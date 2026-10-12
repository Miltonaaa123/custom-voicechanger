"""Streaming speech: chunk text, synthesize ahead, play in order.

The naive approach -- synthesize a chunk, play it, synthesize the next -- leaves an
audible hole the size of every synthesis call. Measured on this machine a call
costs ~0.6-0.7 s and yields 1.5-6.5 s of audio, so instead the two halves run
concurrently: a producer thread keeps speech ready ahead of playback, and the
player thread writes it into the virtual microphone without waiting.

That works because synthesis is faster than speech (2.5-5x real time here). The
pipeline still guards the case where it is not: pending text and ready audio are
both capped, and when the caps are hit the oldest work is dropped rather than
letting the voice drift further and further behind what was said.

Chunking is the other half of the problem. One word at a time is the most
responsive but the least natural -- the model gets almost no context and produces
choppy, over-articulated speech. `chunk` mode (the default) groups a few words,
which keeps the voice close behind while giving the model enough to sound like a
sentence.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .tts import Speech, TtsClient, TtsError
from .virtual_mic import VirtualMic, pcm16_to_float, trim_silence
from . import audio_effects, word_timing

_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")

# Words per synthesis call, by mode: (release size, forced maximum, ceiling).
#
# "release size" is where text is spoken when nothing better is available, and
# "forced maximum" is the hard cap that guarantees speech cannot stall. They differ
# for sentence mode, which waits for real punctuation up to the ceiling and only
# falls back to the smaller release size when the sentence never completes.
DEFAULTS = {
    "word": (1, 2, 3),
    "chunk": (4, 9, 12),
    "phrase": (8, 18, 24),
    # Speech recognition very often returns text with NO punctuation at all
    # (Chrome's Web Speech API frequently does). Waiting strictly for a full stop
    # then means nothing is ever spoken -- a transcript of 152 characters produced
    # no audio at all. After this many words the text is spoken anyway, so the
    # sentence break is missed rather than the audio being lost.
    "sentence": (14, 24, 40),
}


@dataclass
class PipelineConfig:
    mode: str = "chunk"
    """word | chunk | phrase | sentence -- how much text each synthesis call gets."""
    target_words: int = 0
    """0 derives it from `mode`."""
    max_words: int = 0
    queue_size: int = 12
    """Chunks of finished audio held ready to play."""
    pending_limit: int = 16
    """Total backlog (text waiting to be synthesized + audio waiting to play) before
    the oldest work is dropped. Bounding the sum is what bounds lag; bounding one
    queue alone does not, because the producer keeps draining it."""
    gap_ms: float = 0.0
    """Extra silence between chunks; 0 keeps the model's own pacing."""
    word_delay_replay: bool = False
    """Put the speaker's own delays between the words back into the speech.

    Off (the default) the timing marks in the text are simply stripped, which is
    the behaviour this pipeline had before delays were recorded at all.
    """
    word_gap_min_ms: float = 80.0
    """Shortest measured delay worth a synthesis call; see `WordDelaySettings`."""
    word_gap_max_ms: float = word_timing.MAX_GAP_MS
    """Cap on a replayed delay, so a hesitant speaker does not stall the clone."""
    trim_silence_ms: float = 220.0
    """Maximum silence trimmed from each end of a chunk.

    StyleTTS 2 pads every call, which is invisible on a sentence but very audible
    word by word: a one-word call can be mostly silence, so words sound spaced
    out. Trimming is what makes streaming speech sound connected. 0 disables it.
    """
    retries: int = 1
    rate_scale: float = 1.0
    """How much faster than the model's own rate the clip is played.

    Above 1.0 the clip is shorter and higher; below 1.0 it is longer and lower.
    The virtual microphone opens its stream at ``sample_rate * rate_scale`` so the
    cable sees an ordinary stream while the voice comes out shifted.
    """
    apply_effects: bool = False
    """Whether to run `live_transcribe.audio_effects` over each clip."""


@dataclass
class PipelineStats:
    chunks_spoken: int = 0
    chunks_dropped: int = 0
    synthesis_s: float = 0.0
    audio_s: float = 0.0
    failures: int = 0
    backlog_peak: int = 0
    trimmed_s: float = 0.0
    """Silence removed from chunk edges -- what makes streamed words sound joined."""

    segments: int = 0
    """Synthesis calls made. Above the chunk count while delays are being replayed."""

    gaps_written: int = 0
    """Delays put back into the microphone between words."""

    gap_ms_total: float = 0.0
    """How long those delays added up to."""

    gap_ms_peak: float = 0.0
    """The longest delay replayed."""

    @property
    def realtime_factor(self) -> float:
        return (self.audio_s / self.synthesis_s) if self.synthesis_s else 0.0


class SpeechChunker:
    """Splits a growing stream of text into speakable chunks.

    Text arrives a word at a time from the transcriber, so a chunk may only be
    released once it is certainly complete: either it ends a sentence, or it has
    been held long enough that waiting any longer would mean producing no audio at
    all.

    The second rule matters more than it looks. Speech recognition regularly
    returns text with no punctuation whatsoever, and a chunker that waits only for
    a full stop then never releases anything -- the transcript fills up and the
    virtual microphone stays silent.
    """

    def __init__(self, mode: str = "chunk", target_words: int = 0, max_words: int = 0) -> None:
        default_release, default_max, _ = DEFAULTS.get(mode, DEFAULTS["chunk"])
        self.target_words = max(1, target_words or default_release)
        # `max_words` doubles as the point where a missing sentence break is
        # forgiven: without it, an unpunctuated transcript is never spoken.
        self.max_words = max(self.target_words, max_words or default_max)
        self.pending = ""
        """Text not yet spoken, as a raw string.

        Deliberately a string rather than a word list: text can arrive split
        anywhere, including in the middle of a word, so the trailing fragment has to
        stay attached for the next call. Only text up to the final whitespace is
        ever released -- speaking half a word would be wrong, and rebuilding from
        split tokens is what fused words together ("there" + " hello" ->
        "therehello").
        """

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _words(text: str) -> int:
        """Words in a chunk, ignoring the timing marks between them."""
        return sum(1 for t in word_timing.split_tokens(text) if word_timing.use_mark(t) is None)

    def _take(self, count: int) -> list[str]:
        """Move `count` complete words out of pending, leaving the fragment.

        Words arrive with their own leading spaces, so the consumed text is put
        back together from its own parts and never re-joined by guesswork. A
        trailing fragment stays behind: a word is not finished until something
        follows it.

        The cut lands on a word boundary, which means a delay stays in front of its
        own word. Cutting between them instead is what left a chunk starting with
        the mark and ending with the word: "[[300]] what" was split into "[[300]]"
        and "what", and the delay was then replayed on the far side of the word it
        belonged to.
        """
        text = self.pending
        mark_spans = [(m.start(), m.end()) for m in word_timing.MARK_RE.finditer(text)]
        at = 0
        taken = 0
        while taken < count:
            nxt = text.find(" ", at)
            if nxt < 0:
                break
            # A space ends a word only when the next token is another word: a mark
            # there means the delay belongs to the word after it, so the cut has to
            # happen past the mark rather than before it.
            next_token = text[nxt + 1 :].split(" ", 1)[0]
            if word_timing.use_mark(next_token) is not None:
                at = nxt + 1
                continue
            if any(start < nxt < end for start, end in mark_spans):
                at = nxt + 1
                continue
            at = nxt + 1
            taken += 1
        head, tail = text[:at], text[at:]
        self.pending = tail
        return [head.strip()] if head.strip() else []

    def _cut_tokens(self, tokens: list[str], target: int, cap: int | None = None) -> list[str]:
        """Group finished tokens into speakable pieces of at most `cap` words.

        Every token is spoken: this runs on text that is already complete, so the
        only question is where the calls are cut. Two callers want different
        sizes: the long text splitter breaks every `target` words, while the pause
        flush keeps a short utterance as one call, which is what makes it sound
        like a sentence.

        A delay stays with the word that follows it -- it is collected into the
        next piece rather than trailing the one before, or the delay would be
        replayed on the wrong side of the word it belongs to.
        """
        ceiling = max(target, cap if cap is not None else target)
        pieces: list[list[str]] = []
        prefix: list[str] = []
        piece: list[str] = []
        words = 0
        for token in tokens:
            if word_timing.use_mark(token) is not None:
                # A delay belongs to whatever word comes next; if it lands on a
                # cut it starts the following call.
                (piece if piece else prefix).append(token)
                continue
            piece.append(token)
            words += 1
            if words >= ceiling:
                pieces.append(prefix + piece)
                prefix, piece, words = [], [], 0
        if piece:
            pieces.append(prefix + piece)
        return [" ".join(p).strip() for p in pieces if p]

    def _cut_text(self, text: str, target: int, cap: int | None = None) -> list[str]:
        """Split complete text into pieces, keeping whole words and their delays."""
        return self._cut_tokens(word_timing.split_tokens(text), target, cap)

    def _cut_at_words(self, target: int, cap: int | None = None) -> list[str]:
        """`_cut_text` over what is still pending, leaving nothing behind.

        The trailing token is only a fragment while a word is still being written;
        a flush is the end of the line, so it is spoken like any other.
        """
        tokens = word_timing.split_tokens(self.pending)
        if not self.pending.endswith((" ", "\n")) and word_timing.looks_like_mark_prefix(self.pending):
            # Half a mark: the completion is not here yet, so it cannot be spoken.
            tokens = tokens[:-1]
        self.pending = ""
        return self._cut_tokens(tokens, target, cap)

    def add(self, text: str) -> list[str]:
        """Feed new text; returns the chunks that are ready to speak."""
        self.pending += text
        ready: list[str] = []

        while True:
            # A word is only finished once something follows it, so ignore any
            # trailing fragment when looking for a sentence break. Marks are not
            # words and cannot carry punctuation.
            boundary = self.pending.rfind(" ")
            complete = self.pending[: boundary + 1] if boundary >= 0 else ""
            match = _SENTENCE_END.search(word_timing.strip_marks(complete))

            if match:
                consumed = len(complete[: match.end()].split())
                ready.extend(self._take(consumed))
                continue

            words = self._words(complete)
            # Release at the target once there is clearly more than one chunk's
            # worth, so a real sentence break still gets its chance to land first.
            if words >= self.target_words * 2:
                ready.extend(self._take(self.target_words))
                continue
            # Hard cap: speak what we have rather than waiting forever for
            # punctuation that may never come.
            if words >= self.max_words:
                ready.extend(self._take(words))
                continue
            break
        return ready

    def flush(self) -> list[str]:
        """Anything still held back, for shutdown -- speak it rather than lose it."""
        text = self.pending.strip()
        self.pending = ""
        return self._split_long(text) if text else []

    def _split_long(self, text: str) -> list[str]:
        """Break text that exceeds the maximum at a word boundary.

        A piece that already fits the cap is spoken as one call; a longer one is
        broken into target-sized pieces. Timing marks ride along with their words.
        """
        if not text:
            return []
        if self._words(text) <= self.max_words:
            return [text]
        return self._cut_text(text, self.target_words, self.max_words)


@dataclass
class _Job:
    """One chunk of text waiting to be synthesized.

    `text` is what arrived, which may carry timing marks; `words` is the same text
    with them removed, filled in on the producer thread.
    """

    text: str
    words: str = ""
    marks: list = field(default_factory=list)
    error: str = ""


@dataclass
class _Segment:
    """One synthesized piece of a chunk, and the delay that precedes it."""

    speech: Speech | None = None
    text: str = ""
    gap_ms: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""


@dataclass
class _Ready:
    job: _Job
    segments: list = field(default_factory=list)


class SpeechPipeline:
    """Producer/consumer speech: synthesize ahead, play without gaps."""

    def __init__(
        self,
        client: TtsClient,
        mic: VirtualMic,
        config: PipelineConfig | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.mic = mic
        self.cfg = config or PipelineConfig()
        self.log = logger or (lambda _m: None)
        self.chunker = SpeechChunker(self.cfg.mode, self.cfg.target_words, self.cfg.max_words)
        self.stats = PipelineStats()
        self.active_voice = None

        self._pending: "queue.Queue[_Job]" = queue.Queue(maxsize=max(4, self.cfg.pending_limit))
        self._ready: "queue.Queue[_Ready]" = queue.Queue(maxsize=max(2, self.cfg.queue_size))
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._player: threading.Thread | None = None

        # Work the producer has taken but not yet handed to the player. During
        # synthesis a chunk is in *neither* queue, so emptiness alone cannot tell
        # the drain that everything is finished -- it would exit, stop the player,
        # and silently lose the sentence being synthesized.
        self._in_flight = 0
        self._idle = threading.Condition()
        self._active_reference: str | None = None
        """The reference the server is currently cloning, so an emotion switch is
        only issued when it actually changes."""
        self._timing_resolver: Callable[[], bool] | None = None
        """Asked before every chunk whether to replay the delays.

        A callback rather than a flag because the switch lives in the browser and
        must take effect on the next sentence, not on the next restart.
        """
        self.delay_settings = word_timing.WordDelaySettings(
            enabled=self.cfg.word_delay_replay,
            min_gap_ms=self.cfg.word_gap_min_ms,
            max_gap_ms=self.cfg.word_gap_max_ms,
        )
        # The live audio shaping (pitch, volume, gate). Held on the pipeline so the
        # control panel can adjust it while speech is playing; `apply_effects` is
        # what decides whether the player thread looks at it at all.
        self.audio: audio_effects.AudioOptions = audio_effects.AudioOptions()

    # -- rhythm ------------------------------------------------------------ #
    def set_timing_resolver(self, resolver: Callable[[], bool] | None) -> None:
        """Install the live "replay the delays" switch (called per chunk)."""
        self._timing_resolver = resolver

    def set_word_delay_replay(self, enabled: bool) -> None:
        """Turn delay replaying on or off, with no resolver installed."""
        self.delay_settings.enabled = bool(enabled)
        self.cfg.word_delay_replay = bool(enabled)

    def _refresh_timing(self) -> None:
        wanted = self.cfg.word_delay_replay
        if self._timing_resolver is not None:
            try:
                wanted = bool(self._timing_resolver())
            except Exception as exc:  # noqa: BLE001 - a bad switch must not stop speech
                self.log(f"  ! could not read the delay setting: {exc}")
                wanted = self.cfg.word_delay_replay
        if wanted != self.delay_settings.enabled:
            self.delay_settings.enabled = wanted
            self.log(
                f"  word delays: {'replaying your own pace' if wanted else 'off (model pace)'}"
            )

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        if self._producer is not None:
            return
        self._stop.clear()
        self._producer = threading.Thread(target=self._produce, name="tts-producer", daemon=True)
        self._player = threading.Thread(target=self._play, name="tts-player", daemon=True)
        self._producer.start()
        self._player.start()

    def stop(self, drain: bool = True, timeout: float = 60.0) -> None:
        """Finish queued speech (default), then shut the threads down.

        Shutdown is signalled with sentinels rather than only a flag. A worker
        blocked on `Queue.get(timeout=...)` never re-checks a flag while the queue
        stays empty, so a flag-only stop made the threads spin until the timeout --
        and `drain=False` (abort) appeared to hang. Each worker exits as soon as it
        dequeues the sentinel.

        Draining also cannot just watch the queues for emptiness: a chunk is
        briefly in *neither* queue while the producer hands it to the player, and
        stopping then killed the player mid-write and silently lost the audio. The
        drain therefore waits for the hand-off and joins the player, which is
        inside a blocking device write and so cannot finish early.
        """
        if not drain:
            # Abandoning: ask the device to cut any write short, or stopping would
            # wait for the current sentence to finish playing.
            self._stop.set()
            try:
                self.mic.request_abort()  # type: ignore[attr-defined]
            except AttributeError:
                pass

        if self._producer is not None:
            deadline = time.time() + timeout
            with self._idle:
                while drain and time.time() < deadline:
                    # Nothing queued and nothing mid-synthesis means everything that
                    # was asked for has been handed to the player and played.
                    if self._pending.empty() and self._ready.empty() and self._in_flight == 0:
                        break
                    self._idle.wait(0.1)

        self._stop.set()
        self._wake_workers()

        # A bounded join: the abort path must stay responsive even if a worker is
        # stuck inside a device call that cannot be interrupted.
        join_timeout = timeout if drain else min(timeout, 5.0)
        for thread in (self._producer, self._player):
            if thread is not None:
                thread.join(timeout=join_timeout)
        self._producer = self._player = None
        self._clear_queues()

    def _clear_queues(self) -> None:
        """Drop anything still queued, including shutdown sentinels.

        Without this the sentinels used to wake the workers would be counted as
        pending work, so `pending_count()` would never reach zero after stopping.
        """
        for target in (self._pending, self._ready):
            while True:
                try:
                    target.get_nowait()
                except queue.Empty:
                    break
                else:
                    target.task_done()

    def _wake_workers(self) -> None:
        """Poke both queues so a blocked worker sees the stop flag.

        Sentinels are dropped when a queue is full, so nothing is missed: a full
        queue means the worker still has items to consume and will re-check the
        flag on its next pass.
        """
        for _ in range(3):
            for target in (self._pending, self._ready):
                try:
                    target.put_nowait(_Ready(job=_Job(text="")) if target is self._ready else _Job(text=""))
                except queue.Full:
                    break
            time.sleep(0.01)

    # -- input ------------------------------------------------------------- #
    def feed(self, text: str) -> int:
        """Add text; returns how many chunks became ready to speak."""
        chunks = self.chunker.add(text)
        for chunk in chunks:
            self._enqueue(chunk)
        return len(chunks)

    def say(self, text: str) -> None:
        """Queue a complete piece of text as one chunk."""
        text = text.strip()
        if not text:
            return
        for chunk in self.chunker._split_long(text):
            self._enqueue(chunk)

    def flush_partial(self) -> int:
        """Speak whatever is held, without shutting down.

        Needed because the size cap alone is not enough: a short utterance can end
        and leave fewer words pending than any threshold, and then nothing is ever
        spoken. The caller flushes on a pause, which is exactly where the
        transcriber commits its text, so the sentence break is not lost.

        Returns how many chunks were queued.
        """
        if not self.chunker.pending.strip():
            return 0
        count = 0
        for chunk in self.chunker._cut_at_words(self.chunker.target_words, self.chunker.max_words):
            self._enqueue(chunk)
            count += 1
        return count

    def flush(self) -> None:
        """Speak anything the chunker is holding back, at shutdown."""
        for chunk in self.chunker.flush():
            self._enqueue(chunk)

    def _enqueue(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        # The delays stay in the text until the producer reads it, so the setting
        # in force when the chunk is *spoken* is the one that applies.
        job = _Job(text=text)
        while not self._stop.is_set():
            # Enforce a cap on total backlog, not just on one queue. Bounding only
            # the pending queue does not bound lag at all: the producer keeps
            # draining it, so the queue never looks full while the ready queue and
            # the player fall further and further behind.
            self._shed_backlog()
            try:
                self._pending.put(job, timeout=0.5)
                self.stats.backlog_peak = max(
                    self.stats.backlog_peak, self._pending.qsize() + self._ready.qsize()
                )
                return
            except queue.Full:
                continue

    def _shed_backlog(self) -> None:
        """Drop the oldest waiting work when too much has piled up."""
        limit = max(1, self.cfg.pending_limit)
        backlog = self._pending.qsize() + self._ready.qsize()
        if backlog < limit:
            return
        # Oldest first: in-flight audio is what the listener is about to hear, so
        # superseded text is the better thing to lose.
        dropped: _Job | None = None
        try:
            dropped = self._pending.get_nowait()
            self._pending.task_done()
        except queue.Empty:
            try:
                item = self._ready.get_nowait()
                self._ready.task_done()
                dropped = item.job
            except queue.Empty:
                return
        if dropped is not None and dropped.text:
            self.stats.chunks_dropped += 1
            self.log(f"  ! behind - dropped: {dropped.text[:40]}")

    # -- voice ------------------------------------------------------------- #
    def set_reference_resolver(self, resolver: Callable[[object], object] | None) -> None:
        """Install a per-chunk reference chooser: (job) -> wav path or None.

        Called on the producer thread for every chunk, which is what lets a chunk be
        spoken in a voice picked for the emotion measured in the audio it came from.
        It must be quick and must not raise -- a slow chooser would stall speech, and
        a raise would drop the chunk -- so callers get the failure logged and the
        previous voice kept.
        """
        self._reference_resolver = resolver

    def set_emotion_note(self, note: str) -> None:
        """Record what the current emotion is, purely for reporting."""
        self._emotion_note = note

    @property
    def emotion_note(self) -> str:
        return getattr(self, "_emotion_note", "")

    def _resolve_reference(self, job: object) -> object:
        resolver = getattr(self, "_reference_resolver", None)
        if resolver is None:
            return None
        try:
            return resolver(job)
        except Exception as exc:  # noqa: BLE001 - styling must never break the words
            self.log(f"  ! could not choose an emotion voice: {exc}")
            return None

    def set_voice_resolver(self, resolver: Callable[[], object]) -> None:
        """Install a callback that returns the voice to clone (or None).

        It is called on the producer thread before each chunk, which is what makes
        a voice change take effect mid-stream without restarting anything.
        """
        self._voice_resolver = resolver
        self._apply_voice()

    def _apply_voice(self) -> None:
        resolver = getattr(self, "_voice_resolver", None)
        if resolver is None:
            return
        try:
            wanted = resolver()
        except Exception as exc:  # noqa: BLE001 - a bad selector must not stop speech
            self.log(f"  ! voice selector failed: {exc}")
            return
        if wanted is None:
            return
        if self.active_voice is not None and getattr(wanted, "slug", None) == self.active_voice:
            return
        try:
            answer = self.client.set_voice(wanted.path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            self.log(f"  ! could not switch voice: {exc}")
            return
        if answer.get("ok") is False:
            self.log(f"  ! server refused the voice: {answer.get('error')}")
            return
        self.active_voice = getattr(wanted, "slug", str(wanted))
        # Keep the reference bookkeeping in step, so the next emotion switch knows
        # which voice the server is actually holding.
        self._active_reference = str(wanted.path)
        self.log(f"  voice: {self.active_voice}")

    def _synthesize(self, text: str, reference: object) -> Speech:
        """Synthesize `text`, switching the cloned voice first if the chunk needs one.

        The per-request ``reference_wav`` field cannot be used for this. Measured
        against the running server, a request carrying such a path is rejected with
        "a reference_wav in a request must be inside <riko-project>", because that
        field is the browser-facing one and is confined to the server's own folder;
        this project keeps its voices elsewhere. ``set_voice`` (the
        ``/reload-reference`` route) is the path the server does trust for local
        files, and it is what `voice.txt` switching already uses.

        Switching costs one round trip, so it only happens when the wanted voice
        differs from the one already loaded -- in sentence mode that is once per
        utterance rather than once per chunk.
        """
        if reference is None:
            return self.client.synthesize(text)
        wanted = str(reference)
        if wanted != self._active_reference:
            try:
                self.client.set_voice(reference)  # type: ignore[attr-defined]
                self._active_reference = wanted
            except Exception as exc:  # noqa: BLE001 - speak in the old voice, not silence
                self.log(f"  ! could not switch to {Path(wanted).name}: {exc}")
        return self.client.synthesize(text)

    def _synth_one(self, text: str, reference: object) -> Speech:
        """Synthesize one piece, retrying a failure the configured number of times."""
        last_error = ""
        for attempt in range(self.cfg.retries + 1):
            try:
                return self._synthesize(text, reference)
            except TtsError as exc:
                last_error = str(exc)
                if attempt < self.cfg.retries:
                    time.sleep(0.2)
        raise TtsError(last_error or "synthesis failed")

    def _synthesize_segments(
        self, job: _Job, reference: object | None
    ) -> tuple[list[_Segment], str, float]:
        """Synthesize a chunk as one or more pieces, with the delays between them.

        A chunk carrying timing marks is spoken one marked group at a time, because
        a delay can only be inserted between two synthesis calls. A chunk with no
        marks -- an ordinary sentence -- is a single call, exactly as before.

        Returns the segments, the last error, and how long synthesis took.
        """
        plain = job.words or word_timing.strip_marks(job.text)
        pieces = job.marks or [word_timing.TimingMark(word=plain, gap_ms=0.0)]
        segments: list[_Segment] = []
        error = ""
        started = time.perf_counter()
        for index, mark in enumerate(pieces):
            # An empty first piece means the delay belongs to the gap after the
            # previous chunk, which has already been played: dropping it is right.
            if index == 0 and not mark.word.strip():
                continue
            # Resolve the emotion voice once per chunk. Per piece would ask the
            # same question several times and could flip the voice mid-sentence.
            if index == 0 or reference is None:
                reference = self._resolve_reference(job)
            if reference is None:
                self._apply_voice()
            self.stats.segments += 1
            try:
                speech = self._synth_one(mark.word, reference)
            except TtsError as exc:
                error = str(exc)
                segments.append(_Segment(text=mark.word, gap_ms=mark.gap_ms, error=error))
                self.log(f"  ! synthesis failed: {error[:120]}")
                continue
            if speech is not None:
                self.stats.audio_s += speech.seconds
            segments.append(
                _Segment(
                    speech=speech,
                    text=mark.word,
                    gap_ms=mark.gap_ms,
                    elapsed_s=speech.elapsed_s if speech is not None else 0.0,
                )
            )
        return segments, error, time.perf_counter() - started

    # -- worker threads ---------------------------------------------------- #
    def _produce(self) -> None:
        """Synthesize chunks ahead of playback."""
        while True:
            try:
                job = self._pending.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if not job.text:
                # Shutdown sentinel, or an empty job from the drop path.
                self._pending.task_done()
                if self._stop.is_set():
                    return
                continue
            with self._idle:
                self._in_flight += 1
            try:
                self._refresh_timing()
                raw_marks = word_timing.split_marks(job.text, self.delay_settings)
                job.marks = word_timing.segment_marks(raw_marks)
                job.words = word_timing.strip_marks(job.text)
                segments, error, elapsed = self._synthesize_segments(job, None)
                self.stats.synthesis_s += elapsed
                job.error = error
                self.stats.failures += sum(1 for s in segments if s.speech is None)
                if len(segments) > 1:
                    self.log(f"  {len(segments)} pieces (your own delays): {job.words[:44]}")

                # Block until the player has room, so audio is never dropped for
                # lack of a place to put it.
                while not self._stop.is_set():
                    try:
                        self._ready.put(_Ready(job=job, segments=segments), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            finally:
                with self._idle:
                    self._in_flight -= 1
                    self._idle.notify_all()
                self._pending.task_done()

    def _play(self) -> None:
        """Write speech to the virtual microphone as soon as it is ready."""
        while True:
            try:
                item = self._ready.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                job = item.job
                if not job.text:
                    # Shutdown sentinel.
                    if self._stop.is_set():
                        return
                    continue
                for segment in item.segments:
                    if segment.gap_ms:
                        # The delay the speaker left in front of this word. Written
                        # rather than slept so it lands in the microphone exactly
                        # like the words do, and so a stop can still cut it short.
                        self.mic.write_silence(float(segment.gap_ms))
                        self.stats.gaps_written += 1
                        self.stats.gap_ms_total += float(segment.gap_ms)
                        self.stats.gap_ms_peak = max(self.stats.gap_ms_peak, float(segment.gap_ms))
                    if segment.speech is None:
                        continue
                    if self.cfg.gap_ms:
                        self.mic.write_silence(self.cfg.gap_ms)
                    samples = segment.speech.samples
                    original = samples
                    if self.cfg.trim_silence_ms > 0 or self.cfg.apply_effects:
                        # Trim here rather than in the client, so the saved/served
                        # audio is untouched and only the live stream is tightened.
                        # A word whose delay is being replayed keeps a wider edge:
                        # the padding trimmed here would be shorter than the delay
                        # being restored, and the rhythm would come out clipped.
                        keep_ms = 15.0 if not segment.gap_ms else min(45.0, segment.gap_ms / 2.0)
                        working = pcm16_to_float(samples)
                        if self.cfg.trim_silence_ms > 0:
                            working = trim_silence(
                                working,
                                segment.speech.sample_rate,
                                keep_ms=max(10.0, keep_ms),
                                max_trim_ms=max(60.0, self.cfg.trim_silence_ms),
                            )
                        if self.cfg.apply_effects and self.audio is not None:
                            # Gate and volume first: the gate has to measure the
                            # level the listener will actually hear, and it must
                            # run before the pitch shift changes the clip's length.
                            working = audio_effects.process(
                                working, segment.speech.sample_rate, self.audio
                            )
                        if abs(self.cfg.rate_scale - 1.0) > 1e-9:
                            working = audio_effects.resample(working, self.cfg.rate_scale)
                        samples = np.clip(working * 32767.0, -32768, 32767).astype("<i2").tobytes()
                    started = time.perf_counter()
                    self.mic.speak(samples, leading_silence=False)
                    played = time.perf_counter() - started
                    self.stats.chunks_spoken += 1
                    trimmed_s = (len(original) - len(samples)) / 2.0 / max(
                        1, segment.speech.sample_rate
                    )
                    self.stats.trimmed_s += max(0.0, trimmed_s)
                    heard = len(samples) / 2.0 / max(1, segment.speech.sample_rate)
                    delay_txt = f" after {segment.gap_ms:.0f}ms" if segment.gap_ms else ""
                    made = segment.speech.elapsed_s if segment.speech is not None else segment.elapsed_s
                    self.log(
                        f"  spoke {heard:4.1f}s (trimmed {max(0.0, trimmed_s):4.1f}s of padding, "
                        f"made in {made:4.2f}s, played in {played:4.2f}s){delay_txt}: {segment.text[:52]}"
                    )
            finally:
                self._ready.task_done()

    # -- reporting --------------------------------------------------------- #
    def pending_count(self) -> int:
        return self._pending.qsize() + self._ready.qsize()

    def summary(self) -> str:
        stat = self.stats
        rhythm = ""
        if stat.gaps_written:
            average = stat.gap_ms_total / stat.gaps_written
            rhythm = (
                f" | {stat.gaps_written} of your own delays replayed "
                f"(avg {average:.0f} ms, longest {stat.gap_ms_peak:.0f} ms) "
                f"in {stat.segments} calls"
            )
        return (
            f"  chunks spoken {stat.chunks_spoken}, dropped {stat.chunks_dropped}, "
            f"failed {stat.failures} | {stat.audio_s:.1f}s speech from "
            f"{stat.synthesis_s:.1f}s synthesis ({stat.realtime_factor:.1f}x real time){rhythm}"
        )
