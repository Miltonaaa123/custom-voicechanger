"""The live transcription loop.

Design notes
------------
* Whisper's encoder always pads to a 30 s window, so decoding 1.5 s of audio
  costs almost exactly what decoding 6 s costs. There is therefore no reason to
  trim the decode window hard: each decode reads from the commit cursor to the
  live edge, which gives the model the best acoustic context available.
* A word is committed exactly once. Decodes overlap by design (the model needs
  the audio immediately before the cursor), so every returned word carries an
  absolute timeline position and is dropped if it ends at or before the cursor.
* While the speaker keeps talking, words inside `edge_guard_s` of the live edge
  are held back, because a word sliced by the chunk boundary cannot be
  transcribed reliably. As soon as the speaker pauses, the guard is released and
  everything is flushed.
"""
from __future__ import annotations

import os
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .asr import AsrEngine
from .audio import TARGET_SR, MicCapture, VadConfig, speech_level_db
from .merge import CommitCursor, Word, cap_committed, join_words, select_new_words, uncommitted_text
from .transcript import TranscriptWriter
from . import word_timing

TARGET_WORDS = 15
TARGET_SECONDS = 0.5

LEAD_IN_SAMPLES = 1600  # 100 ms of already-committed audio for decode context


@dataclass
class EngineConfig:
    # Model
    model: str = "distil-small.en"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "en"
    beam_size: int = 1
    cpu_threads: int = 4

    # Streaming cadence
    chunk_s: float = 2.0
    """Un-committed speech to accumulate before decoding while speech continues.
    Larger windows mean fewer, more efficient decodes and fewer boundary words;
    the delay this adds stays well inside the target of "decoded within 2 s"."""
    min_chunk_s: float = 0.8
    """Smallest amount of un-decoded audio worth a decode."""
    min_window_s: float = 1.0
    """Absolute floor on decode length. Below ~1 s, word timestamps become
    unreliable and the model starts re-transcribing the boundary word."""
    max_window_s: float = 24.0
    """Absolute cap on audio per decode (the encoder's limit is 30 s)."""
    max_decode_window_s: float = 2.0
    """Starting (and maximum) cap on a decode window. Started deliberately small:
    a window can only be as slow as its own length allows, so a small cap bounds
    the worst case from the first decode, and the window grows on its own when the
    GPU has headroom. The cap adapts at runtime -- see `target_decode_ms`."""
    target_decode_ms: float = 900.0
    """Decode time the adaptive window aims for.

    The GPU is shared, so the time a fixed window takes varies several-fold with
    whatever else is running. Rather than guess a window size, the engine measures
    each decode (exponential moving average) and scales the window to hold that
    time near this target, clamped to `min_decode_window_s`..`max_decode_window_s`.
    When the GPU is busy the windows shrink, which is what keeps every decode
    inside the 2 s bar; when it frees up they grow again for efficiency."""
    min_decode_window_s: float = 0.9
    """Floor for the adaptive window (below ~1 s, word timestamps get unreliable)."""
    speech_preroll_s: float = 0.4
    """Audio kept before the detected speech onset, so the first word is not
    clipped. Everything before this is room tone that only pads the decode."""
    edge_guard_s: float = 0.32
    """Audio held back at the live edge while speech continues."""
    cursor_holdback_s: float = 0.0
    """How far before the last committed word the next decode is allowed to start.

    Default 0 (the cursor lands exactly on the last word). A small holdback keeps
    a stretch of audio re-readable in case Whisper silently omits a word, but
    measured on this setup it lowered accuracy rather than raising it: it widens
    every decode overlap, so the model re-decides more boundary words. Left
    configurable for experimentation."""
    overlap_guard_s: float = 0.08
    """A word must extend this far past the cursor to be committed, so words the
    model re-emits from the already-committed prefix are not duplicated."""
    min_decode_interval_s: float = 0.05
    min_word_probability: float = 0.1
    """Loose sanity floor only. A strict floor is the wrong tool: genuine words at
    a decode boundary are low-confidence too (measured 0.29 for a real word that
    a strict floor wrongly discarded). Duplicates are caught by re-emission
    detection, and invented tails by the energy gate on flushes."""
    min_word_probability_flush: float = 0.3
    """Slightly stricter floor for flush decodes; invented tails score 0.04-0.07."""
    reemission_window_s: float = 0.6
    """An identical word landing this soon after itself is a re-decode of the same
    audio, not new speech."""
    repetition_window_s: float = 1.0
    """A third identical word inside this span is treated as a Whisper repetition
    loop (what laughter or noise produces) and dropped. 0 disables the check."""
    flush_speech_db: float = 3.0
    """A flush only decodes audio whose 50 ms peak is at least this far above the
    speech threshold. Flushing a speechless tail is what makes Whisper invent
    trailing words, which is poison for the text-to-speech stage."""
    flush_cursor_margin_s: float = 0.25
    """When a flush finds no speech to decode, the cursor is pushed to the end of
    the audio only after this much slack. Advancing it exactly to the end can skip
    a word the model happened to omit on that pass; leaving a little behind means
    such a word is still inside the next decode window."""

    prompt_words: int = 40
    """Committed words fed back as the next decode's prompt."""
    idle_skip_s: float = 0.02
    hypothesis_interval_s: float = 0.25

    # Microphone
    device_index: int | str | None = None
    gain_db: float = 0.0
    """Software boost applied before the VAD and the model, for quiet mics."""
    vad: VadConfig = field(default_factory=VadConfig)
    ring_seconds: float = 120.0

    # Output
    live_hypothesis: bool = True
    show_status: bool = True
    verbose: bool = False

    record_word_delays: bool = True
    """Write the delays measured between words into the voice changer's view.

    Whisper returns a start and end time for every word, so these are the speaker's
    own pauses rather than an estimate. They travel inside the text -- "hello [[70]]
    what" -- because the transcriber and the voice changer are separate processes
    that meet at a file, and they are written only to ``live.txt``: ``transcript.txt``
    stays plain text for a person to read. Without this the timing is discarded and
    the clone speaks at the model's pace instead of the speaker's.
    """

    utterance_wav: str = ""
    """Where to write the audio of each finished utterance ("" disables it).

    The voice changer reads this to hear *how* something was said, not just what.
    Without the audio there is no emotion signal at all: text carries no pitch or
    loudness."""
    max_utterance_s: float = 12.0
    """Cap on the captured utterance. Also bounds memory: the slice is taken from
    the ring buffer, so a long pause cannot grow it without limit."""


class LatencyTracker:
    """Milliseconds from a word being spoken to it appearing in the transcript.

    A committed word whose timestamp is ``age_s`` behind the newest captured
    audio was spoken ``age_s`` ago (assuming capture keeps up, which is verified
    separately via overflow counts). Adding the capture-to-commit delay gives the
    true spoken-to-written latency -- an exact measurement, not a model-speed
    estimate.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self.samples: deque[float] = deque(maxlen=maxlen)
        self.last_audio_at: float | None = None

    def record(self, age_s: float, committed_at: float) -> float:
        if self.last_audio_at is None:
            return float("nan")
        ms = (committed_at - self.last_audio_at + age_s) * 1000.0
        self.samples.append(ms)
        return ms

    def percentile(self, p: float) -> float:
        if not self.samples:
            return float("nan")
        data = sorted(self.samples)
        idx = min(len(data) - 1, max(0, int(round((p / 100.0) * (len(data) - 1)))))
        return data[idx]

    def median(self) -> float:
        return self.percentile(50)

    def summary(self) -> str:
        if not self.samples:
            return "n/a"
        return (
            f"p50={self.percentile(50):.0f}ms  p90={self.percentile(90):.0f}ms  "
            f"p99={self.percentile(99):.0f}ms  (n={len(self.samples)})"
        )


@dataclass
class EngineStats:
    decodes: int = 0
    words: int = 0
    commits: int = 0
    skipped_flushes: int = 0
    decode_ms: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    words_per_decode: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    flush_flags: deque[int] = field(default_factory=lambda: deque(maxlen=60))
    window_ms: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    """Length of the audio window each decode covered."""
    first_word_at: float | None = None

    def throughput_s_per_15_words(self) -> float:
        """Rough cost of 15 words, derived per decode (cost / words produced).

        Flush decodes are excluded because they re-read audio already charged for.
        Only reported once there are two or more such decodes: with a single
        sample the ratio is meaningless (a short clip can easily produce one
        decode for one word and read as "8 seconds per 15 words").
        """
        pairs = [
            ms / words
            for ms, words, is_flush in zip(self.decode_ms, self.words_per_decode, self.flush_flags)
            if words > 0 and ms > 0 and not is_flush
        ]
        if len(pairs) < 2:
            return float("nan")
        return (sorted(pairs)[len(pairs) // 2] / 1000.0) * TARGET_WORDS

    def words_per_decode_median(self) -> float:
        data = sorted(w for w in self.words_per_decode if w > 0)
        return data[len(data) // 2] if data else float("nan")

    def slowest_decode_ms(self) -> float:
        """Worst single decode time seen recently -- the "under 2 s" check."""
        return max(self.decode_ms) if self.decode_ms else float("nan")

    def longest_window_ms(self) -> float:
        return max(self.window_ms) if self.window_ms else float("nan")

    def percentile_decode_ms(self, p: float) -> float:
        if not self.decode_ms:
            return float("nan")
        data = sorted(self.decode_ms)
        return data[min(len(data) - 1, int(round(p / 100.0 * (len(data) - 1))))]


class StreamingTranscriber:
    """Owns the microphone, the ASR model and the transcript files."""

    def __init__(
        self,
        cfg: EngineConfig,
        writer: TranscriptWriter,
        on_text: Callable[[str, str], None] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.writer = writer
        self.on_text = on_text
        self.log = logger or (lambda msg: None)

        self.cursor = CommitCursor()
        self.latency = LatencyTracker()
        self.stats = EngineStats()
        self.engine = AsrEngine(
            model_name=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
            beam_size=cfg.beam_size,
            cpu_threads=cfg.cpu_threads,
            verbose=cfg.verbose,
        )
        self.mic: MicCapture | None = None
        self.listening = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_decode_at = 0.0
        self._prev_speech = False
        self.idle_since = time.perf_counter()
        """Last moment the VAD reported no speech. Used to keep decoding gated on
        voice activity, so silence is never handed to Whisper."""
        self._last_decode_total = -1
        self._last_decode_cursor = -1.0
        """Window identity of the previous decode. Without it, a flush that ends
        with the cursor behind the live edge re-decodes the same audio forever."""
        self._decode_ema_ms: float | None = None
        self._window_s = cfg.max_decode_window_s
        """Adaptive decode-window cap, driven by `_adapt_window`."""
        self._hypothesis = ""
        self._hypothesis_at = 0.0
        self._last_status_len = 0
        self._utterance_start_s = 0.0
        """Where the current utterance's audio begins, for emotion analysis. Set to
        the end of the previous committed word each time one is committed."""
        self._marks: list[word_timing.TimingMark] = []
        """Committed words with the delays the speaker left in front of them."""
        self._last_word_end: float | None = None
        """Where the last committed word finished, so the next delay is measurable."""

    # -- setup ------------------------------------------------------------- #
    def prepare(self) -> None:
        self.log(f"[model] loading {self.cfg.model} ({self.cfg.device}/{self.cfg.compute_type}) ...")
        self.engine.load()
        self.log(
            f"[model] ready in {self.engine.load_seconds:.1f}s using "
            f"{self.engine.device}/{self.engine.compute_type}"
        )
        self.engine.warmup()
        self.log(f"[model] warm-up decode {self.engine.warmup_seconds * 1000:.0f} ms")

        self.mic = MicCapture(
            device=self.cfg.device_index,
            target_sr=TARGET_SR,
            ring_seconds=self.cfg.ring_seconds,
            vad=self.cfg.vad,
            gain_db=self.cfg.gain_db,
        )
        self.mic.start()
        gain_note = f", gain {self.cfg.gain_db:+.0f} dB" if self.cfg.gain_db else ""
        self.log(
            f"[mic]   {self.mic.device_name} @ {self.mic.native_sr} Hz -> {TARGET_SR} Hz "
            f"(buffer {self.mic._blocksize} frames{gain_note})"
        )
        self.writer.event(
            "start",
            model=self.cfg.model,
            device=self.engine.device,
            compute_type=self.engine.compute_type,
            mic=self.mic.device_name,
            mic_sr=self.mic.native_sr,
        )

    # -- control ----------------------------------------------------------- #
    def toggle_listening(self) -> bool:
        self.listening = not self.listening
        if not self.listening:
            self._hypothesis = ""
            self._publish(force=True)
        self.writer.event("listen", on=self.listening)
        return self.listening

    def clear(self) -> None:
        """Forget everything and blank the transcript without restarting."""
        position = self.mic.total_samples / float(TARGET_SR) if self.mic else 0.0
        self.cursor.reset(position)
        self.stats = EngineStats()
        self.latency = LatencyTracker()
        self._hypothesis = ""
        self._marks = []
        self._last_word_end = None
        # The window identity no longer refers to a decoded position, so the
        # next tick must be allowed to decode even if nothing new arrives.
        self._last_decode_total = -1
        self._last_decode_cursor = -1.0
        self.writer.set_committed("", "")
        self.writer.event("clear")
        if self.on_text:
            self.on_text("", "")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        if self.mic is not None:
            self.mic.stop()
        self.writer.event(
            "stop",
            words=self.stats.words,
            decodes=self.stats.decodes,
            latency_p50_ms=round(self.latency.median(), 1) if self.latency.samples else None,
            latency_p99_ms=round(self.latency.percentile(99), 1) if self.latency.samples else None,
        )

    # -- main loop --------------------------------------------------------- #
    def run_forever(self) -> None:
        self._run(self._stop)

    def run_for(self, seconds: float) -> None:
        """Run the loop for a bounded time (tests and benchmarks)."""
        ev = threading.Event()
        timer = threading.Timer(seconds, ev.set)
        timer.daemon = True
        timer.start()
        try:
            self._run(ev)
        finally:
            timer.cancel()

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self.run_forever, name="transcribe-loop", daemon=True)
        self._thread.start()

    def _run(self, stop_event: threading.Event) -> None:
        assert self.mic is not None, "call prepare() first"
        while not stop_event.is_set() and not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - never let the loop die
                self.log(f"[warn] loop error: {type(exc).__name__}: {exc}")
                self.writer.event("loop_error", error=f"{type(exc).__name__}: {exc}")
                time.sleep(0.1)
        self._final_flush()

    def _tick(self) -> None:
        assert self.mic is not None
        self.mic.pump(0.02)
        now = time.perf_counter()
        self.latency.last_audio_at = now

        if not self.listening:
            time.sleep(self.cfg.idle_skip_s)
            return

        speaking = self.mic.speech_active
        stopped = self.mic.speech_stopped_at is not None and not speaking

        if speaking and not self._prev_speech:
            # Speech just started. Room tone before this moment is room tone:
            # decoding it wastes GPU time, invents words, and makes the first
            # real word look seconds late. Start the cursor at speech onset (minus
            # a short pre-roll so the first word is not clipped) instead.
            total = self.mic.total_samples
            onset = max(0, total - int((self.cfg.speech_preroll_s + self.cfg.min_chunk_s) * TARGET_SR))
            if onset / float(TARGET_SR) < self.cursor.position:
                self.cursor.position = onset / float(TARGET_SR)
            self.writer.event("speech_onset", cursor_s=round(self.cursor.position, 2))
        self._prev_speech = speaking
        if not speaking and not stopped:
            self.idle_since = now

        pending_s = (self.mic.total_samples - int(self.cursor.position * TARGET_SR)) / TARGET_SR
        if pending_s < self.cfg.min_chunk_s:
            self._expire_hypothesis(now)
            time.sleep(self.cfg.idle_skip_s)
            return

        # Decoding the exact same window with the exact same cursor cannot
        # produce anything new. Checking this before requesting a decode is what
        # stops the loop from spinning once the source stops producing audio.
        if self.mic.total_samples == self._last_decode_total and self.cursor.position == self._last_decode_cursor:
            self._expire_hypothesis(now)
            time.sleep(self.cfg.idle_skip_s)
            return

        # Both the decode window and the cursor advance by whole decodes: a word
        # is only committed once the model has read a decent amount of audio
        # after it, otherwise the same word gets re-committed every cycle.
        trigger = stopped or pending_s >= self.cfg.chunk_s
        if (
            not trigger
            and speaking
            and (now - self._last_decode_at) >= 1.2
        ):
            # Safety valve for one long, unbroken utterance.
            trigger = True

        if not trigger or (now - self._last_decode_at) < self.cfg.min_decode_interval_s:
            time.sleep(self.cfg.idle_skip_s)
            return

        self._decode_once(now, flush=stopped)
        if stopped:
            self.mic.speech_stopped_at = None

    # -- one decode -------------------------------------------------------- #
    def _decode_once(self, now: float, flush: bool) -> None:
        assert self.mic is not None
        total = self.mic.total_samples
        cursor_samples = int(round(self.cursor.position * TARGET_SR))

        read_start = max(0, cursor_samples - LEAD_IN_SAMPLES)
        # Two caps: the encoder's hard limit, and the adaptive cap that keeps any
        # single decode cheap enough to stay well inside the 2 s bar.
        max_window = int(min(self.cfg.max_window_s, self._window_s) * TARGET_SR)
        if total - read_start > max_window:
            read_start = total - max_window

        audio = self.mic.read_range(read_start, total)
        self._last_decode_total = total
        self._last_decode_cursor = self.cursor.position
        if not flush and audio.size < int(self.cfg.min_window_s * TARGET_SR):
            # Too little context: word timestamps would be unreliable and the
            # model tends to re-emit the boundary word. A flush is exempt: it is
            # the last chance to capture the trailing words.
            return

        self._last_decode_at = now
        prompt = self.cursor.tail_words(self.cfg.prompt_words) or None
        read_offset = read_start / float(TARGET_SR)
        window_end = read_offset + audio.size / float(TARGET_SR)
        limit = window_end if flush else max(0.0, window_end - self.cfg.edge_guard_s)

        committed_now = 0
        last_segment_words: list[Word] = []

        if flush:
            # Only the un-committed tail matters here. If it holds no speech, skip
            # the decode entirely: Whisper fed silence pads to 30 s and starts
            # inventing words, which then land in the transcript and get spoken
            # aloud by StyleTTS 2.
            tail = audio[int(max(0.0, self.cursor.position - read_offset) * TARGET_SR) :]
            peak_db = speech_level_db(tail)
            need_db = self.cfg.vad.threshold_db + self.cfg.flush_speech_db
            if tail.size == 0 or peak_db < need_db:
                self.cursor.position = max(
                    self.cursor.position,
                    window_end - self.cfg.flush_cursor_margin_s,
                )
                self._hypothesis = ""
                self._publish(force=True)
                self.stats.skipped_flushes += 1
                self.writer.event(
                    "flush_skipped",
                    reason="no speech in tail",
                    peak_db=round(peak_db, 1),
                    need_db=round(need_db, 1),
                    cursor_s=round(self.cursor.position, 2),
                )
                self._write_status()
                return
        # The word straddling the cursor is re-emitted when the same audio is
        # decoded again, and its end time drifts by a few tens of ms between
        # decodes. The guard must therefore apply on flushes too, or that word is
        # appended a second time. It is far smaller than any real word, so no
        # genuine content is ever held back by it.
        overlap_guard = self.cfg.overlap_guard_s
        try:
            for seg_words, _seg_end in self.engine.iter_decode(audio, prompt=prompt):
                # Shift onto the absolute timeline. New objects, not mutation:
                # the ASR layer owns the originals.
                shifted = [
                    Word(text=w.text, start=w.start + read_offset, end=w.end + read_offset, probability=w.probability)
                    for w in seg_words
                ]
                last_segment_words = shifted
                picked = [
                    w
                    for w in select_new_words(
                        shifted,
                        self.cursor,
                        window_end,
                        edge_guard_s=0.0,
                        allow_edge_words=True,
                        overlap_guard_s=overlap_guard,
                        min_probability=(
                            self.cfg.min_word_probability_flush if flush else self.cfg.min_word_probability
                        ),
                        reemission_window_s=self.cfg.reemission_window_s,
                        repetition_window_s=self.cfg.repetition_window_s,
                    )
                    if w.end <= limit
                ]
                if picked:
                    self._commit(picked, now)
                    committed_now += len(picked)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[warn] decode failed: {type(exc).__name__}: {exc}")
            self.writer.event("decode_error", error=f"{type(exc).__name__}: {exc}")
            return

        decode_ms = (time.perf_counter() - now) * 1000.0
        self.stats.decodes += 1
        self.stats.decode_ms.append(decode_ms)
        self.stats.words_per_decode.append(committed_now)
        self.stats.flush_flags.append(1 if flush else 0)
        self.stats.window_ms.append(audio.size / float(TARGET_SR) * 1000.0)
        self._adapt_window(decode_ms)
        # Slow decodes are the interesting event: logged individually so a stall
        # can be told apart from steady load (the window and flush flag are what
        # show whether it was a big window or the machine freezing).
        if decode_ms > max(1500.0, self.cfg.target_decode_ms * 2):
            self.writer.event(
                "slow_decode",
                ms=round(decode_ms),
                window_ms=round(audio.size / float(TARGET_SR) * 1000.0),
                flush=flush,
                words=committed_now,
                adaptive_window_s=round(self._window_s, 2),
            )

        if flush:
            # The speaker stopped, so the cursor must end up at the tail of the
            # audio just decoded -- otherwise it stays where it was, the same
            # window is decoded again, and the loop never settles.
            if last_segment_words:
                self.cursor.position = max(self.cursor.position, last_segment_words[-1].end)
            elif committed_now == 0:
                self.cursor.position = max(
                    self.cursor.position,
                    max(0.0, window_end - self.cfg.flush_cursor_margin_s),
                )
            self._hypothesis = ""
            self._publish(force=True)
            self.writer.event("flush", cursor_s=round(self.cursor.position, 2), words=committed_now)
            if committed_now:
                self._write_utterance_audio(window_end)
        elif committed_now == 0 and last_segment_words:
            self._update_hypothesis(uncommitted_text(last_segment_words, self.cursor), now)
        self._write_status()

    def _write_utterance_audio(self, window_end: float) -> None:
        """Save the audio of the utterance that just finished.

        The voice changer reads this to judge *how* something was said. Emotion
        cannot be recovered from the transcript: text has no pitch, loudness or
        timing, which is where essentially all of it lives. Only the audio since the
        previous commit is written, so the clip is one utterance rather than a
        growing slab of the whole session.

        Failure here is logged and ignored. Losing the emotion signal costs a piece
        of styling; losing the transcript would cost the words.
        """
        if not self.cfg.utterance_wav or self.mic is None:
            return
        try:
            start_s = max(0.0, min(self._utterance_start_s, window_end))
            # Never capture more than the cap, so a long silence cannot turn this
            # into a large read.
            start_s = max(start_s, window_end - self.cfg.max_utterance_s)
            start = int(start_s * TARGET_SR)
            end = int(window_end * TARGET_SR)
            if end - start < int(0.2 * TARGET_SR):
                return
            audio = self.mic.read_range(start, end)
            if audio.size == 0:
                return
            pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
            path = Path(self.cfg.utterance_wav)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written via a temporary file and renamed, so a reader never opens a
            # half-written wav.
            tmp = path.with_suffix(path.suffix + ".part")
            with wave.open(str(tmp), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(TARGET_SR)
                handle.writeframes(pcm.tobytes())
            os.replace(tmp, path)
            self._utterance_start_s = window_end
            self.writer.event(
                "utterance_audio",
                path=str(path),
                seconds=round((end - start) / float(TARGET_SR), 2),
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"[warn] could not save utterance audio: {type(exc).__name__}: {exc}")

    def _adapt_window(self, decode_ms: float) -> None:
        """Scale the decode window so decodes hover near `target_decode_ms`.

        This is what makes the 2 s bar hold on a shared GPU: when something else
        starts using the card, decodes slow down, the window shrinks, and the cost
        per decode falls back under the bar. When the GPU frees up the window
        grows again, which is more efficient per word.

        The reaction to an over-budget decode is immediate and proportional (the
        window shrinks as much as the decode overshot), because a slow decode
        usually means the card just got busy and the next one would be slow too.
        Recovery is gradual, so the window does not thrash.
        """
        self._decode_ema_ms = (
            decode_ms if self._decode_ema_ms is None else 0.7 * self._decode_ema_ms + 0.3 * decode_ms
        )
        if decode_ms <= 1.0 or decode_ms != decode_ms:
            return

        target = self.cfg.target_decode_ms
        if decode_ms > target:
            # Shrink proportionally to the overshoot, immediately.
            ratio = max(0.4, target / decode_ms)
        else:
            # Under budget: grow slowly, damped by how much headroom there is.
            ratio = min(1.15, max(1.0, (target / max(decode_ms, 1.0)) ** 0.35))

        self._window_s = max(
            self.cfg.min_decode_window_s,
            min(self.cfg.max_decode_window_s, self._window_s * ratio),
        )

    def _commit(self, words: list[Word], now: float) -> None:
        assert self.mic is not None
        newest_s = self.mic.total_samples / float(TARGET_SR)
        for w in words:
            self.latency.record(max(0.0, newest_s - w.end), now)
        self.cursor.add(words)
        cap_committed(self.cursor)
        self._remember_delays(words)
        # Hold the cursor slightly behind the last word so a word Whisper failed
        # to emit this pass stays inside the next decode window. Anything the
        # re-read repeats is dropped by re-emission detection, and the text
        # always holds the full history, so the cursor lagging the text is fine.
        self.cursor.position = max(0.0, words[-1].end - self.cfg.cursor_holdback_s)
        self.stats.words += len(words)
        self.stats.commits += 1
        if self.stats.first_word_at is None:
            self.stats.first_word_at = now
            self.writer.event("first_word", text=join_words([w.text for w in words]).strip())
        self.writer.event(
            "commit",
            words=len(words),
            text=join_words([w.text for w in words]).strip(),
            cursor_s=round(self.cursor.position, 3),
            latency_ms=round(self.latency.samples[-1], 1) if self.latency.samples else None,
        )
        self._hypothesis = ""
        self._publish()

    def _remember_delays(self, words: list[Word]) -> None:
        """Record how long the speaker left before each word just committed.

        The timing comes from the words' own start and end times, not from when the
        decoder finished, so what is measured is the speaker's rhythm. A word the
        model skipped leaves a gap that belongs to the silence it was not heard in,
        which is the closest thing to that word's own delay.
        """
        if not self.cfg.record_word_delays:
            return
        added = 0
        for word in words:
            if not word.clean:
                continue
            self._marks.append(
                word_timing.TimingMark(
                    word=word.clean, gap_ms=self._delay_before(word.start)
                )
            )
            self._last_word_end = word.end
            added += 1
        if added:
            # Bound the memory on a long dictation session. Only the tail can still
            # be in the text the voice changer has not read yet, and the text itself
            # is capped separately.
            del self._marks[:-600]

    def _delay_before(self, start_s: float) -> float:
        """Milliseconds between the previous word ending and `start_s`."""
        previous = self._last_word_end
        if previous is None:
            return 0.0
        gap_s = float(start_s) - float(previous)
        if gap_s < 0:
            return 0.0
        if gap_s > word_timing.UTTERANCE_GAP_S:
            # The end of a thought, not rhythm inside one: replaying that silence
            # would make the clone wait for a pause nobody is waiting for.
            return 0.0
        return min(gap_s * 1000.0, word_timing.MAX_GAP_MS)

    def _marked_text(self) -> str:
        """The committed words with their delays, for the voice changer's view."""
        if not self.cfg.record_word_delays or not self._marks:
            return ""
        return word_timing.marks_to_text(self._marks)

    def _update_hypothesis(self, text: str, now: float) -> None:
        if not text or not self.cfg.live_hypothesis:
            return
        self._hypothesis = text
        # Throttled: the live guess is cosmetic and must not thrash the disk.
        if (now - self._hypothesis_at) >= self.cfg.hypothesis_interval_s:
            self._publish()

    def _expire_hypothesis(self, now: float) -> None:
        if self._hypothesis and (now - self._hypothesis_at) > 0.5:
            self._hypothesis = ""
            self._publish()

    def _publish(self, force: bool = False) -> None:
        self._hypothesis_at = time.perf_counter()
        hypothesis = self._hypothesis if self.cfg.live_hypothesis else ""
        marked = self._marked_text()
        self.writer.set_committed(
            self.cursor.text,
            hypothesis,
            live_text=marked if marked else None,
        )
        if self.on_text:
            self.on_text(self.cursor.text, self._hypothesis)

    def _write_status(self) -> None:
        if not self.cfg.show_status:
            return
        t = self.stats
        worst = t.slowest_decode_ms()
        worst_txt = f"{worst:4.0f}ms" if worst == worst else " n/a"
        p50 = self.latency.median()
        p50_txt = f"{p50:4.0f}ms" if p50 == p50 else " n/a"
        line = f"[{t.words:5d} words | decode {worst_txt} | delay {p50_txt}]"
        pad = max(0, self._last_status_len - len(line))
        print("\r" + line + " " * pad, end="", flush=True)
        self._last_status_len = len(line)

    # -- shutdown ----------------------------------------------------------- #
    def _final_flush(self) -> None:
        if self.mic is None:
            return
        try:
            self.mic.pump(0.35)
            self._last_decode_total = -1  # allow a final look at the same audio
            self._last_decode_cursor = -1.0
            if (self.mic.total_samples - int(self.cursor.position * TARGET_SR)) > int(0.3 * TARGET_SR):
                self._decode_once(time.perf_counter(), flush=True)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[warn] final flush failed: {exc}")
        self._hypothesis = ""
        self._publish(force=True)
        if self.cfg.show_status:
            print()

    def summary(self) -> str:
        t = self.stats
        s = self.engine.stats()
        tput = t.throughput_s_per_15_words()
        # Only shown when it has enough samples to mean anything; the headline
        # number is the worst decode time, which is what the 2 s bar checks.
        budget = (
            f"  ({tput * 1000:.0f} ms per 15 words)"
            if tput == tput
            else ""
        )
        worst = t.slowest_decode_ms()
        worst_txt = (
            f"{worst:.0f} ms  [{'OVER' if worst > 2000 else 'under'} 2000 ms]"
            if worst == worst
            else "n/a"
        )
        overflows = self.mic.overflows if self.mic else 0
        return "\n".join(
            [
                "",
                "-" * 74,
                f"  words committed      : {t.words}",
                f"  decodes              : {t.decodes}   (avg {s['avg_decode_ms']:.0f} ms){budget}",
                f"  worst decode         : {worst_txt}",
                f"  longest window       : {self.stats.longest_window_ms():.0f} ms of audio",
                f"  spoken -> written    : {self.latency.summary()}",
                f"  mic overflows        : {overflows}",
                f"  transcript           : {self.writer.paths.clean}",
                f"  live view            : {self.writer.paths.live}",
                f"  telemetry            : {self.writer.paths.events}",
                "-" * 74,
            ]
        )
