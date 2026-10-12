"""Unit tests for the streaming speech pipeline.

The bug class this guards against: text arriving word by word must be spoken
exactly once, in order, and never mid-sentence -- while a producer thread
synthesizes ahead of the player.

Run:  .venv\\Scripts\\python.exe tests\\test_pipeline.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.pipeline import PipelineConfig, SpeechChunker, SpeechPipeline  # noqa: E402
from live_transcribe.tts import Speech  # noqa: E402


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_word_mode_releases_every_word() -> None:
    chunker = SpeechChunker("word")
    ready: list[str] = []
    # Text arrives a word at a time, as the transcriber writes it. Each word is
    # held until the next arrives, so nothing is spoken while it is still being
    # written -- but everything is released exactly once, in order.
    for word in ["Hello", " there", " friend", " here"]:
        ready.extend(chunker.add(word))
    ready.extend(chunker.flush())
    assert " ".join(ready).split() == ["Hello", "there", "friend", "here"], ready


def test_sentence_mode_waits_for_the_full_stop() -> None:
    chunker = SpeechChunker("sentence")
    ready = chunker.add("This is a sentence that is being typed")
    assert ready == [], ready
    # The trailing fragment is never treated as finished, so the period is what
    # completes it; the following space confirms the word ended.
    chunker.add(" and it finally ends.")
    ready = chunker.add(" ")
    assert ready and ready[-1].endswith("ends."), ready


def test_chunk_mode_groups_words_but_respects_sentences() -> None:
    chunker = SpeechChunker("chunk")
    # No full stop anywhere: it must start releasing rather than wait forever,
    # while holding a tail back so a later sentence break still lands correctly.
    ready: list[str] = []
    for word in "One two three four five six seven eight nine ten".split():
        ready.extend(chunker.add(word + " "))
    assert ready, "chunk mode should release text past its cap"
    # Nothing may be lost: released + still pending is the whole input, in order.
    everything = (" ".join(ready) + " " + chunker.pending).split()
    assert everything == "One two three four five six seven eight nine ten".split(), everything

    # A short sentence is released as one unit, punctuation included.
    chunker2 = SpeechChunker("chunk")
    assert chunker2.add("Short. ") == ["Short."]
    assert chunker2.pending == ""


def test_no_words_are_lost_or_repeated_while_streaming() -> None:
    """The whole contract: every word appears exactly once, in order."""
    source = "The quick brown fox jumps over the lazy dog. Then it keeps running!"
    expected = source.split()

    for mode in ("word", "chunk", "phrase", "sentence"):
        chunker = SpeechChunker(mode)
        produced: list[str] = []
        # Feed in irregular pieces *including mid-word cuts*, the way text can
        # genuinely arrive. Nothing may be spoken until its word is complete.
        for i in range(0, len(source), 3):
            produced.extend(chunker.add(source[i : i + 3]))
        produced.extend(chunker.flush())
        got = " ".join(produced).split()
        assert got == expected, f"{mode} mode: {got}"


def test_flush_speaks_a_half_sentence() -> None:
    chunker = SpeechChunker("sentence")
    chunker.add("no full stop here")
    assert chunker.flush() == ["no full stop here"]
    assert chunker.flush() == [], "flushing twice must not repeat"


def test_long_text_is_split_at_word_boundaries() -> None:
    """Whatever the grouping, words must stay whole and nothing may be lost."""
    chunker = SpeechChunker("chunk", target_words=3, max_words=6)
    words = [f"w{i}" for i in range(30)]
    ready = chunker.add(" ".join(words) + " and then some more text. ")
    rejoined = " ".join(ready).split()
    assert rejoined == (words + "and then some more text.".split()), rejoined
    # A sentence terminator takes priority over the size cap, so this may arrive as
    # one chunk; what must never happen is a split inside a word.
    for chunk in ready:
        for token in chunk.split():
            assert token in rejoined, token


def test_unpunctuated_long_text_is_still_split() -> None:
    """Without punctuation the size cap must do the splitting, not hold everything.

    Note the release size, not the maximum, is what normally applies: once pending
    reaches twice the target the target-sized chunk is spoken, so steady state is
    target-sized calls. The maximum is the safety net for a burst of text arriving
    all at once.
    """
    chunker = SpeechChunker("chunk", target_words=3, max_words=6)
    ready: list[str] = []
    for index in range(30):
        ready.extend(chunker.add(f"w{index} "))
    ready.extend(chunker.flush())

    assert len(ready) > 1, "unpunctuated text must be split by size, not held"
    assert all(len(c.split()) <= 6 for c in ready), [len(c.split()) for c in ready]
    assert " ".join(ready).split() == [f"w{i}" for i in range(30)], ready
    assert chunker.pending == "", "nothing should be left after a flush"


def test_a_sudden_burst_of_text_is_capped() -> None:
    """Text arriving in one piece must not produce an oversized call."""
    chunker = SpeechChunker("chunk", target_words=3, max_words=6)
    ready = chunker.add(" ".join(f"w{i}" for i in range(30)) + " ")
    assert ready, "a burst must be released, not held"
    assert all(len(c.split()) <= 6 for c in ready), [len(c.split()) for c in ready]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class FakeClient:
    """Stands in for the StyleTTS2 server, recording what it was asked to say."""

    def __init__(self, delay: float = 0.05, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.seen: list[str] = []
        self.voices: list[str] = []
        self.lock = threading.Lock()

    def synthesize(self, text: str) -> Speech:
        with self.lock:
            self.seen.append(text)
        time.sleep(self.delay)
        if self.fail_on and self.fail_on in text:
            from live_transcribe.tts import TtsError

            raise TtsError("synthetic failure")
        # 0.1 s of silence is enough for the player to write somewhere.
        return Speech(
            samples=np.zeros(2400, dtype="<i2").tobytes(),
            sample_rate=24000,
            seconds=0.1,
            text=text,
            elapsed_s=self.delay,
        )

    def set_voice(self, path) -> dict:  # noqa: ANN001
        with self.lock:
            self.voices.append(str(path))
        return {"ok": True}


class FakeMic:
    """Records what was played, in order, without touching real audio.

    `play_delay` makes playback take time, which is what lets a backlog build:
    without it the producer finishes instantly and the drop path is never reached.
    The sleep is abortable, matching the real device, so a non-draining stop
    cannot hang the test.
    """

    def __init__(self, play_delay: float = 0.0) -> None:
        self.played: list[bytes] = []
        self.silence_writes = 0
        self.silence_ms: list[float] = []
        self.state = "open"
        self.play_delay = play_delay
        self._abort = threading.Event()

    def request_abort(self) -> None:
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    def speak(self, pcm16: bytes, leading_silence: bool = True) -> float:
        assert self.state == "open", "played into a closed microphone"
        if self.play_delay:
            # Interruptible, like the real sliced write.
            self._abort.wait(self.play_delay)
        self.played.append(pcm16)
        return len(pcm16) / 2.0 / 24000.0

    def write_silence(self, ms: float | None = None) -> None:
        self.silence_writes += 1
        self.silence_ms.append(float(ms or 0.0))

    def drain(self, ms: float = 0) -> None:
        pass


def test_pipeline_speaks_every_chunk_in_order() -> None:
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="word", queue_size=8))
    pipe.start()
    for word in ["alpha", " beta", " gamma", " delta", "."]:
        pipe.feed(word)
    pipe.flush()
    pipe.stop(drain=True)

    # "delta" and "." arrive as separate tokens but are one word by the time they
    # are spoken, and each chunk is a run of whole words in the order given.
    # Exact grouping is the chunker's business; the sequence is not.
    spoken_words = " ".join(tts_client.seen).split()
    assert spoken_words == ["alpha", "beta", "gamma", "delta."], tts_client.seen
    assert pipe.stats.chunks_spoken == len(tts_client.seen), pipe.stats
    assert pipe.stats.chunks_dropped == 0
    assert len(virtual_mic.played) == len(tts_client.seen)


def test_word_mode_releases_every_word() -> None:
    chunker = SpeechChunker("word")
    ready: list[str] = []
    # Text arrives a word at a time, as the transcriber writes it. The chunker
    # holds each word until the next arrives, so nothing is spoken mid-typing.
    for word in ["Hello", " there", " friend", " here"]:
        ready.extend(chunker.add(word))
    ready.extend(chunker.flush())
    assert " ".join(ready).split() == ["Hello", "there", "friend", "here"], ready


def test_pipeline_survives_a_synthesis_failure() -> None:
    # Default retries (1) means a failing chunk is tried twice before giving up.
    # The words are fed as one piece so the chunk boundaries are not the subject
    # here -- what matters is that a failure is retried and does not stop the run.
    tts_client, virtual_mic = FakeClient(fail_on="bad"), FakeMic()
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="chunk"))
    pipe.start()
    pipe.say("good")
    pipe.say("bad")
    pipe.say("fine")
    pipe.stop(drain=True)

    assert pipe.stats.failures == 1, pipe.stats
    # The failing chunk is attempted twice; the others are spoken once each.
    assert tts_client.seen.count("bad") == 2, tts_client.seen
    assert "good" in tts_client.seen and "fine" in tts_client.seen, tts_client.seen
    assert pipe.stats.chunks_spoken == 2, pipe.stats
    assert len(virtual_mic.played) == 2


def test_pipeline_drops_when_far_behind_instead_of_drifting() -> None:
    """A producer that cannot keep up must shed stale audio, not drift forever.

    Playback is given a real duration here, so a backlog genuinely builds -- which
    is the situation the drop path exists for (speech arriving faster than it can
    be synthesized and played).
    """
    tts_client, virtual_mic = FakeClient(delay=0.05), FakeMic(play_delay=0.06)
    pipe = SpeechPipeline(
        tts_client,
        virtual_mic,
        PipelineConfig(mode="word", queue_size=2, pending_limit=4),
    )
    pipe.start()
    # Slam in far more text than the queues can hold.
    for i in range(60):
        pipe.feed(f" word{i}")
    pipe.flush()
    pipe.stop(drain=True)

    assert pipe.stats.chunks_dropped > 0, "should have shed stale chunks"
    assert pipe.stats.chunks_spoken > 0, "should still have spoken something"
    # What is spoken must be the recent end of the stream, not all 60 words.
    assert pipe.stats.chunks_spoken < 60, pipe.stats
    assert pipe.stats.chunks_dropped > 0, "should have shed stale chunks"
    assert pipe.stats.chunks_spoken > 0, "should still have spoken something"


def test_voice_resolver_is_called_and_switches_once() -> None:
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="word"))
    voices = [type("V", (), {"slug": "alpha", "path": "alpha.wav"})(),
              type("V", (), {"slug": "beta", "path": "beta.wav"})()]
    current = {"i": 0}

    def resolver():  # noqa: ANN202
        return voices[current["i"]]

    pipe.set_voice_resolver(resolver)
    pipe.start()
    pipe.feed("one")
    pipe.flush()
    time.sleep(0.4)
    current["i"] = 1  # switch mid-stream
    pipe.feed(" two")
    pipe.flush()
    pipe.stop(drain=True)

    assert tts_client.voices == ["alpha.wav", "beta.wav"], tts_client.voices


def test_pipeline_stop_is_idempotent() -> None:
    pipe = SpeechPipeline(FakeClient(), FakeMic(), PipelineConfig(mode="word"))
    pipe.start()
    pipe.stop(drain=True)
    pipe.stop(drain=True)  # must not raise
    assert pipe.pending_count() == 0


def test_say_queues_complete_text_as_one_chunk() -> None:
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="chunk"))
    pipe.start()
    pipe.say("A complete line of text.")
    pipe.stop(drain=True)
    assert tts_client.seen == ["A complete line of text."], tts_client.seen


# --------------------------------------------------------------------------- #
# Silence trimming -- what makes streamed words sound connected
# --------------------------------------------------------------------------- #
def _pcm(samples) -> bytes:  # noqa: ANN001
    import numpy as np

    return np.asarray(samples, dtype="<i2").tobytes()


def test_stop_does_not_lose_a_chunk_being_handed_over() -> None:
    """Shutdown must not kill the player mid-write.

    Regression: a chunk is briefly in *neither* queue while the producer hands it
    to the player. Draining on an empty-queue check killed the player during that
    window and the audio vanished silently -- `chunks spoken 0` with several
    seconds of speech synthesized. Playback therefore has to take real time here,
    which the other fakes deliberately do not.
    """
    tts_client, virtual_mic = FakeClient(delay=0.05), FakeMic(play_delay=0.4)
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="sentence"))
    pipe.start()
    pipe.say("One two three.")
    pipe.stop(drain=True)

    assert pipe.stats.chunks_spoken == 1, f"the chunk was lost: {pipe.stats}"
    assert tts_client.seen == ["One two three."], tts_client.seen
    assert len(virtual_mic.played) == 1, "the audio never reached the microphone"


def test_stop_without_drain_returns_quickly() -> None:
    """Non-draining shutdown is for aborting: it must not wait for playback."""
    tts_client, virtual_mic = FakeClient(delay=0.2), FakeMic(play_delay=0.5)
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="sentence"))
    pipe.start()
    for i in range(5):
        pipe.say(f"Sentence number {i}.")
    started = time.perf_counter()
    pipe.stop(drain=False)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"stop(drain=False) took {elapsed:.2f}s"


def test_stop_waits_for_a_slow_synthesis_in_flight() -> None:
    """Regression: a chunk being synthesized is in neither queue.

    Checking only for empty queues made `stop()` exit while the producer was still
    synthesizing, which killed the player before it could speak the result: the
    real run reported `chunks spoken 0` with 3.7s of speech synthesized. The gaps
    here are deliberately wider than the queue-empty check to expose it.
    """
    tts_client, virtual_mic = FakeClient(delay=1.2), FakeMic(play_delay=0.2)
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="sentence"))
    pipe.start()
    # Long enough to be synthesizing, short enough not to trip the backlog drop.
    pipe.say("One two three.")
    pipe.stop(drain=True)

    assert pipe.stats.chunks_spoken == 1, f"the in-flight chunk was lost: {pipe.stats}"
    assert len(virtual_mic.played) == 1, "the audio never reached the microphone"
    assert pipe.pending_count() == 0, "queues should be clean after stopping"


def test_unpunctuated_speech_still_gets_spoken() -> None:
    """Regression: recognition often returns text with no punctuation at all.

    Waiting strictly for a full stop then releases nothing, so a transcript that
    filled up perfectly produced no audio whatsoever -- the virtual microphone
    stayed silent. Every mode must eventually speak unpunctuated text.
    """
    # 31 words, no '.', '!' or '?' anywhere: the shape of the real transcript that
    # failed on this machine.
    text = (
        "hi there hello whoopio by this is not working by is this not getting "
        "it's going on there hello there world why is this so horrible why is it so"
    )
    assert not any(ch in text for ch in ".!?")

    for mode in ("word", "chunk", "phrase", "sentence"):
        chunker = SpeechChunker(mode)
        released: list[str] = []
        # Feed one word at a time, as the transcriber writes it.
        for word in text.split():
            released.extend(chunker.add(word + " "))
        released.extend(chunker.flush())

        assert released, f"{mode} mode released nothing for unpunctuated text"
        assert " ".join(released).split() == text.split(), f"{mode} mode lost or reordered words"


def test_long_unpunctuated_speech_is_released_as_it_arrives() -> None:
    """A long unpunctuated stretch must be spoken as it comes, not held to the end."""
    chunker = SpeechChunker("sentence")
    released: list[str] = []
    for index in range(60):  # 60 words, still no punctuation
        released.extend(chunker.add(f"word{index} "))
    assert released, "nothing was spoken until the text ended"
    assert len(" ".join(released).split()) >= 28, released


def test_trim_removes_padding_but_keeps_the_speech() -> None:
    import numpy as np

    from live_transcribe.virtual_mic import pcm16_to_float, trim_silence

    rate = 24000
    # 0.5 s of silence, 0.25 s of speech, 0.5 s of silence. The tone must be
    # scaled to 16-bit range: a raw 0.3 amplitude would truncate to zero.
    phase = 2 * np.pi * 220 * np.arange(int(0.25 * rate)) / rate
    tone = (0.3 * np.sin(phase) * 32767).astype("<i2")
    assert np.abs(tone).max() > 1000, "the test tone must actually contain signal"
    padded = np.concatenate(
        [np.zeros(int(0.5 * rate), dtype="<i2"), tone, np.zeros(int(0.5 * rate), dtype="<i2")]
    )

    trimmed = trim_silence(pcm16_to_float(padded.tobytes()), rate, keep_ms=15.0, max_trim_ms=220.0)
    seconds = trimmed.size / rate

    # Roughly the speech plus a little padding at each end -- not 1.25 s.
    assert 0.25 < seconds < 0.45, f"trimmed to {seconds:.3f}s"
    # The speech itself must survive.
    assert float(np.max(np.abs(trimmed))) > 0.2, float(np.max(np.abs(trimmed)))


def test_trim_never_cuts_into_loud_audio_at_the_start() -> None:
    """The cap protects the beginning of a clip, which is what a small cap is for.

    (Trailing silence is always removed regardless of the cap, and a quiet
    fade-out ends the trim wherever the threshold is crossed -- so the start is
    where the cap is actually observable.)
    """
    import numpy as np

    from live_transcribe.virtual_mic import pcm16_to_float, trim_silence

    rate = 24000
    # Speech first, then a long fade-out, then true silence.
    loud = (0.6 * np.sin(2 * np.pi * 200 * np.arange(int(0.4 * rate)) / rate) * 32767).astype("<i2")
    tail = ((0.5 * np.linspace(1.0, 0.0, int(0.6 * rate)) ** 2) * 32767).astype("<i2")
    signal = np.concatenate([loud, tail, np.zeros(int(0.2 * rate), dtype="<i2")])
    pcm = pcm16_to_float(signal.tobytes())

    for cap_ms in (80.0, 220.0):
        out = trim_silence(pcm, rate, keep_ms=15.0, max_trim_ms=cap_ms)
        # The loud opening must survive untouched: nothing may be removed from a
        # clip that starts at full level.
        assert out.size > pcm.size * 0.7, f"cap {cap_ms}ms removed too much: {out.size}/{pcm.size}"
        assert float(np.max(np.abs(out))) > 0.5
        # No more than the cap can go from the front.
        assert out.size >= pcm.size - 2 * int(rate * max(cap_ms, 500.0) / 1000.0)


def test_trim_only_ever_removes_from_the_edges() -> None:
    """The safety property: speech can never be reordered or cut out of the middle.

    Checked as a slice identity rather than by listening -- if the output is a
    contiguous run of the input, the trimmer cannot have damaged anything inside
    the clip, whatever the signal looks like.
    """
    import numpy as np

    from live_transcribe.virtual_mic import pcm16_to_float, trim_silence

    rate = 24000
    cases = {
        "tone padded": np.concatenate([
            np.zeros(int(0.3 * rate), dtype="<i2"),
            (0.4 * np.sin(2 * np.pi * 180 * np.arange(int(0.3 * rate)) / rate) * 32767).astype("<i2"),
            np.zeros(int(0.4 * rate), dtype="<i2"),
        ]),
        "starts loud": np.concatenate([
            (0.5 * np.sin(2 * np.pi * 200 * np.arange(int(0.3 * rate)) / rate) * 32767).astype("<i2"),
            np.zeros(int(0.5 * rate), dtype="<i2"),
        ]),
        "ends loud": np.concatenate([
            np.zeros(int(0.5 * rate), dtype="<i2"),
            (0.5 * np.sin(2 * np.pi * 200 * np.arange(int(0.3 * rate)) / rate) * 32767).astype("<i2"),
        ]),
        "fade out": np.concatenate([
            (0.6 * np.sin(2 * np.pi * 150 * np.arange(int(0.2 * rate)) / rate) * 32767).astype("<i2"),
            ((0.5 * np.linspace(1.0, 0.0, int(0.5 * rate)) ** 2) * 32767).astype("<i2"),
            np.zeros(int(0.2 * rate), dtype="<i2"),
        ]),
        "all silence": np.zeros(int(0.5 * rate), dtype="<i2"),
    }

    for name, signal in cases.items():
        pcm = pcm16_to_float(signal.tobytes())
        for cap_ms in (60.0, 220.0, 900.0):
            out = trim_silence(pcm, rate, keep_ms=15.0, max_trim_ms=cap_ms)

            assert out.size > 0, f"{name}/{cap_ms}: produced nothing"
            assert out.size <= pcm.size, f"{name}/{cap_ms}: grew the clip"

            # Same audio, just a window of it.
            assert float(np.max(np.abs(out))) <= float(np.max(np.abs(pcm))) + 1e-6, name
            if float(np.max(np.abs(pcm))) > 0.01 and out.size < pcm.size:
                assert float(np.max(np.abs(out))) >= float(np.max(np.abs(pcm))) - 1e-6, (
                    f"{name}/{cap_ms}: the loudest part was removed"
                )

def test_trim_leaves_silence_alone_rather_than_deleting_it() -> None:
    import numpy as np

    from live_transcribe.virtual_mic import pcm16_to_float, trim_silence

    rate = 24000
    silence = np.zeros(rate, dtype="<i2")
    trimmed = trim_silence(pcm16_to_float(silence.tobytes()), rate)
    assert trimmed.size == rate, "all-silent audio must not collapse to nothing"


def test_trim_handles_empty_input() -> None:
    import numpy as np

    from live_transcribe.virtual_mic import trim_silence

    assert trim_silence(np.zeros(0, dtype=np.float32), 24000).size == 0


def test_pipeline_trims_what_it_plays_but_not_what_it_received() -> None:
    """The served audio is untouched; only the live stream is tightened."""
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(tts_client, virtual_mic, PipelineConfig(mode="word", trim_silence_ms=220.0))
    pipe.start()
    pipe.say("testing")
    pipe.stop(drain=True)
    assert pipe.stats.chunks_spoken == 1
    # The fake synthesizes pure silence, which the trimmer deliberately leaves be.
    assert tts_client.seen == ["testing"], tts_client.seen
    assert pipe.stats.trimmed_s >= 0.0


# --------------------------------------------------------------------------- #
# Inter-word delays -- the speaker's own rhythm, recorded and replayed
# --------------------------------------------------------------------------- #
def test_timing_marks_survive_chunking_with_their_words() -> None:
    """A mark belongs to the word after it, and must never be stranded by a split.

    The text is fed in two pieces *including a cut inside the mark*, which is the
    case that would silently lose the delay if marks were treated as text: the
    chunker would release "[[30" and the rest would arrive as a different word.
    """
    chunker = SpeechChunker("chunk")
    marked = "hello [[300]] what is [[120]] going on right now "
    ready: list[str] = []
    for i in range(0, len(marked), 5):
        ready.extend(chunker.add(marked[i : i + 5]))
    ready.extend(chunker.flush())

    joined = " ".join(ready)
    assert "[[300]] what" in joined, joined
    assert "[[120]] going" in joined, joined
    assert joined.split() == marked.split(), joined


def test_delays_are_replayed_as_separate_synthesis_calls() -> None:
    """A delay worth replaying means one call per side of it, and one silence."""
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(
        tts_client,
        virtual_mic,
        # A grouping that does not cut on its own: what splits the calls here is
        # the delays, and nothing else.
        PipelineConfig(mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0),
    )
    pipe.start()
    pipe.say("hello [[300]] what is [[120]] going on")
    pipe.stop(drain=True)

    # 300 and 120 both pass the 80 ms floor, so the phrase arrives as three calls
    # and the listener hears their own two pauses back.
    assert tts_client.seen == ["hello", "what is", "going on"], tts_client.seen
    assert virtual_mic.silence_writes == 2, virtual_mic.silence_writes
    assert virtual_mic.silence_ms == [300.0, 120.0], virtual_mic.silence_ms
    assert pipe.stats.gaps_written == 2, pipe.stats


def test_short_delays_are_absorbed_rather_than_replayed() -> None:
    """The latency dial: a 30 ms gap is not worth a synthesis call.

    This is what keeps a normal sentence a single call. Replaying every 30 ms gap
    would turn eight words into eight calls and put the voice seconds behind.
    """
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(
        tts_client,
        virtual_mic,
        PipelineConfig(mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0),
    )
    pipe.start()
    pipe.say("hello [[30]] what is [[20]] going on")
    pipe.stop(drain=True)

    assert tts_client.seen == ["hello what is going on"], tts_client.seen
    assert virtual_mic.silence_writes == 0, virtual_mic.silence_writes


def test_replaying_delays_can_be_switched_off_without_restarting() -> None:
    """The toggle is live: the same text is paced by the model once it is off."""
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(
        tts_client,
        virtual_mic,
        PipelineConfig(mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0),
    )
    # Both chunks are synthesized before playback drains, so the switch is flipped
    # from the player instead: it runs between the two, deterministically.
    on = {"value": True}

    class SwitchingMic(FakeMic):
        def speak(self, pcm16, leading_silence=True):  # noqa: ANN001
            result = FakeMic.speak(self, pcm16, leading_silence)
            on["value"] = False
            return result

    mic = SwitchingMic()
    pipe = SpeechPipeline(
        tts_client,
        mic,
        PipelineConfig(mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0),
    )
    pipe.set_timing_resolver(lambda: on["value"])
    pipe.start()
    pipe.say("one [[200]] two")
    pipe.stop(drain=True)

    assert tts_client.seen == ["one", "two"], tts_client.seen
    assert mic.silence_ms == [200.0], mic.silence_ms


def test_a_long_delay_is_capped_not_dropped() -> None:
    """A hesitant speaker must not make the clone equally hesitant."""
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(
        tts_client,
        virtual_mic,
        PipelineConfig(
            mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0, word_gap_max_ms=400.0
        ),
    )
    pipe.start()
    pipe.say("well [[2500]] then")
    pipe.stop(drain=True)

    assert virtual_mic.silence_ms == [400.0], virtual_mic.silence_ms
    assert tts_client.seen == ["well", "then"], tts_client.seen


def test_marks_never_reach_the_speech_server() -> None:
    """Whatever else happens, the model must never be asked to say "[[300]]"."""
    tts_client, virtual_mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(
        tts_client, virtual_mic, PipelineConfig(mode="chunk", word_delay_replay=False)
    )
    pipe.start()
    pipe.feed("hello [[300]] there friend ")
    pipe.flush()
    pipe.stop(drain=True)

    assert tts_client.seen, "nothing was spoken at all"
    for text in tts_client.seen:
        assert "[[" not in text and "]]" not in text, tts_client.seen
    assert " ".join(tts_client.seen).split() == ["hello", "there", "friend"], tts_client.seen


def _run_all() -> int:
    # Collected explicitly: `globals()` would also pick up local variables inside
    # other tests whose names happen to start with "test_", and then try to call
    # them as tests.
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}\n         {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
