"""Unit tests for the parts most likely to break silently: streaming resampling,
the audio ring buffer, VAD, and the word-commit rule that prevents duplicated or
lost words at decode boundaries.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
      (or simply: .venv\\Scripts\\python.exe tests\\test_core.py)
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.audio import (  # noqa: E402
    TARGET_SR,
    AudioRing,
    EnergyVad,
    StreamingResampler,
    VadConfig,
    block_rms_db,
)
from live_transcribe.merge import (  # noqa: E402
    CommitCursor,
    Word,
    cap_committed,
    join_words,
    select_new_words,
    uncommitted_text,
)
from live_transcribe.transcript import TranscriptWriter  # noqa: E402
from live_transcribe.engine import LatencyTracker  # noqa: E402
from live_transcribe import word_timing  # noqa: E402


# --------------------------------------------------------------------------- #
def test_resampler_length_and_tone() -> None:
    """48k -> 16k must produce ~1/3 the samples and preserve a sine's frequency."""
    src_sr, dst_sr = 48000, 16000
    rs = StreamingResampler(src_sr, dst_sr)
    freq = 440.0
    t = np.arange(src_sr * 3, dtype=np.float32) / src_sr
    tone = np.sin(2 * math.pi * freq * t).astype(np.float32)

    outs = []
    for i in range(0, len(tone), 480):  # 10 ms blocks
        outs.append(rs.process(tone[i : i + 480]))
    out = np.concatenate(outs)

    expected = len(tone) // 3
    assert abs(len(out) - expected) <= 2, f"got {len(out)}, expected ~{expected}"

    # Dominant frequency should still be 440 Hz.
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak = np.fft.rfftfreq(len(out), 1 / dst_sr)[int(np.argmax(spec))]
    assert abs(peak - freq) < 6, f"peak {peak} Hz, expected {freq}"

    # Amplitude must survive (no gain blow-up or collapse).
    assert 0.6 < np.sqrt(np.mean(out**2)) / (1 / math.sqrt(2)) < 1.05


def test_resampler_block_boundaries_are_continuous() -> None:
    """No clicks: block-wise output should match a single-shot resample closely."""
    from scipy.signal import resample_poly

    src_sr, dst_sr = 44100, 16000
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(src_sr * 2).astype(np.float32) * 0.1

    rs = StreamingResampler(src_sr, dst_sr)
    outs = [rs.process(audio[i : i + 441]) for i in range(0, len(audio), 441)]
    streamed = np.concatenate(outs)

    reference = resample_poly(audio, dst_sr // math.gcd(src_sr, dst_sr), src_sr // math.gcd(src_sr, dst_sr))
    n = min(len(streamed), len(reference))
    # Compare in the steady-state interior (edges differ by design).
    a = streamed[800 : n - 800]
    b = reference[800 : n - 800]
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.995, f"streamed vs single-shot correlation {corr:.4f}"


def test_resampler_passthrough() -> None:
    rs = StreamingResampler(16000, 16000)
    block = np.arange(100, dtype=np.float32)
    out = rs.process(block)
    assert np.array_equal(out, block)


def test_resampler_long_run_does_not_drift() -> None:
    """Minutes of audio must not accumulate sample-count error."""
    rs = StreamingResampler(48000, 16000)
    total_in = 0
    total_out = 0
    for _ in range(int(48000 * 60 / 480)):  # 60 seconds in 10 ms blocks
        block = np.zeros(480, dtype=np.float32)
        total_in += len(block)
        total_out += len(rs.process(block))
    assert total_out == total_in // 3, f"in={total_in} out={total_out}"


# --------------------------------------------------------------------------- #
def test_ring_wraparound_read_last() -> None:
    ring = AudioRing(100)
    ring.write(np.arange(80, dtype=np.float32))
    ring.write(np.arange(80, 160, dtype=np.float32))  # wraps
    got = ring.read_last(100)
    assert len(got) == 100
    assert np.array_equal(got, np.arange(60, 160, dtype=np.float32))


def test_ring_read_range_absolute_positions() -> None:
    ring = AudioRing(50)
    ring.write(np.arange(0, 120, dtype=np.float32))
    # 120 samples offered, 50 retained: positions are numbered by retained audio.
    assert ring.total_written == 50
    assert np.array_equal(ring.read_last(50), np.arange(70, 120, dtype=np.float32))

    got = ring.read_range(30, 40)
    assert np.array_equal(got, np.arange(100, 110, dtype=np.float32)), got

    # A request reaching before the retained window is clamped, never crashes.
    old = ring.read_range(0, 10)
    assert np.array_equal(old, np.arange(70, 80, dtype=np.float32)), old

    # Fully-eaten positions return nothing.
    assert len(ring.read_range(0, 0)) == 0
    assert len(ring.read_range(-500, -400)) == 0
    # Empty / inverted ranges.
    assert len(ring.read_range(25, 25)) == 0
    assert len(ring.read_range(40, 10)) == 0


def test_ring_write_larger_than_capacity() -> None:
    ring = AudioRing(10)
    ring.write(np.arange(25, dtype=np.float32))
    assert np.array_equal(ring.read_last(10), np.arange(15, 25, dtype=np.float32))


# --------------------------------------------------------------------------- #
def test_vad_detects_speech_and_stop() -> None:
    vad = EnergyVad(VadConfig(threshold_db=-42.0, release_db=-50.0, hangover_s=0.2))
    sr = 16000
    silence = np.zeros(sr // 10, dtype=np.float32)
    loud = (np.random.default_rng(0).standard_normal(sr // 10) * 0.2).astype(np.float32)

    speaking, stopped = vad.update(silence, sr)
    assert not speaking and not stopped
    assert block_rms_db(loud) > -30

    speaking, _ = vad.update(loud, sr)
    assert speaking, "loud audio must be detected as speech"

    # hangover_s=0.2 with 0.1 s blocks: the first silent block is tolerated.
    speaking, stopped = vad.update(silence, sr)
    assert speaking and not stopped, "hangover must delay the stop signal"

    # Silence run reaches the hangover threshold on the second block.
    speaking, stopped = vad.update(silence, sr)
    assert not speaking and stopped, "stop must fire after the hangover"


# --------------------------------------------------------------------------- #
def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_select_new_words_skips_already_committed() -> None:
    """Overlapping decodes must not duplicate words."""
    cur = CommitCursor()
    cur.add([w(" hello", 0.0, 0.5), w(" world", 0.5, 1.0)])  # cursor now at 1.0

    # Next decode re-reads 100 ms of lead-in and returns the old words again.
    decoded = [w(" hello", 0.0, 0.5), w(" world", 0.5, 1.0), w(" again", 1.0, 1.4)]
    picked = select_new_words(decoded, cur, window_end=1.4, edge_guard_s=0.0, allow_edge_words=True)
    assert [p.clean for p in picked] == ["again"]


def test_select_new_words_edge_guard_holds_back_boundary_word() -> None:
    cur = CommitCursor()
    decoded = [w(" one", 0.0, 0.4), w(" two", 0.4, 0.8), w(" three", 0.8, 1.2)]
    picked = select_new_words(decoded, cur, window_end=1.2, edge_guard_s=0.3, allow_edge_words=False)
    # 'three' ends inside the 0.3 s guard (limit = 0.9) so it is deferred.
    assert [p.clean for p in picked] == ["one", "two"]

    # Once the speaker pauses the guard is released and the word is committed.
    picked = select_new_words(decoded, cur, window_end=1.2, edge_guard_s=0.3, allow_edge_words=True)
    assert [p.clean for p in picked] == ["one", "two", "three"]


def test_full_streaming_sequence_no_duplicates_no_loss() -> None:
    """Simulate consecutive overlapping decodes over one sentence."""
    reference = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    timeline = [(i * 0.4, (i + 1) * 0.4) for i in range(len(reference))]

    cur = CommitCursor()
    committed: list[str] = []
    # Decode windows of 4 words, each overlapping the previous by 1 word.
    for start_i in range(0, len(reference), 3):
        window = reference[start_i : start_i + 4]
        spans = timeline[start_i : start_i + 4]
        decoded = [w(" " + t if i else t, s, e) for i, (t, (s, e)) in enumerate(zip(window, spans))]
        window_end = spans[-1][1]
        picked = select_new_words(decoded, cur, window_end, edge_guard_s=0.0, allow_edge_words=True)
        cur.add(picked)
        committed.extend(p.clean for p in picked)

    assert committed == reference, f"got {committed}"


def test_join_words_spacing_and_punctuation() -> None:
    assert join_words(["hello", " world"]) == "hello world"
    assert join_words(["Hello", ",", " world", "."]) == "Hello, world."
    assert join_words([" don't", " stop"]) == "don't stop"
    assert join_words(["a", "b"]) == "a b"


def test_cursor_add_maintains_incremental_text() -> None:
    cur = CommitCursor()
    cur.add([w(" Hello", 0.0, 0.4)])
    cur.add([w(" there", 0.4, 0.8), w(".", 0.8, 0.9)])
    assert cur.text == "Hello there."
    assert cur.position == 0.9
    assert cur.tail_words(2) == "there ."

    cur.reset(5.0)
    assert cur.text == "" and cur.position == 5.0 and cur.committed == []


def test_cap_committed_keeps_text_but_limits_words() -> None:
    cur = CommitCursor()
    for i in range(500):
        cur.add([w(f" w{i}", i * 0.1, (i + 1) * 0.1)])
    cap_committed(cur, max_words=50)
    assert len(cur.committed) == 50
    assert cur.text.startswith("w0") and cur.text.endswith("w499")


def test_uncommitted_text_is_the_live_guess() -> None:
    cur = CommitCursor()
    cur.add([w(" hello", 0.0, 0.5)])
    decoded = [w(" hello", 0.0, 0.5), w(" brave", 0.5, 0.9)]
    assert uncommitted_text(decoded, cur) == "brave"


# --------------------------------------------------------------------------- #
def test_repetition_loop_is_cut() -> None:
    """Whisper repeating one token over laughter must not flood the transcript."""
    from live_transcribe.merge import is_repetition_loop

    cur = CommitCursor()
    # Two "oh"s already committed inside a second.
    cur.add([w(" oh,", 0.0, 0.3), w(" oh,", 0.3, 0.6)])
    assert is_repetition_loop(w(" oh,", 0.6, 0.9), cur, 1.0), "third repeat is a loop"

    # Emphasis with fewer repeats must survive.
    cur2 = CommitCursor()
    cur2.add([w(" no,", 0.0, 0.4)])
    assert not is_repetition_loop(w(" no,", 0.4, 0.8), cur2, 1.0), "'no, no!' is not a loop"

    # A genuine repeat far apart in time is not a loop either.
    cur3 = CommitCursor()
    cur3.add([w(" really", 0.0, 0.4), w(" really", 4.0, 4.4)])
    assert not is_repetition_loop(w(" really", 8.0, 8.4), cur3, 1.0), "spaced repeats are fine"


def test_select_new_words_drops_repetition_loop() -> None:
    cur = CommitCursor()
    cur.add([w(" oh", 0.0, 0.3), w(" oh", 0.3, 0.6)])
    decoded = [w(" oh", 0.6, 0.9), w(" oh", 0.9, 1.2)]
    picked = select_new_words(
        decoded, cur, window_end=1.2, edge_guard_s=0.0, allow_edge_words=True, repetition_window_s=1.0
    )
    assert picked == [], f"loop words must be dropped, got {[p.clean for p in picked]}"


def test_transcript_writer_clears_on_start_and_writes() -> None:
    with tempfile.TemporaryDirectory() as d:
        clean = Path(d) / "transcript.txt"
        clean.write_text("OLD CONTENT FROM LAST SESSION", encoding="utf-8")

        wt = TranscriptWriter(clean_path=clean, clear_on_start=True)
        assert clean.read_text(encoding="utf-8") == "", "transcript must be cleared on start"

        wt.set_committed("hello world")
        assert clean.read_text(encoding="utf-8") == "hello world"

        wt.set_committed("hello world again", "and more")
        assert clean.read_text(encoding="utf-8") == "hello world again"
        assert "and more" in wt.paths.live.read_text(encoding="utf-8")

        # No rewrite when nothing changed.
        before = clean.stat().st_mtime_ns
        wt.set_committed("hello world again", "and more")
        assert clean.stat().st_mtime_ns == before

        wt.event("commit", words=2)
        assert wt.paths.events.read_text(encoding="utf-8").strip().startswith("{")


def test_transcript_writer_append_mode_preserves_content() -> None:
    with tempfile.TemporaryDirectory() as d:
        clean = Path(d) / "t.txt"
        clean.write_text("keep me", encoding="utf-8")
        TranscriptWriter(clean_path=clean, clear_on_start=False)
        assert clean.read_text(encoding="utf-8") == "keep me"


def test_the_word_changer_view_carries_delays_the_transcript_does_not() -> None:
    """The delays are for the speech stage; a person reading the file sees words.

    live.txt is what the voice changer follows, so the inter-word delays go there.
    transcript.txt is what a person reads, so it must not contain "[[70]]".
    """
    with tempfile.TemporaryDirectory() as d:
        clean = Path(d) / "transcript.txt"
        wt = TranscriptWriter(clean_path=clean, clear_on_start=True)
        wt.set_committed("hello what", live_text="hello [[70]] what")
        assert clean.read_text(encoding="utf-8") == "hello what"
        assert wt.paths.live.read_text(encoding="utf-8") == "hello [[70]] what"

        # The unfinished guess still goes in the live file, after the marked words.
        wt.set_committed("hello what is", "go", live_text="hello [[70]] what is")
        live = wt.paths.live.read_text(encoding="utf-8")
        assert live.startswith("hello [[70]] what is"), live
        assert "go" in live, live
        assert clean.read_text(encoding="utf-8") == "hello what is"


def test_whisper_word_times_become_the_delays_the_voice_reader_sees() -> None:
    """The local engine measures the speaker's pauses from the word timestamps."""
    from live_transcribe.engine import EngineConfig, StreamingTranscriber
    from live_transcribe.merge import Word

    with tempfile.TemporaryDirectory() as d:
        cfg = EngineConfig()
        writer = TranscriptWriter(clean_path=Path(d) / "transcript.txt", clear_on_start=True)
        engine = StreamingTranscriber.__new__(StreamingTranscriber)
        engine.cfg = cfg
        engine.writer = writer
        engine.on_text = None
        engine.log = lambda _m: None
        engine._marks = []
        engine._last_word_end = None
        engine._hypothesis = ""
        engine._hypothesis_at = 0.0
        engine.latency = LatencyTracker()
        engine.cursor = CommitCursor()

        words = [
            Word(text=" hello", start=1.00, end=1.30),
            Word(text=" what", start=1.33, end=1.60),   # 30 ms later
            Word(text=" is", start=2.10, end=2.30),     # 500 ms later
        ]
        engine._remember_delays(words)
        engine.cursor.add(words)
        engine._publish()

        live = writer.paths.live.read_text(encoding="utf-8")
        clean = (Path(d) / "transcript.txt").read_text(encoding="utf-8")
        assert "[[" in live, live
        assert "[[" not in clean, clean
        assert clean == "hello what is", clean

        builder = word_timing
        marks = builder.marks_from_tokens(builder.split_tokens(live))
        assert [m.word for m in marks] == ["hello", "what", "is"], marks
        assert marks[0].gap_ms == 0.0, "the first word of a session has no delay"
        assert 25 <= marks[1].gap_ms <= 35, marks
        assert 490 <= marks[2].gap_ms <= 510, marks


def test_a_pause_between_thoughts_is_left_out_of_the_delays() -> None:
    from live_transcribe.engine import EngineConfig, StreamingTranscriber
    from live_transcribe.merge import Word

    with tempfile.TemporaryDirectory() as d:
        writer = TranscriptWriter(clean_path=Path(d) / "t.txt", clear_on_start=True)
        engine = StreamingTranscriber.__new__(StreamingTranscriber)
        engine.cfg = EngineConfig()
        engine.writer = writer
        engine.on_text = None
        engine._marks = []
        engine._last_word_end = None
        engine._hypothesis = ""
        engine._hypothesis_at = 0.0
        engine.latency = LatencyTracker()
        engine.cursor = CommitCursor()

        engine._remember_delays([Word(text=" done", start=1.0, end=1.4)])
        engine._remember_delays([Word(text=" next", start=9.0, end=9.4)])
        engine.cursor.add([Word(text=" done", start=1.0, end=1.4), Word(text=" next", start=9.0, end=9.4)])
        engine._publish()

        live = writer.paths.live.read_text(encoding="utf-8")
        assert "[[" not in live, f"a five second silence must not be replayed: {live}"


def test_the_delays_can_be_turned_off_at_the_engine() -> None:
    from live_transcribe.engine import EngineConfig, StreamingTranscriber
    from live_transcribe.merge import Word

    with tempfile.TemporaryDirectory() as d:
        writer = TranscriptWriter(clean_path=Path(d) / "t.txt", clear_on_start=True)
        engine = StreamingTranscriber.__new__(StreamingTranscriber)
        engine.cfg = EngineConfig(record_word_delays=False)
        engine.writer = writer
        engine.on_text = None
        engine._marks = []
        engine._last_word_end = None
        engine._hypothesis = ""
        engine._hypothesis_at = 0.0
        engine.latency = LatencyTracker()
        engine.cursor = CommitCursor()

        engine._remember_delays([
            Word(text=" hello", start=1.0, end=1.3),
            Word(text=" what", start=1.4, end=1.7),
        ])
        engine.cursor.add([
            Word(text=" hello", start=1.0, end=1.3),
            Word(text=" what", start=1.4, end=1.7),
        ])
        engine._publish()
        assert "[[" not in writer.paths.live.read_text(encoding="utf-8")
        assert writer.paths.live.read_text(encoding="utf-8") == "hello what"


def _run_all() -> int:
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
