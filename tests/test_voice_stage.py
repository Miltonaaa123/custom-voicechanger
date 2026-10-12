"""Unit tests for the voice-changer stage: text splitting, transcript following,
voice discovery and virtual-microphone audio handling.

Run:  .venv\\Scripts\\python.exe tests\\test_voice_stage.py
"""
from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe import voices as voices_mod  # noqa: E402
from live_transcribe.tts import Speech  # noqa: E402
from live_transcribe.virtual_mic import (  # noqa: E402
    apply_fade,
    find_device,
    list_cable_devices,
    pcm16_to_float,
)
from voice_changer import TranscriptFollower, split_complete_sentences  # noqa: E402


# --------------------------------------------------------------------------- #
def test_split_holds_back_incomplete_sentence() -> None:
    """A sentence still being typed must not be spoken yet."""
    ready, leftover = split_complete_sentences("hello there")
    assert ready == [], ready
    assert leftover == "hello there"

    ready, leftover = split_complete_sentences("hello there. and more")
    assert ready == ["hello there."], ready
    assert leftover.strip() == "and more", leftover


def test_split_handles_multiple_and_punctuation() -> None:
    ready, leftover = split_complete_sentences("One. Two! Three? Four")
    assert ready == ["One.", "Two!", "Three?"], ready
    assert leftover.strip() == "Four"


def test_split_streaming_accumulation() -> None:
    """The pattern the watcher uses: append, split, keep the remainder."""
    pending = ""
    spoken: list[str] = []
    for chunk in ["Hel", "lo the", "re. How ", "are you", "? Fine"]:
        pending += chunk
        ready, pending = split_complete_sentences(pending)
        spoken.extend(ready)
    assert spoken == ["Hello there.", "How are you?"], spoken
    assert pending.strip() == "Fine"


def test_split_breaks_very_long_clauses() -> None:
    long_sentence = ("word " * 200).strip() + "."
    ready, leftover = split_complete_sentences(long_sentence)
    assert len(ready) > 1, "a long clause should be split for the model"
    assert all(len(s) <= 340 for s in ready), [len(s) for s in ready]
    assert leftover == ""


# --------------------------------------------------------------------------- #
def test_transcript_follower_reads_only_new_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transcript.txt"
        path.write_text("first.", encoding="utf-8")

        follower = TranscriptFollower(path)
        assert follower.read_new() == "", "existing text must not be replayed"

        with path.open("a", encoding="utf-8") as handle:
            handle.write(" second.")
        assert follower.read_new().strip() == "second."
        assert follower.read_new() == "", "nothing new the second time"


def test_transcript_follower_recovers_from_clear() -> None:
    """transcribe.py truncates the file on restart; the follower must not stall."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transcript.txt"
        path.write_text("some old text that is quite long indeed.", encoding="utf-8")
        follower = TranscriptFollower(path)

        path.write_text("brand new.", encoding="utf-8")  # truncated, not appended
        assert follower.read_new().strip() == "brand new."


def test_transcript_follower_tolerates_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        follower = TranscriptFollower(Path(tmp) / "nope.txt")
        assert follower.read_new() == ""


# --------------------------------------------------------------------------- #
def make_wav(path: Path, seconds: float, rate: int = 24000, channels: int = 1) -> None:
    """A real wav of the requested length, so the size checks are exercised."""
    frames = int(seconds * rate)
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(frames) / rate)).astype("<i2")
    if channels == 2:
        tone = np.repeat(tone, 2)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(tone.tobytes())


def test_voice_discovery_accepts_good_clip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_wav(directory / "good.wav", 6.0)
        found = voices_mod.discover(directory)
        assert len(found) == 1
        voice = found[0]
        assert voice.usable, voice.problem
        assert abs(voice.seconds - 6.0) < 0.05, voice.seconds
        assert voice.sample_rate == 24000


def test_voice_discovery_rejects_too_short_and_too_long() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_wav(directory / "short.wav", 1.0)
        make_wav(directory / "long.wav", 25.0)
        found = {v.slug: v for v in voices_mod.discover(directory)}
        assert not found["short"].usable and "at least" in found["short"].problem
        assert not found["long"].usable and "over" in found["long"].problem


def test_voice_discovery_rejects_non_wav_content() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "fake.wav").write_bytes(b"this is not a wav file at all")
        found = voices_mod.discover(directory)
        assert len(found) == 1 and not found[0].usable
        assert "not a readable wav" in found[0].problem


def test_voice_selection_prefers_default_then_first_usable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_wav(directory / "aaa.wav", 5.0)
        make_wav(directory / "default.wav", 5.0)

        chosen = voices_mod.find(None, directory)
        assert chosen.slug == "default", chosen.slug

        chosen = voices_mod.find("aaa", directory)
        assert chosen.slug == "aaa"

        # A selector may be the filename or the slug.
        assert voices_mod.find("aaa.wav", directory).slug == "aaa"

        try:
            voices_mod.find("missing", directory)
        except KeyError as exc:
            assert "Available" in str(exc) or "available" in str(exc)
        else:
            raise AssertionError("an unknown voice should raise")


def test_voice_selection_skips_unusable_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_wav(directory / "default.wav", 1.0)   # too short to clone
        make_wav(directory / "ok.wav", 6.0)
        assert voices_mod.find(None, directory).slug == "ok"


def test_voice_manifest_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_wav(directory / "v.wav", 6.0)
        path = voices_mod.write_manifest(directory)
        assert path.is_file()

        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "v.wav" in data

        # Editing the name must survive a rewrite (no clobbering).
        data["v.wav"]["name"] = "Custom Name"
        path.write_text(json.dumps(data), encoding="utf-8")
        voices_mod.write_manifest(directory)
        again = json.loads(path.read_text(encoding="utf-8"))
        assert again["v.wav"]["name"] == "Custom Name"

        found = voices_mod.discover(directory)
        assert found[0].name == "Custom Name", found[0].name


# --------------------------------------------------------------------------- #
def test_apply_fade_removes_edge_discontinuities() -> None:
    samples = np.ones(2000, dtype=np.float32)
    faded = apply_fade(samples, 24000, fade_ms=8.0)
    assert faded.size == samples.size
    assert abs(float(faded[0])) < 1e-6, "must start at zero to avoid a click"
    assert abs(float(faded[-1])) < 1e-6, "must end at zero to avoid a click"
    assert abs(float(faded[1000]) - 1.0) < 1e-6, "the middle must be untouched"


def test_apply_fade_handles_tiny_and_empty_input() -> None:
    assert apply_fade(np.zeros(0, dtype=np.float32), 24000).size == 0
    tiny = np.ones(3, dtype=np.float32)
    assert apply_fade(tiny, 24000).size == 3


def test_pcm16_conversion_matches_bit_depth() -> None:
    pcm = np.array([0, 32767, -32768], dtype="<i2").tobytes()
    floats = pcm16_to_float(pcm)
    assert floats.shape == (3,)
    assert abs(float(floats[1]) - 0.99997) < 1e-4
    assert float(floats[2]) == -1.0


def test_cable_devices_are_found_on_this_machine() -> None:
    """VB-CABLE must be present, or the whole premise of stage 2 fails."""
    write_end, mic_end = list_cable_devices()
    assert write_end is not None, "CABLE Input playback device not found"
    assert mic_end is not None, "CABLE Output capture device not found"
    assert find_device("CABLE Input", output=True) is not None
    assert find_device("CABLE Input", output=False) is None, "the write end is playback-only"


def test_speech_wav_roundtrip() -> None:
    speech = Speech(samples=np.zeros(2400, dtype="<i2").tobytes(), sample_rate=24000, seconds=0.1, text="x")
    raw = speech.as_wav_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    import io

    with wave.open(io.BytesIO(raw), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1


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
