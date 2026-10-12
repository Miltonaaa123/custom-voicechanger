"""Tests for emotion -> voice mapping and for reading the transcriber's utterance wav.

The mapping decides which reference voice an emotion uses, so a mistake there means
the wrong voice comes out of the microphone. The tracker is what notices that the
transcriber has finished an utterance, and it must never raise or re-analyse the
same audio twice.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from live_transcribe import emotion  # noqa: E402
from live_transcribe import emotion_voices as ev  # noqa: E402


def write_wav(path: Path, seconds: float = 1.0, freq: float = 200.0, rate: int = 16000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    signal = 0.2 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 4 * t))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.clip(signal * 32767, -32768, 32767).astype("<i2").tobytes())
    return path


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def test_slug_is_filename_safe() -> None:
    assert ev.slug_for("very happy") == "very_happy"
    assert ev.slug_for("  Happy  ") == "happy"
    assert ev.slug_for("very-happy") == "very_happy"


def test_no_emotion_voices_means_nothing_is_offered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mapping = ev.load(tmp)
        assert mapping.emotions_covered == []
        assert mapping.get("happy") is None
        assert "no emotion voices" in mapping.describe()


def test_a_wav_named_after_the_emotion_is_picked_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        write_wav(voices / "emotions" / "happy.wav", seconds=4.0)
        mapping = ev.load(voices)
        assert "happy" in mapping.emotions_covered
        assert mapping.get("happy") is not None
        assert mapping.source["happy"] == "file"


def test_extremes_fall_back_to_their_neighbour() -> None:
    """One "happy" file must serve "very happy" too, or the feature needs 8 files."""
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        write_wav(voices / "emotions" / "happy.wav", seconds=4.0)
        mapping = ev.load(voices)
        assert mapping.get("very happy") is not None
        assert mapping.get("very happy") == mapping.get("happy")
        # An emotion with no relative and no file stays absent.
        assert mapping.get("angry") is None


def test_mapping_file_can_point_anywhere() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        target = write_wav(voices / "my_voice.wav", seconds=5.0)
        (voices / "emotions.json").write_text(
            json.dumps({"sad": "my_voice.wav"}), encoding="utf-8"
        )
        mapping = ev.load(voices)
        assert mapping.get("sad") == target
        assert mapping.source["sad"] == "mapping"


def test_mapping_file_ignores_unknown_emotions_and_missing_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        (voices / "emotions.json").write_text(
            json.dumps({"delirious": "x.wav", "sad": "does_not_exist.wav"}), encoding="utf-8"
        )
        mapping = ev.load(voices)
        assert "delirious" not in mapping.emotions_covered
        assert mapping.get("sad") is None


def test_broken_mapping_file_is_not_fatal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        (voices / "emotions.json").write_text("{not json at all", encoding="utf-8")
        mapping = ev.load(voices)
        assert mapping.emotions_covered == []


def test_write_mapping_stores_paths_relative_to_the_voices_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        write_wav(voices / "emotions" / "happy.wav", seconds=4.0)
        path = ev.write_mapping(voices, {"happy": str(voices / "emotions" / "happy.wav")})
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["happy"] == "emotions/happy.wav"
        # And it round-trips.
        assert ev.load(voices).get("happy") is not None


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
def test_tracker_notices_a_new_utterance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utterance.wav"
        write_wav(path, seconds=1.5, freq=250.0)
        tracker = emotion.EmotionTracker(path)
        first = tracker.poll()
        assert first is not None, "a fresh utterance should be analysed"
        assert first.emotion in emotion.EMOTIONS

        # Unchanged file: nothing to do.
        assert tracker.poll() is None

        # Rewritten: analysed again.
        time.sleep(0.02)
        write_wav(path, seconds=1.5, freq=140.0)
        assert tracker.poll() is not None


def test_tracker_ignores_a_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tracker = emotion.EmotionTracker(Path(tmp) / "nothing.wav")
        assert tracker.poll() is None
        assert tracker.reading.emotion == "neutral"


def test_prime_suppresses_a_stale_utterance() -> None:
    """A wav left from a previous run must not style the first sentence."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utterance.wav"
        write_wav(path, seconds=1.5, freq=300.0)
        tracker = emotion.EmotionTracker(path)
        tracker.prime()
        assert tracker.poll() is None, "a primed tracker should ignore the old file"
        time.sleep(0.02)
        write_wav(path, seconds=1.5, freq=130.0)
        assert tracker.poll() is not None, "but a new one must still be noticed"


def test_tracker_skips_audio_too_short_to_judge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utterance.wav"
        write_wav(path, seconds=0.1, freq=250.0)
        tracker = emotion.EmotionTracker(path)
        assert tracker.poll() is None


def test_tracker_writes_the_tag_for_other_tools() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utterance.wav"
        tags = Path(tmp) / "emotion.txt"
        write_wav(path, seconds=2.0, freq=250.0)
        tracker = emotion.EmotionTracker(path, tags_path=tags)
        tracker.poll()
        assert tags.is_file()
        text = tags.read_text(encoding="utf-8")
        assert text.startswith("[")
        assert "pitch" in text


def test_tracker_survives_an_unreadable_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utterance.wav"
        path.write_bytes(b"this is not a wav file at all")
        tracker = emotion.EmotionTracker(path)
        # Must not raise: a styling problem cannot be allowed to stop the words.
        assert tracker.poll() is None


def test_format_emotion_line_is_readable() -> None:
    reading = emotion.classify(emotion.analyse(np.zeros(16000, dtype=np.float32), 16000))
    line = emotion.format_emotion_line(reading)
    assert "[neutral]" in line
    assert "arousal" in line


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
