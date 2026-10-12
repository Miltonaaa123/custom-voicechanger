"""Tests for the voice-emotion detector.

Synthetic speech is used rather than recordings so each case has a known ground
truth: a signal built with a high, moving pitch, a fast syllable rate and a loud
envelope *is* excited by construction, so a detector that calls it "sad" is wrong
for a reason that can be pointed at.

The realistic requirement comes first: the detector must recover the pitch of a
signal whose frequency is known, and it must separate loud/excited from
quiet/flat. Exact emotion labels are heuristic, so those tests assert the ordering
and the arousal direction rather than pretending to reach a ground truth that
acoustics alone cannot supply.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe import emotion  # noqa: E402

SR = 16000


def make_voice(
    pitch: float,
    seconds: float = 2.0,
    pitch_swing: float = 0.0,
    pitch_wobble_hz: float = 3.0,
    amplitude: float = 0.05,
    syllables_per_second: float = 4.0,
    jitter: float = 0.0,
    breath: float = 0.0005,
) -> np.ndarray:
    """Build a voiced, speech-shaped signal with the requested properties.

    A sum of harmonics with a syllable-rate envelope: crude, but it gives the pitch
    tracker a clear fundamental and gives the energy envelope real structure to
    measure, which is all the feature extraction needs.

    Default amplitude is deliberately low. Real speech arrives around -25 to -18
    dBFS median here; synthesising at 0.6 puts it at -6 dBFS, which is louder than
    anyone talks and pins the loudness measure to the top of its range, hiding
    whatever the test meant to show.
    """
    t = np.arange(int(seconds * SR)) / SR
    freq = pitch * (1.0 + pitch_swing * 0.35 * np.sin(2 * math.pi * pitch_wobble_hz * t))
    phase = 2 * math.pi * np.cumsum(freq) / SR

    signal = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.1)):
        signal += weight * np.sin(harmonic * phase)

    # Syllable envelope: a raised cosine per syllable, never fully closing, which is
    # what keeps the frame voiced.
    envelope = 0.5 + 0.5 * np.cos(2 * math.pi * syllables_per_second * t)
    envelope = 0.25 + 0.75 * envelope

    # Optional stop-like gaps to exercise the voiced/unvoiced split.
    if jitter > 0:
        rng = np.random.default_rng(7)
        gate = (rng.random(t.size) > jitter).astype(np.float32)
        envelope = envelope * gate

    signal = signal * envelope * amplitude
    if breath:
        signal = signal + np.random.default_rng(3).normal(0, breath, t.size)
    return signal.astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def test_pitch_is_recovered_for_a_known_frequency() -> None:
    """The pitch tracker must find the fundamental it was given."""
    for expected in (110.0, 150.0, 220.0, 300.0):
        signal = make_voice(expected, seconds=1.5)
        features = emotion.analyse(signal, SR)
        assert abs(features.pitch_median - expected) < 12.0, (
            f"expected ~{expected} Hz, measured {features.pitch_median:.1f} Hz"
        )


def test_louder_signal_measures_as_louder() -> None:
    quiet = emotion.analyse(make_voice(180, amplitude=0.05), SR)
    loud = emotion.analyse(make_voice(180, amplitude=0.6), SR)
    assert loud.energy_median_db > quiet.energy_median_db + 10, (
        quiet.energy_median_db,
        loud.energy_median_db,
    )


def test_moving_pitch_measures_as_more_varied() -> None:
    flat = emotion.analyse(make_voice(180, pitch_swing=0.0), SR)
    moving = emotion.analyse(make_voice(180, pitch_swing=0.8), SR)
    assert moving.pitch_std > flat.pitch_std * 1.5, (flat.pitch_std, moving.pitch_std)


def test_silence_is_not_voiced() -> None:
    features = emotion.analyse(np.zeros(SR, dtype=np.float32), SR)
    assert features.voiced_ratio == 0.0
    assert features.pitch_median == 0.0


def test_speech_rate_uses_speaking_time_not_total_time() -> None:
    """A sentence with long pauses must not look slow."""
    signal = make_voice(180, seconds=2.0)
    padded = np.concatenate([signal, np.zeros(SR, dtype=np.float32)])
    features = emotion.analyse(padded, SR, words=8)
    assert features.seconds > 2.5, features.seconds
    # 8 words in ~2 s of actual speech, not 3 s of file.
    assert features.speech_rate > 3.0, features.speech_rate


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_excited_and_flat_are_separated_by_arousal() -> None:
    excited = emotion.classify(emotion.analyse(
        make_voice(260, amplitude=0.30, pitch_swing=0.9, syllables_per_second=6.0), SR, words=9
    ))
    flat = emotion.classify(emotion.analyse(
        make_voice(120, amplitude=0.02, pitch_swing=0.0, syllables_per_second=3.0), SR, words=4
    ))
    assert excited.arousal > flat.arousal + 0.3, (excited.arousal, flat.arousal)
    assert excited.emotion in ("happy", "very happy", "excited"), excited.emotion
    assert flat.emotion in ("calm", "sad", "tired", "neutral"), flat.emotion


def test_loud_low_harsh_speech_reads_as_negative() -> None:
    """Loud with a low pitch is the angry profile, not the happy one.

    Only the valence direction is asserted. Telling anger from mere loudness needs a
    per-speaker pitch baseline that acoustics alone cannot supply, so demanding the
    exact label would be asserting something the method cannot do.
    """
    reading = emotion.classify(emotion.analyse(
        make_voice(100, amplitude=0.35, pitch_swing=0.6, syllables_per_second=5.0), SR, words=10
    ))
    assert reading.valence < 0.25, reading.describe()
    assert reading.emotion in ("angry", "excited", "tired", "neutral"), reading.emotion

    happy_reference = emotion.classify(emotion.analyse(
        make_voice(280, amplitude=0.35, pitch_swing=0.6, syllables_per_second=5.0), SR, words=10
    ))
    assert reading.valence < happy_reference.valence, (reading.describe(), happy_reference.describe())


def test_high_bright_voice_reads_as_positive() -> None:
    reading = emotion.classify(emotion.analyse(
        make_voice(280, amplitude=0.20, pitch_swing=0.7, syllables_per_second=8.0), SR, words=12
    ))
    assert reading.valence > 0.2, reading.valence
    assert reading.emotion in ("very happy", "happy", "excited"), reading.emotion


def test_very_happy_is_reachable_and_is_the_most_intense() -> None:
    """The extreme the user asked for must be possible, not decorative."""
    loud = emotion.classify(emotion.analyse(
        make_voice(300, amplitude=0.45, pitch_swing=1.0, syllables_per_second=6.0), SR, words=14
    ))
    ordinary = emotion.classify(emotion.analyse(
        make_voice(180, amplitude=0.04, pitch_swing=0.2, syllables_per_second=4.0), SR, words=8
    ))
    # The extreme must be reachable, and it must outrank ordinary speech.
    assert loud.arousal > ordinary.arousal, (loud.describe(), ordinary.describe())
    assert loud.intensity > ordinary.intensity, (loud.intensity, ordinary.intensity)
    assert loud.emotion in ("very happy", "happy", "excited"), loud.describe()
    assert ordinary.emotion in ("neutral", "calm", "tired", "sad"), ordinary.describe()


def test_too_short_or_unvoiced_is_neutral_with_no_confidence() -> None:
    reading = emotion.classify(emotion.analyse(make_voice(200, seconds=0.1), SR))
    assert reading.emotion == "neutral"
    assert reading.confidence == 0.0


def test_every_emotion_has_a_tag_and_reads_back() -> None:
    for name in emotion.EMOTIONS:
        reading = emotion.EmotionReading(emotion=name)
        assert reading.tag == f"[{name}]"
        assert name in reading.describe()


def test_intensity_stays_in_range() -> None:
    for amplitude in (0.01, 0.1, 0.4, 0.9):
        for pitch in (90, 150, 240, 340):
            reading = emotion.classify(emotion.analyse(
                make_voice(pitch, amplitude=amplitude, pitch_swing=0.5), SR, words=8
            ))
            assert 0.0 <= reading.intensity <= 1.0, (amplitude, pitch, reading.intensity)
            assert 0.0 <= reading.confidence <= 1.0
            assert -1.0 <= reading.arousal <= 1.0
            assert -1.0 <= reading.valence <= 1.0


def test_analyse_never_raises_on_odd_input() -> None:
    for signal in (
        np.zeros(0, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        np.ones(400, dtype=np.float32),
        np.full(4000, 0.5, dtype=np.float32),
        np.random.default_rng(1).normal(0, 0.2, 9000).astype(np.float32),
    ):
        features = emotion.analyse(signal, SR)
        assert features.seconds >= 0.0
        emotion.classify(features)


def test_note_helper_is_readable() -> None:
    assert emotion.pitch_hz_to_note(440.0) == "A4"
    assert emotion.pitch_hz_to_note(0) == "-"


def _run_all() -> int:
    # Collected explicitly: `globals()` would also pick up local variables inside
    # other tests whose names happen to start with "test_" and call them as tests.
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
