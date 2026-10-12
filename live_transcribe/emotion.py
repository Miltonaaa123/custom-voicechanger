"""Guess how someone sounded from the audio, using features that actually track arousal.

No emotion model is involved. The measurements here are the ones that carry most of
the emotional signal in speech and can be computed reliably without a training set:

* **Arousal** (calm <-> excited) from loudness and its variation, and from how much
  the pitch moves.
* **Valence** (negative <-> positive), which is much weaker from acoustics alone.
  Pitch height and speech rate are the usual proxies: a raised, faster voice reads
  as happier, a low, slow, flat one as sad, and a loud one with a high pitch and a
  hard onset as angry.
* **Speech rate** from the syllable rate, which separates calm from sad and helps
  tell anger from fear.

This is a heuristic, and it is honest about that: it reports the features it used so
a wrong guess can be seen for what it is. What it is good at is separating *excited*
from *calm* from *flat*, which is exactly the axis an emotional voice transfer needs.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Categories the rest of the system understands, ordered from high to low arousal.
EMOTIONS = (
    "very happy",   # high arousal, positive
    "happy",
    "excited",
    "neutral",
    "calm",
    "sad",
    "angry",
    "tired",
)


@dataclass
class VoiceFeatures:
    """Measured properties of one utterance."""

    seconds: float = 0.0
    speech_seconds: float = 0.0
    words: int = 0

    pitch_median: float = 0.0
    """Median voiced pitch in Hz."""
    pitch_std: float = 0.0
    """Pitch spread -- how much the melody moves."""
    pitch_range: float = 0.0

    energy_median_db: float = -60.0
    energy_std_db: float = 0.0
    energy_range_db: float = 0.0

    voiced_ratio: float = 0.0
    speech_rate: float = 0.0
    """Words per second of speech (not counting the pauses)."""
    syllable_rate: float = 0.0
    onset_sharpness: float = 0.0
    """How abruptly energy rises -- high for shouted/angry speech."""

    pitch_iqr: float = 0.0
    pitch_movement: float = 0.0
    """Pitch spread from the interquartile range -- robust to octave errors."""
    centroid_hz: float = 0.0
    """Spectral centroid: how bright the voice sounds."""

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class EmotionReading:
    """The verdict, plus enough detail to see why."""

    emotion: str = "neutral"
    confidence: float = 0.0
    arousal: float = 0.0
    """-1 (flat) .. +1 (excited)."""
    valence: float = 0.0
    """-1 (negative) .. +1 (positive)."""
    intensity: float = 0.0
    """0 .. 1, how strongly to apply the style."""
    features: VoiceFeatures = field(default_factory=VoiceFeatures)

    @property
    def tag(self) -> str:
        return f"[{self.emotion}]"

    def describe(self) -> str:
        return (
            f"{self.emotion} (confidence {self.confidence:.2f}, "
            f"arousal {self.arousal:+.2f}, valence {self.valence:+.2f}, "
            f"pitch {self.features.pitch_median:.0f} Hz, "
            f"rate {self.features.speech_rate:.1f} w/s)"
        )


# --------------------------------------------------------------------------- #
# Reading audio
# --------------------------------------------------------------------------- #
def load_wav_mono(path: str | Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read a wav as mono float32 at `target_rate`."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    dtype = {2: "<i2", 4: "<i4", 1: "i1"}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported sample width: {width * 8}-bit")
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 2:
        data /= 32768.0
    elif width == 4:
        data /= 2147483648.0
    else:
        data = (data - 128.0) / 128.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and data.size:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(target_rate, rate)
        data = resample_poly(data, target_rate // g, rate // g).astype(np.float32)
        rate = target_rate
    return np.ascontiguousarray(data, dtype=np.float32), rate


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _frame_signal(signal: np.ndarray, size: int, hop: int) -> np.ndarray:
    if signal.size < size:
        return np.zeros((0, size), dtype=np.float32)
    count = 1 + (signal.size - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return signal[index]


def _autocorrelation_pitch(frames: np.ndarray, rate: int, fmin: float = 70.0, fmax: float = 400.0) -> np.ndarray:
    """Per-frame pitch by autocorrelation. 0 where the frame is unvoiced.

    Autocorrelation is used rather than a neural tracker: it needs no model, runs in
    microseconds per frame, and its failure modes (octave errors, noise) are
    tolerable for the coarse high/low distinction this needs.
    """
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    window = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window
    # Subtract the mean so a DC offset cannot dominate the correlation.
    windowed = windowed - windowed.mean(axis=1, keepdims=True)

    size = 1
    while size < 2 * frames.shape[1]:
        size *= 2
    spectrum = np.fft.rfft(windowed, n=size, axis=1)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, : frames.shape[1]]

    min_lag = max(1, int(rate / fmax))
    max_lag = min(correlation.shape[1] - 1, int(rate / fmin))
    if max_lag <= min_lag:
        return np.zeros(frames.shape[0], dtype=np.float32)

    band = correlation[:, min_lag:max_lag]
    peak = np.argmax(band, axis=1)
    peak_value = band[np.arange(band.shape[0]), peak]
    energy = correlation[:, 0]
    # Voiced when the peak stands clearly above nothing; the 0.3 is a conventional
    # threshold that tolerates breathy speech without admitting noise.
    voiced = (energy > 1e-9) & (peak_value > 0.3 * energy)

    pitch = np.zeros(frames.shape[0], dtype=np.float32)
    lags = peak[min_lag:].astype(np.float32) if False else (peak + min_lag).astype(np.float32)
    pitch[voiced] = rate / np.maximum(lags[voiced], 1.0)
    return pitch


def analyse(signal: np.ndarray, rate: int = 16000, words: int = 0) -> VoiceFeatures:
    """Measure one utterance."""
    features = VoiceFeatures()
    if signal.size == 0 or rate <= 0:
        return features

    features.seconds = signal.size / rate

    frame_len = int(0.032 * rate)   # 32 ms
    hop = int(0.010 * rate)         # 10 ms
    frames = _frame_signal(signal, frame_len, hop)
    if frames.shape[0] == 0:
        return features

    energy = np.sqrt(np.mean(np.square(frames), axis=1))
    energy_db = 20.0 * np.log10(np.maximum(energy, 1e-7))
    pitch = _autocorrelation_pitch(frames, rate)
    voiced = pitch > 0

    features.voiced_ratio = float(np.mean(voiced))
    features.energy_median_db = float(np.median(energy_db))

    # Speech time: frames loud enough to be speech rather than a pause.
    threshold = max(features.energy_median_db - 12.0, -55.0)
    speaking = energy_db > threshold
    features.speech_seconds = float(speaking.sum()) * hop / rate

    # Energy variation is measured over *speech frames only*. Including the pauses
    # between words put a 15-35 dB spread on perfectly ordinary recordings, because
    # the measure was mostly reporting how long the gaps were rather than how the
    # voice behaved. That swamped the arousal score.
    if speaking.sum() >= 3:
        within = energy_db[speaking]
        features.energy_std_db = float(np.std(within))
        features.energy_range_db = float(np.percentile(within, 95) - np.percentile(within, 5))
    else:
        features.energy_std_db = float(np.std(energy_db))
        features.energy_range_db = 0.0

    if voiced.any():
        voiced_pitch = pitch[voiced]
        features.pitch_median = float(np.median(voiced_pitch))
        features.pitch_std = float(np.std(voiced_pitch))
        features.pitch_iqr = float(np.percentile(voiced_pitch, 75) - np.percentile(voiced_pitch, 25))
        features.pitch_range = float(np.percentile(voiced_pitch, 95) - np.percentile(voiced_pitch, 5))
        # Raw pitch spread is inflated by octave errors, which autocorrelation
        # produces on noisy frames. The interquartile range ignores those tails, so
        # "pitch movement" uses it; the raw spread is kept for reporting only.
        features.pitch_movement = features.pitch_iqr

    if features.speech_seconds > 0.2 and words:
        features.speech_rate = words / features.speech_seconds

    # Syllable rate: peaks in the energy envelope, a standard cheap estimate.
    if speaking.any():
        smooth = np.convolve(energy, np.ones(3) / 3.0, mode="same")
        peaks = 0
        for i in range(1, len(smooth) - 1):
            if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > np.mean(smooth):
                peaks += 1
        if features.speech_seconds > 0.2:
            features.syllable_rate = peaks / features.speech_seconds

    # Onset sharpness: the steepest rise in the envelope, normalised by its range.
    if energy_db.size > 2:
        rise = np.diff(energy_db)
        features.onset_sharpness = float(np.percentile(rise, 95)) if rise.size else 0.0

    # Spectral centroid: how "bright" the sound is, in Hz. Sung and lively speech
    # sits higher than heavy, closed speech, which makes it a useful second cue for
    # whether a voice sounds positive. Cheap here because the frames are in hand.
    if frames.shape[0]:
        window = np.hanning(frames.shape[1]).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
        freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / rate)
        total = spectrum.sum(axis=1)
        usable = total > 1e-9
        if usable.any():
            centroids = (spectrum[usable] * freqs).sum(axis=1) / total[usable]
            features.centroid_hz = float(np.median(centroids))

    return features


# --------------------------------------------------------------------------- #
# Turning features into an emotion
# --------------------------------------------------------------------------- #
def _norm(value: float, low: float, high: float) -> float:
    """Map a measurement onto -1..+1 across a plausible speaking range."""
    if high <= low:
        return 0.0
    return max(-1.0, min(1.0, (value - low) / (high - low) * 2.0 - 1.0))


def classify(features: VoiceFeatures) -> EmotionReading:
    """Score arousal and valence, then pick the closest emotion.

    A caveat that matters when reading the result: median pitch differs by speaker,
    so an identical delivery reads as more positive for a higher-pitched voice. The
    bands are wide enough that ordinary speech lands on "neutral" for both, but a
    quiet, high voice can still read happier than it sounds. This is a limit of
    acoustics without a per-speaker baseline, not a bug to be tuned away.
    """
    reading = EmotionReading(features=features)
    if features.seconds < 0.3 or features.voiced_ratio < 0.05:
        reading.emotion = "neutral"
        reading.confidence = 0.0
        return reading

    # Loudness relative to ordinary speech, and how much it swings. The ranges were
    # calibrated against measurement of real speech, not guessed: within-speech
    # energy variation sits around 4-8 dB (a steady voice ~4, a lively one ~8), and
    # the earlier 4-16 dB span clamped ordinary speech to the bottom of the scale.
    loudness = _norm(features.energy_median_db, -45.0, -10.0)
    loudness_swing = _norm(features.energy_std_db, 3.5, 8.0)
    # Pitch height and movement. 200 Hz sits about mid-range for adult speech.
    pitch_height = _norm(features.pitch_median, 110.0, 260.0)
    pitch_movement = _norm(features.pitch_movement, 4.0, 30.0)
    brightness = _norm(features.centroid_hz, 500.0, 3500.0)

    # Arousal is what acoustics measure well, so it carries the decision.
    arousal = float(np.clip(
        0.34 * loudness + 0.26 * loudness_swing + 0.26 * pitch_movement + 0.14 * brightness,
        -1.0, 1.0,
    ))
    # Valence is much weaker from acoustics alone, so it stays a minority signal.
    # It is also confounded by speaker pitch range, which is why it never decides an
    # emotion on its own -- see the note in `classify`.
    valence = float(np.clip(
        0.50 * pitch_height + 0.28 * pitch_movement + 0.22 * brightness,
        -1.0, 1.0,
    ))

    # Speech rate, only when the caller actually knows the word count. It cannot be
    # derived from the audio alone, and inventing it would make the rate term a
    # constant that quietly does nothing.
    if features.speech_rate > 0:
        rate = _norm(features.speech_rate, 1.5, 4.0)
        arousal = float(np.clip(arousal * 0.88 + 0.12 * rate, -1.0, 1.0))
        valence = float(np.clip(valence * 0.85 + 0.15 * rate, -1.0, 1.0))

    reading.arousal = arousal
    reading.valence = valence

    # Bands calibrated so ordinary speech lands on "neutral" and only real
    # variation moves it. Measured on this machine: matter-of-fact speech scores
    # around 0.15, lively speech around 0.45, sustained shouting higher still.
    if arousal > 0.50 and valence > 0.30:
        emotion = "very happy"
    elif arousal > 0.50 and valence < -0.10:
        emotion = "angry"
    elif arousal > 0.32:
        emotion = "happy" if valence > 0.05 else "excited"
    elif arousal > 0.05:
        emotion = "happy" if valence > 0.45 else ("neutral" if valence > -0.20 else "tired")
    elif arousal > -0.25:
        emotion = "calm" if valence > -0.15 else "tired"
    else:
        emotion = "sad" if valence < -0.20 else "calm"

    reading.emotion = emotion
    # Confidence from how far the reading sits from the middle of the scale.
    reading.confidence = float(np.clip(0.35 + 0.45 * abs(arousal) + 0.20 * abs(valence), 0.0, 1.0))
    # Intensity drives how hard the style is pushed: excited speech gets more.
    reading.intensity = float(np.clip(0.25 + 0.75 * abs(arousal), 0.0, 1.0))
    return reading


def read_utterance(path: str | Path, words: int = 0) -> EmotionReading:
    """Convenience: measure and classify a wav in one call."""
    try:
        signal, rate = load_wav_mono(path)
    except (OSError, wave.Error, ValueError) as exc:
        reading = EmotionReading()
        reading.features = VoiceFeatures()
        reading.confidence = 0.0
        reading.emotion = "neutral"
        return reading
    return classify(analyse(signal, rate, words))


def pitch_hz_to_note(hz: float) -> str:
    """For logging: the nearest musical note, which makes pitch readable."""
    if hz <= 0:
        return "-"
    import math as _math

    midi = 69 + 12 * _math.log2(hz / 440.0)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    index = int(round(midi))
    return f"{names[index % 12]}{index // 12 - 1}"


# --------------------------------------------------------------------------- #
# Following the transcriber's utterance audio
# --------------------------------------------------------------------------- #
class EmotionTracker:
    """Watch the utterance wav the transcriber writes and read the emotion in it.

    The two stages are separate processes, so they communicate through a file. The
    transcriber rewrites ``utterance.wav`` once per finished utterance (via a
    temporary file and a rename, so a reader never sees a half-written wav), and
    this notices the change by modification time and analyses it.

    Detection is deliberately best-effort: any failure leaves the previous reading
    in place rather than raising, because a styling problem must never take the
    words with it.
    """

    def __init__(
        self,
        path: str | Path,
        min_seconds: float = 0.35,
        tags_path: str | Path | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.min_seconds = min_seconds
        self.tags_path = Path(tags_path) if tags_path else None
        self.log = logger or (lambda _m: None)
        self._stamp: tuple[float, int] | None = None
        self.reading = EmotionReading()

    def prime(self) -> None:
        """Treat whatever utterance file exists now as already seen.

        Without this, a wav left over from a previous run would be analysed at
        startup and would style the first sentence of this session with an emotion
        nobody just expressed.
        """
        try:
            info = self.path.stat()
        except OSError:
            self._stamp = None
            return
        self._stamp = (info.st_mtime, info.st_size)

    def poll(self) -> EmotionReading | None:
        """Re-analyse the utterance if it changed. Returns the new reading, or None.

        None means "nothing new", which is the common case: this runs in the
        watcher's loop and only does work when the transcriber has finished an
        utterance.
        """
        try:
            info = self.path.stat()
        except OSError:
            return None
        stamp = (info.st_mtime, info.st_size)
        if stamp == self._stamp:
            return None
        self._stamp = stamp

        try:
            signal, rate = load_wav_mono(self.path)
        except (OSError, ValueError, wave.Error) as exc:
            self.log(f"  ! could not read the utterance audio: {exc}")
            return None
        if signal.size < int(self.min_seconds * rate):
            return None

        self.reading = classify(analyse(signal, rate))
        self._write_tag()
        return self.reading

    def _write_tag(self) -> None:
        """Record the current emotion, so other tools (and the user) can see it."""
        if self.tags_path is None:
            return
        try:
            self.tags_path.parent.mkdir(parents=True, exist_ok=True)
            self.tags_path.write_text(
                f"{self.reading.tag} {self.reading.describe()}\n", encoding="utf-8"
            )
        except OSError:
            pass


def format_emotion_line(reading: EmotionReading) -> str:
    """One-line report for the console."""
    return (
        f"  {reading.tag:<14} {reading.emotion:<10} "
        f"pitch {reading.features.pitch_median:5.0f} Hz ({pitch_hz_to_note(reading.features.pitch_median)}), "
        f"arousal {reading.arousal:+.2f}, valence {reading.valence:+.2f}, "
        f"intensity {reading.intensity:.2f}"
    )
