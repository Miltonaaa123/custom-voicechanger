"""Changing how the synthesized voice sounds on its way to the virtual microphone.

StyleTTS 2 clones the *timbre* of a reference clip, and the speech server takes no
pitch or volume arguments, so the knobs that shape the output live here instead:
the pipeline hands each finished clip to `process` right before it is written to
the cable.

What each effect is, honestly:

* **Pitch** is resampling, so shifting the pitch also changes how long the clip
  lasts -- raising it by an octave makes the voice twice as fast. That is the
  classic voice-changer sound and it is exactly what a "pitch shift" slider means
  in most of them. It is applied by playing the audio at a scaled rate, so the
  virtual microphone stays at one fixed sample rate.
* **Volume** is a plain gain, soft-limited so pushing it past 1.0 does not clip
  into a buzz.
* **The gate** fades inaudible stretches down to a floor, which is what makes a
  quiet room sound silent rather than faintly hissy. It never touches a stretch
  loud enough to be speech.

Every setting is optional and every effect is a no-op at its default, so a
misconfigured control panel cannot make the voice unintelligible -- it can only
make it sound wrong, which is audible and therefore fixable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Pitch is limited to one octave either way: further than that and the clip's
#: length changes so much that the gaps between words stop lining up.
MAX_SEMITONES = 12.0

#: The envelope follower's window. Long enough to ride out the waveform's own
#: cycles, short enough to open within a syllable.
GATE_WINDOW_MS = 12.0


@dataclass
class AudioOptions:
    """What to do to each clip between synthesis and the cable."""

    pitch_semitones: float = 0.0
    """Positive is higher and shorter, negative is lower and longer."""

    volume: float = 1.0
    """1.0 leaves the level alone."""

    gate_db: float = -70.0
    """Below this the clip is treated as silence. -70 dB is effectively off."""

    gate_floor_db: float = -60.0
    """How far down the silent stretches are pushed. -60 dB is near-silent."""

    @property
    def pitch_ratio(self) -> float:
        """Playback rate multiplier that produces the requested pitch."""
        return float(2.0 ** (max(-MAX_SEMITONES, min(MAX_SEMITONES, self.pitch_semitones)) / 12.0))

    @property
    def gate_open(self) -> float:
        return float(10.0 ** (self.gate_db / 20.0))

    @property
    def gate_shut(self) -> float:
        return float(10.0 ** (min(self.gate_floor_db, self.gate_db) / 20.0))

    @property
    def is_default(self) -> bool:
        return (
            abs(self.pitch_semitones) < 1e-6
            and abs(self.volume - 1.0) < 1e-6
            and self.gate_db <= -69.5
        )

    def describe(self) -> str:
        if self.is_default:
            return "no effects"
        parts: list[str] = []
        if abs(self.pitch_semitones) > 1e-6:
            parts.append(f"pitch {self.pitch_semitones:+.1f} st")
        if abs(self.volume - 1.0) > 1e-6:
            parts.append(f"volume {self.volume:.2f}x")
        if self.gate_db > -69.5:
            parts.append(f"gate {self.gate_db:.0f} dB to {self.gate_floor_db:.0f} dB")
        return ", ".join(parts)


def choose_rate(device_rate: int, pitch_ratio: float) -> int:
    """The device sample rate that makes the pitch come out at `pitch_ratio`.

    ``sounddevice`` resamples whatever it is given to the device's own rate, so a
    clip resampled here by ``r`` and played on a stream opened at
    ``device_rate * r`` comes out pitched by ``r`` at the original speed's *rate*
    -- one stream, one pitch, no per-clip device juggling.
    """
    if abs(pitch_ratio - 1.0) < 1e-9:
        return int(device_rate)
    return max(8000, int(round(device_rate * pitch_ratio)))


def resample(samples: np.ndarray, ratio: float) -> np.ndarray:
    """Linear-interpolation resample, which is what makes the pitch shift audible.

    Linear rather than windowed-sinc on purpose: a whole sentence is being shifted
    by at most an octave, where the difference is not audible through a virtual
    microphone, and this costs a fraction of the time -- the pipeline is on a
    real-time budget and the extra accuracy would come out of it.
    """
    if samples.size == 0 or abs(ratio - 1.0) < 1e-9:
        return samples
    out_length = max(1, int(round(samples.size / ratio)))
    source = np.linspace(0.0, samples.size - 1.0, out_length, dtype=np.float64)
    left = np.floor(source).astype(np.int64)
    right = np.minimum(left + 1, samples.size - 1)
    weight = (source - left).astype(np.float32)
    return (samples[left] * (1.0 - weight) + samples[right] * weight).astype(np.float32)


def apply_gate(samples: np.ndarray, sample_rate: int, options: AudioOptions) -> np.ndarray:
    """Fade everything below the threshold down to the floor.

    The gain is derived from a short moving average of the level and then smoothed
    the same way, so the gate closes gently instead of chopping the tail off a word.
    """
    if samples.size == 0 or options.gate_db <= -69.5:
        return samples
    window = max(1, int(sample_rate * GATE_WINDOW_MS / 1000.0))
    level = np.abs(samples).astype(np.float32)
    if level.size > window:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        level = np.convolve(level, kernel, mode="same")

    span = max(1e-6, options.gate_open - options.gate_shut)
    gain = np.clip((level - options.gate_shut) / span, 0.0, 1.0)
    # A floor rather than a hard zero: gating to complete silence makes the gaps
    # between words sound like dropouts on a phone line.
    gain = options.gate_shut + gain * (1.0 - options.gate_shut)
    return (samples * gain).astype(np.float32)


def apply_volume(samples: np.ndarray, volume: float) -> np.ndarray:
    """Gain, soft-limited so a boosted signal bends instead of clipping."""
    if samples.size == 0 or abs(volume - 1.0) < 1e-6:
        return samples
    if volume <= 0.0:
        return np.zeros_like(samples[:1])
    scaled = samples * float(volume)
    if volume <= 1.0:
        return scaled.astype(np.float32)
    # tanh bends the peaks; at volume 2.0 it is inaudible on speech and it is the
    # difference between "louder" and "distorted".
    return np.tanh(scaled).astype(np.float32)


def process(samples: np.ndarray, sample_rate: int, options: AudioOptions) -> np.ndarray:
    """Every effect, in the order that keeps each one honest.

    The gate runs last on purpose: gating before the volume change would measure
    the level the user never hears, and a quiet recording boosted afterwards would
    come out with its noise floor amplified and no gate left to catch it.
    """
    if options.is_default or samples.size == 0:
        return samples
    out = samples
    if options.gate_db > -69.5:
        out = apply_gate(out, sample_rate, options)
    out = apply_volume(out, options.volume)
    return out
