"""Playing synthesized speech into a virtual microphone.

VB-CABLE is a loopback pair: anything written to the playback device
``CABLE Input`` appears at the capture device ``CABLE Output``, which Windows and
every other application sees as an ordinary microphone. So "becoming a microphone"
is really just: play the audio to the right output device.

Two details matter for it to sound right rather than like a glitch:

* **Fade the edges.** Speech that starts or ends on a non-zero sample clicks
  audibly. Every clip is faded in and out over a few milliseconds.
* **Pad the gaps.** The cable is a live stream, so silence has to be written
  between clips; otherwise the previous audio is simply held and repeats.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# VB-CABLE's endpoints, matched by name so this works whatever the indices are.
CABLE_OUTPUT_NAME = "CABLE Input"   # the playback device we write into
CABLE_MIC_NAME = "CABLE Output"     # the capture device apps record from

FADE_MS = 8.0
WRITE_SLICE_MS = 100.0
"""Audio is written in slices this long, so an abort can interrupt quickly."""


class VirtualMicError(RuntimeError):
    pass


@dataclass
class DeviceInfo:
    index: int
    name: str
    sample_rate: int
    channels: int


def find_device(name_part: str, *, output: bool) -> DeviceInfo | None:
    """First device whose name contains `name_part` and has the wanted direction."""
    wanted = "max_output_channels" if output else "max_input_channels"
    for index, device in enumerate(sd.query_devices()):
        if device[wanted] <= 0:
            continue
        if name_part.lower() in device["name"].lower():
            return DeviceInfo(
                index=index,
                name=device["name"],
                sample_rate=int(device["default_samplerate"]),
                channels=int(device[wanted]),
            )
    return None


def resolve_virtual_mic(name: str = CABLE_OUTPUT_NAME) -> DeviceInfo:
    """The playback device that feeds the virtual microphone."""
    found = find_device(name, output=True)
    if found is None:
        raise VirtualMicError(
            f"no playback device matching {name!r}. VB-CABLE is required:\n"
            f"  install C:\\Program Files\\VB\\CABLE\\VBCABLE_Setup_x64.exe as administrator, "
            f"reboot, then run tools\\vbcable_check.py"
        )
    return found


def list_cable_devices() -> tuple[DeviceInfo | None, DeviceInfo | None]:
    return (
        find_device(CABLE_OUTPUT_NAME, output=True),
        find_device(CABLE_MIC_NAME, output=False),
    )


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def apply_fade(samples: np.ndarray, sample_rate: int, fade_ms: float = FADE_MS) -> np.ndarray:
    """Ramp the first and last few milliseconds so clips do not click."""
    if samples.size == 0:
        return samples
    fade = int(sample_rate * fade_ms / 1000.0)
    fade = max(1, min(fade, samples.size // 2))
    if fade <= 1:
        return samples
    out = samples.copy()
    out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = 0.004,
    keep_ms: float = 25.0,
    max_trim_ms: float = 500.0,
) -> np.ndarray:
    """Remove dead air from the start and end of a clip.

    StyleTTS 2 pads every call with silence, which is harmless for a whole
    sentence but ruins word-by-word speech: a one-word call for "the" came back as
    2.2 s of audio, nearly all of it silence, so consecutive words sounded like
    they had long pauses between them.

    The trim is capped at `max_trim_ms` per side so a quiet word cannot be eaten,
    and `keep_ms` of silence is left behind so the speech does not sound clipped.
    """
    if samples.size == 0:
        return samples

    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        # Nothing above the threshold: leave it alone rather than deleting audio.
        return samples

    keep = int(sample_rate * keep_ms / 1000.0)
    cap = int(sample_rate * max_trim_ms / 1000.0)
    first, last = int(loud[0]), int(loud[-1])

    start = max(0, first - keep)
    if first - start > cap:
        start = first - cap
    end = min(samples.size, last + 1 + keep)
    if end - (last + 1) > cap:
        end = last + 1 + cap

    if end <= start:
        return samples
    return samples[start:end]


class VirtualMic:
    """A live output stream into the virtual microphone.

    Held open across clips so the cable sees one continuous device rather than
    repeated open/close cycles (which some applications experience as the
    microphone dropping out).
    """

    def __init__(
        self,
        device: DeviceInfo | None = None,
        sample_rate: int = 24000,
        silence_ms: float = 60.0,
        monitor: bool = False,
        rate_scale: float = 1.0,
    ) -> None:
        self.device = device or resolve_virtual_mic()
        self.sample_rate = int(sample_rate)
        self.silence_ms = float(silence_ms)
        self.monitor = monitor
        # Pitch is applied as a playback rate, so the stream is opened at the
        # scaled rate: the cable still receives a normal-speed stream, and the
        # pitch shift happens because the clip is resampled on the way in (see
        # `live_transcribe.audio_effects.choose_rate`). Opening a second stream per
        # pitch change is what this avoids.
        self.rate_scale = float(rate_scale) if rate_scale > 0 else 1.0
        self._stream: sd.OutputStream | None = None
        self._monitor_stream: sd.OutputStream | None = None
        self._abort = threading.Event()
        """Set to abandon a write in progress, so shutdown cannot hang on a long
        buffer. Threading (not asyncio) is already used elsewhere here."""
        self.blocks_written = 0
        self.seconds_written = 0.0
        self.underruns = 0

    def request_abort(self) -> None:
        """Ask any in-progress write to return early."""
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    # -- lifecycle --------------------------------------------------------- #
    def __enter__(self) -> "VirtualMic":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def device_rate(self) -> int:
        """The rate the output stream is actually opened at."""
        return max(8000, int(round(self.sample_rate * self.rate_scale)))

    def open(self) -> None:
        if self._stream is not None:
            return
        rate = self.device_rate
        try:
            self._stream = sd.OutputStream(
                device=self.device.index,
                samplerate=rate,
                channels=1,
                dtype="float32",
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise VirtualMicError(
                f"could not open {self.device.name!r} at {rate} Hz: {exc}"
            ) from exc

        if self.monitor:
            # Optional: hear the voice locally too, so it can be checked without
            # another application recording the cable.
            try:
                default_out = sd.default.device[1]
                self._monitor_stream = sd.OutputStream(
                    device=default_out, samplerate=rate, channels=1, dtype="float32"
                )
                self._monitor_stream.start()
            except Exception:  # noqa: BLE001
                self._monitor_stream = None

    def close(self) -> None:
        for stream in (self._stream, self._monitor_stream):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._stream = None
        self._monitor_stream = None

    # -- output ------------------------------------------------------------ #
    def write(self, samples: np.ndarray) -> None:
        """Blocking write; returns once the device has taken the audio.

        Written in slices so an abort can cut it short: a single write of several
        seconds blocks the whole time, which would make stopping the voice changer
        wait for the sentence to finish.
        """
        if self._stream is None:
            raise VirtualMicError("the virtual microphone is not open")
        if samples.size == 0:
            return

        chunk = max(1, int(self.sample_rate * WRITE_SLICE_MS / 1000.0))
        for start in range(0, samples.size, chunk):
            if self._abort.is_set():
                break
            block = samples[start : start + chunk].reshape(-1, 1).astype(np.float32, copy=False)
            try:
                self._stream.write(block)
            except Exception as exc:  # noqa: BLE001
                self.underruns += 1
                raise VirtualMicError(f"writing to the virtual microphone failed: {exc}") from exc
            if self._monitor_stream is not None:
                try:
                    self._monitor_stream.write(block)
                except Exception:  # noqa: BLE001
                    pass
            self.blocks_written += 1
            self.seconds_written += block.shape[0] / float(self.sample_rate)

    def write_silence(self, milliseconds: float | None = None) -> None:
        ms = self.silence_ms if milliseconds is None else milliseconds
        self.write(np.zeros(int(self.sample_rate * ms / 1000.0), dtype=np.float32))

    def speak(self, pcm16: bytes, leading_silence: bool = True) -> float:
        """Play one synthesized clip. Returns the audio duration in seconds."""
        samples = pcm16_to_float(pcm16)
        samples = apply_fade(samples, self.sample_rate)
        if leading_silence:
            self.write_silence()
        self.write(samples)
        return samples.size / float(self.sample_rate)

    def drain(self, milliseconds: float = 250.0) -> None:
        """Let the device's buffer empty before closing, so the tail is not cut."""
        self.write_silence(milliseconds)
        time.sleep(milliseconds / 1000.0)
