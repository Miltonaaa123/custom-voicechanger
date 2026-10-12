"""Microphone capture, streaming resampling and a lightweight energy VAD.

The transcriber needs 16 kHz mono float32. Microphones here typically expose
44.1/48 kHz, so audio is resampled continuously while it streams rather than
after the fact (the stream never ends).
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Streaming resampler
# --------------------------------------------------------------------------- #
class StreamingResampler:
    """Rate conversion for an endless stream.

    ``resample_poly`` is applied block-wise with an overlap so that each block
    sees the samples before it. Output is then trimmed to exactly the number of
    samples the input timing implies, which keeps the stream gap-free and
    drift-free without carrying scipy's internal filter state (that state is not
    exposed in a form that survives an input-length change).
    """

    def __init__(self, src_sr: int, dst_sr: int = TARGET_SR) -> None:
        if src_sr <= 0 or dst_sr <= 0:
            raise ValueError("sample rates must be positive")
        self.src_sr = int(src_sr)
        self.dst_sr = int(dst_sr)
        g = math.gcd(self.src_sr, self.dst_sr)
        self.up = self.dst_sr // g
        self.down = self.src_sr // g
        self.passthrough = self.src_sr == self.dst_sr

        # Input history retained so block boundaries are filtered identically to
        # a single-shot resample. ~20 ms is comfortably past the filter's reach.
        self._history_len = max(64, int(round(0.020 * self.src_sr)))
        self._history = np.zeros(0, dtype=np.float32)

        # Fractional input position of the next output sample.
        self._next_pos = 0.0
        self._ratio = self.src_sr / self.dst_sr

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32).ravel()
        if self.passthrough:
            self._next_pos += len(block)
            return block
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        work = np.concatenate((self._history, block))
        base = len(self._history)

        from scipy.signal import resample_poly

        filtered = resample_poly(work, self.up, self.down).astype(np.float32, copy=False)

        # Output-sample index that corresponds to the start of `work`.
        # Each output sample k sits at fractional input position k / up * down.
        scale = self.up / self.down  # output samples per input sample
        # We need the output samples that fall at or after input offset `base`.
        first_out = int(math.ceil(base * scale))

        # Emit as many samples as the consumed input justifies.
        self._next_pos += len(block)
        want = int(math.floor(self._next_pos * scale)) - first_out
        if want <= 0:
            out = np.zeros(0, dtype=np.float32)
        else:
            out = filtered[first_out : first_out + want]

        # Keep the tail as history for the next block.
        if len(work) > self._history_len:
            self._history = work[-self._history_len :].copy()
        else:
            self._history = work.copy()

        # Drop the input samples that will never be needed again so `_next_pos`
        # stays a small number instead of growing without bound.
        consumed = len(work) - len(self._history)
        if consumed > 0:
            self._next_pos -= consumed

        return np.ascontiguousarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Energy VAD with hysteresis
# --------------------------------------------------------------------------- #
@dataclass
class VadConfig:
    threshold_db: float = -42.0
    """Speech is declared when the block RMS exceeds this level."""
    release_db: float = -50.0
    """Speech ends only once RMS falls below this (hysteresis)."""
    hangover_s: float = 0.35
    """Silence tolerated before the utterance is considered finished."""

    @property
    def threshold(self) -> float:
        return float(10.0 ** (self.threshold_db / 20.0))

    @property
    def release(self) -> float:
        return float(10.0 ** (self.release_db / 20.0))


def block_rms_db(block: np.ndarray) -> float:
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-9))


def speech_level_db(block: np.ndarray, window_samples: int = 800) -> float:
    """Loudest short-term level in a buffer (default 50 ms at 16 kHz).

    Used to decide whether a window to be decoded actually contains speech.
    Speech has a strongly modulated envelope and always contains a loud 50 ms
    somewhere; room tone does not. Decoding a speechless window is what makes
    Whisper invent words, so this gates those windows out.
    """
    if block.size == 0:
        return -120.0
    window_samples = max(1, int(window_samples))
    if block.size <= window_samples:
        return block_rms_db(block)
    # Strided view of consecutive windows; cheap and allocation-light.
    n = block.size // window_samples
    trimmed = block[: n * window_samples].reshape(n, window_samples)
    rms = np.sqrt(np.mean(np.square(trimmed.astype(np.float64)), axis=1))
    return float(20.0 * np.log10(max(float(rms.max()), 1e-9)))


def voiced_fraction(block: np.ndarray, threshold_db: float, window_samples: int = 800) -> float:
    """Fraction of short windows above `threshold_db`."""
    if block.size == 0:
        return 0.0
    window_samples = max(1, int(window_samples))
    n = max(1, block.size // window_samples)
    trimmed = block[: n * window_samples].reshape(n, window_samples)
    if trimmed.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.square(trimmed.astype(np.float64)), axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return float(np.mean(db >= threshold_db))


class EnergyVad:
    """Simple, dependency-free voice activity detector.

    Good enough to avoid running Whisper on an idle room, and it gives the
    engine a "speaker stopped" signal so the last words can be committed.
    """

    def __init__(self, cfg: VadConfig | None = None) -> None:
        self.cfg = cfg or VadConfig()
        self.speaking = False
        self._silence_run = 0.0

    def update(self, block: np.ndarray, sr: int) -> tuple[bool, bool]:
        """Feed a block; returns ``(speaking_now, just_stopped)``."""
        level = block_rms_db(block)
        dur = len(block) / float(sr)
        just_stopped = False

        if level >= self.cfg.threshold_db:
            self.speaking = True
            self._silence_run = 0.0
        elif self.speaking:
            if level < self.cfg.release_db:
                self._silence_run += dur
                if self._silence_run >= self.cfg.hangover_s:
                    self.speaking = False
                    self._silence_run = 0.0
                    just_stopped = True
            # Between release and threshold: keep speaking, hold the silence timer.
        return self.speaking, just_stopped


def supports_input_rate(device_index: int, rate: int) -> bool:
    """Ask PortAudio whether this device accepts `rate` for mono input."""
    try:
        sd.check_input_settings(device=device_index, channels=1, samplerate=rate, dtype="int16")
        return True
    except Exception:  # noqa: BLE001
        return False


def calibrate_noise_floor(
    device: int | str | None = None,
    seconds: float = 2.0,
    margin_db: float = 12.0,
) -> tuple[float, float]:
    """Measure the room's noise floor and suggest a speech threshold.

    Returns ``(noise_floor_db, suggested_threshold_db)``. A fixed threshold of
    -42 dBFS is fine on a quiet desk but wrong for a noisy room or a hot mic
    gain, so the default is measured at startup instead.
    """
    device_index = MicCapture._resolve_device(device)["index"]
    native = int(sd.query_devices(device_index)["default_samplerate"])
    rate = TARGET_SR if supports_input_rate(device_index, TARGET_SR) else native
    frames = max(256, rate // 50)
    levels: list[float] = []

    def cb(indata, _frames, _time, status) -> None:  # noqa: ANN001
        pcm = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size:
            levels.append(block_rms_db(pcm))

    stream = sd.RawInputStream(
        samplerate=rate,
        blocksize=frames,
        device=device_index,
        channels=1,
        dtype="int16",
        callback=cb,
    )
    with stream:
        time.sleep(seconds)

    if not levels:
        return -60.0, -42.0
    # Median is robust against a cough or a door slam during calibration.
    floor_db = float(np.median(levels))
    threshold = max(floor_db + margin_db, -65.0)
    return floor_db, threshold


# --------------------------------------------------------------------------- #
# Ring buffer of recent audio
# --------------------------------------------------------------------------- #
class AudioRing:
    """Fixed-capacity mono float32 buffer holding the most recent audio."""

    def __init__(self, capacity_samples: int) -> None:
        self._buf = np.zeros(max(1, int(capacity_samples)), dtype=np.float32)
        self._cap = len(self._buf)
        self._len = 0
        self._start = 0
        self._total = 0
        self._lock = threading.Lock()

    def write(self, block: np.ndarray) -> None:
        block = np.ascontiguousarray(block, dtype=np.float32).ravel()
        n = len(block)
        if n == 0:
            return
        if n >= self._cap:
            block = block[-self._cap :]
            n = self._cap
        with self._lock:
            end = (self._start + self._len) % self._cap
            # First chunk fills to the physical end of the buffer.
            first = min(n, self._cap - end)
            self._buf[end : end + first] = block[:first]
            if n > first:
                self._buf[: n - first] = block[first:]

            over = self._len + n - self._cap
            if over > 0:
                # Oldest samples are being overwritten: slide the read start.
                self._start = (self._start + over) % self._cap
                self._len = self._cap
            else:
                self._len += n
            # Count what was actually retained, not what was offered, so
            # absolute positions stay meaningful after a capacity overflow.
            self._total += n

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total

    def read_last(self, n: int) -> np.ndarray:
        with self._lock:
            n = min(int(n), self._len)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._start + self._len - n) % self._cap
            if start + n <= self._cap:
                return self._buf[start : start + n].copy()
            first = self._cap - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: n - first]
            return out

    def read_range(self, start_pos: int, end_pos: int) -> np.ndarray:
        """Read absolute sample positions (as returned by ``total_written``)."""
        with self._lock:
            start_pos = max(int(start_pos), self._total - self._len)
            end_pos = min(int(end_pos), self._total)
            if end_pos <= start_pos:
                return np.zeros(0, dtype=np.float32)
            n = end_pos - start_pos
            offset_from_oldest = start_pos - (self._total - self._len)
            start = (self._start + offset_from_oldest) % self._cap
            if start + n <= self._cap:
                return self._buf[start : start + n].copy()
            first = self._cap - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: n - first]
            return out


# --------------------------------------------------------------------------- #
# Microphone
# --------------------------------------------------------------------------- #
class MicError(RuntimeError):
    pass


class MicCapture:
    """Continuous raw microphone capture on a PortAudio callback thread."""

    def __init__(
        self,
        device: int | str | None = None,
        target_sr: int = TARGET_SR,
        blocksize: int = 0,
        ring_seconds: float = 90.0,
        vad: VadConfig | None = None,
        gain_db: float = 0.0,
        max_gain_factor: float = 12.0,
    ) -> None:
        self.device = device
        self.target_sr = int(target_sr)
        info = self._resolve_device(device)
        self.device_index = info["index"]
        self.device_name = info["name"]

        native = int(info["default_samplerate"])
        # WASAPI handles 16 kHz directly on most devices and avoids resampling.
        self.native_sr = self.target_sr if self._supports_rate(info, self.target_sr) else native

        self._resampler = StreamingResampler(self.native_sr, self.target_sr)
        self._ring = AudioRing(int(ring_seconds * self.target_sr))
        self.vad = EnergyVad(vad)

        # Software gain for microphones whose hardware level is very low. This
        # runs before the VAD, so the threshold is measured on the gained signal.
        self.gain_db = float(gain_db)
        self._gain = float(10.0 ** (self.gain_db / 20.0)) if gain_db else 1.0
        self.clip_factor = float(max_gain_factor)
        self.clipped_blocks = 0

        self._blocksize = int(blocksize) if blocksize else max(256, self.native_sr // 100)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.RawInputStream | None = None
        self._lock = threading.Lock()

        self.blocks_seen = 0
        self.last_level_db = -120.0
        self.overflows = 0
        self.speech_active = False
        self.speech_stopped_at: float | None = None
        self._last_speech_time: float | None = None
        self.audio_seconds = 0.0

    # -- device helpers ---------------------------------------------------- #
    @staticmethod
    def _resolve_device(device: int | str | None) -> dict:
        try:
            if device is None:
                idx = sd.default.device[0]
                if idx is None or idx < 0:
                    # Fall back to the first device that can record.
                    for i, d in enumerate(sd.query_devices()):
                        if d["max_input_channels"] > 0:
                            idx = i
                            break
                if idx is None or idx < 0:
                    raise MicError("no input device found")
                info = sd.query_devices(idx)
                return {"index": idx, "name": info["name"], **info}

            if isinstance(device, str):
                needle = device.lower()
                for i, d in enumerate(sd.query_devices()):
                    if d["max_input_channels"] > 0 and needle in d["name"].lower():
                        return {"index": i, "name": d["name"], **d}
                raise MicError(f"no input device matches {device!r}")

            info = sd.query_devices(int(device))
            if info["max_input_channels"] <= 0:
                raise MicError(f"device {device} has no input channels")
            return {"index": int(device), "name": info["name"], **info}
        except MicError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MicError(f"cannot open audio device {device!r}: {exc}") from exc

    def _supports_rate(self, info: dict, rate: int) -> bool:
        return supports_input_rate(info["index"], rate)

    # -- stream lifecycle -------------------------------------------------- #
    def start(self) -> None:
        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                if getattr(status, "input_overflow", False):
                    self.overflows += 1
            # Copy: PortAudio reuses this buffer once we return.
            self._queue.put(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.native_sr,
                blocksize=self._blocksize,
                device=self.device_index,
                channels=1,
                dtype="int16",
                latency="low",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise MicError(
                f"failed to start microphone '{self.device_name}' "
                f"({self.native_sr} Hz): {exc}"
            ) from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    # -- data path --------------------------------------------------------- #
    def pump(self, max_seconds: float = 0.25) -> int:
        """Drain queued callback buffers into the ring. Returns samples added.

        Must be called from the consumer thread; the PortAudio callback only
        pushes bytes so it never blocks on our processing.
        """
        deadline = time.perf_counter() + max_seconds
        added = 0
        while True:
            try:
                raw = self._queue.get_nowait()
            except queue.Empty:
                if added == 0 and time.perf_counter() < deadline:
                    time.sleep(0.002)
                    continue
                break
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size == 0:
                continue
            resampled = self._resampler.process(pcm)
            if resampled.size == 0:
                continue
            if self._gain != 1.0:
                resampled = resampled * self._gain
                peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
                if peak > 1.0:
                    # Soft-limit instead of hard clipping: a clipped waveform
                    # transcribes far worse than a slightly compressed one.
                    resampled = np.tanh(resampled).astype(np.float32)
                    self.clipped_blocks += 1
                else:
                    resampled = resampled.astype(np.float32, copy=False)
            self.blocks_seen += 1
            self.audio_seconds += resampled.size / float(self.target_sr)
            self.last_level_db = block_rms_db(resampled)
            speaking, just_stopped = self.vad.update(resampled, self.target_sr)
            self.speech_active = speaking
            if speaking:
                self._last_speech_time = time.perf_counter()
            if just_stopped:
                self.speech_stopped_at = time.perf_counter()
            self._ring.write(resampled)
            added += resampled.size
            if time.perf_counter() >= deadline:
                break
        return added

    @property
    def total_speech_seconds(self) -> float:
        """Total audio accepted from the device (including silence)."""
        return self.audio_seconds

    @property
    def total_samples(self) -> int:
        return self._ring.total_written

    def read_recent(self, samples: int) -> np.ndarray:
        return self._ring.read_last(samples)

    def read_range(self, start: int, end: int) -> np.ndarray:
        return self._ring.read_range(start, end)

    def seconds_since_speech(self) -> float:
        if self._last_speech_time is None:
            return float("inf")
        return time.perf_counter() - self._last_speech_time
