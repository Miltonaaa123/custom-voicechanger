"""Talking to Riko's StyleTTS2 server.

The heavy lifting (model, checkpoint, style guard, saturation retry) lives in
``riko-project/scripts/tts_server.py``, which exposes a small loopback HTTP API:

* ``GET  /health``            -> what backend, device and voice are loaded
* ``POST /synthesize``        -> ``{"text": ..., "reference_wav": ...}`` -> base64 wav
* ``POST /reload-reference``  -> ``{"path": ...}`` to swap the cloned voice

This module is only the client: send text, get a WAV back, in a shape the rest of
the voice changer can play.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9881


class TtsError(RuntimeError):
    """Raised with the server's own message so failures are actionable."""


@dataclass
class Speech:
    """One synthesized line."""

    samples: bytes
    """Raw little-endian 16-bit mono PCM."""
    sample_rate: int
    seconds: float
    text: str
    peak: float | None = None
    rms: float | None = None
    style_norm: float | None = None
    reference_wav: str | None = None
    elapsed_s: float = 0.0

    def as_wav_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.samples)
        return buffer.getvalue()


class TtsClient:
    """Blocking client for the local StyleTTS2 service."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 300.0) -> None:
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # -- basics ------------------------------------------------------------ #
    def health(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=15) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise TtsError(
                f"the speech server is not answering on {self.base} ({exc}). "
                "Start it with tts.bat (or: run-tts-server.bat)."
            ) from exc

    def wait_until_ready(self, timeout_s: float = 240.0, poll_s: float = 3.0) -> dict:
        """Block until /health answers, which it only does once the model is up."""
        import time

        deadline = time.time() + timeout_s
        last: Exception | None = None
        while time.time() < deadline:
            try:
                info = self.health()
                return info
            except TtsError as exc:
                last = exc
            time.sleep(poll_s)
        raise TtsError(f"the speech server did not become ready within {timeout_s:.0f}s ({last})")

    # -- voices ------------------------------------------------------------ #
    def set_voice(self, wav: str | Path) -> dict:
        """Make `wav` the voice every later sentence clones."""
        body = json.dumps({"path": str(wav)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}/reload-reference",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise TtsError(f"the server refused the voice file: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TtsError(f"could not reach the speech server ({exc})") from exc

    # -- the shared word-delay switch -------------------------------------- #
    def timing(self, timeout: float = 3.0) -> dict | None:
        """The page's word-delay setting, or None if no page server is answering.

        The switch lives with the page (that is where the delays are measured) and
        is polled by the voice changer, which is a different process and cannot be
        told directly that it changed. None means "no page server running", which
        is the normal case for the local Whisper engine -- and then the command
        line is in charge instead.
        """
        try:
            with urllib.request.urlopen(f"{self.base}/timing", timeout=timeout) as response:
                answer = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return None
        return answer if isinstance(answer, dict) else None

    # -- speech ------------------------------------------------------------ #
    def synthesize(self, text: str, reference_wav: str | Path | None = None) -> Speech:
        import time

        body: dict = {"text": text}
        if reference_wav:
            body["reference_wav"] = str(reference_wav)

        request = urllib.request.Request(
            f"{self.base}/synthesize",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                answer = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise TtsError(f"speech synthesis failed: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise TtsError(f"could not reach the speech server ({exc})") from exc

        elapsed = time.perf_counter() - started
        encoded = answer.get("audio_base64") or answer.get("audio") or ""
        if not encoded:
            raise TtsError(f"the server returned no audio: {str(answer)[:300]}")

        raw = base64.b64decode(encoded)
        with wave.open(io.BytesIO(raw), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())

        if width != 2:
            raise TtsError(f"expected 16-bit audio, got {width * 8}-bit")
        if channels == 2:
            # Downmix rather than refuse; the virtual cable is mono for our purposes.
            import numpy as np

            data = np.frombuffer(frames, dtype="<i2").reshape(-1, 2).mean(axis=1)
            frames = data.astype("<i2").tobytes()
        elif channels != 1:
            raise TtsError(f"expected mono or stereo audio, got {channels} channels")

        seconds = len(frames) / 2.0 / float(sample_rate)
        return Speech(
            samples=frames,
            sample_rate=sample_rate,
            seconds=seconds,
            text=text,
            peak=answer.get("peak"),
            rms=answer.get("rms"),
            style_norm=answer.get("style_norm"),
            reference_wav=answer.get("reference_wav"),
            elapsed_s=elapsed,
        )
