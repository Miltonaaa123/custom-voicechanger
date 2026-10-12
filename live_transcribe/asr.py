"""faster-whisper wrapper: model lifecycle, warm-up and one decode call."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from . import cuda_env
from .merge import Word

cuda_env.add_cuda_dll_dirs()


class ModelLoadError(RuntimeError):
    pass


@dataclass
class DecodeResult:
    words: list[Word]
    duration_s: float
    audio_s: float
    rtf: float


class AsrEngine:
    """Thin wrapper around a faster-whisper model.

    One model instance is shared for the process lifetime. Decoding is
    synchronous and single-threaded; a dedicated thread in the streaming engine
    calls it so the PortAudio callback never blocks.
    """

    def __init__(
        self,
        model_name: str = "distil-small.en",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "en",
        beam_size: int = 1,
        cpu_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.language = language
        self.beam_size = beam_size
        self.verbose = verbose

        self.device = device
        self.compute_type = compute_type
        self._model = None
        self.load_seconds = 0.0
        self.warmup_seconds = 0.0
        self.decode_count = 0
        self.audio_seconds = 0.0
        self.decode_seconds = 0.0
        self._cpu_threads = cpu_threads

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> None:
        from faster_whisper import WhisperModel

        attempts = [(self.requested_device, self.compute_type)]
        if self.requested_device == "cuda":
            # cuDNN/cuBLAS may be missing; fall back rather than refusing to run.
            attempts.append(("cuda", "float16"))
            attempts.append(("cpu", "int8"))

        errors: list[str] = []
        for device, compute_type in attempts:
            try:
                t0 = time.perf_counter()
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self._cpu_threads,
                )
                self.load_seconds = time.perf_counter() - t0
                self.device = device
                self.compute_type = compute_type
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{device}/{compute_type}: {type(exc).__name__}: {exc}")
                self._model = None
                continue

        if self._model is None:
            detail = "\n  ".join(errors)
            raise ModelLoadError(
                f"could not load model '{self.model_name}'.\n  {detail}\n"
                "If this mentions cublas/cudnn, run:\n"
                "  .venv\\Scripts\\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
                "Or run with --device cpu."
            )

        if self.requested_device == "cuda" and self.device != "cuda":
            print("[warn] CUDA could not be used; running on CPU instead.")
        if device != self.requested_device and self.device == "cuda":
            print(f"[warn] fell back to compute_type={self.compute_type}")

    def warmup(self, seconds: float = 3.0, sr: int = 16000) -> None:
        """First decode allocates workspaces; do it before the user speaks."""
        if self._model is None:
            raise RuntimeError("call load() first")
        silence = np.zeros(int(seconds * sr), dtype=np.float32)
        t0 = time.perf_counter()
        try:
            list(self._model.transcribe(silence, language=self.language, beam_size=self.beam_size)[0])
        except Exception:  # noqa: BLE001 - warm-up failures are not fatal
            pass
        self.warmup_seconds = time.perf_counter() - t0

    # -- decoding ---------------------------------------------------------- #
    def decode(
        self,
        audio: np.ndarray,
        prompt: str | None = None,
        word_timestamps: bool = True,
    ) -> DecodeResult:
        """Decode one audio window and return words in window-relative time."""
        if self._model is None:
            raise RuntimeError("call load() first")
        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            temperature=0.0,
            vad_filter=False,
        )
        words: list[Word] = []
        for seg in segments:
            if getattr(seg, "words", None):
                for w in seg.words:
                    words.append(
                        Word(text=w.word, start=float(w.start), end=float(w.end), probability=float(w.probability))
                    )
            else:
                # Defensive: if timestamps are unavailable, approximate.
                toks = [t for t in seg.text.split(" ") if t]
                if toks:
                    span = max(1e-3, float(seg.end) - float(seg.start))
                    step = span / len(toks)
                    for i, tok in enumerate(toks):
                        words.append(
                            Word(
                                text=(" " + tok) if i else tok,
                                start=float(seg.start) + i * step,
                                end=float(seg.start) + (i + 1) * step,
                            )
                        )
        elapsed = time.perf_counter() - t0
        audio_s = len(audio) / 16000.0
        self.decode_count += 1
        self.audio_seconds += audio_s
        self.decode_seconds += elapsed
        return DecodeResult(
            words=words,
            duration_s=elapsed,
            audio_s=audio_s,
            rtf=(elapsed / audio_s) if audio_s > 0 else 0.0,
        )

    def iter_decode(
        self,
        audio: np.ndarray,
        prompt: str | None = None,
    ) -> Iterator[tuple[list[Word], float]]:
        """Yield ``(words_in_segment, segment_end)`` as segments become ready.

        Used so text can be written to the transcript before the whole window has
        finished decoding.
        """
        if self._model is None:
            raise RuntimeError("call load() first")
        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            temperature=0.0,
            vad_filter=False,
        )
        produced: list[Word] = []
        for seg in segments:
            seg_words: list[Word] = []
            for w in getattr(seg, "words", None) or []:
                seg_words.append(
                    Word(text=w.word, start=float(w.start), end=float(w.end), probability=float(w.probability))
                )
            if not seg_words and seg.text.strip():
                toks = [t for t in seg.text.split(" ") if t]
                span = max(1e-3, float(seg.end) - float(seg.start))
                step = span / max(1, len(toks))
                for i, tok in enumerate(toks):
                    seg_words.append(
                        Word(
                            text=(" " + tok) if i else tok,
                            start=float(seg.start) + i * step,
                            end=float(seg.start) + (i + 1) * step,
                        )
                    )
            produced.extend(seg_words)
            yield seg_words, float(seg.end)

        elapsed = time.perf_counter() - t0
        audio_s = len(audio) / 16000.0
        self.decode_count += 1
        self.audio_seconds += audio_s
        self.decode_seconds += elapsed

    def stats(self) -> dict[str, float]:
        avg = self.decode_seconds / self.decode_count if self.decode_count else 0.0
        return {
            "decodes": float(self.decode_count),
            "audio_s": self.audio_seconds,
            "decode_s": self.decode_seconds,
            "avg_decode_ms": avg * 1000.0,
            "overall_rtf": (self.decode_seconds / self.audio_seconds) if self.audio_seconds else 0.0,
        }
