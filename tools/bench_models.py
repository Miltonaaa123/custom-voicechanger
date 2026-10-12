"""Benchmark faster-whisper models on this GPU against the user's latency target.

Target: transcribe 15 words in under 0.5 s of wall-clock time.
Reports real-time factor (RTF) = processing_time / audio_duration, plus
projected seconds to handle a 15-word chunk.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from live_transcribe import cuda_env  # noqa: E402

cuda_env.add_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402

SR = 16000


def read_wav_mono_16k(path: str) -> np.ndarray:
    import wave

    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 2:
        data /= 32768.0
    elif width == 4:
        data /= 2147483648.0
    elif width == 1:
        data = (data - 128.0) / 128.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly

        from math import gcd

        g = gcd(SR, sr)
        data = resample_poly(data, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["distil-small.en", "small.en"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--clips", default="bench/audio/*.wav")
    args = ap.parse_args()

    clips = []
    for p in sorted(glob.glob(args.clips)):
        audio = read_wav_mono_16k(p)
        clips.append((p, audio))
    if not clips:
        print("no benchmark clips found; run tools/make_bench_audio.ps1")
        return 1

    print(f"device={args.device} compute_type={args.compute_type} repeats={args.repeats}")
    print("=" * 96)

    for name in args.models:
        try:
            t0 = time.perf_counter()
            model = WhisperModel(name, device=args.device, compute_type=args.compute_type)
            load = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20} LOAD FAILED: {type(exc).__name__}: {exc}")
            continue

        # Warm-up (first call compiles kernels / allocates workspace).
        warm_audio = clips[0][1][: SR * 3]
        t0 = time.perf_counter()
        list(model.transcribe(warm_audio, language="en", beam_size=1)[0])
        warm = time.perf_counter() - t0

        print(f"\n{name}  (load {load:.1f}s, warmup {warm * 1000:.0f}ms)")
        for path, audio in clips:
            dur = len(audio) / SR
            times, words, text = [], 0, ""
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                segs, _info = model.transcribe(
                    audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
                )
                text = " ".join(s.text.strip() for s in segs).strip()
                times.append(time.perf_counter() - t0)
                words = len(text.split())
            med = statistics.median(times)
            rtf = med / dur
            # seconds of wall clock to cover a 15-word chunk
            sec_per_15w = rtf * (dur / words * 15) if words else float("nan")
            verdict = "PASS" if sec_per_15w < 0.5 else "FAIL"
            print(
                f"   {path.split(chr(92))[-1]:<14} audio={dur:5.2f}s words={words:3d} "
                f"median={med * 1000:6.0f}ms  RTF={rtf:5.3f}  -> 15 words = {sec_per_15w * 1000:5.0f}ms  [{verdict}]"
            )
        print(f"   sample transcript: {text[:110]}")
        del model

    print("\nRTF < 1.0 means faster than real time. Target: 15 words in < 500 ms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
