"""Benchmark batched inference: N independent audio chunks in one encoder pass.

Whisper's cost on short chunks is dominated by the fixed 30 s-padded encoder
pass, so encoding several chunks as one batch is much cheaper per chunk. This
script picks the configuration used by the live transcriber.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_transcribe import cuda_env  # noqa: E402

cuda_env.add_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402

from tools.bench_models import read_wav_mono_16k  # noqa: E402

SR = 16000


def timeit(fn, repeats=6):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    configs = [
        ("tiny.en", "int8_float16"),
        ("tiny.en", "float16"),
        ("base.en", "int8_float16"),
        ("distil-small.en", "int8_float16"),
        ("distil-small.en", "float16"),
    ]
    # Simulate live chunks: 1.6 s of audio with 0.4 s lead-in.
    chunk = read_wav_mono_16k("bench/audio_hard/hard1.wav")[: int(2.0 * SR)]
    n_words_ish = 8

    print(f"{'model':<17}{'compute':<14}{'single':>10}{'batch2':>10}{'batch3':>10}{'batch4':>10}   per-chunk @batch4")
    print("-" * 92)
    for name, ct in configs:
        try:
            model = WhisperModel(name, device="cuda", compute_type=ct)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<17}{ct:<14} LOAD FAILED: {exc}")
            continue
        # warm-up
        list(model.transcribe(chunk, language="en", beam_size=1, word_timestamps=True)[0])

        results = {}
        for b in (1, 2, 3, 4):
            audio = np.stack([chunk] * b)
            results[b] = timeit(
                lambda a=audio: model.transcribe(
                    a, language="en", beam_size=1, word_timestamps=True, condition_on_previous_text=False
                )
            )
        per_chunk = results[4] / 4
        words_per_15 = n_words_ish * (1.6 / 1.6)
        eta_15w = per_chunk * (15 / max(words_per_15, 1)) * 1000
        print(
            f"{name:<17}{ct:<14}{results[1] * 1000:8.0f}ms{results[2] * 1000:8.0f}ms"
            f"{results[3] * 1000:8.0f}ms{results[4] * 1000:8.0f}ms   "
            f"{per_chunk * 1000:5.0f}ms/chunk  (~{eta_15w:.0f}ms per 15 words) [{'PASS' if eta_15w < 500 else 'FAIL'}]"
        )
        del model


if __name__ == "__main__":
    main()
