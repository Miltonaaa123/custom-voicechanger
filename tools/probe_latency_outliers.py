"""Look for periodic slow decodes: time many identical decodes and print each.

If a "stall" appears here at a fixed interval it is an environment/driver effect
(VRAM pressure, WDDM paging) rather than something the streaming logic causes.

    .venv\\Scripts\\python.exe tools\\probe_latency_outliers.py --seconds 2 --n 40
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.engine import AsrEngine  # noqa: E402

SR = 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distil-small.en")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    # Speech-like noise burst so the model has something to chew on.
    audio = (rng.standard_normal(int(args.seconds * SR)) * 0.05).astype(np.float32)
    audio[int(0.2 * SR) : int(0.8 * SR)] *= 8.0

    eng = AsrEngine(model_name=args.model)
    eng.load()
    eng.warmup()

    times = []
    print(f"{args.n} decodes of {args.seconds:.1f}s audio, model={args.model}")
    for i in range(args.n):
        t0 = time.perf_counter()
        eng.decode(audio, prompt=None, word_timestamps=True)
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        flag = "  <-- SLOW" if ms > 1500 else ""
        print(f"  #{i + 1:3d} {ms:8.0f} ms{flag}")

    times_sorted = sorted(times)
    print(
        f"\n  min {min(times):.0f}  p50 {statistics.median(times):.0f}  "
        f"p95 {times_sorted[int(0.95 * (len(times_sorted) - 1))]:.0f}  max {max(times):.0f} ms"
    )
    slow = [i for i, t in enumerate(times) if t > 1500]
    print(f"  slow decodes at iterations: {slow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
