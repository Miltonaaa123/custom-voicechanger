"""Does a retry at fewer diffusion steps actually produce a clean short render?

Run inside the engine venv (it needs styletts2 + torch):

    engine\\styletts2\\.venv\\Scripts\\python.exe tools\\probe_steps.py

Renders short words at 5 steps (the default) and at 3 (the retry setting), and
reports duration and the share of energy above 4 kHz -- what tells a voice from a
buzz.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe import engine_paths as p  # noqa: E402
from live_transcribe import engine_tts_server as s  # noqa: E402

SR = s.SAMPLE_RATE


def noise_share(x: np.ndarray) -> float:
    if x.size < 64:
        return 0.0
    X = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    hi = X[(f >= 4000) & (f < 11000)].sum()
    lo = X[(f >= 100) & (f < 2000)].sum()
    return float(hi / (hi + lo + 1e-12))


WORDS = ["you", "yo", "bro", "what", "that", "was"]
REPEATS = 3

engine = s.StyleEngine(p.checkpoint(), p.config(), p.default_reference())
t0 = time.perf_counter()
engine.load()
print(f"loaded in {time.perf_counter() - t0:.1f}s on {s.torch_report()}\n")

print(f"{'word':>6} {'steps':>6} {'run':>4} {'seconds':>8} {'peak':>6} {':>4k':>7}  verdict")
counts = {5: 0, 3: 0}
totals = {5: 0, 3: 0}
for word in WORDS:
    for steps in (5, 3):
        for run in range(1, REPEATS + 1):
            r = engine.synthesize(word, steps=steps)
            x = np.asarray(r["samples"], dtype=np.float32)
            ns = noise_share(x)
            bad = r["seconds"] > 1.2 or ns > 0.5
            counts[steps] += 1 if bad else 0
            totals[steps] += 1
            print(f"{word:>6} {steps:>6} {run:>4} {r['seconds']:8.2f} {r['peak']:6.3f} "
                  f"{ns:7.3f}  {'RUNAWAY' if bad else 'ok'}")

print()
for steps in (5, 3):
    print(f"{steps} steps: {counts[steps]}/{totals[steps]} renders blew up")
