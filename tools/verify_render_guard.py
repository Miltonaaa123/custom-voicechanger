"""Verify the render guard end to end, on the real model, in its own process.

Renders the words that blew up most often and reports, for each: how many renders it
took, how long the result lasts, and how noisy it is. Run with the engine venv:

    engine\\styletts2\\.venv\\Scripts\\python.exe tools\\verify_render_guard.py
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

WORDS = ["you", "yo", "what", "that", "was", "bro"]
REPEATS = 2

print("guard thresholds:")
print(f"  seconds per word : {s.MAX_SPOKEN_SECONDS_PER_WORD} (floor {s.MIN_SPOKEN_SECONDS}s)")
print(f"  max noise share  : {s.MAX_NOISE_SHARE}")
print(f"  attempt ladder   : {s.RENDER_ATTEMPTS}\n")

engine = s.StyleEngine(p.checkpoint(), p.config(), p.default_reference())
t0 = time.perf_counter()
engine.load()
print(f"loaded in {time.perf_counter() - t0:.1f}s\n")

print(f"{'text':>6} {'run':>4} {'attempts':>9} {'seconds':>8} {'noise':>7} {'peak':>6} "
      f"{'ms':>6}  problem")
fails = 0
total = 0
for word in WORDS:
    for run in range(1, REPEATS + 1):
        r = engine.synthesize(word)
        total += 1
        bad = bool(r["render_problem"])
        fails += 1 if bad else 0
        print(f"{word:>6} {run:>4} {r['attempts']:>9} {r['seconds']:8.2f} "
              f"{r['noise_share']:7.3f} {r['peak']:6.3f} {r['elapsed_ms']:>6}  "
              f"{r['render_problem'] or 'ok'}")

print(f"\nrenders returned still bad: {fails}/{total}")
print(f"requests that needed more than one render: {engine.runaway_retries}/{total}")
print(f"ladder in use: {[int(x or engine.diffusion_steps) for x in s.RENDER_ATTEMPTS]}")
print(f"\nengine.info() render guard -> {engine.info()['render_guard']}")
print(f"engine.info() rerendered   -> {engine.info()['rerendered']}")
