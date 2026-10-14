"""How often does the engine render a runaway (over-long / noisy) clip?

Renders the same short word repeatedly and reports the duration of each, which is
the one number that tells a good render from a duration-predictor blow-up.
"""
from __future__ import annotations

import base64
import io
import json
import statistics
import sys
import urllib.request
import wave

import numpy as np

BASE = "http://127.0.0.1:9881"


def synth(text: str) -> tuple[np.ndarray, dict]:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE}/synthesize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        a = json.loads(r.read())
    raw = base64.b64decode(a["audio_base64"])
    with wave.open(io.BytesIO(raw), "rb") as h:
        sr = h.getframerate()
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, a


def noise_ratio(x: np.ndarray, sr: int) -> float:
    """Share of energy above 4 kHz: a voice keeps this low, a buzz does not."""
    if x.size < 64:
        return 0.0
    X = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / sr)
    hi = X[(f >= 4000) & (f < 11000)].sum()
    lo = X[(f >= 100) & (f < 2000)].sum()
    return float(hi / (hi + lo + 1e-12))


text = sys.argv[1] if len(sys.argv) > 1 else "you"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
print(f"rendering {text!r} x{n} through the live engine\n")
print(f"{'#':>3} {'seconds':>8} {'peak':>6} {'rms':>7} {'>4k share':>10} {'ms':>6}  verdict")
durs, noisy = [], []
for i in range(1, n + 1):
    x, a = synth(text)
    nr = noise_ratio(x, a["sample_rate"])
    bad = a["seconds"] > 1.6 or nr > 0.35
    durs.append(a["seconds"])
    if bad:
        noisy.append(i)
    print(f"{i:>3} {a['seconds']:8.2f} {a['peak']:6.3f} {a['rms']:7.4f} {nr:10.3f} "
          f"{a['elapsed_ms']:>6}  {'<-- RUNAWAY' if bad else 'ok'}")

print(f"\nduration: min {min(durs):.2f} median {statistics.median(durs):.2f} max {max(durs):.2f}")
print(f"runaway renders: {len(noisy)}/{n}  at {noisy}")
