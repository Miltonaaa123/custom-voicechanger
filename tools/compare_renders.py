"""Inspect the reproducing render: find the runaway clips and measure them."""
from __future__ import annotations

import sys, wave
from pathlib import Path
import numpy as np

HERE = Path(r"C:\Users\oxsta\Desktop\custom voicechanger")

def load(p):
    with wave.open(str(p), "rb") as h:
        sr = h.getframerate()
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, sr

def stats(x, sr, tag):
    n = len(x)
    if n == 0:
        print(f"{tag}: empty")
        return
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    def band(a, b):
        return 20 * np.log10(np.sqrt((X[(f >= a) & (f < b)] ** 2).mean()) + 1e-9)
    # 20 ms frames, rolling
    hop = sr // 50
    m = n // hop
    fr = x[:m * hop].reshape(m, hop)
    r = np.sqrt((fr ** 2).mean(axis=1))
    z = (np.diff(np.sign(fr), axis=1) != 0).sum(axis=1)
    print(f"{tag}: {n/sr:5.2f}s peak {np.abs(x).max():.3f} rms {np.sqrt((x**2).mean()):.4f}")
    print(f"    bands 0-300 {band(0,300):6.1f} | 300-1k {band(300,1000):6.1f} | 1k-3k {band(1000,3000):6.1f}"
          f" | 3k-6k {band(3000,6000):6.1f} | 6k-12k {band(6000,12000):6.1f} dB")
    print(f"    frames: rms mean {r.mean():.4f} max {r.max():.4f} | zcr mean {z.mean():.1f} max {z.max()}")
    print(f"    low-level (<0.004) frames: {float((r<0.004).mean())*100:.1f}%")
    # profile every 0.25 s
    step = sr // 4
    row = []
    for i in range(0, n - step, step):
        seg = x[i:i + step]
        rr = float(np.sqrt((seg ** 2).mean()))
        zz = int((np.diff(np.sign(seg)) != 0).sum())
        row.append(f"{i/sr:4.1f}:{rr:.3f}/{zz:3d}")
    print("    " + "  ".join(row))

print("=" * 100)
print("A) rendered by tos/probe_fragment (isolated call)")
for name in ("yo", "you", "bro", "what_was_that", "borrow_what_did_you_say_yo_what_did_you_say_bro"):
    p = HERE / "out" / "probe" / f"{name}.wav"
    if p.exists():
        x, sr = load(p)
        stats(x, sr, f"probe/{name}")
    else:
        print(f"probe/{name}: missing")

print("=" * 100)
print("B) the whole reproduced live run")
p = HERE / "out" / "repro" / "repro_word.wav"
if p.exists():
    x, sr = load(p)
    stats(x, sr, "repro_word (whole)")
else:
    print("no repro wav")

print("=" * 100)
print("C) the audio captured from the user's clip")
p = HERE / "out" / "clip" / "a2.wav"
x, sr = load(p)
seg = x[int(3.60 * sr):int(9.40 * sr)]
stats(seg, sr, "user clip (voice-changer track)")
