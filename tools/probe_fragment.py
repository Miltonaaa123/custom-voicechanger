"""Probe the live engine: does a short fragment render as noise instead of speech?

Measures, per render: duration, peak, rms, and the share of energy above 4 kHz
(with so little low-frequency structure, that reads as a hiss/buzz rather than a
voice).
"""
from __future__ import annotations

import base64
import io
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

SR = 24000
BASE = "http://127.0.0.1:9881"


def synth(text: str) -> dict:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE}/synthesize", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        answer = json.loads(r.read())
    raw = base64.b64decode(answer["audio_base64"])
    with wave.open(io.BytesIO(raw), "rb") as h:
        sr = h.getframerate()
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return {"x": x, "sr": sr, "answer": answer, "wav": raw}


def describe(text: str, tag: str) -> dict:
    res = synth(text)
    x, sr = res["x"], res["sr"]
    a = res["answer"]
    n = len(x)
    # energy split
    X = np.abs(np.fft.rfft(x * np.hanning(n))) if n else np.zeros(1)
    f = np.fft.rfftfreq(n, 1 / sr) if n else np.zeros(1)
    low = 20 * np.log10(np.sqrt((X[(f >= 100) & (f < 2000)] ** 2).mean()) + 1e-9)
    high = 20 * np.log10(np.sqrt((X[(f >= 4000) & (f < 11000)] ** 2).mean()) + 1e-9)
    # how much of the clip is actually above the silence threshold
    voiced = float(np.mean(np.abs(x) > 0.004)) if n else 0.0
    # spectral flatness: near 1.0 means noise, low means harmonic/voiced
    p = X[1:] ** 2 + 1e-12
    flat = float(np.exp(np.mean(np.log(p))) / np.mean(p))
    print(
        f"{tag:>26} {len(text):>3} ch | {a['seconds']:5.2f}s peak {a['peak']:.3f} "
        f"rms {a['rms']:.4f} | voiced {voiced*100:5.1f}% | low {low:6.1f} high {high:6.1f} "
        f"| slope {low-high:6.1f} dB | flatness {flat:.4f} | {a['elapsed_ms']:>5} ms"
    )
    return {"text": text, "x": x, "sr": sr, "wav_bytes": res["wav"], **a}


CASES = [
    ("single short word", "yo"),
    ("single short word", "bro"),
    ("single short word", "what"),
    ("single short word", "that"),
    ("single short word", "was"),
    ("two words", "what was"),
    ("three words", "what was that"),
    ("full utterance", "borrow what did you say yo what did you say bro what was that"),
    ("control sentence", "the quick brown fox jumps over the lazy dog"),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    out_dir = Path(r"C:\Users\oxsta\Desktop\custom voicechanger\out\probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, text in CASES:
        if only and text not in only:
            continue
        try:
            r = describe(text, tag)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag:>26} FAILED: {exc}")
            continue
        name = "".join(c if c.isalnum() else "_" for c in text)[:40]
        (out_dir / f"{name}.wav").write_bytes(r["wav_bytes"])
    print(f"\nwavs written to {out_dir}")
