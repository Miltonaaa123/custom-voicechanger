"""Capture 1.5 s from every input device and report which ones actually work.

Finds a microphone that produces real signal, instead of guessing from names.

    .venv\\Scripts\\python.exe tools\\scan_inputs.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.audio import block_rms_db  # noqa: E402

SECONDS = 1.5


def probe(index: int) -> tuple[float, float, float] | None:
    """Return (floor_db, p95_db, peak_db) or None if the device won't open."""
    sr = int(sd.query_devices(index)["default_samplerate"])
    levels: list[float] = []

    def cb(indata, _f, _t, status) -> None:  # noqa: ANN001
        pcm = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size:
            levels.append(block_rms_db(pcm))

    try:
        stream = sd.RawInputStream(
            samplerate=sr, blocksize=max(256, sr // 50), device=index, channels=1, dtype="int16", callback=cb
        )
        with stream:
            time.sleep(SECONDS)
    except Exception:  # noqa: BLE001
        return None
    if not levels:
        return None
    arr = np.array(levels)
    return float(np.percentile(arr, 10)), float(np.percentile(arr, 95)), float(arr.max())


def main() -> int:
    devices = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0 and sd.query_hostapis(d["hostapi"])["name"] == "Windows WASAPI"
    ]
    try:
        default_in = sd.default.device[0]
    except Exception:  # noqa: BLE001
        default_in = None

    print(f"probing {len(devices)} WASAPI input devices, {SECONDS}s each ...\n")
    print(f"{'idx':>4}  {'device':<44}{'floor':>9}{'p95':>9}{'peak':>9}   verdict")
    print("-" * 96)
    good: list[int] = []
    for index, d in devices:
        res = probe(index)
        mark = " (default)" if index == default_in else ""
        if res is None:
            print(f"{index:>4}  {(d['name'][:42] + mark):<44}{'--':>9}{'--':>9}{'--':>9}   cannot open")
            continue
        floor, p95, peak = res
        if peak < -80:
            verdict = "SILENT (muted or no signal)"
        elif p95 - floor < 4:
            verdict = "flat / no speech captured"
        else:
            verdict = "signal present"
            good.append(index)
        print(
            f"{index:>4}  {(d['name'][:42] + mark):<44}{floor:9.1f}{p95:9.1f}{peak:9.1f}   {verdict}"
        )

    print("\nDevices with signal:", ", ".join(str(g) for g in good) if good else "none")
    if good:
        print(f"Try:  .venv\\Scripts\\python.exe tools\\mic_check.py --input-device {good[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
