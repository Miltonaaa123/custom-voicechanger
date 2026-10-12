"""Microphone sanity check: show the level the transcriber actually sees.

Speak normally for a few seconds. The goal is to see your voice clearly above
the noise floor -- roughly above -40 dBFS peaks and above -30 dBFS while
talking, with the floor at least 10 dB below that.

    .venv\\Scripts\\python.exe tools\\mic_check.py --seconds 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.audio import TARGET_SR, MicCapture, VadConfig, block_rms_db  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--input-device", default=None)
    ap.add_argument("--vad-db", type=float, default=None)
    ap.add_argument("--gain-db", type=float, default=0.0, help="test a software boost, e.g. 18")
    args = ap.parse_args()

    print("opening microphone ...")
    mic = MicCapture(
        device=args.input_device,
        vad=VadConfig(threshold_db=-100.0),
        gain_db=args.gain_db,
    )
    mic.start()
    print(f"device: {mic.device_name} @ {mic.native_sr} Hz -> {TARGET_SR} Hz"
          + (f"  (gain {args.gain_db:+.0f} dB)" if args.gain_db else ""))
    print("SPEAK NOW. Levels every 0.25 s:\n")

    levels: list[float] = []
    deadline = time.perf_counter() + args.seconds
    next_report = time.perf_counter()
    while time.perf_counter() < deadline:
        mic.pump(0.05)
        block = mic.read_recent(int(0.25 * TARGET_SR))
        if block.size:
            levels.append(block_rms_db(block))
        if time.perf_counter() >= next_report:
            next_report += 0.25
            db = levels[-1] if levels else -120.0
            bar = "#" * max(0, int((db + 80) / 2))
            print(f"  {db:7.1f} dBFS |{bar}")
    mic.stop()

    if not levels:
        print("no audio captured at all")
        return 1

    arr = np.array(levels)
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    print(f"\n  10th pct : {p10:7.1f} dBFS   (should be the room floor)")
    print(f"  median   : {p50:7.1f} dBFS")
    print(f"  90th pct : {p90:7.1f} dBFS   (should be your voice)")
    print(f"  dynamic range: {p90 - p10:.1f} dB")

    if p90 < -55:
        print("\n  WARNING: the signal is very quiet. Raise the mic level in Windows,")
        print("           or pick a different device (--input-device).")
    elif p90 - p10 < 6:
        print("\n  WARNING: voice barely stands out from the floor. Move closer to")
        print("           the mic, or lower the threshold with --vad-db.")
    else:
        suggested = max(p10 + 12, -65.0)
        print(f"\n  A speech threshold around {suggested:.0f} dBFS should work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
