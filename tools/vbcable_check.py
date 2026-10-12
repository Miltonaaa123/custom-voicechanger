"""Find the VB-CABLE loopback endpoints.

The TTS has to be *played* somewhere that Windows then exposes as a microphone:
``CABLE Input`` (a playback device) is the write end, and whatever app records
from ``CABLE Output`` receives it. This prints those two indices and does a short
audible-through-the-cable test so the routing can be confirmed without guessing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 48000


def find_output(name_part: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if name_part.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    return None


def find_input(name_part: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if name_part.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--play-to", default="CABLE Input", help="playback device to write into")
    ap.add_argument("--record-from", default="CABLE Output", help="capture device apps read")
    ap.add_argument("--seconds", type=float, default=0.0, help="play a test tone for this long")
    args = ap.parse_args()

    out = find_output(args.play_to)
    inp = find_input(args.record_from)
    print(f"playback (write end) : [{out}] {sd.query_devices(out)['name'] if out is not None else 'NOT FOUND'}")
    print(f"capture  (mic end)   : [{inp}] {sd.query_devices(inp)['name'] if inp is not None else 'NOT FOUND'}")

    if out is None:
        print("\nVB-CABLE's 'CABLE Input' playback device is missing.")
        print("Install it once from: C:\\Program Files\\VB\\CABLE\\VBCABLE_Setup_x64.exe (run as admin),")
        print("then reboot. Without it there is no virtual microphone for other apps to read.")
        return 1

    if args.seconds <= 0:
        return 0

    # A 440 Hz tone played into the cable, captured back from it, proves the pair
    # is wired together -- if the recording is silent, Windows is not routing.
    print(f"\nplaying {args.seconds:.1f}s of tone into the cable ...")
    t = np.arange(int(args.seconds * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    fade = int(0.02 * SR)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)

    captured: list[np.ndarray] = []

    def cb(indata, _frames, _time, status):  # noqa: ANN001
        captured.append(indata.copy())

    stream_in = None
    if inp is not None:
        try:
            stream_in = sd.InputStream(device=inp, channels=1, samplerate=SR, callback=cb, blocksize=1024)
            stream_in.start()
        except Exception as exc:  # noqa: BLE001
            print(f"  could not open the capture side: {exc}")
            stream_in = None

    sd.play(tone, SR, device=out, blocking=True)
    time.sleep(0.3)

    if stream_in is not None:
        stream_in.stop()
        stream_in.close()
        if captured:
            data = np.concatenate(captured).ravel()
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            print(f"  captured {data.size / SR:.2f}s from the cable, peak {peak:.3f}")
            if peak > 0.01:
                print("  ROUTING WORKS - other apps can record this as a microphone.")
                return 0
            print("  captured silence: the cable endpoints are not connected.")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
