"""End-to-end proof that the voice changer works as a microphone.

Runs the real path -- ``speak.py`` synthesizes with StyleTTS 2 and plays into
VB-CABLE, while this records from the cable's microphone end -- then reports what
was actually captured. If audio comes back, any application that selects
"CABLE Output" as its microphone will hear the synthesized voice.

    .venv\\Scripts\\python.exe tools\\verify_voice_mic.py --text "testing one two three"
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe.virtual_mic import (  # noqa: E402
    CABLE_MIC_NAME,
    CABLE_OUTPUT_NAME,
    find_device,
)

SR = 48000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Testing one two three. The voice changer is routing audio.")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--mic-name", default=CABLE_MIC_NAME)
    ap.add_argument("--out", default="out/verify_voice_mic.wav")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    mic = find_device(args.mic_name, output=False)
    if mic is None:
        print(f"FAIL: no capture device matching {args.mic_name!r} - is VB-CABLE installed?")
        return 2
    print(f"recording from : {mic.name}")
    print(f"speaking into  : {CABLE_OUTPUT_NAME}")

    captured: list[np.ndarray] = []
    stop = threading.Event()

    def callback(indata, _frames, _time, status):  # noqa: ANN001
        if status:
            pass
        captured.append(indata.copy())

    # One channel is enough: the cable is fed mono.
    stream = sd.InputStream(device=mic.index, channels=1, samplerate=SR, callback=callback, blocksize=2048)
    stream.start()
    time.sleep(0.4)

    command = [str(HERE / ".venv" / "Scripts" / "python.exe"), str(HERE / "speak.py"), "--quiet", args.text]
    if args.voice:
        command += ["--voice", args.voice]
    print("running speak.py ...")
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=str(HERE), capture_output=True, text=True, timeout=args.timeout)
    if proc.returncode != 0:
        print(f"speak.py exited {proc.returncode}")
        print(proc.stdout[-1200:])
        print(proc.stderr[-1200:])
        stream.stop(); stream.close()
        return 3
    elapsed = time.perf_counter() - started

    time.sleep(0.8)
    stream.stop()
    stream.close()

    if not captured:
        print("FAIL: nothing was captured at all")
        return 4

    data = np.concatenate(captured).ravel()
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    loud = data[np.abs(data) > 0.005]
    print(f"\nspeak.py finished in {elapsed:.1f}s")
    print(f"captured {data.size / SR:.2f}s from the cable")
    print(f"  peak {peak:.4f}   rms {rms:.5f}")
    if loud.size:
        print(f"  non-silent samples: {loud.size} ({100.0 * loud.size / data.size:.1f}%)")
        print(f"  median level while speaking: {20 * np.log10(max(statistics.median(np.abs(loud)), 1e-9)):.1f} dBFS")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    print(f"  saved to {out}")

    if peak < 0.01:
        print("\nFAIL: the cable carried silence. Check that nothing is muting CABLE Output")
        print("      and that speak.py did not fall back to another playback device.")
        return 5
    print("\nPASS: the synthesized voice reaches the virtual microphone.")
    print(f"      In any app, select '{CABLE_MIC_NAME}' as the microphone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
