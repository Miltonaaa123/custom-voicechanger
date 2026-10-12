"""Prove the live path: transcript text -> cable, verified by recording the cable.

``verify_voice_mic.py`` runs ``speak.py`` directly. This instead starts the real
``voice_changer.py``, writes a line into the transcript the way the transcriber
does, and records the cable's microphone end to confirm the audio came out of it.

The distinction matters: it exercises the transcript watcher and the chunker,
which is where transcription could succeed while the cable stayed silent.

    .venv\\Scripts\\python.exe tools\\verify_transcript_to_mic.py
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe.virtual_mic import CABLE_MIC_NAME, find_device  # noqa: E402

SR = 48000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0, help="how long to watch")
    ap.add_argument("--out", default="out/verify_transcript_to_mic.wav")
    ap.add_argument(
        "--text",
        default="hi there hello this is not working why is this so horrible",
        help="line to append to the transcript (no punctuation, like real recognition)",
    )
    args = ap.parse_args()

    mic = find_device(CABLE_MIC_NAME, output=False)
    if mic is None:
        print(f"FAIL: no capture device matching {CABLE_MIC_NAME!r} - is VB-CABLE installed?")
        return 2
    print(f"recording from : {mic.name}")

    with tempfile.TemporaryDirectory() as tmp:
        transcript = Path(tmp) / "transcript.txt"
        # Start with content already there: it must NOT be spoken (old session).
        transcript.write_text("this old line must never be spoken ", encoding="utf-8")

        proc = subprocess.Popen(
            [
                str(HERE / ".venv" / "Scripts" / "python.exe"),
                "-u",
                str(HERE / "voice_changer.py"),
                "--transcript",
                str(transcript),
                "--duration",
                str(args.seconds),
            ],
            cwd=str(HERE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        captured: list[np.ndarray] = []
        output_lines: list[str] = []
        output_queue: "queue.Queue[str]" = queue.Queue()

        def reader() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                output_queue.put(raw.rstrip())

        threading.Thread(target=reader, daemon=True).start()

        def callback(indata, _frames, _time, status):  # noqa: ANN001
            captured.append(indata.copy())
            while True:
                try:
                    output_lines.append(output_queue.get_nowait())
                except queue.Empty:
                    break

        stream = sd.InputStream(device=mic.index, channels=1, samplerate=SR, callback=callback, blocksize=2048)
        stream.start()

        # Give the changer time to connect, then append the line.
        print("waiting for the voice changer to be ready ...")
        time.sleep(20)
        print(f"appending: {args.text!r}")
        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(args.text)

        remaining = max(5.0, args.seconds - 20)
        time.sleep(remaining)
        stream.stop()
        stream.close()

        try:
            proc.terminate()
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        while True:
            try:
                output_lines.append(output_queue.get_nowait())
            except queue.Empty:
                break

    print("\n--- voice changer output ---")
    for line in output_lines:
        print(f"  | {line}")

    if not captured:
        print("FAIL: nothing was captured from the cable")
        return 3

    data = np.concatenate(captured).ravel()
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data))))
    loud = data[np.abs(data) > 0.005]
    louder_pct = 100.0 * loud.size / data.size if data.size else 0.0

    print(f"\ncaptured {data.size / SR:.1f}s from the cable")
    print(f"  peak {peak:.4f}   rms {rms:.5f}")
    print(f"  non-silent samples: {louder_pct:.1f}%")

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
        print("\nFAIL: the transcript was written but the cable carried silence.")
        print("      That is the 'transcribes fine, no audio out' failure.")
        return 4
    print("\nPASS: text appended to the transcript came out of the virtual microphone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
