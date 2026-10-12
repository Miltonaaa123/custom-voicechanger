"""Talk to a running speech engine exactly as the voice changer does.

    .venv\\Scripts\\python.exe tools\\engine_check.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe.tts import TtsClient, TtsError  # noqa: E402

BASE = "http://127.0.0.1:9881"


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as response:
            health = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        print(f"  the engine is not answering on {BASE}: {exc}")
        print("  start it with:  run-tts.bat")
        return 1

    print("health:")
    for key in ("engine", "backend", "loaded", "device", "reference_path", "vram_mb",
                "diffusion_steps", "load_seconds"):
        if key in health:
            print(f"    {key:16} {health[key]}")
    check("the engine reports itself loaded", bool(health.get("loaded")), health.get("load_error"))

    client = TtsClient()
    for text in ("Hello. This is the Rixers Custom Voice Changer speech engine.",
                 "And here is a second, shorter line."):
        started = time.time()
        try:
            speech = client.synthesize(text)
        except TtsError as exc:
            check(f"synthesize {text[:24]!r}", False, str(exc)[:200])
            continue
        check(f"synthesize {text[:24]!r}", speech.seconds > 0.2,
              f"{speech.seconds:.2f}s in {time.time() - started:.2f}s, "
              f"peak={speech.peak}, rms={speech.rms}")
        out = HERE / "out" / "engine_check.wav"
        out.write_bytes(speech.as_wav_bytes())
        print(f"    wrote {out} ({speech.seconds:.2f}s, {speech.sample_rate} Hz)")

    # The voice switch the emotion mapping uses.
    voices = sorted((HERE / "voices").glob("*.wav"))
    if len(voices) >= 2:
        answer = client.set_voice(voices[1])
        check(f"reload-reference -> {voices[1].name}", bool(answer.get("ok")), answer)
        speech = client.synthesize("Switched voice.")
        check("it speaks in the new voice", speech.seconds > 0.2,
              answer.get("reference_path"))
        client.set_voice(voices[0])

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
