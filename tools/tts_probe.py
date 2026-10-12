"""Ask Riko's StyleTTS2 server to synthesize a line and report what came back.

    .venv\\Scripts\\python.exe tools\\tts_probe.py --text "hello there" --out out/probe.wav
    .venv\\Scripts\\python.exe tools\\tts_probe.py --health
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def post(url: str, body: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--text", default="Hello, this is the voice changer speaking.")
    ap.add_argument("--reference", default=None, help="wav to clone for this line")
    ap.add_argument("--out", default="out/probe.wav")
    ap.add_argument("--health", action="store_true")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    if args.health:
        info = get(f"{base}/health")
        interesting = {
            k: info.get(k)
            for k in (
                "backend", "device", "reference_loaded", "reference_path",
                "reference_problem", "cloned", "diffusion_steps", "frame_ms",
                "style_max_norm", "sample_rate",
            )
        }
        print(json.dumps(interesting, indent=2))
        return 0

    body: dict = {"text": args.text}
    if args.reference:
        body["reference_wav"] = args.reference

    started = time.perf_counter()
    try:
        answer = post(f"{base}/synthesize", body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"HTTP {exc.code}: {detail}")
        return 1
    elapsed = time.perf_counter() - started

    audio_b64 = answer.get("audio_base64") or answer.get("audio") or ""
    if not audio_b64:
        print("no audio in the response:")
        print(json.dumps({k: v for k, v in answer.items() if k != "audio_base64"}, indent=2)[:1500])
        return 2

    raw = base64.b64decode(audio_b64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)

    with wave.open(str(out), "rb") as w:
        seconds = w.getnframes() / float(w.getframerate())
        rate = w.getframerate()
        channels = w.getnchannels()

    print(f"synthesized in {elapsed:.2f}s -> {out}")
    print(f"  {seconds:.2f}s of audio, {rate} Hz, {channels} channel(s), {len(raw)} bytes")
    for key in ("peak", "rms", "clipped", "style_norm", "diffusion_steps", "reference_wav", "cloned"):
        if key in answer:
            print(f"  {key}: {answer[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
