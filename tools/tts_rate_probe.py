"""Measure per-call synthesis cost against audio produced.

Word-by-word TTS only works if synthesizing a short chunk is faster than speaking
it. StyleTTS 2 has a large fixed cost per call (style sampling, duration
prediction, a 3-step diffusion pass, the vocoder), so this measures the real
break-even: audio seconds produced per wall-clock second spent.

    .venv\\Scripts\\python.exe tools\\tts_rate_probe.py
    .venv\\Scripts\\python.exe tools\\tts_rate_probe.py --texts "the" "the quick brown fox"
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.tts import TtsClient, TtsError  # noqa: E402

CASES = [
    ("1 word  (short)", "the"),
    ("1 word  (long)", "understanding"),
    ("2 words", "hello there"),
    ("3 words", "this is fine"),
    ("5 words", "the quick brown fox jumps"),
    ("8 words", "this is a slightly longer phrase to measure"),
    ("15 words", "the quick brown fox jumps over the lazy dog while the crowd watches in total silence"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument("--warm", type=int, default=2, help="warm-up calls discarded")
    args = ap.parse_args()

    cases = [(f"{len(t.split())} words", t) for t in args.texts] if args.texts else CASES

    client = TtsClient(port=args.port)
    try:
        info = client.health()
    except TtsError as exc:
        print(f"{exc}")
        return 2
    print(f"server: {info.get('backend')} on {info.get('device')}\n")

    print(f"{'case':<18}{'audio s':>9}{'synth s':>9}{'x realtime':>12}{'words/audio s':>15}")
    print("-" * 66)
    for label, text in cases:
        for _ in range(args.warm):
            try:
                client.synthesize(text)
            except TtsError as exc:
                print(f"{label:<18} failed: {exc}")
                break
        audios, times = [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            speech = client.synthesize(text)
            times.append(time.perf_counter() - started)
            audios.append(speech.seconds)
        if not audios:
            continue
        audio = statistics.median(audios)
        synth = statistics.median(times)
        words = len(text.split())
        print(
            f"{label:<18}{audio:9.2f}{synth:9.2f}{audio / synth if synth else 0:11.2f}x"
            f"{(words / audio) if audio else 0:15.2f}"
        )

    print(
        "\n'x realtime' is how much audio one second of synthesis buys."
        "\nAbove 1.0 means a pipelined word-by-word speaker can keep up."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
