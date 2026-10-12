"""Examine word probabilities around a decode boundary.

When consecutive decodes overlap, the word straddling the cursor is sometimes
re-emitted. This shows whether such words carry a distinguishable confidence, so
the commit rule can be tuned on evidence rather than guesswork.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.engine import AsrEngine  # noqa: E402
from tools.bench_models import read_wav_mono_16k  # noqa: E402

SR = 16000


def show(eng: AsrEngine, audio, offset: float, tag: str) -> None:
    res = eng.decode(audio, prompt=None, word_timestamps=True)
    print(f"\n--- {tag}  offset={offset:.2f} len={len(audio)/SR:.2f}s  ({res.duration_s*1000:.0f}ms) ---")
    for w in res.words:
        abs_end = w.end + offset
        print(f"   {w.start + offset:6.2f} -> {abs_end:6.2f}  p={w.probability:5.3f}  {w.text!r}")


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "small.en"
    clip = sys.argv[2] if len(sys.argv) > 2 else "bench/audio/clip15b.wav"
    audio = read_wav_mono_16k(clip)
    eng = AsrEngine(model_name=model)
    eng.load()
    eng.warmup()

    # Reproduce the two overlapping windows from the failing run.
    show(eng, audio[0 : int(3.3 * SR)], 0.0, "decode 1 (cursor would stop ~2.9)")
    show(eng, audio[int(2.8 * SR) :], 2.8, "decode 2 (re-reads from cursor-0.1)")
    show(eng, audio[int(4.6 * SR) :], 4.6, "decode 3 (tail, cursor ~4.3)")
    show(eng, audio[int(4.9 * SR) :], 4.9, "decode 4 (tail from an advanced cursor)")


if __name__ == "__main__":
    main()
