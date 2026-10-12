"""Measure the cost of word-level timestamps (needed to avoid duplicated words
when consecutive ASR windows overlap)."""
from __future__ import annotations

import glob
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_transcribe import cuda_env  # noqa: E402

cuda_env.add_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402

from tools.bench_models import read_wav_mono_16k  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "distil-small.en"


def main() -> None:
    clips = [(p, read_wav_mono_16k(p)) for p in sorted(glob.glob("bench/audio*/*.wav"))]
    model = WhisperModel(MODEL, device="cuda", compute_type="int8_float16")
    # warm up
    list(model.transcribe(clips[0][1][:16000 * 3], language="en", beam_size=1)[0])

    print(f"model={MODEL}")
    print(f"{'clip':<12}{'audio':>7}{'seg-only':>11}{'word-ts':>10}{'words':>7}  {'pct(words ending in last 15% of clip)':>0}")
    for path, audio in clips:
        dur = len(audio) / 16000
        seg_times, word_times, words = [], [], []
        for _ in range(3):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(audio, language="en", beam_size=1, word_timestamps=False)
            list(segs)
            seg_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            segs, _ = model.transcribe(audio, language="en", beam_size=1, word_timestamps=True)
            for s in segs:
                words.extend(s.words or [])
            word_times.append(time.perf_counter() - t0)

        edge = [w for w in words if w.end > dur * 0.85]
        print(
            f"{Path(path).name:<12}{dur:7.2f}{statistics.median(seg_times) * 1000:9.0f}ms"
            f"{statistics.median(word_times) * 1000:8.0f}ms{len(words):7d}  "
            f"{len(edge)}/{len(words)} = {100 * len(edge) / max(1, len(words)):.0f}%"
        )
    if words:
        print("\nsample word timings (last 6):")
        for w in words[-6:]:
            print(f"   {w.start:6.2f} -> {w.end:6.2f}  {w.word!r}")


if __name__ == "__main__":
    main()
