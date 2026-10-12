"""Back-to-back model comparison under the current machine conditions.

Prints one compact line per clip plus a summary, so two models can be compared
without drowning in output.

    .venv\\Scripts\\python.exe tools\\compare_models.py --models small.en distil-small.en
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.audio import TARGET_SR  # noqa: E402
from live_transcribe.engine import EngineConfig, StreamingTranscriber  # noqa: E402
from live_transcribe.transcript import TranscriptWriter  # noqa: E402
from tools.bench_models import read_wav_mono_16k  # noqa: E402
from tools.verify_pipeline import CLIPS_REF, WavSource, word_error_rate  # noqa: E402


def one(clip: str, path: str, reference: str, model: str, chunk: float, real_time: bool, out: Path) -> dict:
    audio = read_wav_mono_16k(path)
    cfg = EngineConfig(model=model, chunk_s=chunk, live_hypothesis=False, show_status=False)
    tr = StreamingTranscriber(cfg, TranscriptWriter(clean_path=out / f"{clip}.{model}.txt", clear_on_start=True))
    tr.engine.load()
    tr.engine.warmup()
    tr.mic = WavSource(audio, src_sr=TARGET_SR, real_time=real_time, vad=cfg.vad)

    deadline = time.perf_counter() + len(audio) / TARGET_SR * 6 + 45
    while True:
        pending = tr.mic.total_samples - int(tr.cursor.position * TARGET_SR)
        if tr.mic.exhausted and pending < int(0.4 * TARGET_SR):
            break
        tr._tick()
        tr.mic.set_consumer_samples(int(tr.cursor.position * TARGET_SR))
        if time.perf_counter() > deadline:
            break
    tr._final_flush()

    wer, errors, words, _ = word_error_rate(reference, tr.cursor.text)
    budget = tr.stats.throughput_s_per_15_words() * 1000
    return {
        "clip": clip,
        "wer": wer,
        "errors": errors,
        "words": words,
        "budget": budget if budget == budget else float("nan"),
        "p50": tr.latency.median(),
        "text": tr.cursor.text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small.en", "distil-small.en"])
    ap.add_argument("--chunk", type=float, default=1.6)
    ap.add_argument("--real-time", action="store_true")
    ap.add_argument("--out-dir", default="out/compare")
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"chunk={args.chunk}  pacing={'real-time' if args.real_time else 'fast'}  clips={len(CLIPS_REF)}")
    summary = []
    for model in args.models:
        rows = []
        for clip, (path, reference) in CLIPS_REF.items():
            r = one(clip, path, reference, model, args.chunk, args.real_time, out)
            rows.append(r)
            flag = "" if r["wer"] <= 0.15 else "  <-- check"
            print(f"  {model:<16}{clip:<9} WER {r['wer'] * 100:5.1f}%  {r['errors']}/{r['words']:<3d}  "
                  f"15w {r['budget']:6.0f}ms  p50 {r['p50']:5.0f}ms{flag}")
            if args.show_text:
                print(f"        {r['text'][:100]}")
        wer = statistics.mean(r["wer"] for r in rows)
        budgets = [r["budget"] for r in rows if r["budget"] == r["budget"]]
        summary.append((model, wer, sum(r["errors"] for r in rows), sum(r["words"] for r in rows),
                        statistics.median(budgets) if budgets else float("nan")))
        print()

    print(f"{'model':<18}{'mean WER':>10}{'errors':>10}{'15 words (median)':>20}")
    print("-" * 60)
    for model, wer, errors, words, budget in summary:
        print(f"{model:<18}{wer * 100:9.1f}%{errors:6d}/{words:<4d}{budget:17.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
