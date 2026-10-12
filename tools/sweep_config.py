"""Settle the defaults: compare models and cadence settings over all test clips.

Runs the full pipeline offline over every benchmark clip and reports word error
rate, the processing budget and the delay for each configuration, so the shipped
defaults are chosen from measurements rather than guesswork.

    .venv\\Scripts\\python.exe tools\\sweep_config.py
    .venv\\Scripts\\python.exe tools\\sweep_config.py --models small.en distil-small.en --chunks 1.6 2.4
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
from tools.verify_pipeline import WavSource, word_error_rate  # noqa: E402

CLIPS = {
    "clip15a": ("bench/audio/clip15a.wav", "the quick brown fox jumps over the lazy dog while the crowd watches in total silence and waits"),
    "clip15b": ("bench/audio/clip15b.wav", "please write these words into the text file so the speech system can read them back out loud"),
    "clip15c": ("bench/audio/clip15c.wav", "machine learning models running on a graphics card can transcribe speech much faster than real time"),
    "hard1": ("bench/audio_hard/hard1.wav", "so what I want to happen is whenever you reboot the program the text file gets cleared and it updates to whatever you are saying right now"),
    "hard2": ("bench/audio_hard/hard2.wav", "I need the transcriber to be lightweight because I am going to feed the output straight into a speech synthesis model afterwards"),
    "hard3": ("bench/audio_hard/hard3.wav", "honestly it should probably run on the graphics card since the central processing unit would struggle to keep up with continuous dictation"),
}


def run(clip: str, reference: str, cfg_kwargs: dict, real_time: bool, out_dir: Path) -> dict:
    audio = read_wav_mono_16k(CLIPS[clip][0])
    cfg = EngineConfig(live_hypothesis=False, show_status=False, **cfg_kwargs)
    writer = TranscriptWriter(clean_path=out_dir / f"{clip}.{cfg.model}.txt", clear_on_start=True)
    tr = StreamingTranscriber(cfg, writer)
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

    wer, errors, nwords, _ = word_error_rate(reference, tr.cursor.text)
    return {
        "clip": clip,
        "wer": wer,
        "errors": errors,
        "words": nwords,
        "text": tr.cursor.text,
        "budget_ms": tr.stats.throughput_s_per_15_words() * 1000,
        "delay_p50": tr.latency.median(),
        "delay_p99": tr.latency.percentile(99),
        "decodes": tr.stats.decodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small.en", "distil-small.en"])
    ap.add_argument("--chunks", nargs="+", type=float, default=[1.6, 2.4])
    ap.add_argument("--real-time", action="store_true")
    ap.add_argument("--out-dir", default="out/sweep")
    ap.add_argument("--runs", type=int, default=1, help="repeat each config to gauge variance")
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"pacing={'real-time' if args.real_time else 'as-fast-as-possible'}  runs={args.runs}")
    print(f"{'model':<17}{'chunk':>6}{'WER':>8}{'errors':>8}{'15 words':>11}{'decode':>9}{'delay p50':>11}{'p99':>9}")
    print("-" * 80)
    best = None
    for model in args.models:
        for chunk in args.chunks:
            allruns = []
            for _ in range(args.runs):
                allruns.extend(
                    run(c, ref, {"model": model, "chunk_s": chunk}, args.real_time, out_dir)
                    for c, (_p, ref) in CLIPS.items()
                )
            wer = statistics.mean(r["wer"] for r in allruns)
            errors = sum(r["errors"] for r in allruns)
            words = sum(r["words"] for r in allruns)
            budgets = [r["budget_ms"] for r in allruns if r["budget_ms"] == r["budget_ms"]]
            p50 = statistics.median(r["delay_p50"] for r in allruns if r["delay_p50"] == r["delay_p50"])
            p99 = statistics.median(r["delay_p99"] for r in allruns if r["delay_p99"] == r["delay_p99"])
            print(
                f"{model:<17}{chunk:6.1f}{wer * 100:7.1f}%{errors:4d}/{words:<4d}"
                f"{(statistics.median(budgets) if budgets else float('nan')):9.0f}ms"
                f"{statistics.median(r['decodes'] for r in allruns):8.0f}"
                f"{p50:9.0f}ms{p99:7.0f}ms"
            )
            score = (wer, statistics.median(budgets) if budgets else 9e9)
            if best is None or score < best[0]:
                best = (score, model, chunk)
            if args.show_text:
                for r in allruns:
                    print(f"      {r['clip']:<9} WER {r['wer'] * 100:5.1f}%  {r['text'][:90]}")

    if best:
        print(f"\nBest: model={best[1]} chunk={best[2]}  (WER {best[0][0] * 100:.1f}%, {best[0][1]:.0f} ms/15 words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
