"""Find the decode-window size that best meets the 15-words-in-0.5s budget.

Whisper's encoder pass costs nearly the same for a 1.5 s window as for a 6 s one,
so the cost *per word* falls as the window grows. This measures the trade against
the delay it adds.

    .venv\\Scripts\\python.exe tools\\sweep_window.py --model distil-small.en
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
    "hard1": ("bench/audio_hard/hard1.wav", "so what I want to happen is whenever you reboot the program the text file gets cleared and it updates to whatever you are saying right now"),
    "hard2": ("bench/audio_hard/hard2.wav", "I need the transcriber to be lightweight because I am going to feed the output straight into a speech synthesis model afterwards"),
    "hard3": ("bench/audio_hard/hard3.wav", "honestly it should probably run on the graphics card since the central processing unit would struggle to keep up with continuous dictation"),
}


def run(clip_path: str, reference: str, model: str, chunk: float, real_time: bool, out_dir: Path) -> dict:
    audio = read_wav_mono_16k(clip_path)
    cfg = EngineConfig(model=model, chunk_s=chunk, live_hypothesis=False, show_status=False)
    writer = TranscriptWriter(clean_path=out_dir / f"sweep.{Path(clip_path).stem}.txt", clear_on_start=True)
    tr = StreamingTranscriber(cfg, writer)
    tr.engine.load()
    tr.engine.warmup()
    tr.mic = WavSource(audio, src_sr=TARGET_SR, real_time=real_time, vad=cfg.vad)

    deadline = time.perf_counter() + len(audio) / TARGET_SR * 5 + 40
    while True:
        pending = tr.mic.total_samples - int(tr.cursor.position * TARGET_SR)
        if tr.mic.exhausted and pending < int(0.4 * TARGET_SR):
            break
        tr._tick()
        tr.mic.set_consumer_samples(int(tr.cursor.position * TARGET_SR))
        if time.perf_counter() > deadline:
            break
    tr._final_flush()

    wer, _e, _n, _s = word_error_rate(reference, tr.cursor.text)
    pairs = [
        ms / w
        for ms, w in zip(tr.stats.decode_ms, tr.stats.words_per_decode)
        if w > 0 and ms > 0
    ]
    return {
        "wer": wer,
        "budget_ms": (sorted(pairs)[len(pairs) // 2] * 15) if pairs else float("nan"),
        "decodes": tr.stats.decodes,
        "words": tr.stats.words,
        "delay_p50": tr.latency.median(),
        "delay_p99": tr.latency.percentile(99),
        "words_per_decode": statistics.mean(
            [w for w in tr.stats.words_per_decode if w > 0] or [0]
        ),
        "decode_ms": statistics.median(tr.stats.decode_ms) if tr.stats.decode_ms else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distil-small.en")
    ap.add_argument("--chunks", nargs="+", type=float, default=[1.4, 2.2, 3.0, 4.0])
    ap.add_argument("--real-time", action="store_true")
    ap.add_argument("--out-dir", default="out/sweep")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model}  pacing={'real-time' if args.real_time else 'as-fast-as-possible'}")
    print(f"{'chunk':>6}{'15 words':>10}{'decode':>9}{'words/dec':>11}{'decodes':>9}{'WER':>8}{'delay p50':>11}{'delay p99':>11}")
    print("-" * 78)
    for chunk in args.chunks:
        rows = [run(p, ref, args.model, chunk, args.real_time, out_dir) for p, ref in CLIPS.values()]
        budget = statistics.median([r["budget_ms"] for r in rows])
        verdict = "PASS" if budget < 500 else "over"
        print(
            f"{chunk:6.1f}{budget:8.0f}ms{statistics.median([r['decode_ms'] for r in rows]):8.0f}ms"
            f"{statistics.mean([r['words_per_decode'] for r in rows]):11.1f}"
            f"{statistics.mean([r['decodes'] for r in rows]):9.0f}"
            f"{statistics.mean([r['wer'] for r in rows]) * 100:7.1f}%"
            f"{statistics.median([r['delay_p50'] for r in rows]):9.0f}ms"
            f"{statistics.median([r['delay_p99'] for r in rows]):9.0f}ms   [{verdict}]"
        )
    print("\nTarget: 15 words in under 500 ms. 'delay' is spoken -> written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
