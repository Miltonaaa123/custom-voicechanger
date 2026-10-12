"""Offline end-to-end verification: play a WAV file through the real pipeline.

This drives the actual capture ring, VAD, decode loop and transcript writer --
only the microphone is replaced by a file. It measures word accuracy and the
true spoken-to-written latency, so the pipeline can be validated and tuned
without talking into a microphone.

    .venv\\Scripts\\python.exe tools\\verify_pipeline.py --input bench/audio_hard/hard1.wav ^
        --reference "so what I want to happen is ..." --model distil-small.en
    .venv\\Scripts\\python.exe tools\\verify_pipeline.py --all --model tiny.en
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe.audio import TARGET_SR, AudioRing, EnergyVad, StreamingResampler  # noqa: E402
from live_transcribe.engine import EngineConfig, StreamingTranscriber  # noqa: E402
from live_transcribe.transcript import TranscriptWriter  # noqa: E402

try:
    from tools.bench_models import read_wav_mono_16k
except ImportError:  # pragma: no cover
    from bench_models import read_wav_mono_16k  # type: ignore


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int, int]:
    """Levenshtein WER with substitutions/insertions/deletions."""
    ref, hyp = words_of(reference), words_of(hypothesis)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    errors = d[len(ref)][len(hyp)]
    wer = errors / max(1, len(ref))
    # Rough breakdown
    subs = sum(1 for a, b in zip(ref, hyp) if a != b)
    return wer, errors, len(ref), subs


class WavSource:
    """Drop-in replacement for MicCapture, fed from a numpy array.

    Mirrors the public surface the engine and tests rely on, so the rest of the
    pipeline is exercised unmodified.
    """

    def __init__(self, audio, src_sr: int = TARGET_SR, real_time: bool = True, vad=None) -> None:
        self.device_name = "WAV file"
        self.device_index = -1
        self.native_sr = src_sr
        self._step = max(1, src_sr // 20)  # 50 ms slices
        self._blocksize = self._step
        self._resampler = StreamingResampler(src_sr, TARGET_SR)
        self._ring = AudioRing(120 * TARGET_SR)
        self.vad = EnergyVad(vad)
        self.overflows = 0
        self._audio = audio
        self._src_pos = 0
        self._started_at = time.perf_counter()
        self._next_emit = self._started_at
        self._real_time = real_time
        self.lead_s = 0.5
        """How far the source may run ahead of the wall clock in fast mode."""
        self._consumer_samples = 0
        self.speech_active = False
        self.speech_stopped_at = None
        self.exhausted = False

    @property
    def total_samples(self) -> int:
        return self._ring.total_written

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _feed(self, block) -> None:
        resampled = self._resampler.process(block)
        if resampled.size == 0:
            return
        speaking, just_stopped = self.vad.update(resampled, TARGET_SR)
        self.speech_active = speaking
        if just_stopped:
            self.speech_stopped_at = time.perf_counter()
        self._ring.write(resampled)

    def pump(self, max_seconds: float = 0.25) -> int:
        """Feed audio.

        Real-time mode paces to the wall clock, so latency numbers are exact.
        Otherwise the source runs as fast as the consumer accepts but never more
        than `lead_s` ahead of wall-clock time -- which is the situation the engine
        faces with a live device (audio exists only as fast as it is spoken) while
        still letting the pipeline run its decode loop flat out.
        """
        step = self._step  # 50 ms of source audio per block
        added = 0
        deadline = time.perf_counter() + max(0.02, max_seconds)

        if self._real_time:
            while self._src_pos < len(self._audio):
                now = time.perf_counter()
                if now < self._next_emit:
                    time.sleep(min(0.004, self._next_emit - now))
                    if time.perf_counter() > deadline:
                        break
                    continue
                self._next_emit = max(self._next_emit, now) + 0.010
                block = self._audio[self._src_pos : self._src_pos + step]
                self._src_pos += len(block)
                self._feed(block)
                added += len(block)
                if time.perf_counter() > deadline:
                    break
        else:
            # Never be more than `lead_s` ahead of the wall clock.
            elapsed = time.perf_counter() - self._started_at
            max_pos = int((elapsed + self.lead_s) * self.native_sr)
            while self._src_pos < len(self._audio) and self._src_pos < max_pos:
                block = self._audio[self._src_pos : self._src_pos + step]
                self._src_pos += len(block)
                self._feed(block)
                added += len(block)
            if added == 0 and self._src_pos < len(self._audio):
                time.sleep(0.005)

        if self._src_pos >= len(self._audio):
            self.exhausted = True
        return added

    def set_consumer_samples(self, samples: int) -> None:
        self._consumer_samples = int(samples)

    def read_recent(self, samples: int):
        return self._ring.read_last(samples)

    def read_range(self, start: int, end: int):
        return self._ring.read_range(start, end)

    def seconds_since_speech(self) -> float:
        return 0.0 if self.speech_active else 999.0


CLIPS_REF: dict[str, tuple[str, str]] = {
    "clip15a": (
        "bench/audio/clip15a.wav",
        "the quick brown fox jumps over the lazy dog while the crowd watches in total silence and waits",
    ),
    "clip15b": (
        "bench/audio/clip15b.wav",
        "please write these words into the text file so the speech system can read them back out loud",
    ),
    "clip15c": (
        "bench/audio/clip15c.wav",
        "machine learning models running on a graphics card can transcribe speech much faster than real time",
    ),
    "hard1": (
        "bench/audio_hard/hard1.wav",
        "so what I want to happen is whenever you reboot the program the text file gets cleared and it updates to whatever you are saying right now",
    ),
    "hard2": (
        "bench/audio_hard/hard2.wav",
        "I need the transcriber to be lightweight because I am going to feed the output straight into a speech synthesis model afterwards",
    ),
    "hard3": (
        "bench/audio_hard/hard3.wav",
        "honestly it should probably run on the graphics card since the central processing unit would struggle to keep up with continuous dictation",
    ),
}


def run_one(args, path: Path, reference: str) -> tuple[bool, str]:
    audio = read_wav_mono_16k(str(path))
    duration = len(audio) / TARGET_SR

    cfg = EngineConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        chunk_s=args.chunk,
        edge_guard_s=args.edge_guard,
        live_hypothesis=False,
        show_status=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / f"{path.stem}.{args.model.replace('/', '_')}.txt"
    writer = TranscriptWriter(clean_path=clean, clear_on_start=True)
    engine = StreamingTranscriber(cfg, writer, logger=lambda m: print(m, flush=True))
    engine.engine.load()
    engine.engine.warmup()
    engine.mic = WavSource(audio, src_sr=TARGET_SR, real_time=args.real_time, vad=cfg.vad)

    deadline = time.perf_counter() + duration * 4 + 30
    while True:
        pending = engine.mic.total_samples - int(engine.cursor.position * TARGET_SR)
        if engine.mic.exhausted and pending < int(0.4 * TARGET_SR):
            break
        engine._tick()
        engine.mic.set_consumer_samples(int(engine.cursor.position * TARGET_SR))
        if time.perf_counter() > deadline:
            print("  [warn] timed out")
            break
    engine._final_flush()

    text = engine.cursor.text
    wer, errors, ref_n, _subs = word_error_rate(reference, text)
    tput = engine.stats.throughput_s_per_15_words()

    ok = True
    detail = []
    if reference:
        detail.append(f"WER {wer * 100:5.1f}% ({errors}/{ref_n} words)")
        ok = ok and wer <= args.max_wer
    if engine.stats.decode_ms:
        worst = engine.stats.slowest_decode_ms()
        p95 = engine.stats.percentile_decode_ms(95)
        longest = engine.stats.longest_window_ms()
        detail.append(f"decode p95 {p95:4.0f}ms worst {worst:5.0f}ms (longest window {longest:.0f}ms)")
        # The bar that matters: no single decode may take longer than 2 s.
        ok = ok and worst <= args.max_decode_ms
    if engine.latency.samples:
        p50, p99 = engine.latency.median(), engine.latency.percentile(99)
        detail.append(f"delay p50 {p50:5.0f}ms p99 {p99:5.0f}ms")
        if args.real_time:
            ok = ok and p99 <= args.max_latency_ms

    print(f"\n{'=' * 78}")
    print(f"{path.name}  ({duration:.1f}s audio, model={args.model}, {'real-time' if args.real_time else 'fast'})")
    print(f"  {' | '.join(detail) if detail else 'no measurements'}")
    print(f"  reference : {reference[:150]}")
    print(f"  transcript: {text[:150]}")
    if not ok:
        print(f"  >>> FAILED (max WER {args.max_wer * 100:.0f}%, max decode {args.max_decode_ms:.0f}ms)")
    print(f"  file: {clean}")
    return ok, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="a .wav file")
    ap.add_argument("--all", action="store_true", help="run every bench clip with known reference text")
    ap.add_argument("--reference", default="", help="expected text for --input")
    ap.add_argument("--model", default="distil-small.en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--chunk", type=float, default=1.4)
    ap.add_argument("--edge-guard", type=float, default=0.32)
    ap.add_argument("--out-dir", default="out/verify")
    ap.add_argument("--real-time", action="store_true", help="pace playback to wall clock (latency is meaningful)")
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--max-decode-ms", type=float, default=2000.0, help="no single decode may exceed this")
    ap.add_argument("--max-latency-ms", type=float, default=2500.0)
    args = ap.parse_args()

    references = {
        "clip15a": "the quick brown fox jumps over the lazy dog while the crowd watches in total silence and waits",
        "clip15b": "please write these words into the text file so the speech system can read them back out loud",
        "clip15c": "machine learning models running on a graphics card can transcribe speech much faster than real time",
        "hard1": "so what I want to happen is whenever you reboot the program the text file gets cleared and it updates to whatever you are saying right now",
        "hard2": "I need the transcriber to be lightweight because I am going to feed the output straight into a speech synthesis model afterwards",
        "hard3": "honestly it should probably run on the graphics card since the central processing unit would struggle to keep up with continuous dictation",
    }

    targets: list[tuple[Path, str]] = []
    if args.all:
        for folder in ("bench/audio", "bench/audio_hard"):
            for p in sorted(Path(folder).glob("*.wav")):
                targets.append((p, references.get(p.stem, "")))
    elif args.input:
        targets.append((Path(args.input), args.reference))
    else:
        ap.error("pass --input or --all")

    results = [run_one(args, p, ref) for p, ref in targets]
    passed = sum(1 for ok, _ in results if ok)
    print(f"\n{'=' * 78}")
    print(f"SUMMARY: {passed}/{len(results)} clips within thresholds (model={args.model})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
