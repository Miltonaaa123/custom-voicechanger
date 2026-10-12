"""One-clip debug trace: prints every decode's window, the words the model
returned, and exactly which of them were committed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe import engine as eng
from live_transcribe.audio import TARGET_SR
from live_transcribe.engine import EngineConfig, StreamingTranscriber
from live_transcribe.transcript import TranscriptWriter
from tools.bench_models import read_wav_mono_16k
from tools.verify_pipeline import WavSource

path = sys.argv[1] if len(sys.argv) > 1 else "bench/audio_hard/hard1.wav"
audio = read_wav_mono_16k(path)
duration = len(audio) / TARGET_SR

cfg = EngineConfig(model="distil-small.en", show_status=False, live_hypothesis=False)
writer = TranscriptWriter(clean_path="out/debug/trace.txt", clear_on_start=True)
tr = StreamingTranscriber(cfg, writer)
tr.engine.load()
tr.engine.warmup()
tr.mic = WavSource(audio, src_sr=TARGET_SR, real_time=False, vad=cfg.vad)

original_decode = tr._decode_once
calls = {"n": 0}

import live_transcribe.engine as eng_mod  # noqa: E402

_orig_select = eng_mod.select_new_words


def traced_select(words, cursor, window_end, edge_guard_s, allow_edge_words, **kw):
    picked = _orig_select(
        words,
        cursor,
        window_end,
        edge_guard_s,
        allow_edge_words,
        **kw,
    )
    picked_ids = {id(w) for w in picked}
    dropped = []
    for w in words:
        if id(w) in picked_ids:
            continue
        if w.end <= cursor.position + kw.get("overlap_guard_s", 0.0):
            why = "past-cursor"
        elif w.probability < kw.get("min_probability", 0.0):
            why = f"low-prob({w.probability:.2f})"
        elif w.end > window_end:
            why = "edge"
        else:
            why = "reemission"
        dropped.append(f"{w.clean}({why})")
    if dropped:
        print(f"      DROPPED from window_end={window_end:.2f}: {dropped}")
    return picked


eng_mod.select_new_words = traced_select


def traced_decode(now: float, flush: bool) -> None:
    calls["n"] += 1
    mic = tr.mic
    cursor_before = tr.cursor.position
    words_before = len(tr.cursor.committed)

    # Reproduce the window math for logging.
    total = mic.total_samples
    cursor_samples = int(round(cursor_before * TARGET_SR))
    read_start = max(0, cursor_samples - 1600)
    max_window = int(cfg.max_window_s * TARGET_SR)
    if total - read_start > max_window:
        read_start = total - max_window
    read_offset = read_start / TARGET_SR
    window_end = read_offset + (total - read_start) / TARGET_SR

    original_decode(now, flush)

    added = len(tr.cursor.committed) - words_before
    new_words = [w.clean for w in tr.cursor.committed[words_before:]]
    print(
        f"#{calls['n']:3d} flush={int(flush)} total={total / TARGET_SR:6.2f}s "
        f"win=[{read_offset:6.2f},{window_end:6.2f}] cursor {cursor_before:6.2f}->{tr.cursor.position:6.2f} "
        f"(+{added}) {new_words}"
    )
    if calls["n"] <= 6 or flush:
        # Show raw model output for this exact window.
        mic = tr.mic
        t = mic.total_samples
        cs = int(round(cursor_before * TARGET_SR))
        rs = max(0, cs - 1600)
        if t - rs > int(cfg.max_window_s * TARGET_SR):
            rs = t - int(cfg.max_window_s * TARGET_SR)
        seg = mic.read_range(rs, t)
        off = rs / TARGET_SR
        print(f"      raw window {seg.size / TARGET_SR:.2f}s offset={off:.2f} prompt={tr.cursor.tail_words(8)!r}")
        try:
            res = tr.engine.decode(seg, prompt=tr.cursor.tail_words(cfg.prompt_words) or None)
            for w in res.words[:14]:
                print(f"        {w.start + off:6.2f} -> {w.end + off:6.2f}  {w.text!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"        raw decode failed: {exc}")


tr._decode_once = traced_decode  # type: ignore[method-assign]

deadline = time.perf_counter() + duration * 6 + 30
while True:
    pending = tr.mic.total_samples - int(tr.cursor.position * TARGET_SR)
    if tr.mic.exhausted and pending < int(0.2 * TARGET_SR):
        break
    tr._tick()
    tr.mic.set_consumer_samples(int(tr.cursor.position * TARGET_SR))
    if time.perf_counter() > deadline:
        print("TIMEOUT")
        break
tr._final_flush()

print("\nTRANSCRIPT:")
print(tr.cursor.text)
print(f"\ndecodes={calls['n']} words={tr.stats.words} audio={duration:.2f}s")
