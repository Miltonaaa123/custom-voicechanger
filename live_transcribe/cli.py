"""Command line interface for the live transcriber."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import cuda_env
from .audio import VadConfig, calibrate_noise_floor
from .engine import EngineConfig, StreamingTranscriber
from .transcript import TranscriptWriter

HERE = Path(__file__).resolve().parent.parent
DEFAULT_OUT = HERE / "out"

BANNER = r"""
  live-transcribe  --  continuous microphone -> text file (faster-whisper on GPU)
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live-transcribe",
        description="Continuously transcribe the microphone into a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="distil-small.en", help="faster-whisper model name or local path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="inference device")
    p.add_argument("--compute-type", default="int8_float16", help="ctranslate2 compute type")
    p.add_argument("--language", default="en", help="spoken language code")
    p.add_argument("--beam-size", type=int, default=1, help="1 is fastest")

    p.add_argument("--input-device", default=None, help="microphone index or name substring (see --list-devices)")
    p.add_argument("--gain-db", type=float, default=0.0,
                   help="software boost for a quiet microphone, e.g. 18 (soft-limited, never clipped)")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    p.add_argument("--list-models", action="store_true", help="show suggested models and exit")
    p.add_argument("--doctor", action="store_true", help="print GPU/library diagnostics and exit")

    p.add_argument("--out", default=str(DEFAULT_OUT / "transcript.txt"), help="transcript file to maintain")
    p.add_argument("--live-file", default=None, help="file with the in-flight guess appended (default: live.txt)")
    p.add_argument("--events", default=None, help="telemetry jsonl path (default: events.jsonl)")
    p.add_argument("--append", action="store_true", help="do NOT clear the transcript on start")
    p.add_argument("--utterance-wav", default=str(DEFAULT_OUT / "utterance.wav"),
                   help="wav of each finished utterance, for emotion detection ('' to disable)")
    p.add_argument("--no-utterance-wav", dest="utterance_wav", action="store_const", const="",
                   help="do not write utterance audio (disables emotion matching)")

    p.add_argument("--chunk", type=float, default=1.4, help="seconds of speech per decode while talking")
    p.add_argument("--edge-guard", type=float, default=0.32, help="seconds held back at the live edge")
    p.add_argument("--silence", type=float, default=0.35, help="silence before an utterance is flushed")
    p.add_argument("--min-word-prob", type=float, default=0.1,
                   help="drop words the model is less confident than this (0 disables)")
    p.add_argument("--no-dedupe", action="store_true",
                   help="disable re-emission detection (keeps every repeat)")
    p.add_argument("--vad-db", type=float, default=None, help="speech threshold in dBFS (default: auto-calibrated)")
    p.add_argument("--calibrate", type=float, nargs="?", const=2.0, default=None,
                   help="measure the room noise floor for N seconds (default 2) before listening")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip calibration and use the built-in -42 dBFS threshold")
    p.add_argument(
        "--no-vad-gate",
        action="store_true",
        help="decode continuously instead of waiting for voice activity",
    )

    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl+C)")
    p.add_argument("--no-live-guess", action="store_true", help="do not write partial words to the live file")
    p.add_argument("--no-status", action="store_true", help="disable the live terminal status line")
    p.add_argument("--quiet", action="store_true", help="suppress the banner")
    p.add_argument("--clear-flag", default=None, help="watch this path; if it exists the transcript is cleared")
    return p


MODELS = [
    ("tiny.en", "fastest, ~130 ms / 15 words, lowest accuracy"),
    ("base.en", "~200 ms / 15 words, decent accuracy"),
    ("distil-small.en", "default: ~190 ms / 15 words, close to small.en accuracy"),
    ("small.en", "~400 ms / 15 words, best accuracy that still fits the budget"),
    ("distil-medium.en", "may exceed the 0.5 s budget on a 3060 Ti"),
]


def cmd_list_devices() -> int:
    import sounddevice as sd

    apis = sd.query_hostapis()
    try:
        default_in = sd.default.device[0]
    except Exception:  # noqa: BLE001
        default_in = None
    print("INPUT DEVICES")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        api = apis[d["hostapi"]]["name"]
        mark = "  <- system default" if i == default_in else ""
        print(f"  [{i:>2}] {d['name'][:52]:<52} sr={int(d['default_samplerate']):<6} api={api}{mark}")
    print("\nPass one with:  --input-device 3      (index)")
    print("           or:  --input-device HyperX  (name substring)")
    return 0


def cmd_list_models() -> int:
    print("SUGGESTED MODELS (measured on this machine, RTX 3060 Ti, int8_float16)\n")
    for name, note in MODELS:
        print(f"  {name:<20} {note}")
    print("\nUse:  --model tiny.en")
    return 0


def cmd_doctor() -> int:
    print(BANNER)
    print(cuda_env.diagnose())
    print()
    try:
        import sounddevice as sd

        print(f"default input device : {sd.query_devices(sd.default.device[0])['name']}")
    except Exception as exc:  # noqa: BLE001
        print(f"audio backend        : FAILED {exc}")
    return 0


def _read_keyboard():
    """Non-blocking single keypress read on Windows (no admin rights needed)."""
    if os.name != "nt":
        return None
    try:
        import msvcrt
    except ImportError:
        return None
    try:
        if not msvcrt.kbhit():
            return None
        return msvcrt.getwch()
    except Exception:  # noqa: BLE001 - no console attached
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        return cmd_list_devices()
    if args.list_models:
        return cmd_list_models()
    if args.doctor:
        return cmd_doctor()

    cuda_env.add_cuda_dll_dirs()

    if not args.quiet:
        print(BANNER)
        print(f"  output file : {args.out}")
        print("  keys        : [c] clear   [l] pause/resume   [q] quit")
        print()

    device = args.device
    if device == "auto":
        device = "cuda"

    # Decide the speech threshold: explicit flag > measured > built-in default.
    vad_db = args.vad_db
    if args.no_vad_gate:
        vad_db = -80.0
    elif vad_db is None:
        calibrate_s = args.calibrate
        if calibrate_s is None and not args.no_calibrate:
            calibrate_s = 1.5  # short and automatic by default
        if calibrate_s:
            print(f"  calibrating: stay quiet for {calibrate_s:.1f} s ...", flush=True)
            try:
                floor_db, measured = calibrate_noise_floor(args.input_device, float(calibrate_s))
                vad_db = measured
                print(f"  noise floor {floor_db:.1f} dBFS -> speech threshold {vad_db:.1f} dBFS")
            except Exception as exc:  # noqa: BLE001
                vad_db = -42.0
                print(f"  calibration failed ({exc}); using {vad_db:.0f} dBFS")
        else:
            vad_db = -42.0

    cfg = EngineConfig(
        model=args.model,
        device=device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        chunk_s=args.chunk,
        edge_guard_s=args.edge_guard,
        device_index=args.input_device,
        gain_db=args.gain_db,
        min_word_probability=args.min_word_prob,
        min_word_probability_flush=max(args.min_word_prob, 0.3) if args.min_word_prob else 0.0,
        reemission_window_s=0.0 if args.no_dedupe else 0.6,
        live_hypothesis=not args.no_live_guess,
        show_status=not args.no_status,
        utterance_wav=args.utterance_wav,
        vad=VadConfig(
            threshold_db=vad_db,
            release_db=(vad_db - 8.0) if not args.no_vad_gate else -90.0,
            hangover_s=args.silence if not args.no_vad_gate else 0.0,
        ),
    )

    writer = TranscriptWriter(
        clean_path=args.out,
        live_path=args.live_file,
        events_path=args.events,
        clear_on_start=not args.append,
    )

    engine = StreamingTranscriber(cfg, writer, logger=lambda m: print(m, flush=True))

    def clear_output() -> None:
        print()
        engine.clear()
        print("[clear] transcript emptied")

    try:
        engine.prepare()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[fatal] {exc}", file=sys.stderr)
        return 2

    engine.writer.event("listening")
    deadline = time.perf_counter() + args.duration if args.duration > 0 else None

    engine.start_background()
    try:
        while engine._thread is not None and engine._thread.is_alive():
            key = _read_keyboard()
            if key:
                k = key.lower()
                if k == "q" or key == "\x03":  # q or Ctrl+C
                    break
                if k == "c":
                    clear_output()
                elif k == "l":
                    on = engine.toggle_listening()
                    print(f"\n[listen] {'resumed' if on else 'paused'}")
            if args.clear_flag and Path(args.clear_flag).exists():
                try:
                    Path(args.clear_flag).unlink()
                except OSError:
                    pass
                clear_output()
            if deadline is not None and time.perf_counter() >= deadline:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()

    print(engine.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
