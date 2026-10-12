#!/usr/bin/env python
"""Speak text out of a virtual microphone.

Text in, StyleTTS 2 speech out, played into VB-CABLE so Windows and every other
application see it as a microphone. This is the "make it a microphone" half of the
voice changer.

    .venv\\Scripts\\python.exe speak.py "hello there"
    .venv\\Scripts\\python.exe speak.py --list-voices
    .venv\\Scripts\\python.exe speak.py --voice narrator "switch voices any time"
    echo some text | .venv\\Scripts\\python.exe speak.py --stdin
    .venv\\Scripts\\python.exe speak.py --file out\\transcript.txt --watch

The speech server must be running first (tts.bat).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_transcribe import voices as voices_mod  # noqa: E402
from live_transcribe.pipeline import PipelineConfig, SpeechPipeline  # noqa: E402
from live_transcribe.tts import TtsClient, TtsError  # noqa: E402
from live_transcribe.virtual_mic import (  # noqa: E402
    CABLE_MIC_NAME,
    CABLE_OUTPUT_NAME,
    VirtualMic,
    VirtualMicError,
    list_cable_devices,
)

BANNER = """
  speak  --  StyleTTS 2 voice -> virtual microphone (VB-CABLE)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speak",
        description="Speak text through a virtual microphone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="what to say")
    parser.add_argument("--stdin", action="store_true", help="read lines from standard input")
    parser.add_argument("--file", default=None, help="read text from a file")
    parser.add_argument("--watch", action="store_true",
                        help="with --file, keep speaking new lines as the file grows")

    parser.add_argument("--voice", default=None, help="voice name from the voices folder")
    parser.add_argument("--voices-dir", default=str(voices_mod.VOICES_DIR), help="where voices live")
    parser.add_argument("--list-voices", action="store_true", help="show the voices and exit")

    parser.add_argument("--host", default="127.0.0.1", help="speech server host")
    parser.add_argument("--port", type=int, default=9881, help="speech server port")
    parser.add_argument("--device", default=CABLE_OUTPUT_NAME,
                        help="playback device to write into (the virtual mic's write end)")
    parser.add_argument("--mode", default="sentence",
                        choices=["word", "chunk", "phrase", "sentence"],
                        help="words per synthesis call; 'word' streams most closely")
    parser.add_argument("--stream", action="store_true",
                        help="read standard input continuously and speak as it arrives")
    parser.add_argument("--queue", dest="queue_size", type=int, default=12,
                        help="chunks of audio kept ready to play")
    parser.add_argument("--backlog", type=int, default=16, help="queued work before the oldest is dropped")
    parser.add_argument("--monitor", action="store_true",
                        help="also play through the default speakers so you can hear it")
    parser.add_argument("--no-silence", action="store_true",
                        help="do not pad clips with silence")
    parser.add_argument("--save", default=None, help="write the audio to this wav instead of playing")
    parser.add_argument("--check", action="store_true", help="report the routing setup and exit")
    parser.add_argument("--quiet", action="store_true", help="less output")
    return parser


def cmd_list_voices(directory: Path) -> int:
    found = voices_mod.discover(directory)
    if not found:
        print(f"no voices in {directory}")
        print("Drop a 3-16 second wav of the voice you want in there.")
        return 1
    print(f"voices in {directory}:\n")
    for voice in found:
        print(f"  {voice.describe()}")
    print("\nUse one with:  --voice <name>")
    print(f"Rename any file to default.wav to make it the default.")
    return 0


def cmd_check(client: TtsClient, device_name: str) -> int:
    print(BANNER)
    write_end, mic_end = list_cable_devices()
    write_txt = write_end.name if write_end else f"MISSING ({CABLE_OUTPUT_NAME})"
    mic_txt = mic_end.name if mic_end else f"MISSING ({CABLE_MIC_NAME})"
    print("Routing")
    print(f"  write end (playback) : {write_txt}")
    print(f"  mic end   (capture)  : {mic_txt}")
    if write_end:
        print(f"    -> other apps should record from '{CABLE_MIC_NAME}'")
    print("\nSpeech server")
    try:
        info = client.health()
        for key in ("backend", "device", "reference_loaded", "reference_path", "reference_problem"):
            print(f"  {key:<18}: {info.get(key)}")
    except TtsError as exc:
        print(f"  not reachable: {exc}")
        return 2
    print("\nVoices")
    found = voices_mod.discover()
    if not found:
        print("  none - drop a wav in voices/")
    for voice in found:
        print(f"  {voice.describe()}")
    return 0


def read_sources(args) -> list[str]:
    chunks: list[str] = []
    if args.text:
        chunks.append(" ".join(args.text))
    if args.file:
        try:
            chunks.append(Path(args.file).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"could not read {args.file}: {exc}", file=sys.stderr)
    if args.stdin:
        chunks.append(sys.stdin.read())
    return [c.strip() for c in chunks if c and c.strip()]


def split_sentences(text: str, max_chars: int = 320) -> list[str]:
    """Break text into speakable lines.

    Synthesis quality and latency both degrade on very long inputs, and the
    style guard works per call, so long text is sent sentence by sentence.
    """
    import re

    text = " ".join(text.split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    lines: list[str] = []
    for part in parts:
        part = part.strip()
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            lines.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            lines.append(part)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    voices_dir = Path(args.voices_dir).expanduser()

    client = TtsClient(host=args.host, port=args.port)

    if args.list_voices:
        return cmd_list_voices(voices_dir)
    if args.check:
        return cmd_check(client, args.device)

    if not args.quiet:
        print(BANNER)

    # Resolve the voice before touching audio, so a bad name fails fast.
    try:
        voice = None if args.save else voices_mod.find(args.voice, voices_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"voice error: {exc}", file=sys.stderr)
        return 2

    if voice and not args.quiet:
        print(f"  voice : {voice.slug}  ({voice.seconds:.1f}s reference)")
    elif args.save and not args.quiet:
        print("  voice : (server default)")

    # Tell the server which voice to clone for this session.
    if voice is not None:
        try:
            answer = client.set_voice(voice.path)
            if answer.get("ok") is False:
                print(f"  the server would not use that voice: {answer.get('error')}", file=sys.stderr)
                return 3
        except TtsError as exc:
            print(f"  {exc}", file=sys.stderr)
            return 3

    pending = read_sources(args)
    if not pending and not (args.watch and args.file) and not args.stream:
        print("nothing to say. Pass text, --stdin, --stream, or --file.", file=sys.stderr)
        return 1

    mic: VirtualMic | None = None
    if not args.save:
        try:
            mic = VirtualMic(device=None if args.device else None, monitor=args.monitor)
            # VirtualMic resolves the cable itself; honour --device when given.
            if args.device:
                from live_transcribe.virtual_mic import find_device

                found = find_device(args.device, output=True)
                if found is None:
                    print(f"no playback device matching {args.device!r}", file=sys.stderr)
                    return 4
                mic.device = found
            mic.open()
        except VirtualMicError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 4
        if not args.quiet:
            print(f"  mic   : writing to {mic.device.name} (apps record '{CABLE_MIC_NAME}')")

    saved_samples: list[bytes] = []
    saved_rate = 0
    spoken = 0

    # A streaming source (stdin, a watched file, or any mode other than
    # sentence-at-a-time) goes through the pipelined engine, which synthesizes
    # ahead so the voice does not pause between chunks.
    streaming = (args.stream or args.watch) and not args.save
    pipeline: SpeechPipeline | None = None
    if streaming and mic is not None:
        pipeline = SpeechPipeline(
            client,
            mic,
            PipelineConfig(mode=args.mode, queue_size=args.queue_size, pending_limit=args.backlog),
            logger=(lambda _m: None) if args.quiet else (lambda m: print(m, flush=True)),
        )
        pipeline.start()

    def say(line: str) -> bool:
        """One-shot path: synthesize a complete line and play it immediately."""
        nonlocal saved_rate
        try:
            speech = client.synthesize(line)
        except TtsError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return False
        if args.save:
            saved_samples.append(speech.samples)
            saved_rate = speech.sample_rate
        elif mic is not None:
            mic.speak(speech.samples, leading_silence=not args.no_silence)
        if not args.quiet:
            print(f"  spoke {speech.seconds:4.1f}s in {speech.elapsed_s:4.2f}s: {line[:70]}")
        return True

    try:
        if pipeline is None:
            for chunk in pending:
                for line in split_sentences(chunk):
                    if say(line):
                        spoken += 1

        if args.stream:
            if not args.quiet:
                print("\n  reading text from standard input - speak each line (Ctrl+C to stop)")
            for raw in sys.stdin:
                line = raw.strip()
                if line:
                    pipeline.feed(line + " ")  # type: ignore[union-attr]

        if args.watch and args.file:
            path = Path(args.file)
            if not args.quiet:
                print(f"\n  watching {path} - speaking new text as it appears (Ctrl+C to stop)")
            seen = path.stat().st_size if path.exists() else 0
            while True:
                time.sleep(0.25)
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < seen:  # the file was cleared: a new session
                    seen = 0
                if size == seen:
                    continue
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        handle.seek(seen)
                        fresh = handle.read()
                except OSError:
                    continue
                seen = size
                if pipeline is not None:
                    pipeline.feed(fresh)
                else:
                    for line in split_sentences(fresh):
                        if say(line):
                            spoken += 1
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n  stopped")
    finally:
        if pipeline is not None:
            pipeline.flush()
            pipeline.stop(drain=True)
            spoken += pipeline.stats.chunks_spoken
        if mic is not None:
            try:
                mic.drain()
            finally:
                mic.close()

    if args.save and saved_samples:
        import wave

        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(saved_rate or 24000)
            for block in saved_samples:
                handle.writeframes(block)
        print(f"\n  saved {len(saved_samples)} line(s) to {out}")

    if pipeline is not None and not args.quiet:
        print(pipeline.summary())
    if not args.quiet:
        print(f"\n  spoke {spoken} line(s)")
    return 0 if spoken or args.watch or args.stream else 5


if __name__ == "__main__":
    sys.exit(main())
