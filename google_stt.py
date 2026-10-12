#!/usr/bin/env python
"""Google speech -> text file, driven from a browser tab.

Why this exists: Chrome and Edge ship a speech recognition engine that uses
Google's service. It is free, needs no API key, and streams results as you speak.
This script serves a local page to the browser, receives the recognised text over
a WebSocket, and keeps ``out/transcript.txt`` in sync -- the same file the local
Whisper engine writes, so the text-to-speech stage works with either.

    run-google.bat
    .venv\\Scripts\\python.exe google_stt.py            (same thing)
    .venv\\Scripts\\python.exe google_stt.py --no-open  (do not launch a browser)

Keep the tab open while you dictate. Ctrl+C here stops the server.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from web.server import BridgeServer, LiveMirror, TranscriptWriter  # noqa: E402

DEFAULT_OUT = HERE / "out" / "transcript.txt"

BANNER = """
  google-stt  --  browser speech recognition -> text file
  Uses Chrome/Edge's built-in speech service (free, no API key).
"""


def find_free_port(preferred: int = 8777) -> int:
    """Return a free TCP port, preferring `preferred` for a stable URL.

    8777 rather than the more obvious 8765: that one is commonly already taken by
    other local tools, and landing on a stranger's server produces confusing
    404s rather than an obvious error.
    """
    for port in [preferred, *range(preferred + 1, preferred + 25), 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port available")


def find_browser(explicit: str | None = None) -> str | None:
    """Locate a Chromium browser that supports the Web Speech API."""
    if explicit:
        return explicit if Path(explicit).exists() else None

    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="google-stt",
        description="Browser (Google) speech recognition into a text file.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="transcript file to maintain")
    parser.add_argument("--port", type=int, default=8777, help="local port (falls back if taken)")
    parser.add_argument("--language", default=None, help="preselect a language, e.g. en-US")
    parser.add_argument("--append", action="store_true", help="do NOT clear the transcript on start")
    parser.add_argument("--no-open", action="store_true", help="do not launch a browser")
    parser.add_argument("--browser", default=None, help="path to chrome.exe / msedge.exe")
    parser.add_argument("--verbose", action="store_true", help="log HTTP requests too")
    parser.add_argument("--quiet", action="store_true", help="suppress the banner")
    parser.add_argument(
        "--word-delay",
        default="auto",
        choices=["auto", "on", "off"],
        help="whether the voice changer replays the delays measured between your words. "
        "'auto' lets the Start/Stop button in the page decide",
    )
    parser.add_argument(
        "--word-delay-min-ms",
        type=float,
        default=80.0,
        help="shortest gap worth replaying; shorter gaps are absorbed into the words",
    )
    parser.add_argument(
        "--word-delay-max-ms",
        type=float,
        default=1500.0,
        help="cap on a replayed gap, so a hesitant speaker does not stall the voice",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out).expanduser().resolve()
    live_path = out_path.with_name("live.txt")
    # Stored beside the transcript: the page writes it, the voice changer reads it.
    timing_path = out_path.with_name("word_delay.json")

    if not args.quiet:
        print(BANNER)
        print(f"  transcript : {out_path}", flush=True)

    writer = TranscriptWriter(out_path, clear_on_start=not args.append)
    mirror = LiveMirror(live_path)

    def emit_interim(text: str, marked: str | None = None) -> None:
        # `marked` carries the inter-word delays in front of the words they belong
        # to. It already includes the in-flight guess, so it replaces the plain text
        # rather than being joined to it -- joining would say the phrase twice.
        if marked:
            mirror.write(marked, "")
        else:
            mirror.write(writer.get(), text)

    writer.emit_interim = emit_interim  # type: ignore[attr-defined]

    def on_event(message: str) -> None:
        print(message, flush=True)

    page = (HERE / "web" / "index.html").read_text(encoding="utf-8")
    # A tiny injected default saves selecting the language in the page every time.
    page = page.replace("__LANGUAGE__", args.language or "")

    port = find_free_port(args.port)
    server = BridgeServer(
        ("127.0.0.1", port),
        writer,
        on_event,
        page,
        timing_path=timing_path,
        timing_mode=args.word_delay,
        timing_min_gap_ms=args.word_delay_min_ms,
        timing_max_gap_ms=args.word_delay_max_ms,
    )
    server.verbose = args.verbose  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{port}/"

    if not args.quiet:
        state = server.timing_state()
        print(
            f"  word delays: {state['mode']} "
            f"(replay gaps of {state['min_gap_ms']:.0f}-{state['max_gap_ms']:.0f} ms)",
            flush=True,
        )

    if port != args.port:
        # Worth saying out loud: with two instances running, the second one lands
        # on a different port, so a browser still pointed at the first would talk
        # to the wrong server -- which looks like the transcript going nowhere.
        print(
            f"  note       : port {args.port} was busy, using {port} instead\n"
            f"               (another copy may already be running - open the URL below)",
            flush=True,
        )

    thread = threading.Thread(target=server.serve_forever, name="bridge-server", daemon=True)
    thread.start()

    # flush=True throughout: these lines are the user's only feedback that the
    # server is up, and stdout is block-buffered whenever it is redirected.
    print(f"  page       : {url}", flush=True)
    print(f"  live view  : {live_path}", flush=True)

    if not args.no_open:
        browser = find_browser(args.browser)
        if browser:
            subprocess.Popen([browser, url], close_fds=True)
            print(f"  opened     : {Path(browser).name}", flush=True)
        else:
            print("  no Chrome/Edge found - open the page above manually", flush=True)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass
    else:
        print("  open the page above in Chrome or Edge", flush=True)

    print("\n  Speak in the tab. Ctrl+C here to stop.\n", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[stop] shutting down")
    finally:
        server.shutdown()
        server.server_close()

    if writer.updates:
        print(f"\n  transcript : {out_path}")
        print(f"  updates    : {writer.updates}")
        preview = writer.get()
        print(f"  final text : {preview[-160:] if preview else '(empty)'}")
    else:
        print(f"\n  nothing was transcribed - check the microphone level in the page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
