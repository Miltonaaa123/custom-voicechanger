"""Test the Google-STT bridge without a browser.

Acts as the page: connects over WebSocket, sends the same messages the real page
sends, and checks that out/transcript.txt and live.txt end up correct. This
exercises the WebSocket handshake, frame parsing and file writes.

    .venv\\Scripts\\python.exe tests\\test_bridge.py
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.server import BridgeServer, LiveMirror, TranscriptWriter  # noqa: E402


def ws_connect(port: int, timeout: float = 5.0) -> socket.socket:
    """Minimal client-side WebSocket handshake."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall(
        (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    # Read the handshake response headers.
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(1024)
        if not chunk:
            raise RuntimeError("server closed during handshake")
        buf += chunk
    head, _, _rest = buf.partition(b"\r\n\r\n")
    assert b"101" in head.split(b"\r\n")[0], head
    assert b"sec-websocket-accept" in head.lower(), head
    return sock


def ws_send(sock: socket.socket, text: str) -> None:
    """Send a masked client text frame."""
    payload = text.encode("utf-8")
    mask = os.urandom(4)
    header = bytearray([0x81])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < (1 << 16):
        header.append(0x80 | 126)
        header += struct.pack(">H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack(">Q", n)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def ws_recv(sock: socket.socket) -> str:
    head = sock.recv(2)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", sock.recv(8))[0]
    data = b""
    while len(data) < length:
        data += sock.recv(length - len(data))
    return data.decode("utf-8")


def main() -> int:
    failures = 0
    events: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transcript.txt"
        out.write_text("STALE FROM LAST SESSION", encoding="utf-8")
        writer = TranscriptWriter(out, clear_on_start=True)
        mirror = LiveMirror(Path(tmp) / "live.txt")

        def emit_interim(text: str, marked: str | None = None) -> None:
            mirror.write(marked if marked else writer.get(), text)

        writer.emit_interim = emit_interim  # type: ignore[attr-defined]

        # Serve the real page so the page-level contract is actually tested.
        real_page = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8")
        page = real_page.replace("__LANGUAGE__", "en-GB")
        # The word-delay switch the page and the voice changer share, stored where
        # a restart of either side can find it again.
        timing_store = Path(tmp) / "word_delay.json"
        server = BridgeServer(
            ("127.0.0.1", 0),
            writer,
            events.append,
            page,
            timing_path=timing_store,
            timing_mode="auto",
        )
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def check(name: str, condition: bool, detail: str = "") -> None:
            nonlocal failures
            if condition:
                print(f"  PASS  {name}")
            else:
                failures += 1
                print(f"  FAIL  {name}  {detail}")

        try:
            # -- clears on start ------------------------------------------ #
            check("transcript cleared on start", out.read_text(encoding="utf-8") == "")

            # -- HTTP routes ---------------------------------------------- #
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                check("GET /health", json.loads(r.read())["ok"] is True)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                body = r.read().decode()
                check("GET / serves the page", "Live transcript" in body and "<script" in body)
                check("language placeholder substituted", "__LANGUAGE__" not in body and "en-GB" in body)
                check("the page carries the delay switch", 'id="timing"' in body, "no #timing in the page")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5) as r:
                check("GET /state", "text" in json.loads(r.read()))

            # -- the shared word-delay switch ----------------------------- #
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/timing", timeout=5) as r:
                timing = json.loads(r.read())
            check("GET /timing reports a mode", timing.get("mode") == "auto", timing)
            check("auto means the delays are replayed", timing.get("enabled") is True, timing)
            check("it reports the gap window",
                timing.get("min_gap_ms") == 80.0 and timing.get("max_gap_ms") == 1500.0, timing)

            def post_timing(mode: str) -> dict:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/timing",
                    data=json.dumps({"mode": mode}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    return json.loads(r.read())

            check("POST /timing turns the delays off", post_timing("off").get("enabled") is False)
            check("the choice is written down for the voice changer",
                json.loads(timing_store.read_text(encoding="utf-8"))["mode"] == "off",
                timing_store.read_text(encoding="utf-8") if timing_store.exists() else "missing")
            try:
                bad = urllib.request.Request(
                    f"http://127.0.0.1:{port}/timing",
                    data=json.dumps({"mode": "banana"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(bad, timeout=5)
                check("an unknown mode is refused", False, "the server accepted it")
            except urllib.error.HTTPError as exc:
                check("an unknown mode is refused", exc.code == 400, exc.code)
            check("POST /timing turns them back on", post_timing("on").get("enabled") is True)

            # -- websocket flow ------------------------------------------- #
            sock = ws_connect(port)
            ws_send(sock, json.dumps({"kind": "hello", "language": "en-US"}))
            welcome = json.loads(ws_recv(sock))
            check("websocket handshake + welcome", welcome.get("kind") == "welcome", welcome)
            setting = json.loads(ws_recv(sock))
            check("the page is told where the switch is", setting.get("kind") == "timing", setting)
            check("and it is the setting just made", setting.get("mode") == "on", setting)

            ws_send(sock, json.dumps({"kind": "timing", "mode": "auto"}))
            time.sleep(0.2)
            check("the page can set it over the socket too", server.timing_mode == "auto", server.timing_mode)

            ws_send(sock, json.dumps({"kind": "interim", "text": "hello wor"}))
            time.sleep(0.25)
            check(
                "interim goes to live.txt only",
                "hello wor" in (Path(tmp) / "live.txt").read_text(encoding="utf-8")
                and out.read_text(encoding="utf-8") == "",
                repr(out.read_text(encoding="utf-8")),
            )

            ws_send(sock, json.dumps({"kind": "final", "text": "hello world "}))
            ack = json.loads(ws_recv(sock))
            check("final acknowledged", ack.get("ok") is not False, ack)
            time.sleep(0.25)
            check(
                "final text written",
                out.read_text(encoding="utf-8") == "hello world",
                repr(out.read_text(encoding="utf-8")),
            )

            # The page sends its interim text twice over: the plain words for the
            # transcript and the same words carrying the measured delays, which is
            # what the voice changer follows. Only the second may reach live.txt.
            ws_send(sock, json.dumps({
                "kind": "interim",
                "text": "and more words",
                "marked": "hello world [[250]] and more words",
            }))
            time.sleep(0.25)
            live = (Path(tmp) / "live.txt").read_text(encoding="utf-8")
            check("the delays reach the voice changer's view through live.txt",
                "[[250]]" in live, repr(live))
            check("and not the transcript", out.read_text(encoding="utf-8") == "hello world",
                repr(out.read_text(encoding="utf-8")))

            ws_send(sock, json.dumps({"kind": "final", "text": "and more words "}))
            time.sleep(0.25)
            check(
                "successive phrases join with a space",
                out.read_text(encoding="utf-8") == "hello world and more words",
                repr(out.read_text(encoding="utf-8")),
            )

            ws_send(sock, json.dumps({"kind": "clear"}))
            time.sleep(0.25)
            check("clear empties the file", out.read_text(encoding="utf-8") == "")

            # -- HTTP submit path (used as a fallback by the page) -------- #
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/submit",
                data=json.dumps({"text": "via http", "mode": "append"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                check("POST /submit", json.loads(r.read())["ok"] is True)
            check(
                "http submit appended",
                out.read_text(encoding="utf-8") == "via http",
                repr(out.read_text(encoding="utf-8")),
            )

            sock.close()
            time.sleep(0.3)
            check("disconnect noted", not server.connected.is_set())
            check("events were logged", any("browser connected" in e for e in events), events[:3])

            # -- append mode preserves existing text ---------------------- #
            out2 = Path(tmp) / "keep.txt"
            out2.write_text("KEEP ME", encoding="utf-8")
            w2 = TranscriptWriter(out2, clear_on_start=False)
            check("append mode keeps old text", out2.read_text(encoding="utf-8") == "KEEP ME")
            check("append mode remembers it", w2.get() == "KEEP ME", w2.get())
        finally:
            server.shutdown()
            server.server_close()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
