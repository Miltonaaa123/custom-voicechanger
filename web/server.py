"""Serve the browser page and receive its transcript over WebSocket.

Deliberately dependency-free: a small threaded HTTP server plus just enough of
RFC 6455 to accept one WebSocket, so nothing needs installing beyond the existing
virtual environment.

Used by ``google_stt.py``; not intended to be run directly.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

WEB_DIR = Path(__file__).resolve().parent
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Transcript text longer than this is rejected -- a guard against a runaway client
# filling the disk. Long dictation sessions are still well inside it.
MAX_TEXT = 200_000


def encode_text_frame(payload: str) -> bytes:
    """Build a masked-less (server->client) text frame."""
    data = payload.encode("utf-8")
    header = bytearray([0x81])  # FIN + text opcode
    n = len(data)
    if n < 126:
        header.append(n)
    elif n < (1 << 16):
        header.append(126)
        header += struct.pack(">H", n)
    else:
        header.append(127)
        header += struct.pack(">Q", n)
    return bytes(header) + data


def read_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes, or None once the peer closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[int, bytes] | None:
    """Read one WebSocket frame, returning (opcode, unmasked payload)."""
    head = read_exact(sock, 2)
    if head is None:
        return None
    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F
    if length == 126:
        raw = read_exact(sock, 2)
        if raw is None:
            return None
        length = struct.unpack(">H", raw)[0]
    elif length == 127:
        raw = read_exact(sock, 8)
        if raw is None:
            return None
        length = struct.unpack(">Q", raw)[0]
    if length > 8 << 20:  # 8 MB is far beyond anything a transcript post needs
        return None

    mask = read_exact(sock, 4) if masked else b"\x00" * 4
    if mask is None:
        return None
    payload = read_exact(sock, length)
    if payload is None:
        return None
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


class TranscriptWriter:
    """Keeps out/transcript.txt in sync with what the browser heard.

    The page sends its current settled text on every update, so the file is simply
    replaced when it differs -- the same "only rewrite on real change" behaviour
    the Whisper engine uses, which keeps downstream file watchers calm.
    """

    def __init__(self, path: Path, clear_on_start: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read() if not clear_on_start else ""
        if clear_on_start:
            self._write("")
        self.updates = 0
        self.started = time.time()

    def _read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _write(self, text: str) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def set_text(self, text: str) -> bool:
        with self._lock:
            if text == self._last:
                return False
            self._write(text)
            self._last = text
            self.updates += 1
            return True

    def append(self, text: str) -> str:
        clean = text.strip()
        with self._lock:
            current = self._last
            if not clean:
                updated = current
            elif not current:
                updated = clean
            elif current[-1].isspace():
                updated = current + clean
            else:
                updated = f"{current} {clean}"
            if updated != current:
                self._write(updated)
                self._last = updated
                self.updates += 1
            return updated

    def get(self) -> str:
        with self._lock:
            return self._last


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    TIMING_MODES = ("auto", "on", "off")

    def __init__(
        self,
        addr,
        writer: TranscriptWriter,
        on_event: Callable[[str], None],
        page: str,
        timing_path: Path | None = None,
        timing_mode: str = "auto",
        timing_min_gap_ms: float = 80.0,
        timing_max_gap_ms: float = 1500.0,
    ):
        super().__init__(addr, BridgeHandler)
        self.writer = writer
        self.on_event = on_event
        self.page = page
        self.connected = threading.Event()
        self.client_language = ""
        self.session_started = time.time()

        # -- word-delay replay -------------------------------------------- #
        # The page can measure how long the speaker left between words, but only
        # the voice changer can put those delays back into the speech. This is the
        # switch they share: the page sets it, the voice changer polls it, and it
        # is stored so the choice survives a restart of either side.
        self.timing_path = Path(timing_path) if timing_path else None
        self.timing_mode = timing_mode if timing_mode in self.TIMING_MODES else "auto"
        self.timing_min_gap_ms = float(timing_min_gap_ms)
        self.timing_max_gap_ms = float(timing_max_gap_ms)
        self._load_timing()

    # -- shared word-delay setting ----------------------------------------- #
    def _load_timing(self) -> None:
        if self.timing_path is None or not self.timing_path.exists():
            return
        try:
            data = json.loads(self.timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        mode = str(data.get("mode", ""))
        if mode in self.TIMING_MODES:
            self.timing_mode = mode
        for key, attr in (("min_gap_ms", "timing_min_gap_ms"), ("max_gap_ms", "timing_max_gap_ms")):
            try:
                value = float(data[key])
            except (KeyError, TypeError, ValueError):
                continue
            setattr(self, attr, value)

    def set_timing(self, mode: str) -> bool:
        """Set 'auto' | 'on' | 'off'; returns False for anything else."""
        if mode not in self.TIMING_MODES:
            return False
        self.timing_mode = mode
        if self.timing_path is not None:
            try:
                self.timing_path.parent.mkdir(parents=True, exist_ok=True)
                self.timing_path.write_text(
                    json.dumps(
                        {
                            "mode": self.timing_mode,
                            "min_gap_ms": self.timing_min_gap_ms,
                            "max_gap_ms": self.timing_max_gap_ms,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        return True

    def timing_state(self) -> dict:
        """What the voice changer needs to decide, and what the page displays."""
        return {
            "mode": self.timing_mode,
            "enabled": self.timing_mode != "off",
            "min_gap_ms": self.timing_min_gap_ms,
            "max_gap_ms": self.timing_max_gap_ms,
            "replay": (
                "the voice changer replays the delays it is sent"
                if self.timing_mode != "off"
                else "measured delays are ignored; the model keeps its own pace"
            ),
        }


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LiveTranscriptBridge/1.0"

    def setup(self) -> None:
        """Install the peeked-line replay before the base class builds its reader."""
        super().setup()
        self._pushback: list[bytes] = []
        # Bound to the bound method so the patch cannot recurse into itself.
        self._real_readline = self.rfile.readline
        self.rfile.readline = self._readline_with_pushback  # type: ignore[method-assign]

    def finish(self) -> None:
        try:
            self.rfile.readline = self._real_readline  # type: ignore[method-assign]
        except Exception:  # noqa: BLE001
            pass
        super().finish()

    # -- helpers ----------------------------------------------------------- #
    def log_message(self, fmt, *args):  # noqa: ANN001 - silence default logging
        if getattr(self.server, "verbose", False):
            self.server.on_event(f"[http] {fmt % args}")  # type: ignore[attr-defined]

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, status: int = 200) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json", status)

    # -- routes ------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 - required name
        server: BridgeServer = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            self._send_bytes(server.page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            self._send_json(
                {
                    "text": server.writer.get(),
                    "updates": server.writer.updates,
                    "connected": server.connected.is_set(),
                    "language": server.client_language,
                }
            )
        elif path == "/timing":
            # Polled by the voice changer, which lives in another process and
            # cannot be told directly when the page's switch is flipped.
            self._send_json(server.timing_state())
        elif path == "/health":
            self._send_json({"ok": True})
        else:
            self._send_bytes(b"not found", "text/plain", 404)

    def do_POST(self) -> None:  # noqa: N802 - required name
        from urllib.parse import urlparse

        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 16 << 20:
            self._send_json({"ok": False, "error": "bad content length"}, 400)
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"ok": False, "error": "bad json"}, 400)
            return

        if path == "/submit":
            text = str(payload.get("text", ""))
            if len(text) > MAX_TEXT:
                self._send_json({"ok": False, "error": "text too long"}, 413)
                return
            mode = payload.get("mode", "set")
            if mode == "append":
                final = self.server.writer.append(text)  # type: ignore[attr-defined]
            else:
                self.server.writer.set_text(text)  # type: ignore[attr-defined]
                final = self.server.writer.get()
            self._send_json({"ok": True, "length": len(final), "updates": self.server.writer.updates})  # type: ignore[attr-defined]
        elif path == "/event":
            kind = str(payload.get("kind", "event"))
            detail = str(payload.get("detail", ""))[:400]
            self.server.on_event(f"[page] {kind}: {detail}" if detail else f"[page] {kind}")  # type: ignore[attr-defined]
            self._send_json({"ok": True})
        elif path == "/timing":
            mode = str(payload.get("mode", "auto"))
            if not self.server.set_timing(mode):  # type: ignore[attr-defined]
                self._send_json({"ok": False, "error": f"unknown mode {mode!r}"}, 400)
                return
            state = self.server.timing_state()  # type: ignore[attr-defined]
            self.server.on_event(f"[page] word delays -> {state['mode']}")  # type: ignore[attr-defined]
            self._send_json({"ok": True, **state})
        else:
            self._send_json({"ok": False, "error": "unknown path"}, 404)

    # -- websocket --------------------------------------------------------- #
    def handle_one_request(self) -> None:
        """Serve WebSocket upgrades; hand every other request to http.server.

        `http.server` cannot route an upgrade, so the request line and headers are
        peeked at first. When this is not an upgrade, every peeked line is pushed
        back and the parent implementation runs untouched -- so the normal HTTP
        path stays exactly as the standard library implements it.

        (Pushing lines back rather than re-parsing here matters:
        `BaseHTTPRequestHandler.rfile` is a write-only property, so swapping in a
        BytesIO has no effect and the request would be parsed twice.)
        """
        peeked: list[bytes] = []

        def peek_line() -> bytes:
            if peeked:
                return peeked.pop(0)
            return self._real_readline(65537)

        try:
            request_line = peek_line()
        except (ConnectionError, OSError):
            self.close_connection = True
            return
        if not request_line:
            self.close_connection = True
            return

        headers: dict[str, str] = {}
        header_lines: list[bytes] = []
        is_upgrade = False
        try:
            while True:
                raw = peek_line()
                if not raw or raw in (b"\r\n", b"\n"):
                    break
                header_lines.append(raw)
                text = raw.decode("iso-8859-1")
                name, _, value = text.partition(":")
                headers[name.strip().lower()] = value.strip()
                if name.strip().lower() == "upgrade":
                    is_upgrade = value.strip().lower() == "websocket"
        except (ConnectionError, OSError, UnicodeDecodeError):
            self.close_connection = True
            return

        if not is_upgrade:
            # Not ours: replay the original bytes for the parent implementation.
            self._pushback = [request_line, *header_lines, b"\r\n"]
            try:
                super().handle_one_request()
            finally:
                self._pushback = []
            return

        self._handle_websocket(headers)

    def _readline_with_pushback(self, size: int = -1) -> bytes:
        """Replay peeked lines first, then read normally."""
        if self._pushback:
            return self._pushback.pop(0)
        return self._real_readline(size)
    def _handle_websocket(self, headers: dict[str, str]) -> None:
        server: BridgeServer = self.server  # type: ignore[assignment]
        key = headers.get("sec-websocket-key", "")
        if not key:
            self.send_error(400, "missing Sec-WebSocket-Key")
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()

        self.wfile.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        self.wfile.flush()

        sock = self.connection
        sock.settimeout(None)
        server.connected.set()
        server.on_event("[ws] browser connected")

        try:
            while True:
                frame = read_frame(sock)
                if frame is None:
                    break
                opcode, payload = frame
                if opcode == 0x8:  # close
                    break
                if opcode == 0x9:  # ping -> pong
                    sock.sendall(b"\x8a\x00")
                    continue
                if opcode not in (0x1, 0x2):
                    continue
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

                kind = msg.get("kind")
                if kind == "hello":
                    server.client_language = str(msg.get("language", ""))
                    server.on_event(f"[ws] page ready (lang={server.client_language or 'default'})")
                    sock.sendall(encode_text_frame(json.dumps({"kind": "welcome"})))
                    # Tell the page where the shared word-delay switch is set, so a
                    # reloaded page shows the choice that is actually in force.
                    sock.sendall(
                        encode_text_frame(
                            json.dumps({"kind": "timing", **server.timing_state()})
                        )
                    )
                elif kind == "timing":
                    server.set_timing(str(msg.get("mode", "auto")))
                    server.on_event(f"[page] word delays -> {server.timing_mode}")
                elif kind == "final":
                    text = str(msg.get("text", ""))
                    if len(text) <= MAX_TEXT:
                        before = server.writer.get()
                        after = server.writer.append(text)
                        if after != before:
                            server.on_event(
                                f'[heard] {text.strip()}\n        -> transcript: "...{after[-70:]}"'
                            )
                            sock.sendall(encode_text_frame(json.dumps({"kind": "ack", "length": len(after)})))
                elif kind == "interim":
                    text = str(msg.get("text", ""))
                    # The delays the page measured, in front of the words they
                    # belong to. They go to live.txt -- the file the voice changer
                    # follows -- and never into the clean transcript.
                    marked = str(msg.get("marked", "")) or None
                    if len(text) <= MAX_TEXT and (marked is None or len(marked) <= MAX_TEXT):
                        server.writer.emit_interim(text, marked)
                elif kind == "event":
                    server.on_event(f"[page] {msg.get('detail', kind)}")
                elif kind == "clear":
                    server.writer.set_text("")
                    server.on_event("[page] transcript cleared")
        except (ConnectionError, OSError):
            pass
        finally:
            server.connected.clear()
            server.on_event("[ws] browser disconnected")
        self.close_connection = True


class LiveMirror:
    """Writes live.txt with the in-flight guess appended, mirroring the Whisper engine."""

    MARKER = " \u2039 "

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = ""
        # Created up front so the file exists even before the first word, and so a
        # viewer opened early does not report "missing".
        try:
            self.path.write_text("", encoding="utf-8")
        except OSError:
            pass

    def write(self, committed: str, interim: str) -> None:
        live = f"{committed}{self.MARKER}{interim}" if interim else committed
        with self._lock:
            if live == self._last:
                return
            self._last = live
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(live, encoding="utf-8")
            tmp.replace(self.path)


def _attach_interim(writer: TranscriptWriter, mirror: LiveMirror) -> None:
    """Give the writer an emit_interim hook (kept separate for clarity)."""

    def emit_interim(text: str, marked: str | None = None) -> None:
        """The page's in-flight guess, and optionally the words with their delays.

        `marked` already holds the committed words *and* the same guess, because
        that is what the voice changer has to read: it is a stream of text, and the
        delays belong in front of the words they were measured for. Appending the
        guess a second time behind it would make the voice changer say the sentence
        twice, so when the page supplies the marked version it replaces the plain
        one rather than joining it.
        """
        if marked:
            mirror.write(marked, "")
        else:
            mirror.write(writer.get(), text.strip())

    writer.emit_interim = emit_interim  # type: ignore[attr-defined]
