"""End-to-end: what the page measures comes back out of the virtual microphone.

The three stages are separate processes that meet at a file, which is where this
kind of feature usually breaks: the browser measures the pauses between words, the
bridge writes them into the file the voice changer follows, and the speech pipeline
has to put them back into the audio it plays. Each stage is tested on its own
elsewhere; this drives all three in one process to prove the handoff works.

No audio hardware and no speech server are involved: the synthesizer and the
microphone are stand-ins that record what they were asked to do.

    .venv\\Scripts\\python.exe tests\\test_timing_flow.py
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from live_transcribe.pipeline import PipelineConfig, SpeechPipeline  # noqa: E402
from live_transcribe.tts import Speech  # noqa: E402
from tests.test_bridge import ws_connect, ws_send  # noqa: E402
from voice_changer import TranscriptFollower  # noqa: E402
from web.server import BridgeServer, LiveMirror, TranscriptWriter  # noqa: E402


def page_session(port: int) -> socket.socket:
    """Connect as the page does and get past the handshake."""
    sock = ws_connect(port)
    ws_send(sock, json.dumps({"kind": "hello", "language": "en-US"}))
    from tests.test_bridge import ws_recv

    ws_recv(sock)  # welcome
    ws_recv(sock)  # the timing setting the server pushes on connect
    return sock


class RecordingTts:
    """Stands in for StyleTTS 2, recording each call it is asked to make."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def synthesize(self, text: str) -> Speech:
        self.seen.append(text)
        # A tone, not silence: a clip with no signal at all is left alone by the
        # trimmer, and then the test would not exercise trimming at all.
        rate = 24000
        phase = 2 * np.pi * 220 * np.arange(int(0.2 * rate)) / rate
        tone = (0.4 * np.sin(phase) * 32767).astype("<i2")
        padded = np.concatenate([np.zeros(int(0.05 * rate), dtype="<i2"), tone,
                                 np.zeros(int(0.05 * rate), dtype="<i2")])
        return Speech(
            samples=padded.tobytes(),
            sample_rate=rate,
            seconds=padded.size / rate,
            text=text,
            elapsed_s=0.01,
        )

    def set_voice(self, _path) -> dict:  # noqa: ANN001
        return {"ok": True}

    def timing(self) -> dict | None:
        return None


class RecordingMic:
    """Stands in for the virtual microphone, recording silence and speech in order."""

    def __init__(self) -> None:
        self.silence_ms: list[float] = []
        self.played: list[bytes] = []

    def speak(self, pcm16: bytes, leading_silence: bool = True) -> float:
        self.played.append(pcm16)
        return len(pcm16) / 2.0 / 24000.0

    def write_silence(self, ms: float | None = None) -> None:
        self.silence_ms.append(float(ms or 0.0))

    def drain(self, ms: float = 0) -> None:
        pass

    def request_abort(self) -> None:
        pass


def main() -> int:
    failures = 0

    def check(name: str, condition: bool, detail: object = "") -> None:
        nonlocal failures
        if condition:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transcript.txt"
        live = Path(tmp) / "live.txt"
        writer = TranscriptWriter(out, clear_on_start=True)
        mirror = LiveMirror(live)

        def emit_interim(text: str, marked: str | None = None) -> None:
            if marked:
                mirror.write(marked, "")
            else:
                mirror.write(writer.get(), text)

        writer.emit_interim = emit_interim  # type: ignore[attr-defined]

        page = (HERE.parent / "web" / "index.html").read_text(encoding="utf-8")
        timing_store = Path(tmp) / "word_delay.json"
        server = BridgeServer(
            ("127.0.0.1", 0), writer, lambda _m: None, page,
            timing_path=timing_store, timing_mode="on",
        )
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        tts, mic = RecordingTts(), RecordingMic()

        def new_pipeline() -> SpeechPipeline:
            # Phrase grouping so that what splits the synthesis calls is the delays
            # and nothing else; the pipeline's own chunking is tested separately.
            pipe = SpeechPipeline(
                tts,
                mic,
                PipelineConfig(mode="phrase", word_delay_replay=True, word_gap_min_ms=80.0),
            )
            pipe.set_timing_resolver(lambda: server.timing_state()["enabled"])
            return pipe

        # The voice changer follows live.txt, so that is what it reads here.
        follower = TranscriptFollower(live)
        pipe = new_pipeline()

        def run_once(pipeline: SpeechPipeline, text: str) -> None:
            """Drive one turn of the voice changer's loop: read, flush, speak."""
            pipeline.start()
            time.sleep(0.3)
            pipeline.feed(text)
            pipeline.flush_partial()
            time.sleep(0.4)
            pipeline.flush()
            pipeline.stop(drain=True)

        def speak_as_the_page(text: str, marked: str) -> str:
            """Send an interim the way the page does, then read it as the follower does.

            The page updates one result in place, so the file grows a word at a time
            and the follower sees only the new tail each time. Driving it in steps is
            what makes this the real path rather than an idealised one: the words and
            their delays have to arrive correctly when they are delivered piecemeal.
            """
            collected = ""
            words = marked.split(" ")
            for end in range(2, len(words) + 1, 2):
                ws_send(sock, json.dumps({
                    "kind": "interim", "text": text, "marked": " ".join(words[:end]),
                }))
                time.sleep(0.15)
                collected += follower.read_new()
            ws_send(sock, json.dumps({"kind": "interim", "text": text, "marked": marked}))
            time.sleep(0.2)
            collected += follower.read_new()
            return collected

        try:
            sock = page_session(port)

            # -- the page speaks, with the pauses it measured ------------------ #
            fresh = speak_as_the_page("hello what is", "hello [[250]] what [[400]] is")

            check("the transcript stays plain for a person to read",
                  out.read_text(encoding="utf-8") == "", repr(out.read_text(encoding="utf-8")))
            check("the voice changer's view carries the delays",
                  "[[250]]" in live.read_text(encoding="utf-8"),
                  live.read_text(encoding="utf-8"))
            check("the follower collects the marked text as it grows",
                  "[[250]]" in fresh and "[[400]]" in fresh and "is" in fresh, fresh)
            run_once(pipe, fresh)

            # -- and it comes back out in the speaker's own rhythm ------------- #
            check("each replayed delay is one synthesis call per side",
                  tts.seen == ["hello", "what", "is"], tts.seen)
            check("the model was never asked to say a mark",
                  all("[[" not in t for t in tts.seen), tts.seen)
            check("the pauses are written into the microphone",
                  mic.silence_ms == [250.0, 400.0], mic.silence_ms)
            check("every piece reached the microphone", len(mic.played) == 3, len(mic.played))

            # -- the switch in the page turns it off live ---------------------- #
            tts.seen.clear()
            mic.silence_ms.clear()
            ws_send(sock, json.dumps({"kind": "timing", "mode": "off"}))
            time.sleep(0.3)
            check("the switch reaches the server", server.timing_state()["enabled"] is False)

            fresh = speak_as_the_page("and now this", "hello what is [[900]] and now this")
            check("the follower sees what the page wrote after the switch",
                  "and now this" in fresh, fresh)
            # The delays are still in the text; it is the pipeline's live switch that
            # decides whether they come back out, which is what makes this usable
            # mid-conversation without restarting anything.
            run_once(new_pipeline(), fresh)

            check("with the switch off the marked words become one call",
                  tts.seen == ["hello what is and now this"], tts.seen)
            check("and no silence is inserted", mic.silence_ms == [], mic.silence_ms)
            check("the marks still never reach the speech server",
                  all("[[" not in t for t in tts.seen), tts.seen)

            sock.close()
        finally:
            try:
                pipe.stop(drain=False)
            except Exception:  # noqa: BLE001
                pass
            server.shutdown()
            server.server_close()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
