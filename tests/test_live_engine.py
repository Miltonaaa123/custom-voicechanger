"""Does a settings change reach a *running* voice changer?

Starts the real ``voice_changer.py`` against a throwaway transcript and settings
file, waits for it to be up, then rewrites the settings the way the control panel
does and checks the process reports the change -- without being restarted.

Needs the StyleTTS 2 speech server on 127.0.0.1:9881 (run-tts.bat). Skips cleanly
when it is not running, so it is safe to run as part of the suite.

    .venv\\Scripts\\python.exe tests\\test_live_engine.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
sys.path.insert(0, str(HERE))

from live_transcribe import live_settings  # noqa: E402


def server_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:9881/health", timeout=2) as response:
            return bool(json.loads(response.read()).get("loaded"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return False


class Engine:
    """The voice changer, with its console collected on a thread.

    Reading the pipe inline would block until the next line arrives, and a quiet
    voice changer prints nothing for seconds at a time -- so the reads happen on a
    thread and the test only ever looks at what has already arrived.
    """

    def __init__(self, settings_path: Path, transcript: Path) -> None:
        self.lines: list[str] = []
        self.process = subprocess.Popen(
            [str(PYTHON), "-u", str(HERE / "voice_changer.py"),
             "--settings", str(settings_path),
             "--transcript", str(transcript),
             "--word-delay", "auto"],
            cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.lines.append(line.rstrip())
        except (OSError, ValueError):
            pass

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def wait_for(self, needle: str, timeout: float = 45.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.text:
                return True
            if self.process.poll() is not None:
                return needle in self.text
            time.sleep(0.25)
        return False

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main() -> int:
    if not server_up():
        print("  SKIP  the speech server is not answering on 127.0.0.1:9881")
        print("\nALL PASS (skipped)")
        return 0

    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    with tempfile.TemporaryDirectory() as tmp:
        settings_path = Path(tmp) / "settings.json"
        transcript = Path(tmp) / "transcript.txt"
        transcript.write_text("", encoding="utf-8")
        settings = live_settings.LiveSettings(settings_path)
        settings.update(word_delay="off", mode="sentence")
        settings.save()

        engine = Engine(settings_path, transcript)
        try:
            check("the engine reached the watch loop", engine.wait_for("Ctrl+C to stop", 60.0),
                  engine.text[-600:])
            check("it reports the settings file it obeys",
                  str(settings_path) in engine.text, engine.text[-600:])
            check("the default effects reach it without changing anything",
                  "no effects" in engine.text, engine.text[-600:])
            check("it uses the panel's chunking mode",
                  "mode       : sentence" in engine.text, engine.text[-600:])

            # -- a live change, exactly as the control panel writes it --------- #
            panel = live_settings.LiveSettings(settings_path)
            panel.load(force=True)
            panel.update(pause_flush_s=1.75, mode="chunk", word_gap_min_ms=140.0)
            panel.save()
            check("a live pause change is picked up", engine.wait_for("pause flush: 1.75s", 15.0),
                  engine.text[-400:])
            check("a live chunking change is picked up", engine.wait_for("mode: chunk", 15.0),
                  engine.text[-400:])
            check("a live rhythm change is picked up",
                  engine.wait_for("replaying gaps of 140-", 15.0), engine.text[-400:])

            # -- effects, including the one that reopens the device ------------ #
            panel.update(pitch_semitones=6.0, volume=1.5, gate_db=-45.0)
            panel.save()
            check("a pitch change is announced", engine.wait_for("pitch changed", 20.0),
                  engine.text[-400:])
            # A 24 kHz stream shifted up six semitones is a rate of 24000 * 2^0.5.
            check("the stream is reopened at the shifted rate",
                  engine.wait_for("stream at 33941 Hz", 20.0), engine.text[-400:])
            check("it survived reopening the virtual microphone",
                  engine.process.poll() is None, engine.process.poll())

            # -- going back to no effects must also work ----------------------- #
            panel.update(pitch_semitones=0.0, volume=1.0, gate_db=-70.0)
            panel.save()
            check("the pitch can be put back",
                  engine.wait_for("stream at 24000 Hz", 20.0), engine.text[-400:])
            check("the engine is still alive at the end", engine.process.poll() is None,
                  engine.process.poll())
        finally:
            engine.stop()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
