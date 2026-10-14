"""Start the speech engine as a process that outlives the shell that starts it.

``run-tts.bat`` is the way to start the engine by hand: its console *is* the
server's console, which is the point -- Ctrl+C stops it and the window shows what it
is saying. A tool that restarts the engine on the user's behalf cannot use that,
because a process started as a child of the tool dies when the tool's session ends.
This starts the same server with the same arguments, in its own process group, so
the voice keeps working after the window that restarted it is gone.

    .venv\\Scripts\\python.exe tools\\engine_detach.py
    .venv\\Scripts\\python.exe tools\\engine_detach.py --reference voices\\default.wav
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE_PY = HERE / "engine" / "styletts2" / ".venv" / "Scripts" / "python.exe"
SERVER = HERE / "live_transcribe" / "engine_tts_server.py"
DEFAULT_REFERENCE = HERE / "voices" / "default.wav"
LOG = HERE / "out" / "engine.log"
HEALTH = "http://127.0.0.1:9881/health"

#: Windows flags: no console of its own, and its own process group so it is not
#: killed as part of the caller's process tree.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


def health() -> dict | None:
    try:
        import json

        with urllib.request.urlopen(HEALTH, timeout=3) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--wait", type=float, default=240.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    current = health()
    if current and current.get("loaded"):
        print(f"the engine is already answering on {HEALTH}")
        return 0
    if not ENGINE_PY.is_file():
        print(f"the engine is not installed: {ENGINE_PY}", file=sys.stderr)
        return 1

    LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = LOG.open("a", encoding="utf-8", errors="replace")
    handle.write(f"\n--- starting {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    handle.flush()

    # PYTHONUTF8 matters: the model prints IPA phonemes, and a code page that cannot
    # encode them makes synthesis fail and the model fall back to the CPU.
    environment = {**__import__("os").environ, "PYTHONUTF8": "1",
                   "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        [str(ENGINE_PY), "-u", str(SERVER), "--reference", str(args.reference)],
        cwd=str(HERE),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        close_fds=True,
        env=environment,
    )
    print(f"started the engine detached (pid {process.pid}); log: {LOG}")

    if args.no_wait:
        return 0
    deadline = time.time() + args.wait
    while time.time() < deadline:
        info = health()
        if info and info.get("loaded"):
            print(f"ready in the background: {info.get('device')}")
            return 0
        if process.poll() is not None:
            print(f"the engine exited at once (code {process.returncode}); see {LOG}",
                  file=sys.stderr)
            return 1
        time.sleep(3)
    print(f"the engine did not become ready within {args.wait:.0f}s; see {LOG}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
