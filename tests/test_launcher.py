"""Launch google_stt.py the way run-google.bat does, then verify it end to end.

Checks the banner and detected port, fetches the served page and confirms the
language was injected, submits a phrase over HTTP, and confirms it lands in
out/transcript.txt. This is the "does the launcher actually work" test.

    .venv\\Scripts\\python.exe tests\\test_launcher.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "transcript.txt"
        out.write_text("STALE", encoding="utf-8")
        server_proc: subprocess.Popen[str] | None = None

        def shutdown_server() -> None:
            """Always reap the server, or the next run collides with it."""
            nonlocal server_proc
            if server_proc is None:
                return
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5)
            except OSError:
                pass
            finally:
                server_proc = None

        try:
            server_proc = subprocess.Popen(
                [
                    str(PYTHON),
                    "-u",  # unbuffered, so the banner arrives immediately
                    str(HERE / "google_stt.py"),
                    "--no-open",
                    "--language",
                    "en-GB",
                    "--out",
                    str(out),
                    # A dedicated port. A copy the user is already running must not
                    # collide with this, and this test must never end up talking to
                    # their live server -- it would inject phrases into their
                    # real transcript.
                    "--port",
                    "8799",
                ],
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            port = None
            banner_lines: list[str] = []
            deadline = time.time() + 25
            assert server_proc.stdout is not None
            while time.time() < deadline:
                line = server_proc.stdout.readline()
                if not line:
                    break
                banner_lines.append(line.rstrip())
                match = re.search(r"127\.0\.0\.1:(\d+)", line)
                if match:
                    port = int(match.group(1))
                    break
                if "Traceback" in line:
                    break

            check("server started and reported a port", port is not None, banner_lines)
            check("transcript declared in banner", any("transcript" in ln for ln in banner_lines), banner_lines)
            check("transcript cleared on start", out.read_text(encoding="utf-8") == "")
            check(
                "used the requested port (no silent fallback)",
                port == 8799,
                f"got {port}; a stale server is probably holding it",
            )

            if port:
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(f"{base}/", timeout=5) as response:
                    page = response.read().decode()
                check("page served", "Live transcript" in page and "<script" in page)
                check("language injected", 'data-default="en-GB"' in page and "__LANGUAGE__" not in page)
                check("page has level meter", "levelbar" in page)
                check("page tells the user it needs Chrome", "Chrome" in page)

                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    check("health endpoint", json.loads(response.read())["ok"] is True)

                # Simulate the page posting a phrase.
                request = urllib.request.Request(
                    f"{base}/submit",
                    data=json.dumps({"text": "hello from the launcher", "mode": "append"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    check("phrase accepted", json.loads(response.read())["ok"] is True)

                time.sleep(0.3)
                check(
                    "phrase written to the transcript",
                    out.read_text(encoding="utf-8") == "hello from the launcher",
                    out.read_text(encoding="utf-8"),
                )
                check("live.txt created", (out.parent / "live.txt").exists())
        finally:
            shutdown_server()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
