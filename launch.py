"""Wait for the pipeline's services to be ready, and report what is missing.

Exists so the .bat launchers can show a definite "ready" moment instead of leaving
the user guessing whether a server that printed nothing is still loading or has
already died. Standard library only, so the project virtual environment is
optional.

    python launch.py check        # preflight: paths, venv, GPU, cable, voices
    python launch.py wait-tts     # block until the speech server answers /health
    python launch.py wait-stt     # block until the transcription page server is up
    python launch.py stop         # stop the voice changer and the speech engine
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TTS_PORT = 9881
STT_PORT = 8777

RIKO = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop" / "riko-project"
TTS_VENV_PYTHON = RIKO / "scripts" / "models" / "styletts2" / ".venv" / "Scripts" / "python.exe"
TTS_SERVER = RIKO / "scripts" / "tts_server.py"
CHECKPOINT = RIKO / "scripts" / "models" / "styletts2" / "epochs_2nd_00020.pth"

#: What this project installs for itself. Preferred over the legacy paths above,
#: which only exist on a machine that also has the Riko checkout.
try:
    sys.path.insert(0, str(HERE))
    from live_transcribe import engine_paths as engine_paths  # noqa: PLC0415

    ENGINE_VENV_PYTHON = engine_paths.venv_python()
    ENGINE_SERVER = HERE / "live_transcribe" / "engine_tts_server.py"
    ENGINE_CHECKPOINT = engine_paths.checkpoint()
except Exception:  # noqa: BLE001 - a missing package must not break the launcher
    engine_paths = None  # type: ignore[assignment]
    ENGINE_VENV_PYTHON = HERE / "engine" / "styletts2" / ".venv" / "Scripts" / "python.exe"
    ENGINE_SERVER = HERE / "live_transcribe" / "engine_tts_server.py"
    ENGINE_CHECKPOINT = HERE / "engine" / "styletts2" / "epochs_2nd_00020.pth"


def http_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) == 0


def wait_for_tts(timeout: float, quiet: bool = False) -> int:
    """Return 0 once the speech server reports itself loaded."""
    started = time.perf_counter()
    last_note = 0.0
    while time.perf_counter() - started < timeout:
        info = http_json(f"http://127.0.0.1:{TTS_PORT}/health")
        if info:
            # "loaded" means the checkpoint is in memory and synthesis can run.
            # "reference_loaded" only says the reference wav exists, so it must
            # not be enough on its own - the launcher would announce READY while
            # the model is still loading and the first synthesis would fail.
            # It stays as the fallback for engines that never report "loaded".
            ready = info.get("loaded")
            if ready is None:
                ready = info.get("reference_loaded")
            if ready:
                device = info.get("device")
                voice = Path(str(info.get("reference_path") or "")).name or "default"
                vram = info.get("vram_mb")
                if not quiet:
                    print(
                        f"  ready after {time.perf_counter() - started:.0f}s"
                        f"  ({device}{f', {vram} MB VRAM' if vram else ''}, voice: {voice})",
                        flush=True,
                    )
                return 0
            if not quiet and time.perf_counter() - last_note > 10:
                last_note = time.perf_counter()
                print(f"  answering but not loaded yet ({time.perf_counter() - started:.0f}s)", flush=True)
        elif not quiet:
            elapsed = time.perf_counter() - started
            if elapsed - last_note > 10:
                last_note = elapsed
                note = "still loading the model"
                if port_open(TTS_PORT):
                    note = "port is open, waiting for the model"
                print(f"  {elapsed:.0f}s - {note}", flush=True)
        time.sleep(0.5)
    print(
        f"  TIMEOUT after {timeout:.0f}s - the speech server never became ready.\n"
        f"  Run run-tts.bat on its own to see its error.",
        file=sys.stderr,
        flush=True,
    )
    return 1


def wait_for_stt(timeout: float, quiet: bool = False) -> int:
    """Return 0 once the transcription page server answers."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if http_json(f"http://127.0.0.1:{STT_PORT}/health"):
            if not quiet:
                print(f"  ready after {time.perf_counter() - started:.0f}s", flush=True)
            return 0
        time.sleep(0.4)
    print(f"  TIMEOUT after {timeout:.0f}s - the page server never answered.", file=sys.stderr, flush=True)
    return 1


def check() -> int:
    """Preflight: everything a full start needs. Prints one line per item."""
    problems: list[str] = []

    def line(ok: bool, label: str, detail: str = "", fatal: bool = True) -> None:
        mark = "OK  " if ok else ("FAIL" if fatal else "warn")
        print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}", flush=True)
        if not ok and fatal:
            problems.append(label)

    print("Checking what the launcher needs:", flush=True)

    line(Path(sys.executable).is_file(), "python", sys.executable)

    # The engine this project installs for itself comes first; the Riko paths are
    # only relevant on a machine that happens to have that checkout too.
    engine_installed = ENGINE_VENV_PYTHON.is_file() and ENGINE_CHECKPOINT.is_file()
    legacy_installed = TTS_VENV_PYTHON.is_file() and CHECKPOINT.is_file()
    if engine_installed:
        interpreter, checkpoint = ENGINE_VENV_PYTHON, ENGINE_CHECKPOINT
    else:
        interpreter, checkpoint = TTS_VENV_PYTHON, CHECKPOINT
    line(engine_installed, "speech engine environment", str(interpreter),
         fatal=not legacy_installed)
    line(checkpoint.is_file(), "voice checkpoint",
         f"{checkpoint.stat().st_size // (1024 * 1024)} MB" if checkpoint.is_file() else str(checkpoint),
         fatal=not legacy_installed)
    if not engine_installed:
        line(False, "self-contained engine", "run Setup.bat to install it here", fatal=False)
    line((HERE / "voices" / "default.wav").is_file(), "default voice", "voices\\default.wav", fatal=False)

    # Virtual cable endpoints, via the project's own device lookup.
    try:
        sys.path.insert(0, str(HERE))
        import sounddevice as sd  # noqa: PLC0415

        names = [d["name"] for d in sd.query_devices()]
        has_out = any("CABLE Input" in n for n in names)
        has_in = any("CABLE Output" in n for n in names)
        line(has_out, "virtual microphone (write end)", "CABLE Input")
        line(has_in, "virtual microphone (read end)", "CABLE Output")
    except Exception as exc:  # noqa: BLE001
        line(False, "audio devices", f"{type(exc).__name__}: {exc}")

    # GPU, straight from the speech environment that will use it.
    if TTS_VENV_PYTHON.is_file():
        try:
            import subprocess  # noqa: PLC0415

            result = subprocess.run(
                [str(TTS_VENV_PYTHON), "-c", "import torch;print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            ok = result.stdout.strip().endswith("True")
            line(ok, "CUDA for the speech model", result.stdout.strip() or result.stderr.strip()[:80])
        except Exception as exc:  # noqa: BLE001
            line(False, "CUDA for the speech model", f"{type(exc).__name__}: {exc}", fatal=False)

    if problems:
        print(f"\n  {len(problems)} problem(s) - fix these before starting:", flush=True)
        for item in problems:
            print(f"    - {item}", flush=True)
        return 1
    print("\n  Everything the launcher needs is present.", flush=True)
    return 0


def stop() -> int:
    """Terminate the processes this project starts.

    Matched on the exact command lines, so nothing else on the machine is
    touched. The microphone -> text stage is deliberately left alone: it may be a
    browser tab that is still in use.
    """
    import subprocess  # noqa: PLC0415

    targets = {
        "voice_changer.py": "voice changer",
        "tts_server.py": "speech engine",
    }
    if os.name != "nt":
        print("  this helper only stops processes on Windows", flush=True)
        return 1

    query = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }"
    )
    try:
        listing = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  could not list processes: {exc}", file=sys.stderr, flush=True)
        return 1

    found = 0
    for line in listing.splitlines():
        pid, _, command = line.partition("|")
        if not pid.strip().isdigit():
            continue
        for needle, label in targets.items():
            if needle in command:
                found += 1
                print(f"  stopping {label} (pid {pid.strip()})", flush=True)
                subprocess.run(
                    ["taskkill", "/PID", pid.strip(), "/T", "/F"],
                    capture_output=True,
                    timeout=30,
                )
                break

    if not found:
        print("  nothing was running.", flush=True)
    else:
        print(f"  stopped {found} process(es).", flush=True)
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command = argv[0]
    quiet = "--quiet" in argv
    timeout = 240.0
    for item in argv[1:]:
        if item.startswith("--timeout="):
            timeout = float(item.split("=", 1)[1])
    if command == "check":
        return check()
    if command == "wait-tts":
        return wait_for_tts(timeout, quiet)
    if command == "wait-stt":
        return wait_for_stt(timeout, quiet)
    if command == "stop":
        return stop()
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
