#!/usr/bin/env python
"""Install everything Rixers Custom Voice Changer needs, then verify it.

One entry point for both halves of the installation:

    python setup.py               # the app and the speech engine
    python setup.py app           # just this project's virtual environment
    python setup.py engine        # just StyleTTS 2 (its own Python 3.11 venv + models)
    python setup.py check         # report what is installed; change nothing
    python setup.py engine --reuse-legacy
                                  # adopt an existing Riko StyleTTS 2 install
                                  # (its venv and model files) instead of downloading

Why two environments
--------------------
The voice changer itself runs on this project's ``.venv`` (3.10 here) and needs
faster-whisper, sounddevice and numpy. StyleTTS 2 cannot live there: its
dependency chain ends at ``transformers<5``, which pins ``tokenizers<0.20``, and
those releases publish no wheel for Python 3.12+. ``py -3.11`` is the version that
resolves cleanly, so the model gets its own interpreter under
``engine/styletts2/.venv`` -- exactly as the upstream instructions describe, but
automated.

What it downloads
-----------------
Roughly 1.6 GB: the two PyTorch wheels, the LibriTTS checkpoint, and four smaller
model files the checkpoint config refers to. Every download is skipped when the
file is already there, so re-running this is cheap and safe. Nothing here touches
the network except to fetch a missing file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_transcribe import engine_paths as paths  # noqa: E402

APP_VENV = HERE / ".venv"
APP_PYTHON = APP_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

#: torch builds, newest first, with the compute capability each is for. The
#: installer picks the first one the driver supports; `--torch cu118` overrides.
TORCH_BUILDS = [
    ("cu128", "https://download.pytorch.org/whl/cu128", "CUDA 12.8 (RTX 50-series / Blackwell)"),
    ("cu126", "https://download.pytorch.org/whl/cu126", "CUDA 12.6"),
    ("cu121", "https://download.pytorch.org/whl/cu121", "CUDA 12.1"),
    ("cu118", "https://download.pytorch.org/whl/cu118", "CUDA 11.8"),
]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
DEFAULT_TORCH = "cu121"
"""What the original install on this project's machine used (GTX 10-series era)."""

DOWNLOAD_TIMEOUT = 3600
CHUNK = 1024 * 1024

#: Where uv keeps its cache, when it is used. It is deliberately inside the project:
#: uv's own default lives under the user profile, which a restricted environment can
#: refuse to write ("Failed to initialize cache ... Access is denied"), and a cache
#: next to the environment it feeds is also the one that gets deleted with it.
UV_CACHE_DIR = HERE / ".uv-cache"


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
class Log:
    """Console output with a consistent prefix, and a record for the marker file."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.steps: list[str] = []

    def step(self, text: str) -> None:
        self.steps.append(text)
        if not self.quiet:
            print(f"\n==> {text}", flush=True)

    def info(self, text: str) -> None:
        if not self.quiet:
            print(f"    {text}", flush=True)

    def warn(self, text: str) -> None:
        print(f"    ! {text}", flush=True)

    def error(self, text: str) -> None:
        print(f"    x {text}", file=sys.stderr, flush=True)


def run(command: list[str], log: Log, what: str, cwd: Path | None = None,
        env: dict | None = None) -> bool:
    """Run a command, streaming its output, and report whether it worked."""
    log.info(f"$ {' '.join(command[:3])} ...")
    try:
        result = subprocess.run(command, cwd=str(cwd or HERE), env=env)
    except OSError as exc:
        log.error(f"{what} could not start: {exc}")
        return False
    if result.returncode != 0:
        log.error(f"{what} failed (exit {result.returncode})")
        return False
    return True


def find_uv() -> Path | None:
    """uv, if it is installed.

    StyleTTS 2 pulls a large dependency graph (torch, transformers, librosa, numba,
    langchain), and pip's resolver spent more than twenty minutes on it in testing --
    most of it backtracking, not downloading. uv resolves the same graph in about
    three. It is used when present and pip when it is not, so the installer never
    *requires* a tool the user has not got.
    """
    found = shutil.which("uv")
    if found:
        return Path(found)
    for candidate in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python313" / "Scripts" / "uv.exe",
        Path.home() / ".local" / "bin" / "uv.exe",
        Path.home() / ".cargo" / "bin" / "uv.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def uv_env() -> dict:
    """The environment uv needs to work here: a cache it can actually write."""
    environment = dict(os.environ)
    environment["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
    environment.setdefault("UV_HTTP_TIMEOUT", "300")
    return environment


def install_packages(
    interpreter: Path,
    packages: list[str],
    log: Log,
    what: str,
    index_url: str | None = None,
    upgrade: bool = False,
) -> bool:
    """Install packages with uv when it is available, pip otherwise."""
    uv = find_uv()
    if uv is not None:
        log.info(f"using uv ({uv.name})")
        command = [str(uv), "pip", "install", "--python", str(interpreter)]
        if upgrade:
            command.append("--upgrade")
        command += packages
        if index_url:
            command += ["--index-url", index_url]
        if run(command, log, what, env=uv_env()):
            return True
        log.warn("uv could not do it - falling back to pip")
    command = [str(interpreter), "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    command += packages
    if index_url:
        command += ["--index-url", index_url]
    return run(command, log, what)


# --------------------------------------------------------------------------- #
# finding an interpreter
# --------------------------------------------------------------------------- #
def find_python(version: str = paths.ENGINE_PYTHON) -> Path | None:
    """The interpreter for `version`, from the py launcher or PATH.

    `py -3.11` is the launcher Windows installs with python.org builds; it is
    checked first because it is the only thing that reliably finds a specific
    version. `python3.11`/`python3` cover the other cases.
    """
    candidates: list[list[str]] = []
    if os.name == "nt":
        candidates.append(["py", f"-{version}"])
    candidates += [[f"python{version}"], ["python3"]]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [*candidate, "-c", "import sys;print(sys.version_info[:2])"],
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        try:
            found = eval(result.stdout.strip())  # noqa: S307 - our own print
        except (SyntaxError, ValueError):
            continue
        if found == tuple(int(part) for part in version.split(".")):
            try:
                which = subprocess.run(
                    [*candidate, "-c", "import sys;print(sys.executable)"],
                    capture_output=True, text=True, timeout=60,
                )
                return Path(which.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                continue
    return None


def app_python() -> Path:
    """The interpreter to use for the project itself."""
    if APP_PYTHON.is_file():
        return APP_PYTHON
    return Path(sys.executable)


# --------------------------------------------------------------------------- #
# downloads
# --------------------------------------------------------------------------- #
def download(url: str, target: Path, minimum: int, log: Log, label: str = "") -> bool:
    """Fetch `url` to `target` unless it is already there and big enough.

    Downloaded to a `.part` file and moved into place only on success, so an
    interrupted download can never be mistaken for a finished one -- which matters
    when the finished one is a 736 MB checkpoint whose absence only shows up as a
    cryptic unpickling error much later.
    """
    label = label or target.name
    if target.is_file() and target.stat().st_size >= minimum:
        log.info(f"{label}: already installed ({target.stat().st_size // 1048576} MB)")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.time()
    log.info(f"{label}: downloading {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rixers-setup/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            got = 0
            last_report = 0.0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                got += len(block)
                now = time.time()
                if not log.quiet and now - last_report > 2.0:
                    last_report = now
                    if total:
                        share = got / total
                        speed = got / max(0.001, now - started) / 1048576
                        print(f"      {share * 100:5.1f}%  {got // 1048576:5d}/{total // 1048576} MB"
                              f"  ({speed:5.1f} MB/s)", end="\r", flush=True)
                    else:
                        print(f"      {got // 1048576} MB", end="\r", flush=True)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        partial.unlink(missing_ok=True)
        log.error(f"{label} could not be downloaded: {exc}")
        return False
    if not log.quiet:
        print(" " * 60, end="\r")
    size = partial.stat().st_size
    if size < minimum:
        partial.unlink(missing_ok=True)
        log.error(
            f"{label} came back as {size // 1024} KB, which is far too small - "
            "usually an error page from a proxy or a truncated transfer. Try again."
        )
        return False
    partial.replace(target)
    log.info(f"{label}: {size // 1048576} MB in {time.time() - started:.0f}s")
    return True


def download_all_models(log: Log) -> bool:
    """Every model file the engine needs, skipping the ones already present."""
    ok = True
    for name, spec in paths.FILES.items():
        if not download(spec["url"], Path(spec["path"]), int(spec["min_bytes"]), log,
                        label=spec["label"]):
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# the app environment
# --------------------------------------------------------------------------- #
def install_app(log: Log, recreate: bool = False) -> bool:
    log.step("This project's environment (.venv)")
    if recreate and APP_VENV.exists():
        log.info("removing the old environment (--recreate)")
        shutil.rmtree(APP_VENV, ignore_errors=True)
    if APP_PYTHON.is_file():
        log.info(f"reusing {APP_VENV}")
    else:
        log.info(f"creating {APP_VENV}")
        if not run([sys.executable, "-m", "venv", str(APP_VENV)], log, "venv creation"):
            return False
    interpreter = app_python()
    install_packages(interpreter, ["pip"], log, "pip upgrade", upgrade=True)
    requirements = HERE / "requirements.txt"
    if not install_packages(interpreter, ["-r", str(requirements)], log,
                            "installing requirements.txt"):
        return False
    log.info("checking the packages import")
    probe = (
        "import faster_whisper, ctranslate2, numpy, scipy, sounddevice, tkinter;"
        "print('app packages OK')"
    )
    if not run([str(interpreter), "-c", probe], log, "package check"):
        log.warn("some packages did not import - the messages above say which")
        return False
    return True


# --------------------------------------------------------------------------- #
# the engine environment
# --------------------------------------------------------------------------- #
def reuse_legacy(log: Log, legacy: Path) -> bool:
    """Adopt an existing Riko install: its venv and its model files.

    Described as *reuse* rather than install because that is what happens: the
    files are moved (copied) into this project's engine directory, and an existing
    engine venv is copied across if it is Python 3.11. Nothing is downloaded.
    """
    log.step(f"Reusing the StyleTTS 2 install at {legacy}")
    paths.STYLE_DIR.mkdir(parents=True, exist_ok=True)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = legacy / paths.CHECKPOINT_NAME
    config = legacy / "config.yml"
    for source, target in ((checkpoint, paths.checkpoint()), (config, paths.config())):
        if not source.is_file():
            log.error(f"{source} is missing")
            return False
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            log.info(f"{target.name}: already there")
            continue
        log.info(f"copying {source.name} ({source.stat().st_size // 1048576} MB)")
        shutil.copy2(source, target)

    source_venv = legacy / ".venv"
    target_venv = paths.VENV_DIR
    if paths.venv_python(target_venv).is_file():
        log.info("engine venv: already there")
    elif paths.venv_python(source_venv).is_file():
        log.info(f"copying the engine venv from {source_venv}")
        shutil.copytree(source_venv, target_venv, dirs_exist_ok=True)
    else:
        log.warn("no usable engine venv to reuse - it will be built instead")
        return install_engine_venv(log)

    # The four auxiliary models are not in a Riko model directory: the library
    # downloads them into the HuggingFace cache. Adopting them from there is the
    # difference between working offline and a 1 GB surprise.
    return adopt_auxiliary_models(log)


def adopt_auxiliary_models(log: Log) -> bool:
    """Copy the auxiliary models out of the HuggingFace cache when they are there."""
    cache = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    wanted = {
        "asr": ("epoch_00080.pth", "epoch_00080.pth"),
        "f0": ("bst.t7", "bst.t7"),
        "bert": ("step_1000000.t7", "step_1000000.t7"),
    }
    found_any = False
    for key, (_label, filename) in wanted.items():
        target = Path(paths.FILES[key]["path"])
        if target.is_file():
            continue
        matches = list(cache.rglob(filename)) if cache.exists() else []
        if not matches:
            continue
        source = max(matches, key=lambda path: path.stat().st_size)
        log.info(f"{filename}: taking {source.stat().st_size // 1048576} MB from the cache")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        found_any = True
    if found_any:
        log.info("any file still missing will be downloaded")
    return download_all_models(log)


def install_engine_venv(log: Log, torch_key: str | None = None, recreate: bool = False) -> bool:
    """Build the Python 3.11 environment StyleTTS 2 can actually install into."""
    log.step("The StyleTTS 2 environment (engine/styletts2/.venv)")
    interpreter = paths.venv_python(paths.VENV_DIR)
    if recreate and paths.VENV_DIR.exists():
        log.info("removing the old engine environment (--recreate)")
        shutil.rmtree(paths.VENV_DIR, ignore_errors=True)
    if interpreter.is_file():
        log.info(f"reusing {paths.VENV_DIR}")
    else:
        python = find_python(paths.ENGINE_PYTHON)
        if python is None:
            log.error(
                f"Python {paths.ENGINE_PYTHON} is required for StyleTTS 2 and was not found.\n"
                f"      Install it from https://www.python.org/downloads/release/python-3119/\n"
                f"      and make sure `py -{paths.ENGINE_PYTHON} --version` works.\n"
                "      (Newer Pythons cannot install StyleTTS 2: its tokenizers pin has no wheel.)"
            )
            return False
        log.info(f"python {paths.ENGINE_PYTHON}: {python}")
        log.info(f"creating {paths.VENV_DIR}")
        if not run([str(python), "-m", "venv", str(paths.VENV_DIR)], log, "engine venv creation"):
            return False

    install_packages(interpreter, ["pip"], log, "pip upgrade", upgrade=True)

    index, label = torch_index(torch_key, log)
    log.info(f"torch: {label}")
    if not install_packages(interpreter, ["torch", "torchaudio"], log, "installing torch",
                            index_url=index):
        log.error(
            "torch could not be installed from that index. If the driver is older than "
            "CUDA 12.1 use --torch cu118; with no usable GPU use --torch cpu."
        )
        return False
    if not install_packages(interpreter, ["styletts2"], log, "installing styletts2"):
        return False
    # styletts2 tokenizes with NLTK, and on NLTK >= 3.9 that reads "punkt_tab",
    # which is a separate download from "punkt". Without it the model loads
    # perfectly and then dies on the first sentence with a LookupError.
    log.info("NLTK data (punkt, punkt_tab, cmudict)")
    nltk_probe = (
        "import nltk;"
        "[nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','cmudict')];"
        "print('nltk data OK')"
    )
    if not run([str(interpreter), "-c", nltk_probe], log, "NLTK data download"):
        log.warn("NLTK data could not be downloaded - the first sentence will fail until it can be")
    return True


def torch_index(key: str | None, log: Log) -> tuple[str, str]:
    """Which torch wheel index to use, defaulting to one this machine can drive."""
    if key:
        key = key.lower()
        if key == "cpu":
            return CPU_INDEX, "CPU only (slow, but it always works)"
        for name, index, label in TORCH_BUILDS:
            if name == key:
                return index, label
        log.warn(f"unknown --torch {key}; using {DEFAULT_TORCH}")
    for name, index, label in TORCH_BUILDS:
        if name == DEFAULT_TORCH:
            return index, label
    return CPU_INDEX, "CPU only"


def install_engine(log: Log, torch_key: str | None = None, recreate: bool = False,
                   reuse_legacy_install: bool = False) -> bool:
    log.step("The speech engine (StyleTTS 2)")
    paths.STYLE_DIR.mkdir(parents=True, exist_ok=True)

    if reuse_legacy_install:
        legacy = paths.legacy_install()
        if legacy is None:
            log.error("no existing StyleTTS 2 install was found to reuse")
            return False
        return reuse_legacy(log, legacy)

    if not install_engine_venv(log, torch_key, recreate):
        return False
    if not download_all_models(log):
        return False
    return True


def clean_uv_cache(log: Log) -> None:
    """Remove uv's cache once the installation is done.

    It reaches several gigabytes -- the torch wheels are most of it -- and it exists
    only to make a *second* install fast, which is not what it is being used for
    here. Leaving it would quietly double the disk cost of setting this project up.
    A reinstall simply downloads again.
    """
    if not UV_CACHE_DIR.exists():
        return
    try:
        size = sum(path.stat().st_size for path in UV_CACHE_DIR.rglob("*") if path.is_file())
    except OSError:
        size = 0
    shutil.rmtree(UV_CACHE_DIR, ignore_errors=True)
    if not UV_CACHE_DIR.exists():
        log.info(f"removed the uv cache ({size / 1073741824:.1f} GB)")
    else:
        log.info(f"could not remove {UV_CACHE_DIR} - it can be deleted by hand ({size / 1073741824:.1f} GB)")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
SMOKE_TEXT = "Hello. This is the Rixers Custom Voice Changer speech engine."


def verify_engine(log: Log) -> bool:
    """Load the model in the engine venv and make it speak one line."""
    log.step("Checking that the engine can load the model and speak")
    interpreter = paths.venv_python(paths.VENV_DIR)
    if not interpreter.is_file():
        log.error("the engine environment is not installed")
        return False
    smoke = (
        "import sys, time, json;"
        "sys.path.insert(0, r'%s');"
        "from live_transcribe import engine_tts_server as s, engine_paths as p;"
        "engine = s.StyleEngine(p.checkpoint(), p.config(), p.default_reference());"
        "t0=time.time();engine.load();load=time.time()-t0;"
        "t1=time.time();r=engine.synthesize(%r);synth=time.time()-t1;"
        "print(json.dumps({'load_s':round(load,1),'synth_s':round(synth,2),"
        "'seconds':round(r['seconds'],2),'peak':r['peak'],'rms':r['rms'],"
        "'attempts':r['attempts'],'noise_share':r['noise_share'],"
        "'render_problem':r['render_problem'],"
        "'device':s.torch_report(),'steps':r['diffusion_steps']}))"
        % (str(HERE), SMOKE_TEXT)
    )
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    log.info("loading the model (60-120 s is normal) and synthesizing one sentence")
    try:
        result = subprocess.run([str(interpreter), "-c", smoke], cwd=str(HERE),
                                env=environment, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        log.error(f"the smoke test could not run: {exc}")
        return False
    if result.returncode != 0:
        log.error("the engine did not speak:")
        for line in (result.stdout + result.stderr).strip().splitlines()[-12:]:
            log.error(f"  {line}")
        return False
    payload = {}
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                payload = {}
            break
    if not payload.get("seconds"):
        log.error(f"the engine ran but produced no audio: {result.stdout[-400:]}")
        return False
    if payload.get("render_problem"):
        log.warn(f"the test sentence needed re-rolling: {payload['render_problem']} "
                 f"(took {payload.get('attempts')} renders) - this is the guard working, "
                 "but a voice that keeps doing it is worth changing")
    log.info(f"ok: loaded in {payload.get('load_s')}s, spoke {payload.get('seconds')}s "
             f"in {payload.get('synth_s')}s on {payload.get('device')} "
             f"(noise {payload.get('noise_share')}, {payload.get('attempts')} render(s))")
    paths.write_marker(
        installed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        python=paths.ENGINE_PYTHON,
        device=str(payload.get("device")),
        load_seconds=payload.get("load_s"),
        synth_seconds=payload.get("synth_s"),
        verified=True,
    )
    return True


def verify_app(log: Log) -> bool:
    log.step("Checking the voice changer itself")
    interpreter = app_python()
    checks = [
        ([str(interpreter), "-c", "import tkinter;print('tkinter OK')"], "tkinter"),
        ([str(interpreter), "-c",
          "import faster_whisper;print('faster-whisper OK')"], "faster-whisper"),
        ([str(interpreter), "-c",
          "import sounddevice as sd;print(len(sd.query_devices()),'audio devices')"], "sounddevice"),
        ([str(interpreter), "live_transcribe/engine_paths.py"], "engine paths"),
    ]
    ok = True
    for command, what in checks:
        if not run(command, log, what):
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def report(log: Log) -> int:
    """Print what is installed, and return 0 only when everything is."""
    print("Rixers Custom Voice Changer - installation\n", flush=True)
    print(f"  project      : {HERE}")
    print(f"  app venv     : {APP_VENV if APP_PYTHON.is_file() else 'MISSING (run setup.py app)'}")

    engine = paths.inspect(deep=APP_PYTHON.is_file())
    print(f"  engine venv  : {engine.venv_python}  "
          f"{'OK' if engine.venv_exists else 'MISSING (run setup.py engine)'}")
    if engine.venv_exists:
        print(f"  engine python: {_python_version(engine.venv_python)}")
        print(f"  packages     : {'OK' if engine.packages_ok else 'INCOMPLETE'} "
              f"({engine.packages_detail})")
    print(f"  torch        : {engine.torch_build or '?'}   cuda={engine.cuda}")
    if engine.files_ok:
        total = sum(Path(spec["path"]).stat().st_size
                    for spec in paths.FILES.values() if Path(spec["path"]).is_file())
        print(f"  model files  : OK ({total / 1073741824:.2f} GB)")
    else:
        print(f"  model files  : MISSING -> {', '.join(engine.missing_files)}")
        print("                 (run setup.py engine)")
    marker = paths.read_marker()
    if marker.get("verified"):
        print(f"  verified     : {marker.get('installed_at')} on {marker.get('device')}")
    legacy = paths.legacy_install()
    if legacy:
        print(f"  note         : an existing StyleTTS 2 install was found at {legacy}")
        print("                 (setup.py engine --reuse-legacy adopts it)")
    print(f"\n  {engine.summary()}", flush=True)
    return 0 if engine.ready else 1


def _python_version(interpreter: Path) -> str:
    try:
        result = subprocess.run([str(interpreter), "-c", "import sys;print(sys.version.split()[0])"],
                                capture_output=True, text=True, timeout=120)
        return result.stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="setup.py",
        description="Install and verify everything Rixers Custom Voice Changer needs.",
    )
    parser.add_argument("what", nargs="?", default="all",
                        choices=["all", "app", "engine", "check"],
                        help="what to install (default: all)")
    parser.add_argument("--torch", default=None,
                        help="torch build: " + ", ".join(name for name, _i, _l in TORCH_BUILDS) + ", cpu")
    parser.add_argument("--recreate", action="store_true",
                        help="delete and rebuild the environment being installed")
    parser.add_argument("--reuse-legacy", action="store_true",
                        help="adopt an existing Riko StyleTTS 2 install instead of downloading")
    parser.add_argument("--skip-verify", action="store_true",
                        help="install without loading the model to check it")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = Log(quiet=args.quiet)
    started = time.time()

    if args.what == "check":
        return report(log)

    if args.what in ("all", "app"):
        if not install_app(log, recreate=args.recreate):
            log.error("the project environment did not install completely")
            return 1
    if args.what in ("all", "engine"):
        if not install_engine(log, torch_key=args.torch, recreate=args.recreate,
                              reuse_legacy_install=args.reuse_legacy):
            log.error("the speech engine did not install completely")
            return 1

    ok = True
    if not args.skip_verify and args.what in ("all", "engine"):
        ok = verify_engine(log) and ok
    if args.what in ("all", "app"):
        ok = verify_app(log) and ok
    clean_uv_cache(log)

    print(f"\n  finished in {time.time() - started:.0f}s", flush=True)
    report(log)
    if ok:
        print("\n  Everything is installed. Start it with:  RixersVoiceChanger.bat\n", flush=True)
        return 0
    print("\n  Something is still missing - see the messages above.\n", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
