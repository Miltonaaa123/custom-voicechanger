"""Where the speech engine's files live.

One module owns every path the engine needs, so the server, the installer, the
`.bat` files and the tests cannot disagree about them:

    engine/
      styletts2/
        .venv/                       Python 3.11 with torch + styletts2
        epochs_2nd_00020.pth         the LibriTTS checkpoint (~736 MB)
        config.yml                   its config
        installed.json               what setup.py wrote, and what it verified
      models/                        the smaller models, downloaded on first use
        epochs_00080.pth             ASR (text aligner)
        ASR_config.yml
        bst.t7                       F0 / pitch extractor
        step_1000000.t7              PLBERT (style encoder)
        PLBERT_config.yml

Everything under `engine/` is downloaded, never committed: it is several gigabytes.
`EngineInstall` answers "is this machine ready", which is what the installer,
`launch.py check` and the control panel all ask.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENGINE_DIR = PROJECT_ROOT / "engine"
STYLE_DIR = ENGINE_DIR / "styletts2"
MODELS_DIR = ENGINE_DIR / "models"
VENV_DIR = STYLE_DIR / ".venv"
MARKER_NAME = "installed.json"

CHECKPOINT_NAME = "epochs_2nd_00020.pth"
CONFIG_NAME = "config.yml"

#: The model files, with the URL each comes from and a sanity floor in bytes.
#:
#: Only *weights* are downloaded. The two small YAML files that describe the
#: architectures ship inside the `styletts2` package and are read straight from
#: there, so the description always matches the installed library -- a downloaded
#: copy could drift from it.
#:
#: The floors are deliberately well under the real sizes and well over an error
#: page: they are a truncation check, not an integrity check. The real sizes are
#: in the comment beside each one.
FILES: dict[str, dict] = {
    "checkpoint": {
        "path": STYLE_DIR / CHECKPOINT_NAME,
        "url": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth",
        "min_bytes": 600 * 1024 * 1024,        # real: 735.7 MB
        "label": "StyleTTS 2 checkpoint (LibriTTS)",
    },
    "config": {
        "path": STYLE_DIR / CONFIG_NAME,
        "url": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml",
        "min_bytes": 1_000,                    # real: 1,859 bytes
        "label": "StyleTTS 2 config",
    },
    "asr": {
        "path": MODELS_DIR / "epochs_00080.pth",
        "url": "https://github.com/yl4579/StyleTTS2/raw/main/Utils/ASR/epoch_00080.pth",
        "min_bytes": 60 * 1024 * 1024,         # real: 90.2 MB
        "label": "text aligner weights (ASR)",
    },
    "f0": {
        "path": MODELS_DIR / "bst.t7",
        "url": "https://github.com/yl4579/StyleTTS2/raw/main/Utils/JDC/bst.t7",
        "min_bytes": 15 * 1024 * 1024,         # real: 20.1 MB
        "label": "pitch extractor weights (JDC)",
    },
    "bert": {
        "path": MODELS_DIR / "PLBERT" / "step_1000000.t7",
        "url": "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/step_1000000.t7",
        "min_bytes": 18 * 1024 * 1024,         # real: 24.0 MB
        "label": "style encoder weights (PLBERT)",
    },
    "bert_config": {
        # PLBERT is loaded from a *directory* that must contain both a
        # "config.yml" and a "step_*.t7", so the checkpoint and its config share a
        # folder rather than living loose in engine/models.
        "path": MODELS_DIR / "PLBERT" / "config.yml",
        "url": "https://github.com/yl4579/StyleTTS2/raw/main/Utils/PLBERT/config.yml",
        "min_bytes": 256,                      # real: 915 bytes
        "label": "style encoder config",
    },
}


@dataclass
class LibraryConfigs:
    """The architecture configs that ship inside the installed `styletts2` package."""

    asr: Path
    plbert_dir: Path

    @property
    def ok(self) -> bool:
        return self.asr.is_file() and (self.plbert_dir / "config.yml").is_file()

    def missing(self) -> list[str]:
        problems = []
        if not self.asr.is_file():
            problems.append(str(self.asr))
        if not (self.plbert_dir / "config.yml").is_file():
            problems.append(str(self.plbert_dir / "config.yml"))
        return problems


def library_configs() -> LibraryConfigs | None:
    """Where the `styletts2` package keeps its own ASR and PLBERT configs.

    These are read from the installed package rather than downloaded, so the
    architecture description can never disagree with the library that loads it.
    None means the package is not importable, which is the installer's situation
    until it has built the engine environment.
    """
    try:
        import styletts2  # noqa: PLC0415
    except ImportError:
        return None
    root = Path(styletts2.__file__).resolve().parent
    return LibraryConfigs(asr=root / "Utils" / "ASR" / "config.yml",
                          plbert_dir=root / "Utils" / "PLBERT")


def venv_python(venv: Path | None = None) -> Path:
    """The interpreter inside an engine venv, on this platform."""
    venv = Path(venv) if venv else VENV_DIR
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def checkpoint() -> Path:
    return FILES["checkpoint"]["path"]


def config() -> Path:
    return FILES["config"]["path"]


def default_reference() -> Path:
    """The voice the engine starts with: this project's default, if it exists."""
    return PROJECT_ROOT / "voices" / "default.wav"


def marker_path() -> Path:
    return STYLE_DIR / MARKER_NAME


def read_marker() -> dict:
    path = marker_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_marker(**values) -> Path:
    path = marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_marker()
    existing.update(values)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass
class EngineInstall:
    """What is present, what is missing, and whether it can run."""

    venv_python: Path = field(default_factory=venv_python)
    missing_files: list[str] = field(default_factory=list)
    python_ok: bool = False
    packages_ok: bool = False
    packages_detail: str = ""
    torch_build: str = ""
    cuda: bool = False
    marker: dict = field(default_factory=read_marker)

    @property
    def venv_exists(self) -> bool:
        return self.venv_python.is_file()

    @property
    def files_ok(self) -> bool:
        return not self.missing_files

    @property
    def ready(self) -> bool:
        return self.venv_exists and self.python_ok and self.packages_ok and self.files_ok

    def summary(self) -> str:
        if self.ready:
            gpu = "CUDA" if self.cuda else "CPU"
            return f"engine ready ({gpu}, {self.torch_build or 'torch'})"
        problems: list[str] = []
        if not self.venv_exists:
            problems.append("no engine environment")
        elif not self.packages_ok:
            problems.append(f"packages incomplete ({self.packages_detail})")
        if self.missing_files:
            problems.append(f"{len(self.missing_files)} model file(s) missing")
        return "engine not ready: " + ", ".join(problems or ["unknown"])


#: Package that must be importable inside the engine venv for it to be usable.
REQUIRED_IMPORTS = ("torch", "torchaudio", "styletts2", "numpy", "librosa", "nltk")

#: The Python the engine venv must be built on. styletts2 -> transformers<5 ->
#: tokenizers<0.20, and those tokenizers releases publish no wheel for 3.12+, so
#: pip falls back to building tokenizers from source and the install fails.
ENGINE_PYTHON = "3.11"


def inspect(python: Path | None = None, deep: bool = True) -> EngineInstall:
    """Look at the installed engine.

    `deep` runs the interpreter to check its packages and CUDA, which costs a
    second or two; the control panel asks for the shallow answer while it is
    drawing and the deep one when the user presses a button.
    """
    import subprocess

    interpreter = Path(python) if python else venv_python()
    report = EngineInstall(venv_python=interpreter)
    report.missing_files = [
        name for name, spec in FILES.items() if not Path(spec["path"]).is_file()
    ]
    if not report.venv_exists or not deep:
        return report
    try:
        result = subprocess.run(
            [str(interpreter), "-c", PROBE_SNIPPET],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.packages_detail = f"{type(exc).__name__}: {exc}"
        return report
    payload = _parse_probe(result.stdout)
    if payload is None:
        report.packages_detail = (result.stdout or result.stderr or "no output").strip()[:120]
        return report
    report.python_ok = True
    report.packages_detail = ", ".join(payload.get("missing") or []) or "all present"
    report.packages_ok = not payload.get("missing") and bool(payload.get("torch"))
    report.torch_build = str(payload.get("torch") or "")
    report.cuda = bool(payload.get("cuda"))
    return report


PROBE_SNIPPET = """
import importlib, json, sys
missing = []
versions = {}
for name in %r:
    try:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "?")
    except Exception:
        missing.append(name)
cuda = False
torch_version = versions.get("torch", "")
if "torch" not in missing:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        torch_version = torch.__version__
    except Exception:
        pass
print(json.dumps({"python": sys.version.split()[0], "missing": missing,
                  "torch": torch_version, "cuda": cuda, "versions": versions}))
""" % (REQUIRED_IMPORTS,)


def _parse_probe(text: str) -> dict | None:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            return data if isinstance(data, dict) else None
    return None


def legacy_install() -> Path | None:
    """A Riko checkout that already has a working StyleTTS 2, if there is one.

    Only used to *offer* the choice of reusing it: nothing here depends on it, and
    a clone on another machine will not have it. Checked in the two places it has
    lived so the offer works on an older layout too.
    """
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    candidates = [
        PROJECT_ROOT / "riko-project" / "scripts" / "models" / "styletts2",
        home / "Desktop" / "riko-project" / "scripts" / "models" / "styletts2",
    ]
    for candidate in candidates:
        if (candidate / CHECKPOINT_NAME).is_file() and venv_python(candidate / ".venv").is_file():
            return candidate
    return None


if __name__ == "__main__":  # a quick "where is everything" when run directly
    report = inspect()
    print(f"project      : {PROJECT_ROOT}")
    print(f"engine dir   : {ENGINE_DIR}")
    print(f"venv python  : {report.venv_python}  {'OK' if report.venv_exists else 'MISSING'}")
    print(f"packages     : {'OK' if report.packages_ok else 'MISSING'} ({report.packages_detail})")
    print(f"model files  : {'OK' if report.files_ok else 'MISSING -> ' + ', '.join(report.missing_files)}")
    print(f"torch        : {report.torch_build or '?'}  cuda={report.cuda}")
    print(f"summary      : {report.summary()}")
    legacy = legacy_install()
    if legacy:
        print(f"legacy       : {legacy} (can be reused with setup.py engine --reuse-legacy)")
    sys.exit(0 if report.ready else 1)
