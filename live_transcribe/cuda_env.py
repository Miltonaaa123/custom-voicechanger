"""Make pip-installed NVIDIA CUDA runtime DLLs visible to CTranslate2.

CTranslate2's CUDA backend loads ``cublas64_12.dll`` and ``cudnn64_9.dll`` at
inference time. The ``nvidia-*`` pip wheels ship those DLLs inside
``site-packages/nvidia/*/bin`` but Windows does not add them to the DLL search
path automatically, which produces:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Import this module (or call :func:`add_cuda_dll_dirs`) before importing
``ctranslate2`` / ``faster_whisper``.
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_registered = False


def _candidate_bin_dirs() -> list[Path]:
    """Return every plausible directory holding NVIDIA runtime DLLs."""
    roots: list[Path] = []

    # 1. Wheels installed into this interpreter.
    for sp in {*site.getsitepackages(), site.getusersitepackages()}:
        if sp:
            roots.append(Path(sp))

    # 2. Whatever is on sys.path (covers venvs, src layouts, embedded installs).
    for entry in sys.path:
        if entry:
            roots.append(Path(entry))

    # 3. An optional fully local copy shipped next to this file.
    roots.append(Path(__file__).resolve().parent.parent / ".cuda_dlls")

    bin_dirs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            for leaf in ("bin", "lib"):
                d = nvidia / sub / leaf
                if d.is_dir():
                    key = str(d).lower()
                    if key not in seen:
                        seen.add(key)
                        bin_dirs.append(d)
        # Some layouts flatten everything into nvidia/bin
        flat = nvidia / "bin"
        if flat.is_dir():
            key = str(flat).lower()
            if key not in seen:
                seen.add(key)
                bin_dirs.append(flat)
    return bin_dirs


def add_cuda_dll_dirs(verbose: bool = False) -> list[str]:
    """Register NVIDIA DLL directories with the Windows loader.

    Returns the list of directories that were registered. Safe to call more
    than once and safe to call on non-Windows platforms (no-op there).
    """
    global _registered
    if os.name != "nt":
        return []

    dirs = _candidate_bin_dirs()
    added: list[str] = []
    for d in dirs:
        text = str(d)
        # os.add_dll_directory is the reliable mechanism on Python 3.8+.
        try:
            os.add_dll_directory(text)
            added.append(text)
        except (OSError, AttributeError):
            continue
        # Also prepend to PATH: some libraries use a plain LoadLibrary call
        # that ignores the add_dll_directory list.
        if text not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")

    _registered = True
    if verbose:
        print(f"[cuda] registered {len(added)} DLL director{'y' if len(added) == 1 else 'ies'}")
        for a in added:
            print(f"[cuda]   {a}")
    return added


def diagnose() -> str:
    """Return a human-readable report used by --doctor and error messages."""
    lines = [f"platform        : {sys.platform}", f"executable      : {sys.executable}"]
    try:
        import ctranslate2

        lines.append(f"ctranslate2     : {ctranslate2.__version__}")
        try:
            n = ctranslate2.get_cuda_device_count()
            lines.append(f"cuda devices    : {n}")
            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            lines.append(f"cuda compute    : {', '.join(types) if types else 'none'}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"cuda query      : FAILED {exc}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"ctranslate2     : NOT IMPORTABLE ({exc})")

    dirs = _candidate_bin_dirs()
    lines.append(f"nvidia dll dirs : {len(dirs)} found")
    for d in dirs:
        names = sorted(p.name for p in d.glob("*.dll"))
        lines.append(f"  - {d}")
        for n in names:
            if n.lower().startswith(("cublas", "cudnn")):
                lines.append(f"      {n}")
    if not any(d.glob("cublas64_*.dll") for d in dirs):
        lines.append("WARNING: cublas64_12.dll not found -> run:")
        lines.append("         .venv\\Scripts\\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    return "\n".join(lines)
