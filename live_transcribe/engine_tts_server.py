"""The speech engine: StyleTTS 2 behind the same little HTTP API as before.

This project used to borrow Riko's server (``riko-project/scripts/tts_server.py``),
which meant a fresh clone had no speech engine at all. This is the replacement: it
depends only on the ``styletts2`` package from PyPI and its own model files, all of
which ``setup.py engine`` installs, so the voice changer is self-contained.

The API is the one ``live_transcribe/tts.py`` already speaks:

    GET  /health            -> what is loaded, on what device, with which voice
    POST /synthesize        -> {"text": ...} -> {"audio_base64": ...}
    POST /reload-reference  -> {"path": ...} swaps the cloned voice

Two details are not obvious and both matter:

* **torch.load.** PyTorch 2.6 flipped ``weights_only`` to True by default, and
  StyleTTS 2 predates that, so its checkpoint and its ASR model fail to unpickle
  with "Weights only load failed". The patch is applied here, before the model is
  built, and it is why this file patches rather than merely calling the library.
* **Saturation.** The duration predictor occasionally runs away and the decoder
  returns a full-scale, clipped burst -- "earrape" -- instead of speech. It is
  rare at the default five diffusion steps and much more likely above them, so a
  saturated render is retried at three steps, which is the setting measured to be
  safe. Retrying is much cheaper than the alternative, which is the user hearing
  it.

    python -m live_transcribe.engine_tts_server
    python -m live_transcribe.engine_tts_server --cpu
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 9881
SAMPLE_RATE = 24000
"""StyleTTS 2's own output rate. The pipeline expects exactly this."""

DEFAULT_DIFFUSION_STEPS = 5
"""The library's default. More is more varied and slower, and likelier to blow up."""

SAFE_DIFFUSION_STEPS = 3
"""What a saturated render is retried with."""

MAX_TEXT_CHARS = 1200
"""Roughly the point where the library splits the request itself."""

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 16.0
MAX_BODY_BYTES = 64 * 1024 * 1024

EXIT_MISSING_MODEL = 3


# --------------------------------------------------------------------------- #
# torch compatibility
# --------------------------------------------------------------------------- #
def resample_audio(audio, orig_rate: int, target_rate: int):
    """Resample a mono float array, the way ``librosa.load(..., sr=...)`` would.

    ``scipy.signal.resample_poly`` is the same polyphase filter librosa uses
    underneath, and it is what keeps a 44.1 kHz reference clip from aliasing when it
    becomes 24 kHz. The rate pair is reduced by their greatest common divisor first,
    so 44100 -> 24000 becomes the 147/80 filter rather than something enormous.
    """
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    if orig_rate == target_rate or audio.size == 0:
        return audio
    divisor = gcd(int(orig_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(orig_rate) // divisor
    return resample_poly(np.asarray(audio, dtype=np.float32), up, down).astype(np.float32)


def patch_style_loader() -> bool:
    """Load the reference clip without librosa.

    ``StyleTTS2.compute_style`` opens the reference voice with

        wave, sr = librosa.load(path, sr=24000)

    and that one call is the only reason the whole librosa stack -- lazy_loader,
    numba, audioread, the lot -- is needed at runtime. On this machine importing any
    of librosa's heavy core submodules hangs indefinitely (measured: 10+ minutes,
    600 s of CPU, no output) because of numba's JIT machinery, which is also why the
    model appeared to load and then never speak.

    ``soundfile`` reads a wav directly, and ``torchaudio``'s mel transform -- which
    the library has already built at import time -- does the rest. The replacement
    reproduces the original exactly, including the ``top_db=30`` trim and the 24 kHz
    conversion, so a reference clip becomes the same style vector it always did.

    Returns True when the patch was applied.
    """
    try:
        import numpy as np
        from styletts2 import tts as tts_module
    except ImportError:
        return False

    if getattr(tts_module, "_rixers_style_patch_applied", False):
        return False

    def compute_style(self, path):  # noqa: ANN001, ANN201 - mirrors the library
        import soundfile as sf

        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)          # mono, the way librosa.load does it
        if audio.size == 0:
            raise ValueError(f"{path} contains no audio")
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            # The same 30 dB-below-peak trim librosa.effects.trim performs.
            threshold = peak * (10.0 ** (-30.0 / 20.0))
            loud = np.flatnonzero(np.abs(audio) > threshold)
            if loud.size:
                audio = audio[int(loud[0]): int(loud[-1]) + 1]
        if int(rate) != SAMPLE_RATE:
            audio = resample_audio(audio, int(rate), SAMPLE_RATE)
        mel_tensor = tts_module.preprocess(audio).to(self.device)
        with tts_module.torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        return tts_module.torch.cat([ref_s, ref_p], dim=1)

    tts_module.StyleTTS2.compute_style = compute_style
    tts_module._rixers_style_patch_applied = True  # type: ignore[attr-defined]
    return True


def warm_imports() -> None:
    """Resolve the imports the first sentence needs, before the server starts.

    Two separate reasons, both learned the hard way while this server was written:

    * ``librosa`` is a lazy-loader package: ``import librosa`` binds a stub and each
      submodule is imported by ``lazy_loader.__getattr__`` the first time it is
      touched. Doing that from the load thread while the interpreter is already busy
      importing can deadlock inside ``importlib._bootstrap``. Resolving them on the
      main thread first removes the race, and moves the cost out of the first
      sentence where it would otherwise look like unexplained latency.
    * ``librosa.core.audio`` must **not** be imported eagerly. It pulls in
      ``audioread``, whose import probes every audio backend on the machine
      (ffmpeg, gstreamer, ...) and was measured hanging for minutes on this machine.
      It is never needed: ``librosa.load`` goes through ``soundfile`` when soundfile
      and numpy are importable, which they always are here, so ``audioread`` stays
      an unused fallback. ``librosa.effects`` -- which ``compute_style`` does use --
      is cheap and is imported deliberately.
    """
    import importlib

    for name in ("numpy", "torch", "torchaudio", "soundfile", "yaml", "nltk"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            print(f"  ! could not import {name}: {exc}", file=sys.stderr, flush=True)
    # librosa is deliberately absent: `patch_style_loader` removes the only call to
    # it at runtime, and importing its core submodules hangs on some machines (see
    # that function's docstring). `styletts2.tts` imports the package itself at
    # module level, which is a cheap stub under lazy_loader and does not pull in the
    # heavy submodules.


def patch_torch_load() -> bool:
    """Let StyleTTS 2's checkpoints load on PyTorch 2.6 and later.

    Returns True when the patch was needed and applied. Written as a module-level
    function so it runs before any model is built, and so it can be undone.
    """
    try:
        import torch
    except ImportError:
        return False
    original = getattr(torch, "_rixers_original_load", None)
    if original is None:
        original = torch.load
        torch._rixers_original_load = original  # type: ignore[attr-defined]

    def load(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors torch.load exactly
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load  # type: ignore[assignment]
    return True


def patch_library_bug() -> bool:
    """Fix the inverted path check that makes the pitch extractor undownloadable.

    ``StyleTTS2.load_model`` validates its auxiliary model paths like this:

        if not ASR_config or not Path(ASR_config).exists():   # correct
        if not ASR_path or not Path(ASR_path).exists():       # correct
        if F0_path or not Path(F0_path).exists():             # INVERTED
        if not BERT_dir_path or not Path(BERT_dir_path).exists():   # correct

    Every sibling check is right; that one lost its ``not``. The consequence is that
    ``F0_path`` from the config is *never* used -- the condition is true for any
    non-empty path -- so the pitch extractor is always fetched by URL. With no
    network, or behind a proxy, that is a hang inside an uninterruptible download
    rather than an error, and it is what this project's installer hit while it was
    being written: the same 20 MB file that took one second through this project's
    own downloader stalled indefinitely behind the library's.

    The method's source is edited in memory and recompiled, so nothing on disk is
    touched and reinstalling the package restores the original. Returns True when the
    fix was applied, False when the library already behaves.
    """
    try:
        import inspect
        import re
        import textwrap
        from styletts2 import tts as tts_module
    except ImportError:
        return False

    # The patched method is an exec'd function with no source file, so asking for
    # its source again would raise. The flag makes this idempotent, which matters
    # because the engine can be built more than once in a process (and a test suite
    # will).
    if getattr(tts_module, "_rixers_f0_patch_applied", False):
        return False

    method = tts_module.StyleTTS2.load_model
    try:
        source = textwrap.dedent(inspect.getsource(method))
    except OSError:
        # No source to read: either already patched by something else, or the
        # package is a frozen build with no .py on disk. Nothing to do.
        return False
    if "if F0_path or not Path(F0_path).exists():" not in source:
        return False
    fixed = source.replace(
        "if F0_path or not Path(F0_path).exists():",
        "if not F0_path or not Path(F0_path).exists():",
    )
    # `getsource` hands back a method still indented for the class body. Dedented it
    # is an ordinary function, so it is compiled as one and then bound to the class
    # -- exec'ing the indented text directly is what raised "unexpected indent".
    fixed = re.sub(r"^def load_model\(", "def _load_model_patched(", fixed, count=1)
    namespace: dict = {}
    exec(compile(fixed, "<patched StyleTTS2.load_model>", "exec"),  # noqa: S102
         dict(vars(tts_module)), namespace)
    tts_module.StyleTTS2.load_model = namespace["_load_model_patched"]
    tts_module._rixers_f0_patch_applied = True  # type: ignore[attr-defined]
    return True


def patch_plbert_bug() -> bool:
    """Fix the string path join that breaks loading PLBERT from a configured folder.

    ``styletts2.Utils.PLBERT.util.load_plbert`` finds its checkpoint like this:

        if not checkpoint_path:
            log_dir = <the PLBERT_dir from the config, if any>
            ...
            checkpoint_path = log_dir / f"step_{iters}.t7"      # log_dir is a str

    ``log_dir`` arrives as a string and is never converted, so the very moment the
    config supplies a ``PLBERT_dir`` -- which is exactly what makes it load from
    disk instead of downloading -- the call dies with::

        TypeError: unsupported operand type(s) for /: 'str' and 'str'

    The published config omits ``PLBERT_dir``, which is why this is not hit in
    normal use: the download path passes both a config and a checkpoint explicitly
    and never reaches that line.

    The replacement resolves the same directory, in the same way, and passes both
    paths explicitly. Returns True when the patch was applied.
    """
    try:
        import inspect
        import re
        import textwrap
        from styletts2.Utils.PLBERT import util as plbert_module
    except ImportError:
        return False

    if getattr(plbert_module, "_rixers_plbert_patch_applied", False):
        return False

    try:
        source = textwrap.dedent(inspect.getsource(plbert_module.load_plbert))
    except OSError:
        return False
    if "checkpoint_path = log_dir / f\"step_{iters}.t7\"" not in source:
        return False
    fixed = re.sub(r"^def load_plbert\(", "def _load_plbert_patched(", source, count=1)
    fixed = fixed.replace(
        "checkpoint_path = log_dir / f\"step_{iters}.t7\"",
        "checkpoint_path = os.path.join(str(log_dir), f\"step_{iters}.t7\")",
    )
    namespace: dict = {}
    exec(compile(fixed, "<patched load_plbert>", "exec"),  # noqa: S102
         dict(vars(plbert_module)), namespace)
    patched = namespace["_load_plbert_patched"]
    plbert_module.load_plbert = patched
    # `styletts2.tts` does `from .Utils.PLBERT.util import load_plbert` at import
    # time, so the name it calls is a *separate reference* to the original: patching
    # only the defining module leaves the broken one in use. Both are rebound.
    try:
        from styletts2 import tts as tts_module

        tts_module.load_plbert = patched  # type: ignore[attr-defined]
    except ImportError:
        pass
    plbert_module._rixers_plbert_patch_applied = True  # type: ignore[attr-defined]
    return True


def patch_config(config_path: Path) -> dict:
    """Point the checkpoint's config at the weights this project installed.

    The published ``config.yml`` names its auxiliary models relative to an empty
    ``root_path`` and against the *working directory*:

        ASR_config: Utils/ASR/config.yml       resolved from the CWD
        ASR_path:   Utils/ASR/epoch_00080.pth
        F0_path:    Utils/JDC/bst.t7

    so it only resolves when the process is started from a checkout of StyleTTS 2.
    Rather than make the working directory load-bearing, those entries are replaced
    with absolute paths: the two architecture configs are read from the installed
    package (so they cannot drift from the library that loads them) and the three
    weight files are read from ``engine/models``. ``PLBERT_dir`` is added for the
    same reason -- without it the library downloads PLBERT every time.

    Returns the parsed config, for the log and for tests.
    """
    import live_transcribe.engine_paths as paths
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} is not a StyleTTS 2 config")

    bundled = paths.library_configs()
    if bundled is None or not bundled.ok:
        detail = f": {', '.join(bundled.missing())}" if bundled else ""
        raise FileNotFoundError(
            "the styletts2 package is installed but its Utils/ configs are missing"
            f"{detail}\n  run:  Setup.bat        (or: python setup.py engine)"
        )
    config["ASR_config"] = str(bundled.asr)
    config["ASR_path"] = str(paths.MODELS_DIR / "epochs_00080.pth")
    config["F0_path"] = str(paths.MODELS_DIR / "bst.t7")
    # PLBERT is loaded by *directory*: the library looks for `config.yml` and a
    # `step_*.t7` inside `PLBERT_dir`, so that folder holds both. Without the entry
    # the library downloads PLBERT from the network on every start.
    config["PLBERT_dir"] = str(paths.MODELS_DIR / "PLBERT")
    return config


def redirect_model_urls() -> list[str]:
    """Stop the library reaching the network for models we have already installed.

    ``load_model`` takes paths for the checkpoint and the config, but three
    auxiliary models -- the text aligner, the pitch extractor and PLBERT -- are
    resolved through ``cached_path``. Two things make that a problem rather than an
    inconvenience:

    * **One of the config entries is ignored.** The library checks the pitch
      extractor with ``if F0_path or not Path(F0_path).exists()`` -- inverted, so
      ``F0_path`` from the config can *never* be used and it always downloads. With
      no network, or with a proxy, that is a hang rather than an error.
    * **It does not retry.** A dropped transfer stalls or fails the load outright,
      which is exactly what happened while this installer was being written: the
      20 MB pitch extractor stalled indefinitely behind ``cached_path`` while the
      same file downloaded in one second through this project's own downloader.

    So the resolver itself is replaced: a URL we have installed is answered from
    disk, and a URL for a file that is *missing* raises with the fix in the
    message instead of downloading it again. Anything else is passed through to the
    library's own resolver.

    Returns the keys it has redirected, for the log and for tests.
    """
    import live_transcribe.engine_paths as paths

    try:
        from styletts2 import tts as tts_module
    except ImportError:
        return []

    original = getattr(tts_module, "_rixers_original_cached_path", None)
    if original is None:
        original = tts_module.cached_path
        tts_module._rixers_original_cached_path = original  # type: ignore[attr-defined]

    by_url = {
        spec["url"]: (name, Path(spec["path"]))
        for name, spec in paths.FILES.items()
        if spec.get("url")
    }
    served: list[str] = []

    def cached_path(url_or_filename, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        entry = by_url.get(str(url_or_filename))
        if entry is None:
            return original(url_or_filename, *args, **kwargs)
        name, path = entry
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} is not installed: {path}\n"
                "  run:  Setup.bat        (or: python setup.py engine)"
            )
        served.append(name)
        return str(path)

    tts_module.cached_path = cached_path  # type: ignore[attr-defined]
    return served

def torch_report() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name}, torch {torch.__version__})"
        return f"cpu (torch {torch.__version__}, no usable CUDA device)"
    except ImportError as exc:
        return f"torch is not installed ({exc})"


def vram_mb() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
    except Exception:  # noqa: BLE001 - reporting only
        return None


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def float_to_pcm16(samples) -> bytes:
    """StyleTTS 2 returns floats in [-1, 1]; the client wants 16-bit mono PCM."""
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    return np.clip(data * 32767.0, -32768, 32767).astype("<i2").tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_info(path: Path) -> tuple[float, int, str]:
    """(seconds, sample_rate, problem) for a candidate reference clip.

    Bounds are checked here rather than only in the client so a bad reference is
    refused with a reason instead of quietly cloning badly.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return 0.0, 0, f"cannot read the file ({exc})"
    if size > 6 * 1024 * 1024:
        return 0.0, 0, f"{size // (1024 * 1024)} MB is over the 6 MB limit"
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except wave.Error:
        # librosa reads far more formats than `wave` does, so a non-wav is only
        # rejected if it is also unreadable -- the length check is skipped.
        return 0.0, 0, ""
    if rate <= 0:
        return 0.0, 0, "unreadable sample rate"
    seconds = frames / float(rate)
    if seconds < MIN_REFERENCE_SECONDS:
        return seconds, rate, (
            f"only {seconds:.1f}s - a clone needs at least {MIN_REFERENCE_SECONDS:.0f}s of speech"
        )
    if seconds > MAX_REFERENCE_SECONDS:
        return seconds, rate, (
            f"{seconds:.1f}s is over the {MAX_REFERENCE_SECONDS:.0f}s limit - "
            "trim it to one or two sentences"
        )
    return seconds, rate, ""


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class StyleEngine:
    """Loads StyleTTS 2 once and answers synthesis requests.

    Every public method takes ``self._lock``: one GPU, one synthesis at a time.
    """

    def __init__(
        self,
        checkpoint: Path,
        config: Path,
        reference: Path | None = None,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        force_cpu: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.config = Path(config)
        self.reference = Path(reference) if reference else None
        self.diffusion_steps = int(diffusion_steps)
        self.force_cpu = bool(force_cpu)

        self.model = None
        self.load_error = ""
        self.load_seconds = 0.0
        self.synthesised = 0
        self.saturation_retries = 0
        self.config_data: dict = {}
        self._config_file: Path | None = None
        self._styles: dict[str, object] = {}
        self._lock = threading.RLock()
        self._loading = threading.Event()
        self._ready = threading.Event()
        """Set only once the checkpoint is fully loaded, so `/health` cannot claim
        to be ready while the weights are still being read."""

    # -- loading ----------------------------------------------------------- #
    @property
    def loaded(self) -> bool:
        """True only when the checkpoint is fully loaded and synthesis can run."""
        return self._ready.is_set() and self.model is not None

    def missing_files(self) -> list[Path]:
        return [path for path in (self.checkpoint, self.config) if not path.is_file()]

    def load(self) -> None:
        """Build the model. Safe to call twice; the second call does nothing."""
        with self._lock:
            if self.model is not None:
                return
            missing = self.missing_files()
            if missing:
                raise FileNotFoundError(
                    "the StyleTTS 2 model is not installed:\n"
                    + "\n".join(f"  missing: {path}" for path in missing)
                    + "\n  run:  Setup.bat        (or: python setup.py engine)"
                )
            started = time.perf_counter()
            patch_torch_load()
            # Order matters: `patch_library_bug` recompiles `load_model` with a
            # *copy* of the module's globals, so whatever `load_plbert` is bound to
            # at that moment is the one it will keep calling. Patching PLBERT first
            # means the copy contains the fixed version.
            if patch_plbert_bug():
                print("  patch     : fixed the library's string join in load_plbert", flush=True)
            if patch_library_bug():
                print("  patch     : fixed the library's inverted F0_path check", flush=True)
            if patch_style_loader():
                print("  patch     : reference clips are read with soundfile, not librosa",
                      flush=True)
            self.config_data = patch_config(self.config)
            # A temp file, because the library only takes a path: this is the
            # published config with our absolute auxiliary-model paths in it.
            import tempfile  # noqa: PLC0415

            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".yml", delete=False, encoding="utf-8", prefix="rixers-config-"
            )
            import yaml  # noqa: PLC0415

            yaml.safe_dump(self.config_data, handle, allow_unicode=True)
            handle.close()
            self._config_file = Path(handle.name)
            served = redirect_model_urls()
            if served:
                print(f"  models    : served from disk ({', '.join(sorted(set(served)))})",
                      flush=True)
            import torch  # noqa: PLC0415 - only needed once the files exist
            from styletts2 import tts  # noqa: PLC0415

            if self.force_cpu:
                torch.cuda.is_available = lambda: False  # type: ignore[assignment]
            # Assigning to a local first matters: the library sets up the model
            # object before it has loaded any weights, so publishing it on `self`
            # early would make `/health` report ready while the checkpoint is still
            # being read -- and a client that believed it would then sit waiting on
            # the lock this method holds. Seen in testing: the health endpoint said
            # "loaded" for a minute before synthesis could actually run.
            model = tts.StyleTTS2(
                model_checkpoint_path=str(self.checkpoint),
                config_path=str(self._config_file),
            )
            self.model = model
            self._ready.set()
            self.load_seconds = time.perf_counter() - started
            # Warm the style cache for the starting voice so the first sentence is
            # not slower than the rest.
            if self.reference is not None and self.reference.is_file():
                try:
                    self.style_for(self.reference)
                except Exception as exc:  # noqa: BLE001 - a bad reference is not fatal
                    print(f"  ! could not prepare {self.reference.name}: {exc}", flush=True)

    def load_in_background(self) -> threading.Thread:
        """Start loading; `/health` reports `loaded: false` until it finishes."""

        def work() -> None:
            try:
                self.load()
                print(f"  ready in {self.load_seconds:.1f}s "
                      f"({torch_report()}, reference: "
                      f"{self.reference.name if self.reference else 'model default'})",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - reported through /health
                self.load_error = f"{type(exc).__name__}: {exc}"
                print(f"  ! the model could not be loaded: {self.load_error}", flush=True)
                traceback.print_exc()
            finally:
                self._loading.set()

        thread = threading.Thread(target=work, name="styletts2-load", daemon=True)
        thread.start()
        return thread

    # -- voices ------------------------------------------------------------ #
    def style_for(self, path: Path):
        """The style vector for a reference clip, computed once and kept."""
        key = str(path.resolve()).lower()
        cached = self._styles.get(key)
        if cached is not None:
            return cached
        if self.model is None:
            raise RuntimeError("the model is not loaded")
        vector = self.model.compute_style(str(path))
        self._styles[key] = vector
        return vector

    def set_reference(self, path: Path) -> dict:
        """Make `path` the voice every later request clones."""
        path = Path(path).expanduser()
        if not path.is_file():
            return {"ok": False, "error": f"no file at {path}"}
        seconds, rate, problem = wav_info(path)
        if problem:
            return {"ok": False, "error": problem}
        with self._lock:
            if self.model is None:
                return {"ok": False, "error": "the model is not loaded"}
            try:
                self.style_for(path)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                return {"ok": False, "error": f"could not read the reference: {exc}"}
            self.reference = path
        return {
            "ok": True,
            "reference_path": str(path),
            "reference_seconds": round(seconds, 2),
            "sample_rate": rate,
        }

    # -- speech ------------------------------------------------------------ #
    def synthesize(self, text: str, steps: int | None = None) -> dict:
        """Render `text` in the current voice. Returns PCM and the numbers about it."""
        import numpy as np  # noqa: PLC0415

        with self._lock:
            if self.model is None:
                raise RuntimeError(self.load_error or "the model is not loaded")
            reference = self.reference if (self.reference and self.reference.is_file()) else None
            style = self.style_for(reference) if reference is not None else None
            wanted = int(steps or self.diffusion_steps)
            started = time.perf_counter()
            audio = self.model.inference(
                text,
                target_voice_path=(str(reference) if reference else None),
                output_sample_rate=SAMPLE_RATE,
                diffusion_steps=wanted,
                ref_s=style,
            )
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak >= 0.999 and wanted > SAFE_DIFFUSION_STEPS:
                # A full-scale, flat-topped render is the duration predictor having
                # run away, not a loud voice. Retry lower, where it does not happen.
                self.saturation_retries += 1
                audio = self.model.inference(
                    text,
                    target_voice_path=(str(reference) if reference else None),
                    output_sample_rate=SAMPLE_RATE,
                    diffusion_steps=SAFE_DIFFUSION_STEPS,
                    ref_s=style,
                )
                peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
                wanted = SAFE_DIFFUSION_STEPS
            self.synthesised += 1
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            return {
                "samples": audio,
                "peak": round(peak, 4),
                "rms": round(rms, 4),
                "sample_rate": SAMPLE_RATE,
                "seconds": len(audio) / float(SAMPLE_RATE),
                "diffusion_steps": wanted,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "reference_path": str(reference) if reference else None,
                "cloned": reference is not None,
            }

    def info(self) -> dict:
        return {
            "backend": "styletts2",
            "engine": "rixers-styletts2",
            "loaded": self.loaded,
            "loading": self._loading.is_set() is False and not self.loaded,
            "device": torch_report(),
            "reference_loaded": bool(self.reference and self.reference.is_file()),
            "reference_path": str(self.reference) if self.reference else None,
            "checkpoint": str(self.checkpoint),
            "diffusion_steps": self.diffusion_steps,
            "synthesised": self.synthesised,
            "saturation_retries": self.saturation_retries,
            "load_seconds": round(self.load_seconds, 1),
            "vram_mb": vram_mb(),
            "load_error": self.load_error,
            "endpoints": ["GET /health", "POST /synthesize", "POST /reload-reference"],
        }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "RixersStyleTTS2/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------- #
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": "body must be a JSON object"})
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._json(400, {"error": "body is not valid JSON"})
            return None
        if not isinstance(body, dict):
            self._json(400, {"error": "body must be a JSON object"})
            return None
        return body

    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args
        if " 200" not in message:
            sys.stderr.write(f"  [engine] {message}\n")

    # -- routes ------------------------------------------------------------ #
    def do_OPTIONS(self) -> None:  # noqa: N802 - the name the base class calls
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?")[0].rstrip("/") or "/"
        if route in ("/health", "/", "/healthz"):
            self._json(200, self.server.engine.info())  # type: ignore[attr-defined]
            return
        self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        engine: StyleEngine = self.server.engine  # type: ignore[attr-defined]
        route = self.path.split("?")[0].rstrip("/") or "/"

        if route == "/reload-reference":
            body = self._read_body()
            if body is None:
                return
            raw = body.get("path") or body.get("reference_wav")
            if not isinstance(raw, str) or not raw.strip():
                self._json(400, {"error": "path must be a non-empty string"})
                return
            answer = engine.set_reference(Path(raw))
            self._json(200 if answer.get("ok") else 400, answer)
            return

        if route != "/synthesize":
            self._json(404, {"error": "unknown endpoint"})
            return

        body = self._read_body()
        if body is None:
            return
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            self._json(400, {"error": "text must be a non-empty string"})
            return
        if not engine.loaded:
            self._json(503, {
                "error": "the voice is not loaded",
                "detail": engine.load_error or "the model is still starting",
            })
            return
        try:
            result = engine.synthesize(text[:MAX_TEXT_CHARS])
        except Exception as error:  # noqa: BLE001 - answer, do not die
            traceback.print_exc()
            self._json(500, {"error": f"{type(error).__name__}: {error}"})
            return

        encoded = base64.b64encode(
            pcm16_to_wav(float_to_pcm16(result["samples"]), result["sample_rate"])
        ).decode("ascii")
        self._json(200, {
            "audio_base64": encoded,
            "sample_rate": result["sample_rate"],
            "seconds": round(result["seconds"], 3),
            "peak": result["peak"],
            "rms": result["rms"],
            "diffusion_steps": result["diffusion_steps"],
            "elapsed_ms": result["elapsed_ms"],
            "reference_path": result["reference_path"],
            "cloned": result["cloned"],
            "text_chars": len(text),
        })


def serve(engine: StyleEngine, host: str = HOST, port: int = PORT) -> int:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.engine = engine  # type: ignore[attr-defined]
    print(f"  listening : http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping", flush=True)
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    import live_transcribe.engine_paths as engine_paths

    parser = argparse.ArgumentParser(
        prog="rixers-speech-engine",
        description="StyleTTS 2 voice engine for Rixers Custom Voice Changer.",
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--checkpoint", default=None, help="StyleTTS 2 .pth (default: the installed one)")
    parser.add_argument("--config", default=None, help="config.yml (default: the installed one)")
    parser.add_argument("--reference", default=None, help="wav to clone (default: voices/default.wav)")
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS,
                        help=f"more steps = more varied and slower (default: {DEFAULT_DIFFUSION_STEPS})")
    parser.add_argument("--cpu", action="store_true", help="skip the GPU on purpose")
    parser.add_argument("--wait", action="store_true",
                        help="load the model before answering /health, instead of in the background")
    parser.add_argument("--check", action="store_true",
                        help="report what is installed and exit (no server)")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint) if args.checkpoint else engine_paths.checkpoint()
    config = Path(args.config) if args.config else engine_paths.config()
    reference = Path(args.reference) if args.reference else engine_paths.default_reference()

    engine = StyleEngine(
        checkpoint=checkpoint,
        config=config,
        reference=reference,
        diffusion_steps=args.diffusion_steps,
        force_cpu=args.cpu,
    )

    if args.check:
        missing = engine.missing_files()
        print(f"  python    : {sys.version.split()[0]}  ({sys.executable})")
        print(f"  torch     : {torch_report()}")
        print(f"  checkpoint: {checkpoint}  {'OK' if checkpoint.is_file() else 'MISSING'}")
        print(f"  config    : {config}  {'OK' if config.is_file() else 'MISSING'}")
        print(f"  reference : {reference}  {'OK' if reference.is_file() else 'MISSING'}")
        return EXIT_MISSING_MODEL if missing else 0

    print("  Rixers Custom Voice Changer - speech engine", flush=True)
    print(f"  python    : {sys.version.split()[0]}", flush=True)
    print(f"  torch     : {torch_report()}", flush=True)
    # Before the load thread exists: see `warm_imports` for why this is not optional.
    warm_imports()
    print(f"  checkpoint: {checkpoint}", flush=True)
    print(f"  reference : {reference if reference.is_file() else '(model default voice)'}",
          flush=True)

    missing = engine.missing_files()
    if missing:
        print("\n  ! the StyleTTS 2 model is not installed:", flush=True)
        for path in missing:
            print(f"      missing: {path}", flush=True)
        print("  ! run  Setup.bat   (or: python setup.py engine)", file=sys.stderr, flush=True)
        return EXIT_MISSING_MODEL

    if args.wait:
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {exc}", file=sys.stderr, flush=True)
            return 1
        print(f"  ready in {engine.load_seconds:.1f}s", flush=True)
    else:
        print("  loading the model (60-90 s is normal) ...", flush=True)
        engine.load_in_background()

    return serve(engine, args.host, args.port)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
