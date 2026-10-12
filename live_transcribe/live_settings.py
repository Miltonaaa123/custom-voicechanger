"""One settings file shared by the control panel and the running voice changer.

The control panel (``rixers_gui.py``) and the voice changer (``voice_changer.py``)
are separate processes: one draws windows, the other owns the audio. They meet at
``out/rixers_settings.json``, the same way the transcriber and the voice changer
meet at ``out/transcript.txt``.

Two things make that work without either process knowing about the other:

* The file is written atomically (temp file, then replace), so a reader never sees
  half a document.
* The reader only re-parses it when its modification time changes, so polling costs
  nothing while nobody is editing.

Every setting a running process can honour without being restarted is listed in
`FIELDS` with its type, bounds and default. Anything outside those bounds -- a
hand-edited file, a stale version, a typo -- is clamped or ignored rather than
allowed to reach the audio, and `coerce` is the single place that decision is made.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
DEFAULT_OUT = PROJECT_DIR / "out"
SETTINGS_NAME = "rixers_settings.json"

#: Live values, with the bounds that keep a bad file from breaking the audio.
#: ``kind`` also tells the control panel which widget to draw.
FIELDS: dict[str, dict[str, Any]] = {
    # -- voice -----------------------------------------------------------------
    "voice": {"kind": "text", "default": ""},
    # -- sentence timing -------------------------------------------------------
    "pause_flush_s": {"kind": "float", "default": 0.7, "min": 0.1, "max": 10.0},
    "min_chars": {"kind": "int", "default": 1, "min": 1, "max": 200},
    # -- your own rhythm -------------------------------------------------------
    "word_delay": {"kind": "choice", "default": "auto", "choices": ["auto", "on", "off"]},
    "word_gap_min_ms": {"kind": "float", "default": 80.0, "min": 0.0, "max": 5000.0},
    "word_gap_max_ms": {"kind": "float", "default": 5000.0, "min": 0.0, "max": 5000.0},
    # -- how much text per synthesis call --------------------------------------
    "mode": {"kind": "choice", "default": "sentence",
             "choices": ["word", "chunk", "phrase", "sentence"]},
    "target_words": {"kind": "int", "default": 0, "min": 0, "max": 200},
    "max_words": {"kind": "int", "default": 0, "min": 0, "max": 400},
    # -- keeping up ------------------------------------------------------------
    "queue": {"kind": "int", "default": 12, "min": 1, "max": 64},
    "backlog": {"kind": "int", "default": 16, "min": 1, "max": 128},
    "gap_ms": {"kind": "float", "default": 0.0, "min": 0.0, "max": 2000.0},
    "trim_silence_ms": {"kind": "float", "default": 220.0, "min": 0.0, "max": 1000.0},
    "retries": {"kind": "int", "default": 1, "min": 0, "max": 5},
    # -- the sound of the output ----------------------------------------------
    "monitor": {"kind": "bool", "default": False},
    "pitch_semitones": {"kind": "float", "default": 0.0, "min": -12.0, "max": 12.0},
    "volume": {"kind": "float", "default": 1.0, "min": 0.0, "max": 4.0},
    "gate_db": {"kind": "float", "default": -70.0, "min": -70.0, "max": 0.0},
    "gate_floor_db": {"kind": "float", "default": -60.0, "min": -70.0, "max": 0.0},
    # The control panel's own view of the gate, kept in step with `gate_db` so the
    # switch means the same thing when the file is read back.
    "gate_on": {"kind": "bool", "default": False},
    # Which playback device to write into. Empty means the CABLE Input pair.
    "device": {"kind": "text", "default": ""},
    # -- stage 1: your microphone -> text --------------------------------------
    "stt_engine": {"kind": "choice", "default": "whisper",
                   "choices": ["whisper", "google"]},
    "stt_start": {"kind": "bool", "default": False},    # -- spoken engine arguments, edited only in the Advanced section ----------
    "tts_args": {"kind": "text", "default": ""},
    "stt_args": {"kind": "text", "default": ""},
}

#: Settings the running voice changer can change from one sentence to the next.
LIVE_FIELDS: tuple[str, ...] = (
    "voice",
    "pause_flush_s",
    "min_chars",
    "word_delay",
    "word_gap_min_ms",
    "word_gap_max_ms",
    "mode",
    "target_words",
    "max_words",
    "queue",
    "backlog",
    "gap_ms",
    "trim_silence_ms",
    "retries",
    "pitch_semitones",
    "volume",
    "gate_db",
    "gate_floor_db",
)

#: Settings the transcription stage owns. They are applied when it next starts,
#: because the recogniser is a separate process from the voice changer.
STT_FIELDS: tuple[str, ...] = ("stt_engine", "stt_start", "stt_args")

#: Written by the control panel, obeyed by the voice changer at start-up.
COMMAND_FIELDS: tuple[str, ...] = (
    "voice",
    "monitor",
    "mode",
    "target_words",
    "max_words",
    "queue",
    "backlog",
    "gap_ms",
    "trim_silence_ms",
    "retries",
    "pause_flush_s",
    "min_chars",
    "word_delay",
    "word_gap_min_ms",
    "word_gap_max_ms",
    "pitch_semitones",
    "volume",
    "gate_db",
    "gate_floor_db",
)


def settings_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Where the shared settings file lives."""
    if path:
        return Path(path).expanduser()
    return DEFAULT_OUT / SETTINGS_NAME


def defaults() -> dict[str, Any]:
    return {name: spec["default"] for name, spec in FIELDS.items()}


def coerce(name: str, value: Any) -> Any:
    """Force one value into the shape and range its field declares.

    Returns the field's default when the value cannot be used at all, so a
    corrupted file degrades to a working voice changer rather than a crash.
    """
    spec = FIELDS.get(name)
    if spec is None:
        return value
    default = spec["default"]
    kind = spec["kind"]
    try:
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if kind == "int":
            return max(int(spec["min"]), min(int(spec["max"]), int(round(float(value)))))
        if kind == "float":
            return max(float(spec["min"]), min(float(spec["max"]), float(value)))
        if kind == "choice":
            text = str(value)
            return text if text in spec["choices"] else default
        if kind == "text":
            return str(value)
    except (TypeError, ValueError):
        return default
    return value


def sanitise(values: dict[str, Any]) -> dict[str, Any]:
    """Every known field, coerced, with defaults filled in for anything missing."""
    clean = defaults()
    for name in FIELDS:
        if name in values:
            clean[name] = coerce(name, values[name])
    # A window that cannot exist is worse than an ignored setting.
    if clean["word_gap_max_ms"] < clean["word_gap_min_ms"]:
        clean["word_gap_max_ms"] = clean["word_gap_min_ms"]
    if clean["gate_floor_db"] > clean["gate_db"]:
        clean["gate_floor_db"] = clean["gate_db"]
    return clean


class LiveSettings:
    """The settings file, read whenever it changes and written atomically."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = settings_path(path)
        self.values: dict[str, Any] = defaults()
        self._stamp: float | None = None
        self._size: int | None = None

    # -- reading ----------------------------------------------------------- #
    def load(self, force: bool = False) -> bool:
        """Read the file if it changed. True when new values were taken on."""
        try:
            stat = self.path.stat()
        except OSError:
            return False
        if not force and stat.st_mtime == self._stamp and stat.st_size == self._size:
            return False
        self._stamp = stat.st_mtime
        self._size = stat.st_size
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            # A half-written file is not worth reporting: the writer retries.
            return False
        if not isinstance(raw, dict):
            return False
        self.values = sanitise(raw)
        return True

    def get(self, name: str, fallback: Any = None) -> Any:
        if name in self.values:
            return self.values[name]
        return FIELDS.get(name, {}).get("default", fallback)

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    # -- writing ----------------------------------------------------------- #
    def update(self, **changes: Any) -> dict[str, Any]:
        self.values = sanitise({**self.values, **changes})
        return self.values

    def save(self) -> Path:
        """Write the file so a reader sees either the old one or the new one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(sanitise(self.values), indent=2, sort_keys=True) + "\n"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)
        try:
            stat = self.path.stat()
            self._stamp, self._size = stat.st_mtime, stat.st_size
        except OSError:
            self._stamp = self._size = None
        return self.path


class SettingsWatcher:
    """Polls `LiveSettings` and calls back with what actually changed.

    The callback runs on whatever thread calls :meth:`poll` -- the voice changer's
    own watch loop -- so it must be quick. It only ever fires for values a running
    process can honour, and only when they differ from the last time it fired.
    """

    def __init__(
        self,
        settings: LiveSettings,
        on_change,
        names: tuple[str, ...] = LIVE_FIELDS,
        interval_s: float = 0.5,
    ) -> None:
        self.settings = settings
        self.on_change = on_change
        self.names = tuple(names)
        self.interval_s = float(interval_s)
        self._last_checked = 0.0
        self._applied: dict[str, Any] = {name: settings.get(name) for name in self.names}

    def poll(self, force: bool = False) -> list[str]:
        """Check the file now (rate-limited); returns the names that changed."""
        now = time.time()
        if not force and now - self._last_checked < self.interval_s:
            return []
        self._last_checked = now
        if not self.settings.load():
            return []
        changed = [
            name for name in self.names
            if self.settings.get(name) != self._applied.get(name)
        ]
        if not changed:
            return []
        for name in changed:
            self._applied[name] = self.settings.get(name)
        try:
            self.on_change(changed, self.settings)
        except Exception as exc:  # noqa: BLE001 - a bad setting must not stop speech
            print(f"  ! could not apply the changed settings: {exc}", flush=True)
        return changed
