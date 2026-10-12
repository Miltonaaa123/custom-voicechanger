"""The voices folder: drop in a wav, get a voice.

A StyleTTS 2 "voice" is just a reference clip — the model clones the timbre from
it. So managing voices is managing wav files:

* every ``*.wav`` in ``voices/`` is a voice, named by its filename
* the one named ``default.wav`` (or the first alphabetically) is used unless
  ``--voice`` says otherwise
* optional per-voice settings live in ``voices/voices.json``: a spoken transcript
  of the clip, and a human-friendly name

No database, no registration step: copy a file in, it appears.
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
MANIFEST_NAME = "voices.json"

# The server enforces the same bounds; checked here too so the error arrives
# before a slow upload rather than after it.
MIN_SECONDS = 3.0
MAX_SECONDS = 16.0
MAX_BYTES = 6 * 1024 * 1024


@dataclass
class Voice:
    path: Path
    name: str
    seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bytes: int = 0
    transcript: str = ""
    problem: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problem

    @property
    def slug(self) -> str:
        return self.path.stem

    def describe(self) -> str:
        if self.problem:
            return f"{self.slug:<22} UNUSABLE - {self.problem}"
        note = f"{self.seconds:5.1f}s {self.sample_rate} Hz {self.channels}ch"
        extras = f"  tags: {', '.join(self.tags)}" if self.tags else ""
        return f"{self.slug:<22} {note}  {self.name}{extras}"


def _inspect(path: Path) -> tuple[float, int, int, str]:
    """(seconds, sample_rate, channels, problem) for a candidate wav."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        return 0.0, 0, 0, f"{size // (1024 * 1024)} MB is over the {MAX_BYTES // (1024 * 1024)} MB limit"
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
    except wave.Error as exc:
        return 0.0, 0, 0, f"not a readable wav ({exc})"
    if rate <= 0:
        return 0.0, 0, 0, "unreadable sample rate"
    if width not in (2, 3, 4):
        return 0.0, rate, channels, f"{width * 8}-bit audio is not supported"
    seconds = frames / float(rate)
    if seconds < MIN_SECONDS:
        return seconds, rate, channels, (
            f"only {seconds:.1f}s - a clone needs at least {MIN_SECONDS:.0f}s of speech"
        )
    if seconds > MAX_SECONDS:
        return seconds, rate, channels, (
            f"{seconds:.1f}s is over the {MAX_SECONDS:.0f}s limit - trim it to one or two sentences"
        )
    return seconds, rate, channels, ""


def load_manifest(directory: Path) -> dict:
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def discover(directory: Path | None = None) -> list[Voice]:
    """Every wav in the voices folder, ready or not."""
    directory = Path(directory) if directory else VOICES_DIR
    if not directory.is_dir():
        return []
    manifest = load_manifest(directory)

    voices: list[Voice] = []
    for path in sorted(directory.glob("*.wav")):
        entry = manifest.get(path.name, {}) if isinstance(manifest, dict) else {}
        if not isinstance(entry, dict):
            entry = {}
        seconds, rate, channels, problem = _inspect(path)
        voices.append(
            Voice(
                path=path,
                name=str(entry.get("name") or path.stem.replace("_", " ").title()),
                seconds=seconds,
                sample_rate=rate,
                channels=channels,
                bytes=path.stat().st_size,
                transcript=str(entry.get("transcript") or ""),
                tags=[str(t) for t in entry.get("tags", [])] if isinstance(entry.get("tags"), list) else [],
                problem=problem,
            )
        )
    return voices


def find(selector: str | None, directory: Path | None = None) -> Voice:
    """Resolve a voice by filename stem, filename, or displayed name.

    With no selector, ``default.wav`` wins, otherwise the first usable voice.
    """
    directory = Path(directory) if directory else VOICES_DIR
    voices = discover(directory)
    if not voices:
        raise FileNotFoundError(
            f"no voices found in {directory}.\n"
            "Drop a 3-16 second wav of the voice you want into that folder."
        )

    if selector:
        wanted = selector.strip().lower()
        for voice in voices:
            if wanted in (voice.slug.lower(), voice.path.name.lower(), voice.name.lower()):
                if not voice.usable:
                    raise ValueError(f"voice {voice.slug!r} cannot be used: {voice.problem}")
                return voice
        available = ", ".join(v.slug for v in voices)
        raise KeyError(f"no voice called {selector!r}. Available: {available}")

    for voice in voices:
        if voice.slug.lower() == "default" and voice.usable:
            return voice
    for voice in voices:
        if voice.usable:
            return voice
    raise ValueError(
        "no usable voice in "
        f"{directory}: " + "; ".join(f"{v.slug} ({v.problem})" for v in voices)
    )


def write_manifest(directory: Path | None = None) -> Path:
    """Create a starter manifest listing every voice, without clobbering edits."""
    directory = Path(directory) if directory else VOICES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    existing = load_manifest(directory)

    for voice in discover(directory):
        entry = existing.get(voice.path.name)
        if isinstance(entry, dict):
            continue
        existing[voice.path.name] = {
            "name": voice.name,
            "transcript": voice.transcript,
            "tags": voice.tags,
        }
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return path
