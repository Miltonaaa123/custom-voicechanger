"""Emotion -> voice mapping, so a happy delivery can use a happy reference.

StyleTTS 2 clones whatever reference wav it is given, and how the reference sounds
is by far the strongest control over how the output sounds: measured on this
machine, swapping the reference changes pitch and energy by far more than any of
the API's numeric knobs (``embedding_scale``, ``alpha``, ``beta`` moved energy by
under 10% -- see ``tools/style_probe.py``).

So emotion transfer works by voice selection:

* ``voices/emotions/<emotion>.wav`` is used when it exists, and
* ``voices/emotions.json`` records which voice the user chose for each emotion.

There is no emotion *tag* in this checkpoint -- the server has no tag parsing and
the model conditions only on reference audio and scalars -- so rather than pretend
a ``[very happy]`` token is understood, the detection picks a different reference.

Nothing here requires an emotion file to exist. With none present the mapping simply
reports that it has nothing to offer, and the voice changer carries on unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .emotion import EMOTIONS
from .voices import Voice, discover

EMOTION_DIR_NAME = "emotions"
MAPPING_NAME = "emotions.json"


def emotion_dir(voices_dir: str | Path) -> Path:
    return Path(voices_dir) / EMOTION_DIR_NAME


def mapping_path(voices_dir: str | Path) -> Path:
    return Path(voices_dir) / MAPPING_NAME


def slug_for(emotion: str) -> str:
    """Filename-safe form of an emotion: "very happy" -> "very_happy"."""
    return "_".join(str(emotion).strip().lower().replace("-", " ").split())


@dataclass
class EmotionVoices:
    """Which reference to use for each emotion, and where it came from."""

    voices_dir: Path
    by_emotion: dict[str, Path]
    source: dict[str, str]
    """emotion -> "file" | "mapping" | "default"."""

    def get(self, emotion: str) -> Path | None:
        """The reference for an emotion, falling back to its arousal family.

        A missing ``very happy`` falls back to ``happy`` before giving up: the two
        differ only in intensity, so a voice recorded for one is a sensible stand-in
        for the other. Falling back keeps the feature usable with a single file
        rather than demanding the full set.
        """
        direct = self.by_emotion.get(emotion)
        if direct is not None:
            return direct
        for fallback in _FALLBACKS.get(emotion, ()):
            found = self.by_emotion.get(fallback)
            if found is not None:
                return found
        return None

    @property
    def emotions_covered(self) -> list[str]:
        return sorted(self.by_emotion)

    def describe(self) -> str:
        if not self.by_emotion:
            return "no emotion voices (add wavs to voices\\emotions\\)"
        parts = [f"{name} <- {path.name}" for name, path in sorted(self.by_emotion.items())]
        return ", ".join(parts)


# "very happy" is just a stronger "happy", so each extreme reuses its neighbour.
_FALLBACKS: dict[str, tuple[str, ...]] = {
    "very happy": ("happy", "excited"),
    "happy": ("excited", "very happy"),
    "excited": ("happy", "very happy"),
    "calm": ("neutral", "tired"),
    "neutral": ("calm",),
    "tired": ("sad", "calm"),
    "sad": ("tired",),
    "angry": ("excited",),
}


def load(voices_dir: str | Path) -> EmotionVoices:
    """Find the per-emotion references available in `voices_dir`.

    Two sources, checked in this order:

    1. an explicit entry in ``emotions.json`` (so a voice elsewhere on disk, or a
       differently named clip, can be assigned to an emotion), then
    2. a wav in ``voices/emotions/`` whose filename matches the emotion.
    """
    voices_dir = Path(voices_dir)
    found: dict[str, Path] = {}
    source: dict[str, str] = {}

    # 1. the mapping file
    path = mapping_path(voices_dir)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for emotion, value in raw.items():
                key = str(emotion).strip().lower()
                if key not in EMOTIONS:
                    continue
                candidate = Path(str(value)).expanduser()
                if not candidate.is_absolute():
                    candidate = voices_dir / candidate
                if candidate.is_file():
                    found[key] = candidate
                    source[key] = "mapping"

    # 2. wavs named after the emotion
    directory = emotion_dir(voices_dir)
    if directory.is_dir():
        for voice in discover(directory):
            if not voice.usable:
                continue
            key = voice.slug.replace("_", " ").lower()
            if key in EMOTIONS and key not in found:
                found[key] = voice.path
                source[key] = "file"

    # 3. a whole subfolder per emotion: voices/emotions/happy/*.wav
    if directory.is_dir():
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            key = child.name.replace("_", " ").lower()
            if key not in EMOTIONS or key in found:
                continue
            candidates = [v for v in discover(child) if v.usable]
            if candidates:
                found[key] = candidates[0].path
                source[key] = "file"

    return EmotionVoices(voices_dir=voices_dir, by_emotion=found, source=source)


def write_mapping(voices_dir: str | Path, mapping: dict[str, str]) -> Path:
    """Record emotion -> voice choices, relative to the voices folder where possible."""
    voices_dir = Path(voices_dir)
    stored: dict[str, str] = {}
    for emotion, value in mapping.items():
        candidate = Path(str(value)).expanduser()
        try:
            stored[emotion] = str(candidate.relative_to(voices_dir)).replace("\\", "/")
        except ValueError:
            stored[emotion] = str(candidate)
    path = mapping_path(voices_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
