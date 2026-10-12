"""Writing the transcript files.

Three outputs, each with a different job:

* ``transcript.txt``  - exactly what has been said, nothing else. This is the
  file to hand to StyleTTS 2; it is rewritten only when the text really changes.
* ``live.txt``        - the same text plus a dimmed in-flight hypothesis, for a
  human watching the words appear.
* ``events.jsonl``    - append-only timing/telemetry, survives restarts.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

LIVE_MARKER = " \u2039 "  # ' ‹ ' separates settled text from the live guess


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class TranscriptPaths:
    clean: Path
    live: Path
    events: Path


class TranscriptWriter:
    """Keeps the on-disk transcript in sync with the current committed text."""

    def __init__(
        self,
        clean_path: str | Path,
        live_path: str | Path | None = None,
        events_path: str | Path | None = None,
        clear_on_start: bool = True,
        newline_per_utterance: bool = True,
    ) -> None:
        clean = Path(clean_path).expanduser().resolve()
        self.paths = TranscriptPaths(
            clean=clean,
            live=Path(live_path).expanduser().resolve() if live_path else clean.with_name("live.txt"),
            events=Path(events_path).expanduser().resolve() if events_path else clean.with_name("events.jsonl"),
        )
        self.newline_per_utterance = newline_per_utterance
        self._lock = threading.Lock()
        self._last_clean: str | None = None
        self._last_live: str | None = None
        self.existing_text = ""
        """Transcript text already on disk when running with clear_on_start=False."""
        self._start = time.time()
        self.write_count = 0

        for p in (self.paths.clean, self.paths.live, self.paths.events):
            p.parent.mkdir(parents=True, exist_ok=True)

        if clear_on_start:
            # Requirement: restarting the program clears the transcript.
            _atomic_write(self.paths.clean, "")
            _atomic_write(self.paths.live, "")
            self._last_clean = ""
            self._last_live = ""
        else:
            # Append mode: adopt whatever is already on disk, otherwise the first
            # publish would treat it as a change and overwrite the old text.
            self._last_clean = self._read_existing(self.paths.clean)
            self._last_live = self._read_existing(self.paths.live)
            if self._last_clean:
                self.existing_text = self._last_clean

    def _read_existing(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError:
            return ""

    # -- public API -------------------------------------------------------- #
    def _with_prefix(self, text: str) -> str:
        """Prepend text already on disk when running in append mode."""
        if not self.existing_text or not text:
            return text
        if self.existing_text[-1].isspace() or text[:1].isspace():
            return self.existing_text + text
        return f"{self.existing_text} {text}"

    def set_committed(self, text: str, hypothesis: str = "", live_text: str | None = None) -> None:
        """Update the settled transcript (and optionally the live guess).

        ``live_text`` is the same words carrying the inter-word delays the speaker
        made, for the file the voice changer reads. It is deliberately a separate
        argument rather than extra text: a person reads ``transcript.txt`` and must
        not be shown "[[70]]", while the speech stage needs the delays to put the
        speaker's own rhythm back. The hypothesis goes in the live file only, so
        the timing marks -- which belong to words already committed -- are placed
        before it rather than mixed into the unfinished guess.
        """
        # In append mode an empty transcription must leave the file untouched:
        # collapsing to the prefix here would make the "no change" check below
        # compare the prefix against itself and could blank a fresh file.
        if self.existing_text and text:
            text = self._with_prefix(text)
            if live_text is not None:
                live_text = self._with_prefix(live_text)
        with self._lock:
            if text != self._last_clean:
                _atomic_write(self.paths.clean, text)
                self._last_clean = text
                self.write_count += 1
            live = live_text if live_text is not None else text
            if hypothesis:
                live = f"{live}{LIVE_MARKER}{hypothesis}" if live else hypothesis
            if live != self._last_live:
                _atomic_write(self.paths.live, live)
                self._last_live = live

    def event(self, kind: str, **fields: object) -> None:
        """Append one telemetry record."""
        record = {"t": round(time.time() - self._start, 3), "wall": time.strftime("%H:%M:%S"), "kind": kind}
        record.update(fields)
        try:
            with self._lock, self.paths.events.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def describe(self) -> str:
        return str(self.paths.clean)
