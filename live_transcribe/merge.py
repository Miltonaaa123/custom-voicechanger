"""Text assembly: turning timestamped ASR words into a clean, non-duplicated stream.

The transcriber deliberately re-reads a little audio it has already seen (a
"lead-in") so Whisper has acoustic context at chunk boundaries. Word-level
timestamps are what make that safe: a word is committed exactly once, decided by
whether its end time lies past the commit cursor. No text-level overlap guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that carry no information for overlap comparison.
_FILLER = {"", " ", "\n"}


@dataclass
class Word:
    """One recognized word with its position in the stream timeline."""

    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def clean(self) -> str:
        return self.text.strip()


def normalise(word: str) -> str:
    """Comparison key for a word: lowercase letters and digits only."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def join_words(words: list[str]) -> str:
    """Join word fragments the way Whisper intends them to be rendered."""
    out = ""
    for w in words:
        if not w:
            continue
        if not out:
            out = w.lstrip()
        elif w.startswith(" ") or out.endswith((" ", "\n")):
            out += w
        elif w[0] in ",.!?;:%)]}'\"":
            out += w
        else:
            out += " " + w
    return out


@dataclass
class CommitCursor:
    """Tracks which point in the audio timeline has been turned into text."""

    position: float = 0.0
    """Timeline seconds up to which words have been committed."""
    committed: list[Word] = field(default_factory=list)
    text: str = ""
    """Incrementally maintained join of every committed word (never O(n^2))."""

    def add(self, words: list[Word]) -> None:
        if not words:
            return
        self.committed.extend(words)
        self.text = join_words([self.text, *[w.text for w in words]]) if self.text else join_words(
            [w.text for w in words]
        )
        self.position = max(self.position, words[-1].end)

    def reset(self, position: float = 0.0) -> None:
        self.committed.clear()
        self.text = ""
        self.position = position

    def last_word_end(self) -> float:
        return self.committed[-1].end if self.committed else 0.0

    def tail_words(self, n: int) -> str:
        """Last `n` committed words, used as the next decode's prompt."""
        return " ".join(w.clean for w in self.committed[-n:] if w.clean)


def uncommitted_text(words: list[Word], cursor: CommitCursor) -> str:
    """Text of `words` that lies beyond the commit cursor (the live guess)."""
    tail = [w.text for w in words if w.clean and w.end > cursor.position + 1e-3]
    return join_words(tail).strip()


def is_repetition_loop(word: Word, cursor: CommitCursor, window_s: float = 1.0) -> bool:
    """True if this word continues a runaway repetition.

    Fed laughter, music or loud noise (or hallucinating over silence), Whisper can
    lock into repeating one token ("oh, oh, oh, oh...") and never stop. Left alone
    this floods the transcript, and it is the single worst failure mode for a
    text-to-speech handoff.

    Two earlier occurrences of the same word inside a short span mark the start of
    a loop; everything after that is dropped until the model returns to real
    speech. Two, not one, so that emphasis ("no, no, no!") still gets through.
    """
    key = normalise(word.text)
    if not key:
        return False
    recent = [w for w in reversed(cursor.committed) if w.end >= word.start - window_s]
    same = sum(1 for w in recent if normalise(w.text) == key)
    return same >= 2


def is_reemission(word: Word, cursor: CommitCursor, window_s: float = 0.6) -> bool:
    """True if this word looks like a duplicate of one just committed.

    When the word straddling the commit cursor is re-decoded, the model emits it
    again with a slightly shifted timestamp. No timing rule separates that from
    genuine new speech, but the text does: an identical word landing within a few
    tenths of a second of itself is a repeat of the same audio, not new speech.

    ``window_s`` is deliberately tight. Real immediate repeats ("very very") sit
    further apart than this, so they survive.
    """
    key = normalise(word.text)
    if not key:
        return False
    for prev in reversed(cursor.committed):
        if prev.end < word.start - window_s:
            break
        if normalise(prev.text) == key:
            return True
    return False


def select_new_words(
    words: list[Word],
    cursor: CommitCursor,
    window_end: float,
    edge_guard_s: float,
    allow_edge_words: bool,
    overlap_guard_s: float = 0.0,
    min_probability: float = 0.0,
    reemission_window_s: float = 0.6,
    repetition_window_s: float = 1.2,
) -> list[Word]:
    """Pick the words from a decode that are new and safe to commit.

    Args:
        words: word timestamps for the decoded window, timeline-absolute.
        cursor: current commit position.
        window_end: timeline end of the audio the model actually saw.
        edge_guard_s: keep this much audio un-committed at the window edge so a
            word truncated by the chunk boundary is re-read next round.
        allow_edge_words: set when the speaker has paused, meaning the audio
            after the window is silence and nothing can be truncated.
        overlap_guard_s: a word must end at least this far past the cursor to be
            committed. Catches the common case of the cursor sitting inside the
            boundary word.
        min_probability: optional confidence floor. Kept loose by default --
            genuine words at a decode boundary are also low-confidence, so a
            strict floor loses real speech. Hallucinations are handled by the
            energy gate on flushes instead.
        reemission_window_s: how close an identical word must be to count as a
            repeat of the same audio.

    Returns:
        New words in order, contiguous after the cursor.
    """
    limit = window_end if allow_edge_words else max(0.0, window_end - edge_guard_s)
    threshold = cursor.position + max(0.0, overlap_guard_s)
    out: list[Word] = []
    for w in words:
        if not w.clean:
            continue
        if min_probability > 0 and w.probability < min_probability:
            continue
        # Words wholly in the past are already committed.
        if w.end <= threshold:
            continue
        if w.start >= limit:
            break
        if w.end > limit:
            # Straddles the guard: leave it for the next, better-context decode.
            break
        if is_reemission(w, cursor, reemission_window_s):
            continue
        if repetition_window_s > 0 and is_repetition_loop(w, cursor, repetition_window_s):
            continue
        out.append(w)
    return out


def cap_committed(cursor: CommitCursor, max_words: int = 400) -> None:
    """Bound memory on very long dictation sessions.

    Only the word list is trimmed: ``cursor.text`` holds the full history, which
    is what gets written to the transcript file.
    """
    if len(cursor.committed) > max_words:
        del cursor.committed[: len(cursor.committed) - max_words]


def similarity(a: list[str], b: list[str]) -> float:
    """Fraction of identical words between two sequences (debug helper)."""
    if not a or not b:
        return 0.0
    hits = sum(1 for x, y in zip(a, b) if normalise(x) == normalise(y))
    return hits / max(len(a), len(b))
