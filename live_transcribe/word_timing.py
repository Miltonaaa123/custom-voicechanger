"""Inter-word delays: measuring them, carrying them, and replaying them.

What this is for
----------------
How long you leave between words is part of how you speak. "hello what is going
on" is not the same rhythm as "hello ... what is going on", and a voice changer
that throws the timing away hands you back a sentence delivered at a pace you
never used. This module records the delay in front of each word and puts it back
when the clone speaks.

How a delay travels
-------------------
The transcriber and the voice changer are separate processes that meet at a text
file, so the delay has to travel *inside the text* or it would be lost the moment
the words were written down. A delay is therefore carried as a mark:

    hello [[70]] what is [[240]] going on

meaning "wait 70 ms before 'what'" and "wait 240 ms before 'going'". The mark
belongs to the word that follows it, which is what makes it survive chunking: a
chunk cut at a token boundary can never strand a mark from its word.

The mark is deliberately not human-friendly. It is only ever in the stream handed
to the speech pipeline, never in ``transcript.txt``, which stays plain text -- the
page sends the words and the marks separately for exactly that reason.

What is replayed, and what it costs
-----------------------------------
Gaps are filtered before they are replayed (see `WordDelaySettings`). A gap has to
be long enough to be worth a synthesis call, because that is the price: a gap is
reproduced by ending one call and starting the next, and each call costs a few
hundred milliseconds on this machine. So the defaults keep the gaps you can hear
and absorb the ones you cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A mark: [[123]] -> 123 ms before the next word. At most four digits, so the
# longest a mark can express is 9999 ms and the pattern and the parser agree
# exactly: anything `strip_marks` removes, `use_mark` accepts, and anything else --
# "[the] [2]", a stray "[[not this]]", a five-digit number -- is ordinary text that
# a mis-typed transcript can still speak aloud.
MARK_RE = re.compile(r"\[\[(\d{1,4})\]\]")

# An unterminated mark at the very end of a piece of text. Text can arrive split
# anywhere, so the tail of a feed may hold half a mark; it is never emitted until
# it is complete.
MARK_PREFIX_RE = re.compile(r"\[\[\d{0,4}$")

# Longest gap that may be replayed. Longer than this is a pause between sentences,
# not rhythm inside one, and reproducing it would just make the clone as slow as
# the original speaker was hesitant.
MAX_GAP_MS = 5000.0

# Gaps at or below this are measurement noise rather than speech. One 60 Hz frame
# is 16.7 ms, so anything under this is not a pause anyone made on purpose.
MIN_MEASURED_MS = 1.0

# A gap this long is the end of a thought rather than rhythm inside one. Replaying
# it would make the clone wait for a silence nobody else is waiting for, so a delay
# measured across it is dropped (see TimingMarkBuilder).
UTTERANCE_GAP_S = 2.0


@dataclass(frozen=True)
class TimingMark:
    """One measured delay, in front of the word that follows it."""

    word: str
    gap_ms: float


@dataclass
class WordDelaySettings:
    """How much of the measured rhythm to actually replay."""

    enabled: bool = False
    """Off means the marks are stripped and ignored: the model keeps its own pacing."""
    min_gap_ms: float = 80.0
    """Gaps shorter than this are absorbed into the words around them.

    This is the latency/rhythm dial. Every gap that is replayed costs a synthesis
    call, so replaying a 30 ms gap between every word in an eight-word sentence
    turns one call into eight and the voice falls seconds behind the speaker.
    """
    max_gap_ms: float = MAX_GAP_MS
    """Longer gaps are capped rather than dropped, so a hesitant speaker does not
    make the clone equally hesitant."""
    use_cap: bool = True
    """When the speech server cannot report a total duration, the cap cannot be
    applied honestly and raw delays are replayed instead."""

    def replay_ms(self, gap_ms: float) -> float:
        """The delay to actually insert, or 0.0 to leave the gap absorbed."""
        if not self.enabled:
            return 0.0
        gap = float(gap_ms)
        if gap != gap or gap < self.min_gap_ms:  # NaN or too short
            return 0.0
        if self.use_cap and gap > self.max_gap_ms:
            return float(self.max_gap_ms)
        return gap


def strip_marks(text: str) -> str:
    """The text without any timing marks, for display or for speaking plainly."""
    return " ".join(MARK_RE.sub(" ", text).split())


def use_mark(token: str) -> int | None:
    """The delay a token carries, or None if the token is not a mark."""
    match = MARK_RE.fullmatch(token)
    return int(match.group(1)) if match else None


def has_marks(text: str) -> bool:
    return MARK_RE.search(text) is not None


def looks_like_mark_prefix(text: str) -> bool:
    """True when text ends in half a mark, which must not be released yet."""
    tail = text[text.rfind("[[") :] if "[[" in text else ""
    return bool(tail) and MARK_PREFIX_RE.fullmatch(tail) is not None


def split_tokens(text: str) -> list[str]:
    """Split text into words and marks.

    Whitespace only: whether the last token is finished is a property of the text
    as a whole (does it end in whitespace?), not of the token, so the caller decides
    with `join_tokens(..., terminated=...)`. Words arrive one at a time and the
    trailing piece is often half a word or half a mark, and keeping the caller in
    charge is what stops it being released early.
    """
    if not text:
        return []
    return [p for p in text.split(" ") if p]


def join_tokens(tokens: list[str], terminated: bool = True) -> str:
    """Rebuild text from tokens.

    The separator between tokens is recovered from their spacing, which is what
    makes the round trip exact: the first token is stripped of its leading space
    and the rest keep theirs. Text assembled from these parts cannot fuse two
    words together, which an earlier design did ("there" + " hello").

    `terminated` says whether the last token is finished. Text that arrives word by
    word ends in a fragment until something follows it, and a fragment ends the
    whole string -- the next feed continues it, so no separator may be added after
    it. That is what turned a split mark "[[120]" into "[[120] ]" and lost the
    delay it carried.
    """
    if not tokens:
        return ""
    text = tokens[0].lstrip()
    for token in tokens[1:]:
        text += token if token[:1].isspace() else " " + token
    return text if terminated else text.rstrip()


def marks_from_tokens(tokens: list[str]) -> list[TimingMark]:
    """Pair each word with the delays in front of it.

    Consecutive marks are summed: two delays before one word mean the same as
    their total.
    """
    marks: list[TimingMark] = []
    pending = 0.0
    for token in tokens:
        gap = use_mark(token)
        if gap is not None:
            pending += gap
            continue
        marks.append(TimingMark(word=token, gap_ms=pending))
        pending = 0.0
    return marks


def marks_to_text(marks: list[TimingMark]) -> str:
    """Text carrying its delays, in the order they were spoken."""
    parts: list[str] = []
    for mark in marks:
        word = mark.word.strip()
        if not word:
            continue
        if mark.gap_ms and mark.gap_ms > 0:
            parts.append(f"[[{int(round(mark.gap_ms))}]]")
        parts.append(word)
    return " ".join(parts)


def split_marks(text: str, settings: WordDelaySettings) -> list[TimingMark]:
    """Break marked text into words with the delays that will be replayed.

    A word whose delay is not worth replaying comes back with a gap of 0 and is
    therefore grouped with its neighbours by `segment_marks`.
    """
    return [
        TimingMark(word=mark.word, gap_ms=settings.replay_ms(mark.gap_ms))
        for mark in marks_from_tokens(split_tokens(text))
    ]


def segment_marks(marks: list[TimingMark]) -> list[TimingMark]:
    """Group words that belong in one synthesis call.

    Only a replayed delay splits a call. Everything else stays joined, which is
    what keeps an ordinary sentence one call -- the model has more context and the
    stream keeps up -- while still honouring the pauses the speaker made.
    """
    out: list[TimingMark] = []
    for mark in marks:
        word = mark.word.strip()
        if not word:
            continue
        if out and not mark.gap_ms:
            out[-1] = TimingMark(word=f"{out[-1].word} {word}", gap_ms=out[-1].gap_ms)
        else:
            out.append(TimingMark(word=word, gap_ms=mark.gap_ms))
    return out


class TimingMarkBuilder:
    """Records how long the speaker left between words, as they are committed.

    Call `record()` with each word and the moment it started, in timeline order.
    Gaps are computed from the timing of the words themselves, not from when the
    recogniser produced them, so the measurement is the speaker's own rhythm rather
    than the service's latency.

    Delays between separate utterances are dropped: a pause after a full stop is
    the end of a thought, not rhythm inside one, and replaying it would make the
    clone wait for a silence the listener is not waiting for either.
    """

    #: A gap this long ends the utterance as far as rhythm is concerned.
    utterance_gap_s: float = UTTERANCE_GAP_S

    #: Clamp on a single measured delay, to survive a stalled stream.
    max_gap_ms: float = 3000.0

    #: Below this a delay is measurement noise, not speech.
    min_gap_ms: float = 15.0

    def __init__(self, utterance_gap_s: float | None = None) -> None:
        if utterance_gap_s is not None:
            self.utterance_gap_s = float(utterance_gap_s)
        self.marks: list[TimingMark] = []
        self._last_end: float | None = None

    def reset(self) -> None:
        self.marks = []
        self._last_end = None

    def record(self, word: str, start_s: float, end_s: float | None = None) -> float:
        """Add one word; returns the delay recorded in front of it (ms)."""
        clean = (word or "").strip()
        if not clean:
            return 0.0
        end = float(end_s) if end_s is not None else float(start_s)

        gap_ms = 0.0
        if self._last_end is not None:
            gap_s = float(start_s) - self._last_end
            if 0.0 <= gap_s <= self.utterance_gap_s:
                gap_ms = max(self.min_gap_ms, min(gap_s * 1000.0, self.max_gap_ms))
        self._last_end = end
        self.marks.append(TimingMark(word=clean, gap_ms=gap_ms))
        return gap_ms

    def text(self, since: int = 0) -> str:
        """Words from `since` onwards, carrying their delays."""
        return marks_to_text(self.marks[since:])

    def as_dicts(self, since: int = 0) -> list[dict]:
        return [
            {"word": m.word, "gap_ms": round(m.gap_ms, 1)} for m in self.marks[since:]
        ]


def describe(marks: list[TimingMark]) -> str:
    """One-line report of what was measured, for the console."""
    if not marks:
        return "no words"
    gaps = [m.gap_ms for m in marks if m.gap_ms > 0]
    if not gaps:
        return f"{len(marks)} words, no gap above {MIN_MEASURED_MS:.0f} ms"
    return (
        f"{len(marks)} words, {len(gaps)} gaps: "
        f"median {sorted(gaps)[len(gaps) // 2]:.0f} ms, "
        f"max {max(gaps):.0f} ms"
    )
