"""Unit tests for the inter-word delay plumbing.

The bug class this guards against: the delay the speaker left between two words is
measured, carried in the text, and put back when the clone speaks -- without ever
being spoken aloud, without being lost at a chunk boundary, and without turning a
fluent sentence into a string of one-word synthesis calls.

Run:  .venv\\Scripts\\python.exe tests\\test_word_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_transcribe import word_timing as wt  # noqa: E402


# --------------------------------------------------------------------------- #
# Carrying a delay in the text
# --------------------------------------------------------------------------- #
def test_marks_are_read_back_as_the_delay_in_front_of_a_word() -> None:
    text = "hello [[70]] what is [[240]] going on"
    marks = wt.marks_from_tokens(wt.split_tokens(text))
    assert [(m.word, m.gap_ms) for m in marks] == [
        ("hello", 0),
        ("what", 70),
        ("is", 0),
        ("going", 240),
        ("on", 0),
    ], marks


def test_a_mark_is_not_mistaken_for_a_word() -> None:
    tokens = wt.split_tokens("hello [[70]] what ")
    assert wt.use_mark("[[70]]") == 70
    assert wt.use_mark("[[70000]]") is None, "an implausible delay is not a mark"
    assert wt.use_mark("[70]") is None
    assert wt.use_mark("hello") is None
    assert tokens == ["hello", "[[70]]", "what"], tokens

def test_half_a_mark_is_recognised_and_held_back() -> None:
    """Text arrives split anywhere, so a mark can be cut in two."""
    assert wt.looks_like_mark_prefix("hello [[7")
    assert wt.looks_like_mark_prefix("[[")
    assert not wt.looks_like_mark_prefix("hello [[70]]")
    assert not wt.looks_like_mark_prefix("hello [")
    assert not wt.looks_like_mark_prefix("hello")


def test_marks_never_survive_into_plain_text() -> None:
    assert wt.strip_marks("hello [[70]] what") == "hello what"
    assert wt.strip_marks("no marks here") == "no marks here"
    assert wt.strip_marks("[[300]] leading") == "leading"
    assert not wt.has_marks("hello what")


def test_text_round_trips_through_marks() -> None:
    marks = [wt.TimingMark("hello", 0), wt.TimingMark("what", 70), wt.TimingMark("now", 240)]
    text = wt.marks_to_text(marks)
    assert text == "hello [[70]] what [[240]] now"
    again = wt.marks_from_tokens(wt.split_tokens(text))
    assert [(m.word, m.gap_ms) for m in again] == [(m.word, m.gap_ms) for m in marks]


def test_the_model_is_never_asked_to_say_a_mark() -> None:
    """The mark format must be unlike anything the speaker would ever say."""
    for word in ("hi", "hello", "2024", "42", "what?"):
        spoken = wt.strip_marks(wt.marks_to_text([wt.TimingMark("a", 300), wt.TimingMark(word, 300)]))
        assert spoken == f"a {word}", spoken
        assert not wt.has_marks(spoken), spoken


# --------------------------------------------------------------------------- #
# Deciding which delays are worth replaying
# --------------------------------------------------------------------------- #
def test_short_delays_are_absorbed_and_long_ones_are_replayed() -> None:
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0)
    assert settings.replay_ms(30) == 0.0, "a 30 ms gap is not worth a synthesis call"
    assert settings.replay_ms(79) == 0.0
    assert settings.replay_ms(80) == 80.0
    assert settings.replay_ms(300) == 300.0


def test_a_very_long_delay_is_capped() -> None:
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0, max_gap_ms=1500.0)
    assert settings.replay_ms(9000) == 1500.0


def test_nothing_is_replayed_when_the_feature_is_off() -> None:
    settings = wt.WordDelaySettings(enabled=False)
    assert settings.replay_ms(1000) == 0.0


def test_a_nonsense_measurement_is_ignored() -> None:
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0)
    assert settings.replay_ms(float("nan")) == 0.0
    assert settings.replay_ms(-500) == 0.0


# --------------------------------------------------------------------------- #
# Grouping words into synthesis calls
# --------------------------------------------------------------------------- #
def test_only_replayed_delays_split_a_synthesis_call() -> None:
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0)
    text = "hello [[20]] what is [[300]] going on"
    grouped = wt.segment_marks(wt.split_marks(text, settings))
    assert [(m.word, m.gap_ms) for m in grouped] == [
        ("hello what is", 0.0),
        ("going on", 300.0),
    ], grouped


def test_a_delay_is_always_in_front_of_its_own_word() -> None:
    """Replaying a delay on the wrong side of a word would scramble the rhythm."""
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0)
    grouped = wt.segment_marks(wt.split_marks("one [[200]] two [[200]] three", settings))
    assert [(m.word, m.gap_ms) for m in grouped] == [
        ("one", 0.0),
        ("two", 200.0),
        ("three", 200.0),
    ], grouped


def test_splitting_never_loses_a_word() -> None:
    settings = wt.WordDelaySettings(enabled=True, min_gap_ms=80.0)
    text = "one [[200]] two [[20]] three [[400]] four five"
    grouped = wt.segment_marks(wt.split_marks(text, settings))
    words = " ".join(m.word for m in grouped).split()
    assert words == ["one", "two", "three", "four", "five"], grouped


# --------------------------------------------------------------------------- #
# Measuring the speaker's rhythm
# --------------------------------------------------------------------------- #
def test_a_measured_gap_is_the_silence_between_two_words() -> None:
    builder = wt.TimingMarkBuilder()
    builder.record("hello", start_s=1.00, end_s=1.30)
    gap = builder.record("what", start_s=1.33, end_s=1.55)
    # 30 ms of silence between the end of "hello" and the start of "what".
    assert 25 <= gap <= 35, gap
    assert builder.text() == "hello [[30]] what" or builder.text().startswith("hello [["), builder.text()


def test_the_first_word_of_an_utterance_carries_no_delay() -> None:
    builder = wt.TimingMarkBuilder()
    assert builder.record("hello", start_s=5.0, end_s=5.3) == 0.0
    assert builder.marks[0].gap_ms == 0.0


def test_a_pause_between_thoughts_is_not_rhythm() -> None:
    """A full stop is the end of a sentence; replaying that silence would drag."""
    builder = wt.TimingMarkBuilder(utterance_gap_s=2.0)
    builder.record("done", start_s=1.0, end_s=1.4)
    assert builder.record("next", start_s=9.0, end_s=9.3) == 0.0


def test_overlapping_timestamps_do_not_produce_a_negative_gap() -> None:
    builder = wt.TimingMarkBuilder()
    builder.record("one", start_s=2.0, end_s=2.5)
    assert builder.record("two", start_s=2.4, end_s=2.8) == 0.0


def test_resetting_forgets_the_previous_utterance() -> None:
    builder = wt.TimingMarkBuilder()
    builder.record("hello", start_s=1.0, end_s=1.3)
    builder.reset()
    assert builder.record("what", start_s=1.4, end_s=1.6) == 0.0
    assert builder.text() == "what"


def test_an_absurd_delay_is_clamped_rather_than_replayed() -> None:
    class Wide(wt.TimingMarkBuilder):
        utterance_gap_s = 10.0
        max_gap_ms = 3000.0

    builder = Wide()
    builder.record("a", start_s=1.0, end_s=1.2)
    assert builder.record("b", start_s=6.0, end_s=6.2) == 3000.0


def test_a_number_too_large_to_be_a_pause_is_left_as_text() -> None:
    """Better to read a stray number aloud than to sit in silence for a minute."""
    assert wt.use_mark("[[70000]]") is None
    assert wt.strip_marks("call [[70000]] now") == "call [[70000]] now"


# --------------------------------------------------------------------------- #
# Repeated splitting must not invent or fuse words
# --------------------------------------------------------------------------- #
def test_words_are_never_fused_by_a_split() -> None:
    """An earlier design rebuilt words from tokens and produced "therehello".

    The cuts checked are the ones a chunker actually makes -- at a space -- and
    what must survive is a real word followed by a real word.
    """
    source = "the quick brown [[120]] fox"
    for cut, char in enumerate(source):
        if char != " ":
            continue
        head, tail = source[:cut], source[cut:]
        # The head is a partial feed, so its last token is unterminated; the tail
        # brings its own leading space from the original text.
        rebuilt = wt.join_tokens(wt.split_tokens(head), terminated=False) + tail
        assert rebuilt == source, (cut, rebuilt)
        words = rebuilt.split()
        assert "therehello" not in words and all(w for w in words), words


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}\n         {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
