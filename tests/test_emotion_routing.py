"""End-to-end tests for emotion-driven voice switching, without the GPU.

`test_voice_integration.py` proves the real thing works against the live server, but
it needs the model loaded and takes minutes. These tests cover the same wiring
against stand-ins, so the part most likely to break -- a chunk being synthesized in
the wrong voice -- fails fast and deterministically.

The mechanism being checked: a per-chunk reference resolver supplies a voice, the
pipeline switches the server to it, and it switches *only* when the voice actually
changes, because `set_voice` is a round trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe.pipeline import PipelineConfig, SpeechPipeline  # noqa: E402
from live_transcribe.tts import Speech  # noqa: E402


class FakeClient:
    """Records every synthesise call and every voice switch."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.voices_set: list[str] = []
        self.per_call_reference: list[str | None] = []
        self._reference: str | None = None

    def synthesize(self, text: str) -> Speech:
        self.seen.append(text)
        self.per_call_reference.append(self._reference)
        return Speech(
            samples=np.zeros(2400, dtype="<i2").tobytes(),
            sample_rate=24000,
            seconds=0.1,
            text=text,
            elapsed_s=0.0,
        )

    def set_voice(self, path) -> dict:  # noqa: ANN001
        self.voices_set.append(str(path))
        self._reference = str(path)
        return {"ok": True}


class FakeMic:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.state = "open"

    def speak(self, pcm16: bytes, leading_silence: bool = True) -> float:
        assert self.state == "open"
        self.played.append(pcm16)
        return len(pcm16) / 2.0 / 24000.0

    def request_abort(self) -> None:
        pass

    def write_silence(self, ms: float) -> None:
        pass


def run_with_resolver(references: list[str | None], lines: list[str]) -> tuple[FakeClient, FakeMic]:
    """Speak `lines`, handing out `references` one per chunk in order."""
    client, mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(client, mic, PipelineConfig(mode="chunk"))
    handed = iter(references)
    pipe.set_reference_resolver(lambda _job: next(handed, None))
    pipe.start()
    for line in lines:
        pipe.say(line)
    pipe.stop(drain=True)
    return client, mic


def test_a_reference_can_be_chosen_per_chunk() -> None:
    client, mic = run_with_resolver(
        ["voices/emotions/happy.wav", "voices/emotions/sad.wav"],
        ["first line", "second line"],
    )
    assert client.seen == ["first line", "second line"], client.seen
    assert client.per_call_reference == [
        "voices/emotions/happy.wav",
        "voices/emotions/sad.wav",
    ], client.per_call_reference
    assert len(mic.played) == 2


def test_the_same_voice_is_not_switched_twice() -> None:
    """`set_voice` is a round trip, so an unchanged voice must not re-issue it."""
    client, _mic = run_with_resolver(
        ["voices/emotions/happy.wav"] * 3,
        ["one", "two", "three"],
    )
    assert client.voices_set == ["voices/emotions/happy.wav"], client.voices_set
    assert len(client.seen) == 3


def test_no_reference_leaves_the_selected_voice_alone() -> None:
    """With no emotion voices installed, nothing about the voice may change."""
    client, _mic = run_with_resolver([None, None], ["one", "two"])
    assert client.voices_set == [], client.voices_set
    assert len(client.seen) == 2


def test_a_switch_back_and_forth_is_issued_each_time() -> None:
    references = ["a.wav", "b.wav", "a.wav"]
    client, _mic = run_with_resolver(references, ["one", "two", "three"])
    assert client.voices_set == ["a.wav", "b.wav", "a.wav"], client.voices_set
    assert client.per_call_reference == references


def test_a_broken_resolver_does_not_lose_the_words() -> None:
    """A styling failure must never cost the sentence."""
    client, mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(client, mic, PipelineConfig(mode="chunk"))

    def exploding_resolver(_job):  # noqa: ANN001
        raise RuntimeError("the emotion map is on fire")

    pipe.set_reference_resolver(exploding_resolver)
    pipe.start()
    pipe.say("this must still be spoken")
    pipe.stop(drain=True)

    assert client.seen == ["this must still be spoken"], client.seen
    assert len(mic.played) == 1
    assert pipe.stats.failures == 0


def test_a_refused_voice_switch_still_speaks() -> None:
    """If the server refuses the emotion voice, speak in the old one, not silence."""
    client, mic = FakeClient(), FakeMic()
    client.set_voice = lambda path: {"ok": False, "error": "not a usable wav"}  # type: ignore[method-assign]
    pipe = SpeechPipeline(client, mic, PipelineConfig(mode="chunk"))
    pipe.set_reference_resolver(lambda _job: "voices/emotions/broken.wav")
    pipe.start()
    pipe.say("still speaks")
    pipe.stop(drain=True)

    assert client.seen == ["still speaks"], client.seen
    assert len(mic.played) == 1


def test_emotion_note_is_reported() -> None:
    client, mic = FakeClient(), FakeMic()
    pipe = SpeechPipeline(client, mic, PipelineConfig(mode="chunk"))
    assert pipe.emotion_note == ""
    pipe.set_emotion_note("[happy]")
    assert pipe.emotion_note == "[happy]"


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
