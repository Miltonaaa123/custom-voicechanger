"""The render guard: does a blown-up StyleTTS 2 render get caught and re-rolled?

This is the check that stops the voice "tweaking out". StyleTTS 2 fails at random on
short text, and the failure is not always loud -- measured through the live engine,
9 of 12 renders of the single word "you" came back as 1.1-4.6 s of hiss peaking at
0.04-0.76, which the old peak-only guard could never see.

Nothing here needs the model or a GPU: the guard is a measurement on decoded audio,
and the re-roll is driven through a stub engine that returns a failing render first
and a good one second. That is what makes it testable at all.

    .venv\\Scripts\\python.exe tests\\test_render_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe import engine_tts_server as engine_mod  # noqa: E402

SR = engine_mod.SAMPLE_RATE


# --------------------------------------------------------------------------- #
# synthetic renders
# --------------------------------------------------------------------------- #
def voiced(seconds: float, f0: float = 120.0) -> np.ndarray:
    """A stand-in for speech: harmonics at a speaking pitch, nothing up high."""
    t = np.arange(int(SR * seconds)) / SR
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 3.0 * t))       # syllable-ish
    out = np.zeros_like(t)
    for harmonic in range(1, 12):
        if f0 * harmonic > 2000:
            break
        out += np.sin(2 * np.pi * f0 * harmonic * t) / harmonic
    return (0.4 * envelope * out).astype(np.float32)


def buzz(seconds: float, level: float = 0.3) -> np.ndarray:
    """A stand-in for a failed render: full-band noise, no voice in it."""
    rng = np.random.default_rng(12345)
    t = np.arange(int(SR * seconds)) / SR
    # A slow wobble on top of the noise, which is what makes a failed render sound
    # like a warble rather than a plain hiss.
    return (level * rng.standard_normal(t.size) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t))
            ).astype(np.float32)


class StubModel:
    """Returns the renders it is handed, one per inference call."""

    def __init__(self, renders: list[np.ndarray]) -> None:
        self.renders = list(renders)
        self.steps: list[int] = []

    def inference(self, text, target_voice_path=None, output_sample_rate=None,
                  diffusion_steps=5, ref_s=None):  # noqa: ANN001, ANN003
        self.steps.append(int(diffusion_steps))
        return self.renders.pop(0) if self.renders else voiced(0.5)


def stub_engine(renders: list[np.ndarray]) -> engine_mod.StyleEngine:
    engine = engine_mod.StyleEngine.__new__(engine_mod.StyleEngine)
    engine.model = StubModel(renders)
    engine.reference = None
    engine.diffusion_steps = engine_mod.DEFAULT_DIFFUSION_STEPS
    engine.load_error = ""
    engine.load_seconds = 0.0
    engine.synthesised = 0
    engine.runaway_retries = 0
    engine.config_data = {}
    engine._config_file = None
    engine._styles = {}
    import threading
    engine._lock = threading.RLock()
    engine._loading = threading.Event()
    engine._ready = threading.Event()
    return engine


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    good = voiced(0.5)
    print("\n-- what the guard can see --")
    check("a clean short render passes", engine_mod.render_problem(good, "you") == "",
          engine_mod.render_problem(good, "you"))
    check("full-scale saturation is caught",
          "saturat" in engine_mod.render_problem(np.clip(good * 40, -1, 1), "you"))
    # Over-long but a voice, so length is the only thing wrong with it.
    check("a 4.5 s render of one word is caught",
          "runaway" in engine_mod.render_problem(voiced(4.5), "you"),
          engine_mod.render_problem(voiced(4.5), "you"))
    # The failure this guard exists for: an ordinary level, no clipping, no voice.
    quiet_hiss = buzz(0.9, level=0.001)
    problem = engine_mod.render_problem(quiet_hiss, "you")
    check("a quiet hiss is caught even though it never clips",
          "noise" in problem and float(np.max(np.abs(quiet_hiss))) < 0.5, problem)
    check("a long sentence is allowed to be long",
          engine_mod.render_problem(voiced(5.0), "borrow what did you say what did you say") == "",
          engine_mod.render_problem(voiced(5.0), "borrow what did you say what did you say"))
    check("the length cap scales with the words",
          engine_mod.spoken_seconds_ceiling("you") < engine_mod.spoken_seconds_ceiling("a b c d"))
    check("the length cap never falls below the floor",
          engine_mod.spoken_seconds_ceiling("a") == engine_mod.MIN_SPOKEN_SECONDS,
          engine_mod.spoken_seconds_ceiling("a"))
    check("noise_share is low for a voice and high for noise",
          engine_mod.noise_share(good) < engine_mod.MAX_NOISE_SHARE
          < engine_mod.noise_share(buzz(1.0)),
          (engine_mod.noise_share(good), engine_mod.noise_share(buzz(1.0))))

    print("\n-- re-rolling a failed render --")
    engine = stub_engine([buzz(4.5), good])
    result = engine.synthesize("you")
    check("a failed render is re-rolled", result["attempts"] == 2, result["attempts"])
    check("the good render is the one returned", result["render_problem"] == "")
    check("the returned audio is the good one",
          abs(result["seconds"] - 0.5) < 0.05, result["seconds"])
    check("the re-roll uses a different step count",
          engine.model.steps == [engine_mod.DEFAULT_DIFFUSION_STEPS,
                                 engine_mod.SAFE_DIFFUSION_STEPS], engine.model.steps)
    check("the re-roll is reported", engine.runaway_retries == 1, engine.runaway_retries)

    print("\n-- when every render fails --")
    engine = stub_engine([buzz(4.5), buzz(4.0), buzz(3.5), buzz(3.0)])
    result = engine.synthesize("you")
    check("every step in the ladder is tried",
          result["attempts"] == len({engine_mod.DEFAULT_DIFFUSION_STEPS,
                                     engine_mod.SAFE_DIFFUSION_STEPS, 4, 6}),
          result["attempts"])
    check("something is returned rather than silence", result["seconds"] > 0)
    check("the failure is reported honestly", bool(result["render_problem"]),
          result["render_problem"])
    check("it is counted as a runaway", engine.runaway_retries == 1)

    print("\n-- a good first render must cost nothing extra --")
    engine = stub_engine([good])
    result = engine.synthesize("you")
    check("only one inference call", engine.model.steps == [engine_mod.DEFAULT_DIFFUSION_STEPS],
          engine.model.steps)
    check("nothing is counted as a re-roll", engine.runaway_retries == 0)
    check("the response still carries the render numbers",
          {"attempts", "noise_share", "render_problem"} <= set(result), sorted(result))

    print("\n-- the ladder reaches the requested steps --")
    engine = stub_engine([good])
    result = engine.synthesize("you", steps=9)
    check("an explicit step count is used first", engine.model.steps == [9], engine.model.steps)
    check("the step count comes back", result["diffusion_steps"] == 9, result["diffusion_steps"])

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
