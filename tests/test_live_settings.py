"""The live settings bridge and the output effects.

Two things are checked here that nothing else can:

* that a change written to the settings file reaches a running voice changer's
  callback exactly once, and that a hand-edited file with nonsense in it degrades
  to the defaults rather than to broken audio;
* that the effects actually change the samples the way the labels say -- pitch up
  makes a clip shorter and the stream faster, the gate leaves speech alone and
  pushes silence down, volume scales.

    .venv\\Scripts\\python.exe tests\\test_live_settings.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe import audio_effects, live_settings  # noqa: E402


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.json"
        settings = live_settings.LiveSettings(path)

        # -- defaults ------------------------------------------------------ #
        settings.save()
        check("saving creates the file", path.is_file())
        settings.load(force=True)
        check("the defaults survive a round trip",
              settings.get("pause_flush_s") == 0.7 and settings.get("mode") == "sentence",
              settings.values)

        # -- the watcher --------------------------------------------------- #
        # The watcher is given a *reader* instance, which is what the voice changer
        # has: the control panel is a different process with its own settings object,
        # so a change only becomes visible through the file. Mutating the watcher's
        # own object would be testing nothing -- it already knows those values.
        reader = live_settings.LiveSettings(path)
        reader.load(force=True)
        seen: list[tuple[list[str], dict]] = []
        watcher = live_settings.SettingsWatcher(
            reader, lambda changed, values: seen.append((changed, dict(values.values))),
            interval_s=0.0,
        )
        check("nothing fires when nothing changed", watcher.poll() == [])
        settings.update(pause_flush_s=1.5)
        settings.save()
        changed = watcher.poll()
        check("a change is reported once", changed == ["pause_flush_s"], changed)
        check("the callback saw the new value", seen and seen[0][1]["pause_flush_s"] == 1.5, seen)
        check("it does not fire again", watcher.poll() == [])

        settings.update(volume=1.5, gate_db=-40.0)
        settings.save()
        changed = watcher.poll()
        check("several changes arrive together", sorted(changed) == ["gate_db", "volume"], changed)
        check("every change reached the callback", len(seen) == 2, seen)

        # A setting the running process cannot use must be ignored, not applied.
        settings.update(mode="nonsense", pause_flush_s=99.0)
        settings.save()
        values = live_settings.LiveSettings(path)
        values.load(force=True)
        check("a nonsense choice falls back to the default", values.get("mode") == "sentence",
              values.get("mode"))
        check("an out-of-range number is clamped to the field's maximum",
              values.get("pause_flush_s") == live_settings.FIELDS["pause_flush_s"]["max"],
              values.get("pause_flush_s"))

        # A hand-broken file must not stop the voice changer.
        path.write_text("{not json at all", encoding="utf-8")
        broken = live_settings.LiveSettings(path)
        check("a corrupt file is refused, not raised", broken.load(force=True) is False)
        check("and the defaults are still there", broken.get("pause_flush_s") == 0.7)

        # The order rule: the cap can never end up below the floor.
        values = live_settings.sanitise({"word_gap_min_ms": 400.0, "word_gap_max_ms": 100.0})
        check("min and max gaps cannot cross",
              values["word_gap_max_ms"] >= values["word_gap_min_ms"], values)

    # -- the effects ------------------------------------------------------- #
    rate = 24000
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * np.arange(rate // 2) / rate)).astype(np.float32)

    up = audio_effects.AudioOptions(pitch_semitones=12.0)
    check("an octave up is twice the rate", abs(up.pitch_ratio - 2.0) < 1e-9, up.pitch_ratio)
    check("the stream rate follows the pitch", audio_effects.choose_rate(24000, 2.0) == 48000,
          audio_effects.choose_rate(24000, 2.0))
    shifted = audio_effects.resample(tone, up.pitch_ratio)
    check("pitch up makes the clip shorter", len(shifted) < len(tone),
          (len(tone), len(shifted)))
    check("resampling keeps the shape", shifted.dtype == np.float32 and shifted.size > 0)
    check("no shift means no copy", audio_effects.resample(tone, 1.0) is tone)

    louder = audio_effects.process(tone, rate, audio_effects.AudioOptions(volume=2.0))
    check("volume raises the level", float(np.abs(louder).mean()) > float(np.abs(tone).mean()),
          (float(np.abs(tone).mean()), float(np.abs(louder).mean())))
    check("volume is soft-limited", float(np.abs(louder).max()) <= 1.0,
          float(np.abs(louder).max()))
    quieter = audio_effects.process(tone, rate, audio_effects.AudioOptions(volume=0.25))
    check("volume below 1.0 scales down", float(np.abs(quieter).max()) < float(np.abs(tone).max()))

    # Speech-like signal: loud in the middle, near-silent at the edges.
    speech = np.zeros(rate, dtype=np.float32)
    speech[rate // 4: 3 * rate // 4] = tone[: rate // 2]
    speech += 0.0005 * np.random.default_rng(7).standard_normal(rate).astype(np.float32)
    gated = audio_effects.process(speech, rate, audio_effects.AudioOptions(gate_db=-40.0,
                                                                          gate_floor_db=-60.0))
    middle = slice(rate // 4, 3 * rate // 4)
    edge_slice = np.r_[np.arange(0, rate // 8), np.arange(7 * rate // 8, rate)]
    check("the gate leaves speech alone",
          abs(float(np.abs(gated[middle]).max()) - float(np.abs(speech[middle]).max())) < 0.05)
    check("the gate pushes the quiet parts down",
          float(np.abs(gated[edge_slice]).max()) < float(np.abs(speech[edge_slice]).max()),
          (float(np.abs(speech[edge_slice]).max()), float(np.abs(gated[edge_slice]).max())))

    clean = audio_effects.process(tone, rate, audio_effects.AudioOptions())
    check("defaults change nothing", np.array_equal(clean, tone))
    check("defaults are reported as such", audio_effects.AudioOptions().is_default)
    check("a pitch shift is described", "pitch" in up.describe(), up.describe())

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
