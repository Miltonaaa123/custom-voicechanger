"""Does any StyleTTS2 knob actually change the emotion of the speech?

StyleTTS2's `inference()` takes `alpha`, `beta`, `embedding_scale` and
`diffusion_steps`. The docstring says `embedding_scale` makes the style "more
conditional to the input text and hence more emotional", which is exactly what an
emotion feature would need -- but claiming that is not the same as it being audible.

This measures it: the same sentence is synthesized at several settings and the
acoustic result is compared (duration, pitch, energy, pitch variability). If nothing
moves, the feature is not worth building on this checkpoint.

Run with the StyleTTS2 environment, not the project one:
    <riko>\\scripts\\models\\styletts2\\.venv\\Scripts\\python.exe tools\\style_probe.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RIKO = Path.home() / "Desktop" / "riko-project"
MODELS = RIKO / "scripts" / "models" / "styletts2"
CHECKPOINT = MODELS / "epochs_2nd_00020.pth"
CONFIG = MODELS / "config.yml"

TEXT = "I cannot believe this is actually happening right now, it is incredible."


def measure(samples: np.ndarray, rate: int) -> dict:
    """Cheap acoustic summary: enough to tell whether the delivery changed."""
    if samples.size == 0:
        return {}
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    # Frame energy envelope, 25 ms frames.
    frame = max(1, int(0.025 * rate))
    usable = samples[: (samples.size // frame) * frame].reshape(-1, frame)
    energy = np.sqrt(np.mean(np.square(usable), axis=1)) if usable.size else np.zeros(1)
    # Zero-crossing rate approximates pitch movement without a full pitch tracker.
    zcr = np.mean(np.abs(np.diff(np.sign(usable), axis=1)) > 0, axis=1) if usable.size else np.zeros(1)
    return {
        "seconds": round(samples.size / rate, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 5),
        "energy_mean": round(float(np.mean(energy)), 5),
        "energy_std": round(float(np.std(energy)), 5),
        "zcr_mean": round(float(np.mean(zcr)), 5),
        "zcr_std": round(float(np.std(zcr)), 5),
        "silence_pct": round(100.0 * float(np.mean(energy < 0.01)), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(HERE / "voices" / "default.wav"))
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out-dir", default="out/style_probe")
    ap.add_argument("--port", type=int, default=9881)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    import styletts2.tts as tts_module

    print(f"loading {CHECKPOINT.name} ...", flush=True)
    model = tts_module.StyleTTS2(
        model_checkpoint_path=str(CHECKPOINT) if CHECKPOINT.is_file() else None,
        config_path=str(CONFIG) if CONFIG.is_file() else None,
    )
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    reference = Path(args.reference)
    if not reference.is_file():
        print(f"no reference wav at {reference}")
        return 1
    ref_s = model.compute_style(str(reference))
    print(f"reference: {reference.name}\n", flush=True)

    # Baseline is what the server uses today (alpha .3, beta .7, 3 steps, scale 1).
    cases = [
        ("baseline (what you hear now)", dict(alpha=0.3, beta=0.7, diffusion_steps=3, embedding_scale=1)),
        ("embedding_scale 1.5", dict(alpha=0.3, beta=0.7, diffusion_steps=3, embedding_scale=1.5)),
        ("embedding_scale 2", dict(alpha=0.3, beta=0.7, diffusion_steps=3, embedding_scale=2)),
        ("embedding_scale 3", dict(alpha=0.3, beta=0.7, diffusion_steps=3, embedding_scale=3)),
        ("beta 0.5 (prosody to text)", dict(alpha=0.3, beta=0.5, diffusion_steps=3, embedding_scale=1)),
        ("beta 0.9 (prosody to voice)", dict(alpha=0.3, beta=0.9, diffusion_steps=3, embedding_scale=1)),
        ("alpha 0.5 (timbre to text)", dict(alpha=0.5, beta=0.7, diffusion_steps=3, embedding_scale=1)),
        ("more steps (5)", dict(alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)),
        ("steps 5 + scale 2", dict(alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=2)),
    ]

    results = []
    print(f"{'setting':<30}{'sec':>6}{'peak':>7}{'rms':>8}{'E mean':>8}{'E std':>8}{'zcr':>7}{'sil%':>7}{'t':>6}")
    print("-" * 92)
    for label, kwargs in cases:
        started = time.perf_counter()
        try:
            audio = model.inference(args.text, ref_s=ref_s, output_sample_rate=24000, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<30} FAILED: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        samples = np.asarray(audio, dtype=np.float32).ravel()
        stats = measure(samples, 24000)
        if not stats:
            print(f"{label:<30} produced no audio")
            continue
        stats.update({"label": label, "kwargs": kwargs, "synth_s": round(elapsed, 2)})
        results.append(stats)

        path = out_dir / f"{len(results):02d}_{label.split()[0].replace('(','')}.wav"
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(pcm.tobytes())

        print(
            f"{label:<30}{stats['seconds']:6.2f}{stats['peak']:7.3f}{stats['rms']:8.4f}"
            f"{stats['energy_mean']:8.4f}{stats['energy_std']:8.4f}{stats['zcr_mean']:7.4f}"
            f"{stats['silence_pct']:7.1f}{elapsed:6.2f}"
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if len(results) >= 2:
        base = results[0]
        print("\nchange versus the baseline (what the server does today):")
        for stats in results[1:]:
            parts = []
            for key, fmt, label in (
                ("seconds", "{:+.0f}%", "length"),
                ("energy_std", "{:+.0f}%", "energy variation"),
                ("zcr_mean", "{:+.0f}%", "pitch (zcr)"),
                ("zcr_std", "{:+.0f}%", "pitch variation"),
            ):
                if base.get(key):
                    delta = 100.0 * (stats[key] - base[key]) / base[key]
                    parts.append(f"{label} {fmt.format(delta)}")
            print(f"  {stats['label']:<30} {'  '.join(parts)}")

        print(
            "\nIf those percentages are all near zero, this checkpoint ignores these knobs"
            " and an emotion feature cannot be built on them."
        )
    print(f"\nwavs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
