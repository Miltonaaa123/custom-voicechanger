"""Reproduce the live run: real engine, real settings, capturing microphone.

The control panel's own file (out/rixers_settings.json) is loaded, so the pipeline
runs with exactly the settings the user had when the clip was recorded. Text is fed
the way out/transcript.txt actually grows: one committed word at a time.

    .venv\\Scripts\\python.exe tools/repro_live.py
    .venv\\Scripts\\python.exe tools/repro_live.py --mode sentence
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from live_transcribe import live_settings  # noqa: E402
from live_transcribe.pipeline import PipelineConfig, SpeechPipeline  # noqa: E402
from live_transcribe.tts import TtsClient  # noqa: E402

UTTERANCE = "borrow what did you say yo what did you say bro what was that"

# What Chrome's Web Speech API actually commits, and roughly when. The recogniser
# returns the whole utterance at once, so the file grows in one or two writes.
COMMITS = [
    (0.0, "borrow what did you say yo "),
    (0.15, "what did you say bro what was that"),
]


class CapturingMic:
    """A virtual microphone that records instead of playing, at real speed.

    `write` sleeps for the duration of the audio it is given, exactly as the real
    device does, so the producer/player race is the same one the user hits.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.device_rate = sample_rate
        self.buffer: list[np.ndarray] = []
        self.blocks = 0
        self.seconds = 0.0
        self.silence_ms: list[float] = []
        self.play_calls: list[tuple[float, float]] = []   # (t_rel, seconds)
        self._abort = threading.Event()
        self._t0 = time.perf_counter()

    def request_abort(self) -> None:
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    def write(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        block = np.asarray(samples, dtype=np.float32)
        self.buffer.append(block)
        self.blocks += 1
        self.seconds += block.size / float(self.sample_rate)
        # Real time passes while the device plays it.
        self._abort.wait(block.size / float(self.sample_rate))

    def write_silence(self, milliseconds: float | None = None) -> None:
        ms = 60.0 if milliseconds is None else float(milliseconds)
        self.silence_ms.append(ms)
        self.write(np.zeros(int(self.sample_rate * ms / 1000.0), dtype=np.float32))

    def speak(self, pcm16: bytes, leading_silence: bool = True) -> float:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.play_calls.append((time.perf_counter() - self._t0, samples.size / self.sample_rate))
        if leading_silence:
            self.write_silence()
        self.write(samples)
        return samples.size / float(self.sample_rate)

    def drain(self, milliseconds: float = 250.0) -> None:
        pass

    def wav(self, path: Path) -> float:
        if not self.buffer:
            return 0.0
        data = np.concatenate(self.buffer)
        pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as h:
            h.setnchannels(1)
            h.setsampwidth(2)
            h.setframerate(self.sample_rate)
            h.writeframes(pcm.tobytes())
        return data.size / float(self.sample_rate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, help="override the saved mode")
    ap.add_argument("--out", default=str(HERE / "out" / "repro"))
    ap.add_argument("--settle", type=float, default=25.0,
                    help="how long to wait after the last word before stopping")
    args = ap.parse_args()

    settings = live_settings.LiveSettings()
    settings.load()
    values = settings.values
    mode = args.mode or str(values["mode"])
    print("settings in force:")
    for key in ("mode", "target_words", "max_words", "queue", "backlog", "gap_ms",
                "trim_silence_ms", "retries", "word_delay", "word_gap_min_ms",
                "word_gap_max_ms", "pause_flush_s", "pitch_semitones", "volume",
                "gate_db", "gate_floor_db", "voice"):
        print(f"   {key:>18}: {values[key]}")
    print(f"   {'(mode override)':>18}: {mode}" if args.mode else "")

    config = PipelineConfig(
        mode=mode,
        target_words=int(values["target_words"]),
        max_words=int(values["max_words"]),
        queue_size=int(values["queue"]),
        pending_limit=int(values["backlog"]),
        gap_ms=float(values["gap_ms"]),
        trim_silence_ms=float(values["trim_silence_ms"]),
        retries=int(values["retries"]),
        word_delay_replay=str(values["word_delay"]) != "off",
        word_gap_min_ms=float(values["word_gap_min_ms"]),
        word_gap_max_ms=float(values["word_gap_max_ms"]),
    )

    mic = CapturingMic()
    client = TtsClient()

    def log(message: str) -> None:
        print(f"{time.perf_counter() - t0:7.2f}s {message}", flush=True)

    pipe = SpeechPipeline(client, mic, config, logger=log)
    pipe.set_timing_resolver(lambda: str(values["word_delay"]) != "off")
    pipe.audio.pitch_semitones = float(values["pitch_semitones"])
    pipe.audio.volume = float(values["volume"])
    pipe.audio.gate_db = float(values["gate_db"])
    pipe.audio.gate_floor_db = float(values["gate_floor_db"])
    pipe.start()

    t0 = time.perf_counter()
    print(f"\nfeeding {len(UTTERANCE.split())} words over {sum(1 for _ in COMMITS)} commits")
    for delay, text in COMMITS:
        while time.perf_counter() - t0 < delay:
            time.sleep(0.01)
        n = pipe.feed(text)
        print(f"{time.perf_counter() - t0:7.2f}s fed {n} chunk(s): {text!r}", flush=True)

    # Let the pipeline work through its queue at real speed, exactly as it would
    # while the user goes quiet.
    last = time.perf_counter()
    while time.perf_counter() - last < args.settle:
        time.sleep(0.5)
        if pipe.pending_count() == 0:
            last = time.perf_counter()
        if time.perf_counter() - t0 > 180:
            break

    pipe.flush()
    pipe.stop(drain=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"repro_{mode}.wav"
    seconds = mic.wav(path)
    print(f"\n{pipe.summary()}")
    print(f"\nrecorded {seconds:.2f}s to {path}")
    print(f"  play calls      : {len(mic.play_calls)}")
    print(f"  silence writes  : {len(mic.silence_ms)}")
    print(f"  expected words  : {len(UTTERANCE.split())}")
    if mic.play_calls:
        print("  plays (t, seconds):")
        for t, s in mic.play_calls:
            print(f"     {t:7.2f}s  {s:5.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
