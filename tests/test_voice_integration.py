"""Integration test: the voice changer follows the transcript and swaps voices.

Starts the real ``voice_changer.py``, appends text to a transcript file the way
``transcribe.py`` does, and checks that:

* only complete new sentences are spoken (not the old contents, not half-lines)
* writing a different name into ``voices/voice.txt`` changes the voice mid-run

    .venv\\Scripts\\python.exe tests\\test_voice_integration.py
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"


def server_voice(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
        return str(json.loads(response.read()).get("reference_path") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--mode", default="sentence", choices=["word", "chunk", "phrase", "sentence"])
    args = ap.parse_args()

    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        transcript = tmpdir / "transcript.txt"
        voices_dir = tmpdir / "voices"
        voices_dir.mkdir()

        # Two distinct voices, copied from the real folder so they are valid clips.
        import shutil

        shutil.copy(HERE / "voices" / "main_sample.wav", voices_dir / "alpha.wav")
        shutil.copy(HERE / "voices" / "testing1.wav", voices_dir / "beta.wav")

        # Pre-existing text must NOT be spoken: it belongs to an earlier session.
        transcript.write_text("This old sentence must never be spoken aloud. ", encoding="utf-8")

        proc = subprocess.Popen(
            [
                str(PYTHON), "-u", str(HERE / "voice_changer.py"),
                "--transcript", str(transcript),
                "--voices-dir", str(voices_dir),
                "--voice", "alpha",
                "--port", str(args.port),
                "--mode", args.mode,
            ],
            cwd=str(HERE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        lines: list[str] = []
        output_queue: "queue.Queue[str]" = queue.Queue()

        def reader() -> None:
            """Drain stdout on its own thread.

            readline() blocks until a line arrives, so a deadline cannot be
            enforced on it directly; a queue gives the timed wait instead.
            """
            assert proc.stdout is not None
            for raw in proc.stdout:
                output_queue.put(raw.rstrip())
            output_queue.put("")  # sentinel: the process closed its output

        threading.Thread(target=reader, daemon=True).start()

        def pump(seconds: float) -> None:
            """Collect output for a while (never blocks past the deadline)."""
            deadline = time.time() + seconds
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return
                try:
                    line = output_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if line == "":
                    return
                lines.append(line)

        def wait_for(predicate, timeout: float, what: str) -> bool:
            """Poll until the output satisfies `predicate`, or give up.

            Synthesis on a contended GPU takes tens of seconds, so fixed sleeps
            make this test flaky; watching the output is the reliable signal.
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                if predicate("\n".join(lines)):
                    return True
                pump(0.5)
            return predicate("\n".join(lines))

        def all_words_present(text: str, words: str) -> bool:
            """Check every word was spoken.

            In word mode each word is its own output line, so a phrase can only be
            verified word by word rather than as one string.
            """
            spoken = " ".join(
                line.split("): ", 1)[1] for line in text.splitlines() if "): " in line
            ).lower()
            return all(word.lower() in spoken for word in words.split())

        try:
            # Startup: wait for the server handshake and the first voice switch.
            # The GPU may be busy, so allow generous time and watch the output.
            check(
                "started and found the speech server",
                wait_for(lambda t: "speech server" in t, 90, "server"),
                "\n".join(lines[-8:]),
            )
            check(
                "applied the startup voice",
                wait_for(lambda t: "voice: alpha" in t, 60, "alpha"),
                "\n".join(lines[-8:]),
            )
            joined = "\n".join(lines)
            check("did not speak pre-existing text", "old sentence must never" not in joined, joined[-400:])

            mark = len(lines)
            # In sentence mode nothing may be spoken until the sentence ends --
            # that is the mode's whole contract.
            with transcript.open("a", encoding="utf-8") as handle:
                handle.write("This first sentence is complete")
            pump(8)
            so_far = "\n".join(lines[mark:])
            check(
                "an unfinished sentence is not spoken",
                "spoke" not in so_far,
                so_far[-400:],
            )

            # ...until it is finished.
            with transcript.open("a", encoding="utf-8") as handle:
                handle.write(". And here is a second one.")

            def all_words_present(text: str, words: str) -> bool:
                """Every word must appear, in any line.

                Word mode puts each word on its own line, so a phrase can only be
                verified word by word rather than as one string.
                """
                spoken = " ".join(
                    line.split("): ", 1)[1] for line in text.splitlines() if "): " in line
                ).lower()
                return all(word.lower() in spoken for word in words.split())

            check(
                "the finished sentence was spoken",
                wait_for(
                    lambda t: all_words_present(t, "This first sentence is complete"),
                    150,
                    "sentence 1",
                ),
                "\n".join(lines[mark:])[-500:],
            )
            check(
                "the second sentence was spoken",
                wait_for(
                    lambda t: all_words_present(t, "And here is a second one"),
                    150,
                    "sentence 2",
                ),
                "\n".join(lines[mark:])[-500:],
            )

            # Live voice switch: write a name into voice.txt, no restart.
            before = server_voice(args.port)
            (voices_dir / "voice.txt").write_text("beta", encoding="utf-8")
            with transcript.open("a", encoding="utf-8") as handle:
                handle.write(" Now the voice should change.")
            check(
                "voice switched live",
                wait_for(lambda t: "voice: beta" in t, 120, "beta"),
                "\n".join(lines[-8:]),
            )
            # The switch is applied just before the sentence is synthesized, so
            # give the server a moment to have taken it.
            after_switch = before
            for _ in range(40):
                after_switch = server_voice(args.port)
                if "beta" in Path(after_switch).name:
                    break
                time.sleep(0.5)
            check(
                "the server is now cloning the new voice",
                after_switch != before and "beta" in Path(after_switch).name,
                f"{before} -> {after_switch}",
            )

            after_all = "\n".join(lines)
            check(
                "no sentence was spoken twice",
                len(re.findall(r"spoke .*?: This first sentence", after_all)) <= 1,
                after_all[-600:],
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            # Put the real voice.txt back the way it was.
            real = HERE / "voices" / "voice.txt"
            if real.exists():
                real.unlink()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
