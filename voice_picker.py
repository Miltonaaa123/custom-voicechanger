#!/usr/bin/env python
"""Pick the voice, without touching a terminal.

The voice changer reads ``voices/voice.txt`` before every sentence, so a picker
only has to write that file -- which means this window and the changer never need
to talk to each other, and a change takes effect on the next sentence.

    .venv\\Scripts\\python.exe voice_picker.py
    .venv\\Scripts\\python.exe voice_picker.py --no-test-button

Also shows which voice the speech server is actually cloning, so a switch that did
not take is visible immediately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import font as tkfont, messagebox

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_transcribe import voices as voices_mod  # noqa: E402

SELECTOR = "voice.txt"
POLL_MS = 1500

BACKGROUND = "#14161a"
PANEL = "#1b1e24"
EDGE = "#2a2f38"
TEXT = "#e8eaed"
DIM = "#8b929c"
OK = "#4ade80"
WARN = "#fbbf24"
BAD = "#f87171"


def server_voice(port: int) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            info = json.loads(response.read())
        path = str(info.get("reference_path") or "")
        return Path(path).stem if path else None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


class Picker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.voices_dir = Path(args.voices_dir).expanduser()
        self.selector = self.voices_dir / SELECTOR
        self.voices: list[voices_mod.Voice] = []

        self.root = tk.Tk()
        self.root.title("Voice picker")
        self.root.geometry(f"{args.width}x{args.height}+80+80")
        self.root.configure(bg=BACKGROUND)
        self.root.attributes("-topmost", bool(args.topmost))

        title_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        body_font = tkfont.Font(family="Segoe UI", size=10)
        small_font = tkfont.Font(family="Segoe UI", size=8)

        header = tk.Label(
            self.root,
            text="  Which voice should come out of the virtual microphone?",
            anchor="w",
            bg=PANEL,
            fg=TEXT,
            font=title_font,
            padx=10,
            pady=8,
        )
        header.pack(fill="x")

        self.listbox = tk.Listbox(
            self.root,
            bg=PANEL,
            fg=TEXT,
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            font=body_font,
            relief="flat",
            highlightthickness=0,
            activestyle="none",
            borderwidth=0,
        )
        self.listbox.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        self.listbox.bind("<Double-Button-1>", lambda _e: self.apply())
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.on_select())

        buttons = tk.Frame(self.root, bg=BACKGROUND)
        buttons.pack(fill="x", padx=10)

        tk.Button(
            buttons, text="Use this voice", command=self.apply, bg="#1d4ed8", fg="white",
            activebackground="#2563eb", relief="flat", padx=14, pady=6, font=body_font,
        ).pack(side="left")

        if not args.no_test_button:
            tk.Button(
                buttons, text="Hear it", command=self.test_voice, bg="#2a2f38", fg=TEXT,
                activebackground="#343a45", relief="flat", padx=14, pady=6, font=body_font,
            ).pack(side="left", padx=6)

        tk.Button(
            buttons, text="Reload list", command=self.reload, bg="#2a2f38", fg=TEXT,
            activebackground="#343a45", relief="flat", padx=14, pady=6, font=body_font,
        ).pack(side="right")

        self.status = tk.Label(
            self.root, text="", anchor="w", bg="#0e1013", fg=DIM, font=small_font, padx=10, pady=6
        )
        self.status.pack(fill="x", pady=(8, 0))

        hint = tk.Label(
            self.root,
            # Kept ASCII: this runs under pythonw, whose stdout encoding on a
            # non-UTF-8 code page cannot encode box-drawing characters.
            text=(
                f"  Add a voice by dropping a 3-16 s wav into {self.voices_dir.name}\\ "
                "  |  changes apply on the next sentence"
            ),
            anchor="w",
            bg=BACKGROUND,
            fg=DIM,
            font=small_font,
            padx=6,
            pady=4,
        )
        hint.pack(fill="x")

        self.reload()
        self.poll()

    # -- voices ------------------------------------------------------------ #
    def reload(self) -> None:
        self.voices = voices_mod.discover(self.voices_dir)
        self.listbox.delete(0, "end")
        for voice in self.voices:
            label = f"{voice.slug}   ({voice.seconds:.1f}s)"
            if voice.problem:
                label = f"{voice.slug}   - {voice.problem}"
            elif voice.name and voice.name.lower() != voice.slug.replace("_", " "):
                label = f"{voice.slug}   ({voice.seconds:.1f}s)   {voice.name}"
            self.listbox.insert("end", label)
            if voice.problem:
                self.listbox.itemconfig("end", foreground=BAD)

        active = self.active_slug()
        if active:
            for index, voice in enumerate(self.voices):
                if voice.slug == active:
                    self.listbox.selection_clear(0, "end")
                    self.listbox.selection_set(index)
                    self.listbox.see(index)
                    break
        self.refresh_status()

    def selected(self) -> voices_mod.Voice | None:
        picked = self.listbox.curselection()
        if not picked:
            return None
        return self.voices[picked[0]]

    def active_slug(self) -> str | None:
        try:
            if self.selector.is_file():
                name = self.selector.read_text(encoding="utf-8").strip()
                return name or None
        except OSError:
            pass
        return None

    def on_select(self) -> None:
        self.refresh_status()

    # -- actions ----------------------------------------------------------- #
    def apply(self) -> None:
        voice = self.selected()
        if voice is None:
            return
        if not voice.usable:
            messagebox.showwarning(
                "That voice cannot be used",
                f"{voice.slug}: {voice.problem}\n\nPick another, or replace the file.",
                parent=self.root,
            )
            return
        try:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            self.selector.write_text(voice.slug, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not write the selector", str(exc), parent=self.root)
            return
        self.refresh_status(f"Switched to {voice.slug} - it applies on the next sentence.")

    def test_voice(self) -> None:
        """Ask the running changer to speak a short line in the selected voice."""
        voice = self.selected()
        if voice is None:
            return
        self.apply()
        try:
            subprocess.Popen(
                [
                    str(HERE / ".venv" / "Scripts" / "python.exe"),
                    "-u",
                    str(HERE / "speak.py"),
                    "--voice",
                    voice.slug,
                    "This is the selected voice.",
                ],
                cwd=str(HERE),
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            self.refresh_status(f"Speaking a test line in {voice.slug} ...")
        except OSError as exc:
            messagebox.showerror("Could not start the test", str(exc), parent=self.root)

    # -- status ------------------------------------------------------------ #
    def refresh_status(self, note: str = "") -> None:
        chosen = self.selected()
        active = self.active_slug()
        live = server_voice(self.args.port)

        parts = []
        if chosen:
            parts.append(f"selected: {chosen.slug}")
        parts.append(f"voice.txt: {active or 'unset'}")
        parts.append(f"engine: {live or 'not running'}")

        colour = DIM
        if live is None:
            colour = BAD
            parts.append("- start run-tts.bat")
        elif active and live != active:
            colour = WARN
            parts.append("- applies on the next sentence")
        elif live:
            colour = OK

        if note:
            parts.append(f"| {note}")
        self.status.configure(text="   ".join(parts), fg=colour)

    def poll(self) -> None:
        self.refresh_status()
        self.root.after(POLL_MS, self.poll)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-picker", description="Choose the cloned voice.")
    parser.add_argument("--voices-dir", default=str(voices_mod.VOICES_DIR))
    parser.add_argument("--port", type=int, default=9881, help="speech server port")
    parser.add_argument("--width", type=int, default=430)
    parser.add_argument("--height", type=int, default=330)
    parser.add_argument("--topmost", action="store_true", default=True)
    parser.add_argument("--no-topmost", dest="topmost", action="store_false")
    parser.add_argument("--no-test-button", action="store_true",
                        help="hide the button that speaks a test line")
    args = parser.parse_args(argv)

    if not Path(args.voices_dir).is_dir():
        print(f"no voices folder at {args.voices_dir}", file=sys.stderr)
        return 1
    Picker(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
