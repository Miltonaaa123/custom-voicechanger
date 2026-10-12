#!/usr/bin/env python
"""Always-on-top window showing the live transcript.

Reads the transcript files written by transcribe.py; it never touches the audio
pipeline. Handy because the words appear where you can see them while the file
quietly updates for StyleTTS 2.

    python viewer.py                       # watches out/live.txt
    python viewer.py --file out/transcript.txt
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live transcript viewer")
    p.add_argument("--file", default=str(HERE / "out" / "live.txt"), help="file to display")
    p.add_argument("--poll-ms", type=int, default=120, help="refresh interval")
    p.add_argument("--font-size", type=int, default=15)
    p.add_argument("--width", type=int, default=760)
    p.add_argument("--height", type=int, default=220)
    p.add_argument("--topmost", action="store_true", default=True)
    p.add_argument("--no-topmost", dest="topmost", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.file)

    root = tk.Tk()
    root.title(f"live transcript - {path.name}")
    root.geometry(f"{args.width}x{args.height}+60+60")
    root.attributes("-topmost", args.topmost)
    root.configure(bg="#14161a")

    body = tkfont.Font(family="Consolas", size=args.font_size)
    footer = tkfont.Font(family="Segoe UI", size=9)
    text = tk.Text(
        root,
        wrap="word",
        bg="#14161a",
        fg="#e8eaed",
        insertbackground="#e8eaed",
        font=body,
        relief="flat",
        padx=14,
        pady=10,
    )
    text.pack(fill="both", expand=True)
    status = tk.Label(root, text="waiting...", anchor="w", bg="#0e1013", fg="#8a8f98", font=footer, padx=10)
    status.pack(fill="x")

    state = {"content": None, "error": None}

    def refresh() -> None:
        try:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            state["error"] = None
        except OSError as exc:
            current = state["content"] or ""
            state["error"] = str(exc)

        if current != state["content"]:
            state["content"] = current
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", current)
            # Keep the newest words in view.
            text.see("end")
            size = len(current)
            status.configure(
                text=f"{size} chars  |  {path}  |  click to copy, drag to move"
                if not state["error"]
                else f"read error: {state['error']}"
            )
        root.after(args.poll_ms, refresh)

    def copy_all(_event=None) -> None:
        root.clipboard_clear()
        root.clipboard_append(state["content"] or "")
        status.configure(text="copied to clipboard")

    text.bind("<Button-1>", copy_all)
    root.bind("<Escape>", lambda _e: root.destroy())
    refresh()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
