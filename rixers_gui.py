#!/usr/bin/env python
"""Rixers Custom Voice Changer - the control panel.

A real Windows window (Tkinter, no browser and no web server) for everything the
voice changer can do: which voice to clone, how long to wait after you stop
speaking, how much of your own rhythm to put back, how much text each synthesis
call gets, the audio devices, the output effects, and the microphone -> text stage
that feeds it.

How it talks to the voice changer
---------------------------------
The window and the voice changer are separate processes that meet at one file:

    out/rixers_settings.json

The window writes it; the voice changer re-reads it between sentences. So a change
to a live setting (rhythm, chunking, volume, gate) is in force on the next sentence
without restarting anything, and a change to a start-up setting (the output device,
whether to monitor, the transcription engine) is in force the next time it starts.
`live_transcribe.live_settings` is the schema both sides agree on.

Nothing here invents a setting. If a control is on these pages, it changes
something real; the pages that would need a knob the speech server does not have
say so instead of pretending.

    .venv\\Scripts\\pythonw.exe rixers_gui.py
    RixersVoiceChanger.bat          (double-click launcher)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_transcribe import live_settings  # noqa: E402
from live_transcribe import emotion as emotion_mod  # noqa: E402
from live_transcribe import emotion_voices, voices as voices_mod  # noqa: E402

APP_NAME = "Rixers Custom Voice Changer"
TAGLINE = "Control Panel & Settings"
VERSION = "v1.0.0"
PROFILE_NAME = "Custom Build"
PYTHON = HERE / ".venv" / "Scripts" / "python.exe"
PYTHONW = HERE / ".venv" / "Scripts" / "pythonw.exe"
TTS_PORT = 9881
STT_PORT = 8777

# --------------------------------------------------------------------------- #
# Palette: the same values the reference design uses, so the window and the
# mock-up are recognisably the same application.
# --------------------------------------------------------------------------- #
BG = "#090D16"
PANEL = "#101622"
PANEL_2 = "#1A2234"
EDGE = "#232C42"
EDGE_SOFT = "#1E2637"
SUNKEN = "#0C1119"
ROW = "#111826"
CYAN = "#06B6D4"
INDIGO = "#6366F1"
TEXT = "#E5E7EB"
BRIGHT = "#FFFFFF"
DIM = "#8B94A7"
FAINT = "#5D6678"
OK = "#34D399"
WARN = "#FBBF24"
BAD = "#F87171"
RED = "#EF4444"

FONT = "Segoe UI"
SIDEBAR_W = 272
RADIUS = 14

PAGES = [
    ("dashboard", "Dashboard"),
    ("voices", "Voices"),
    ("timing", "Timing & Rhythm"),
    ("engine", "Speech Engine"),
    ("audio", "Audio & Devices"),
    ("effects", "Effects"),
    ("input", "Transcription"),
    ("advanced", "Advanced"),
]


# --------------------------------------------------------------------------- #
# Small drawing helpers: Tkinter has no rounded rectangles, so the cards and
# toggles are drawn onto canvases.
# --------------------------------------------------------------------------- #
def rounded(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float,
            radius: float = RADIUS, **kwargs):
    """A rounded rectangle on `canvas`; returns the item id."""
    radius = max(0.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


class Card(tk.Frame):
    """A panel with a heading, holding rows of settings.

    The border is two frames rather than a drawn rounded rectangle. A canvas is the
    obvious way to get rounded corners, but a canvas does not reliably claim the
    space its siblings need: it either asks for Tk's default 7cm and starves the
    contents, or -- told to ask for nothing -- collapses to a 1px strip and draws no
    border at all. A one-pixel frame of the border colour around a panel-coloured
    frame sizes correctly, always.
    """

    def __init__(self, master: tk.Misc, title: str = "", note: str = "", accent: str = CYAN):
        super().__init__(master, bg=EDGE)
        self.panel = tk.Frame(self, bg=PANEL)
        self.panel.pack(fill="both", expand=True, padx=1, pady=1)
        self.inner = tk.Frame(self.panel, bg=PANEL)
        self.inner.pack(fill="both", expand=True)

        if title:
            head = tk.Frame(self.inner, bg=PANEL)
            head.pack(fill="x", padx=20, pady=(16, 0))
            tk.Label(head, text="\u25CF", bg=PANEL, fg=accent,
                     font=(FONT, 9)).pack(side="left", padx=(0, 8))
            tk.Label(head, text=title, bg=PANEL, fg=TEXT,
                     font=(FONT, 11, "bold")).pack(side="left")
            if note:
                tk.Label(self.inner, text=note, bg=PANEL, fg=FAINT, font=(FONT, 9),
                         justify="left", wraplength=780, anchor="w").pack(
                    fill="x", padx=20, pady=(4, 0))
        # Everything a page adds goes in here.
        self.rows = tk.Frame(self.inner, bg=PANEL)
        self.rows.pack(fill="both", expand=True, padx=12, pady=(10, 12))


def table(parent: tk.Misc, headers: list[tuple[str, int]], height: int = 7) -> ttk.Treeview:
    """A dark table in the card style."""
    tree = ttk.Treeview(parent, columns=[name for name, _width in headers],
                        show="headings", height=height)
    for name, width in headers:
        tree.heading(name, text=name)
        tree.column(name, width=width, anchor="w")
    return tree


def accent_row(master: tk.Misc, accent: str = CYAN, padx: int = 12, pady: int = 9) -> tk.Frame:
    """A setting row: a sunken panel with an accent bar down its left edge.

    The bar is a thin frame of the accent colour with the panel packed on top of
    it, so the row is exactly as tall as its contents -- the row cannot collapse
    the way a canvas-backed one does.
    """
    bar = tk.Frame(master, bg=accent)
    bar.pack(fill="x", padx=6, pady=4)
    content = tk.Frame(bar, bg=ROW, padx=padx, pady=pady)
    content.pack(side="left", fill="both", expand=True, padx=(3, 0))
    return content


class Toggle(tk.Canvas):
    """The pill switch from the design, drawn because Tk has no such widget."""

    W, H = 40, 22

    def __init__(self, master: tk.Misc, value: bool = False, command=None):
        super().__init__(master, width=self.W, height=self.H, bg=ROW,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.value = bool(value)
        self.command = command
        self._track = rounded(self, 1, 1, self.W - 1, self.H - 1, (self.H - 2) / 2,
                              fill=PANEL_2, outline="")
        self._knob = self.create_oval(3, 3, self.H - 3, self.H - 3, fill="#7A8398", outline="")
        self.bind("<Button-1>", self._click)
        self._paint()

    def _paint(self) -> None:
        self.itemconfig(self._track, fill=CYAN if self.value else PANEL_2)
        self.itemconfig(self._knob, fill=BRIGHT if self.value else "#7A8398")
        self.coords(self._knob, *( (self.W - self.H + 3, 3, self.W - 3, self.H - 3)
                                   if self.value else (3, 3, self.H - 3, self.H - 3) ))

    def _click(self, _event=None) -> None:
        self.value = not self.value
        self._paint()
        if self.command:
            self.command(self.value)

    def set(self, value: bool, notify: bool = False) -> None:
        self.value = bool(value)
        self._paint()
        if notify and self.command:
            self.command(self.value)


class Slider(tk.Canvas):
    """A slider that shows its own value, with a click-to-type readout."""

    def __init__(self, master: tk.Misc, field: dict, on_change, width: int = 250,
                 decimals: int | None = None, unit: str = ""):
        self.fmin = float(field.get("min", 0.0))
        self.fmax = float(field.get("max", 1.0))
        self.step = float(field.get("step", 0.0) or 0.0)
        self.unit = unit or field.get("unit", "")
        if decimals is None:
            decimals = 0 if float(field.get("default", 0)).is_integer() else 1
        self.decimals = decimals
        self.on_change = on_change
        self.value = float(field.get("default", self.fmin))

        super().__init__(master, width=width, height=34, bg=ROW, highlightthickness=0, bd=0)
        self.width = width
        self._track = rounded(self, 0, 14, width, 20, 3, fill=PANEL_2, outline="")
        self._fill = rounded(self, 0, 14, width, 20, 3, fill=CYAN, outline="")
        self._knob = self.create_oval(0, 0, 0, 0, fill=BRIGHT, outline=CYAN, width=2)
        self._text = self.create_text(width, 8, text="", anchor="e", fill=CYAN,
                                      font=(FONT, 10, "bold"))
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._drag)
        self._paint()

    # -- geometry ---------------------------------------------------------- #
    def _fraction(self) -> float:
        span = self.fmax - self.fmin
        return 0.0 if span <= 0 else (self.value - self.fmin) / span

    def _paint(self) -> None:
        fraction = max(0.0, min(1.0, self._fraction()))
        fill_x = 8 + (self.width - 16) * fraction
        self.coords(self._fill, *self._rounded_points(0, 14, fill_x, 20, 3))
        self.coords(self._knob, fill_x - 6, 11, fill_x + 6, 23)
        self.itemconfig(self._text, text=self.label())

    @staticmethod
    def _rounded_points(x1, y1, x2, y2, radius) -> list[float]:
        radius = max(0.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
        return [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]

    def label(self) -> str:
        if self.decimals <= 0:
            number = f"{self.value:.0f}"
        else:
            number = f"{self.value:.{self.decimals}f}"
        return f"{number}{self.unit}"

    # -- interaction ------------------------------------------------------- #
    def _from_x(self, x: float) -> float:
        fraction = max(0.0, min(1.0, (float(x) - 8.0) / max(1.0, self.width - 16.0)))
        value = self.fmin + fraction * (self.fmax - self.fmin)
        if self.step > 0:
            value = round(value / self.step) * self.step
        return max(self.fmin, min(self.fmax, value))

    def _commit(self, value: float) -> None:
        if abs(value - self.value) < 1e-9:
            return
        self.value = value
        self._paint()
        if self.on_change:
            self.on_change(self.value)

    def _click(self, event) -> None:
        self._commit(self._from_x(event.x))

    def _drag(self, event) -> None:
        self._commit(self._from_x(event.x))

    def set(self, value: float, notify: bool = False) -> None:
        try:
            value = max(self.fmin, min(self.fmax, float(value)))
        except (TypeError, ValueError):
            return
        self.value = value
        self._paint()
        if notify and self.on_change:
            self.on_change(self.value)


def pill_button(master: tk.Misc, text: str, command, kind: str = "ghost",
                width: int | None = None) -> tk.Button:
    """A flat button in one of the design's three styles."""
    styles = {
        "primary": (CYAN, BRIGHT, "#0891B2"),
        "ghost": (PANEL_2, TEXT, "#232C42"),
        "danger": ("#3A1519", "#FCA5A5", "#4C1B20"),
        "quiet": (PANEL, DIM, PANEL_2),
        "indigo": (INDIGO, BRIGHT, "#4F46E5"),
    }
    bg, fg, active = styles.get(kind, styles["ghost"])
    button = tk.Button(
        master, text=text, command=command, bg=bg, fg=fg, activebackground=active,
        activeforeground=fg, relief="flat", bd=0, highlightthickness=0,
        font=(FONT, 9, "bold" if kind in ("primary", "indigo") else "normal"),
        padx=12, pady=6, cursor="hand2",
    )
    if width:
        button.configure(width=width)
    return button


def dark_combo(master: tk.Misc, values: list[str], width: int = 28) -> ttk.Combobox:
    box = ttk.Combobox(master, values=values, state="readonly", width=width,
                       style="Rixers.TCombobox", font=(FONT, 9))
    return box


def dark_entry(master: tk.Misc, width: int = 26, textvariable=None) -> tk.Entry:
    return tk.Entry(
        master, width=width, textvariable=textvariable, bg=SUNKEN, fg=TEXT,
        insertbackground=CYAN, relief="flat", bd=0, highlightthickness=1,
        highlightbackground=EDGE, highlightcolor=CYAN, font=(FONT, 9),
    )


# --------------------------------------------------------------------------- #
# The settings bridge: one place that knows how a value is read, validated and
# written, so no page has to.
# --------------------------------------------------------------------------- #
class SettingsBridge:
    """Pushes edits at the settings file and keeps the window in step with it."""

    def __init__(self, path: Path) -> None:
        self.live = live_settings.LiveSettings(path)
        self.live.load(force=True)
        self.draft: dict = dict(self.live.values)
        self.dirty = False
        self.on_live_write = None
        self.on_dirty = None
        self.fields = self._field_specs()

    @staticmethod
    def _field_specs() -> dict:
        """Slider fields for the numeric settings, with the labels they deserve."""
        return {
            "pause_flush_s": {"min": 0.1, "max": 5.0, "step": 0.05, "default": 0.7,
                              "unit": " s", "decimals": 2},
            "word_gap_min_ms": {"min": 0.0, "max": 400.0, "step": 5.0, "default": 80.0,
                                "unit": " ms", "decimals": 0},
            "word_gap_max_ms": {"min": 0.0, "max": 5000.0, "step": 50.0, "default": 1500.0,
                                "unit": " ms", "decimals": 0},
            "queue": {"min": 1, "max": 64, "step": 1, "default": 12, "unit": "", "decimals": 0},
            "backlog": {"min": 1, "max": 128, "step": 1, "default": 16, "unit": "", "decimals": 0},
            "gap_ms": {"min": 0.0, "max": 400.0, "step": 5.0, "default": 0.0,
                       "unit": " ms", "decimals": 0},
            "trim_silence_ms": {"min": 0.0, "max": 600.0, "step": 10.0, "default": 220.0,
                                "unit": " ms", "decimals": 0},
            "target_words": {"min": 0, "max": 40, "step": 1, "default": 0, "unit": " words",
                             "decimals": 0},
            "max_words": {"min": 0, "max": 80, "step": 1, "default": 0, "unit": " words",
                          "decimals": 0},
            "retries": {"min": 0, "max": 5, "step": 1, "default": 1, "unit": "", "decimals": 0},
            "pitch_semitones": {"min": -12.0, "max": 12.0, "step": 0.5, "default": 0.0,
                                "unit": " st", "decimals": 1},
            "volume": {"min": 0.0, "max": 4.0, "step": 0.05, "default": 1.0,
                       "unit": "x", "decimals": 2},
            "gate_db": {"min": -70.0, "max": 0.0, "step": 1.0, "default": -70.0,
                        "unit": " dB", "decimals": 0},
            "gate_floor_db": {"min": -70.0, "max": 0.0, "step": 1.0, "default": -60.0,
                              "unit": " dB", "decimals": 0},
            "min_chars": {"min": 1, "max": 60, "step": 1, "default": 1, "unit": " chars",
                          "decimals": 0},
        }

    # -- reading ----------------------------------------------------------- #
    def get(self, name: str):
        return self.draft.get(name, live_settings.FIELDS.get(name, {}).get("default"))

    def is_dirty(self) -> bool:
        return self.draft != self.live.values

    # -- writing ----------------------------------------------------------- #
    def set(self, name: str, value, live: bool = False) -> None:
        """Take on a value, and (for live settings) tell the engine straight away."""
        coerced = live_settings.coerce(name, value)
        if coerced == self.get(name):
            return
        self.draft[name] = coerced
        if live:
            self.live.update(**{name: coerced})
            try:
                self.live.save()
            except OSError:
                pass
            if self.on_live_write:
                self.on_live_write(name, coerced)
        if self.on_dirty:
            self.on_dirty()

    def reload(self) -> None:
        self.live.load(force=True)
        self.draft = dict(self.live.values)
        if self.on_dirty:
            self.on_dirty()

    def save(self) -> Path:
        self.live.update(**self.draft)
        path = self.live.save()
        self.draft = dict(self.live.values)
        if self.on_dirty:
            self.on_dirty()
        return path

    def reset(self, names) -> None:
        for name in names:
            self.draft[name] = live_settings.FIELDS[name]["default"]
        if self.on_dirty:
            self.on_dirty()


# --------------------------------------------------------------------------- #
# The engine: the voice changer process the Start/Stop buttons own.
# --------------------------------------------------------------------------- #
class Engine:
    """Runs `voice_changer.py` as a child process and collects its output."""

    def __init__(self, log) -> None:
        self.log = log
        self.process: subprocess.Popen | None = None
        self.started_at = 0.0
        self._reader: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, extra_args: list[str] | None = None) -> tuple[bool, str]:
        if self.running:
            return False, "the voice changer is already running"
        python = PYTHON if PYTHON.exists() else Path(sys.executable)
        command = [str(python), "-u", str(HERE / "voice_changer.py")]
        if extra_args:
            command += list(extra_args)
        self.log(f"$ {' '.join(command[1:])}\n")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.process = None
            return False, f"could not start the voice changer: {exc}"
        self.started_at = time.time()
        self._reader = threading.Thread(target=self._pump, name="engine-log", daemon=True)
        self._reader.start()
        return True, "starting"

    def _pump(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self.log(line.rstrip("\n"))
        except (OSError, ValueError):
            pass
        finally:
            code = process.poll()
            self.log(f"-- the voice changer exited (code {code})")

    def stop(self, timeout: float = 6.0) -> bool:
        process = self.process
        if process is None:
            return False
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                except OSError:
                    pass
        self.process = None
        return True

    def uptime(self) -> float:
        return time.time() - self.started_at if self.running else 0.0


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
class ControlPanel:
    def __init__(self, settings_path: Path | None = None) -> None:
        self.bridge = SettingsBridge(settings_path or live_settings.settings_path())
        self.voices_dir = voices_mod.VOICES_DIR
        self.selector = self.voices_dir / "voice.txt"
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.status_queue: "queue.Queue[dict]" = queue.Queue()
        self.device_queue: "queue.Queue[tuple[list[str], list[str]]]" = queue.Queue()
        self.engine = Engine(self._enqueue_log)
        self.voices: list[voices_mod.Voice] = []
        self.widgets: dict[str, list] = {}
        self.nav_buttons: dict[str, tk.Frame] = {}
        self.pages: dict[str, tk.Frame] = {}
        self.current_page = PAGES[0][0]
        self.devices: dict[str, list[str]] = {"input": [], "output": []}
        self._device_busy = False
        self._status_busy = False
        self._toast_job = None
        self._log_lines = 0
        self._last_status: dict = {}
        self._closing = False
        """Set when the window is going away.

        The polling loops reschedule themselves with `after`, and worker threads post
        their results to queues this thread drains. Both have to stop *before*
        `destroy()`: a worker that touches the interpreter after it is gone does not
        raise a Python error, it takes the process down with an access violation
        (0xC0000005). That is exactly how this was found -- the GUI test passed four
        times and then killed its own interpreter on the fifth.
        """

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} \u2014 Settings")
        self.root.configure(bg=BG)
        self.root.minsize(1120, 720)
        self._centre(1320, 860)
        self._build_fonts()
        self._build_styles()
        self._build_layout()

        self.bridge.on_dirty = self._refresh_dirty_state
        self.bridge.on_live_write = self._live_note
        # Write the file straight away if it is not there yet. The voice changer
        # reads it at start-up, so leaving it absent until the first Save would mean
        # a first run silently used the command-line defaults.
        if not self.bridge.live.path.exists():
            try:
                self.bridge.save()
            except OSError:
                pass

        # The pages own the voice picker and the library table, so the voices are
        # read *after* they exist -- calling this first found no picker to fill and
        # quietly left every voice list empty.
        self.build_pages()
        self.reload_voices()
        self.show_page(self.current_page)
        # The user has changed nothing yet; without this the panel opens claiming
        # there are unsaved changes, because reading the active voice writes to the
        # draft.
        self.bridge.live.values = dict(self.bridge.draft)
        self._refresh_dirty_state()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.after(220, self._drain_log)
        self.root.after(400, self._poll_status)
        self.root.after(900, self._refresh_devices)
        self.root.after(1400, self._apply_startup_voice)

    # -- window furniture -------------------------------------------------- #
    def _centre(self, width: int, height: int) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(width, screen_w - 40)
        height = min(height, screen_h - 80)
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_fonts(self) -> None:
        self.font_body = tkfont.Font(family=FONT, size=9)
        self.font_small = tkfont.Font(family=FONT, size=8)
        self.font_bold = tkfont.Font(family=FONT, size=9, weight="bold")
        self.font_head = tkfont.Font(family=FONT, size=15, weight="bold")
        self.font_sub = tkfont.Font(family=FONT, size=9)
        self.font_card = tkfont.Font(family=FONT, size=11, weight="bold")
        self.font_mono = tkfont.Font(family="Consolas", size=8)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Rixers.TCombobox",
                        fieldbackground=SUNKEN, background=PANEL_2, foreground=TEXT,
                        arrowcolor=CYAN, bordercolor=PANEL_2, lightcolor=PANEL_2,
                        darkcolor=PANEL_2, borderwidth=0, padding=6, relief="flat")
        style.map("Rixers.TCombobox",
                  fieldbackground=[("readonly", SUNKEN)],
                  foreground=[("readonly", TEXT)],
                  background=[("readonly", PANEL_2)])
        self.root.option_add("*TCombobox*Listbox.background", PANEL_2)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", CYAN)
        self.root.option_add("*TCombobox*Listbox.selectForeground", BG)
        style.configure("Rixers.Vertical.TScrollbar",
                        background=PANEL_2, troughcolor=BG, bordercolor=BG,
                        arrowcolor=DIM, relief="flat", width=10)
        style.map("Rixers.Vertical.TScrollbar", background=[("active", EDGE)])
        style.configure("Treeview", background=SUNKEN, fieldbackground=SUNKEN, foreground=TEXT,
                        bordercolor=EDGE, rowheight=24, font=self.font_small)
        style.configure("Treeview.Heading", background=PANEL_2, foreground=DIM,
                        font=self.font_small, relief="flat", padding=4)
        style.map("Treeview", background=[("selected", "#123A47")],
                  foreground=[("selected", BRIGHT)])

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self._build_sidebar()
        self._build_main()

    # -- sidebar ----------------------------------------------------------- #
    def _build_sidebar(self) -> None:
        bar = tk.Frame(self.root, bg=PANEL, width=SIDEBAR_W)
        bar.grid(row=0, column=0, sticky="nsw")
        bar.grid_propagate(False)
        tk.Frame(bar, bg=EDGE_SOFT, width=1).pack(side="right", fill="y")

        brand = tk.Frame(bar, bg=PANEL, padx=20, pady=20)
        brand.pack(fill="x")
        logo = tk.Canvas(brand, width=40, height=40, bg=PANEL, highlightthickness=0)
        logo.pack(side="left")
        rounded(logo, 1, 1, 39, 39, 12, fill=INDIGO, outline="")
        logo.create_text(20, 21, text="RV", fill=BRIGHT, font=(FONT, 11, "bold"))
        titles = tk.Frame(brand, bg=PANEL)
        titles.pack(side="left", padx=10)
        tk.Label(titles, text="Rixers", bg=PANEL, fg=BRIGHT,
                 font=(FONT, 12, "bold")).pack(anchor="w")
        tk.Label(titles, text="Custom Voice Changer", bg=PANEL, fg=CYAN,
                 font=(FONT, 8)).pack(anchor="w")
        tk.Label(titles, text=TAGLINE, bg=PANEL, fg=FAINT,
                 font=(FONT, 7)).pack(anchor="w")

        nav = tk.Frame(bar, bg=PANEL, padx=12)
        nav.pack(fill="both", expand=True)
        for key, label in PAGES:
            self.nav_buttons[key] = self._nav_button(nav, key, label)

        footer = tk.Frame(bar, bg=PANEL, padx=16, pady=14)
        footer.pack(side="bottom", fill="x")
        tk.Frame(footer, bg=EDGE_SOFT, height=1).pack(fill="x", pady=(0, 12))
        line = tk.Frame(footer, bg=PANEL)
        line.pack(fill="x")
        badge = tk.Canvas(line, width=30, height=30, bg=PANEL, highlightthickness=0)
        badge.pack(side="left")
        rounded(badge, 1, 1, 29, 29, 15, fill="#0F3B33", outline="")
        badge.create_text(15, 16, text="RC", fill=OK, font=(FONT, 8, "bold"))
        info = tk.Frame(line, bg=PANEL)
        info.pack(side="left", padx=8)
        tk.Label(info, text=PROFILE_NAME, bg=PANEL, fg=BRIGHT,
                 font=(FONT, 8, "bold")).pack(anchor="w")
        tk.Label(info, text="\u25CF Local \u00b7 Offline", bg=PANEL, fg=OK,
                 font=(FONT, 7)).pack(anchor="w")
        tk.Label(line, text=VERSION, bg=PANEL_2, fg=DIM, font=(FONT, 7),
                 padx=6, pady=2).pack(side="right")
        tk.Label(footer, text=f"{APP_NAME} \u2022 Settings Core", bg=PANEL, fg=FAINT,
                 font=(FONT, 7)).pack(pady=(10, 0))

    def _nav_button(self, parent: tk.Misc, key: str, label: str) -> tk.Frame:
        """One sidebar entry: a coloured marker, a bullet, and the label.

        Built from frames and labels rather than a canvas item so the row is exactly
        as tall as its text -- a canvas reports a height of 1 until something fills
        it, which is what made the first version's marker a dot in the corner.
        """
        holder = tk.Frame(parent, bg=PANEL, cursor="hand2")
        holder.pack(fill="x", pady=1)
        marker = tk.Frame(holder, bg=PANEL, width=3)
        marker.pack(side="left", fill="y")
        body = tk.Frame(holder, bg=PANEL, padx=10, pady=9)
        body.pack(side="left", fill="both", expand=True)
        dot = tk.Label(body, text="\u25CF", bg=PANEL, fg=DIM, font=(FONT, 7))
        dot.pack(side="left", padx=(0, 10))
        text = tk.Label(body, text=label, bg=PANEL, fg=DIM, font=(FONT, 9))
        text.pack(side="left")

        holder.marker = marker  # type: ignore[attr-defined]
        holder.dot = dot  # type: ignore[attr-defined]
        holder.text = text  # type: ignore[attr-defined]
        holder.body = body  # type: ignore[attr-defined]
        holder.accent = INDIGO if key in ("voices", "effects", "advanced") else CYAN
        for widget in (holder, body, dot, text, marker):
            widget.bind("<Button-1>", lambda _e, k=key: self.show_page(k))
            widget.bind("<Enter>", lambda _e, k=key: self._nav_hover(k, True))
            widget.bind("<Leave>", lambda _e, k=key: self._nav_hover(k, False))
        return holder

    def _nav_hover(self, key: str, entering: bool) -> None:
        if key == self.current_page:
            return
        holder = self.nav_buttons[key]
        holder.text.configure(fg=TEXT if entering else DIM)  # type: ignore[attr-defined]
        holder.dot.configure(fg=CYAN if entering else DIM)  # type: ignore[attr-defined]

    def _paint_nav(self) -> None:
        for key, holder in self.nav_buttons.items():
            active = key == self.current_page
            accent = holder.accent  # type: ignore[attr-defined]
            background = PANEL_2 if active else PANEL
            for widget in (holder, holder.body, holder.dot, holder.text):  # type: ignore[attr-defined]
                widget.configure(bg=background)
            holder.marker.configure(bg=accent if active else background)  # type: ignore[attr-defined]
            holder.text.configure(fg=accent if active else DIM)  # type: ignore[attr-defined]
            holder.dot.configure(fg=accent if active else DIM)  # type: ignore[attr-defined]

    # -- main area --------------------------------------------------------- #
    def _build_main(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        header = tk.Frame(main, bg=PANEL, height=72)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        left = tk.Frame(header, bg=PANEL)
        left.pack(side="left", padx=24, pady=14)
        self.header_title = tk.Label(left, text="", bg=PANEL, fg=BRIGHT, font=self.font_head)
        self.header_title.pack(anchor="w")
        self.header_sub = tk.Label(left, text="", bg=PANEL, fg=DIM, font=self.font_sub)
        self.header_sub.pack(anchor="w")

        actions = tk.Frame(header, bg=PANEL)
        actions.pack(side="right", padx=20)
        self.save_button = pill_button(actions, "\u2713  Save Changes", self.save_all, "primary")
        self.save_button.pack(side="right", padx=(8, 0))
        pill_button(actions, "\u21BA  Reset Defaults", self.reset_page, "ghost").pack(side="right")
        self.dirty_label = tk.Label(actions, text="", bg=PANEL, fg=WARN, font=self.font_small)
        self.dirty_label.pack(side="right", padx=12)

        tk.Frame(main, bg=EDGE_SOFT, height=1).grid(row=0, column=0, sticky="sew")

        body = tk.Frame(main, bg=BG)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview,
                               style="Rixers.Vertical.TScrollbar")
        scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.content = tk.Frame(self.canvas, bg=BG)
        self._content_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._sync_scroll)
        self.canvas.bind("<Configure>", self._sync_scroll)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

        self.toast = tk.Frame(self.root, bg="#10322A", highlightthickness=1,
                              highlightbackground="#1F6F5C")
        self.toast_label = tk.Label(self.toast, text="", bg="#10322A", fg="#A7F3D0",
                                    font=self.font_small, justify="left")
        self.toast_label.pack(padx=14, pady=10)

    def _sync_scroll(self, event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfigure(self._content_id, width=self.canvas.winfo_width())

    def _on_wheel(self, event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if isinstance(widget, tk.Text):
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # -- pages ------------------------------------------------------------- #
    def build_pages(self) -> None:
        """Build every page inside one holder, showing one at a time.

        Only the visible page is managed: the others are unpacked, so they cost no
        space and no redraw. (Gridding them all into one cell and raising the active
        one looks equivalent but is not -- every page stays mapped, the cell takes
        the height of the *tallest* page, and the shorter pages get squeezed. That
        is what produced a page whose cards were clipped and blank.)
        """
        holder = tk.Frame(self.content, bg=BG)
        holder.pack(fill="both", expand=True)
        self.page_holder = holder
        builders = {
            "dashboard": self._page_dashboard,
            "voices": self._page_voices,
            "timing": self._page_timing,
            "engine": self._page_engine,
            "audio": self._page_audio,
            "effects": self._page_effects,
            "input": self._page_input,
            "advanced": self._page_advanced,
        }
        for key, _label in PAGES:
            page = tk.Frame(holder, bg=BG)
            self.pages[key] = page
            builders[key](page)

    def show_page(self, key: str) -> None:
        self.current_page = key
        for name, page in self.pages.items():
            if name == key:
                page.pack(fill="both", expand=True, padx=28, pady=(18, 40))
            else:
                page.pack_forget()
        titles = {
            "dashboard": ("Dashboard", "Everything at a glance, and the Start / Stop buttons"),
            "voices": ("Voices", "Clone a voice from a 3-16 second wav, or map one to each emotion"),
            "timing": ("Timing & Rhythm", "How long to wait after you stop speaking, and your own pace"),
            "engine": ("Speech Engine", "How much text each synthesis call gets, and how far ahead it runs"),
            "audio": ("Audio & Devices", "Which microphone is transcribed, and where the cloned voice goes"),
            "effects": ("Effects", "Pitch, volume and the noise gate, applied to the finished voice"),
            "input": ("Transcription", "The microphone -> text stage that feeds the voice changer"),
            "advanced": ("Advanced", "Raw settings, the paths in use, and the last-resort buttons"),
        }
        title, subtitle = titles[key]
        self.header_title.configure(text=title)
        self.header_sub.configure(text=subtitle)
        self._paint_nav()
        self.canvas.yview_moveto(0.0)
        self._refresh_dirty_state()

    # -- generic widgets --------------------------------------------------- #
    def _register(self, name: str, widget) -> None:
        self.widgets.setdefault(name, []).append(widget)
        setter = getattr(widget, "set", None)
        if callable(setter):
            try:
                setter(self.bridge.get(name))
            except Exception:  # noqa: BLE001 - a widget that cannot show a value is a bug
                pass

    def add_toggle(self, parent: tk.Misc, name: str, title: str, note: str,
                   live: bool = True, command=None) -> Toggle:
        content = accent_row(parent, CYAN)
        left = tk.Frame(content, bg=ROW)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=title, bg=ROW, fg=BRIGHT, font=self.font_bold).pack(anchor="w")
        if note:
            tk.Label(left, text=note, bg=ROW, fg=FAINT, font=self.font_small,
                     justify="left", wraplength=620, anchor="w").pack(anchor="w", pady=(2, 0))

        def changed(value: bool) -> None:
            self.bridge.set(name, value, live=live)
            if command:
                command(value)

        toggle = Toggle(content, value=bool(self.bridge.get(name)), command=changed)
        toggle.pack(side="right", padx=(12, 4))
        self._register(name, toggle)
        return toggle

    def add_slider(self, parent: tk.Misc, name: str, title: str, note: str,
                   live: bool = True, accent: str = CYAN, decimals: int | None = None,
                   command=None) -> Slider:
        content = accent_row(parent, accent)
        tk.Label(content, text=title, bg=ROW, fg=BRIGHT, font=self.font_bold).pack(anchor="w")
        if note:
            tk.Label(content, text=note, bg=ROW, fg=FAINT, font=self.font_small,
                     justify="left", wraplength=700, anchor="w").pack(anchor="w", pady=(2, 0))

        def changed(value: float) -> None:
            self.bridge.set(name, value, live=live)
            if command:
                command(value)

        spec = dict(self.bridge.fields.get(name, {}))
        spec["default"] = self.bridge.get(name)
        slider = Slider(content, spec, changed, width=300, decimals=decimals)
        slider.pack(anchor="w", pady=(8, 0), fill="x")
        self._register(name, slider)
        return slider

    def add_choice(self, parent: tk.Misc, name: str, title: str, note: str,
                   options: list[tuple[str, str]], live: bool = True,
                   accent: str = INDIGO, command=None) -> ttk.Combobox:
        content = accent_row(parent, accent)
        tk.Label(content, text=title, bg=ROW, fg=BRIGHT, font=self.font_bold).pack(anchor="w")
        if note:
            tk.Label(content, text=note, bg=ROW, fg=FAINT, font=self.font_small,
                     justify="left", wraplength=700, anchor="w").pack(anchor="w", pady=(2, 0))
        labels = [label for _value, label in options]
        lookup = {label: value for value, label in options}
        current = str(self.bridge.get(name))
        label_now = next((lab for val, lab in options if val == current), labels[0])

        def picked(_event=None) -> None:
            box.selection_clear()
            self.bridge.set(name, lookup[box.get()], live=live)
            if command:
                command(lookup[box.get()])

        box = dark_combo(content, labels, width=44)
        box.set(label_now)
        box.pack(anchor="w", pady=(8, 0))
        box.bind("<<ComboboxSelected>>", picked)
        box.lookup = lookup  # type: ignore[attr-defined]
        box.labels = labels  # type: ignore[attr-defined]
        self._register(name, box)
        return box

    def add_text(self, parent: tk.Misc, name: str, title: str, note: str,
                 live: bool = False, accent: str = INDIGO, width: int = 60) -> tk.Entry:
        content = accent_row(parent, accent)
        tk.Label(content, text=title, bg=ROW, fg=BRIGHT, font=self.font_bold).pack(anchor="w")
        if note:
            tk.Label(content, text=note, bg=ROW, fg=FAINT, font=self.font_small,
                     justify="left", wraplength=700, anchor="w").pack(anchor="w", pady=(2, 0))
        variable = tk.StringVar(value=str(self.bridge.get(name) or ""))
        entry = dark_entry(content, width=width, textvariable=variable)
        entry.pack(anchor="w", pady=(8, 0), ipady=6)

        def commit(_event=None) -> None:
            self.bridge.set(name, variable.get(), live=live)

        entry.bind("<FocusOut>", commit)
        entry.bind("<Return>", commit)
        self._register(name, entry)
        entry.variable = variable  # type: ignore[attr-defined]
        return entry

    # -- pages: each one a handful of cards -------------------------------- #
    def _page_dashboard(self, page: tk.Frame) -> None:
        hero = Card(page, "Status", "What the voice changer is doing right now.")
        hero.pack(fill="x", pady=(0, 14))
        grid = tk.Frame(hero.rows, bg=PANEL)
        grid.pack(fill="x", padx=6, pady=2)
        self.status_boxes: dict[str, tk.Label] = {}
        for index, (key, label, note) in enumerate([
            ("engine", "Voice changer", "the process that speaks into the cable"),
            ("tts", "Speech engine", "StyleTTS 2 on 127.0.0.1:9881"),
            ("stt", "Transcription", "the microphone -> text stage"),
            ("voice", "Active voice", "the reference being cloned"),
        ]):
            cell = tk.Frame(grid, bg=ROW, padx=14, pady=12)
            cell.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
            grid.grid_columnconfigure(index % 2, weight=1)
            tk.Label(cell, text=label, bg=ROW, fg=FAINT, font=self.font_small).pack(anchor="w")
            value = tk.Label(cell, text="checking ...", bg=ROW, fg=TEXT, font=self.font_bold)
            value.pack(anchor="w", pady=(3, 0))
            tk.Label(cell, text=note, bg=ROW, fg=FAINT, font=self.font_small).pack(anchor="w")
            self.status_boxes[key] = value

        run = Card(page, "Run", "Start the voice changer, or stop it and free the microphone.")
        run.pack(fill="x", pady=(0, 14))
        buttons = tk.Frame(run.rows, bg=PANEL)
        buttons.pack(fill="x", padx=6, pady=4)
        self.start_button = pill_button(buttons, "\u25B6  Start Voice Changer",
                                        self.start_engine, "primary")
        self.start_button.pack(side="left")
        self.stop_button = pill_button(buttons, "\u25A0  Stop", self.stop_engine, "ghost")
        self.stop_button.pack(side="left", padx=8)
        pill_button(buttons, "Test: speak a line", self.test_line, "ghost").pack(side="left", padx=8)
        pill_button(buttons, "Speech engine ++", self.start_speech_engine, "quiet").pack(side="left", padx=8)
        pill_button(buttons, "Stop everything", self.stop_everything, "danger").pack(side="right")

        log_card = Card(page, "Live output", "Everything the voice changer prints, as it happens.", accent=INDIGO)
        log_card.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_card.rows, height=16, bg=SUNKEN, fg="#B9C4D6", insertbackground=CYAN,
            relief="flat", bd=0, highlightthickness=1, highlightbackground=EDGE,
            font=self.font_mono, wrap="none", padx=10, pady=8,
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=4)
        self.log_text.configure(state="disabled")
        actions = tk.Frame(log_card.rows, bg=PANEL)
        actions.pack(fill="x", padx=6, pady=(4, 2))
        pill_button(actions, "Clear", self._clear_log, "quiet").pack(side="right")

    def _page_voices(self, page: tk.Frame) -> None:
        card = Card(page, "Active voice", "The voice every new sentence is spoken in.", accent=INDIGO)
        card.pack(fill="x", pady=(0, 14))
        content = tk.Frame(card.rows, bg=PANEL)
        content.pack(fill="x", padx=6, pady=4)
        tk.Label(content, text="Voice to clone", bg=PANEL, fg=BRIGHT,
                 font=self.font_bold).pack(anchor="w")
        self.voice_box = dark_combo(content, [], width=52)
        self.voice_box.pack(anchor="w", pady=(6, 0))
        self.voice_box.bind("<<ComboboxSelected>>", lambda _e: self._voice_sel_changed())
        self.voice_note = tk.Label(content, text="", bg=PANEL, fg=FAINT, font=self.font_small,
                                   justify="left", wraplength=760, anchor="w")
        self.voice_note.pack(anchor="w", pady=(6, 8))
        buttons = tk.Frame(content, bg=PANEL)
        buttons.pack(anchor="w")
        pill_button(buttons, "Use this voice now", self.apply_voice, "primary").pack(side="left")
        pill_button(buttons, "Hear a preview", self.preview_voice, "ghost").pack(side="left", padx=8)
        pill_button(buttons, "Import a wav ...", self.import_voice, "ghost").pack(side="left", padx=8)
        pill_button(buttons, "Reload list", self.reload_voices, "quiet").pack(side="left", padx=8)
        pill_button(buttons, "Open voices folder", self.open_voices_folder, "quiet").pack(side="left", padx=8)
        tk.Label(content, text="A voice is just a wav: 3-16 seconds of clean speech of the voice "
                               "you want. Anything shorter or longer is refused by the engine.",
                 bg=PANEL, fg=FAINT, font=self.font_small, justify="left",
                 wraplength=760, anchor="w").pack(anchor="w", pady=(10, 0))

        table = Card(page, "Voice library", "Every wav in the voices folder, with what the engine "
                                            "makes of it.", accent=INDIGO)
        table.pack(fill="both", expand=True, pady=(0, 14))
        self.voice_tree = ttk.Treeview(table.rows, columns=("time", "rate", "state", "name"),
                                       show="headings", height=7)
        for column, label, width in (("time", "Length", 90), ("rate", "Rate", 90),
                                     ("state", "State", 190), ("name", "Name", 260)):
            self.voice_tree.heading(column, text=label)
            self.voice_tree.column(column, width=width, anchor="w")
        self.voice_tree.pack(fill="both", expand=True, padx=6, pady=4)
        self.voice_tree.bind("<Double-Button-1>", lambda _e: self.apply_voice())

        emotions = Card(page, "Emotion voices", "How you sounded picks the reference, so a happy "
                                               "delivery uses a happy voice.", accent=INDIGO)
        emotions.pack(fill="x")
        self.emotion_boxes: dict[str, ttk.Combobox] = {}
        holder = tk.Frame(emotions.rows, bg=PANEL)
        holder.pack(fill="x", padx=6, pady=4)
        choices = ["(use the active voice)"] + [v.slug for v in self.voices if v.usable]
        for index, name in enumerate(emotion_mod.EMOTIONS):
            cell = tk.Frame(holder, bg=PANEL)
            cell.grid(row=index // 2, column=index % 2, sticky="ew", padx=6, pady=5)
            holder.grid_columnconfigure(index % 2, weight=1)
            tk.Label(cell, text=name.title(), bg=PANEL, fg=TEXT,
                     font=self.font_small).pack(anchor="w")
            box = dark_combo(cell, choices, width=30)
            box.pack(anchor="w")
            box.bind("<<ComboboxSelected>>", lambda _e, n=name: self.save_emotion(n))
            self.emotion_boxes[name] = box
        tk.Label(emotions.rows, text="Point one emotion at a voice and it applies on the next "
                                     "sentence. Leave the rest alone and the active voice is used.",
                 bg=PANEL, fg=FAINT, font=self.font_small, justify="left",
                 wraplength=760, anchor="w").pack(anchor="w", padx=12, pady=(6, 8))

    def _page_timing(self, page: tk.Frame) -> None:
        card = Card(page, "Sentence timing", "How long the voice changer waits after you stop "
                                             "speaking before it speaks that sentence.")
        card.pack(fill="x", pady=(0, 14))
        self.add_slider(card.rows, "pause_flush_s", "Wait after you stop speaking",
                        "Sets the delay before a sentence is spoken. Lower means it answers "
                        "sooner and cuts you off more often; higher means a longer wait for "
                        "the sentence to be committed. 0.7 s is the default.",
                        live=True, decimals=2)
        self.add_slider(card.rows, "min_chars", "Shortest fragment worth speaking",
                        "Anything shorter than this many characters is dropped instead of "
                        "being synthesized. 1 means never drop anything.",
                        live=True, decimals=0)

        rhythm = Card(page, "Your own rhythm", "How much of the timing between your words is "
                                               "put back when the clone speaks.", accent=INDIGO)
        rhythm.pack(fill="x", pady=(0, 14))
        self.add_choice(rhythm.rows, "word_delay", "Replay the delays between your words",
                        "auto follows the browser page's Word delays button, on always replays "
                        "them, off always strips them. Every replayed delay costs one more "
                        "synthesis call, so turning this off is the lowest-latency setting.",
                        options=[("auto", "Auto - follow the page"),
                                 ("on", "On - always replay my pace"),
                                 ("off", "Off - use the model's own pace")],
                        live=True)
        self.add_slider(rhythm.rows, "word_gap_min_ms", "Shortest delay worth replaying",
                        "Gaps shorter than this are absorbed into the words around them. This is "
                        "the latency dial: replaying a 30 ms gap between every word turns one "
                        "synthesis call into eight.", live=True, accent=INDIGO, decimals=0)
        self.add_slider(rhythm.rows, "word_gap_max_ms", "Longest delay to replay",
                        "A hesitation longer than this is capped, so a hesitant speaker does not "
                        "make the clone equally hesitant.", live=True, accent=INDIGO, decimals=0)

        flow = Card(page, "Backlog", "What happens when the model cannot keep up with you.")
        flow.pack(fill="x")
        self.add_slider(flow.rows, "queue", "Finished audio kept ready to play",
                        "Chunks of synthesized speech waiting to be written to the cable. More "
                        "means it survives a slow sentence without a hole, at the cost of "
                        "reacting later to a voice change.", live=True)
        self.add_slider(flow.rows, "backlog", "Total queued work before the oldest is dropped",
                        "Bounds how far behind the voice is allowed to fall. When it is exceeded "
                        "the oldest waiting text is dropped, so the voice stays current instead "
                        "of drifting behind what you said.", live=True)

    def _page_engine(self, page: tk.Frame) -> None:
        card = Card(page, "How much text per synthesis call", "The single biggest trade-off in "
                                                              "the pipeline: latency against "
                                                              "naturalness.")
        card.pack(fill="x", pady=(0, 14))
        self.add_choice(card.rows, "mode", "Chunking mode",
                        "word follows your speech most closely and sounds choppiest; sentence "
                        "waits for punctuation and sounds most natural; chunk and phrase are in "
                        "between. Sentence is the default and what the pipeline is tuned for.",
                        options=[("word", "Word - one or two words per call (lowest latency)"),
                                 ("chunk", "Chunk - about four words per call"),
                                 ("phrase", "Phrase - about eight words per call"),
                                 ("sentence", "Sentence - wait for punctuation (most natural)")],
                        live=True)
        self.add_slider(card.rows, "target_words", "Words per call (0 = from the mode)",
                        "Overrides the mode's own release size. 0 leaves it alone.", live=True)
        self.add_slider(card.rows, "max_words", "Hard cap before a chunk is released",
                        "Speech recognition often returns text with no punctuation at all. After "
                        "this many words the text is spoken anyway, so the sentence break is "
                        "missed rather than the audio being lost. 0 = from the mode.", live=True)

        polish = Card(page, "Pacing and retries", "Small adjustments to the gaps between chunks.",
                      accent=INDIGO)
        polish.pack(fill="x", pady=(0, 14))
        self.add_slider(polish.rows, "gap_ms", "Extra silence between chunks",
                        "Adds a little air between one synthesis call and the next. 0 keeps the "
                        "model's own pacing, which is what sounds most natural.",
                        live=True, accent=INDIGO, decimals=0)
        self.add_slider(polish.rows, "trim_silence_ms", "Trim padding from each clip",
                        "StyleTTS 2 pads every call with silence, which is what makes streamed "
                        "words sound spaced out. This trims it back. 0 keeps the model's own "
                        "padding (useful when comparing voices).", live=True, accent=INDIGO)
        self.add_slider(polish.rows, "retries", "Retries when a call fails",
                        "How many times a failed synthesis call is repeated before the chunk is "
                        "given up on and logged.", live=True, accent=INDIGO, decimals=0)

        engine = Card(page, "Speech engine")
        engine.pack(fill="x")
        info = tk.Frame(engine.rows, bg=PANEL)
        info.pack(fill="x", padx=6, pady=4)
        self.engine_info = tk.Label(info, text="checking ...", bg=PANEL, fg=DIM,
                                    font=self.font_small, justify="left", anchor="w",
                                    wraplength=760)
        self.engine_info.pack(anchor="w")
        buttons = tk.Frame(engine.rows, bg=PANEL)
        buttons.pack(fill="x", padx=6, pady=(8, 2))
        pill_button(buttons, "Start the speech engine", self.start_speech_engine, "primary").pack(side="left")
        pill_button(buttons, "Open its window (run-tts.bat)", self.open_tts_window, "ghost").pack(side="left", padx=8)
        pill_button(buttons, "Recheck", self._poll_status, "quiet").pack(side="left", padx=8)

    def _page_audio(self, page: tk.Frame) -> None:
        devices = Card(page, "Output device", "Where the cloned voice is written. This is what "
                                              "applications record from.")
        devices.pack(fill="x", pady=(0, 14))
        content = tk.Frame(devices.rows, bg=PANEL)
        content.pack(fill="x", padx=6, pady=4)
        tk.Label(content, text="Playback device", bg=PANEL, fg=BRIGHT,
                 font=self.font_bold).pack(anchor="w")
        self.output_box = dark_combo(content, [voices_mod.VOICES_DIR.name], width=54)
        self.output_box.pack(anchor="w", pady=(6, 0))
        self.output_box.bind("<<ComboboxSelected>>",
                             lambda _e: self.bridge.set("device", self._device_substring()))
        tk.Label(content, text="VB-CABLE is the loopback pair the rest of this project is built "
                               "around: write to 'CABLE Input' and every application can record "
                               "'CABLE Output' as an ordinary microphone. Pick another device only "
                               "to test through your speakers. Changes here take effect the next "
                               "time the voice changer starts.",
                 bg=PANEL, fg=FAINT, font=self.font_small, justify="left",
                 wraplength=760, anchor="w").pack(anchor="w", pady=(8, 0))
        line = tk.Frame(content, bg=PANEL)
        line.pack(anchor="w", pady=(10, 0))
        pill_button(line, "Refresh devices", self._refresh_devices, "ghost").pack(side="left")
        self.device_note = tk.Label(line, text="", bg=PANEL, fg=FAINT, font=self.font_small)
        self.device_note.pack(side="left", padx=12)

        monitor = Card(page, "Monitoring", "Hear the voice yourself while you use it.", accent=INDIGO)
        monitor.pack(fill="x", pady=(0, 14))
        self.add_toggle(monitor.rows, "monitor", "Also play it through my speakers",
                        "Plays everything the cable receives through the default output as well, "
                        "so you can check a voice without another application recording it.")

        chain = Card(page, "Signal chain")
        chain.pack(fill="x")
        tk.Label(chain.rows,
                 text="  your microphone\n"
                      "        \u2193   (1)  the transcription stage: live_transcribe / google_stt\n"
                      "  out\\transcript.txt\n"
                      "        \u2193   (2)  the voice changer: chunker \u2192 StyleTTS 2 \u2192 effects\n"
                      "  CABLE Input \u2192 CABLE Output\n"
                      "        \u2193   (3)  Discord / a game / OBS records 'CABLE Output'\n"
                      "  a voice that is not yours",
                 bg=ROW, fg=DIM, font=self.font_mono, justify="left", anchor="w").pack(
            fill="x", padx=12, pady=10)

    def _page_effects(self, page: tk.Frame) -> None:
        card = Card(page, "Output effects", "Applied to every clip after StyleTTS 2, on its way "
                                            "into the cable.", accent=INDIGO)
        card.pack(fill="x", pady=(0, 14))
        self.add_slider(card.rows, "pitch_semitones", "Pitch shift",
                        "Twelve semitones is one octave. The shift is a playback rate, so it also "
                        "changes how long the clip lasts: raising it makes the voice higher and "
                        "faster, which is the classic voice-changer sound. This one reopens the "
                        "virtual microphone, so expect a short gap when you move it.",
                        live=True, accent=INDIGO, decimals=1)
        self.add_slider(card.rows, "volume", "Output volume",
                        "A plain gain with the peaks soft-limited, so pushing it past 1.0 gets "
                        "louder without clipping into a buzz.", live=True, accent=INDIGO, decimals=2)

        gate = Card(page, "Noise gate", "Fades the inaudible stretches down, so a quiet room "
                                        "sounds silent instead of faintly hissy.")
        gate.pack(fill="x", pady=(0, 14))
        self.add_toggle(gate.rows, "gate_on", "Enable the noise gate",
                        "-70 dB, the default, is off: nothing is touched. Turning this on sets a "
                        "-45 dB threshold, which is a good starting point.",
                        live=True, command=self._gate_toggle)
        self.add_slider(gate.rows, "gate_db", "Open above this level",
                        "Anything below this is treated as room noise rather than speech. Too high "
                        "and quiet words get swallowed; too low and the gate does nothing.",
                        live=True, decimals=0)
        self.add_slider(gate.rows, "gate_floor_db", "Push it down to",
                        "How far the gated stretches are pushed. -60 dB is close to silent; a "
                        "floor rather than full silence is deliberate, because gating to a hard "
                        "zero sounds like dropouts on a phone line.", live=True, decimals=0)

        note = Card(page, "What is not here")
        note.pack(fill="x")
        tk.Label(note.rows,
                 text="Reverb, EQ, formant and 'AI noise suppression' are not implemented. Those "
                      "controls would need the speech server to expose parameters it does not "
                      "have, and a slider that changes nothing is worse than no slider. Pitch, "
                      "volume and the gate above are real: they are applied by this project, to "
                      "the audio, every clip.",
                 bg=ROW, fg=DIM, font=self.font_small, justify="left", anchor="w",
                 wraplength=740).pack(fill="x", padx=14, pady=12)

    def _page_input(self, page: tk.Frame) -> None:
        card = Card(page, "Microphone -> text", "Stage one. The voice changer can only speak what "
                                                "this stage writes down.")
        card.pack(fill="x", pady=(0, 14))
        self.add_choice(card.rows, "stt_engine", "Which recogniser to use",
                        "whisper runs locally on the GPU and works offline. google uses "
                        "Chrome's own recogniser through a local page, which is more accurate "
                        "on some voices and needs a browser tab kept open.",
                        options=[("whisper", "Local Whisper (offline, GPU)"),
                                 ("google", "Google via Chrome (needs a browser tab)")],
                        live=False)
        self.add_toggle(card.rows, "stt_start", "Start transcription with the voice changer",
                        "Opens the transcription stage in its own window when the voice changer "
                        "starts, so one button gets both stages running.", live=False)
        self.add_text(card.rows, "stt_args", "Extra arguments",
                      "Passed straight to the recogniser, e.g. --model small.en or --language en-GB. "
                      "Leave empty for the defaults.", live=False)
        buttons = tk.Frame(card.rows, bg=PANEL)
        buttons.pack(fill="x", padx=6, pady=(8, 2))
        pill_button(buttons, "\u25B6  Start transcription now", self.start_transcription, "primary").pack(side="left")
        pill_button(buttons, "Open the transcription page", self.open_stt_page, "ghost").pack(side="left", padx=8)
        pill_button(buttons, "List microphones", self.list_microphones, "quiet").pack(side="left", padx=8)

        transcript = Card(page, "Transcript file", "The meeting point between the two stages.",
                          accent=INDIGO)
        transcript.pack(fill="x", pady=(0, 14))
        content = tk.Frame(transcript.rows, bg=PANEL)
        content.pack(fill="x", padx=6, pady=4)
        self.transcript_view = tk.Text(content, height=5, bg=SUNKEN, fg="#B9C4D6",
                                       relief="flat", bd=0, highlightthickness=1,
                                       highlightbackground=EDGE, font=self.font_mono,
                                       wrap="word", padx=10, pady=8)
        self.transcript_view.pack(fill="x", pady=(0, 8))
        line = tk.Frame(content, bg=PANEL)
        line.pack(anchor="w")
        pill_button(line, "Re-read", self.refresh_transcript, "ghost").pack(side="left")
        pill_button(line, "Clear the transcript", self.clear_transcript, "danger").pack(side="left", padx=8)

        test = Card(page, "Test the voice changer without speaking", "Speaks the transcript's "
                                                                    "current contents once.")
        test.pack(fill="x")
        pill_button(test.rows, "Speak the transcript now", self.speak_transcript_once, "primary").pack(
            anchor="w", padx=12, pady=10)

    def _page_advanced(self, page: tk.Frame) -> None:
        raw = Card(page, "Raw settings", "The engine arguments the control panel does not model "
                                         "with its own control.", accent=INDIGO)
        raw.pack(fill="x", pady=(0, 14))
        self.add_text(raw.rows, "tts_args", "Speech server arguments",
                      "Passed to the voice changer's speech-server connection, e.g. --port 9881. "
                      "Leave empty for the defaults.", live=False, width=64)
        self.add_text(raw.rows, "voice", "Starting voice",
                      "The voice the engine starts with. The Voices page sets this for you; it is "
                      "here so a settings file can be written by hand.", live=False, width=40)

        paths = Card(page, "Paths in use")
        paths.pack(fill="x", pady=(0, 14))
        self.paths_label = tk.Label(paths.rows, text="", bg=ROW, fg=DIM, font=self.font_mono,
                                    justify="left", anchor="w")
        self.paths_label.pack(fill="x", padx=14, pady=12)
        line = tk.Frame(paths.rows, bg=PANEL)
        line.pack(fill="x", padx=6, pady=(4, 2))
        pill_button(line, "Open the project folder", self.open_project_folder, "ghost").pack(side="left")
        pill_button(line, "Open the settings file", self.open_settings_file, "ghost").pack(side="left", padx=8)
        pill_button(line, "Run the preflight check", self.run_preflight, "ghost").pack(side="left", padx=8)

        hotkeys = Card(page, "Keyboard shortcuts", "These work while this window is focused.",
                       accent=INDIGO)
        hotkeys.pack(fill="x", pady=(0, 14))
        for keys, what in [
            ("Ctrl + Enter", "Start or stop the voice changer"),
            ("F5", "Re-read the transcript and every status"),
            ("Ctrl + 1 .. 8", "Jump to a sidebar page"),
            ("Ctrl + S", "Save changes"),
            ("Esc", "Hide the window (it keeps running)"),
        ]:
            line = tk.Frame(hotkeys.rows, bg=PANEL)
            line.pack(fill="x", padx=6, pady=3)
            tk.Label(line, text=what, bg=PANEL, fg=TEXT, font=self.font_small).pack(side="left")
            tk.Label(line, text=f" {keys} ", bg=PANEL_2, fg=CYAN, font=self.font_mono).pack(side="right")

        danger = Card(page, "Reset", "The last-resort buttons.")
        danger.pack(fill="x")
        content = tk.Frame(danger.rows, bg=PANEL)
        content.pack(fill="x", padx=6, pady=6)
        tk.Label(content, text="Factory reset", bg=PANEL, fg=TEXT,
                 font=self.font_bold).pack(anchor="w")
        tk.Label(content, text="Puts every setting on this panel back to its default and writes "
                               "that to the settings file. Voices, the transcript and the "
                               "emotion mapping are files and are left alone.",
                 bg=PANEL, fg=FAINT, font=self.font_small, justify="left",
                 wraplength=740, anchor="w").pack(anchor="w", pady=(2, 8))
        pill_button(content, "Factory reset all settings", self.factory_reset, "danger").pack(anchor="w")

    # -- voice helpers ----------------------------------------------------- #
    def reload_voices(self) -> None:
        """Re-read the voices folder and show it in the picker, table and mapping.

        Note the `is not None` checks. Tkinter widgets define `__bool__` from their
        Tcl name, which is something like ``.!frame.!combobox2``, and Tk answers
        "0" for anything it does not recognise as a boolean -- so ``if self.voice_box:``
        is False for a perfectly good widget, and the whole reload silently did
        nothing.
        """
        self.voices = voices_mod.discover(self.voices_dir)
        labels = [self._voice_label(voice) for voice in self.voices]
        if getattr(self, "voice_box", None) is not None:
            self.voice_box.configure(values=labels)
            active = self.active_voice_slug()
            index = next((i for i, v in enumerate(self.voices) if v.slug == active), None)
            if index is None:
                index = next((i for i, v in enumerate(self.voices) if v.usable), None)
            if index is not None:
                self.voice_box.current(index)
            self._voice_sel_changed()
        for box in (getattr(self, "emotion_boxes", None) or {}).values():
            box.configure(values=["(use the active voice)"]
                                 + [v.slug for v in self.voices if v.usable])
        tree = getattr(self, "voice_tree", None)
        if tree is not None:
            tree.delete(*tree.get_children())
            active = self.active_voice_slug()
            for voice in self.voices:
                state = voice.problem or ("in use" if voice.slug == active else "ready")
                tree.insert("", "end", values=(f"{voice.seconds:.1f}s", f"{voice.sample_rate} Hz",
                                               state, voice.name))
        self.load_emotion_mapping()

    @staticmethod
    def _voice_label(voice: voices_mod.Voice) -> str:
        if voice.problem:
            return f"{voice.slug}  \u2014 {voice.problem}"
        return f"{voice.slug}   ({voice.seconds:.1f}s, {voice.sample_rate} Hz)"

    def active_voice_slug(self) -> str:
        try:
            if self.selector.is_file():
                return self.selector.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        return str(self.bridge.get("voice") or "")

    def selected_voice(self) -> voices_mod.Voice | None:
        box = getattr(self, "voice_box", None)
        index = box.current() if box is not None else -1
        if 0 <= index < len(self.voices):
            return self.voices[index]
        return None

    def _voice_sel_changed(self) -> None:
        voice = self.selected_voice()
        if voice is None or not hasattr(self, "voice_note"):
            return
        if voice.problem:
            self.voice_note.configure(text=f"\u26A0  {voice.problem}", fg=BAD)
        else:
            detail = f"{voice.seconds:.1f} s, {voice.sample_rate} Hz, {voice.channels} channel(s)"
            if voice.transcript:
                detail += f"\nTranscript: {voice.transcript[:90]}"
            self.voice_note.configure(text=detail, fg=FAINT)

    def apply_voice(self) -> None:
        voice = self.selected_voice()
        if voice is None:
            return
        if not voice.usable:
            messagebox.showwarning("That voice cannot be used",
                                   f"{voice.slug}: {voice.problem}", parent=self.root)
            return
        try:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            self.selector.write_text(voice.slug, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not write the selector", str(exc), parent=self.root)
            return
        self.bridge.set("voice", voice.slug)
        self.bridge.save()
        self.load_emotion_mapping()
        self.show_toast("Voice changed", f"Now speaking as {voice.slug}. It applies on the next "
                                         "sentence - no restart needed.")

    def preview_voice(self) -> None:
        voice = self.selected_voice()
        if voice is None:
            return
        if not self.engine_live():
            self.show_toast("The speech engine is not running",
                            "Start it first (Speech engine page), then preview a voice.",
                            kind="warn")
            return
        try:
            subprocess.Popen(
                [str(PYTHON if PYTHON.exists() else sys.executable), "-u",
                 str(HERE / "speak.py"), "--voice", voice.slug,
                 "This is the selected voice."],
                cwd=str(HERE), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except OSError as exc:
            messagebox.showerror("Could not start the preview", str(exc), parent=self.root)
            return
        self.show_toast("Preview started", f"A test line is being spoken as {voice.slug}.")

    def import_voice(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a wav to use as a voice",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        source = Path(path)
        target = self.voices_dir / source.name
        if target.exists() and not messagebox.askyesno(
                "Replace the existing voice?", f"{target.name} already exists. Replace it?",
                parent=self.root):
            return
        try:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        except OSError as exc:
            messagebox.showerror("Could not copy the file", str(exc), parent=self.root)
            return
        self.reload_voices()
        self.show_toast("Voice imported", f"{target.name} is in the voices folder. "
                                          "It must be 3-16 seconds long to be usable.")

    def open_voices_folder(self) -> None:
        self._open_path(self.voices_dir)

    def load_emotion_mapping(self) -> None:
        boxes = getattr(self, "emotion_boxes", None)
        if not boxes:
            return
        mapping = emotion_voices.load(self.voices_dir)
        for name, box in boxes.items():
            path = mapping.by_emotion.get(name)
            if path is None:
                box.set("(use the active voice)")
            else:
                box.set(path.stem)
                if path.stem not in box["values"]:
                    box.configure(values=list(box["values"]) + [path.stem])

    def save_emotion(self, emotion: str) -> None:
        boxes = getattr(self, "emotion_boxes", {})
        mapping: dict[str, str] = {}
        for name, box in boxes.items():
            choice = box.get()
            if choice and choice != "(use the active voice)":
                mapping[name] = choice
        pairs: dict[str, str] = {}
        for name, slug in mapping.items():
            path = next((v.path for v in self.voices if v.slug == slug), None)
            if path is not None:
                pairs[name] = str(path)
        try:
            emotion_voices.write_mapping(self.voices_dir, pairs)
        except OSError as exc:
            messagebox.showerror("Could not save the emotion mapping", str(exc), parent=self.root)
            return
        if emotion in pairs:
            self.show_toast("Emotion voice set", f"{emotion.title()} now uses {pairs[emotion]}.")
        else:
            self.show_toast("Emotion voice cleared", f"{emotion.title()} uses the active voice.")

    # -- engine control ---------------------------------------------------- #
    def engine_live(self) -> bool:
        return self._http_ok(TTS_PORT)

    @staticmethod
    def _http_ok(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.2) as response:
                json.loads(response.read())
            return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return False

    def start_engine(self) -> None:
        if self.engine.running:
            self.show_toast("Already running", "The voice changer is up.", kind="warn")
            return
        if not self.engine_live() and not messagebox.askyesno(
                "The speech engine is not running",
                "The StyleTTS 2 speech server does not answer on port 9881.\n\n"
                "The voice changer will wait for it. Start it now?", parent=self.root):
            return
        if not self.engine_live():
            self.start_speech_engine()
        self.bridge.save()
        extra: list[str] = []
        device = str(self.bridge.get("device") or "").strip()
        if device:
            extra += ["--device", device]
        extra += str(self.bridge.get("tts_args") or "").split()
        ok, message = self.engine.start(extra)
        if not ok:
            messagebox.showerror("Could not start", message, parent=self.root)
        else:
            self.show_toast("Voice changer starting", "Watch the Live output card on the Dashboard.")
        self._poll_status()

    def stop_engine(self) -> None:
        if self.engine.stop():
            self.show_toast("Stopped", "The voice changer was stopped and the microphone released.")
        else:
            self.show_toast("Not running", "There was nothing to stop.", kind="warn")
        self._poll_status()

    def stop_everything(self) -> None:
        self.engine.stop()
        try:
            subprocess.Popen(
                [str(PYTHON if PYTHON.exists() else sys.executable), "-u",
                 str(HERE / "launch.py"), "stop"],
                cwd=str(HERE), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
        except OSError as exc:
            messagebox.showerror("Could not run the stop helper", str(exc), parent=self.root)
            return
        self.show_toast("Stopping", "The voice changer and the speech engine are being stopped.")
        self._poll_status()

    def start_speech_engine(self) -> None:
        self.open_tts_window()

    def open_tts_window(self) -> None:
        try:
            subprocess.Popen(["cmd", "/c", "start", "Rixers speech engine", "/D", str(HERE),
                              "run-tts.bat"],
                             cwd=str(HERE), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except OSError as exc:
            messagebox.showerror("Could not start the speech engine", str(exc), parent=self.root)
            return
        self.show_toast("Speech engine starting",
                        "It takes 60-90 seconds to load the model. Its window shows the progress.")

    def start_transcription(self) -> None:
        engine = str(self.bridge.get("stt_engine"))
        script = "google_stt.py" if engine == "google" else "transcribe.py"
        args = [str(PYTHON if PYTHON.exists() else sys.executable), "-u", str(HERE / script)]
        args += str(self.bridge.get("stt_args") or "").split()
        try:
            subprocess.Popen(args, cwd=str(HERE),
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except OSError as exc:
            messagebox.showerror("Could not start transcription", str(exc), parent=self.root)
            return
        self.show_toast("Transcription starting", f"{script} is opening in its own window.")

    def test_line(self) -> None:
        self.speak_transcript_once()

    def speak_transcript_once(self) -> None:
        if not self.engine_live():
            self.show_toast("The speech engine is not running",
                            "Start it first, then a test line can be spoken.", kind="warn")
            return
        args = [str(PYTHON if PYTHON.exists() else sys.executable), "-u",
                str(HERE / "voice_changer.py"), "--once", "--voice", str(self.active_voice_slug() or "default")]
        args += str(self.bridge.get("tts_args") or "").split()
        try:
            subprocess.Popen(args, cwd=str(HERE),
                             creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except OSError as exc:
            messagebox.showerror("Could not speak the transcript", str(exc), parent=self.root)
            return
        self.show_toast("Speaking the transcript", "A one-shot run is speaking what is in "
                                                   "out\\transcript.txt.")

    # -- transcript helpers ------------------------------------------------ #
    def refresh_transcript(self) -> None:
        view = getattr(self, "transcript_view", None)
        if view is None:
            return
        try:
            text = (HERE / "out" / "transcript.txt").read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"could not read out\\transcript.txt: {exc}"
        view.configure(state="normal")
        view.delete("1.0", "end")
        view.insert("1.0", text.strip() or "(the transcript is empty - stage one has not run yet)")
        view.configure(state="disabled")

    def clear_transcript(self) -> None:
        if not messagebox.askyesno("Clear the transcript?",
                                   "The transcription stage will refill it as you speak.",
                                   parent=self.root):
            return
        try:
            (HERE / "out" / "transcript.txt").write_text("", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not clear it", str(exc), parent=self.root)
            return
        self.refresh_transcript()
        self.show_toast("Transcript cleared", "Stage one writes to it again as you speak.")

    def list_microphones(self) -> None:
        try:
            subprocess.Popen(
                [str(PYTHON if PYTHON.exists() else sys.executable), "-u",
                 str(HERE / "transcribe.py"), "--list-devices"],
                cwd=str(HERE), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except OSError as exc:
            messagebox.showerror("Could not list the microphones", str(exc), parent=self.root)

    def open_stt_page(self) -> None:
        if self._http_ok(STT_PORT):
            webbrowser.open(f"http://127.0.0.1:{STT_PORT}/")
        else:
            self.show_toast("The transcription page is not serving",
                            "Start transcription first; the page lives on 127.0.0.1:8777.",
                            kind="warn")

    # -- misc actions ------------------------------------------------------ #
    def open_project_folder(self) -> None:
        self._open_path(HERE)

    def open_settings_file(self) -> None:
        path = self.bridge.live.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                self.bridge.save()
            os.startfile(str(path))  # noqa: S606 - the user asked to open their own file
        except OSError as exc:
            messagebox.showerror("Could not open the settings file", str(exc), parent=self.root)

    def run_preflight(self) -> None:
        try:
            subprocess.Popen(
                [str(PYTHON if PYTHON.exists() else sys.executable), "-u",
                 str(HERE / "launch.py"), "check"],
                cwd=str(HERE), creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        except OSError as exc:
            messagebox.showerror("Could not run the check", str(exc), parent=self.root)

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            os.startfile(str(path))  # noqa: S606
        except OSError:
            webbrowser.open(path.as_uri())

    def factory_reset(self) -> None:
        if not messagebox.askyesno(
                "Factory reset?",
                "Every setting on this panel goes back to its default. Voices, the transcript "
                "and the emotion mapping are files and are left alone.", parent=self.root):
            return
        self.bridge.live.values = live_settings.defaults()
        self.bridge.live.save()
        self.bridge.reload()
        for name, widgets in self.widgets.items():
            value = self.bridge.get(name)
            for widget in widgets:
                setter = getattr(widget, "set", None)
                if callable(setter):
                    try:
                        setter(value)
                    except Exception:  # noqa: BLE001
                        pass
        self.show_toast("Factory reset", "Every setting is back to its default.")

    # -- the gate toggle is a view of gate_db ------------------------------ #
    def _gate_toggle(self, enabled: bool) -> None:
        self.bridge.set("gate_on", enabled)
        if enabled and float(self.bridge.get("gate_db")) <= -69.5:
            self.bridge.set("gate_db", -45.0, live=True)
        elif not enabled:
            self.bridge.set("gate_db", -70.0, live=True)
        for widget in self.widgets.get("gate_db", []):
            if isinstance(widget, Slider):
                widget.set(self.bridge.get("gate_db"))
        if not self.bridge.is_dirty():
            self.bridge.save()

    # -- status polling ---------------------------------------------------- #
    def _poll_status(self) -> None:
        """Ask for the slow facts on a worker thread; draw them here.

        The worker never touches a widget -- not even through ``after``. Tkinter is
        only safe from the thread running the event loop, and calling into it from
        anywhere else raises "main thread is not in main loop" the moment the window
        is being pumped with ``update`` rather than ``mainloop`` (which is exactly
        what the test harness does, and what a modal dialog does to the real one).
        So the worker's only output is a queue entry, and every widget change
        happens on this thread.
        """
        if self._closing:
            return
        if not self._status_busy:
            self._status_busy = True
            threading.Thread(target=self._gather_status, name="rixers-status",
                             daemon=True).start()
        self._drain_results()
        self._draw_status()
        self.refresh_transcript()
        self.root.after(1500, self._poll_status)

    def _drain_results(self) -> None:
        """Take anything the worker threads finished, on the UI thread."""
        latest: tuple[list[str], list[str]] | None = None
        while True:
            try:
                latest = self.device_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.devices = {"output": latest[0], "input": latest[1]}
            self._apply_device_list()

    def _gather_status(self) -> None:
        report: dict = {}
        try:
            report["tts"] = self._tts_health()
            report["stt"] = self._http_ok(STT_PORT)
            report["voice"] = Path(str(report["tts"].get("reference_path") or "")).stem
        except Exception as exc:  # noqa: BLE001 - a status panel must never be fatal
            report["error"] = f"{type(exc).__name__}: {exc}"
        self.status_queue.put(report)
        self._status_busy = False

    @staticmethod
    def _tts_health() -> dict:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{TTS_PORT}/health", timeout=1.5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return {}

    def _draw_status(self) -> None:
        while True:
            try:
                self._last_status = self.status_queue.get_nowait()
            except queue.Empty:
                break
        report = self._last_status
        running = self.engine.running
        if hasattr(self, "status_boxes"):
            self._set_status("engine",
                             "running" if running else "stopped",
                             OK if running else DIM,
                             f"up {self.engine.uptime():.0f}s" if running
                             else "press Start on the Dashboard")
            health = report.get("tts") or {}
            if health:
                device = health.get("device") or "?"
                loaded = health.get("loaded") or health.get("reference_loaded")
                self._set_status("tts", f"ready on {device}" if loaded else "loading ...",
                                 OK if loaded else WARN, str(health.get("engine") or ""))
            else:
                self._set_status("tts", "not running", BAD, "start it from the Speech engine page")
            stt = report.get("stt")
            self._set_status("stt", "serving" if stt else "not running",
                             OK if stt else DIM,
                             "127.0.0.1:8777" if stt else "start it from the Transcription page")
            self._set_status("voice", report.get("voice") or self.active_voice_slug() or "none",
                             TEXT, "out\\transcript.txt is the input")
        if hasattr(self, "engine_info"):
            health = report.get("tts") or {}
            if health:
                lines = [
                    f"backend      : {health.get('backend')} on {health.get('device')}",
                    f"loaded       : {health.get('loaded')}",
                    f"reference    : {health.get('reference_path') or '-'}",
                    f"vram         : {health.get('vram_mb') or '-'} MB",
                    f"endpoints    : {', '.join(health.get('endpoints') or [])}",
                ]
                self.engine_info.configure(text="\n".join(lines), fg=DIM)
            else:
                self.engine_info.configure(
                    text="The speech server is not answering on 127.0.0.1:9881.\n"
                         "Start it with the button below; it takes 60-90 seconds to load the "
                         "model onto the GPU.", fg=WARN)
        if hasattr(self, "start_button"):
            state = "disabled" if running else "normal"
            self.start_button.configure(state=state,
                                        bg="#155E63" if running else CYAN)
            self.stop_button.configure(state="normal" if running else "disabled")
        if hasattr(self, "paths_label"):
            self.paths_label.configure(text=self._paths_text())

    def _paths_text(self) -> str:
        rows = [
            ("project", HERE),
            ("settings", self.bridge.live.path),
            ("voices", self.voices_dir),
            ("voice.txt", self.selector),
            ("transcript", HERE / "out" / "transcript.txt"),
            ("python", PYTHON if PYTHON.exists() else Path(sys.executable)),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}}  {path}" for name, path in rows)

    def _set_status(self, key: str, text: str, colour: str, note: str = "") -> None:
        label = self.status_boxes.get(key)
        if label is not None:
            label.configure(text=str(text)[:46], fg=colour)

    # -- devices ----------------------------------------------------------- #
    def _refresh_devices(self) -> None:
        """List the audio devices on a worker thread; the UI thread applies it.

        Enumerating every endpoint is slow enough to freeze the window, so it is
        done off the UI thread -- and the result is posted to a queue rather than
        pushed at a widget, for the reason spelled out in `_poll_status`.
        """
        if self._device_busy:
            return
        self._device_busy = True

        def work() -> None:
            try:
                import sounddevice as sd

                outputs, inputs = [], []
                for index, device in enumerate(sd.query_devices()):
                    name = str(device["name"])
                    if device["max_output_channels"] > 0:
                        outputs.append(f"{index}: {name}")
                    if device["max_input_channels"] > 0:
                        inputs.append(f"{index}: {name}")
                self.device_queue.put((outputs, inputs))
            except Exception:  # noqa: BLE001 - the list is a convenience, not a need
                self.device_queue.put(([], []))
            finally:
                self._device_busy = False

        threading.Thread(target=work, name="rixers-devices", daemon=True).start()

    def _apply_device_list(self) -> None:
        if hasattr(self, "output_box"):
            self.output_box.configure(values=self.devices["output"])
            current = str(self.bridge.get("device") or "")
            cable = next((d for d in self.devices["output"] if "CABLE Input" in d), None)
            self.output_box.set(current or cable or "(no output devices found)")
        if not self.devices["output"]:
            self._device_note("no output devices could be listed", WARN)
            return
        found = any("CABLE Input" in name for name in self.devices["output"])
        self._device_note(
            "VB-CABLE found" if found else "VB-CABLE not found - install it before starting",
            OK if found else WARN)

    def _device_note(self, text: str, colour: str = FAINT) -> None:
        label = getattr(self, "device_note", None)
        if label is not None:
            label.configure(text=text, fg=colour)

    def _device_substring(self) -> str:
        """'3: CABLE Input (VB-Audio)' -> the name the engine matches on.

        `find_device` matches a name substring, so the index the list shows is
        dropped: it is not stable across reboots, and a name is.
        """
        chosen = self.output_box.get()
        if chosen.startswith("("):
            return ""
        _, _, name = chosen.partition(": ")
        return name.strip() or chosen

    # -- logging and toasts ------------------------------------------------ #
    def _enqueue_log(self, line: str) -> None:
        self.log_queue.put(line)

    def _drain_log(self) -> None:
        chunks: list[str] = []
        while True:
            try:
                chunks.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        text = getattr(self, "log_text", None)
        if chunks and text is not None and not self._closing:
            text.configure(state="normal")
            for line in chunks[:400]:
                text.insert("end", line + "\n")
                self._log_lines += 1
            if self._log_lines > 2000:
                text.delete("1.0", f"{self._log_lines - 1200}.0")
                self._log_lines = 1200
            text.see("end")
            text.configure(state="disabled")
        if not self._closing:
            self.root.after(250, self._drain_log)

    def _clear_log(self) -> None:
        text = getattr(self, "log_text", None)
        if text is not None:
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.configure(state="disabled")
            self._log_lines = 0

    def show_toast(self, title: str, message: str, kind: str = "ok") -> None:
        palette = {
            "ok": ("#10322A", "#1F6F5C", "#A7F3D0", OK),
            "warn": ("#3A2E12", "#8A6A1F", "#FDE68A", WARN),
            "bad": ("#3A1519", "#7F2C33", "#FECACA", BAD),
        }
        bg, border, fg, dot = palette.get(kind, palette["ok"])
        self.toast.configure(bg=bg, highlightbackground=border)
        self.toast_label.configure(bg=bg, fg=fg,
                                   text=f"{dot}  {title}\n     {message}")
        self.toast.place(relx=1.0, rely=1.0, x=-26, y=-26, anchor="se")
        self.toast.lift()
        if self._toast_job is not None:
            self.root.after_cancel(self._toast_job)
        self._toast_job = self.root.after(4200, self.toast.place_forget)

    # -- saving ------------------------------------------------------------ #
    def save_all(self) -> bool:
        # The live fields are already written as they were moved; this is what
        # commits the rest (the voice, the devices, the transcription stage).
        try:
            path = self.bridge.save()
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc), parent=self.root)
            return False
        self.show_toast("Settings saved", f"Written to {path.name}. Live settings are already in "
                                          "force for the running voice changer.")
        return True

    def _live_note(self, name: str, value) -> None:
        """Called when a live setting was written while the engine is running."""
        if name == "pitch_semitones" and self.engine.running:
            self.show_toast("Pitch changed",
                            "The virtual microphone is being reopened, so expect a short gap.",
                            kind="warn")

    def _refresh_dirty_state(self) -> None:
        dirty = self.bridge.is_dirty()
        if hasattr(self, "dirty_label"):
            self.dirty_label.configure(text="\u25CF unsaved changes" if dirty else "",
                                       fg=WARN)
        if hasattr(self, "save_button"):
            self.save_button.configure(state="normal" if dirty or True else "disabled")

    def reset_page(self) -> None:
        names = [name for name, widgets in self.widgets.items()
                 if any(self._widget_on_page(widget, self.current_page) for widget in widgets)]
        if not names:
            self.show_toast("Nothing to reset", "This page has no settings of its own.",
                            kind="warn")
            return
        self.bridge.reset(names)
        for name in names:
            value = self.bridge.get(name)
            for widget in self.widgets.get(name, []):
                setter = getattr(widget, "set", None)
                if callable(setter):
                    try:
                        setter(value)
                    except Exception:  # noqa: BLE001
                        pass
        self.show_toast("Defaults restored", f"{len(names)} setting(s) on this page went back to "
                                             "their defaults. Press Save to write them.")

    def _widget_on_page(self, widget, page_key: str) -> bool:
        target = self.pages.get(page_key)
        node = widget
        while node is not None:
            if node is target:
                return True
            node = getattr(node, "master", None)
        return False

    # -- keys and closing -------------------------------------------------- #
    def _apply_startup_voice(self) -> None:
        """Make sure voice.txt exists, so the engine and this panel agree."""
        slug = self.active_voice_slug()
        if slug:
            self.bridge.set("voice", slug)
            self._refresh_dirty_state()
        # The gate switch is a view of the threshold, so it starts from the file.
        self.bridge.draft["gate_on"] = float(self.bridge.get("gate_db")) > -69.5
        for widget in self.widgets.get("gate_on", []):
            widget.set(self.bridge.draft["gate_on"])

    def _bind_keys(self) -> None:
        self.root.bind("<Control-Return>", lambda _e: self._toggle_engine())
        self.root.bind("<Control-s>", lambda _e: self.save_all())
        self.root.bind("<F5>", self._refresh_everything)
        self.root.bind("<Escape>", lambda _e: self.root.withdraw())
        for index, (key, _label) in enumerate(PAGES, start=1):
            self.root.bind(f"<Control-Key-{index}>", lambda _e, k=key: self.show_page(k))

    def _toggle_engine(self) -> None:
        if self.engine.running:
            self.stop_engine()
        else:
            self.start_engine()

    def _refresh_everything(self, _event=None) -> None:
        self.refresh_transcript()
        self.reload_voices()
        self._refresh_devices()
        self._poll_status()
        self.show_toast("Refreshed", "Voices, devices and status were re-read.")

    def shutdown(self, ask: bool = True) -> None:
        """Stop everything, wait for the workers, and only then destroy the window.

        Order matters and is the whole point of this method. The polling loops are
        stopped first so nothing reschedules itself; the worker threads are then
        *joined*, because a thread still running when `destroy()` frees the
        interpreter is an access violation rather than a Python exception; and the
        window is destroyed last.
        """
        if self.engine.running and ask:
            if not messagebox.askyesno(
                    "The voice changer is running",
                    "Closing this window stops the voice changer.\n\nStop it and close?",
                    parent=self.root):
                return
        self._closing = True
        self.engine.stop()
        if self._toast_job is not None:
            try:
                self.root.after_cancel(self._toast_job)
            except tk.TclError:
                pass
            self._toast_job = None
        for thread in threading.enumerate():
            if thread is threading.current_thread() or thread.daemon is False:
                continue
            if thread.name.startswith("rixers-") or thread.name == "engine-log":
                thread.join(timeout=5.0)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def on_close(self) -> None:
        self.shutdown(ask=True)

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    settings_path = None
    for index, item in enumerate(argv):
        if item == "--settings" and index + 1 < len(argv):
            settings_path = Path(argv[index + 1])
    try:
        panel = ControlPanel(settings_path)
    except tk.TclError as exc:
        print(f"could not open a window: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        # Started from the .bat through pythonw, this process has no console, so a
        # crash would otherwise be completely silent.
        _report_startup_failure(exc)
        return 1
    panel.run()
    return 0


def _report_startup_failure(exc: BaseException) -> None:
    import traceback

    detail = traceback.format_exc()
    try:
        print(detail, file=sys.stderr)
    except Exception:  # noqa: BLE001 - no console attached
        pass
    try:
        fallback = tk.Tk()
        fallback.withdraw()
        messagebox.showerror(
            f"{APP_NAME} could not start",
            f"{type(exc).__name__}: {exc}\n\n{detail[-1200:]}",
            parent=fallback,
        )
        fallback.destroy()
    except Exception:  # noqa: BLE001 - a failed error report must not raise again
        pass


if __name__ == "__main__":
    sys.exit(main())
