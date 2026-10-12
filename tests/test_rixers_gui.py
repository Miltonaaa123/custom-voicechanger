"""Build the whole control panel off-screen and walk every page.

Catches the failures a byte-compile cannot: a widget that does not exist, a
callback that throws while a page is being drawn, a page that will not switch, a
control that is not wired to a real setting.

Run it from the project folder:

    .venv\\Scripts\\python.exe tests\\test_rixers_gui.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import rixers_gui  # noqa: E402


def quiet_dialogs() -> list[str]:
    """Answer every confirmation and record every complaint.

    The panel asks before doing anything destructive, which is right for a person
    and fatal for a test: `askyesno` blocks until someone clicks. This is also how
    a *failure* inside the panel becomes visible -- a warning box is a test failure.
    """
    complaints: list[str] = []

    rixers_gui.messagebox.askyesno = lambda *a, **k: True
    rixers_gui.messagebox.showwarning = lambda title, message, **k: complaints.append(f"{title}: {message}")
    rixers_gui.messagebox.showerror = lambda title, message, **k: complaints.append(f"{title}: {message}")
    return complaints


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: object = "") -> None:
        nonlocal failures
        if ok:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail!r}")

    complaints = quiet_dialogs()
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        panel = rixers_gui.ControlPanel(settings)
        root = panel.root
        root.withdraw()
        try:
            root.update()
            check("window built", True)
            check("settings file written", settings.is_file(), str(settings))
            check("voices discovered", len(panel.voices) >= 1, [v.slug for v in panel.voices])
            check("the voice picker is filled in", panel.voice_box.current() >= 0,
                  panel.voice_box.get())
            check("the voice library is filled in", len(panel.voice_tree.get_children()) == len(panel.voices))
            check("the voice list is not empty", len(panel.voice_box["values"]) == len(panel.voices),
                  panel.voice_box["values"])

            for key, _label in rixers_gui.PAGES:
                panel.show_page(key)
                # A pack/unpack and the canvas's own item sizing are settled in
                # idle tasks, so both have to run before a size means anything.
                for _ in range(3):
                    root.update_idletasks()
                    root.update()
                page = panel.pages[key]
                check(f"page '{key}' is the one showing", panel.current_page == key)
                # Only the visible page may take space: a page left packed while
                # another is showing is what silently stacked every page together.
                others = [name for name in panel.pages if name != key
                          and panel.pages[name].winfo_ismapped()]
                check(f"page '{key}' hides the others", not others, others)
                # A page that Tk has not laid out yet reports height 1, which is what
                # a clipped page looks like -- so this is the check that would have
                # caught the cards being squeezed to nothing. `winfo_reqheight` is
                # used because the window is deliberately never shown here: an
                # unmapped widget's actual size is 1x1 and says nothing.
                check(f"page '{key}' asks for a real height", page.winfo_reqheight() > 100,
                      (page.winfo_reqheight(), page.winfo_reqwidth()))
                # And it must stay that way: a page whose request shrinks on the next
                # geometry pass is a page being resized by something other than itself.
                height = page.winfo_reqheight()
                root.update()
                check(f"page '{key}' keeps its height", page.winfo_reqheight() == height,
                      (height, page.winfo_reqheight()))

            check("nav buttons all present", len(panel.nav_buttons) == len(rixers_gui.PAGES))

            # Every control must drive a real setting, and every slider must be able
            # to show what the file says.
            known = set(rixers_gui.live_settings.FIELDS)
            unknown = sorted(set(panel.widgets) - known)
            check("every registered control maps to a setting", not unknown, unknown)
            missing = sorted(known - set(panel.widgets) - {"voice", "tts_args", "device", "gate_on"})
            check("no setting is missing a control", not missing, missing)

            # A live edit must reach the file at once...
            panel.bridge.set("pause_flush_s", 1.25, live=True)
            root.update()
            written = rixers_gui.live_settings.LiveSettings(settings)
            written.load(force=True)
            check("a live edit is written immediately", written.get("pause_flush_s") == 1.25,
                  written.get("pause_flush_s"))
            # ...an out-of-range one must be clamped rather than saved...
            panel.bridge.set("volume", 99.0, live=True)
            written.load(force=True)
            check("an out-of-range value is clamped", written.get("volume") == 4.0,
                  written.get("volume"))
            # ...and a nonsense choice must fall back to the default.
            panel.bridge.set("mode", "banana", live=True)
            written.load(force=True)
            check("a nonsense choice falls back", written.get("mode") == "sentence",
                  written.get("mode"))

            # The widgets must follow the values back the other way.
            panel.bridge.set("pause_flush_s", 2.0, live=True)
            for widget in panel.widgets["pause_flush_s"]:
                widget.set(panel.bridge.get("pause_flush_s"))
            check("a slider shows the value it was given",
                  panel.widgets["pause_flush_s"][0].value == 2.0,
                  panel.widgets["pause_flush_s"][0].value)

            check("save reports success", panel.save_all() is True)

            panel.factory_reset()
            root.update()
            written.load(force=True)
            check("factory reset clears the change", written.get("pause_flush_s") == 0.7,
                  written.get("pause_flush_s"))

            # The polling paths must survive without a speech server running.
            panel._refresh_devices()
            panel._poll_status()
            root.update()
            panel._draw_status()
            panel.refresh_transcript()
            root.update()
            check("status and device polling survive", True)
            check("nothing popped a dialog", not complaints, complaints)
        finally:
            # The same teardown the window runs when it closes: stop the polling
            # loops, join the worker threads, and destroy last. Destroying while a
            # worker is still running is an access violation, not a Python error --
            # which is how this test killed its own interpreter once in five runs.
            panel.shutdown(ask=False)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
