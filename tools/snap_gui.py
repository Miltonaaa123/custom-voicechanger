#!/usr/bin/env python
"""Open the control panel, photograph each page, and close it.

For checking the layout without a person watching: builds the real window, lets Tk
draw it, screenshots it through `tools/screenshot.py`, moves to the next page, and
repeats. Nothing here is needed to *use* the app.

    .venv\\Scripts\\python.exe tools\\snap_gui.py out\\shots
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import rixers_gui  # noqa: E402
import screenshot  # noqa: E402


def main(argv: list[str]) -> int:
    out_dir = Path(argv[0]) if argv else HERE / "out" / "shots"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot.set_dpi_aware()

    with tempfile.TemporaryDirectory() as tmp:
        panel = rixers_gui.ControlPanel(Path(tmp) / "snap_settings.json")
        root = panel.root
        # Deliberately not deiconified: the window is drawn off-screen and captured
        # through PrintWindow, so checking the layout never steals focus from
        # whatever the user is actually doing.
        root.update()
        time.sleep(0.4)
        root.update()
        hwnd = int(root.winfo_id())
        # Tk's id is the client area; its parent is the real top-level frame, which
        # is what PrintWindow wants -- otherwise the title bar is cut off.
        parent = screenshot.user32.GetParent(hwnd)
        if parent:
            hwnd = parent
        rect = screenshot.wintypes.RECT()
        screenshot.user32.GetWindowRect(hwnd, screenshot.ctypes.byref(rect))

        for key, _label in rixers_gui.PAGES:
            panel.show_page(key)
            root.update()
            root.update_idletasks()
            time.sleep(0.25)
            root.update()
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            print(f"  {key}: {width}x{height}")
            screenshot.write_png(out_dir / f"{key}.png", width, height,
                                 screenshot.grab(0, 0, width, height, hwnd))
        panel.engine.stop()
        root.destroy()
    print(f"wrote {len(rixers_gui.PAGES)} page(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
