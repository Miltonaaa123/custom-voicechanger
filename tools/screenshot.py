#!/usr/bin/env python
"""Grab a screenshot of a window, using only the standard library.

Written to check the control panel's rendering without a human looking at the
screen. The bitmap comes from GDI through ctypes and the PNG is written through
Tkinter's own image support, so there is nothing to install.

    .venv\\Scripts\\python.exe tools\\screenshot.py out\\shot.png 0 0 1320 860
    .venv\\Scripts\\python.exe tools\\screenshot.py out\\shot.png --window "Rixers"
"""
from __future__ import annotations

import ctypes
import struct
import sys
import zlib
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def set_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - older Windows, or already set
        try:
            user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def find_window(title_part: str) -> int:
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title_part.lower() in buffer.value.lower():
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else 0


def grab(x: int, y: int, width: int, height: int, hwnd: int = 0) -> bytes:
    """Return BGRA bytes, top row first.

    With an `hwnd` the pixels come from that window's own device context via
    ``PrintWindow`` rather than from the screen, so the shot is right even when
    something else (a full-screen game, typically) is covering the window. Without
    it this is a screen grab, which is what you want for a whole desktop.
    """
    if hwnd:
        window_dc = user32.GetWindowDC(hwnd)
        memory = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        gdi32.SelectObject(memory, bitmap)
        # 2 = PW_RENDERFULLCONTENT, needed for windows drawn with DirectComposition.
        if not user32.PrintWindow(hwnd, memory, 2):
            user32.PrintWindow(hwnd, memory, 0)
        buffer = _bits(memory, bitmap, width, height)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(hwnd, window_dc)
        return buffer

    screen = user32.GetDC(0)
    memory = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
    gdi32.SelectObject(memory, bitmap)
    gdi32.BitBlt(memory, 0, 0, width, height, screen, x, y, SRCCOPY)
    buffer = _bits(memory, bitmap, width, height)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memory)
    user32.ReleaseDC(0, screen)
    return buffer


def _bits(memory, bitmap, width: int, height: int) -> bytes:
    info = BITMAPINFO()
    info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    info.bmiHeader.biWidth = width
    info.bmiHeader.biHeight = -height  # negative: top-down rows
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = BI_RGB
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(info), DIB_RGB_COLORS)
    return buffer.raw


def write_png(path: Path, width: int, height: int, bgra: bytes) -> None:
    """Minimal PNG writer: 8-bit RGB, one filter byte per row."""
    rows = bytearray()
    for row in range(height):
        start = row * width * 4
        line = bgra[start:start + width * 4]
        rows.append(0)
        for column in range(0, len(line), 4):
            rows += bytes((line[column + 2], line[column + 1], line[column]))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    target = Path(argv[0])
    set_dpi_aware()
    if argv[1] == "--window":
        hwnd = find_window(argv[2] if len(argv) > 2 else "Rixers")
        if not hwnd:
            print("no visible window with that title", file=sys.stderr)
            return 1
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        x, y = rect.left, rect.top
        width, height = rect.right - rect.left, rect.bottom - rect.top
    else:
        x, y, width, height = (int(v) for v in argv[1:5])
    width = max(1, width)
    height = max(1, height)
    write_png(target, width, height, grab(x, y, width, height))
    print(f"{target}  {width}x{height}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
