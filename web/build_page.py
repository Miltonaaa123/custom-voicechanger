"""Rewrite the page's <script> block from a separate source file.

Keeping the JavaScript in its own file makes it editable and lintable, then this
script splices it into index.html between the script tags. Run it after editing
web/app.js.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "index.html"
APP = HERE / "app.js"


def main() -> int:
    if not APP.exists():
        print(f"missing {APP}")
        return 1
    html = HTML.read_text(encoding="utf-8")
    js = APP.read_text(encoding="utf-8").rstrip() + "\n"

    pattern = re.compile(r"(<script>)(.*?)(</script>)", re.S)
    if not pattern.search(html):
        print("no <script> block found in index.html")
        return 1
    updated = pattern.sub(lambda m: f"{m.group(1)}\n{js}{m.group(3)}", html, count=1)
    HTML.write_text(updated, encoding="utf-8")
    print(f"spliced {len(js)} chars of JS into {HTML.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
