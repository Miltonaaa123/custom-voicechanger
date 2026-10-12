#!/usr/bin/env python
"""Entry point:  python transcribe.py [options]

Examples
--------
    python transcribe.py                      # default mic, GPU, writes out/transcript.txt
    python transcribe.py --list-devices       # find your microphone index
    python transcribe.py --input-device 39    # use a specific mic
    python transcribe.py --gain-db 18         # boost a quiet microphone
    python transcribe.py --model small.en     # try the larger model
    python transcribe.py --doctor             # check CUDA / cuDNN setup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_transcribe.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
