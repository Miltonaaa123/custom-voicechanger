"""Live microphone -> text transcriber built on faster-whisper.

Continuously captures microphone audio, transcribes it incrementally on the
GPU, and maintains a plain-text file that always holds what was just said.
"""
