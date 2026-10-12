#!/usr/bin/env python
"""Voice changer: what you say comes out of a virtual microphone in another voice.

The two halves already exist separately:

* ``transcribe.py`` puts your speech into ``out/transcript.txt``
* ``speak.py`` turns text into StyleTTS 2 speech on the virtual microphone

This is the bridge that runs them as one thing. It watches the transcript file and
speaks each new sentence, so the loop is:

    microphone -> transcript.txt -> StyleTTS 2 -> CABLE Input -> CABLE Output
                                                                     ^
                                        any app selects this as "microphone"

Voices are files: drop a wav into ``voices/`` and it becomes available. Change the
voice while running by writing its name into ``voices/voice.txt`` -- it is re-read
on every line, so no restart.

    .venv\\Scripts\\python.exe voice_changer.py
    .venv\\Scripts\\python.exe voice_changer.py --voice narrator
    .venv\\Scripts\\python.exe voice_changer.py --transcript out\\transcript.txt --monitor
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from live_transcribe import emotion  # noqa: E402
from live_transcribe import emotion_voices  # noqa: E402
from live_transcribe import voices as voices_mod  # noqa: E402
from live_transcribe import word_timing  # noqa: E402
from live_transcribe import audio_effects, live_settings  # noqa: E402
from live_transcribe.pipeline import PipelineConfig, SpeechPipeline  # noqa: E402
from live_transcribe.tts import TtsClient, TtsError  # noqa: E402
from live_transcribe.virtual_mic import (  # noqa: E402
    CABLE_MIC_NAME,
    CABLE_OUTPUT_NAME,
    VirtualMic,
    VirtualMicError,
)

BANNER = """
  voice-changer  --  your speech -> another voice -> virtual microphone
"""

DEFAULT_TRANSCRIPT = HERE / "out" / "transcript.txt"
VOICE_SELECTOR = "voice.txt"
SETTINGS_FILE = Path("out") / "rixers_settings.json"

MODE_HELP = (
    "how much text each synthesis call gets. 'sentence' (default) waits for "
    "punctuation and sounds most natural; 'word' follows your speech most closely "
    "but is choppier; 'chunk'/'phrase' are in between"
)


class TranscriptFollower:
    """Yields new text appended to a file, coping with it being cleared."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = ""
        #: How long a silence ends the sentence. The control panel owns this: it is
        #: the delay the user hears between finishing a sentence and hearing it
        #: again, so it has to be adjustable without restarting anything.
        self.pause_flush_s = 0.7
        #: Fragments shorter than this are not worth a synthesis call.
        self.min_chars = 1
        self.reset()

    def reset(self) -> None:
        try:
            self.offset = self.path.stat().st_size
        except OSError:
            self.offset = 0
        self.pending = ""

    def read_new(self) -> str:
        """Whatever has been added since the last call (or since a clear)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return ""
        if size < self.offset:
            # The file was truncated: the transcription program restarted.
            self.offset = 0
            self.pending = ""
        if size == self.offset:
            return ""
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self.offset)
                fresh = handle.read()
        except OSError:
            return ""
        self.offset = size
        return fresh


def split_complete_sentences(text: str) -> tuple[list[str], str]:
    """Split off finished sentences, returning (sentences, leftover).

    Speaking mid-sentence would be wrong, and the transcript grows one word at a
    time while the transcriber is still working, so anything without terminal
    punctuation is held back until it is finished.
    """
    import re

    sentences: list[str] = []
    remainder = text
    while True:
        match = re.search(r"([.!?]+)(\s+|$)", remainder)
        if not match:
            break
        end = match.end()
        sentence = remainder[:end].strip()
        remainder = remainder[end:]
        # Very long clauses get their own call; the style guard is per request.
        while len(sentence) > 320:
            cut = sentence.rfind(" ", 0, 320)
            cut = cut if cut > 0 else 320
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)
    return sentences, remainder


def start_transcription(settings, note) -> bool:
    """Start stage one in its own window, if the control panel asked for it.

    The two stages are separate programs that meet at a text file, so "start the
    voice changer" starting the transcriber as well is a convenience, not a
    dependency: a transcriber that is already running, or that fails to start,
    changes nothing about the voice side. Its console is kept because the local
    Whisper engine prints the live guess there, which is the only progress report
    it has.
    """
    import subprocess

    engine = str(settings.get("stt_engine") or "whisper").lower()
    script = HERE / ("google_stt.py" if engine == "google" else "transcribe.py")
    if not script.is_file():
        note(f"  ! could not start transcription: {script.name} is missing")
        return False
    command = [sys.executable or "python", "-u", str(script)]
    command += str(settings.get("stt_args") or "").split()
    try:
        subprocess.Popen(
            command,
            cwd=str(HERE),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    except OSError as exc:
        note(f"  ! could not start transcription: {exc}")
        return False
    note(f"  transcription: started {script.name} in its own window")
    return True


def resolve_voice(selector_path: Path, startup_voice: str | None, directory: Path):
    """The voice to use now: voices/voice.txt wins over the startup choice."""
    choice = startup_voice
    try:
        if selector_path.is_file():
            text = selector_path.read_text(encoding="utf-8").strip()
            if text:
                choice = text
    except OSError:
        pass
    return voices_mod.find(choice, directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voice-changer",
        description="Speak a transcript file out of a virtual microphone in a cloned voice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--transcript", default=str(DEFAULT_TRANSCRIPT),
                        help="file written by transcribe.py")
    parser.add_argument("--voice", default=None, help="voice from the voices folder")
    parser.add_argument("--voices-dir", default=str(voices_mod.VOICES_DIR))
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--device", default=None,
                        help="playback device for the virtual mic (default: CABLE Input)")
    parser.add_argument("--settings", default=str(HERE / SETTINGS_FILE),
                        help="the file the control panel writes its settings to")
    parser.add_argument("--mode", default="sentence", choices=["word", "chunk", "phrase", "sentence"],
                        help=MODE_HELP)
    parser.add_argument("--target-words", type=int, default=0,
                        help="words per synthesis call (0 = derived from --mode)")
    parser.add_argument("--max-words", type=int, default=0,
                        help="hard cap before a chunk is released (0 = derived)")
    parser.add_argument("--queue", type=int, default=12,
                        help="chunks of finished audio kept ready to play")
    parser.add_argument("--backlog", type=int, default=16,
                        help="total queued work before the oldest is dropped (keeps the voice current)")
    parser.add_argument("--gap-ms", type=float, default=0.0,
                        help="extra silence between chunks (0 = the model's own pacing)")
    parser.add_argument("--monitor", action="store_true", help="also hear it on your speakers")
    parser.add_argument("--min-chars", type=int, default=1,
                        help="do not speak fragments shorter than this")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="speak the file's current contents, then exit (for testing)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds of watching (0 = run until Ctrl+C)")
    parser.add_argument("--pause-flush-s", type=float, default=0.7,
                        help="speak held text after this long with nothing new (a pause). "
                             "Kept short on purpose: it is dead time before you hear yourself, "
                             "and the transcriber has normally committed by then anyway")
    parser.add_argument("--word-delay", default="auto", choices=["auto", "on", "off"],
                        help="replay the delays measured between your words. 'auto' obeys the "
                             "Start/Stop button in the browser page; 'on'/'off' override it")
    parser.add_argument("--word-delay-min-ms", type=float, default=80.0,
                        help="shortest measured delay worth a synthesis call. Each replayed "
                             "delay means one more call, so this is the latency dial")
    parser.add_argument("--word-delay-max-ms", type=float, default=word_timing.MAX_GAP_MS,
                        help="cap on a replayed delay, so a hesitant speaker does not stall it")
    parser.add_argument("--emotion", dest="emotion", action="store_true", default=True,
                        help="match the voice to how you sounded (default: on)")
    parser.add_argument("--no-emotion", dest="emotion", action="store_false",
                        help="ignore emotion; always use the selected voice")
    parser.add_argument("--utterance-wav",
                        default=str(Path(__file__).resolve().parent / "out" / "utterance.wav"),
                        help="utterance audio written by the transcriber (for emotion)")
    parser.add_argument("--emotion-tags", default=str(Path(__file__).resolve().parent / "out" / "emotion.txt"),
                        help="where to record the detected emotion")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="shift the finished voice by this many semitones "
                             "(-12..12). Positive is higher and slightly faster")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="output gain, soft-limited so it does not clip (1.0 = unchanged)")
    parser.add_argument("--gate-db", type=float, default=-70.0,
                        help="treat anything quieter than this as silence (-70 = off)")
    parser.add_argument("--gate-floor-db", type=float, default=-60.0,
                        help="how far down the gated stretches are pushed")
    parser.add_argument("--no-effects", action="store_true",
                        help="ignore pitch/volume/gate entirely (cleanest signal path)")
    args = parser.parse_args(argv)
    # What the command line left alone is what the browser page is allowed to
    # change; an explicit option here wins over the page's setting.
    default_min_gap = parser.get_default("word_delay_min_ms")
    default_max_gap = parser.get_default("word_delay_max_ms")

    voices_dir = Path(args.voices_dir).expanduser()
    transcript = Path(args.transcript).expanduser()

    # -- the control panel's settings -------------------------------------- #
    # The panel (rixers_gui.py) writes these before it starts this process, and
    # keeps writing them while it runs. Everything the running process can honour
    # without restarting is re-read from here on every pass of the watch loop.
    settings = live_settings.LiveSettings(args.settings)
    settings.load()
    if not settings.path.exists():
        try:
            settings.save()
        except OSError as exc:
            if not args.quiet:
                print(f"  note: could not create {settings.path}: {exc}", file=sys.stderr)

    def setting(name: str, fallback):
        """A setting from the panel, with the command line as the fallback."""
        value = settings.get(name)
        return fallback if value is None else value

    def setting_voice() -> str | None:
        return str(settings.get("voice") or "").strip() or None

    if args.list_voices:
        found = voices_mod.discover(voices_dir)
        if not found:
            print(f"no voices in {voices_dir}")
            return 1
        for voice in found:
            print(f"  {voice.describe()}")
        return 0

    if not args.quiet:
        print(BANNER)
        print(f"  transcript : {transcript}")
        print(f"  voices     : {voices_dir}")
        print(f"  virtual mic: apps record from '{CABLE_MIC_NAME}'")

    client = TtsClient(host=args.host, port=args.port)
    if not args.quiet:
        print("\n  waiting for the speech server ...")
    try:
        info = client.wait_until_ready(timeout_s=240.0)
    except TtsError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print("Start it first:  run-tts.bat", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"  speech server: {info.get('backend')} on {info.get('device')}")

    # -- the audio shaping the panel controls ------------------------------ #
    options = audio_effects.AudioOptions(
        pitch_semitones=setting("pitch_semitones", args.pitch),
        volume=setting("volume", args.volume),
        gate_db=setting("gate_db", args.gate_db),
        gate_floor_db=setting("gate_floor_db", args.gate_floor_db),
    )
    if args.no_effects:
        options = audio_effects.AudioOptions()
    effects_on = not args.no_effects
    monitor_on = args.monitor or bool(settings.get("monitor"))

    from live_transcribe.virtual_mic import find_device

    def locate_device():
        if not args.device:
            return None
        found = find_device(args.device, output=True)
        if found is None:
            print(f"no playback device matching {args.device!r}", file=sys.stderr)
        return found

    mic: VirtualMic | None = None
    try:
        device = locate_device()
        if args.device and device is None:
            return 3
        mic = VirtualMic(
            device=device,
            monitor=monitor_on,
            rate_scale=options.pitch_ratio if effects_on else 1.0,
        )
        mic.open()
    except VirtualMicError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
    if options.is_default:
        device_note = "no effects"
    else:
        device_note = options.describe()
    if not args.quiet:
        print(f"  sound      : {device_note}"
              f"  (stream at {mic.device_rate} Hz, {CABLE_OUTPUT_NAME})")

    # The pipeline synthesizes ahead of playback in its own thread, so the voice
    # follows the transcript instead of pausing to think between every phrase.
    config = PipelineConfig(
        mode=setting("mode", args.mode),
        target_words=setting("target_words", args.target_words),
        max_words=setting("max_words", args.max_words),
        queue_size=setting("queue", args.queue),
        pending_limit=setting("backlog", args.backlog),
        gap_ms=setting("gap_ms", args.gap_ms),
        trim_silence_ms=setting("trim_silence_ms", 220.0),
        retries=setting("retries", 1),
        word_delay_replay=setting("word_delay", args.word_delay) != "off",
        word_gap_min_ms=setting("word_gap_min_ms", args.word_delay_min_ms),
        word_gap_max_ms=setting("word_gap_max_ms", args.word_delay_max_ms),
        rate_scale=options.pitch_ratio if effects_on else 1.0,
        apply_effects=effects_on,
    )

    def note(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    # -- the word-delay switch --------------------------------------------- #
    # Three states, because two things want to control it: this command line, and
    # the Start/Stop button in the browser page (which cannot see this process).
    #   on   - replay the delays, whatever the page says
    #   off  - never replay them
    #   auto - the page decides; with no page server running, on
    timing_state: dict = {
        "mode": "auto",
        "enabled": True,
        "min_gap_ms": args.word_delay_min_ms,
        "max_gap_ms": args.word_delay_max_ms,
    }
    timing_checked = 0.0

    def refresh_timing_setting(force: bool = False) -> None:
        """Re-read the page's setting, at most a few times a second.

        The gap window follows the page too, so changing it there takes effect
        here -- but only when this command line left it at the default, so an
        explicit ``--word-delay-min-ms`` is not silently overruled.
        """
        nonlocal timing_checked
        now = time.time()
        if not force and now - timing_checked < 0.5:
            return
        timing_checked = now
        answer = client.timing()
        if not answer:
            return
        if answer.get("mode") != timing_state.get("mode"):
            note(f"  word delays: {answer['mode']} (set in the browser page)")
        timing_state.update(answer)
        if args.word_delay_min_ms == default_min_gap:
            config.word_gap_min_ms = float(answer.get("min_gap_ms", config.word_gap_min_ms))
        if args.word_delay_max_ms == default_max_gap:
            config.word_gap_max_ms = float(answer.get("max_gap_ms", config.word_gap_max_ms))
        pipe.delay_settings.min_gap_ms = config.word_gap_min_ms
        pipe.delay_settings.max_gap_ms = config.word_gap_max_ms
    # The control panel, when it is driving, wins over the page: it is the window
    # the user is looking at, and `auto` is what it writes when it has no opinion.
    watcher: live_settings.SettingsWatcher
    started: list[bool] = [False]

    def apply_live_settings(changed: list[str], wanted: live_settings.LiveSettings) -> None:
        """Take on whatever the control panel changed while we were speaking.

        Called from the watch loop, so it is already on the right thread -- the same
        one that feeds text to the pipeline. Nothing here may block for long: a
        pitch change rebuilds the output stream, which is the one slow case, and it
        is announced so a brief gap in the audio is not a mystery.
        """
        nonlocal config, follower, mic

        if "pause_flush_s" in changed:
            follower.pause_flush_s = float(wanted.get("pause_flush_s"))
            note(f"  pause flush: {follower.pause_flush_s:.2f}s")
        if "min_chars" in changed and hasattr(follower, "min_chars"):
            follower.min_chars = int(wanted.get("min_chars"))
        if "mode" in changed:
            config.mode = str(wanted.get("mode"))
            note(f"  mode: {config.mode}")
        for name, attribute in (
            ("target_words", "target_words"),
            ("max_words", "max_words"),
            ("queue", "queue_size"),
            ("backlog", "pending_limit"),
            ("gap_ms", "gap_ms"),
            ("trim_silence_ms", "trim_silence_ms"),
            ("retries", "retries"),
        ):
            if name in changed:
                setattr(config, attribute, wanted.get(name))
        if {"mode", "target_words", "max_words"} & set(changed):
            pipe.chunker = type(pipe.chunker)(config.mode, config.target_words, config.max_words)

        # Rhythm. The panel's own switch is read directly on the next chunk, so
        # only the window and the command-line override are handled here.
        gap_window = {"word_gap_min_ms", "word_gap_max_ms"} & set(changed)
        if gap_window:
            config.word_gap_min_ms = float(wanted.get("word_gap_min_ms"))
            config.word_gap_max_ms = float(wanted.get("word_gap_max_ms"))
            pipe.delay_settings.min_gap_ms = config.word_gap_min_ms
            pipe.delay_settings.max_gap_ms = config.word_gap_max_ms
            note(f"  replaying gaps of {config.word_gap_min_ms:.0f}"
                 f"-{config.word_gap_max_ms:.0f} ms")

        # Sound. Pitch is a stream rate, so it can only change by reopening the
        # cable; volume and the gate are read per clip and need nothing done.
        if {"pitch_semitones", "volume", "gate_db", "gate_floor_db"} & set(changed):
            pipe.audio.pitch_semitones = float(wanted.get("pitch_semitones"))
            pipe.audio.volume = float(wanted.get("volume"))
            pipe.audio.gate_db = float(wanted.get("gate_db"))
            pipe.audio.gate_floor_db = float(wanted.get("gate_floor_db"))
            wanted_scale = pipe.audio.pitch_ratio if effects_on else 1.0
            # The first pass is start-up: the stream was opened a moment ago at
            # exactly this rate and has nothing playing, so reopening it there would
            # only be noise in the log (and a closed device if it failed).
            if abs(wanted_scale - config.rate_scale) > 1e-6 and started[0]:
                note("  pitch changed - reopening the virtual microphone ...")
                was_enabled = replay_word_delays()
                pipe.stop(drain=True)
                mic.close()
                try:
                    mic = VirtualMic(
                        device=locate_device(), monitor=monitor_on, rate_scale=wanted_scale
                    )
                    mic.open()
                except VirtualMicError as exc:
                    print(f"  ! could not reopen the virtual microphone: {exc}", file=sys.stderr)
                    return
                config.rate_scale = wanted_scale
                pipe.mic = mic
                pipe.start()
                pipe.delay_settings.enabled = was_enabled
                note(f"  sound      : {pipe.audio.describe()} "
                     f"(stream at {mic.device_rate} Hz)")

    def replay_word_delays() -> bool:
        """Whether to put the speaker's own delays back into the speech.

        The panel's setting wins when it is 'auto' (the default it writes): it is
        the window the user is looking at, and it has no way to change a running
        command line. 'on'/'off' on the command line still override it.
        """
        if args.word_delay == "on":
            return True
        if args.word_delay == "off":
            return False
        mode = str(settings.get("word_delay", "auto"))
        if mode in ("on", "off"):
            return mode == "on"
        refresh_timing_setting()
        return bool(timing_state.get("enabled", True))

    def resolve_current_voice():  # noqa: ANN202
        """Re-read voices/voice.txt before each chunk, so voice changes are live.

        The control panel writes the same file, so a voice picked in the window is
        in force on the next sentence; its saved setting is the fallback for when
        no voice has been picked yet.
        """
        try:
            return resolve_voice(voices_dir / VOICE_SELECTOR, setting_voice() or args.voice, voices_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if first_voice[0] is None:
                print(f"  voice problem: {exc}", file=sys.stderr)
                first_voice[0] = False
            return None

    first_voice: list = [None]
    pipe = SpeechPipeline(client, mic, config, logger=note)
    pipe.set_voice_resolver(resolve_current_voice)
    # Asked before every chunk, so flipping the switch takes effect on the next
    # sentence rather than on the next restart.
    pipe.set_timing_resolver(replay_word_delays)

    # -- emotion matching -------------------------------------------------- #
    # How you sounded is measured from the utterance audio the transcriber writes,
    # not from the text: pitch, loudness and timing carry essentially all of it.
    emotion_map = emotion_voices.load(voices_dir)
    tracker: emotion.EmotionTracker | None = None
    if args.emotion and args.utterance_wav:
        tracker = emotion.EmotionTracker(
            Path(args.utterance_wav),
            tags_path=Path(args.emotion_tags) if args.emotion_tags else None,
            logger=note,
        )
        # Ignore any utterance left over from a previous run.
        tracker.prime()
        if not args.quiet:
            if emotion_map.emotions_covered:
                print(f"  emotion    : {emotion_map.describe()}")
            else:
                print(
                    "  emotion    : detected and shown, but no emotion voices are set up - "
                    f"put wavs in {emotion_voices.EMOTION_DIR_NAME}\\ to hear a difference"
                )

    current: dict = {"emotion": "", "reference": None}

    def choose_reference(_job) -> object:  # noqa: ANN001
        """Pick the voice for this chunk from the emotion last measured.

        Returning None keeps whatever voice `voice.txt` selected, so with no emotion
        voices installed this changes nothing at all.
        """
        return current["reference"]

    pipe.set_reference_resolver(choose_reference if tracker is not None else None)
    pipe.start()

    # Apply the starting voice before anything is spoken, so the first words are
    # already in the right voice.
    try:
        starting = voices_mod.find(args.voice, voices_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  voice problem: {exc}", file=sys.stderr)
        pipe.stop(drain=False)
        mic.close()
        return 4
    first_voice[0] = starting.slug
    try:
        client.set_voice(starting.path)
        if not args.quiet:
            print(f"  voice: {starting.slug}  ({starting.seconds:.1f}s reference)")
    except TtsError as exc:
        print(f"  could not set the starting voice: {exc}", file=sys.stderr)

    follower = TranscriptFollower(transcript)
    follower.pause_flush_s = float(setting("pause_flush_s", args.pause_flush_s))
    follower.min_chars = int(setting("min_chars", args.min_chars))

    # Now that the pipeline, the follower and the microphone all exist, hand them
    # whatever the control panel had already written -- its settings still have to
    # reach the parts that the command line built. Done here rather than earlier
    # because this is the first moment there is a follower to set a pause on.
    watcher = live_settings.SettingsWatcher(settings, apply_live_settings)
    apply_live_settings(list(live_settings.LIVE_FIELDS), settings)
    config.word_delay_replay = settings.get("word_delay", "auto") != "off"
    follower.pause_flush_s = float(setting("pause_flush_s", args.pause_flush_s))
    follower.min_chars = int(setting("min_chars", args.min_chars))
    started[0] = True

    # Stage one, if the control panel asked for it. Started here rather than before
    # the pipeline so a failure to start it cannot stop the voice side coming up.
    if settings.get("stt_start"):
        start_transcription(settings, note)
    if args.once:
        try:
            text = transcript.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"could not read {transcript}: {exc}", file=sys.stderr)
            pipe.stop(drain=False)
            mic.close()
            return 4
        pipe.say(text)
        pipe.stop(drain=True)
        mic.drain()
        mic.close()
        print(f"\n  {pipe.summary()}")
        return 0 if pipe.stats.chunks_spoken else 5

    if not args.quiet:
        print(f"\n  mode       : {config.mode} ({config.target_words or 'auto'} words per call)")
        if args.word_delay == "auto":
            print(f"  word delays: {settings.get('word_delay', 'auto')} "
                  f"(replaying gaps of {config.word_gap_min_ms:.0f}-{config.word_gap_max_ms:.0f} ms)")
        else:
            print(f"  word delays: {args.word_delay} "
                  f"(replaying gaps of {config.word_gap_min_ms:.0f}-{config.word_gap_max_ms:.0f} ms)")
        print(f"  pause flush: {follower.pause_flush_s:.2f}s of silence ends your sentence")
        print(f"  listening  : {transcript} - speak into your transcription program.")
        print(f"  settings   : {settings.path} (the control panel edits this live)")
        print("  Ctrl+C to stop.\n")

    deadline = time.time() + args.duration if args.duration > 0 else None
    last_fresh = time.time()
    try:
        while True:
            if deadline is not None and time.time() >= deadline:
                if not args.quiet:
                    print(f"\n  reached --duration {args.duration:.0f}s - stopping")
                break
            time.sleep(0.2)
            # Pick up anything the control panel changed while we were speaking.
            watcher.poll()

            # Read how the last utterance sounded. Done before the flush below so the
            # text that was just committed is spoken in the voice its own audio calls
            # for, rather than one utterance behind.
            if tracker is not None:
                reading = tracker.poll()
                if reading is not None:
                    chosen = emotion_map.get(reading.emotion)
                    current["emotion"] = reading.emotion
                    current["reference"] = chosen
                    pipe.set_emotion_note(reading.tag)
                    if not args.quiet:
                        print(emotion.format_emotion_line(reading), flush=True)
                        if chosen is not None and chosen.stem != reading.emotion.replace(" ", "_"):
                            print(f"                 using {chosen.name}", flush=True)

            fresh = follower.read_new()
            if fresh:
                last_fresh = time.time()
                pipe.feed(fresh)
                continue
            # Nothing new for a moment: the speaker has paused, which is where the
            # transcriber commits. Speak whatever the chunker is holding, or a short
            # utterance would sit unspoken forever waiting for a size threshold.
            if time.time() - last_fresh >= follower.pause_flush_s:
                if pipe.flush_partial() and not args.quiet:
                    print("  (pause)")
                last_fresh = time.time()
    except KeyboardInterrupt:
        print("\n  stopping - finishing what is queued")
    finally:
        # Whatever is still buffered is unfinished speech; speak it rather than
        # dropping the last thing that was said.
        pipe.flush()
        pipe.stop(drain=True)
        try:
            mic.drain(250)
        finally:
            mic.close()

    print(pipe.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
