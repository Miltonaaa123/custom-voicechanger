# Voice changer: your speech → another voice → virtual microphone

Two complete stages, both working:

| Stage | What it does | Start it with |
| --- | --- | --- |
| **1. Speech → text** | Your voice becomes text in `out\transcript.txt` | `run-google.bat` (Google, accurate) or `run.bat` (local Whisper) |
| **2. Text → voice → mic** | That text is spoken in a cloned voice into a **virtual microphone** | `start-all.bat`, or `run-tts.bat` then `run-voice-changer.bat` |

**New here? Use `RixersVoiceChanger.bat`.** It is a native Windows control panel
for both stages — pick the voice, set how long to wait after you stop speaking,
tune the effects, and press Start. See
[The control panel](#the-control-panel-recommended) below.

```
 your mic ──▶ speech-to-text ──▶ out\transcript.txt
                                        │
                                        ▼
                              StyleTTS 2 (cloned voice)
                                        │
                                        ▼
                          CABLE Input ──▶ CABLE Output
                                                │
                        any app selects this as its "microphone"
```

## Quick start

```
start-all.bat
```

That one file starts everything needed and **closes its own window when done**:

1. checks the environment (paths, GPU, virtual cable, voices) and stops if
   anything is missing;
2. starts the speech engine in its own window;
3. **waits until the engine reports itself loaded**, then says
   `ready after 19s (cuda, 1084 MB VRAM, voice: default.wav)` — a definite moment,
   instead of guessing whether a silent console is still loading or already dead;
4. starts the voice changer in its own window;
5. prints `EVERYTHING IS UP` and closes the launcher.

Two windows remain: **`styletts2 voice server`** and **`voice-changer`**. Closing
the first frees the GPU. Then start the microphone stage with `run-google.bat` (or
`run.bat`), speak, and point Discord/OBS at
`CABLE Output (VB-Audio Virtual Cable)`.

| Option | Effect |
| --- | --- |
| `start-all.bat --no-preflight` | skip the environment check |
| `start-all.bat --no-voice-changer` | start only the speech engine |
| `start-all.bat --voice narrator` | start with a particular voice |
| `start-all.bat --mode chunk` | how much text per synthesis call |

To shut it all down again (frees the GPU and releases the virtual microphone):

```
stop-all.bat
```

That stops the speech engine and the voice changer, matched on their exact
command lines so nothing else on the machine is touched. It deliberately leaves
the microphone → text stage alone, since that may be a browser tab still in use.

**If a window looks empty, that is normal at first** — the speech engine prints
nothing for 20–60 s while it loads a 735 MB checkpoint onto the GPU. Both
`run-tts.bat` and `run-voice-changer.bat` now run a readiness check and end with an
explicit `READY` line. To check things on their own:

```
.venv\Scripts\python.exe launch.py check       # paths, GPU, cable, voices
.venv\Scripts\python.exe launch.py wait-tts    # block until the engine is loaded
```

## Setup

```
Setup.bat
```

Clone, run `Setup.bat`, done. It installs everything and proves it works:

1. **This project's environment** — `.venv` with faster-whisper, sounddevice, numpy.
2. **The speech engine** — StyleTTS 2 in its *own* Python 3.11 environment under
   `engine/`, because its dependency chain cannot be installed on 3.12+.
3. **The models** — about 1 GB, downloaded once into `engine/`.
4. **A test sentence** — loads the model and synthesizes a line, so "installed"
   means "it spoke", not "the files are present".

Nothing is installed globally and nothing outside this folder is touched. Safe to
run again at any time: finished steps are skipped and missing ones are repaired.

| Option | Effect |
| --- | --- |
| `Setup.bat --torch cpu` | no usable GPU (slow, but always works) |
| `Setup.bat --torch cu118` | a driver older than CUDA 12.1 |
| `Setup.bat --reuse-legacy` | adopt an existing Riko StyleTTS 2 install instead of downloading |
| `Setup.bat --skip-verify` | install without the test sentence |
| `Setup.bat check` | report what is installed, change nothing |

Then:

```
RixersVoiceChanger.bat
```

Requirements: **Windows**, and **Python 3.11** on PATH for the engine (the app's own
environment is built from whatever Python runs `Setup.bat`). If 3.11 is missing,
setup says so and stops before downloading anything.

### Why two environments

The voice changer needs `faster-whisper`, `sounddevice` and `numpy`. StyleTTS 2 needs
`transformers<5`, which pins `tokenizers<0.20`, and those releases publish no wheel
for Python 3.12+ — pip falls back to building `tokenizers` from source and that build
fails. So the model lives in its own interpreter, `engine/styletts2/.venv`, and
`live_transcribe/engine_tts_server.py` is started with that interpreter directly. The
two never share a process.

### What gets downloaded, and where

| File | Size | Why |
| --- | --- | --- |
| `engine/styletts2/epochs_2nd_00020.pth` | 736 MB | the voice model |
| `engine/styletts2/config.yml` | 2 KB | its config |
| `engine/models/epochs_00080.pth` | 90 MB | text aligner |
| `engine/models/bst.t7` | 20 MB | pitch extractor |
| `engine/models/PLBERT/step_1000000.t7` | 24 MB | style encoder |
| `engine/models/PLBERT/config.yml` | 1 KB | its config |

Everything under `engine/` is downloaded, never committed — see `.gitignore`.

### Three library defects the engine patches at startup

`live_transcribe/engine_tts_server.py` uses the `styletts2` package from PyPI, which
has three defects that all present the same way: the model appears to load and then
never speaks. Each is patched **in memory** at start-up — nothing on disk is changed
and reinstalling the package restores the original. All three are documented at the
patch site:

* **`torch.load`** — PyTorch 2.6 flipped `weights_only` to True by default and
  StyleTTS 2 predates that, so its checkpoint fails to unpickle.
* **An inverted path check** — `if F0_path or not Path(F0_path).exists():` lost its
  `not`, so the pitch extractor named in the config is *never* used and the library
  tries to download it instead. With no network that is a hang, not an error.
* **A string path join** — `load_plbert` does `log_dir / f"step_{iters}.t7"` where
  `log_dir` is a `str`, which raises the moment the config supplies `PLBERT_dir`.

The reference voice is also read with `soundfile` rather than `librosa.load`. That is
not a library bug: importing librosa's core submodules hangs on some machines (see
[Troubleshooting](#troubleshooting)), and `soundfile` plus the library's own torchaudio
mel transform produce exactly the same style vector.

## The control panel (recommended)

```
RixersVoiceChanger.bat
```

That opens **Rixers Custom Voice Changer** — a real Windows window, not a browser
tab. Everything a settings file can change is on it, and it has its own
**Start** / **Stop** buttons for the voice changer, so this is now the shortest
path from nothing to a working voice:

1. **Dashboard → Speech engine** starts the StyleTTS 2 server (60-90 s to load).
2. **Voices** picks which voice to clone, previews it, and imports new wavs.
3. **Dashboard → Start Voice Changer** runs the voice changer and shows its
   console output live in the window.
4. **Transcription → Start transcription now** runs stage one, or tick
   *Start transcription with the voice changer* so one button starts both.

The window is laid out the way the reference design was: a sidebar with the
settings spread across eight sections, a header with **Reset Defaults** and
**Save Changes**, and a status readout in the bottom-right corner.

| Section | What lives there |
| --- | --- |
| **Dashboard** | status, Start / Stop, a one-shot test, and the live console |
| **Voices** | which voice is cloned, the library with lengths, importing, emotion → voice mapping |
| **Timing & Rhythm** | **how long to wait after you stop speaking**, shortest fragment worth speaking, replaying your own word delays, the backlog |
| **Speech Engine** | words per synthesis call (mode + overrides), gap, silence trimming, retries |
| **Audio & Devices** | the output device (VB-CABLE), monitoring, the signal chain |
| **Effects** | pitch, volume, noise gate |
| **Transcription** | Whisper or Google, extra arguments, the transcript file |
| **Advanced** | raw engine arguments, every path in use, the preflight check, factory reset |

### How it talks to the voice changer

They are separate processes that meet at one file:

```
out\rixers_settings.json
```

The window writes it; the voice changer re-reads it between sentences. So:

* a **live** setting (pause delay, rhythm, chunking, volume, gate, pitch) takes
  effect on the **next sentence** — no restart, no lost GPU load;
* a **start-up** setting (the output device, monitoring, the transcription engine)
  is in force the next time the voice changer starts;
* the file is plain JSON and safe to hand-edit — anything out of range or
  misspelled falls back to its default rather than reaching the audio.

`live_transcribe/live_settings.py` is the schema both sides agree on, and it is
also the list of what is adjustable while running.

### Effects, and what is deliberately missing

`live_transcribe/audio_effects.py` shapes each finished clip on its way into the
cable:

* **Pitch** is a playback rate, so shifting it also changes how long the clip
  lasts — up is higher *and* faster, which is the classic voice-changer sound.
  Changing it reopens the virtual microphone, so there is a short gap.
* **Volume** is a gain with the peaks soft-limited, so past 1.0 it gets louder
  rather than clipping.
* **The gate** fades everything below a threshold down to a floor, so a quiet room
  sounds silent instead of faintly hissy.

Reverb, EQ, formant shifting and "AI noise suppression" are **not** implemented.
They would need the speech server to expose parameters it does not have, and a
slider that changes nothing is worse than no slider — the Effects page says so
rather than pretending.

## Stage 2 quick start (by hand)

1. `run-tts.bat` — loads the voice engine and confirms when it is ready.
2. `run-voice-changer.bat` — watches the transcript and speaks it; also prints
   `READY` once connected to the engine.
3. In Discord / OBS / a game, set the **microphone** to
   `CABLE Output (VB-Audio Virtual Cable)`.

**Why this is a microphone at all:** VB-CABLE is a loopback pair. Anything written
to the playback device `CABLE Input` appears at the capture device
`CABLE Output`, which Windows and every application treat as an ordinary
microphone. So the voice changer is really playing audio to the right device. That
routing is already installed and verified on this machine
(`tools\vbcable_check.py` plays a tone in one end and captures it from the other).

## Matching how you sounded (emotion)

The voice changer listens to *how* you said something, not just what, and picks a
different cloned voice for it. Each finished utterance is measured for pitch,
loudness, energy variation and brightness, and turned into an emotion with a
confidence and an intensity:

```
  [happy]        happy      pitch   302 Hz (D4), arousal +0.11, valence +0.57, intensity 0.34
  [sad]          sad        pitch   121 Hz (B2), arousal -0.75, valence -0.88, intensity 0.81
```

**How the emotion reaches the voice.** The audio is measured, then a different
reference wav is cloned for that sentence. This is the honest mechanism: **this
checkpoint has no emotion tags** — the server does not parse `[very happy]`, and
StyleTTS 2 conditions only on reference audio plus a few scalars. Worse, the scalars
are weak: measured on this machine, `embedding_scale`, `alpha` and `beta` moved
energy and pitch by under 10% (see `tools/style_probe.py`), while swapping the
reference audibly changes the voice. So emotion is transferred by **choosing a
voice**, which is also exactly what you asked for: a voice per emotion.

### Setting it up

Put one wav per emotion in `voices\emotions\`, named after the emotion:

```
voices\emotions\happy.wav        voices\emotions\sad.wav
voices\emotions\angry.wav        voices\emotions\neutral.wav
voices\emotions\very_happy.wav   voices\emotions\calm.wav
```

That is the whole setup — same 3–16 s rules as any other voice. Record each one
*while sounding like that emotion*; the clone copies the delivery, which is the
point. `voices\emotions.json` assigns a voice explicitly if you would rather point
at files elsewhere:

```json
{ "happy": "emotions/happy.wav", "sad": "C:/voices/me_tired.wav" }
```

**You do not need all eight.** `very happy` falls back to `happy`, `calm` to
`neutral`, and so on, and any emotion with no voice keeps the voice you selected —
so one file works, and no files means the feature is simply inert rather than
broken. It says which it found at startup:

```
  emotion    : happy <- happy.wav, neutral <- neutral.wav
  emotion    : detected and shown, but no emotion voices are set up - put wavs in emotions\ to hear a difference
```

The current emotion is also written to `out\emotion.txt` as a `[tag]` for other
tools. `--no-emotion` turns the whole thing off.

### What it can and cannot tell

Acoustics measure **arousal** (calm ↔ excited) well and **valence** (negative ↔
positive) poorly. This reads arousal from loudness, how much the loudness swings
*while you are actually speaking*, pitch movement, and brightness; valence from
pitch height, movement and brightness. Bands are calibrated so ordinary speech
lands on `neutral` and only real variation moves it.

Two limits worth stating rather than hiding:

* **Valence is a weak signal.** Text sentiment would be far better at it, and this
  gets no text signal at all — it is deliberately audio-only, because that is what
  the request was.
* **Pitch differs by speaker**, so an identical delivery reads as more positive for
  a higher-pitched voice. Normalising per speaker would need a calibration step the
  system does not have. The bands are wide enough that ordinary speech still lands
  on `neutral` for both, but a quiet, high voice can read happier than it sounds.

`tools\style_probe.py` is the measurement behind the "no emotion tags" claim, and
`tests\test_emotion.py` (14 tests) pins the feature extraction, including recovering
a known pitch.

## Adding and changing voices

**A voice is a file.** Every `*.wav` in `voices\` is a voice — drop one in and it
appears. Nothing is trained, so a new voice works immediately.

**To change voice while it is running, use the picker** — no typing required:

```
pick-voice.bat
```

That opens a small always-on-top window listing every voice. Select one and press
**Use this voice** (or double-click it); **Hear it** speaks a test line so you can
check before using it. The status line shows what you picked, what `voice.txt`
says, and which voice the engine is *actually* cloning — so a switch that did not
take is visible immediately.

It works by writing `voices\voice.txt`, which the voice changer re-reads before
every sentence. Nothing is restarted, and the change lands on your next sentence.
`start-all.bat --with-picker` opens it as part of startup.

| To do this | Do that |
| --- | --- |
| add a voice | copy a 3–16 s wav of clean speech into `voices\` |
| make it the default | name it `voices\default.wav` |
| **switch voice while running** | `pick-voice.bat`, or write the name into `voices\voice.txt` |
| describe a voice | edit `voices\voices.json` (name, transcript, tags) |

Requirements for a good reference clip: 3–16 seconds, one speaker, no music, no
reverb, no long silences. The folder is checked at startup and anything unusable is
reported with the reason (`only 1.2s`, `22.0s is over the limit`, `not a readable
wav`) rather than failing mid-sentence. `voices\README.txt` has the full guide.

## How much text per synthesis call

Each sentence is synthesized as a unit and sent to the virtual microphone. A
background thread does the synthesizing while the previous sentence is still
playing, so there is no silent gap while the GPU works:

```
transcript ──▶ chunker ──▶ synthesis thread ──▶ play queue ──▶ virtual mic
                             (runs ahead)            │
                                                     └─ player thread, never waits
```

The default is **`sentence`**: nothing is spoken until the sentence ends, which
gives the model the most context and therefore the most natural delivery. Two
things can end a chunk:

* a **sentence terminator** — `.`, `!` or `?` — which is the natural break; or
* a **pause**, plus a size limit as a final fallback.

Both fallbacks are load-bearing, and one of them was a real bug. **Speech
recognition frequently returns text with no punctuation at all** — Chrome's Web
Speech API often does exactly that. A chunker waiting strictly for a full stop then
releases nothing at all: the transcript fills up, and the virtual microphone stays
silent forever. Measured on a real 31-word unpunctuated transcript: no audio was
produced. It now speaks after a 0.7 s pause (`--pause-flush-s`) or 24 words,
whichever comes first. The pause is deliberately short: it is dead time between you
finishing a sentence and hearing it, and the transcriber has normally committed by
then anyway.

| `--mode` | Speaks | Latency | Naturalness |
| --- | --- | --- | --- |
| `sentence` | whole sentence, or on a pause — **default** | waits for punctuation or a pause | most natural |
| `phrase` | ~8 words | medium | smoother |
| `chunk` | ~4 words | short | good balance |
| `word` | one word per call | closest | choppiest; little context per call |

```
run-voice-changer.bat                    (sentence mode, the default)
run-voice-changer.bat --mode chunk       (groups a few words)
run-voice-changer.bat --mode word        (follows speech most closely)
```

### Word delays: speaking at your own pace

How long you leave between words is part of how you speak. "hello what is going on"
is not the same rhythm as "hello … what … is going on", and a voice changer that
throws the timing away hands you back a sentence delivered at a pace you never used.
So the delay in front of every word is measured, and the ones worth replaying are
put back when the clone speaks.

**Measuring.** The page reports words, not their timing — Chrome's
`SpeechRecognitionResult.timestamp` is not implemented — so what is measured is the
interval between words *arriving*, minus the service's own reporting delay. That
delay is estimated from the shortest interval in the same phrase, on the reasoning
that you cannot have paused for less than the service takes to report a word. The
local Whisper engine (`run.bat`) needs none of that: it gets a real start and end
time for every word, so its measurements are exact.

**Carrying.** The transcriber and the voice changer are separate processes that meet
at a file, so a delay travels *inside the text*, in front of the word it belongs to:

```
hello [[250]] what is [[400]] going on
```

`live.txt`, the file the voice changer follows, carries them. `transcript.txt` never
does: that is the file a person reads, and it stays plain words.

**Replaying costs a synthesis call per delay.** That is the whole trade-off. A
replayed delay means ending one synthesis call and starting the next, and each call
costs a few hundred milliseconds on this machine. So only gaps long enough to be
worth it are replayed, and the rest are absorbed into the words around them — which
is what keeps an ordinary sentence a single call:

| Setting | Meaning |
| --- | --- |
| `--word-delay auto\|on\|off` | **auto** (default) obeys the page's `Word delays` button; `on`/`off` override it |
| `--word-delay-min-ms 80` | shortest gap worth a synthesis call. Lower it for more literal rhythm and more lag |
| `--word-delay-max-ms 5000` | cap on a replayed gap, so a hesitant speaker does not stall the clone |

The switch lives with the page, because that is where the delays are measured, and
the voice changer reads it from the local server (stored in `out\word_delay.json`,
so it survives a restart of either side). Flipping it takes effect on the next
sentence, with nothing to restart. With no page server running — the local Whisper
path — the command line decides, and `auto` means on.

> **The 30 ms case.** A 30 ms gap is below what any of this replays, on purpose: it
> is about one frame at 60 Hz and not a pause anybody made on purpose. If you want
> those put back too, `--word-delay-min-ms 0 --word-delay-max-ms 100` replays every
> measurable gap, at the cost of one synthesis call per word.

### If you try word mode

Word-by-word works and keeps up (measured **1.9x real time**), but it is audibly
choppier — the model gets almost no context per call. Two things make it usable:

* **Silence trimming.** StyleTTS 2 pads every call, which is invisible in a
  sentence but very audible word by word: a one-word call for "the" came back as
  2.2 s of audio, almost all silence. Each chunk now has its dead air trimmed to a
  15 ms edge (capped at `--trim-ms`, default 220) before playing. The audio the
  server serves is untouched.
* **Bounded lag.** `--queue` (finished audio kept ready, default 12) and
  `--backlog` (total queued work, default 16) cap how far behind the voice can
  get. When the backlog is hit the *oldest* work is dropped so the voice stays
  current instead of drifting — measured when deliberately overloaded: 56 of 60
  chunks shed, the most recent 4 spoken. Drops are logged.

| Flag | Effect |
| --- | --- |
| `--mode word\|chunk\|phrase\|sentence` | how much text per synthesis call |
| `--gap-ms 40` | add a little air between chunks |
| `--trim-ms 0` | keep the model's own padding |
| `--backlog 24` | tolerate more lag before dropping anything |
| `--target-words 2` | custom grouping size |

## Configuration — voice changer

| Flag | What it does |
| --- | --- |
| `--voice NAME` | which voice to start with |
| `--list-voices` | show every voice and whether it is usable |
| `--transcript PATH` | watch a different file (default `out\transcript.txt`) |
| `--mode sentence` | how much text per synthesis call (default) |
| `--monitor` | also play through your speakers, so you can hear it yourself |
| `--once` | speak the file's current contents and exit (a quick test) |
| `--device "CABLE Input"` | a different playback device for the virtual mic |
| `--settings PATH` | the file the control panel writes (default `out\rixers_settings.json`) |
| `--pitch -3` | shift the finished voice by semitones (also changes its speed) |
| `--volume 1.5` | output gain, soft-limited so it cannot clip |
| `--gate-db -45` | treat anything quieter as silence (`-70` is off) |
| `--no-effects` | ignore pitch/volume/gate entirely (cleanest signal path) |

Everything the control panel marks as live overrides these defaults: the settings
file wins over the command line for the pause delay, the rhythm window and the
audio effects, because it is the window the user is actually looking at. An
explicit `--word-delay on` or `--word-delay off` still wins over the file.

To hear it yourself without another application:

```
run-voice-changer.bat --monitor
```

## Testing it

```
.venv\Scripts\python.exe speak.py --check              # routing + server + voices
.venv\Scripts\python.exe tools\vbcable_check.py --seconds 2   # tone through the cable
.venv\Scripts\python.exe tools\verify_voice_mic.py     # text -> TTS -> captured from the cable
.venv\Scripts\python.exe tools\verify_transcript_to_mic.py  # transcript -> cable (the live path)
.venv\Scripts\python.exe tools\tts_rate_probe.py       # synthesis speed per chunk size
.venv\Scripts\python.exe tests\test_rixers_gui.py      # the control panel: builds, draws, saves
.venv\Scripts\python.exe tests\test_live_settings.py   # 26 tests: the settings bridge + the effects
.venv\Scripts\python.exe tests\test_live_engine.py     # live: a change reaches a running engine
.venv\Scripts\python.exe tests\test_voice_stage.py     # 18 unit tests
`.venv\Scripts\python.exe tests\test_pipeline.py        # 31 tests: chunking, streaming, dropping, delays
.venv\Scripts\python.exe tests\test_word_timing.py     # 21 tests: measuring and carrying delays
.venv\Scripts\python.exe tests\test_timing_flow.py     # live: page -> file -> pipeline -> microphone
.venv\Scripts\python.exe tests\test_core.py            # 23 tests: resampling, VAD, commit rule, transcript
.venv\Scripts\python.exe tests\test_bridge.py          # the page's own contract, incl. the delay switch
.venv\Scripts\python.exe tests\test_emotion.py         # 14 tests: pitch, arousal/valence
.venv\Scripts\python.exe tests\test_emotion_voices.py  # 15 tests: mapping, tracking
.venv\Scripts\python.exe tests\test_emotion_routing.py # 7 tests: per-chunk voice switching
.venv\Scripts\python.exe tests\test_voice_integration.py  # live: streaming + voice swap
node tests\test_page_logic.js                          # the browser page's commit logic
```

To check the control panel's *layout* rather than just its behaviour — for example
after moving a card — photograph every page:

```
.venv\Scripts\python.exe tools\snap_gui.py out\shots
```

That builds the real window, screenshots each page with `PrintWindow` (so it works
even with a game covering the screen) and writes `out\shots\<page>.png`. The
capture is `tools\screenshot.py`, which is standard library only.

`verify_voice_mic.py` and `verify_transcript_to_mic.py` are the honest ones. The
first runs `speak.py`; the second starts the **real voice changer**, appends a line
to a transcript the way the transcriber does, and records the cable's microphone
end. That distinction matters — the bug where transcription succeeded but the cable
stayed silent lived in the transcript watcher, and the first tool cannot see it.

`tts_rate_probe.py` is the one that decided the design: it measures audio seconds
produced per second of synthesis for each chunk size, which is what says whether
streaming can keep up (it can — 2.5x to 5x real time per call).

## How the voice is made

The engine is this project's own: `live_transcribe\engine_tts_server.py`, started with
the Python 3.11 environment `Setup.bat` builds under `engine/styletts2/.venv`. It calls
the `styletts2` package from PyPI with the LibriTTS checkpoint
(`epochs_2nd_00020.pth`), and adds the three in-memory fixes described under
[Setup](#three-library-defects-the-engine-patches-at-startup), a render guard, and
a single-request-at-a-time guard around the one GPU. `live_transcribe\tts.py` is only
the HTTP client, and the API is the one it already spoke:

    GET  /health            what is loaded, on what device, with which voice
    POST /synthesize        {"text": ...} -> base64 wav + peak/rms/timing
    POST /reload-reference  {"path": ...} swaps the cloned voice

A Riko checkout is no longer needed. On a machine that has one, `Setup.bat
--reuse-legacy` will adopt its engine environment and model files instead of
downloading them again.

Text → phonemes → duration predictor → style vector sampled from your reference
wav → pitch/energy → diffusion acoustic model (3 steps) → HiFi-GAN vocoder →
24 kHz mono PCM. **Volume is generated, not applied**: the vocoder decides the
amplitude of every sample, and the only explicit gain in the path is the
quiet-rescue, which scales upward to keep a faint draw audible.

**The render guard.** StyleTTS 2 fails at random on short text, and the failure is
not always loud. Rendering the single word "you" twelve times through the live
engine produced 0.37–4.57 s of audio, and nine of those twelve were unusable —
seconds of broadband hiss for one syllable, which is what "the voice started
tweaking out" sounds like from the other side. Only some of them clipped; the rest
peaked at 0.04–0.4, so the old peak-only check could not see them at all. Every
render is now checked for three things — full-scale saturation, length against a
per-word ceiling, and the share of its energy above 4 kHz, which is what separates
a voice (measured 0.4–19%) from a hiss (measured 64–95%). A render that fails any
of them is thrown away and re-rolled at the next number of diffusion steps in
`RENDER_ATTEMPTS`; measured, that turned 9 bad renders out of 12 into 0, at the
cost of re-rolling 2 of the 12. `GET /health` reports `rerendered` and the
thresholds in use, and `/synthesize` returns `attempts`, `noise_share` and
`render_problem` for the render it just made.

Two details this stage adds on top:

* **Sentence splitting.** Text is spoken sentence by sentence — synthesis quality
  and latency both suffer on long inputs, and a half-typed sentence must not be
  spoken. The watcher holds text back until it ends with `.`, `!` or `?`.
* **Fades and padding.** Every clip is faded over 8 ms at both ends and padded
  with silence, so sentences do not click or run together in the cable stream.

## Known limitations

* **It speaks per sentence, after you pause.** Nothing is sent to the voice until
  the sentence's punctuation arrives, because that is what sounds most natural.
  `--mode word` will follow your speech closely instead, at the cost of choppier
  delivery (see above).
* **`word` mode is the engine's worst case.** Each word becomes its own synthesis
  call, and short inputs are exactly where StyleTTS 2 fails at random — that is what
  the render guard above exists for. It also costs a full inference per word
  (measured 0.3–0.8 s) to produce a third of a second of speech, so a fast speaker
  can outrun it and the pipeline drops the oldest waiting words to stay current.
  `chunk` mode is the recommendation; `word` is there for maximum responsiveness.
* **VRAM.** StyleTTS 2 uses ~1.1 GB on top of the Whisper engine's ~0.5 GB.
  With 8 GB total this is fine, but it competes with games.
* **Stage 1 needs the right microphone.** Both engines currently read the Windows
  default input, which on this machine is a Voicemod virtual device that reports
  **−92 dBFS** (silence). Set your physical mic as the default — see
  [Troubleshooting](#troubleshooting).
* **The speech server throws one bad synthesis per session, sometimes.** Roughly
  one call in a run fails with
  `IndexError: invalid index of a 0-dim tensor` from StyleTTS 2's duration
  predictor. `retries` (3 on the Speech Engine page) covers it: the retry is a
  fresh diffusion sample and normally succeeds. It is a model quirk, not a
  routing problem — a direct `client.synthesize()` of the same text can pass on
  the first try — so if `tests\test_voice_integration.py` fails one sentence while
  the other passes, raise `retries` rather than looking for a bug in the bridge.
* **Effects change the clip's length.** Pitch is a playback rate (see
  [Effects](#effects-and-what-is-deliberately-missing)), so a pitch shift makes the
  voice faster or slower as well as higher or lower. The gaps *between* chunks are
  written at the model's own rate and so are not stretched, which keeps the rhythm
  recognisable.

## One-time setup notes

These are already done on this machine; recorded so they can be repeated.

**UTF-8 is required for the TTS server.** It prints IPA phonemes, and the default
Windows code page cannot encode them, which made synthesis fail with
`'charmap' codec can't encode character '\u025b'` *and* silently pushed the model
onto the CPU. `run-tts.bat` sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`
before starting it. With that, the model loads on CUDA (~1.1 GB VRAM) and
synthesizes a 4.6 s sentence in ~1.8 s.

**VB-CABLE** is installed at `C:\Program Files\VB\CABLE`. If it is ever missing:
run `VBCABLE_Setup_x64.exe` as administrator and reboot.

---

# Stage 1 reference

## Two interchangeable speech-to-text engines

Both write the same `out\transcript.txt`:

| Engine | Command | What it uses |
| --- | --- | --- |
| **Google (browser)** | `run-google.bat` | Chrome/Edge's built-in recogniser → Google's service. **No API key, free.** |
| Local Whisper | `run.bat` | faster-whisper on your GPU. Works offline. |

Pick whichever hears you better; stage 2 reads the same file either way.

No API, no cloud account, no key. The Whisper engine is fully offline; the Google
option sends audio to Google through Chrome, which is what makes it accurate —
see [The Google (browser) engine](#the-google-browser-engine).

## Quick start — Google engine (recommended if Whisper mishears you)

```
run-google.bat
```

A browser tab opens. Press **Start listening** (or <kbd>Space</kbd>) and talk.
Keep the tab open while you dictate. Everything you say is appended to
`out\transcript.txt`; <kbd>Ctrl</kbd>+<kbd>C</kbd> in the console stops it.

Requires **Chrome or Edge** — Firefox, Brave and Safari do not implement the Web
Speech API. No Python packages beyond the existing virtual environment are needed.

## Quick start — local Whisper (works offline)

```
run.bat
```

Speak, and watch `out\transcript.txt`. Press `q` in the terminal to stop. The
first run downloads the model (~500 MB) and calibrates your microphone's noise
floor for 1.5 seconds, so **stay quiet for a moment at startup**.

To see the words appear as you talk, launch the floating viewer as well:

```
view.bat
```

## Measured on this machine

RTX 3060 Ti, `distil-small.en`, `int8_float16`. Measured with other GPU
applications running (Roblox, Discord, Chrome, Wallpaper Engine — 6–7 GB of the
8 GB VRAM in use), which is the realistic case here.

| Metric | Result | Bar |
| --- | --- | --- |
| Typical decode (p95) | **~250–600 ms** | under 2000 ms ✅ |
| Spoken → written delay | p50 **~0.6–0.8 s** | — |
| Accuracy | **0 % WER on 5 of 6 clips**, 5.6 % on the sixth | — |
| Mic overflows | 0 | — |

Every one of the six test clips passes `tools/verify_pipeline.py`, and the ASR
engine alone is very consistent: **40 consecutive decodes of 2 s audio took
239–262 ms** (`tools/probe_latency_outliers.py`).

**One caveat worth knowing.** On this machine the live loop occasionally records a
single 2–6 s decode. It is not the model and not the streaming logic: the probe
above shows the model itself never stalling, and the slow decodes land on a
healthy-sized window whose normal cost is ~400 ms, spaced roughly 25 s apart.
The likely cause is VRAM pressure — Windows paging the CUDA context out while 7
of 8 GB is in use by other programs. When it happens, audio is still retained (no
overflows) and the words appear a moment later; the adaptive governor shrinks the
window in response. Closing other GPU applications removes it. Use
`--model tiny.en` if you want maximum headroom.

### Why faster-whisper

[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) is Whisper
reimplemented on CTranslate2, which is 4–5× faster than the reference
implementation at equal accuracy, and it runs on CUDA. For live English dictation
on this hardware it is the right tool: the other local options are either heavier
to install (NeMo/Parakeet needs a multi-GB toolchain) or less accurate at this
size. `--model` switches between the Whisper variants if you want to trade speed
for accuracy.

`distil-small.en` is the default. Back-to-back over all six clips it was both
**more accurate and faster** than `small.en` (0.8 % vs 3.2 % mean WER, 871 ms vs
1294 ms per 15 words): the distilled model has half the decoder layers, and on
short streaming windows the larger model's extra capacity mostly produced
boundary noise rather than better words.

Note: the benchmark speech was synthesised by Windows SAPI, so a few residual
errors are the model mishearing synthetic speech. Real dictation transcribes more
cleanly.

### Reproducing these numbers

```
.venv\Scripts\python.exe tools\verify_pipeline.py --all --model distil-small.en
.venv\Scripts\python.exe tools\probe_latency_outliers.py --seconds 2 --n 40
.venv\Scripts\python.exe tools\compare_models.py --models distil-small.en small.en
```

`verify_pipeline.py` plays a WAV through the *real* pipeline — same ring buffer,
VAD, decode loop and transcript writer — replacing only the microphone. It reports
word error rate against known text, the worst decode time, and latency. It fails
if any decode exceeds 2 s (change that with `--max-decode-ms`).

## The files it writes

Both engines write the same files:

| File | Purpose |
| --- | --- |
| `out\transcript.txt` | **What has been said, nothing else.** This is your StyleTTS 2 input. |
| `out\live.txt` | The same text plus the in-flight guess after a `‹` marker, for the viewer. |
| `out\events.jsonl` | Append-only timing telemetry (Whisper engine). |

`transcript.txt` is **cleared every time the program starts** (pass `--append` to
keep it instead). It is rewritten atomically, and only when the text actually
changes — so a file watcher downstream sees one clean update per new phrase and
never a half-written file.

## Options — Whisper engine

```
.venv\Scripts\python.exe transcribe.py --help
```

The ones worth knowing:

| Flag | What it does |
| --- | --- |
| `--list-devices` | print every microphone with its index |
| `--input-device 3` | use a specific mic (index or name substring) |
| `--gain-db 18` | software boost for a quiet mic (soft-limited, never clips) |
| `--model small.en` | try the larger model if it suits your voice |
| `--chunk 1.0` | smaller = lower delay, more GPU work (default 1.6 s) |
| `--out D:\path\voice.txt` | write the transcript somewhere else |
| `--min-word-prob 0` | keep even low-confidence words |
| `--no-dedupe` | disable re-emission detection |
| `--calibrate` | re-measure the noise floor (default 1.5 s at startup) |
| `--no-calibrate` | skip calibration, use the built-in −42 dBFS threshold |
| `--vad-db -45` | set the speech threshold manually |
| `--no-vad-gate` | decode continuously, never gate on voice activity |
| `--doctor` | print CUDA / cuDNN / device diagnostics |
| `--append` | do **not** clear the transcript on start |
| `--duration 30` | stop automatically after 30 s (useful for testing) |

### Keys while running

| Key | Action |
| --- | --- |
| `c` | clear the transcript without restarting |
| `l` | pause / resume listening |
| `q` | quit (flushes the last words first) |

## The Google (browser) engine

This is the more accurate option, and it is what to use if the local model keeps
mishearing you — Google's recogniser is trained on vastly more speech than any
model that fits on a desktop GPU.

**How it works.** `run-google.bat` starts a tiny local server (standard library
only, nothing to install) and opens a page in Chrome. The page calls the browser's
built-in `SpeechRecognition` API, which streams your voice to Google and returns
text. The page sends each finished phrase back to the local server over a
WebSocket, and the server appends it to `out\transcript.txt`.

**It is free and needs no API key.** Chrome's speech recognition is a browser
feature, not a paid product — there is no billing, no quota to configure, and
nothing to sign up for. The tradeoff is that it only works while Chrome is open
and online, whereas the Whisper engine runs offline.

### Options

| Flag | What it does |
| --- | --- |
| `--language en-GB` | preset the recognition language |
| `--out D:\path\voice.txt` | write the transcript somewhere else |
| `--port 8777` | change the local port (it moves automatically if taken) |
| `--append` | do **not** clear the transcript on start |
| `--no-open` | start the server without launching a browser |
| `--word-delay auto\|on\|off` | whether the voice changer replays the delays measured between your words; `auto` lets the page's button decide |
| `--word-delay-min-ms 80` | shortest gap worth replaying on the voice changer's side |

### Using the page

* **Start listening** / **Stop listening** — or press <kbd>Space</kbd>. <kbd>Esc</kbd> stops.
* **Word delays** — cycles `auto → on → off`. This is the switch described under
  [Word delays](#word-delays-speaking-at-your-own-pace): the page measures how long
  you leave between words and the voice changer replays the gaps worth replaying. It
  is shared with the voice changer through the local server, so it applies to the
  next sentence without restarting anything, and the choice is remembered in this
  browser and in `out\word_delay.json`.
* **Language** — switch anytime; recognition restarts to apply it.
* **Clear** — empties the transcript file.
* **Mic level** meter — shows the *system default* input, which is what the
  recogniser uses. Watch this to see *why* words are missed. The hint beside it
  shows the shortest gap currently being replayed.
* **Activity** log — every recognition event, every phrase committed, and every
  error, timestamped. The same lines are printed in the console that launched the
  server, so you can see exactly what happened without opening dev tools.

There is no microphone dropdown, deliberately: `SpeechRecognition` cannot be
pointed at a device, so a dropdown could not affect recognition and would only
mislead. Change the device in Windows — see
[If sessions start and abort immediately](#if-sessions-start-and-abort-immediately).

#### If sessions start and abort immediately

Reported symptoms: the log fills with `session #N started → aborted` about once a
second, transcription is unreliable or absent, and the meter reads near `−90 dBFS`
even while you talk.

Two causes were found and fixed, plus one that is purely a Windows setting.

**1. The page was fighting the recogniser for the microphone (fixed).** An earlier
version opened the chosen device with `getUserMedia`, routed it through a
`MediaStreamAudioDestinationNode`, and also ran a level meter. That is *two*
captures competing with the one the recogniser opens for itself — Chrome responds
by starting a session and aborting it. The page now lets `SpeechRecognition` own
the microphone exactly as the working reference implementation does, and the level
meter is the only `getUserMedia` consumer.

**2. Restarts had no backoff (fixed).** A session that produced nothing was
retried instantly, forever. Sessions are now judged on whether they produced
*text*, and failures back off 350 ms → 8 s; after four in a row it stops and
explains the problem. A single `SpeechRecognition` object is reused for the life
of the page rather than creating a new one per session, which is what the
reference does.

**3. The default microphone is a virtual device (Windows setting).** Measured
here, while speaking:

| Device | Meter | Usable? |
| --- | --- | --- |
| `Microphone (Voicemod)` | **−92 dBFS** | no — silence |
| `Microphone (DeskIn Virtual Audio)` | **−91 dBFS** | no — silence |
| `Microphone (HyperX Cloud II Wireless)` | ~−45 dBFS | yes, but quiet |

`SpeechRecognition` has **no device option** — it always captures the *system
default* input, and a page cannot redirect it. So setting your real microphone as
the default is not optional:

1. `Settings → System → Sound → Input` → choose the **physical** mic (headset or
   built-in), not Voicemod / VB-Cable / DeskIn;
2. raise its level there — the meter should enter green (above −35 dBFS) when you
   talk;
3. keep Voicemod on the *output* side if you use it as a voice changer;
4. reload the page and press **Start listening**.

The meter deliberately follows the *system default* too, so what it shows is
exactly what the recogniser is being fed.

**Fallback:** Whisper can be pointed at a device and can amplify it, which the
browser cannot:

```
run.bat --input-device HyperX --gain-db 18
```

Because Chrome is unreliable about marking results final (it often ends a session
having only ever produced *interim* text), the page commits a phrase on whichever
of these happens first: Chrome marks it final, you pause for 0.7 s, the session
ends, or you press stop. Identical or overlapping phrases are de-duplicated, so
nothing is written twice. This is what stops words appearing on screen but never
reaching the file.

## Feeding StyleTTS 2

`out\transcript.txt` always contains the complete text of the session — with
punctuation, capitalisation, and no partial words. A minimal bridge:

```python
from pathlib import Path

transcript = Path(r"C:\Users\oxsta\Desktop\custom voicechanger\out\transcript.txt")

# Speak only the new text, so you do not re-speak every poll
last = ""
while True:
    text = transcript.read_text(encoding="utf-8").strip()
    if text and text != last:
        last = text
        styletts2_speak(text[-400:], voice="target_voice.wav")  # your call
    time.sleep(0.2)
```

Two things to keep in mind when you wire it up:

* **Debounce.** The file changes once per phrase, not once per word, but you
  still want to wait ~200–300 ms of quiet before synthesising, or you will
  re-speak the same sentence as it grows. `out\events.jsonl` records a `flush`
  event at every pause if you would rather trigger on that.
* **Speak the tail, not the whole file.** `transcript.txt` accumulates the whole
  session, so synthesise only the part you have not spoken yet (track an offset
  or use the last N characters).

## How it works, and why

**A word is committed exactly once.** Whisper's encoder always pads its input to
30 seconds, so decoding 1.5 s of audio costs nearly the same as decoding 6 s.
Each decode therefore reads from the commit cursor to the live edge, which gives
the model the best possible context. Because those windows overlap, every
returned word carries an absolute timestamp and is dropped if it ends at or
before the cursor. Two guards make that safe:

* an **edge guard** keeps the last ~0.3 s of audio un-committed while you are
  still talking, so a word sliced by the chunk boundary is re-read with more
  context instead of being guessed;
* an **overlap guard** (80 ms) rejects the word straddling the cursor, which
  Whisper re-emits with slightly shifted timestamps on the next decode.

**Flushes.** When the VAD reports ~0.35 s of silence, the guard is released and
everything left is committed at once, then the cursor jumps to the end of the
decoded audio. That is what makes the last word of a sentence appear promptly.

**Three guards keep the text clean.** Streaming Whisper fails in three specific
ways, and all are fatal when the text is spoken aloud, so each is handled
separately:

* *Duplicated words.* Re-reading the audio before the cursor makes the model
  re-emit the word straddling it. An **80 ms overlap guard** rejects that; a
  tighter **re-emission check** catches the rest by noticing that an identical
  word landed within 0.6 s of itself, which is a re-decode of the same audio
  rather than new speech. A genuine immediate repeat ("very very") is further
  apart than that and survives.
* *Invented trailing words.* Feeding near-silence to Whisper is what makes it
  hallucinate. A flush therefore measures the 50 ms peak level of the
  un-committed tail and, if it holds no speech, advances the cursor without
  decoding at all. This was worth 18 spurious words on one test clip
  (`"…afterwards. to be lightweight be"`), and it also stops the same speechless
  audio being re-decoded from a stale cursor — which previously pulled almost the
  entire clip back through the model.
* *Runaway repetition.* Given laughter, music or noise, Whisper can lock onto one
  token and never stop — a live run produced 57 consecutive `"oh,"`. A third
  identical word inside a second marks the loop and the rest are dropped. Two
  repeats are allowed through, so emphasis like `"no, no, no!"` still works.

**The decode window adapts.** Because the cost of a decode is roughly fixed, the
engine measures each decode's duration and scales the window to hold it near
`target_decode_ms` (900 ms), between 0.9 s and 2 s. When the GPU gets busy the
windows shrink, which is what keeps every decode inside the 2 s bar; when it frees
up they grow again for efficiency. Over-budget decodes shrink the window
immediately and proportionally; recovery is gradual so it does not thrash.

The tail of `transcript.txt` is never a partial word, which matters when the
text is spoken aloud.

## Verification

```
.venv\Scripts\python.exe tests\test_core.py        # 19 unit tests: VAD, resampler, merge rules
.venv\Scripts\python.exe tests\test_bridge.py      # 18 tests: Google bridge, WebSocket, file writes
.venv\Scripts\python.exe tests\test_launcher.py    # launches google_stt.py and drives it
node tests\test_page_logic.js                      # 24 tests: the page's commit + restart logic
.venv\Scripts\python.exe tools\verify_pipeline.py --all              # accuracy + decode time
.venv\Scripts\python.exe tools\verify_pipeline.py --all --real-time  # + exact latency
```

`test_page_logic.js` runs `web/app.js` in a Node VM against a stubbed DOM, and
drives the recognition callbacks directly. That covers the case that caused real
trouble — interim-only results — without needing a browser.

The bridge tests need no browser either: they speak the same WebSocket protocol
the page does, so the handshake, message handling and file writes are all covered.
The launcher test starts the real server, fetches the served page, and posts a
phrase through it.

`verify_pipeline.py` plays a WAV file through the *real* Whisper pipeline — the
same ring buffer, VAD, decode loop and transcript writer — replacing only the
microphone. It reports word error rate against known text, the worst single decode
time and the spoken-to-written latency, so the audio path can be checked without
talking into a mic. It fails if any decode exceeds 2 s (raise that with
`--max-decode-ms`).

## Troubleshooting

**`Setup.bat` says Python 3.11 is missing**
StyleTTS 2 cannot be installed on 3.12+ (its `tokenizers` pin has no wheel, so pip
tries to build it from source and fails). Install 3.11 from
<https://www.python.org/downloads/release/python-3119/> and make sure
`py -3.11 --version` works, then run `Setup.bat` again. It stops *before*
downloading anything, so nothing is wasted.

**The engine loads and then never speaks**
Run `Setup.bat check` first. If the files are all there, start the engine on its own
with `run-tts.bat` and read its output: the three library defects listed under
[Setup](#three-library-defects-the-engine-patches-at-startup) all look like this, and
the engine already patches them. If you are running a *different* server (Riko's, for
example), that one does not have those patches.

**The engine takes minutes to start, or hangs on "loading the model"**
Two known causes, both with the same symptom:

* **A stale download lock.** The `styletts2` package resolves its auxiliary models
  through `cached_path`, which takes an exclusive file lock with no timeout. A
  download interrupted by Ctrl+C leaves that lock behind and the next start waits on
  it *forever*. This project avoids `cached_path` for its own models, but a server
  that uses it (Riko's) can still be caught. The cache is under
  `%USERPROFILE%\.cache\cached_path`; deleting the `.lock` files there clears it.
* **Importing `librosa.core.audio` hangs.** Measured on this machine: 10+ minutes,
  600 s of CPU, no output — because of numba's JIT machinery. `engine_tts_server.py`
  therefore reads reference clips with `soundfile` and never imports librosa's core
  modules. A server that calls `librosa.load` (Riko's does) will hang here regardless
  of this project.

**Nothing at all is transcribed / the transcript stays empty**
Check the microphone itself first — this is by far the most common cause, and it
affects **both** engines:

```
.venv\Scripts\python.exe tools\scan_inputs.py
.venv\Scripts\python.exe tools\mic_check.py --input-device <best index>
```

With the Google engine, also watch the **Mic level** meter on the page — it shows
exactly what the recogniser is being fed.

A working mic shows speech peaks around −30 to −12 dBFS and a floor well below it.
If a device reports **flat** levels or a floor near −80 dBFS or lower, the
microphone is muted or at zero gain — that is a Windows/headset problem, not a
transcription problem. On this machine both the HyperX headset and the built-in
Realtek mic measured near-silence (peaks of −45 and −82 dBFS), which is exactly
the symptom that makes every engine mishear words. Check, in order:

1. the physical mute switch on the headset (USB headsets often mute in hardware,
   which software cannot override);
2. **Settings → Privacy & security → Microphone** → *Let desktop apps access your
   microphone*, and *Microphone access* itself;
3. **Settings → System → Sound → Input →** your device → raise the volume
   (and untick mute). `mmsys.cpl` → *Recording* → device → *Properties* → *Levels*
   gives the same control;
4. pick a different device with `--input-device` (Whisper engine).

If a mic works but is simply quiet, the Whisper engine can boost it in software:
`--gain-db 18`. The boost is soft-limited with `tanh`, so it compresses rather than
clipping — a clipped waveform transcribes far worse than a compressed one. The
Google engine cannot apply a software boost (the browser hands the recogniser the
raw device), so for that one the Windows level is the only fix.

**Words appear in grey on the page but never reach the file**
Fixed — that was a real bug. Chrome's continuous recognition frequently ends a
session with only *interim* results, never marking them final, so the page was
discarding perfectly good phrases. A phrase is now committed on pause, on session
end, on stop, or on a final result, whichever comes first. Watch the **Activity**
log to confirm: each commit prints `heard (reason): …`.

**The Google page keeps restarting sessions and never transcribes anything**
This is a microphone problem, not a Google problem — see
[If sessions start and abort immediately](#if-sessions-start-and-abort-immediately-and-nothing-is-transcribed).
Pick a real microphone in the **Microphone** dropdown; the Windows default is
often a virtual device that passes near-silence.

**A virtual microphone (Voicemod, VB-Cable, DeskIn, etc.) is selected**
These pass audio through and often change its level, and they are the usual reason
a recogniser hears nothing. Chrome's meter reports what the recogniser actually
receives, so if it reads very low, switch to the physical microphone in the
dropdown — or raise the level on the virtual device *and* its source. Note that a
virtual mic used for voice changing also colours what the recogniser hears.

**The browser page is blank / "Speech recognition is not available"**
You are in Firefox, Brave or Safari, which do not implement the Web Speech API.
Open the printed URL in **Chrome or Edge**.

**The Google page opens but never hears anything**
Check the Mic level meter: if it stays flat, the browser has no usable audio —
grant microphone permission via the icon in Chrome's address bar, and check the
device shown in `chrome://settings/content/microphone`.

**`Library cublas64_12.dll is not found`** (Whisper engine only)
The CUDA runtime DLLs are missing:
```
.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
`live_transcribe/cuda_env.py` registers them automatically before CTranslate2
loads. Run `--doctor` to confirm. As a fallback the model silently retries on CPU.

**It transcribes silence, or ignores my voice**
Run `--calibrate 2` and stay quiet during it, or set the threshold by hand with
`--vad-db -45`. If your room is very noisy, try `--vad-db -35`.

**It transcribes other people / music in the room**
That is expected: the VAD gates on loudness only, not on *whose* voice it is. Use
a close-talking headset mic (`--input-device HyperX`) rather than a desk mic.

**The wrong microphone is being used**
`--list-devices`, then `--input-device <index>`. The Windows default input may be
a virtual device (Voicemod, VB-Cable, DeskIn all appear here).

**Delay is higher than I want**
Lower `--chunk` (e.g. `--chunk 1.0`) for lower delay at the cost of more GPU work.
If you are on CPU, add `--model tiny.en --device cpu --compute-type int8`.

**Decoding never stops / the GPU stays busy while I am silent**
It should: the loop is gated on voice activity, and the cursor is moved to the
moment speech starts so idle room noise is never handed to Whisper. If you see
constant decoding anyway, your noise floor is probably above the threshold —
re-run with `--calibrate 3`, or set `--vad-db` manually.

## Layout

```
Setup.bat                   install everything (this project + the speech engine + models)
RixersVoiceChanger.bat      the control panel (native Windows window, no browser)
setup.py                    what Setup.bat runs: environments, downloads, verification
start-all.bat               start everything, then close its own window
stop-all.bat                stop the speech engine and the voice changer
pick-voice.bat              voice picker window (no typing needed)
run-tts.bat                 start the speech engine (waits and confirms readiness)
run-voice-changer.bat       start the voice changer (confirms it connected)
launch.py                   preflight check, readiness waits, stop (used by the .bat files)
speak.py                    text -> StyleTTS 2 -> virtual microphone
rixers_gui.py               the control panel: sidebar, pages, Start/Stop, live console
voice_changer.py            watches the transcript and speaks it (stage 2 bridge)
google_stt.py               Google engine: local server + opens the browser page
run-google.bat              launch the Google engine
transcribe.py               Whisper engine entry point
run.bat                     launch the Whisper engine
viewer.py / view.bat        floating live-text window (works with either engine)
voices/                     one wav per voice, plus voice.txt for live switching
engine/                     downloaded by Setup.bat; never committed (see .gitignore)
  styletts2/
    .venv/                  Python 3.11 with torch + styletts2
    epochs_2nd_00020.pth    the LibriTTS checkpoint
    config.yml              its config, with paths to the models below
    installed.json          what setup.py verified, and when
  models/
    epochs_00080.pth        text aligner
    bst.t7                  pitch extractor
    PLBERT/                 style encoder: step_1000000.t7 + config.yml
out/
  rixers_settings.json      what the control panel writes, what the engine obeys
  transcript.txt            stage 1 writes here, stage 2 reads here
web/
  index.html                the page: markup, styles, activity log
  app.js                    the page's logic (recognition, commits, level meter)
  build_page.py             splices app.js into index.html
  server.py                 HTTP server + WebSocket receiver (stdlib only)
live_transcribe/
  cli.py                    argument parsing, keys, lifecycle
  engine.py                 the streaming loop (decode scheduling, flushing, guards)
  engine_tts_server.py      the speech engine this project installs and runs
  engine_paths.py           where the engine's files live, and whether they are ready
  merge.py                  word selection: no duplicates, no lost words
  asr.py                    faster-whisper wrapper, CUDA fallback, warm-up
  audio.py                  mic capture, gain, streaming resampler, ring buffer, VAD
  transcript.py             atomic transcript writes
  cuda_env.py               finds the pip-installed cuBLAS/cuDNN DLLs
  tts.py                    HTTP client for the speech engine
  pipeline.py               streaming speech: chunking, producer/consumer, drops
  live_settings.py          the settings schema shared by the panel and the engine
  audio_effects.py          pitch, volume and the noise gate on the way into the cable
  emotion.py               voice emotion: features, arousal/valence, utterance tracker
  emotion_voices.py        emotion -> reference voice mapping
  virtual_mic.py            plays speech into VB-CABLE (the "microphone")
  voices.py                 the voices folder: discovery, validation, selection
tests/
  test_core.py              19 unit tests (VAD, resampler, ring buffer, merge rules)
  test_bridge.py            18 tests for the Google bridge (WebSocket, file writes)
  test_launcher.py          launches google_stt.py and drives it over HTTP
  test_page_logic.js        29 tests for the page's commit + restart logic (Node)
  test_rixers_gui.py        builds the whole control panel and walks every page
  test_live_settings.py     the settings bridge, the bounds, and the effects
  test_live_engine.py       live: a settings change reaches a running engine
  test_voice_stage.py       18 unit tests (splitting, following, voices, fades)
  test_pipeline.py          25 tests (chunking, streaming, backpressure, trimming)
  test_emotion.py           14 tests (pitch recovery, arousal/valence separation)
  test_emotion_voices.py    15 tests (mapping, fallbacks, utterance tracking)
  test_emotion_routing.py   7 tests (per-chunk voice switching, failure handling)
  test_voice_integration.py live run: streaming + live voice switching
tools/
  engine_check.py           talk to a running engine the way the app does
  screenshot.py             capture a window (or the screen) to a PNG, stdlib only
  snap_gui.py               photograph every control-panel page, for layout checks
  vbcable_check.py          proves the virtual cable routes audio
  verify_voice_mic.py       text -> TTS -> captured from the microphone end
  verify_transcript_to_mic.py  transcript -> cable, through the real voice changer
  style_probe.py            measures whether StyleTTS2's style knobs do anything
  tts_probe.py              query the speech server directly
  tts_rate_probe.py         synthesis speed vs chunk size (the streaming budget)
  compare_models.py         back-to-back model accuracy/speed comparison
  sweep_config.py           model x cadence matrix
  sweep_window.py           decode-window size vs budget
  verify_pipeline.py        offline accuracy + latency verification
  trace_one.py              decode-by-decode trace of one WAV (debugging)
  probe_boundary.py         word confidence at decode boundaries
  probe_latency_outliers.py checks for periodic slow decodes
  scan_inputs.py            which microphone actually produces signal
  mic_check.py              live level meter, suggests a VAD threshold
  list_devices.py           every input device with its index
  bench_models.py           model speed/accuracy comparison
  bench_batch.py            batched-inference experiments
  bench_word_ts.py          cost of word-level timestamps
  make_bench_audio*.ps1     generates test speech with Windows SAPI
```
