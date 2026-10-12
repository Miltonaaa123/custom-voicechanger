VOICES
======
Every .wav file in this folder is a voice. Drop one in, it appears in the list.
No registration step, no config editing required.


HOW TO ADD A VOICE
------------------
1. Record or download 3-16 seconds of the voice you want.
   - clean speech, one speaker, no music and no background noise
   - ONE or two sentences is ideal; 5-10 seconds works best
   - must be .wav (convert mp3/flac with any converter first)

2. Save it in this folder, for example:
       voices\my_voice.wav

3. Use it:
       run-voice-changer.bat --list-voices
       run-voice-changer.bat --voice my_voice

The clip is the ONLY thing that defines the voice: StyleTTS 2 clones the timbre
from it. Nothing is trained, so a new voice takes effect immediately.


MAKING ONE THE DEFAULT
----------------------
Rename or copy it to:
       voices\default.wav
The speech server loads that at start-up, and it is used when --voice is omitted.


CHANGING VOICE WHILE RUNNING
----------------------------
Write the name of a voice (without .wav) into:
       voices\voice.txt
The voice changer re-reads this file before every sentence, so it switches
mid-sentence-stream with no restart. Delete the file to go back to --voice.


OPTIONAL: DESCRIBE A VOICE
--------------------------
voices\voices.json holds a friendly name, the clip's spoken transcript, and tags.
None of it is required - a missing or broken file is ignored.

    {
      "my_voice.wav": {
        "name": "Narrator",
        "transcript": "the exact words spoken in the clip",
        "tags": ["deep", "calm"]
      }
    }

The transcript is informational for this checkpoint; quality does not depend on it.


WHAT MAKES A GOOD REFERENCE CLIP
--------------------------------
DO    speak steadily, in the tone you want, with a little energy
DO    keep it dry - no reverb, no compression, no music
DON'T use a clip shorter than 3 s or longer than 16 s (both are refused)
DON'T include two speakers, laughter, or silence padding at the ends


TROUBLESHOOTING
---------------
"UNUSABLE - only 1.2s"        too short; the model needs at least 3 s
"UNUSABLE - 22.0s is over"    too long; trim it to one or two sentences
"UNUSABLE - not a readable wav"   actually an mp3/flac, or corrupted
Squealing or screeching output   the reference is unusual (music, clipping,
                                 heavy reverb). The server detects this and
                                 refuses rather than returning noise - try
                                 another clip.
