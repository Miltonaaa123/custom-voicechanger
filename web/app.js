"use strict";

// ---------------------------------------------------------------------------
// Live transcript page
//
// Chrome's Web Speech API in `continuous` mode is unreliable about marking
// results final: it frequently ends the session after a pause having only ever
// produced *interim* text. Committing only `isFinal` results therefore loses
// entire phrases -- you speak, the page shows your words in grey, and the file
// stays empty.
//
// So there are three commit paths, all funnelling into commitPhrase(), which
// de-duplicates against what was last sent:
//   1. a result Chrome marks final;
//   2. a pause: interim text unchanged for COMMIT_AFTER_MS;
//   3. the session ending (onend) or an error.
// A phrase is thus committed whether or not Chrome ever calls it final.
// ---------------------------------------------------------------------------

const COMMIT_AFTER_MS = 700;
const HEARD_HINT_MS = 800;

// ---------------------------------------------------------------------------
// Word delays -- how long you leave between words.
//
// Chrome's speech service reports words, not their timing: there is no word clock
// to read (SpeechRecognitionResult.timestamp is not implemented in Chrome). What
// *is* observable is when each word arrives, and the intervals between those
// arrivals are dominated by how long you were silent between them. So that is
// what is measured, with the recognition lag subtracted.
//
// The delays are carried to the server as marks in the text ("hello [[70]] what"),
// and the voice changer decides which of them are worth replaying -- a delay has
// to be long enough to be worth a separate synthesis call, or the voice falls
// behind the speaker.
// ---------------------------------------------------------------------------
const MARK_MIN_MS = 25;         // below this a gap is rounding noise, not a pause
const MARK_SHORT_MS = 90;       // a gap across a commit is real, but this short is not a pause
// The service never reports two words in the same instant, so a small floor comes
// off an interval measured while someone is talking. It is only a fallback: the
// shortest interval in the phrase is a better estimate and is normally available.
const LAG_FLOOR_MS = 40;
const MARK_UTTERANCE_MS = 2000; // a longer break ends the thought; do not mark it
const TIMING_STATES = ["auto", "on", "off"];
const TIMING_LABELS = {
  auto: "Word delays: auto",
  on: "Word delays: on",
  off: "Word delays: off",
};

let timingMode = "auto";
let timingUserSet = false;
let timingMinMs = 80;
let timingMaxMs = 1500;
let speechLagFloorMs = LAG_FLOOR_MS;
let speechStopAt = 0;      // when the recogniser last reported speech ending
let lastOnsetAt = 0;       // when the last reported word appeared (survives a commit)
let wordMarks = [];        // { index, word, at } for the phrase being built


// Chrome ends a recognition session on its own after a short silence, and if
// sessions start failing (busy service, no audio, rate limiting) restarting
// instantly turns that into an endless abort loop. Restarts therefore back off,
// and a session that produced no audio at all counts as a failure.
const RESTART_BASE_MS = 350;
const RESTART_MAX_MS = 8000;
const UNHEALTHY_AFTER = 4;
const STUCK_WHILE_TYPING_MS = 12000;

let sessionSeq = 0;
let sessionRunning = false;
let sessionSawAudio = false;
let restartDelayMs = RESTART_BASE_MS;
let healthySessions = 0;
let unhealthySessions = 0;
let lastResultAt = 0;
let lastCommitAt = 0;

const el = {
  dot: document.getElementById("dot"),
  status: document.getElementById("status"),
  toggle: document.getElementById("toggle"),
  clear: document.getElementById("clear"),
  lang: document.getElementById("lang"),
  transcript: document.getElementById("transcript"),
  levelbar: document.getElementById("levelbar"),
  leveltext: document.getElementById("leveltext"),
  peakhint: document.getElementById("peakhint"),
  warning: document.getElementById("warning"),
  logpanel: document.getElementById("logpanel"),
  logbody: document.getElementById("logbody"),
  logtoggle: document.getElementById("logtoggle"),
  outpath: document.getElementById("outpath"),
  timing: document.getElementById("timing"),
  timinghint: document.getElementById("timinghint"),
};

let ws = null;
let wsReady = false;
let pending = [];
let recognition = null;
let listening = false;
let restartTimer = null;
let commitTimer = null;
let heardTimer = null;
let lastSent = "";
let finalText = "";
let markedText = "";       // the same words with the measured delays in them
let interimText = "";
let peakDb = -120;
let quietFrames = 0;
let meterStream = null;
let meterGeneration = 0;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (el.lang.dataset.default && el.lang.dataset.default !== "__LANGUAGE__") {
  el.lang.value = el.lang.dataset.default;
}

// ---------------------------------------------------------------------------
// Diagnostics. Without these a silent recognition failure looks identical to a
// working page that simply has not heard anything yet.
// ---------------------------------------------------------------------------
function log(message, cls) {
  send({ kind: "event", detail: message });
  const entry = document.createElement("div");
  entry.className = "logline" + (cls ? " " + cls : "");
  entry.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  el.logbody.appendChild(entry);
  while (el.logbody.childElementCount > 150) el.logbody.removeChild(el.logbody.firstChild);
  el.logbody.scrollTop = el.logbody.scrollHeight;
}

function el_(id) {
  return document.getElementById(id);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function render() {
  el.transcript.textContent = finalText;
  if (interimText) {
    const span = document.createElement("span");
    span.id = "interim";
    span.textContent = (finalText ? " " : "") + interimText;
    el.transcript.appendChild(span);
  }
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function setStatus(text, cls) {
  el.status.textContent = text;
  el.status.className = cls || "";
}

function showWarning(html) {
  el.warning.innerHTML = html;
  el.warning.classList.remove("hidden");
}

function clearWarning() {
  el.warning.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Server socket
// ---------------------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    wsReady = true;
    ws.send(JSON.stringify({ kind: "hello", language: el.lang.value }));
    while (pending.length) ws.send(pending.shift());
    log("connected to the local server");
  };
  ws.onmessage = (event) => {
    let msg = null;
    try { msg = JSON.parse(event.data); } catch (err) { return; }
    if (msg && msg.kind === "timing") applyTimingState(msg);
  };
  ws.onclose = () => {
    wsReady = false;
    log("server connection lost, retrying", "warn");
    setTimeout(connect, 1000);
  };
  ws.onerror = () => { /* onclose handles the retry */ };
}

function send(obj) {
  const data = JSON.stringify(obj);
  if (wsReady && ws && ws.readyState === WebSocket.OPEN) ws.send(data);
  else if (pending.length < 200) pending.push(data);
}

// ---------------------------------------------------------------------------
// Measuring how long you left between words
// ---------------------------------------------------------------------------
function resetWordMarks() {
  wordMarks = [];
}

/** Note where each of a result's words lands on the page's clock. */
function noteWordOnsets(text, startIndex) {
  const words = (text || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const now = Date.now();

  // Measure the recogniser's own delay while nothing is being said: speech ended,
  // then the words finally turned up. That is lag, not silence from the speaker.
  if (speechStopAt && now - speechStopAt < 2500) {
    const lag = now - speechStopAt;
    if (lag > 0 && lag < 1500) speechLagFloorMs = speechLagFloorMs * 0.7 + lag * 0.3;
  }

  const onsets = [];
  words.forEach((word, i) => {
    // Results arrive together when the service catches up, so several words can
    // share one instant. A zero gap is honest: nothing was observed between them.
    const onset = i === 0 ? now : now + i * 10;
    onsets.push({ index: startIndex + i, word, at: onset });
  });
  // The clock is moved on *after* the commit that follows has read it: a commit
  // measures the delay in front of its first word against where the previous word
  // ended, so advancing it here would make every phrase look as if it began after
  // a silence nobody made.
  return onsets;
}

/** Move the onset clock past the words just handled. */
function advanceOnsetClock(onsets) {
  if (onsets.length) lastOnsetAt = onsets[onsets.length - 1].at;
}

/** The delay in front of every recorded word, with the service's own lag removed.
 *
 * There are two kinds of interval here, and they need different treatment.
 *
 * Inside a phrase the floor comes off: an interval between two words of the same
 * result is mostly how long the service took to report the second one, so the
 * smallest interval seen in the phrase is the best estimate of that delay and any
 * pause you really made has to be larger than it.
 *
 * Across a commit -- the first word of a phrase Chrome had already closed off --
 * the interval is a plain gap between two words and is left alone. Subtracting a
 * floor there is what made the pause in front of "what" vanish.
 *
 * The last onset is deliberately *not* cleared by a commit, so that second kind of
 * pause remains measurable: a phrase detected as "what" on its own would otherwise
 * look as if nothing preceded it, and that pause is the one that matters most.
 */
function computeWordDelays() {
  const crossing = lastOnsetAt;   // the word before this commit, if there was one
  const intervals = [];
  let previousAt = lastOnsetAt;
  for (const entry of wordMarks) {
    // Several words can share one instant when the service catches up: that is a
    // batch, not silence, so it cannot inform the floor.
    intervals.push(previousAt > 0 ? entry.at - previousAt : null);
    previousAt = entry.at;
  }
  // Only intervals *inside* this commit teach the floor. The one that spans a
  // commit is a real gap between two words, and letting it set the floor is what
  // made the pause in front of a separately-detected word cancel itself out.
  const inside = intervals.filter((ms, i) => i > 0 && ms !== null && ms > 0);
  const floor = inside.length ? Math.min(...inside) : 0;

  let previousWasEndOfSentence = false;
  return wordMarks.map((entry, i) => {
    const interval = intervals[i];
    let ms = 0;
    if (interval !== null && interval > 0 && interval <= MARK_UTTERANCE_MS) {
      if (i === 0 && crossing > 0) {
        // A gap between two separately reported words: it is the real pause, so
        // nothing comes off it -- but a very short one is still just the service
        // reporting two words separately, not a pause worth replaying.
        if (interval >= MARK_SHORT_MS) ms = interval;
      } else if (floor > 0) {
        ms = interval - floor;
      } else {
        ms = interval;
      }
    }
    // A full stop is not rhythm inside a phrase.
    if (previousWasEndOfSentence) ms = 0;
    previousWasEndOfSentence = /[.!?]$/.test(entry.word);
    return { word: entry.word, ms: ms >= MARK_MIN_MS ? Math.round(ms) : 0 };
  });
}

/** Forget the rhythm: a long silence, a clear, or a new session. */
function resetWordTiming() {
  wordMarks = [];
  lastOnsetAt = 0;
  speechStopAt = 0;
}

/** Remember what the smallest gaps look like, for the next phrase's fallback. */
function calibrateLag(marks) {
  const gaps = marks.map((m) => m.ms).filter((ms) => ms > 0 && ms < 700);
  if (!gaps.length) return;
  speechLagFloorMs = Math.max(MARK_MIN_MS, speechLagFloorMs * 0.7 + Math.min(...gaps) * 0.3);
}

/**
 * Put the measured delays into the text, in front of the words they belong to.
 *
 * They travel with the text because they have to: the transcriber and the voice
 * changer are separate processes that meet at a file, and a delay kept anywhere
 * else would be lost the moment the words were written down.
 */
function withDelays(text, delays) {
  const words = (text || "").trim().split(/\s+/).filter(Boolean);
  const out = [];
  words.forEach((word, i) => {
    const delay = delays[i];
    if (delay && delay.ms > 0) out.push(`[[${delay.ms}]]`);
    out.push(word);
  });
  return out.join(" ");
}

function describeDelays(marks) {
  const gaps = marks.map((m) => m.ms).filter((ms) => ms > 0);
  if (!gaps.length) return "";
  const median = gaps.slice().sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
  return `, delays ${gaps.length} (median ${median} ms)` +
    (timingMode === "off" ? " [not replayed]" : "");
}

// ---------------------------------------------------------------------------
// The shared word-delay switch
//
// The delays are measured here but replayed in the voice changer, which is a
// different process. The server in between is what they share, so the choice
// survives a reload and reaches a voice changer that is already running.
// ---------------------------------------------------------------------------
function setTimingMode(mode, announce) {
  if (!TIMING_STATES.includes(mode)) return;
  timingMode = mode;
  try { localStorage.setItem("wordDelayMode", mode); } catch (err) { /* private mode */ }
  if (el.timing) el.timing.textContent = TIMING_LABELS[mode];
  if (el.timinghint) {
    el.timinghint.textContent = mode === "off"
      ? "measured, not replayed"
      : `replay gaps ${timingMinMs}+ ms`;
  }
  send({ kind: "timing", mode });
  fetch("/timing", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  }).catch(() => {});
  if (announce) log(`word delays -> ${mode}`, "good");
}

function applyTimingState(state) {
  if (!state || !TIMING_STATES.includes(state.mode)) return;
  // A choice the user made in this browser wins over whatever the server was told
  // to start with on its command line; otherwise the server is the authority.
  if (!timingUserSet) setTimingMode(state.mode, false);
  if (typeof state.min_gap_ms === "number") timingMinMs = state.min_gap_ms;
  if (typeof state.max_gap_ms === "number") timingMaxMs = state.max_gap_ms;
  if (el.timinghint && timingMode !== "off") {
    el.timinghint.textContent = `replay gaps ${timingMinMs}+ ms`;
  }
}

// ---------------------------------------------------------------------------
// Committing text
// ---------------------------------------------------------------------------
function normalise(text) {
  return text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
}

/**
 * Send one phrase to the server, skipping anything already sent.
 *
 * Chrome re-reports overlapping text when a session restarts, and the pause
 * commit can race the final result for the same words, so the guard matters.
 */
function commitPhrase(raw, why) {
  const text = (raw || "").replace(/\s+/g, " ").trim();
  if (!text) return false;

  const norm = normalise(text);
  const already = normalise(lastSent);
  if (norm && already) {
    if (norm === already) return false;
    // Wholly contained in what we already sent.
    if (already.endsWith(norm)) return false;
  }

  // If the new phrase extends the previous one (Chrome re-sends a growing
  // phrase), send only the new tail.
  let toSend = text;
  let tailFrom = 0;
  if (already && norm.startsWith(already) && norm.length > already.length) {
    const words = text.split(" ");
    const sentWords = lastSent.split(" ").length;
    toSend = words.slice(sentWords).join(" ").trim();
    tailFrom = sentWords;
    if (!toSend) return false;
  }

  // The delays that belong to the words actually being sent. A phrase whose words
  // straddle the commit boundary only sends one side of the gap, so no delay is
  // invented between words that were not spoken together.
  const delays = computeWordDelays().slice(tailFrom);
  calibrateLag(delays);

  const phrase = withDelays(toSend, delays);
  // The delays go to the server separately from the words: transcript.txt is what
  // a person reads and must stay plain, while the voice changer needs the timing.
  markedText = markedText ? `${markedText} ${phrase}` : phrase;
  finalText = finalText ? `${finalText} ${toSend}` : toSend;
  lastSent = text;
  lastCommitAt = Date.now();
  interimText = "";
  resetWordMarks();
  render();
  send({ kind: "final", text: phrase + " " });
  log(`heard (${why}): ${toSend}${describeDelays(delays)}`, "good");  return true;
}

function scheduleCommit() {
  clearTimeout(commitTimer);
  commitTimer = setTimeout(() => {
    if (interimText) commitPhrase(interimText, "pause");
  }, COMMIT_AFTER_MS);
}

// ---------------------------------------------------------------------------
// Microphone routing -- deliberately absent.
//
// An earlier version opened the chosen device with getUserMedia, ran it through
// a MediaStreamAudioDestinationNode and hoped the recogniser would prefer that
// track. That is the wrong shape for this problem: SpeechRecognition opens the
// microphone itself, so the page ended up holding *two* captures (the routing
// stream and the level meter) against the one device the recogniser was also
// trying to open. Chrome responds to that by starting a session and immediately
// aborting it -- exactly the `session #N started / aborted` loop.
//
// The reference implementation this project is modelled on (riko-project) simply
// lets the recogniser own the microphone and never wraps it, which is why it does
// not hit this. That is what happens here too: no routing stream, and the level
// meter below is the only getUserMedia consumer.
// ---------------------------------------------------------------------------

function stopRecognitionStream() {
  // Kept so callers do not need to know the routing is gone.
}

// ---------------------------------------------------------------------------
// Speech recognition
// One recogniser object for the life of the page, reused for every session --
// the same shape the working reference implementation uses. Chrome ends a session
// on its own (after silence, or roughly every 60 s); when that happens the object
// is simply started again, which is what `onend` below does.
//
// The dangerous case is a session that dies immediately: retried instantly it
// becomes an endless abort loop that hammers the service and transcribes nothing.
// So a session is judged on whether it produced text, and failures back off.
// ---------------------------------------------------------------------------
function ensureRecognition() {
  if (recognition) return recognition;

  const r = new SpeechRecognition();
  r.continuous = true;
  r.interimResults = true;
  r.lang = el.lang.value;
  r.maxAlternatives = 1;

  // Per-session bookkeeping, reset on every start.
  let sawText = false;

  r.onstart = () => {
    sessionRunning = true;
    sessionSawAudio = false;
    sawText = false;
    el.dot.classList.add("live");
    el.toggle.textContent = "Stop listening";
    setStatus("listening", "good");
    clearWarning();
    log(`session #${++sessionSeq} started (lang=${r.lang})`);
  };

  r.onaudiostart = () => {
    sessionSawAudio = true;
    setStatus("listening", "good");
  };

  r.onspeechstart = () => {
    clearTimeout(heardTimer);
    heardTimer = setTimeout(() => log("speech detected", "good"), HEARD_HINT_MS);
  };
  r.onspeechend = () => {
    // Also the moment that makes the next word's arrival measurable: how long
    // after speech stops the recogniser finally reports it is its own latency.
    speechStopAt = Date.now();
    log("speech ended");
  };

  r.onresult = (event) => {
    sawText = true;
    lastResultAt = Date.now();
    let interim = "";
    let interimStart = -1;
    const finals = [];
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const text = result[0].transcript;
      if (result.isFinal) {
        finals.push({ text: text, start: wordMarks.length });
      } else {
        // Chrome updates one result in place as it grows, so its words are
        // re-timed from scratch rather than appended a second time.
        if (interimStart < 0) {
          interimStart = wordMarks.length;
          resetWordMarks();
          interimStart = 0;
        }
        interim += text;
        const onsets = noteWordOnsets(text, interimStart);
        wordMarks.push(...onsets);
        advanceOnsetClock(onsets);
      }
    }
    if (finals.length) {
      clearTimeout(commitTimer);
      for (const final of finals) {
        const onsets = noteWordOnsets(final.text, final.start);
        wordMarks.push(...onsets);
        commitPhrase(final.text, "final");
        advanceOnsetClock(onsets);
      }
    }
    if (interim) {
      interimText = interim.replace(/\s+/g, " ");
      render();
      // `marked` is the same words plus the measured delays, and it is the version
      // the voice changer follows -- live.txt, never transcript.txt.
      send({ kind: "interim", text: interimText, marked: markedText });
      scheduleCommit();
    }
  };

  r.onerror = (event) => {
    const code = event.error || "unknown";

    // "no-speech" and "aborted" are normal for a continuous listener; onend puts
    // the session back, so they are informational only.
    if (code === "no-speech") {
      commitPhrase(interimText, "no-speech");
      log("no speech in that session", "warn");
      return;
    }
    if (code === "aborted") {
      log("session aborted");
      return;
    }

    if (code === "not-allowed" || code === "service-not-allowed") {
      setStatus("microphone blocked", "bad");
      showWarning(
        "<b>Microphone access is blocked.</b> Click the mic icon in Chrome's " +
        "address bar and allow access, then press Start again."
      );
      log(`error: ${code}`, "bad");
      listening = false;
      stopListening();
      return;
    }

    if (code === "audio-capture") {
      setStatus("no microphone", "bad");
      showWarning("<b>No microphone was found.</b> Check that one is connected and enabled.");
      log(`error: ${code}`, "bad");
      listening = false;
      stopListening();
      return;
    }

    // Everything else (network, service failures) is worth retrying with backoff.
    setStatus(`error: ${code}`, "warn");
    log(`error: ${code}`, "bad");
    commitPhrase(interimText, "before-retry");
  };

  r.onend = () => {
    sessionRunning = false;
    clearTimeout(commitTimer);
    // Speech stopping is what makes the next arrival measurable, so treat the end
    // of a session as the end of speech too.
    if (!speechStopAt) speechStopAt = Date.now();
    // Whatever is showing as a guess is real speech Chrome never marked final.
    commitPhrase(interimText, "session-end");
    // A session that ended after a real silence starts the next one fresh: the
    // gap across it was a pause, not the rhythm of a sentence.
    if (Date.now() - speechStopAt > MARK_UTTERANCE_MS) resetWordTiming();
    el.dot.classList.remove("live");

    if (!listening) {
      setStatus("stopped");
      log("stopped");
      return;
    }

    if (sawText) {
      // Produced something we could commit: healthy session.
      healthySessions++;
      unhealthySessions = 0;
      restartDelayMs = RESTART_BASE_MS;
    } else {
      // No text, whether or not audio was captured. Counting a session with audio
      // but no results as healthy is what caused an endless restart loop: the
      // audio event fires, the session is aborted, and it is judged fine forever.
      unhealthySessions++;
      restartDelayMs = Math.min(RESTART_MAX_MS, Math.round(restartDelayMs * 1.8));
      if (unhealthySessions === UNHEALTHY_AFTER) reportStalledSessions();
    }

    scheduleRestart(restartDelayMs);
  };

  recognition = r;
  return r;
}

function scheduleRestart(delayMs) {
  clearTimeout(restartTimer);
  restartTimer = setTimeout(() => {
    if (!listening) return;
    startSession();
  }, delayMs);
}

/** Called when several sessions in a row ended without producing any text. */
function reportStalledSessions() {
  stopListening();
  setStatus("recogniser stalling", "bad");
  showWarning(
    "<b>The recogniser keeps starting and ending without producing any text.</b><br><br>" +
    "That is Chrome's speech service refusing the session, and in practice it is " +
    "almost always the <i>system default</i> microphone rather than the one chosen " +
    "here. Make sure the device you want is the default:<br>" +
    "<code>Settings → System → Sound → Input</code> → choose your real microphone, " +
    "not Voicemod / VB-Cable / DeskIn.<br><br>" +
    "Also worth trying: reload this page after the change, and if a virtual " +
    "microphone (Voicemod) is running, turn its routing off while dictating — " +
    "the Web Speech API does not honour the Microphone dropdown above, it uses " +
    "whatever Windows has as default.<br><br>" +
    "The local Whisper engine is the fallback: <code>run.bat --input-device HyperX</code>."
  );
  log("stopped: repeated sessions produced no text", "bad");
}

// Watches for the case where Chrome is happily "listening" but nothing is coming
// through, and says so rather than leaving a silent screen.
setInterval(() => {
  if (!listening) return;
  const now = Date.now();
  if (lastResultAt && now - lastResultAt < STUCK_WHILE_TYPING_MS) return;
  if (!lastResultAt) return; // never heard anything yet; the backoff handles that
  if (now - lastCommitAt < STUCK_WHILE_TYPING_MS) return;
  if (interimText) return;
  log("listening but nothing recognized for a while - check the mic level", "warn");
}, 5000);

function startListening() {
  if (!SpeechRecognition) {
    setStatus("unsupported browser", "bad");
    showWarning(
      "<b>This browser cannot do speech recognition.</b> Use Google Chrome or " +
      "Microsoft Edge and open this page there. Firefox and Brave do not support it."
    );
    return;
  }
  // A fresh run gets a fresh budget: the backoff must not carry over from a
  // previous failure, or Start would appear to do nothing.
  restartDelayMs = RESTART_BASE_MS;
  unhealthySessions = 0;
  healthySessions = 0;
  lastResultAt = Date.now();
  lastCommitAt = Date.now();
  listening = true;
  el.toggle.textContent = "Stop listening";

  // One recogniser for the life of the page, reused for every session. Creating a
  // new object per session (as an earlier version did) leaks sessions and makes
  // Chrome refuse the next start().
  recognition = ensureRecognition();
  startSession();
}

/** start() the shared recogniser, tolerating Chrome's transient refusals. */
function startSession() {
  if (!listening || !recognition) return;
  if (sessionRunning) return;
  try {
    recognition.start();
    sessionRunning = true;
  } catch (err) {
    // Chrome throws while the previous session is still closing. The pending
    // 'end' event or the next restart tick will pick it up.
    sessionRunning = false;
    scheduleRestart(restartDelayMs);
  }
}

function stopListening() {
  listening = false;
  clearTimeout(restartTimer);
  clearTimeout(commitTimer);
  commitPhrase(interimText, "stop");
  if (recognition) {
    try { recognition.stop(); } catch (err) { /* already stopped */ }
  }
  sessionRunning = false;
  stopRecognitionStream();
  el.dot.classList.remove("live");
  el.toggle.textContent = "Start listening";
  setStatus("stopped");
}

// ---------------------------------------------------------------------------
// Microphone level meter -- reveals a quiet input, which is the most common
// reason a recogniser misses words.
//
// SpeechRecognition offers no device option: it captures the *system default*
// input and the page cannot redirect it. The meter is therefore opened on the
// system default as well, so what it shows is what the recogniser hears. To use a
// different device, change it in Windows:
//     Settings -> System -> Sound -> Input
// ---------------------------------------------------------------------------
async function startMeter() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    el.peakhint.textContent = "level meter unavailable";
    return;
  }
  try {
    const constraints = {
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    };
    meterStream = await navigator.mediaDevices.getUserMedia(constraints);
    const generation = ++meterGeneration;
    const track = meterStream.getAudioTracks()[0];
    const label = track ? track.label : "unknown";
    log(`metering the system default input: ${label}`);
    peakDb = -120;
    quietFrames = 0;

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(meterStream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    src.connect(analyser);

    const buf = new Float32Array(analyser.fftSize);
    let warned = false;

    const tick = () => {
      // Stop drawing once this stream has been replaced by a newer one.
      if (generation !== meterGeneration) return;
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      let peak = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = buf[i];
        sum += v * v;
        const a = Math.abs(v);
        if (a > peak) peak = a;
      }
      const rms = Math.sqrt(sum / buf.length);
      const db = 20 * Math.log10(Math.max(rms, 1e-7));
      const peakNow = 20 * Math.log10(Math.max(peak, 1e-7));
      peakDb = Math.max(peakDb, peakNow);

      const pct = Math.max(0, Math.min(100, ((db + 80) / 70) * 100));
      el.levelbar.style.width = `${pct}%`;
      el.levelbar.style.background = db > -35 ? "#4ade80" : db > -50 ? "#fbbf24" : "#f87171";
      el.leveltext.textContent = `${db.toFixed(0)} dBFS`;
      el.peakhint.textContent = `peak ${peakDb.toFixed(0)} dBFS`;

      if (db > -50) quietFrames = 0; else quietFrames++;

      // Report once, only if speech never reached a usable level.
      if (!warned && quietFrames > 300 && peakDb < -38) {
        warned = true;
        showWarning(
          `<b>Your microphone is very quiet (peak ${peakDb.toFixed(0)} dBFS; speech should reach about −20).</b> ` +
          "Recognition will keep missing words until this is fixed. Open " +
          "<code>Settings → System → Sound → Input</code>, pick this device, and raise its volume. " +
          "Also check the headset's own mute switch or dial."
        );
        log(`microphone very quiet: peak ${peakDb.toFixed(0)} dBFS`, "bad");
      }
      requestAnimationFrame(tick);
    };
    tick();
  } catch (err) {
    el.peakhint.textContent = "level meter unavailable";
    log(`level meter failed: ${err}`, "warn");
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
el.toggle.addEventListener("click", () => {
  if (listening) stopListening();
  else startListening();
});

el.clear.addEventListener("click", () => {
  finalText = "";
  interimText = "";
  markedText = "";
  lastSent = "";
  peakDb = -120;
  quietFrames = 0;
  resetWordTiming();
  render();
  send({ kind: "clear" });
  setStatus("cleared", "good");
  log("cleared the transcript");
});

if (el.timing) {
  el.timing.addEventListener("click", () => {
    // Cycle auto -> on -> off -> auto. The switch is shared with the voice
    // changer through the local server, so it applies without restarting it.
    timingUserSet = true;
    const next = TIMING_STATES[(TIMING_STATES.indexOf(timingMode) + 1) % TIMING_STATES.length];
    setTimingMode(next, true);
    if (next === "on") {
      log("word delays will be replayed; playback waits for the measured pauses");
    }
  });
}

el.lang.addEventListener("change", () => {
  log(`language -> ${el.lang.value}`);
  if (listening) {
    stopListening();
    setTimeout(startListening, 250);
  }
});

el.logtoggle.addEventListener("click", () => {
  el.logpanel.classList.toggle("collapsed");
  el.logtoggle.textContent = el.logpanel.classList.contains("collapsed")
    ? "Show activity"
    : "Hide activity";
});

document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "SELECT") return;
  if (event.code === "Space") {
    event.preventDefault();
    el.toggle.click();
  } else if (event.code === "Escape") {
    stopListening();
  }
});

window.addEventListener("beforeunload", () => {
  if (interimText) commitPhrase(interimText, "page-close");
});

// Restore whatever the server already holds (e.g. after a page refresh).
fetch("/state")
  .then((r) => r.json())
  .then((state) => {
    // Only when the server actually holds something: an empty answer arriving
    // after the first words have been spoken must not wipe what is on screen.
    if (state.text) {
      finalText = state.text;
      lastSent = finalText;
      render();
      log(`restored ${state.text.length} chars from the server`);
    }
  })
  .catch(() => {});

// A word-delay choice made in this browser is remembered here and re-applied, so
// a reload does not quietly undo it. Without one, the server's own default wins.
try {
  const remembered = localStorage.getItem("wordDelayMode");
  if (remembered && TIMING_STATES.includes(remembered)) {
    timingUserSet = true;
    setTimingMode(remembered, false);
  }
} catch (err) { /* private mode */ }

fetch("/timing")
  .then((r) => r.json())
  .then((state) => applyTimingState(state))
  .catch(() => {});

connect();
startMeter();

if (!SpeechRecognition) {
  setStatus("unsupported browser", "bad");
  showWarning(
    "<b>Speech recognition is not available in this browser.</b> Open this page in " +
    "Google Chrome or Microsoft Edge. (Firefox, Brave and Safari do not support it.)"
  );
}
