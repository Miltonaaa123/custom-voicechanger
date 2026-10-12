/**
 * Test the page's commit logic without a browser.
 *
 * The bug this guards against: Chrome's continuous speech recognition often ends
 * a session having only ever produced *interim* text. Committing only `isFinal`
 * results therefore loses whole phrases, which is exactly what happened -- the
 * page showed the words, the file stayed empty.
 *
 * Run:  node tests/test_page_logic.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP = path.join(__dirname, "..", "web", "app.js");
const source = fs.readFileSync(APP, "utf8");

let failures = 0;
function check(name, ok, detail) {
  if (ok) {
    console.log(`  PASS  ${name}`);
  } else {
    failures++;
    console.log(`  FAIL  ${name}  ${detail === undefined ? "" : JSON.stringify(detail)}`);
  }
}

/** Build a fake element good enough for the page's uses. */
function makeElement(id) {
  const el = {
    id,
    children: [],
    className: "",
    style: {},
    dataset: {},
    textContent: "",
    innerHTML: "",
    value: "",
    options: [],
    selectedIndex: 0,
    parentNode: null,
    scrollTop: 0,
    scrollHeight: 0,
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c) { this._set.has(c) ? this._set.delete(c) : this._set.add(c); },
      contains(c) { return this._set.has(c); },
    },
    setAttribute() {},
    insertBefore(child, ref) { this.children.push(child); child.parentNode = this; return child; },
    appendChild(child) { this.children.push(child); child.parentNode = this; return child; },
    removeChild(child) { this.children = this.children.filter((c) => c !== child); },
    addEventListener() {},
    getAudioTracks: () => [],
    focus() {},
    _listeners: {},
    get childElementCount() { return this.children.length; },
    get firstChild() { return this.children[0]; },
  };
  el.addEventListener = (name, fn) => { (el._listeners[name] = el._listeners[name] || []).push(fn); };
  el.click = () => (el._listeners.click || []).forEach((fn) => fn({ preventDefault() {} }));
  return el;
}

/** Run app.js in a sandbox and hand back its internals for testing. */
function loadPage() {
  const elements = new Map();
  const ids = [
    "dot", "status", "toggle", "clear", "lang", "transcript", "levelbar",
    "leveltext", "peakhint", "warning", "logpanel", "logbody", "logtoggle", "outpath",
    "timing", "timinghint",
  ];
  const body = makeElement("body");
  ids.forEach((id) => {
    const element = makeElement(id);
    element.parentNode = body; // the header controls live inside the page body
    elements.set(id, element);
  });

  const sent = [];       // messages the page pushed to the server
  const posts = [];      // HTTP posts the page made
  const logged = [];     // diagnostic lines
  let captured = null;   // the SpeechRecognition instance the page built

  const document = {
    getElementById: (id) => elements.get(id) || null,
    createElement: (tag) => makeElement(tag),
    addEventListener() {},
    body: makeElement("body"),
  };

  function FakeWebSocket(url) {
    this.url = url;
    this.readyState = 1; // OPEN
    this.send = (data) => sent.push(JSON.parse(data));
    this.close = () => {};
    setTimeout(() => this.onopen && this.onopen(), 0);
  }
  FakeWebSocket.OPEN = 1;

  function FakeSpeechRecognition() {
    captured = this;
    this.continuous = false;
    this.interimResults = false;
    this.lang = "";
    this.maxAlternatives = 1;
    this.start = () => { this.started = true; if (this.onstart) this.onstart(); };
    this.stop = () => { this.stopped = true; if (this.onend) this.onend(); };
  }

  const sandbox = {
    console,
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (id) => clearTimeout(id),
    setInterval: () => 0, // the stuck-watchdog is not exercised here
    clearInterval: () => {},
    Date,
    Math,
    JSON,
    requestAnimationFrame: () => 0,
    document,
    location: { protocol: "http:", host: "127.0.0.1:8777" },
    WebSocket: FakeWebSocket,
    fetch: (url, options) => {
      // The page reads /state and /timing on load and posts /timing on a toggle.
      posts.push({ url: String(url), options: options || null });
      if (String(url).includes("/timing")) {
        return Promise.resolve({
          json: () => Promise.resolve({ mode: "auto", enabled: true, min_gap_ms: 80, max_gap_ms: 1500 }),
        });
      }
      // A server holding nothing yet: this must not blank text the page already
      // has, which is what an empty string here used to do.
      return Promise.resolve({ json: () => Promise.resolve({}) });
    },
    navigator: { mediaDevices: undefined }, // level meter disabled in tests
    addEventListener() {},
    localStorage: {
      _data: {},
      getItem(key) { return this._data[key] === undefined ? null : this._data[key]; },
      setItem(key, value) { this._data[key] = String(value); },
    },
  };
  sandbox.window = sandbox;
  sandbox.window.SpeechRecognition = FakeSpeechRecognition;
  sandbox.globalThis = sandbox;

  const wrapped =
    source +
    "\n;globalThis.__t = { commitPhrase, buildRecognition: ensureRecognition, startListening, stopListening," +
    " scheduleCommit, log, getState: () => ({ finalText, interimText, lastSent, listening })," +
    " getRestart: () => ({ delay: restartDelayMs, healthy: healthySessions, unhealthy: unhealthySessions })," +
    " runRestartNow: () => { clearTimeout(restartTimer); if (!listening) return; recognition = ensureRecognition();" +
    "   sessionRunning = false; try { recognition.start(); } catch (e) {} }," +
    " getTiming: () => ({ mode: timingMode, userSet: timingUserSet, lag: speechLagFloorMs, min: timingMinMs," +
    "   marks: wordMarks.slice(), lastOnsetAt: lastOnsetAt, stopAt: speechStopAt })," +
    " setTimingMode: (m) => setTimingMode(m, false)," +
    " computeWordDelays: () => computeWordDelays()," +
    " withDelays: (t, d) => withDelays(t, d)," +
    " stripMarksIn: (t) => t.replace(/\\[\\[\\d+\\]\\]/g, ' ').replace(/\\s+/g, ' ').trim()," +
    " onendNow: () => { if (captured && captured.onend) captured.onend(); }," +
    " flush: () => { wsReady = true; while (pending.length) ws.send(pending.shift()); } };";

  vm.createContext(sandbox);
  vm.runInContext(wrapped, sandbox, { filename: "app.js" });

  // Capture the page's own log lines so we can assert on them.
  const pageLog = sandbox.__t.log;
  sandbox.__t.log = (m, c) => { logged.push(m); return pageLog(m, c); };

  return {
    t: sandbox.__t,
    sent,
    posts,
    logged,
    elements,
    getRecognition: () => captured,
    buildRecognition: sandbox.__t.buildRecognition,
    /**
     * Begin a session the way the page does: this clears the "a session is
     * already running" latch, which start() called directly would not.
     */
    startSession() {
      sandbox.__t.startListening();
      return captured;
    },
  };
}

/** Wait for `ms`, so the page's own clock produces a measurable gap. */
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function finalsFrom(sent) {
  return sent.filter((m) => m.kind === "final").map((m) => m.text);
}

// ---------------------------------------------------------------------------
// 1. A final result is committed.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "hello world" } }],
  });
  const finals = finalsFrom(page.sent);
  check("final result reaches the server", finals.length === 1 && finals[0].trim() === "hello world", finals);
  check("final result shown on the page", page.t.getState().finalText === "hello world", page.t.getState());
}

// ---------------------------------------------------------------------------
// 2. THE BUG: interim-only text must still be committed once the session ends.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: false, 0: { transcript: "what are you doing" } }],
  });
  check("interim is not written to the transcript file", finalsFrom(page.sent).length === 0, page.sent);
  check("interim is sent for live.txt", page.sent.some((m) => m.kind === "interim"), page.sent);

  rec.onend(); // Chrome ends the session without ever marking the result final
  const finals = finalsFrom(page.sent);
  check(
    "interim committed when the session ends",
    finals.length === 1 && finals[0].trim() === "what are you doing",
    finals
  );
}

// ---------------------------------------------------------------------------
// 3. A pause commits interim text even mid-session.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: false, 0: { transcript: "testing the pause path" } }],
  });
  // The page schedules a commit COMMIT_AFTER_MS after the last interim update.
  const done = new Promise((resolve) => setTimeout(resolve, 1100));
  done.then(() => {
    const finals = finalsFrom(page.sent);
    check(
      "pause commits interim text",
      finals.length === 1 && finals[0].trim() === "testing the pause path",
      finals
    );
    finish();
  });
}

// ---------------------------------------------------------------------------
// 4. No duplicates when a final arrives after a pause commit.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: false, 0: { transcript: "no duplicate please" } }],
  });
  // Same phrase then arrives as final.
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "no duplicate please" } }],
  });
  const finals = finalsFrom(page.sent);
  check("an identical final is not sent twice", finals.length === 1, finals);
}

// ---------------------------------------------------------------------------
// 5. A growing phrase only sends the new tail.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "hello what" } }],
  });
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "hello what are you doing" } }],
  });
  const finals = finalsFrom(page.sent);
  check(
    "a re-sent longer phrase only contributes its tail",
    finals.length === 2 && finals[1].trim() === "are you doing",
    finals
  );
  check(
    "combined text is correct",
    page.t.getState().finalText === "hello what are you doing",
    page.t.getState()
  );
}

// ---------------------------------------------------------------------------
// 6. Stop commits what is on screen.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: false, 0: { transcript: "stopping now" } }],
  });
  rec.stop();
  check("stop commits the pending guess", finalsFrom(page.sent).length === 1, page.sent);
}

// ---------------------------------------------------------------------------
// 7. Diagnostics actually report activity.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "diagnostics work" } }],
  });
  const events = page.sent.filter((m) => m.kind === "event").map((m) => m.detail);
  check("session start is logged", events.some((m) => /session #\d+ started/.test(m)), events);
  check("committed text is logged", events.some((m) => /heard/.test(m)), events);
  check("diagnostics reach the server", events.length > 0, events);
}

// ---------------------------------------------------------------------------
// 9. Repeatedly failing sessions back off instead of spinning.
//
// This is the abort loop: sessions that produce no audio and no text must not be
// restarted instantly, or the page hammers the service and transcribes nothing.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();

  check("restart delay starts small", page.t.getRestart().delay === 350, page.t.getRestart());

  // Session that died without producing anything.
  rec.onend();
  const afterFirst = page.t.getRestart();
  check("a silent session backs off", afterFirst.delay > 350, afterFirst);
  check("backoff is recorded as unhealthy", afterFirst.unhealthy === 1, afterFirst);

  // Two more silent failures push the delay up further.
  page.t.runRestartNow();
  const rec2 = page.getRecognition();
  rec2.onstart();
  rec2.onend();
  page.t.runRestartNow();
  const rec3 = page.getRecognition();
  rec3.onstart();
  rec3.onend();
  const afterThree = page.t.getRestart();
  check("backoff keeps growing", afterThree.delay > afterFirst.delay, afterThree);
  check("backoff stays bounded", afterThree.delay <= 8000, afterThree);
}

// ---------------------------------------------------------------------------
// 10. A healthy session resets the backoff.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onend(); // fails first, so the delay grows
  check("delay grew before recovery", page.t.getRestart().delay > 350, page.t.getRestart());

  page.t.runRestartNow();
  const rec2 = page.getRecognition();
  rec2.onstart();
  rec2.onresult({
    resultIndex: 0,
    results: [{ isFinal: true, 0: { transcript: "back in business" } }],
  });
  rec2.onend();
  const after = page.t.getRestart();
  check("a working session resets the backoff", after.delay === 350, after);
  check("healthy sessions are counted", after.healthy >= 1, after);
  check("its text was committed", finalsFrom(page.sent).length === 1, finalsFrom(page.sent));
}

// ---------------------------------------------------------------------------
// 11. A quiet session is reported as a microphone problem, not silence.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onerror({ error: "no-speech" });
  const events = page.sent.filter((m) => m.kind === "event").map((m) => m.detail);
  check(
    "no-speech is explained as an audio problem",
    events.some((m) => /audio too quiet|no speech/i.test(m)),
    events
  );
}

// ---------------------------------------------------------------------------
// 8. An empty result does not commit anything.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();
  rec.onstart();
  rec.onresult({ resultIndex: 0, results: [{ isFinal: false, 0: { transcript: "" } }] });
  rec.onend();
  check("empty results never reach the file", finalsFrom(page.sent).length === 0, page.sent);
}

// ---------------------------------------------------------------------------
// 12. A session with audio but no text still counts as a failure.
//
// This is the restart storm: `onaudiostart` fires, so the session looked healthy,
// it was retried instantly, and the page spun forever producing nothing. Audio
// alone must not count as success.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();
  const rec = page.startSession();

  rec.onstart();
  rec.onaudiostart(); // audio arrived...
  rec.onend();        // ...but the session produced no text

  const state = page.t.getRestart();
  check("audio without text counts as unhealthy", state.unhealthy === 1, state);
  check("backoff engaged despite audio", state.delay > 350, state);
}

// ---------------------------------------------------------------------------
// 13. The restart storm gives up instead of spinning forever.
// ---------------------------------------------------------------------------
{
  const page = loadPage();
  page.t.flush();

  let current = page.startSession();
  current.onaudiostart();
  current.onend();

  // Keep failing: each round still produces no text. The page should give up on
  // its own after UNHEALTHY_AFTER sessions.
  for (let round = 0; round < 6; round++) {
    if (!page.t.getState().listening) break;
    page.t.runRestartNow();
    current = page.getRecognition();
    if (!current) break;
    current.onstart();
    current.onaudiostart();
    current.onend();
  }

  const state = page.t.getRestart();
  check("the storm was counted", state.unhealthy >= 4, state);
  check("the page gave up listening", page.t.getState().listening === false, page.t.getState());
  const events = page.sent.filter((m) => m.kind === "event").map((m) => m.detail);
  check(
    "it explains the stall instead of spinning",
    events.some((m) => /repeated sessions produced no text/.test(m)),
    events
  );
}

let finished = false;
function finish() {
  if (finished) return;
  finished = true;
  console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILED`);
  process.exit(failures === 0 ? 0 : 1);
}

// ---------------------------------------------------------------------------
// 14. The delay between two words is measured and carried with them.
//
// Chrome reports words without timings, so the interval between their arrivals
// is what is measured -- with the smallest interval in the phrase removed, since
// that is the service's own reporting delay rather than silence from the speaker.
// ---------------------------------------------------------------------------
async function delayTests() {
  {
    const page = loadPage();
    page.t.flush();
    const rec = page.startSession();
    rec.onstart();
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "hello" } }],
    });
    // Two words back to back: that interval is the service's own reporting delay,
    // and it is what a real pause has to be measured against.
    await wait(20);
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "there" } }],
    });
    await wait(300);
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "what" } }],
    });

    const finals = finalsFrom(page.sent);
    check("every phrase reaches the server", finals.length === 3, finals);
    check("a word with no pause carries no delay", finals[1].trim() === "there", finals[1]);
    check("the measured pause travels with the word it precedes",
      /^\[\[\d+\]\] what $/.test(finals[2]), finals[2]);
    const shown = page.t.getState().finalText;
    check("the delay is not shown in the transcript", shown === "hello there what", shown);
  }

  // -------------------------------------------------------------------------
  // 15. Words that arrive together carry no delay: a batch is not a pause.
  // -------------------------------------------------------------------------
  {
    const page = loadPage();
    page.t.flush();
    const rec = page.startSession();
    rec.onstart();
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "one two three" } }],
    });
    const finals = finalsFrom(page.sent);
    check("no delay is invented between words reported together",
      finals.length === 1 && finals[0].trim() === "one two three", finals);
  }

  // -------------------------------------------------------------------------
  // 16. The pause between two detected phrases is measured as well.
  //
  // Chrome often ends a result on its own, so the word after a pause arrives as a
  // separate phrase. The onset clock deliberately survives a commit for this.
  // -------------------------------------------------------------------------
  {
    const page = loadPage();
    page.t.flush();
    const rec = page.startSession();
    rec.onstart();
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "hello" } }],
    });
    await wait(20);
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "there" } }],
    });
    await wait(500);
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "what" } }],
    });
    const finals = finalsFrom(page.sent);
    check("the phrase after a pause carries its own delay",
      /^\[\[\d+\]\] what $/.test(finals[2]), finals[2]);
    const ms = parseInt((finals[2].match(/\[\[(\d+)\]\]/) || [0, "0"])[1], 10);
    check("and it is roughly the pause that was heard", ms > 300 && ms < 700, ms);
  }

  // -------------------------------------------------------------------------
  // 17. A growing re-send only marks the new tail.
  // -------------------------------------------------------------------------
  {
    const page = loadPage();
    page.t.flush();
    const rec = page.startSession();
    rec.onstart();
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "hello what" } }],
    });
    await wait(200);
    rec.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: "hello what are you doing" } }],
    });
    const finals = finalsFrom(page.sent);
    check("the tail is what is sent", finals.length === 2 && /doing\s*$/.test(finals[1]), finals);
    check("the tail carries only its own delay",
      (finals[1].match(/\[\[/g) || []).length <= 1, finals[1]);
  }

  // -------------------------------------------------------------------------
  // 18. The switch in the header cycles and reaches the server.
  // -------------------------------------------------------------------------
  {
    const page = loadPage();
    page.t.flush();
    await wait(50); // the page reads the server's setting on load
    const button = page.elements.get("timing");
    check("the switch starts in auto", page.t.getTiming().mode === "auto", page.t.getTiming());
    check("it says so on the button", /auto/.test(button.textContent), button.textContent);
    check("the page asks the server where the switch is",
      page.posts.some((p) => p.url === "/timing" && !p.options), page.posts.map((p) => p.url));

    button.click();
    check("a click turns it on", page.t.getTiming().mode === "on", page.t.getTiming());
    check("which is announced to the voice changer",
      page.sent.some((m) => m.kind === "timing" && m.mode === "on"), page.sent);
    check("and remembered for the next visit",
      page.posts.some((p) => p.url === "/timing" && p.options && /"on"/.test(p.options.body)),
      page.posts.map((p) => p.url));

    button.click();
    check("a second click turns it off", page.t.getTiming().mode === "off", page.t.getTiming());
    button.click();
    check("and a third returns it to auto", page.t.getTiming().mode === "auto", page.t.getTiming());
  }

  finish();
}

delayTests().catch((err) => {
  failures++;
  console.log(`  FAIL  delay tests threw: ${err && err.stack ? err.stack : err}`);
  finish();
});

// Scenario 3 is asynchronous (it waits for the pause commit); give it time, then
// report. A hard exit keeps any stray timer from holding the process open.
setTimeout(finish, 6000);

