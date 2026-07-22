/**
 * Speech-to-task — click MIC to record, click STOP to transcribe (Jetson Whisper).
 */
(function () {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition || null;

  const state = {
    config: null,
    mode: "record",
    listening: false,
    recording: false,
    holdActive: false,
    autoRun: false,
    speakReplies: true,
    recognition: null,
    mediaStream: null,
    mediaRecorder: null,
    recordBlob: null,
    recordTimer: null,
    recordSeconds: 0,
    browserTranscript: "",
    browserInterim: "",
    selectedDeviceId: localStorage.getItem("speech_mic_id") || "",
    lastGoalRunning: false,
    announcedStarts: Object.create(null),
    announcedEnds: Object.create(null),
    speakTimer: null,
    speakGen: 0,
    micRefreshBusy: false,
    recordStarting: false,
    transcribing: false,
    uiWired: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(msg, level) {
    const el = $("speech-status");
    if (!el) return;
    el.textContent = msg;
    el.dataset.level = level || "info";
  }

  function setInterim(msg) {
    const el = $("speech-interim");
    if (el) el.textContent = msg || "";
  }

  function normalizeTranscript(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function fillGoal(text) {
    const input = $("goal-input");
    if (!input) return;
    input.value = text;
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  async function submitGoalFromSpeech(text) {
    const goal = normalizeTranscript(text);
    if (!goal) {
      setStatus("No speech detected — try again.", "warn");
      return;
    }
    fillGoal(goal);
    if (typeof window.submitAgentGoal === "function") {
      await window.submitAgentGoal(goal);
    } else {
      $("run-goal")?.click();
    }
  }

  function speak(text) {
    if (!state.speakReplies || !window.speechSynthesis || !text) return;
    const now = Date.now();
    const msg = String(text);
    // Hard cooldown: never re-speak same (or any) line in a short window.
    if (now - (state.lastSpeakAt || 0) < 2500) return;
    if (msg === state.lastSpeakText && now - (state.lastSpeakAt || 0) < 10000) return;
    state.lastSpeakText = msg;
    state.lastSpeakAt = now;
    state.speakGen += 1;
    const gen = state.speakGen;
    if (state.speakTimer) {
      clearTimeout(state.speakTimer);
      state.speakTimer = null;
    }
    try {
      window.speechSynthesis.cancel();
    } catch (_) {
      /* ignore */
    }
    state.speakTimer = setTimeout(() => {
      state.speakTimer = null;
      if (gen !== state.speakGen) return;
      if (!state.speakReplies || !window.speechSynthesis) return;
      try {
        window.speechSynthesis.cancel();
      } catch (_) {
        /* ignore */
      }
      const u = new SpeechSynthesisUtterance(msg);
      u.rate = 1.05;
      u.pitch = 1;
      u.lang = "en-US";
      // Chromium Linux often re-queues the same utterance; pause/resume clears it.
      window.speechSynthesis.speak(u);
      try {
        window.speechSynthesis.pause();
        window.speechSynthesis.resume();
      } catch (_) {
        /* ignore */
      }
    }, 30);
  }

  function onGoalStatus(gs, events) {
    // Status text only — voice is owned by submitAgentGoal (one tab, one shot).
    if (!gs) return;
    if (gs.running) {
      state.lastGoalRunning = true;
      return;
    }
    if (state.lastGoalRunning && gs.goal) {
      state.lastGoalRunning = false;
      if (gs.error) setStatus(`Goal error: ${gs.error}`, "error");
      else if (gs.ended_at) {
        const note = gs.note ? ` ${gs.note}` : "";
        setStatus(`Finished: ${gs.goal}${note}`, "ok");
      }
    }
  }

  window.speakGoalUpdate = function speakGoalUpdate(phase, gs, events) {
    if (!state.speakReplies) return;
    if (phase === "start") {
      speak("Okay, executing the task.");
      return;
    }
    if (phase === "end") {
      if (gs?.error) {
        speak("Task failed.");
        setStatus(`Goal error: ${gs.error}`, "error");
        return;
      }
      const summary = taskSummaryForSpeech(gs || {}, events || null);
      speak(summary ? `Task finished. ${summary}.` : "Task finished.");
      if (gs?.goal) {
        const note = gs.note ? ` ${gs.note}` : "";
        setStatus(`Finished: ${gs.goal}${note}`, "ok");
      }
    }
  };

  const MOTION_DIR = {
    move_up: "up",
    move_down: "down",
    move_left: "left",
    move_right: "right",
    move_forward: "forward",
    move_backward: "backward",
  };

  const GRIPPER_PHRASE = {
    open_gripper: "opened gripper",
    close_gripper: "closed gripper",
    toggle_gripper: "toggled gripper",
  };

  function formatCentimeters(cm) {
    const n = Math.round(Math.abs(cm));
    if (!n) return "";
    return n === 1 ? "1 centimeter" : `${n} centimeters`;
  }

  function formatDistanceSpeech(meters) {
    if (meters == null || !Number.isFinite(Number(meters))) return "";
    return formatCentimeters(Number(meters) * 100);
  }

  function formatUnitSpeech(amount, unit) {
    const n = parseFloat(amount);
    if (!Number.isFinite(n) || n <= 0) return "";
    const u = (unit || "").replace(/s$/, "");
    if (u === "cm" || u === "centimeter") return formatCentimeters(n);
    if (u === "mm" || u === "millimeter") return formatCentimeters(n / 10);
    if (u === "m" || u === "meter") return formatCentimeters(n * 100);
    return `${n} ${unit}`;
  }

  function summarizeMotionEvent(evt) {
    const tool = evt?.tool;
    const dir = MOTION_DIR[tool];
    if (dir) {
      const dist = formatDistanceSpeech(evt.inputs?.distance_m);
      return dist ? `${dist} ${dir}` : dir;
    }
    return GRIPPER_PHRASE[tool] || null;
  }

  function summarizeFromEvents(events, sinceTs) {
    if (!Array.isArray(events) || !sinceTs) return null;
    const windowStart = sinceTs - 2;
    const phrases = [];
    for (let i = events.length - 1; i >= 0; i--) {
      const evt = events[i];
      if (!evt || evt.ts < windowStart) continue;
      const result = evt.result;
      if (result?.status === "error") continue;
      const phrase = summarizeMotionEvent(evt);
      if (phrase) phrases.push(phrase);
    }
    return phrases.length ? phrases.join(", then ") : null;
  }

  function summarizeFromGoal(goal) {
    const g = (goal || "").trim().toLowerCase();
    if (!g) return null;

    const withDistance = [
      /(?:go|move)\s+(up|down|left|right|forward|back(?:ward)?)\s+(\d+(?:\.\d+)?)\s*(cm|centimeters?|mm|millimeters?|m|meters?)\b/,
      /(\d+(?:\.\d+)?)\s*(cm|centimeters?|mm|millimeters?|m|meters?)\s+(up|down|left|right|forward|back(?:ward)?)\b/,
    ];
    for (const re of withDistance) {
      const m = g.match(re);
      if (!m) continue;
      let dir;
      let amount;
      let unit;
      if (/^(up|down|left|right|forward|back)/.test(m[1])) {
        [, dir, amount, unit] = m;
      } else {
        [, amount, unit, dir] = m;
      }
      dir = dir.replace(/^back(?:ward)?$/, "backward");
      const dist = formatUnitSpeech(amount, unit);
      return dist ? `${dist} ${dir}` : dir;
    }

    const simple = g.match(/(?:go|move)\s+(up|down|left|right|forward|back(?:ward)?)\b/);
    if (simple) return simple[1].replace(/^back(?:ward)?$/, "backward");

    if (/\bopen\b.*\bgripper\b/.test(g)) return "opened gripper";
    if (/\bclose\b.*\bgripper\b/.test(g)) return "closed gripper";

    return goal.trim();
  }

  function taskSummaryForSpeech(gs, events) {
    return summarizeFromEvents(events, gs?.started_at) || summarizeFromGoal(gs?.goal);
  }

  window.onAgentGoalStatus = function (goalStatus, events) {
    onGoalStatus(goalStatus, events);
  };

  function updateMicButton() {
    const btn = $("speech-talk-btn");
    if (!btn) return;
    if (state.transcribing) {
      btn.textContent = "Transcribing…";
      btn.disabled = true;
    } else if (state.recording) {
      btn.textContent = `STOP (${state.recordSeconds}s)`;
      btn.classList.add("listening");
      btn.disabled = false;
      btn.setAttribute("aria-pressed", "true");
    } else if (state.recordStarting) {
      btn.textContent = "Opening mic…";
      btn.classList.remove("listening");
      btn.disabled = true;
    } else {
      btn.textContent = "Mic - Speak";
      btn.classList.remove("listening");
      btn.disabled = false;
      btn.setAttribute("aria-pressed", "false");
    }
  }

  function startRecordTimer() {
    stopRecordTimer();
    state.recordSeconds = 0;
    state.recordTimer = setInterval(() => {
      state.recordSeconds += 1;
      updateMicButton();
    }, 1000);
  }

  function stopRecordTimer() {
    if (state.recordTimer) {
      clearInterval(state.recordTimer);
      state.recordTimer = null;
    }
  }

  async function loadSpeechConfig() {
    try {
      const res = await fetch("/api/speech/config");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.config = await res.json();
    } catch (e) {
      state.config = { browser_stt: !!SpeechRecognition, cloud_available: false, record_available: true };
      setStatus(`Speech config unavailable: ${e.message}`, "warn");
      return;
    }
    updateModeUi();
    const provider = state.config?.cloud_provider || "";
    const model = state.config?.whisper_model || state.config?.local_whisper_model || "?";
    const where = provider === "openai_whisper" ? "OpenAI" : "local on Jetson";
    setStatus(`Ready: ${model} (${where})`, "ok");
  }

  function isChromiumLinux() {
    const ua = navigator.userAgent || "";
    return /Linux/i.test(ua) && /Chromium/i.test(ua);
  }

  function updateModeUi() {
    const hint = $("speech-mode-hint");
    if (!hint) return;
    if (state.mode === "record") {
      hint.textContent = "1) Click MIC  2) Speak  3) Click STOP  4) Press Run";
    } else {
      hint.textContent = "Browser STT — click MIC to start/stop.";
    }
  }

  function switchToRecordMode(reason) {
    state.mode = "record";
    localStorage.setItem("speech_mode", "record");
    updateModeUi();
    if (reason) setStatus(reason, "warn");
  }

  function populateMicSelect(inputs) {
    const select = $("speech-mic-select");
    if (!select) return;
    const prev = state.selectedDeviceId || select.value;
    select.innerHTML = "";
    const def = document.createElement("option");
    def.value = "";
    def.textContent = "Default microphone";
    select.appendChild(def);
    inputs.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      const label = d.label || `Mic ${d.deviceId.slice(0, 8)}`;
      opt.textContent = label.toLowerCase().includes("jabra") ? `★ ${label}` : label;
      select.appendChild(opt);
    });
    if (inputs.some((d) => d.deviceId === prev)) {
      select.value = prev;
      state.selectedDeviceId = prev;
    } else {
      const jabra = inputs.find((d) => (d.label || "").toLowerCase().includes("jabra"));
      if (jabra) {
        select.value = jabra.deviceId;
        state.selectedDeviceId = jabra.deviceId;
        localStorage.setItem("speech_mic_id", jabra.deviceId);
      }
    }
  }

  async function refreshMicList(requestPermission = false) {
    const select = $("speech-mic-select");
    if (!select || state.micRefreshBusy) return;
    state.micRefreshBusy = true;
    select.disabled = true;
    try {
      let devices = await navigator.mediaDevices.enumerateDevices();
      let inputs = devices.filter((d) => d.kind === "audioinput");
      if (requestPermission && inputs.every((d) => !d.label)) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((t) => t.stop());
        devices = await navigator.mediaDevices.enumerateDevices();
        inputs = devices.filter((d) => d.kind === "audioinput");
      }
      populateMicSelect(inputs);
    } catch (e) {
      setStatus(`Mic permission: ${e.message}`, "error");
    } finally {
      select.disabled = false;
      state.micRefreshBusy = false;
    }
  }

  function releaseStream() {
    state.mediaStream?.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
    state.mediaRecorder = null;
    state.recordBlob = null;
  }

  async function openMicStream() {
    const base = { echoCancellation: false, noiseSuppression: false, autoGainControl: true };
    const deviceId = state.selectedDeviceId;
    if (deviceId) {
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: { ...base, deviceId: { ideal: deviceId } },
        });
      } catch (_) {
        /* try default */
      }
    }
    return navigator.mediaDevices.getUserMedia({ audio: base });
  }

  function pickRecorderMime() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"];
    return types.find((t) => window.MediaRecorder?.isTypeSupported(t)) || "";
  }

  function mimeToFilename(mime) {
    if (!mime) return "speech.webm";
    if (mime.includes("ogg")) return "speech.ogg";
    return "speech.webm";
  }

  function setListeningUi(active) {
    document.querySelectorAll(".speech-talk-trigger").forEach((el) => {
      el.classList.toggle("listening", active);
    });
  }

  async function startMicRecord() {
    if (state.recordStarting || state.recording || state.transcribing) return;
    if (!window.MediaRecorder) {
      setStatus("MediaRecorder not supported — use Chrome/Chromium.", "error");
      return;
    }

    state.recordStarting = true;
    updateMicButton();
    setStatus("Allow microphone if prompted…", "active");

    try {
      releaseStream();
      state.mediaStream = await openMicStream();
      state.recordBlob = null;
      state.recordStartedAt = Date.now();

      const mime = pickRecorderMime();
      const options = mime ? { mimeType: mime } : undefined;
      state.mediaRecorder = new MediaRecorder(state.mediaStream, options);

      const stopped = new Promise((resolve) => {
        state.mediaRecorder.onstop = () => resolve();
      });

      state.mediaRecorder.ondataavailable = (ev) => {
        if (ev.data?.size > 0) state.recordBlob = ev.data;
      };

      // Single blob on stop — avoids corrupt WebM from chunked concat
      state.mediaRecorder.start();
      state.recording = true;
      state.recordStarting = false;
      startRecordTimer();
      updateMicButton();
      setStatus("🔴 Recording — speak, then click STOP.", "active");
      setInterim("");
    } catch (e) {
      state.recordStarting = false;
      releaseStream();
      updateMicButton();
      setStatus(`Mic error: ${e.message}. Click Refresh mics, pick ★ Jabra, retry.`, "error");
    }
  }

  async function stopMicRecord() {
    if (!state.recording || !state.mediaRecorder) return;

    state.recording = false;
    stopRecordTimer();
    updateMicButton();

    const heldMs = Date.now() - (state.recordStartedAt || 0);
    if (heldMs < 1000) {
      await new Promise((r) => setTimeout(r, 1000 - heldMs));
    }

    setStatus("Saving audio…", "active");
    const recorder = state.mediaRecorder;

    try {
      if (recorder.state === "recording") {
        recorder.stop();
      }
      await new Promise((resolve) => {
        if (recorder.state === "inactive") resolve();
        else recorder.onstop = () => resolve();
      });
      await new Promise((r) => setTimeout(r, 100));
    } catch (e) {
      releaseStream();
      updateMicButton();
      setStatus(`Stop failed: ${e.message}`, "error");
      return;
    }

    const mime = recorder.mimeType || pickRecorderMime() || "audio/webm";
    const blob = state.recordBlob || new Blob([], { type: mime });
    releaseStream();
    updateMicButton();

    if (!blob.size || blob.size < 1000) {
      setStatus(
        `No audio captured (${blob.size} bytes, ${(heldMs / 1000).toFixed(1)}s). Speak louder, try ★ Jabra.`,
        "warn"
      );
      return;
    }

    state.transcribing = true;
    updateMicButton();
    setStatus(`Transcribing ${(heldMs / 1000).toFixed(1)}s (${blob.size} bytes)…`, "active");

    const form = new FormData();
    form.append("audio", blob, mimeToFilename(mime));
    try {
      const res = await fetch("/api/speech/transcribe", { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus(body.detail || body.reason || `Server error ${res.status}`, "error");
        return;
      }
      const text = normalizeTranscript(body.text);
      if (!text) {
        setStatus("Empty transcript — speak louder and try again.", "warn");
        return;
      }
      fillGoal(text);
      setInterim(`You said: “${text}”`);
      setStatus(`OK (${body.model || "whisper"}): check text above, then Run.`, "ok");
      if (state.autoRun) await submitGoalFromSpeech(text);
    } catch (e) {
      setStatus(`Network error: ${e.message}`, "error");
    } finally {
      state.transcribing = false;
      updateMicButton();
    }
  }

  function stopRecognitionEngine() {
    state.listening = false;
    try {
      state.recognition?.stop();
    } catch (_) {
      /* ignore */
    }
  }

  function bindBrowserRecognition(rec) {
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.continuous = true;
    rec.onresult = (ev) => {
      let interim = "";
      let finals = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) finals += t;
        else interim += t;
      }
      if (finals) state.browserTranscript += finals;
      state.browserInterim = interim;
      const combined = normalizeTranscript(state.browserTranscript + state.browserInterim);
      setInterim(combined);
      if (combined) fillGoal(combined);
    };
    rec.onerror = (ev) => {
      if (ev.error === "network") switchToRecordMode("Browser STT failed — use Record mode.");
      else setStatus(`Speech error: ${ev.error}`, "error");
      state.holdActive = false;
      stopRecognitionEngine();
      setListeningUi(false);
    };
  }

  async function finishBrowserListen() {
    state.holdActive = false;
    stopRecognitionEngine();
    setListeningUi(false);
    const text = normalizeTranscript(state.browserTranscript + state.browserInterim);
    state.browserTranscript = "";
    state.browserInterim = "";
    if (text) {
      fillGoal(text);
      setInterim(`You said: “${text}”`);
      if (state.autoRun) await submitGoalFromSpeech(text);
    }
  }

  function startBrowserListen() {
    if (!SpeechRecognition) {
      setStatus("Use Record mode on this browser.", "error");
      return;
    }
    state.browserTranscript = "";
    state.browserInterim = "";
    state.holdActive = true;
    state.recognition = new SpeechRecognition();
    bindBrowserRecognition(state.recognition);
    state.listening = true;
    setListeningUi(true);
    setStatus("Browser listening… click MIC to stop.", "active");
    try {
      state.recognition.start();
    } catch (e) {
      setStatus(`Browser STT failed: ${e.message}`, "error");
    }
  }

  async function onMicClick(ev) {
    ev.preventDefault();
    if (state.transcribing || state.recordStarting) return;
    if (state.mode === "browser") {
      if (state.listening) await finishBrowserListen();
      else startBrowserListen();
      return;
    }
    if (state.recording) await stopMicRecord();
    else await startMicRecord();
  }

  function wireUi() {
    if (state.uiWired) return;
    state.uiWired = true;
    $("speech-mode-record")?.addEventListener("click", () => {
      state.mode = "record";
      localStorage.setItem("speech_mode", "record");
      updateModeUi();
    });
    $("speech-mode-browser")?.addEventListener("click", () => {
      state.mode = "browser";
      localStorage.setItem("speech_mode", "browser");
      updateModeUi();
    });
    $("speech-mic-refresh")?.addEventListener("click", () => refreshMicList(true));
    $("speech-mic-select")?.addEventListener("change", (ev) => {
      state.selectedDeviceId = ev.target.value;
      localStorage.setItem("speech_mic_id", state.selectedDeviceId);
      setStatus(`Mic: ${ev.target.selectedOptions[0]?.textContent || "default"}`, "ok");
    });
    $("speech-auto-run")?.addEventListener("change", (ev) => {
      state.autoRun = ev.target.checked;
      localStorage.setItem("speech_auto_run", state.autoRun ? "1" : "0");
    });
    $("speech-speak-replies")?.addEventListener("change", (ev) => {
      state.speakReplies = ev.target.checked;
      localStorage.setItem("speech_speak_replies", state.speakReplies ? "1" : "0");
    });
    document.querySelectorAll(".speech-talk-trigger").forEach((btn) => {
      btn.addEventListener("click", onMicClick);
    });
  }

  window.initSpeechTask = async function () {
    state.autoRun = localStorage.getItem("speech_auto_run") === "1";
    state.speakReplies = localStorage.getItem("speech_speak_replies") !== "0";
    state.mode = localStorage.getItem("speech_mode") === "browser" ? "browser" : "record";
    const autoEl = $("speech-auto-run");
    const speakEl = $("speech-speak-replies");
    if (autoEl) autoEl.checked = state.autoRun;
    if (speakEl) speakEl.checked = state.speakReplies;
    wireUi();
    updateMicButton();
    await loadSpeechConfig();
    if (isChromiumLinux() && state.mode === "browser") {
      switchToRecordMode("Use Record mode on Chromium/Ubuntu.");
    }
    refreshMicList(false).catch(() => {});
    updateModeUi();
  };
})();
