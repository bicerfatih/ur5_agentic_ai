const $ = (id) => document.getElementById(id);

const stateEl = $("telemetry");
const eventsEl = $("events");
const resultEl = $("tool-result");
const sitePill = $("site-pill");
const armPill = $("arm-pill");
const modePill = $("mode-pill");
const safetyPill = $("safety-pill");
const motionPill = $("motion-pill");
const cameraPreview = $("camera-preview");
const cameraHint = $("camera-hint");
const detectedLabelsEl = $("detected-labels");
const goalStatusEl = $("goal-status");
const liveFeedToggle = $("live-feed-toggle");
const detectToggle = $("detect-toggle");

let latestJoints = [0, -1.57, 0, -1.57, 0, 0];
let lastTelemetryState = null;
let digitalTwin = null;
let liveFeedTimer = null;
let detectEnabled = false;
let livePreviewObjectUrl = null;

async function postTool(name, inputs = {}) {
  const res = await fetch("/api/tool", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, inputs }),
  });
  const body = await res.json();
  resultEl.textContent = JSON.stringify(body, null, 2);
  if (name === "get_camera_frame") {
    const result = body?.result || {};
    if (result.status === "done" && result.path) {
      refreshCameraPreview();
    } else {
      cameraHint.textContent = `Capture failed: ${result.reason || "unknown error"}`;
    }
  }
}

function renderEvents(events = []) {
  eventsEl.innerHTML = "";
  events.slice(0, 12).forEach((evt) => {
    const div = document.createElement("div");
    div.className = "event";
    div.textContent = `${new Date(evt.ts * 1000).toLocaleTimeString()}  ${evt.tool}  (${evt.elapsed_ms}ms)\n` +
      JSON.stringify(evt.result).slice(0, 240);
    eventsEl.appendChild(div);
  });
}

function renderDetectionLabels(det = {}) {
  const count = Number(det.count) || 0;
  const labels = Array.isArray(det.unique_labels) && det.unique_labels.length
    ? det.unique_labels
    : Array.isArray(det.labels)
      ? [...new Set(det.labels)]
      : [];
  if (!count && !labels.length) {
    detectedLabelsEl.textContent = detectEnabled
      ? "Detected: (none)"
      : "Detected: —";
    detectedLabelsEl.classList.add("empty");
    return;
  }
  detectedLabelsEl.classList.remove("empty");
  const names = labels.length ? labels.join(", ") : "—";
  detectedLabelsEl.textContent = `Detected (${count}): ${names}`;
}

function normalizeRobotState(st) {
  const out = { ...(st || {}) };
  if (!Array.isArray(out.joint_positions_rad) && Array.isArray(out.joint_positions_deg)) {
    out.joint_positions_rad = out.joint_positions_deg.map((d) => (Number(d) * Math.PI) / 180);
  }
  if (Array.isArray(out.joint_positions_rad)) {
    out.joint_positions_rad = out.joint_positions_rad.map((v) => Number(v));
    const maxAbs = Math.max(...out.joint_positions_rad.map((v) => Math.abs(v)));
    if (maxAbs > 6.5) {
      out.joint_positions_rad = out.joint_positions_rad.map((d) => (d * Math.PI) / 180);
    }
  }
  return out;
}

function renderState(payload) {
  const st = normalizeRobotState(payload.state);
  lastTelemetryState = st;
  if (st.error && !st.joint_positions_rad) {
    if (digitalTwin && $("twin-hud")) {
      $("twin-hud").textContent = `Robot read error: ${st.error}`;
    }
  }
  if (Array.isArray(st.joint_positions_rad)) {
    latestJoints = st.joint_positions_rad;
  }
  sitePill.textContent = `site: ${payload.site || "-"}`;
  armPill.textContent = `arm: ${st.arm_model || "-"}`;
  modePill.textContent = `mode: ${st.robot_mode}`;
  safetyPill.textContent = `safety: ${st.safety_mode}`;
  const rtde = st.rtde_control || {};
  const motionOk = rtde.motion_enabled !== false && rtde.connected !== false;
  if (st.simulated) {
    motionPill.textContent = "motion: sim";
    motionPill.style.borderColor = "rgba(34,197,94,.6)";
  } else if (rtde.connected === false || rtde.motion_enabled === false) {
    motionPill.textContent = "motion: OFF";
    motionPill.style.borderColor = "rgba(239,68,68,.85)";
    motionPill.title = rtde.hint || "RTDE control not connected";
  } else {
    motionPill.textContent = "motion: ON";
    motionPill.style.borderColor = "rgba(34,197,94,.6)";
    motionPill.title = "";
  }
  modePill.style.borderColor = st.robot_mode === 7 ? "rgba(34,197,94,.6)" : "rgba(245,158,11,.7)";
  safetyPill.style.borderColor = st.safety_mode === 1 ? "rgba(34,197,94,.6)" : "rgba(245,158,11,.7)";
  stateEl.textContent = JSON.stringify(st, null, 2);
  if (digitalTwin) {
    digitalTwin.updateFromState(st);
  } else if ($("twin-hud")) {
    $("twin-hud").textContent = "Telemetry OK — starting 3D twin…";
  }
  if (payload.camera_path) {
    cameraHint.textContent = payload.camera_path;
  }
  if (payload.detection && !liveFeedTimer) {
    renderDetectionLabels(payload.detection);
  }
  const gs = payload.goal_status || {};
  if (gs.running) {
    goalStatusEl.textContent = `Running: ${gs.goal || ""} (check terminal for live tool output)`;
  } else if (gs.error) {
    goalStatusEl.textContent = `Error: ${gs.error}`;
  } else if (gs.ended_at) {
    const note = gs.note ? ` — ${gs.note}` : "";
    goalStatusEl.textContent = `Finished: ${gs.goal || ""}${note}`;
  }
}

function refreshCameraPreview() {
  const cacheBust = `t=${Date.now()}`;
  cameraPreview.src = `/api/camera/latest?${cacheBust}`;
  cameraPreview.onerror = () => {
    cameraHint.textContent = "No camera image yet. Click capture first.";
  };
  cameraPreview.onload = () => {
    cameraHint.textContent = "Latest capture loaded.";
  };
}

async function refreshLiveFrame() {
  const detect = detectEnabled ? 1 : 0;
  const cacheBust = `t=${Date.now()}`;
  try {
    const res = await fetch(`/api/camera/live.jpg?detect=${detect}&${cacheBust}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const blob = await res.blob();
    if (livePreviewObjectUrl) {
      URL.revokeObjectURL(livePreviewObjectUrl);
    }
    livePreviewObjectUrl = URL.createObjectURL(blob);
    cameraPreview.src = livePreviewObjectUrl;
    if (detectEnabled) {
      const detRes = await fetch("/api/detection");
      if (detRes.ok) {
        renderDetectionLabels(await detRes.json());
      }
      cameraHint.textContent = "Live feed + detection running.";
    } else {
      renderDetectionLabels({ count: 0, labels: [] });
      cameraHint.textContent = "Live feed running.";
    }
  } catch {
    cameraHint.textContent = "Live feed unavailable. Check camera connection.";
  }
}

function setLiveFeed(enabled) {
  if (enabled) {
    if (liveFeedTimer) return;
    refreshLiveFrame();
    liveFeedTimer = setInterval(refreshLiveFrame, 350);
    liveFeedToggle.textContent = "Stop Live Feed";
  } else {
    if (liveFeedTimer) {
      clearInterval(liveFeedTimer);
      liveFeedTimer = null;
    }
    liveFeedToggle.textContent = "Start Live Feed";
    cameraHint.textContent = "Live feed stopped.";
    if (!detectEnabled) {
      renderDetectionLabels({ count: 0, labels: [] });
    }
  }
}

document.querySelectorAll("button[data-tool]").forEach((btn) => {
  btn.addEventListener("click", () => postTool(btn.dataset.tool, {}));
});

$("run-custom").addEventListener("click", () => {
  const name = $("tool-name").value.trim();
  if (!name) return;
  let inputs = {};
  const raw = $("tool-inputs").value.trim();
  if (raw) {
    try { inputs = JSON.parse(raw); }
    catch { resultEl.textContent = "Invalid JSON inputs"; return; }
  }
  postTool(name, inputs);
});

$("capture-and-refresh").addEventListener("click", async () => {
  // Capture action should show a stable still frame preview.
  if (liveFeedTimer) {
    setLiveFeed(false);
  }
  await postTool("get_camera_frame", {});
});
liveFeedToggle.addEventListener("click", () => {
  setLiveFeed(!liveFeedTimer);
});
detectToggle.addEventListener("click", () => {
  detectEnabled = !detectEnabled;
  detectToggle.textContent = detectEnabled ? "Detect ON" : "Detect OFF";
  detectToggle.style.borderColor = detectEnabled ? "rgba(34,197,94,.75)" : "";
  if (liveFeedTimer) {
    refreshLiveFrame();
  } else {
    renderDetectionLabels({ count: 0, labels: [] });
    cameraHint.textContent = detectEnabled
      ? "Detection enabled. Start Live Feed to see boxes and labels below."
      : "Detection disabled.";
  }
});

$("run-goal").addEventListener("click", async () => {
  const goal = $("goal-input").value.trim();
  if (!goal) {
    goalStatusEl.textContent = "Goal is empty.";
    return;
  }
  goalStatusEl.textContent = "Submitting goal...";
  const res = await fetch("/api/goal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal }),
  });
  const body = await res.json();
  if (!res.ok) {
    goalStatusEl.textContent = body.detail || "Failed to submit goal.";
    return;
  }
  goalStatusEl.textContent = `Accepted: ${goal}`;
});

async function pullTelemetry(site) {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) return;
    const data = await res.json();
    renderState({
      site,
      state: data.state || {},
      events: data.events || [],
      goal_status: data.goal_status,
      detection: data.detection,
    });
    renderEvents(data.events || []);
  } catch {
    if ($("twin-hud")) {
      $("twin-hud").textContent = "Cannot reach /api/state — is the console running?";
    }
  }
}

let telemetryPollTimer = null;
let telemetryWs = null;

function connectTelemetryWs(site) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  telemetryWs = ws;

  ws.onopen = () => {
    if (digitalTwin?.ready && lastTelemetryState) {
      digitalTwin.updateFromState(lastTelemetryState);
    }
  };
  ws.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    payload.site = site;
    renderState(payload);
    renderEvents(payload.events || []);
  };
  ws.onclose = () => {
    if (telemetryWs !== ws) return;
    if ($("twin-hud")) {
      $("twin-hud").textContent = "Live telemetry: HTTP (1s) — reconnecting WebSocket…";
    }
    setTimeout(() => connectTelemetryWs(site), 2500);
  };
  ws.onerror = () => {
    if ($("twin-hud")) {
      $("twin-hud").textContent = "Live telemetry: HTTP (1s) — WebSocket retry…";
    }
  };
}

async function init() {
  ensureTwin();
  const cfg = await (await fetch("/api/config")).json();
  const site = cfg.site;
  sitePill.textContent = `site: ${site}`;

  await pullTelemetry(site);
  if (telemetryPollTimer) clearInterval(telemetryPollTimer);
  telemetryPollTimer = setInterval(() => pullTelemetry(site), 1000);

  connectTelemetryWs(site);

  setLiveFeed(true);
}

function loadThreeJs(timeoutMs = 15000) {
  return new Promise((resolve) => {
    if (typeof THREE !== "undefined") {
      resolve(true);
      return;
    }
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      if (typeof THREE !== "undefined") {
        resolve(true);
        return;
      }
      if (Date.now() > deadline) {
        resolve(false);
        return;
      }
      setTimeout(tick, 80);
    };
    tick();
  });
}

function waitForTwinReady(twin, timeoutMs = 5000) {
  return new Promise((resolve) => {
    if (twin?.ready) {
      resolve(twin);
      return;
    }
    const deadline = Date.now() + timeoutMs;
    const tick = () => {
      if (twin?.ready) {
        resolve(twin);
        return;
      }
      if (Date.now() > deadline) {
        resolve(twin);
        return;
      }
      setTimeout(tick, 50);
    };
    tick();
  });
}

function ensureTwin() {
  if (digitalTwin?.ready && digitalTwin.mode === "3d") return true;
  if (digitalTwin && !digitalTwin.ready) return true;
  if (digitalTwin) return false;
  const canvas = $("twin-canvas");
  const canvas2d = $("twin-canvas-2d");
  const hud = $("twin-hud");
  if (!canvas) return false;
  if (!window.UR5DigitalTwin) {
    if (hud) hud.textContent = "twin.js failed to load. Hard refresh (Ctrl+Shift+R).";
    return false;
  }
  if (typeof THREE === "undefined") {
    if (hud) {
      hud.textContent = "Three.js missing — restart Ops Console (server downloads it on startup).";
    }
    return false;
  }
  digitalTwin = new window.UR5DigitalTwin(canvas, canvas2d, hud, { prefer3d: true });
  return true;
}

async function bootTwin() {
  const hud = $("twin-hud");
  let cfg = {};
  try {
    cfg = await (await fetch("/api/config")).json();
  } catch {
    /* ignore */
  }
  if (!cfg.three_js?.ready && hud) {
    hud.textContent = `3D library not on server: ${cfg.three_js?.message || "unknown"}. Run: python3 scripts/fetch_threejs.py`;
  } else if (hud) {
    hud.textContent = "Loading Three.js…";
  }

  const threeOk = await loadThreeJs();
  if (!threeOk) {
    if (hud) {
      hud.textContent = "Three.js did not load. Check GET /assets/vendor/three.min.js in browser (should be ~650KB).";
    }
    return false;
  }

  if (!digitalTwin) ensureTwin();
  await waitForTwinReady(digitalTwin, 6000);
  if (digitalTwin?.ready && digitalTwin.mode === "3d") {
    if (lastTelemetryState) {
      digitalTwin.updateFromState(lastTelemetryState);
    } else {
      digitalTwin.updateFromState({
        joint_positions_rad: latestJoints,
        tcp_pose: [0.4, -0.25, 0.55, 0, 0, 0],
      });
    }
    return true;
  }
  return false;
}

async function boot() {
  await bootTwin();
  requestAnimationFrame(() => bootTwin());
  await init();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
