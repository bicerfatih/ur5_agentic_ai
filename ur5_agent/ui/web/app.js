const $ = (id) => document.getElementById(id);

const stateEl = $("telemetry");
const eventsEl = $("events");
const resultEl = $("tool-result");
const sitePill = $("site-pill");
const armPill = $("arm-pill");
const modePill = $("mode-pill");
const safetyPill = $("safety-pill");
const cameraPreview = $("camera-preview");
const cameraHint = $("camera-hint");
const detectedLabelsEl = $("detected-labels");
const goalStatusEl = $("goal-status");
const liveFeedToggle = $("live-feed-toggle");
const detectToggle = $("detect-toggle");

let latestJoints = [0, -1.57, 0, -1.57, 0, 0];
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

function renderState(payload) {
  const st = payload.state || {};
  if (Array.isArray(st.joint_positions_rad)) {
    latestJoints = st.joint_positions_rad;
  }
  sitePill.textContent = `site: ${payload.site || "-"}`;
  armPill.textContent = `arm: ${st.arm_model || "-"}`;
  modePill.textContent = `mode: ${st.robot_mode}`;
  safetyPill.textContent = `safety: ${st.safety_mode}`;
  modePill.style.borderColor = st.robot_mode === 7 ? "rgba(34,197,94,.6)" : "rgba(245,158,11,.7)";
  safetyPill.style.borderColor = st.safety_mode === 1 ? "rgba(34,197,94,.6)" : "rgba(245,158,11,.7)";
  stateEl.textContent = JSON.stringify(st, null, 2);
  renderTwin(latestJoints);
  if (payload.camera_path) {
    cameraHint.textContent = payload.camera_path;
  }
  if (payload.detection && !liveFeedTimer) {
    renderDetectionLabels(payload.detection);
  }
  const gs = payload.goal_status || {};
  if (gs.running) {
    goalStatusEl.textContent = `Running: ${gs.goal || ""}`;
  } else if (gs.error) {
    goalStatusEl.textContent = `Error: ${gs.error}`;
  } else if (gs.ended_at) {
    goalStatusEl.textContent = `Finished goal: ${gs.goal || ""}`;
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

async function init() {
  const cfg = await (await fetch("/api/config")).json();
  sitePill.textContent = `site: ${cfg.site}`;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    payload.site = cfg.site;
    renderState(payload);
    renderEvents(payload.events || []);
  };
  ws.onclose = () => { stateEl.textContent = "Telemetry socket disconnected."; };
}

// --- lightweight three.js twin ---
let scene, camera, renderer, armGroup, links = [];
function initTwin() {
  const canvas = $("twin-canvas");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x04070f);
  camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 20);
  camera.position.set(1.2, 1.2, 1.2);
  camera.lookAt(0, 0.2, 0);

  const light = new THREE.PointLight(0x66e5ff, 2.5, 10);
  light.position.set(1.8, 1.8, 1.8);
  scene.add(light, new THREE.AmbientLight(0x305070, 1.0));

  const grid = new THREE.GridHelper(2, 20, 0x1e3554, 0x132033);
  scene.add(grid);

  armGroup = new THREE.Group();
  scene.add(armGroup);
  const mat = new THREE.MeshStandardMaterial({ color: 0x22d3ee, metalness: 0.4, roughness: 0.3 });
  const lengths = [0.25, 0.30, 0.24, 0.18, 0.12, 0.08];
  let parent = armGroup;
  for (let i = 0; i < lengths.length; i++) {
    const joint = new THREE.Group();
    const link = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, lengths[i], 16), mat);
    link.position.y = lengths[i] / 2;
    joint.add(link);
    parent.add(joint);
    parent = joint;
    parent.position.y = lengths[i];
    links.push(joint);
  }
  animate();
}

function renderTwin(q) {
  if (!links.length) return;
  for (let i = 0; i < Math.min(6, q.length); i++) {
    const j = links[i];
    if (i === 0) j.rotation.y = q[i] || 0;
    else j.rotation.z = q[i] || 0;
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (armGroup) armGroup.rotation.y += 0.002;
  renderer?.render(scene, camera);
}

initTwin();
init();
