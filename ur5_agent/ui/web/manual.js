/**
 * Manual control — Cartesian TCP jog (moveL) + per-joint jog + gripper.
 */
(function () {
  const JOINT_LABELS = ["J1 Base", "J2 Shoulder", "J3 Elbow", "J4 Wrist 1", "J5 Wrist 2", "J6 Wrist 3"];
  const COOLDOWN_MS = 450;

  const CARTESIAN_TOOLS = {
    ArrowLeft: "move_left",
    ArrowRight: "move_right",
    ArrowUp: "move_forward",
    ArrowDown: "move_backward",
    PageUp: "move_up",
    PageDown: "move_down",
  };

  const state = {
    heldKeys: new Set(),
    lastAxis: null,
    lastAxisAt: 0,
    lastCmdAt: 0,
    busy: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(msg, level) {
    const el = $("manual-status");
    if (!el) return;
    el.textContent = msg;
    el.dataset.level = level || "info";
  }

  function isTyping() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
  }

  function stepDeg() {
    const v = parseFloat($("jog-step-deg")?.value || "5");
    return Number.isFinite(v) && v > 0 ? Math.min(v, 15) : 5;
  }

  function stepM() {
    const v = parseFloat($("jog-step-cm")?.value || "2");
    const cm = Number.isFinite(v) && v > 0 ? Math.min(v, 25) : 2;
    return cm / 100;
  }

  async function runTool(name, inputs) {
    if (typeof window.postTool !== "function") {
      setStatus("Console not ready.", "error");
      return;
    }
    const now = Date.now();
    if (state.busy || now - state.lastCmdAt < COOLDOWN_MS) return;
    state.busy = true;
    state.lastCmdAt = now;
    try {
      setStatus(`Running ${name}…`, "active");
      await window.postTool(name, inputs);
      setStatus(`Done: ${name}`, "ok");
    } catch (e) {
      setStatus(`Error: ${e.message}`, "error");
    } finally {
      state.busy = false;
    }
  }

  function jogJoint(joint, direction) {
    const delta = stepDeg() * direction;
    runTool("jog_joint", { joint, delta_deg: delta, speed: 0.25, acceleration: 0.25 });
  }

  function jogCartesian(toolName) {
    runTool(toolName, { distance_m: stepM() });
  }

  function gripper(open) {
    runTool(open ? "open_gripper" : "close_gripper", {});
  }

  function activeAxisKey() {
    for (let n = 7; n >= 1; n--) {
      if (state.heldKeys.has(String(n))) return String(n);
    }
    if (state.lastAxis && Date.now() - state.lastAxisAt < 2500) {
      return state.lastAxis;
    }
    return null;
  }

  function onArrowKey(key) {
    const axis = activeAxisKey();
    if (axis && (key === "ArrowLeft" || key === "ArrowRight")) {
      const direction = key === "ArrowRight" ? 1 : -1;
      if (axis === "7") {
        gripper(direction > 0);
        return;
      }
      jogJoint(parseInt(axis, 10), direction);
      return;
    }
    const tool = CARTESIAN_TOOLS[key];
    if (tool) jogCartesian(tool);
  }

  function buildUi() {
    const grid = $("manual-joints");
    if (!grid) return;
    grid.innerHTML = "";
    for (let j = 1; j <= 6; j++) {
      const row = document.createElement("div");
      row.className = "manual-row";
      row.innerHTML = `
        <span class="manual-axis-label"><kbd>${j}</kbd> ${JOINT_LABELS[j - 1]}</span>
        <span class="manual-joint-val" id="manual-j${j}-val">—°</span>
        <button type="button" class="manual-btn manual-left" data-joint="${j}" data-dir="-1" title="Joint ${j} −°">−</button>
        <button type="button" class="manual-btn manual-right" data-joint="${j}" data-dir="1" title="Joint ${j} +°">+</button>
      `;
      grid.appendChild(row);
    }

    grid.querySelectorAll(".manual-btn[data-joint]").forEach((btn) => {
      btn.addEventListener("click", () => {
        jogJoint(parseInt(btn.dataset.joint, 10), parseInt(btn.dataset.dir, 10));
      });
    });

    const cart = $("manual-cartesian");
    if (cart) {
      cart.innerHTML = `
        <div class="manual-cartesian-pad">
          <button type="button" class="manual-btn manual-cart" data-tool="move_forward" title="Forward">↑ Fwd</button>
          <div class="manual-cart-row">
            <button type="button" class="manual-btn manual-cart" data-tool="move_left" title="Left">← Left</button>
            <button type="button" class="manual-btn manual-cart" data-tool="move_right" title="Right">Right →</button>
          </div>
          <button type="button" class="manual-btn manual-cart" data-tool="move_backward" title="Backward">↓ Back</button>
          <div class="manual-cart-row manual-cart-z">
            <button type="button" class="manual-btn manual-cart" data-tool="move_up" title="Up">Up</button>
            <button type="button" class="manual-btn manual-cart" data-tool="move_down" title="Down">Down</button>
          </div>
        </div>
      `;
      cart.querySelectorAll("[data-tool]").forEach((btn) => {
        btn.addEventListener("click", () => jogCartesian(btn.dataset.tool));
      });
    }

    const grip = $("manual-gripper-row");
    if (grip) {
      grip.innerHTML = `
        <div class="manual-row manual-gripper">
          <span class="manual-axis-label"><kbd>7</kbd> Gripper</span>
          <span class="manual-joint-val" id="manual-gripper-val">—</span>
          <button type="button" class="manual-btn manual-left" id="manual-grip-close" title="Close gripper">Close</button>
          <button type="button" class="manual-btn manual-right" id="manual-grip-open" title="Open gripper">Open</button>
        </div>
      `;
      $("manual-grip-close")?.addEventListener("click", () => gripper(false));
      $("manual-grip-open")?.addEventListener("click", () => gripper(true));
    }

    $("manual-stop")?.addEventListener("click", () => runTool("stop_robot", {}));
  }

  function updateJointDisplay(robotState) {
    const deg = robotState?.joint_positions_deg;
    if (!Array.isArray(deg)) return;
    for (let j = 1; j <= 6; j++) {
      const el = $(`manual-j${j}-val`);
      if (el && deg[j - 1] != null) {
        el.textContent = `${Number(deg[j - 1]).toFixed(1)}°`;
      }
    }
    const g = robotState?.gripper;
    const gel = $("manual-gripper-val");
    if (gel && g) {
      gel.textContent = g.command_state || g.last_command || "—";
    }
  }

  window.onManualTelemetry = function (payload) {
    updateJointDisplay(payload?.state);
  };

  function wireKeyboard() {
    document.addEventListener("keydown", (e) => {
      if (isTyping()) return;

      const key = e.key;
      if (key >= "1" && key <= "7") {
        if (!e.repeat) {
          state.heldKeys.add(key);
          state.lastAxis = key;
          state.lastAxisAt = Date.now();
          if (key === "7") {
            setStatus("Axis 7 selected — ← close / → open gripper", "active");
          } else {
            setStatus(`Axis ${key} selected — ←/→ jog joint, or arrows alone for TCP`, "active");
          }
        }
        return;
      }

      if (!CARTESIAN_TOOLS[key]) return;
      if (e.repeat) return;
      e.preventDefault();
      onArrowKey(key);
    });

    document.addEventListener("keyup", (e) => {
      if (e.key >= "1" && e.key <= "7") {
        state.heldKeys.delete(e.key);
      }
    });
  }

  window.initManualControl = function () {
    buildUi();
    wireKeyboard();
    setStatus("Ready — arrows alone: TCP · 1–6 + ←/→: joint · 7 + ←/→: gripper", "ok");
  };
})();
