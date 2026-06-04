/**
 * UR5 digital twin — 3D (Three.js) with 2D schematic fallback (Jetson / offline).
 */
(function () {
  // UR5 geometry + FK from ros-industrial ur_kinematics (matches RTDE / controller)
  const UR5 = {
    d1: 0.089159,
    a2: -0.425,
    a3: -0.39225,
    d4: 0.10915,
    d5: 0.09465,
    d6: 0.0823,
  };
  const HOME_JOINTS = [0, -1.57, 0, -1.57, 0, 0];
  const LINK_2D = [0.425, 0.392, 0.2, 0.15, 0.12];
  // Robotiq 2F on UR tool0: +Z approach from flange, fingers open along ±Y (no extra mount twist).
  const GRIPPER_TCP_OFFSET_Z = 0.145;

  function mat4FromUr(c) {
    return new THREE.Matrix4().set(
      c[0], c[4], c[8], c[12],
      c[1], c[5], c[9], c[13],
      c[2], c[6], c[10], c[14],
      c[3], c[7], c[11], c[15],
    );
  }

  /** World transforms T01..T06 for RTDE joint vector q (rad). */
  function urForwardWorld(q) {
    const s1 = Math.sin(q[0]);
    const c1 = Math.cos(q[0]);
    const s2 = Math.sin(q[1]);
    const c2 = Math.cos(q[1]);
    const s3 = Math.sin(q[2]);
    const c3 = Math.cos(q[2]);
    const q23 = q[1] + q[2];
    const q234 = q[1] + q[2] + q[3];
    const s23 = Math.sin(q23);
    const c23 = Math.cos(q23);
    const s234 = Math.sin(q234);
    const c234 = Math.cos(q234);
    const s5 = Math.sin(q[4]);
    const c5 = Math.cos(q[4]);
    const s6 = Math.sin(q[5]);
    const c6 = Math.cos(q[5]);
    const { d1, a2, a3, d4, d5, d6 } = UR5;

    const T1 = mat4FromUr([
      c1, s1, 0, 0,
      0, 0, 1, 0,
      s1, -c1, 0, 0,
      0, 0, d1, 1,
    ]);

    const T2 = mat4FromUr([
      c1 * c2, c2 * s1, s2, 0,
      -c1 * s2, -s1 * s2, c2, 0,
      s1, -c1, 0, 0,
      a2 * c1 * c2, a2 * c2 * s1, d1 + a2 * s2, 1,
    ]);

    const T3 = mat4FromUr([
      c23 * c1, -s23 * c1, s1, 0,
      c23 * s1, -s23 * s1, -c1, 0,
      s23, c23, 0, 0,
      c1 * (a3 * c23 + a2 * c2), s1 * (a3 * c23 + a2 * c2), d1 + a3 * s23 + a2 * s2, 1,
    ]);

    const T4 = mat4FromUr([
      c234 * c1, s1, s234 * c1, 0,
      c234 * s1, -c1, s234 * s1, 0,
      s234, 0, -c234, 0,
      c1 * (a3 * c23 + a2 * c2) + d4 * s1,
      s1 * (a3 * c23 + a2 * c2) - d4 * c1,
      d1 + a3 * s23 + a2 * s2,
      1,
    ]);

    const T5 = mat4FromUr([
      s1 * s5 + c234 * c1 * c5, c234 * c5 * s1 - c1 * s5, s234 * c5, 0,
      -s234 * c1, -s234 * s1, c234, 0,
      c5 * s1 - c234 * c1 * s5, -c1 * c5 - c234 * s1 * s5, -s234 * s5, 0,
      c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
      s1 * (a3 * c23 + a2 * c2) - d4 * c1 + d5 * s234 * s1,
      d1 + a3 * s23 + a2 * s2 - d5 * c234,
      1,
    ]);

    const T6 = mat4FromUr([
      c6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * s6,
      -c6 * (c1 * s5 - c234 * c5 * s1) - s234 * s1 * s6,
      c234 * s6 + s234 * c5 * c6,
      0,
      -s6 * (s1 * s5 + c234 * c1 * c5) - s234 * c1 * c6,
      s6 * (c1 * s5 - c234 * c5 * s1) - s234 * c6 * s1,
      c234 * c6 - s234 * c5 * s6,
      0,
      c5 * s1 - c234 * c1 * s5,
      -c1 * c5 - c234 * s1 * s5,
      -s234 * s5,
      0,
      d6 * (c5 * s1 - c234 * c1 * s5) + c1 * (a3 * c23 + a2 * c2) + d4 * s1 + d5 * s234 * c1,
      s1 * (a3 * c23 + a2 * c2) - d4 * c1 - d6 * (c1 * c5 + c234 * s1 * s5) + d5 * s234 * s1,
      d1 + a3 * s23 + a2 * s2 - d5 * c234 - d6 * s234 * s5,
      1,
    ]);

    return [T1, T2, T3, T4, T5, T6];
  }

  function urTcpToPose(tcp) {
    if (!tcp || tcp.length < 6 || typeof THREE === "undefined") return null;
    const p = new THREE.Vector3(Number(tcp[0]), Number(tcp[1]), Number(tcp[2]));
    const rx = Number(tcp[3]);
    const ry = Number(tcp[4]);
    const rz = Number(tcp[5]);
    const angle = Math.sqrt(rx * rx + ry * ry + rz * rz);
    const q = new THREE.Quaternion();
    if (angle > 1e-8) {
      q.setFromAxisAngle(new THREE.Vector3(rx, ry, rz).normalize(), angle);
    }
    return { p, q };
  }

  class UR5DigitalTwin {
    constructor(canvas, canvas2d, hudEl, options = {}) {
      this.ready = false;
      this.mode = "none";
      this.prefer3d = options.prefer3d !== false;
      this.canvas = canvas;
      this.canvas2d = canvas2d;
      this.hudEl = hudEl;
      this.jointGroups = [];
      this.targetJoints = HOME_JOINTS.slice();
      this.displayJoints = HOME_JOINTS.slice();
      this._userOrbit = false;
      this._gripOpen = true;
      this._fail3d = null;

      if (hudEl) hudEl.textContent = "Starting 3D twin…";
      requestAnimationFrame(() => requestAnimationFrame(() => this._start()));
    }

    _start() {
      if (typeof THREE !== "undefined") {
        try {
          this._initScene3d();
          this._buildArm3d();
          this._initCameraControls();
          this.mode = "3d";
          this.ready = true;
        } catch (err) {
          this._fail3d = err && err.message ? err.message : String(err);
          console.warn("[twin] 3D init failed:", err);
        }
      } else {
        this._fail3d = "Three.js not loaded from /assets/vendor/three.min.js";
      }

      if (!this.ready && !this.prefer3d && this.canvas2d) {
        try {
          this._init2d();
          this.mode = "2d";
          this.ready = true;
        } catch (err) {
          console.warn("[twin] 2D init failed:", err);
        }
      }

      if (!this.ready) {
        if (this.hudEl) {
          const hint = this._fail3d || "unknown error";
          this.hudEl.textContent = `3D twin failed: ${hint}. Restart console (downloads Three.js) or: python3 scripts/fetch_threejs.py`;
        }
        return;
      }

      this._onResize();
      window.addEventListener("resize", () => this._onResize());
      const el = this.mode === "2d" ? this.canvas2d : this.canvas;
      if (el && window.ResizeObserver) {
        new ResizeObserver(() => this._onResize()).observe(el);
      }

      if (this.mode === "3d") {
        this.setJoints(this.targetJoints);
        this._frameCameraOnArm(true);
      }

      if (this.hudEl) {
        this.hudEl.textContent = this.mode === "2d"
          ? `2D fallback (${this._fail3d}) — waiting for telemetry…`
          : "3D twin ready — waiting for robot telemetry…";
      }
      this._animate();
    }

    _createWebGLRenderer() {
      const rect = this.canvas.getBoundingClientRect();
      if (rect.width < 8 || rect.height < 8) {
        throw new Error(`Canvas not laid out yet (${Math.round(rect.width)}×${Math.round(rect.height)})`);
      }
      const attempts = [
        { antialias: true, failIfMajorPerformanceCaveat: false, powerPreference: "high-performance" },
        { antialias: false, failIfMajorPerformanceCaveat: false },
        { antialias: false, failIfMajorPerformanceCaveat: true, powerPreference: "default" },
      ];
      let lastErr = null;
      for (const opts of attempts) {
        let renderer = null;
        try {
          renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            alpha: false,
            ...opts,
          });
          if (!renderer.getContext()) {
            renderer.dispose();
            throw new Error("WebGL context is null");
          }
          return renderer;
        } catch (e) {
          lastErr = e;
          if (renderer) renderer.dispose();
        }
      }
      throw new Error(lastErr ? lastErr.message || String(lastErr) : "WebGL unavailable");
    }

    _initScene3d() {
      this.renderer = this._createWebGLRenderer();
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.setClearColor(0x0a1220, 1);
      if (THREE.SRGBColorSpace && this.renderer.outputColorSpace !== undefined) {
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      }

      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0a1220);
      this.camera = new THREE.PerspectiveCamera(50, 1, 0.02, 20);
      this.camera.up.set(0, 0, 1);

      this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
      this.scene.add(new THREE.HemisphereLight(0xb8e8ff, 0x1a2438, 0.6));
      const key = new THREE.DirectionalLight(0xffffff, 1.1);
      key.position.set(1.5, -1.2, 2.5);
      this.scene.add(key);

      const grid = new THREE.GridHelper(2.4, 24, 0x3d6a8c, 0x1e3348);
      grid.rotation.x = Math.PI / 2;
      this.scene.add(grid);
      this.arm = new THREE.Group();
      this.scene.add(this.arm);

      this.flangeFrame = new THREE.Group();
      this.scene.add(this.flangeFrame);

      this.toolFrame = new THREE.Group();
      this.scene.add(this.toolFrame);

      this.tcpMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.006, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0x4ade80 }),
      );
      this.toolFrame.add(this.tcpMarker);

      this.canvas.classList.remove("hidden");
      if (this.canvas2d) this.canvas2d.classList.add("hidden");
    }

    _init2d() {
      this.ctx2d = this.canvas2d.getContext("2d");
      if (!this.ctx2d) throw new Error("2D canvas context unavailable");
      this.canvas2d.classList.remove("hidden");
      this.canvas.classList.add("hidden");
    }

    _mat(kind) {
      const table = {
        base: { color: 0x3f4654, metalness: 0.55, roughness: 0.38 },
        accent: { color: 0x0078a8, metalness: 0.4, roughness: 0.35 },
        arm: { color: 0xc4c9d2, metalness: 0.62, roughness: 0.32 },
        joint: { color: 0x6d7585, metalness: 0.58, roughness: 0.4 },
        wrist: { color: 0x9aa3b2, metalness: 0.55, roughness: 0.34 },
        grip: { color: 0x2a2f38, metalness: 0.5, roughness: 0.45 },
        finger: { color: 0x4ade80, metalness: 0.35, roughness: 0.42 },
      };
      const p = table[kind] || table.arm;
      return new THREE.MeshStandardMaterial(p);
    }

    _mesh(geo, mat) {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = false;
      m.receiveShadow = false;
      return m;
    }

    /** Place unit cylinder (Y-up) between two FK frame origins */
    _positionLink(mesh, p0, p1, radius) {
      if (!this._segDir) {
        this._segDir = new THREE.Vector3();
        this._segMid = new THREE.Vector3();
        this._segUp = new THREE.Vector3(0, 1, 0);
        this._segQuat = new THREE.Quaternion();
      }
      this._segDir.subVectors(p1, p0);
      const len = this._segDir.length();
      if (len < 1e-4) {
        mesh.visible = false;
        return;
      }
      mesh.visible = true;
      this._segDir.normalize();
      this._segMid.addVectors(p0, p1).multiplyScalar(0.5);
      mesh.position.copy(this._segMid);
      this._segQuat.setFromUnitVectors(this._segUp, this._segDir);
      mesh.quaternion.copy(this._segQuat);
      mesh.scale.set(radius, len, radius);
    }

    _buildGripper(parent) {
      this.gripperGroup = new THREE.Group();
      parent.add(this.gripperGroup);

      const plate = this._mesh(
        new THREE.CylinderGeometry(0.038, 0.042, 0.014, 24),
        this._mat("wrist"),
      );
      plate.position.z = 0.007;
      this.gripperGroup.add(plate);

      const body = this._mesh(
        new THREE.BoxGeometry(0.086, 0.058, 0.052),
        this._mat("grip"),
      );
      body.position.z = 0.048;
      this.gripperGroup.add(body);

      const fingerMat = this._mat("finger");
      this.fingerL = this._mesh(new THREE.BoxGeometry(0.018, 0.011, 0.052), fingerMat);
      this.fingerL.position.set(0.026, 0, 0.108);
      this.fingerR = this._mesh(new THREE.BoxGeometry(0.018, 0.011, 0.052), fingerMat);
      this.fingerR.position.set(-0.026, 0, 0.108);
      this.gripperGroup.add(this.fingerL);
      this.gripperGroup.add(this.fingerR);
      this.gripperMesh = this.gripperGroup;
    }

    _setGripperOpen(open) {
      if (!this.fingerL || !this.fingerR) return;
      const gap = open ? 0.024 : 0.007;
      this.fingerL.position.x = gap;
      this.fingerR.position.x = -gap;
      const col = open ? 0x4ade80 : 0xf87171;
      this.fingerL.material.color.setHex(col);
      this.fingerR.material.color.setHex(col);
    }

    _buildBaseMeshes() {
      const baseMat = this._mat("base");
      const accentMat = this._mat("accent");
      const pedestal = this._mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.028, 32), baseMat);
      pedestal.position.z = 0.014;
      this.arm.add(pedestal);
      const baseTower = this._mesh(new THREE.CylinderGeometry(0.078, 0.088, 0.058, 32), baseMat);
      baseTower.position.z = 0.058;
      this.arm.add(baseTower);
      const ring = this._mesh(new THREE.TorusGeometry(0.092, 0.006, 12, 40), accentMat);
      ring.rotation.x = Math.PI / 2;
      ring.position.z = 0.082;
      this.arm.add(ring);
    }

    _buildArm3d() {
      this.linkSegs = [];
      this.jointMarks = [];
      this._buildBaseMeshes();

      const linkRadii = [0.052, 0.048, 0.044, 0.038, 0.034];
      for (let i = 0; i < 5; i++) {
        const mat = i < 2 ? this._mat("arm") : this._mat("wrist");
        const m = this._mesh(new THREE.CylinderGeometry(1, 1, 1, 20), mat);
        m.scale.set(linkRadii[i], 0.001, linkRadii[i]);
        this.arm.add(m);
        this.linkSegs.push(m);
      }
      for (let i = 0; i < 6; i++) {
        const r = i < 2 ? 0.038 : 0.03;
        const s = this._mesh(new THREE.SphereGeometry(r, 16, 14), this._mat("joint"));
        this.arm.add(s);
        this.jointMarks.push(s);
      }

      this._buildGripper(this.flangeFrame);
      this._setGripperOpen(false);
      this.hasRealTcp = false;
      this._applyFk(HOME_JOINTS);
    }

    _syncToolFrame() {
      if (!this._fkOrigins) return;

      if (this.flangeFrame) {
        this.flangeFrame.position.copy(this._fkOrigins[5]);
        this.flangeFrame.quaternion.copy(this._fkQuats[5]);
      }

      if (!this.toolFrame) return;
      if (this.hasRealTcp && this._rtdeTcp) {
        this.toolFrame.position.copy(this._rtdeTcp.p);
        this.toolFrame.quaternion.copy(this._rtdeTcp.q);
      } else if (this._fkTcp) {
        const z = new THREE.Vector3(0, 0, 1).applyQuaternion(this._fkTcp.q);
        this.toolFrame.position.copy(this._fkTcp.p).addScaledVector(z, GRIPPER_TCP_OFFSET_Z);
        this.toolFrame.quaternion.copy(this._fkTcp.q);
      }

    }

    _applyFk(q) {
      if (!this.linkSegs || !this.linkSegs.length) return;
      const world = urForwardWorld(q);
      if (!this._fkOrigins) {
        this._fkOrigins = Array.from({ length: 6 }, () => new THREE.Vector3());
        this._fkQuats = Array.from({ length: 6 }, () => new THREE.Quaternion());
        this._fkScale = new THREE.Vector3();
      }
      for (let i = 0; i < 6; i++) {
        world[i].decompose(this._fkOrigins[i], this._fkQuats[i], this._fkScale);
        if (this.jointMarks[i]) {
          this.jointMarks[i].position.copy(this._fkOrigins[i]);
          this.jointMarks[i].quaternion.copy(this._fkQuats[i]);
        }
      }
      const radii = [0.052, 0.048, 0.044, 0.038, 0.034];
      for (let i = 0; i < 5; i++) {
        this._positionLink(
          this.linkSegs[i],
          this._fkOrigins[i],
          this._fkOrigins[i + 1],
          radii[i],
        );
      }
      this._fkTcp = {
        p: this._fkOrigins[5].clone(),
        q: this._fkQuats[5].clone(),
      };
      this._syncToolFrame();
    }

    setJoints(q) {
      if (!q || q.length < 6) return;
      this.targetJoints = q.slice(0, 6).map((v) => Number(v));
      if (this.targetJoints.some((v) => !Number.isFinite(v))) return;
      if (this.mode === "3d") this._applyFk(this.targetJoints);
    }

    _frameCameraOnArm(force) {
      if (this.mode !== "3d" || !this.arm || (this._userOrbit && !force)) return;
      const box = new THREE.Box3().setFromObject(this.arm);
      if (this.flangeFrame) box.expandByObject(this.flangeFrame);
      if (this.toolFrame) box.expandByObject(this.toolFrame);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z, 0.35);
      this._orbit.target.copy(center);
      this._orbit.radius = Math.max(0.85, Math.min(2.4, span * 1.8));
      this._orbit.target.lerp(new THREE.Vector3(0, 0, 0.12), 0.25);
    }

    setRealTcp(tcpPose) {
      if (this.mode !== "3d" || !this.toolFrame) return;
      const pose = urTcpToPose(tcpPose);
      if (!pose) return;
      this._rtdeTcp = pose;
      this.hasRealTcp = true;
      this._syncToolFrame();
    }

    _draw2d() {
      const c = this.canvas2d;
      const ctx = this.ctx2d;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = c.getBoundingClientRect();
      const w = Math.max(rect.width, 320);
      const h = Math.max(rect.height, 280);
      c.width = Math.floor(w * dpr);
      c.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.fillStyle = "#0a1220";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#1e3348";
      ctx.lineWidth = 1;
      for (let i = 0; i < w; i += 32) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, h);
        ctx.stroke();
      }

      const q = this.displayJoints;
      const scale = h * 0.42;
      const ox = w * 0.38 + Math.sin(q[0]) * 28;
      const oy = h * 0.86;

      ctx.fillStyle = "#64748b";
      ctx.fillRect(ox - 22, oy - 6, 44, 12);

      let ang = q[1] - Math.PI / 2;
      let x = ox;
      let y = oy - 8;
      ctx.strokeStyle = "#7fa8cc";
      ctx.lineWidth = 12;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(ox, oy - 4);
      for (let i = 0; i < LINK_2D.length; i++) {
        if (i > 0) ang += q[i + 1];
        const len = LINK_2D[i] * scale;
        x += Math.cos(ang) * len;
        y += Math.sin(ang) * len;
        ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = this._gripOpen ? "#4ade80" : "#f87171";
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("2D schematic (side elevation)", 10, 18);
    }

    updateFromState(state) {
      if (!state) return;
      if (state.error && !state.joint_positions_rad && !state.joint_positions_deg) {
        if (this.hudEl) this.hudEl.textContent = `No live telemetry: ${state.error}`;
        return;
      }
      let joints = state.joint_positions_rad;
      if ((!joints || joints.length < 6) && Array.isArray(state.joint_positions_deg)) {
        joints = state.joint_positions_deg.map((d) => (Number(d) * Math.PI) / 180);
      }
      if (Array.isArray(joints) && joints.length >= 6) {
        this.setJoints(joints);
        if (this.mode === "3d" && !this._userOrbit) this._frameCameraOnArm(false);
      }
      if (this.mode === "3d" && Array.isArray(state.tcp_pose) && state.tcp_pose.length >= 6) {
        this.setRealTcp(state.tcp_pose);
      }
      const grip = state.gripper || {};
      this._gripOpen = grip.position === undefined || grip.position < 128;
      this._setGripperOpen(this._gripOpen);
      this._updateHud(state);
    }

    _updateHud(state) {
      if (!this.hudEl) return;
      const q = state.joint_positions_deg || [];
      const tcp = state.tcp_pose || [];
      const j = q.length >= 6
        ? `J: ${q.map((v) => Number(v).toFixed(0)).join("° ")}°`
        : "J: —";
      const t = tcp.length >= 3
        ? `TCP: ${tcp.slice(0, 3).map((v) => Number(v).toFixed(3)).join(", ")} m`
        : "TCP: —";
      const mode = state.robot_mode === 7 ? "RUN" : `M${state.robot_mode ?? "?"}`;
      const view = this.mode === "2d" ? " · 2D" : " · 3D";
      const stale = state.telemetry_stale ? " (cached)" : "";
      const live = state.simulated ? " · sim" : " · live";
      let sync = "";
      if (this._fkTcp && this._rtdeTcp) {
        const err = this._fkTcp.p.distanceTo(this._rtdeTcp.p);
        sync = err < 0.02 ? " · sync OK" : ` · FKΔ ${(err * 1000).toFixed(0)}mm`;
      }
      this.hudEl.textContent = `${j}  |  ${t}  |  ${mode}${live}${view}${sync}${stale}`;
    }

    _initCameraControls() {
      this._orbit = {
        phi: 0.95,
        theta: -0.85,
        radius: 1.5,
        target: new THREE.Vector3(0.25, 0, 0.45),
      };
      this._drag = false;
      this._last = { x: 0, y: 0 };
      this.canvas.addEventListener("pointerdown", (e) => {
        this._drag = true;
        this._userOrbit = true;
        this._last = { x: e.clientX, y: e.clientY };
        this.canvas.setPointerCapture(e.pointerId);
      });
      this.canvas.addEventListener("pointerup", () => { this._drag = false; });
      this.canvas.addEventListener("pointermove", (e) => {
        if (!this._drag) return;
        const dx = e.clientX - this._last.x;
        const dy = e.clientY - this._last.y;
        this._last = { x: e.clientX, y: e.clientY };
        this._orbit.theta -= dx * 0.006;
        this._orbit.phi = Math.max(0.2, Math.min(1.65, this._orbit.phi + dy * 0.006));
      });
      this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        this._userOrbit = true;
        this._orbit.radius = Math.max(0.45, Math.min(3.5, this._orbit.radius + e.deltaY * 0.0015));
      }, { passive: false });
      this.canvas.addEventListener("dblclick", () => {
        this._userOrbit = false;
        this._frameCameraOnArm(true);
      });
    }

    _updateCamera() {
      const { phi, theta, radius, target } = this._orbit;
      const x = target.x + radius * Math.sin(phi) * Math.cos(theta);
      const y = target.y + radius * Math.sin(phi) * Math.sin(theta);
      const z = target.z + radius * Math.cos(phi);
      this.camera.position.set(x, y, z);
      this.camera.lookAt(target);
    }

    _onResize() {
      if (this.mode === "3d" && this.renderer) {
        const rect = this.canvas.getBoundingClientRect();
        const w = Math.max(rect.width || 320, 320);
        const h = Math.max(rect.height || 280, 280);
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
      }
    }

    _animate() {
      if (!this.ready) return;
      requestAnimationFrame(() => this._animate());
      for (let i = 0; i < 6; i++) {
        this.displayJoints[i] += (this.targetJoints[i] - this.displayJoints[i]) * 0.22;
      }
      if (this.mode === "3d" && this.linkSegs && this.linkSegs.length) {
        this._applyFk(this.displayJoints);
      }
      if (this.mode === "3d" && this.renderer) {
        this._updateCamera();
        this.renderer.render(this.scene, this.camera);
      } else if (this.mode === "2d") {
        this._draw2d();
      }
    }
  }

  window.UR5DigitalTwin = UR5DigitalTwin;
})();
