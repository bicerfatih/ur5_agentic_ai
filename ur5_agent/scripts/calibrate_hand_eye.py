#!/usr/bin/env python3
"""Hand-eye calibration for a camera mounted ON the UR5 (eye-in-hand).

Produces data/calibration/hand_eye.json used by detect_objects / camera_reach.

Your setup
----------
Camera is on the robot → mount = eye_in_hand.
The ArUco marker must stay FIXED on the TABLE (not on the gripper).
We solve T_tool_camera: camera pose relative to the TCP.

Why the previous run failed
---------------------------
With the marker on the gripper AND the camera on the arm, they move together.
Pixel/depth barely change → calibration is meaningless. Marker belongs on the table.

Session
-------
0. Print marker:  python3 scripts/calibrate_hand_eye.py --make-marker
   Tape it flat on the TABLE where the wrist camera can see it.
1. STOP the ops console (camera exclusive).
2. Pendant: Remote ON, External Control PLAYING.
3. Auto:          python3 scripts/calibrate_hand_eye.py --auto
   - First: move TCP to TOUCH the marker center, type TEACH
   - Then type YES — robot moves while looking at the table marker
4. Verify:        python3 scripts/calibrate_hand_eye.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import HAND_EYE_CALIB_PATH, ROBOT_HOST
from camera.geometry import pixel_depth_to_camera_xyz, sample_depth_m

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CALIB_DIR = os.path.join(REPO_ROOT, "data", "calibration")
SAMPLES_PATH = os.path.join(CALIB_DIR, "hand_eye_samples.json")
DEFAULT_OUT = os.path.join(CALIB_DIR, "hand_eye.json")
MARKER_PNG = os.path.join(CALIB_DIR, "aruco_marker_4x4_id0.png")


# ── ArUco ────────────────────────────────────────────────


def _aruco_detector():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def detect_marker_center(color_bgr: np.ndarray, marker_id: int | None) -> tuple[float, float] | None:
    import cv2

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector()(gray)
    if ids is None or len(corners) == 0:
        return None
    for quad, mid in zip(corners, ids.flatten()):
        if marker_id is None or int(mid) == int(marker_id):
            c = quad.reshape(4, 2).mean(axis=0)
            return float(c[0]), float(c[1])
    return None


def make_marker(path: str = MARKER_PNG, marker_id: int = 0, size_px: int = 700) -> str:
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    except AttributeError:
        img = cv2.aruco.drawMarker(dictionary, marker_id, size_px)
    bordered = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bordered)
    return path


# ── Math ─────────────────────────────────────────────────


def connect_rtde_receive(host: str):
    from rtde_receive import RTDEReceiveInterface

    return RTDEReceiveInterface(host)


def tcp_pose_to_matrix(tcp_pose: list[float]) -> np.ndarray:
    import cv2

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(tcp_pose[:3], dtype=np.float64)
    rvec = np.asarray(tcp_pose[3:6], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    return T


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Rigid T minimizing || T @ src - dst ||."""
    assert src.shape == dst.shape and src.shape[0] >= 3
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cd - R @ cs
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    pred = (R @ src.T).T + t
    residuals = np.linalg.norm(pred - dst, axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    return T, rms, residuals


def spread_report(pts: np.ndarray) -> str:
    span = pts.max(axis=0) - pts.min(axis=0)
    return f"spread X={span[0]*100:.0f}cm Y={span[1]*100:.0f}cm Z={span[2]*100:.0f}cm"


# ── Capture ──────────────────────────────────────────────


def capture_marker_cam(cam, marker_id) -> tuple[dict | None, str | None]:
    """Return marker observation in camera frame (no TCP)."""
    last_err = "camera capture failed"
    for _ in range(3):
        try:
            cam.warmup(frames=3, timeout_ms=1500)
            rgbd = cam.capture_rgbd(timeout_ms=8000)
            break
        except Exception as e:
            last_err = str(e)
            try:
                cam.reconnect()
            except Exception as re:
                return None, f"{e}; reconnect failed: {re}"
    else:
        return None, f"camera timeout: {last_err}"

    color = rgbd["color"]
    depth = rgbd.get("depth")
    if depth is None:
        return None, "no depth — set CAMERA_DEPTH_ENABLED=true"
    center = detect_marker_center(color, marker_id)
    if center is None:
        return None, "marker not detected — put it on the TABLE in the camera view"
    u, v = center
    depth_m = sample_depth_m(
        depth, int(round(u)), int(round(v)), depth_scale=float(rgbd.get("depth_scale", 0.001))
    )
    if depth_m is None or depth_m <= 0.05:
        return None, f"bad depth at marker ({depth_m})"
    p_cam = pixel_depth_to_camera_xyz(u, v, depth_m, rgbd.get("intrinsics") or {})
    if p_cam is None:
        return None, "no intrinsics"
    return {
        "pixel": [round(u, 1), round(v, 1)],
        "depth_m": round(float(depth_m), 4),
        "p_cam": [round(float(x), 5) for x in p_cam],
        "intrinsics": rgbd.get("intrinsics") or {},
    }, None


def solve_eye_in_hand(
    samples: list[dict],
    marker_base: np.ndarray,
    intrinsics: dict,
    out_path: str,
) -> bool:
    """
    For each sample: p_marker_in_tool = inv(T_base_tool) @ marker_base
    Fit T_tool_camera so T @ p_cam ≈ p_marker_in_tool.
    """
    if len(samples) < 4:
        print(f"\nNeed ≥4 samples, have {len(samples)}. Not solving.")
        return False

    p_cam = []
    p_tool = []
    for s in samples:
        T_bt = tcp_pose_to_matrix(s["tcp_pose"])
        T_tb = np.linalg.inv(T_bt)
        p_h = np.ones(4)
        p_h[:3] = marker_base
        p_t = (T_tb @ p_h)[:3]
        p_cam.append(s["p_cam"])
        p_tool.append(p_t)

    p_cam_a = np.asarray(p_cam, dtype=np.float64)
    p_tool_a = np.asarray(p_tool, dtype=np.float64)
    T, rms, residuals = fit_rigid_transform(p_cam_a, p_tool_a)

    print(f"\n── Eye-in-hand solve: {len(samples)} samples, {spread_report(p_tool_a)}")
    for i, r in enumerate(residuals):
        flag = "  <-- outlier?" if r > max(0.02, 3 * rms) else ""
        print(f"  sample {i+1:2d}: residual {r*1000:6.1f} mm{flag}")
    print(f"  RMS error: {rms*1000:.1f} mm")
    if rms > 0.02:
        print("  WARNING: RMS > 20 mm — retake with marker clearly on the table.")

    # Pixel spread sanity: if camera moved over a fixed marker, pixels should vary.
    pixels = np.array([s["pixel"] for s in samples], dtype=np.float64)
    pix_span = float(np.linalg.norm(pixels.max(axis=0) - pixels.min(axis=0)))
    print(f"  pixel span: {pix_span:.1f} px (want ≫ 20 — proves camera moved over fixed marker)")
    if pix_span < 20:
        print(
            "  ERROR: pixels barely changed. Marker is probably still on the gripper,\n"
            "  or the camera is not looking at a fixed table marker. NOT writing calib."
        )
        return False

    payload = {
        "mount": "eye_in_hand",
        "T_tool_camera": [[round(float(v), 6) for v in row] for row in T],
        "marker_base_m": [round(float(v), 5) for v in marker_base],
        "intrinsics": {k: round(float(v), 2) for k, v in intrinsics.items()},
        "meta": {
            "method": "eye_in_hand_aruco_kabsch",
            "n_samples": len(samples),
            "rms_m": round(rms, 5),
            "pixel_span_px": round(pix_span, 1),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    return True


# ── Modes ────────────────────────────────────────────────


def _pose_grid(step: float, n: int) -> list[tuple[float, float, float]]:
    """Relative TCP deltas from current hover; denser grid → more samples in FOV."""
    s = step
    hs = step * 0.5
    qs = step * 0.35
    grid = [
        (0.0, 0.0, 0.0),
        (s, 0.0, 0.0),
        (-s, 0.0, 0.0),
        (0.0, s, 0.0),
        (0.0, -s, 0.0),
        (hs, hs, 0.0),
        (hs, -hs, 0.0),
        (-hs, hs, 0.0),
        (-hs, -hs, 0.0),
        (qs, 0.0, s * 0.35),
        (-qs, 0.0, s * 0.35),
        (0.0, qs, s * 0.35),
        (0.0, -qs, s * 0.35),
        (hs, 0.0, -s * 0.25),
        (0.0, hs, -s * 0.25),
        (-hs, 0.0, s * 0.2),
        (0.0, -hs, s * 0.2),
        (qs, qs, s * 0.25),
        (qs, -qs, s * 0.25),
        (-qs, qs, 0.0),
        (-qs, -qs, 0.0),
        (s * 0.75, 0.0, 0.0),
        (0.0, s * 0.75, 0.0),
        (-s * 0.75, 0.0, 0.0),
        (0.0, -s * 0.75, 0.0),
    ]
    # Absolute-style extras as small local orbits if more poses requested.
    while len(grid) < n:
        i = len(grid)
        ang = (i * 0.7) % 6.2832
        r = qs + (i % 3) * 0.01
        grid.append((r * float(np.cos(ang)), r * float(np.sin(ang)), (i % 2) * 0.02))
    return grid[:n]


def run_auto(args):
    """Eye-in-hand: teach table marker, then robot moves while looking at it."""
    from camera.realsense_camera import RealSenseCamera
    from robot.ur5_driver import UR5Driver

    n = max(6, int(args.poses))
    step = float(args.step_m)

    print("=" * 60)
    print("EYE-IN-HAND calibration (camera mounted on the UR5)")
    print("=" * 60)
    print("1) Put the ArUco marker FLAT ON THE TABLE (not on the gripper).")
    print("2) Pendant: Remote ON · External Control PLAYING · e-stop ready.")
    if args.reuse_marker:
        print("3) Reusing previous marker_base (no TEACH). Keep marker in the same place.")
    else:
        print("3) You will TEACH the marker by touching it with the TCP tip.")
    print()

    cam = RealSenseCamera(depth_enabled=True)
    cam.connect()
    robot = UR5Driver(host=args.host)
    robot.connect()
    if robot.rtde_c is None:
        print("ERROR: RTDE control missing — play External Control.")
        cam.disconnect()
        robot.disconnect()
        return

    intrinsics = cam.get_intrinsics()
    print(f"Camera intrinsics: {intrinsics}")
    mode = int(robot.rtde_r.getRobotMode())
    safety = int(robot.rtde_r.getSafetyMode())
    print(f"robot_mode={mode} safety_mode={safety}")
    if mode != 7:
        print("ERROR: need robot_mode 7 (RUNNING).")
        cam.disconnect()
        robot.disconnect()
        return

    # ── Teach / reuse marker position in base frame ──
    existing_samples: list[dict] = []
    if args.reuse_marker:
        marker_base = None
        if os.path.isfile(args.out):
            with open(args.out, encoding="utf-8") as f:
                prev = json.load(f)
            mb = prev.get("marker_base_m")
            if mb and len(mb) == 3:
                marker_base = np.asarray(mb, dtype=np.float64)
        if marker_base is None and os.path.isfile(SAMPLES_PATH):
            with open(SAMPLES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            mb = data.get("marker_base")
            if mb and len(mb) == 3:
                marker_base = np.asarray(mb, dtype=np.float64)
            if args.append:
                existing_samples = list(data.get("samples") or [])
        if marker_base is None:
            print("ERROR: --reuse-marker but no marker_base in hand_eye.json / samples.")
            cam.disconnect()
            robot.disconnect()
            return
        print(f"Reusing marker base XYZ = {[round(float(v), 4) for v in marker_base]}")
        # Move to hover above marker (keep orientation).
        now = list(robot.get_tcp_pose())
        hover = [
            float(marker_base[0]),
            float(marker_base[1]),
            float(marker_base[2]) + 0.08,
            now[3],
            now[4],
            now[5],
        ]
        print(f"Moving to hover above marker: {[round(v, 4) for v in hover[:3]]}")
        try:
            robot.move_linear(hover, speed=0.12, accel=0.3)
        except Exception as e:
            print(f"ERROR: could not reach hover: {e}")
            cam.disconnect()
            robot.disconnect()
            return
        home = list(robot.get_tcp_pose())
    else:
        print("\n── TEACH MARKER ──")
        print("Move the TCP so the tool tip touches the CENTER of the paper marker.")
        print("(Freedrive in Local, then switch back to Remote + External Control,")
        print(" OR jog with the ops console / pendant while this waits.)")
        while True:
            cmd = input("When TCP is on the marker center, type TEACH: ").strip().upper()
            if cmd == "TEACH":
                break
            if cmd in ("Q", "QUIT"):
                cam.disconnect()
                robot.disconnect()
                return

        marker_base = np.asarray(robot.get_tcp_pose()[:3], dtype=np.float64)
        print(f"Marker base XYZ = {[round(float(v), 4) for v in marker_base]}")

        # Lift clear of the table so we don't crash during the grid.
        print("Lifting +8 cm above the marker...")
        robot.move_tcp_relative(dz=0.08, speed=0.08, accel=0.3)
        home = list(robot.get_tcp_pose())

    print(f"Hover TCP: {[round(v, 4) for v in home[:3]]}")

    obs, err = capture_marker_cam(cam, args.marker_id)
    if err:
        print(f"WARNING: cannot see marker from hover: {err}")
        print("Aim the wrist camera at the table marker, then continue.")
    else:
        print(f"Marker visible at pixel={obs['pixel']} depth={obs['depth_m']}m — good.")

    steps = _pose_grid(step, n)

    print(
        f"\nWill take {len(steps)} views (step={step*100:.0f} cm)"
        f"{f', keeping {len(existing_samples)} prior samples' if existing_samples else ''}."
    )
    print("Keep marker on the table.")
    if not args.yes:
        if input("Type YES to start: ").strip() != "YES":
            print("Aborted.")
            cam.disconnect()
            robot.disconnect()
            return

    samples: list[dict] = list(existing_samples)
    speed, accel = 0.1, 0.4
    # Return to hover between skips so FOV doesn't drift away permanently.
    try:
        for i, (dx, dy, dz) in enumerate(steps, 1):
            # Absolute offset from hover (not chained relative — avoids walk-off).
            target = [
                home[0] + dx,
                home[1] + dy,
                home[2] + dz,
                home[3],
                home[4],
                home[5],
            ]
            before = list(robot.get_tcp_pose())
            print(f"\n[{i}/{len(steps)}] offset=({dx:+.3f},{dy:+.3f},{dz:+.3f})")
            try:
                robot.move_linear(target, speed=speed, accel=accel)
            except Exception as e:
                print(f"  SKIP move: {e}")
                continue
            time.sleep(0.35)
            after = list(robot.get_tcp_pose())
            moved_mm = float(np.linalg.norm(np.array(after[:3]) - np.array(before[:3]))) * 1000
            print(f"  TCP {[round(v, 4) for v in after[:3]]}  (Δ {moved_mm:.1f} mm)")

            obs, err = capture_marker_cam(cam, args.marker_id)
            if err:
                print(f"  SKIP: {err}")
                continue
            sample = {
                **obs,
                "tcp_pose": [round(float(x), 5) for x in after],
                "marker_base": [round(float(x), 5) for x in marker_base],
            }
            samples.append(sample)
            os.makedirs(CALIB_DIR, exist_ok=True)
            with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"mount": "eye_in_hand", "marker_base": marker_base.tolist(), "samples": samples},
                    f,
                    indent=2,
                )
            print(f"  OK  pixel={sample['pixel']} depth={sample['depth_m']}m p_cam={sample['p_cam']}")
            if len(samples) >= 2:
                pix = np.array([s["pixel"] for s in samples])
                span = float(np.linalg.norm(pix.max(0) - pix.min(0)))
                print(f"  samples={len(samples)}  pixel span so far: {span:.1f} px")
    finally:
        print("\nReturning toward hover pose...")
        try:
            robot.move_linear(home, speed=speed, accel=accel)
        except Exception as e:
            print(f"  (return failed: {e})")
        cam.disconnect()
        robot.disconnect()

    solve_eye_in_hand(samples, marker_base, intrinsics, args.out)


def run_solve(args):
    if not os.path.isfile(SAMPLES_PATH):
        print(f"No samples at {SAMPLES_PATH}")
        sys.exit(1)
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    marker_base = np.asarray(data.get("marker_base") or samples[0].get("marker_base"), dtype=np.float64)
    intrinsics = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            intrinsics = json.load(f).get("intrinsics", {})
    if samples and samples[0].get("intrinsics"):
        intrinsics = samples[0]["intrinsics"]
    solve_eye_in_hand(samples, marker_base, intrinsics, args.out)


def run_verify(args):
    from camera.geometry import HandEyeCalibration
    from camera.realsense_camera import RealSenseCamera

    calib_path = args.out if os.path.isfile(args.out) else HAND_EYE_CALIB_PATH
    calib = HandEyeCalibration(calib_path)
    if not calib.loaded:
        print(f"No calibration at {calib_path}")
        sys.exit(1)
    with open(calib_path, encoding="utf-8") as f:
        raw = json.load(f)
    marker_base = np.asarray(raw.get("marker_base_m") or [], dtype=np.float64)
    if marker_base.size != 3:
        print("Calib file has no marker_base_m — cannot verify.")
        sys.exit(1)

    from robot.ur5_driver import UR5Driver

    cam = RealSenseCamera(depth_enabled=True)
    cam.connect()
    robot = UR5Driver(host=args.host)
    robot.connect()
    print(f"mount={calib.mount}  marker_base={marker_base.tolist()}")
    print("Ctrl+C to stop. Move the arm (or let it sit) while looking at the table marker.\n")
    try:
        while True:
            obs, err = capture_marker_cam(cam, args.marker_id)
            if err:
                print(f"  {err}")
                time.sleep(1.0)
                continue
            tcp = list(robot.get_tcp_pose())
            predicted = calib.camera_point_to_base(np.asarray(obs["p_cam"]), tcp_pose=tcp)
            err_mm = float(np.linalg.norm(predicted - marker_base)) * 1000
            print(
                f"  pred={np.round(predicted, 4).tolist()} "
                f"true={np.round(marker_base, 4).tolist()} error={err_mm:.1f} mm "
                f"pixel={obs['pixel']}"
            )
            time.sleep(1.2)
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        robot.disconnect()


def main():
    p = argparse.ArgumentParser(description="Eye-in-hand calibration (camera on UR5)")
    p.add_argument("--auto", action="store_true", help="Teach marker + auto motion + solve")
    p.add_argument("--poses", type=int, default=20, help="Number of views (default 20)")
    p.add_argument("--step-m", type=float, default=0.04, help="Pose spacing meters (default 4 cm)")
    p.add_argument("--solve", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--make-marker", action="store_true")
    p.add_argument("--reuse-marker", action="store_true", help="Reuse marker_base from last calib (no TEACH)")
    p.add_argument("--append", action="store_true", help="Keep prior samples when reusing marker")
    p.add_argument("--yes", action="store_true", help="Skip YES confirmation")
    p.add_argument("--host", default=ROBOT_HOST)
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    if args.marker_id < 0:
        args.marker_id = None

    if args.make_marker:
        path = make_marker()
        print(f"Marker written: {path}")
        print("Print ~4–6 cm and tape it FLAT ON THE TABLE (not on the gripper).")
        return
    if args.auto:
        run_auto(args)
    elif args.solve:
        run_solve(args)
    elif args.verify:
        run_verify(args)
    else:
        p.print_help()
        print("\nFor your wrist/top-mounted camera use:  --auto")


if __name__ == "__main__":
    main()
