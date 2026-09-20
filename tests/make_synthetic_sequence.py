"""Generate a tiny synthetic sequence in CEAR's on-disk format, with KNOWN ground truth.

Why this exists: the real CEAR download is ~5 GB and the trap-prone parts of the pipeline
(quaternion conventions, timestamp units + offsets, the extrinsic transform chain, the
up-axis conversion, the COLMAP round-trip, TSDF fusion, collider floor height) are all
testable WITHOUT it and WITHOUT a GPU. This module renders an analytic room (floor, walls,
a box) along a camera trajectory, then writes the exact files a CEAR sequence has, using
the exact conventions the plan documents (§2.2/§2.3):

  * MoCap.txt      : t(s)  x y z  qx qy qz qw          (scalar-LAST, Y-up marker frame)
  * vectornav.txt  : t(us) gx gy gz ax ay az mx my mz  qw qx qy qz  (scalar-FIRST)
  * realsense_timestamp.txt : filenames with a leading µs timestamp
  * rgb/, depth/   : uint8 RGB and uint16 depth (mm), depth aligned to the RGB frame
  * calibration/rgb_intrinsics.yaml : K, dist, and the RGB->marker extrinsic
  * ground_truth.json : the TRUE camera poses / geometry, for the tests to assert against

This generator uses scipy directly (NOT scripts/pose_utils), so a matching round-trip in
the pipeline is a genuine cross-check, not the same code cancelling its own bug.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------- #
# Conventions (must match the pipeline config that is emitted alongside)
# --------------------------------------------------------------------------- #
OFFSETS_S = {"event": 0.0, "rgb": 0.004611, "imu": -0.004012, "joints": 0.004611}
DEPTH_UNITS_PER_M = 1000.0
DEPTH_INVALID = 0

# Room (Z-up TRUE world, floor at z=0). Metres.
LX, LY, LZ = 6.0, 4.0, 2.5
OBS = dict(x0=2.6, x1=3.4, y0=1.6, y1=2.4, z1=0.6)   # a low box on the floor
CENTER = np.array([LX / 2.0, LY / 2.0, 0.30])

# Camera intrinsics (smaller than CEAR's 640x480 for speed; same math).
W, H = 320, 240
FX = FY = 190.0
CX, CY = 160.0, 120.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# RGB->marker extrinsic (cam_to_marker): marker sits above/behind the camera, tilted.
_T_MARKER_CAM = np.eye(4)
_T_MARKER_CAM[:3, :3] = Rotation.from_euler("xyz", [4, -3, 2], degrees=True).as_matrix()
_T_MARKER_CAM[:3, 3] = [0.02, -0.05, -0.03]

# Y-up <-> Z-up: +90 deg about X maps (x,y,z)_yup -> (x,-z,y)_zup (definitional).
_M_ZUP_FROM_YUP = np.eye(4)
_M_ZUP_FROM_YUP[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()


# --------------------------------------------------------------------------- #
# Trajectory (TRUE camera poses in the Z-up world, OpenCV optical convention)
# --------------------------------------------------------------------------- #
def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation with OpenCV axes (x right, y down, z forward)."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.column_stack([right, down, fwd])


def cam_pose(tau: float, duration: float, preamble_s: float = 0.0
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_world_cam, C) for trajectory time tau in [0, preamble_s + duration].

    If tau < preamble_s the camera is doing the sync preamble: a near-stationary body with
    a large pitch swing (the authors' time-sync motion, TRAP 8). Afterwards it orbits.
    """
    if preamble_s > 0 and tau < preamble_s:
        # stationary at the orbit's start pose (so the trajectory is CONTINUOUS at the
        # handoff — a teleport there would inject a spurious omega spike), doing a big
        # pitch swing that returns to zero at tau == preamble_s.
        eye = np.array([CENTER[0] + 1.4, CENTER[1], 0.40])
        R0 = _look_at(eye, CENTER)
        pitch = 0.5 * np.sin(2 * np.pi * 1.5 * tau)     # large-amplitude pitch oscillation
        return R0 @ Rotation.from_euler("x", pitch).as_matrix(), eye
    tt = tau - preamble_s
    ph = 2 * np.pi * 0.6 * (tt / duration)
    eye = np.array([CENTER[0] + 1.4 * np.cos(ph),
                    CENTER[1] + 1.4 * np.sin(ph),
                    0.40 + 0.08 * np.sin(3 * ph)])
    R = _look_at(eye, CENTER)
    # time-varying wobble about the camera's down-axis -> spread in |omega| for gating
    wob = 0.15 * np.sin(2 * np.pi * (0.5 + 2.0 * (tt / duration)) * tt)
    R = R @ Rotation.from_euler("y", wob).as_matrix()
    return R, eye


# --------------------------------------------------------------------------- #
# Analytic raycast renderer
# --------------------------------------------------------------------------- #
def _bounded_plane_hit(o, d, axis, value, lo, hi):
    """s (H,W) for ray o+s*d hitting plane axis=value inside the other-two-axis box.

    Returns s with np.inf where invalid (no hit / behind / outside bounds).
    """
    dv = d[..., axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (value - o[..., axis]) / dv
        p = o + s[..., None] * d
    ok = (s > 1e-6) & np.isfinite(s)
    others = [a for a in (0, 1, 2) if a != axis]
    ok &= (p[..., others[0]] >= lo[0]) & (p[..., others[0]] <= lo[1])
    ok &= (p[..., others[1]] >= hi[0]) & (p[..., others[1]] <= hi[1])
    s = np.where(ok, s, np.inf)
    return s, p


def _checker(a, b, scale=2.0):
    return ((np.floor(a * scale).astype(np.int64) +
             np.floor(b * scale).astype(np.int64)) % 2)


def render(R_world_cam: np.ndarray, C: np.ndarray, rng: np.random.Generator,
           noise_mm: float = 2.0):
    """Render one (rgb uint8 HxWx3, depth uint16 HxW mm) view of the room."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dir_cam = np.stack([(us - CX) / FX, (vs - CY) / FY, np.ones_like(us)], axis=-1)
    d = dir_cam @ R_world_cam.T          # (H,W,3) world-frame ray dirs; note z-comp==1 in cam
    o = C.reshape(1, 1, 3)

    surfaces = [
        # (axis, value, bound_lo(min,max), bound_hi(min,max), name, base_color)
        (2, 0.0, (0, LX), (0, LY), "floor", (150, 140, 130)),
        (2, LZ, (0, LX), (0, LY), "ceil", (205, 205, 215)),
        (0, 0.0, (0, LY), (0, LZ), "wx0", (170, 120, 110)),
        (0, LX, (0, LY), (0, LZ), "wx1", (110, 150, 170)),
        (1, 0.0, (0, LX), (0, LZ), "wy0", (120, 170, 120)),
        (1, LY, (0, LX), (0, LZ), "wy1", (170, 160, 110)),
        (2, OBS["z1"], (OBS["x0"], OBS["x1"]), (OBS["y0"], OBS["y1"]), "obs_top", (200, 90, 80)),
        (0, OBS["x0"], (OBS["y0"], OBS["y1"]), (0, OBS["z1"]), "obs_x0", (200, 90, 80)),
        (0, OBS["x1"], (OBS["y0"], OBS["y1"]), (0, OBS["z1"]), "obs_x1", (200, 90, 80)),
        (1, OBS["y0"], (OBS["x0"], OBS["x1"]), (0, OBS["z1"]), "obs_y0", (200, 90, 80)),
        (1, OBS["y1"], (OBS["x0"], OBS["x1"]), (0, OBS["z1"]), "obs_y1", (200, 90, 80)),
    ]

    best_s = np.full((H, W), np.inf)
    best_id = np.full((H, W), -1, dtype=np.int64)
    hit_pts = np.zeros((H, W, 3))
    for i, (axis, value, blo, bhi, _name, _col) in enumerate(surfaces):
        s, p = _bounded_plane_hit(o, d, axis, value, blo, bhi)
        closer = s < best_s
        best_s = np.where(closer, s, best_s)
        best_id = np.where(closer, i, best_id)
        hit_pts = np.where(closer[..., None], p, hit_pts)

    # --- color ---
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    for i, (axis, value, blo, bhi, name, col) in enumerate(surfaces):
        mask = best_id == i
        if not mask.any():
            continue
        p = hit_pts[mask]
        if axis == 2:      # horizontal surface -> checker in x,y
            chk = _checker(p[:, 0], p[:, 1])
        elif axis == 0:    # x-wall -> checker in y,z
            chk = _checker(p[:, 1], p[:, 2])
        else:              # y-wall -> checker in x,z
            chk = _checker(p[:, 0], p[:, 2])
        shade = np.where(chk == 0, 0.72, 1.0)
        rgb[mask] = np.array(col, dtype=np.float64) * shade[:, None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # --- depth (camera-frame z == best_s because dir_cam.z == 1) ---
    depth_m = best_s.copy()
    depth_m[~np.isfinite(depth_m)] = 0.0
    if noise_mm > 0:
        valid = depth_m > 0
        depth_m[valid] += rng.normal(0, noise_mm / 1000.0, size=valid.sum())
    depth_u16 = np.clip(depth_m * DEPTH_UNITS_PER_M, 0, 65535).astype(np.uint16)
    depth_u16[best_s == np.inf] = DEPTH_INVALID
    return rgb, depth_u16


# --------------------------------------------------------------------------- #
# Angular velocity from the trajectory (for the IMU stream)
# --------------------------------------------------------------------------- #
def body_omega(tau: float, duration: float, preamble_s: float = 0.0,
               h: float = 1e-3) -> np.ndarray:
    total = duration + preamble_s
    Ra, _ = cam_pose(max(tau - h, 0.0), duration, preamble_s)
    Rb, _ = cam_pose(min(tau + h, total), duration, preamble_s)
    dR = Ra.T @ Rb
    return Rotation.from_matrix(dR).as_rotvec() / (2 * h)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def make_synthetic_sequence(out_root: str, seq_name: str = "synthetic",
                            n_frames: int = 40, duration: float = 2.0,
                            seed: int = 0, preamble_s: float = 0.0,
                            depth_dt_ms: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    total = duration + preamble_s
    depth_dt = depth_dt_ms / 1000.0            # RGB->depth timestamp gap (TRAP 7)
    seq_dir = os.path.join(out_root, seq_name)
    rgb_dir = os.path.join(seq_dir, "rgb")
    depth_dir = os.path.join(seq_dir, "depth")
    calib_dir = os.path.join(seq_dir, "calibration")
    # clear image dirs so re-generating into the same path can't leave stale frames
    for d in (rgb_dir, depth_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
    for d in (rgb_dir, depth_dir, calib_dir):
        os.makedirs(d, exist_ok=True)

    # ---- MoCap.txt : 120 Hz, seconds, Y-up marker frame, scalar-LAST ----
    mocap_tau = np.arange(0.0, total + 1e-9, 1.0 / 120.0)
    with open(os.path.join(seq_dir, "MoCap.txt"), "w") as f:
        f.write("# timestamp(s) x y z qx qy qz qw  (OptiTrack, marker frame, Y-up)\n")
        for tau in mocap_tau:
            R_wc, C = cam_pose(tau, duration, preamble_s)
            T_wc_z = np.eye(4); T_wc_z[:3, :3] = R_wc; T_wc_z[:3, 3] = C
            # camera -> marker(world) : T_world_marker = T_world_cam @ inv(T_marker_cam)
            T_wm_z = T_wc_z @ np.linalg.inv(_T_MARKER_CAM)
            # express marker pose in a Y-up world (pipeline undoes this)
            T_wm_y = np.linalg.inv(_M_ZUP_FROM_YUP) @ T_wm_z
            q = Rotation.from_matrix(T_wm_y[:3, :3]).as_quat()   # xyzw
            t = T_wm_y[:3, 3]
            f.write(f"{tau:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")

    # ---- vectornav.txt : 400 Hz, microseconds, scalar-FIRST ----
    imu_tau = np.arange(0.0, total + 1e-9, 1.0 / 400.0)
    with open(os.path.join(seq_dir, "vectornav.txt"), "w") as f:
        f.write("# t(us) gx gy gz ax ay az mx my mz qw qx qy qz  (scalar-first)\n")
        for tau in imu_tau:
            w = body_omega(tau, duration, preamble_s)
            R_wc, _ = cam_pose(tau, duration, preamble_s)
            q = Rotation.from_matrix(R_wc).as_quat()             # xyzw
            ts_us = int(round((tau - OFFSETS_S["imu"]) * 1e6))
            f.write(f"{ts_us} {w[0]:.6f} {w[1]:.6f} {w[2]:.6f} "
                    f"0 0 9.81 0 0 0 "
                    f"{q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f}\n")

    # ---- mini_cheetah_joint.txt : 100 Hz, microseconds, 12 joints ----
    with open(os.path.join(seq_dir, "mini_cheetah_joint.txt"), "w") as f:
        f.write("# t(us) 12 joint angles (rad)\n")
        for tau in np.arange(0.0, total + 1e-9, 1.0 / 100.0):
            ts_us = int(round((tau - OFFSETS_S["joints"]) * 1e6))
            j = 0.2 * np.sin(2 * np.pi * tau + np.arange(12))
            f.write(f"{ts_us} " + " ".join(f"{v:.5f}" for v in j) + "\n")

    # ---- frames + realsense_timestamp.txt ----
    # RGB and depth get DIFFERENT timestamps (TRAP 7): render each at its own pose so the
    # pipeline must interpolate each stream at its own time to recover the geometry.
    if preamble_s > 0:
        frame_tau = np.linspace(0.02 * total, 0.98 * total - depth_dt, n_frames)
    else:
        frame_tau = np.linspace(0.05 * duration, 0.95 * duration - depth_dt, n_frames)
    gt_frames = []
    with open(os.path.join(seq_dir, "realsense_timestamp.txt"), "w") as f:
        f.write("# depth_rgb  depth_event  rgb   (leading integer = timestamp in us)\n")
        for tau in frame_tau:
            t_rgb, t_depth = float(tau), float(tau) + depth_dt
            R_rgb, C_rgb = cam_pose(t_rgb, duration, preamble_s)
            R_dep, C_dep = cam_pose(t_depth, duration, preamble_s)
            rgb, _ = render(R_rgb, C_rgb, rng)              # color at the RGB instant
            _, depth = render(R_dep, C_dep, rng)            # geometry at the depth instant
            rgb_us = int(round((t_rgb - OFFSETS_S["rgb"]) * 1e6))
            dep_us = int(round((t_depth - OFFSETS_S["rgb"]) * 1e6))
            rgb_name = f"{rgb_us:012d}_rgb.png"
            depth_name = f"{dep_us:012d}_depth_rgb.png"
            event_name = f"{dep_us:012d}_depth_event.png"
            cv2.imwrite(os.path.join(rgb_dir, rgb_name), rgb[:, :, ::-1])   # RGB->BGR
            cv2.imwrite(os.path.join(depth_dir, depth_name), depth)
            f.write(f"{depth_name} {event_name} {rgb_name}\n")
            T_wc = np.eye(4); T_wc[:3, :3] = R_rgb; T_wc[:3, 3] = C_rgb
            gt_frames.append(dict(tau=t_rgb, rgb=rgb_name, depth=depth_name,
                                  T_world_cam=T_wc.tolist()))

    # ---- calibration ----
    calib = dict(
        K=K.tolist(), dist=[0.0, 0.0, 0.0, 0.0, 0.0], dist_model="opencv",
        width=W, height=H,
        # stored as cam_to_marker (T maps cam-frame points into the marker frame)
        T_rgb_marker=_T_MARKER_CAM.tolist(),
        T_rgb_robot=np.eye(4).tolist(),
    )
    with open(os.path.join(calib_dir, "rgb_intrinsics.yaml"), "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)

    # ---- ground truth for the tests ----
    gt = dict(
        room=dict(LX=LX, LY=LY, LZ=LZ), obstacle=OBS, floor_z=0.0,
        K=[FX, FY, CX, CY], width=W, height=H,
        frames=gt_frames,
        preamble_s=float(preamble_s), depth_dt_ms=float(depth_dt_ms),
        conventions=dict(world_up="y", target_up="z"),
    )
    with open(os.path.join(seq_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    # ---- matching pipeline config ----
    cfg = _emit_config(seq_dir, out_root, seq_name)
    cfg_path = os.path.join(out_root, f"{seq_name}_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return dict(seq_dir=seq_dir, config=cfg_path, ground_truth=os.path.join(seq_dir, "ground_truth.json"))


def _emit_config(seq_dir: str, out_root: str, seq_name: str) -> dict:
    return {
        "sequence": {
            "name": seq_name, "data_root": seq_dir, "pose_file": "MoCap.txt",
            "rgb_dir": "rgb", "depth_dir": "depth", "raw_depth_dir": "depth",
            "timestamp_table": "realsense_timestamp.txt", "imu_file": "vectornav.txt",
            "joint_file": "mini_cheetah_joint.txt",
        },
        "timestamps": {
            "units": {"rgb": "microseconds", "imu": "microseconds",
                      "joints": "microseconds", "pose": "seconds"},
            "offsets_s": dict(OFFSETS_S),
        },
        "quaternions": {"pose_order": "xyzw", "imu_order": "wxyz", "colmap_order": "wxyz"},
        "frames": {
            "world_up": "y", "target_up": "z", "pose_frame": "marker",
            "extrinsic": {"rgb_marker_direction": "cam_to_marker",
                          "rgb_robot_direction": "cam_to_robot", "delta_T": None},
        },
        "photometric": {"auto_exposure_confirmed": True},
        "intrinsics": {
            "calib_file": os.path.join(seq_dir, "calibration", "rgb_intrinsics.yaml"),
            "image_width": W, "image_height": H,
        },
        "depth": {"units_per_meter": DEPTH_UNITS_PER_M, "invalid_value": DEPTH_INVALID,
                  "min_range_m": 0.2, "max_range_m": 4.0},
        "fusion": {"voxel_size_m": 0.025, "sdf_trunc_m": 0.10, "seed_downsample_m": 0.03},
        "gating": {"omega_percentile_keep": 60, "laplacian_min": 5.0,
                   "spatial_min_sep_m": 0.01, "holdout_every": 8,
                   "exclude_preamble": True, "preamble_end_s": None,
                   "preamble_swing_factor": 2.0},
        "sync": {"drift_check": True, "drift_tolerance_ms": 2.0},
        "collider": {"method": "tsdf_marching_cubes", "max_triangles": 50000,
                     "floor_patch": True},
        "paths": {"out_root": os.path.join(out_root, f"{seq_name}_out")},
    }


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic CEAR-format sequence.")
    ap.add_argument("--out", required=True, help="output root directory")
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preamble", type=float, default=0.0, help="sync-preamble seconds (TRAP 8)")
    ap.add_argument("--depth-dt-ms", type=float, default=4.0, help="RGB->depth ts gap (TRAP 7)")
    args = ap.parse_args()
    paths = make_synthetic_sequence(args.out, args.name, n_frames=args.frames, seed=args.seed,
                                    preamble_s=args.preamble, depth_dt_ms=args.depth_dt_ms)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
