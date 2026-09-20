"""Shared pose / frame / COLMAP utilities for the CEAR pipeline (plan §5, Phases 2-5).

EVERY convention the plan's TRAPs warn about is handled HERE and nowhere else. If you
find yourself reordering a quaternion or inverting a transform in another script, stop
and route it through this module instead.

Conventions, stated once (see docs/PLAN.md §2.3 and §5):

  Rotations
    - Internally everything is a 4x4 float64 homogeneous transform T = [[R, t],[0,1]].
    - `T_a_b` reads "transform that maps a point expressed in frame b into frame a",
      i.e. p_a = T_a_b @ p_b. So T_world_cam maps camera-frame points to world (it is the
      camera-to-world / camera pose), and its columns' translation is the camera center.

  Quaternions (TRAP 1 / 1b) — order is ALWAYS explicit:
    - MoCap.txt / FasterLIO.txt : 'xyzw'  (scalar-LAST)
    - vectornav.txt            : 'wxyz'  (scalar-FIRST)
    - COLMAP images.txt        : 'wxyz'  (scalar-FIRST), and stores WORLD-TO-CAMERA.

  Timestamps (TRAP 2 / 3):
    - All returned times are float SECONDS on the EVENT-camera clock.
    - t_event = t_sensor_seconds + offsets_s[stream].  MoCap poses are already on the
      event clock (they are the reference), so their offset is 0.

  Camera model:
    - OpenCV / COLMAP optical frame: +x right, +y down, +z forward.

  World up-axis (§5.4):
    - Source (OptiTrack) world is Y-up; the pipeline converts once to Z-up (USD/Isaac,
      floor at z=0) via a +90° rotation about X:  (x,y,z)_yup -> (x,-z,y)_zup.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    """Load a YAML config and expand ${ENV} references in every string value."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def _expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj

    return _expand(cfg)


# --------------------------------------------------------------------------- #
# Quaternion / transform primitives
# --------------------------------------------------------------------------- #

_QUAT_ORDERS = {"wxyz", "xyzw"}


def quat_to_R(q: Sequence[float], order: str, *, tol: float = 1e-3) -> np.ndarray:
    """Quaternion (in the given order) -> 3x3 rotation matrix. Asserts unit norm."""
    if order not in _QUAT_ORDERS:
        raise ValueError(f"unknown quaternion order {order!r}")
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or abs(n - 1.0) > tol:
        raise AssertionError(f"quaternion not unit-norm (|q|={n:.6f}, order={order})")
    q = q / n
    if order == "wxyz":  # scipy wants scalar-last xyzw
        q = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q).as_matrix()


def R_to_quat(R: np.ndarray, order: str) -> np.ndarray:
    """3x3 rotation matrix -> quaternion in the requested order."""
    if order not in _QUAT_ORDERS:
        raise ValueError(f"unknown quaternion order {order!r}")
    q_xyzw = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    if order == "wxyz":
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return q_xyzw


def make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse (transpose R, adjust t). Faster and exact vs np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_from_quat_trans(q: Sequence[float], t: Sequence[float], order: str) -> np.ndarray:
    return make_T(quat_to_R(q, order), t)


# --------------------------------------------------------------------------- #
# Up-axis conversion (§5.4)
# --------------------------------------------------------------------------- #

def up_axis_R(from_up: str, to_up: str) -> np.ndarray:
    """3x3 rotation mapping a point in a `from_up` world into a `to_up` world."""
    from_up, to_up = from_up.lower(), to_up.lower()
    if from_up == to_up:
        return np.eye(3)
    if {from_up, to_up} == {"y", "z"}:
        # +90 deg about X:  (x,y,z)_yup -> (x,-z,y)_zup ; inverse for z->y.
        Ryz = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        return Ryz if (from_up, to_up) == ("y", "z") else Ryz.T
    raise NotImplementedError(f"up-axis conversion {from_up}->{to_up} not implemented")


def convert_world_up(T_world_cam: np.ndarray, from_up: str, to_up: str) -> np.ndarray:
    """Re-express a camera-to-world transform in a world with a different up-axis."""
    M = np.eye(4)
    M[:3, :3] = up_axis_R(from_up, to_up)
    return M @ T_world_cam


# --------------------------------------------------------------------------- #
# File parsers (TRAP 1 / 2 / 3)
# --------------------------------------------------------------------------- #

@dataclass
class PoseTrack:
    """Time-stamped rigid poses in a single frame convention."""
    t: np.ndarray                 # (N,) float seconds, EVENT clock, sorted
    T: np.ndarray                 # (N,4,4) camera/marker-to-world transforms
    frame: str = "marker"         # which body frame these poses describe
    world_up: str = "y"           # up-axis of the world they live in


def _offset(cfg: dict, stream: str) -> float:
    return float(cfg.get("timestamps", {}).get("offsets_s", {}).get(stream, 0.0))


def parse_mocap(path: str, cfg: dict) -> PoseTrack:
    """Parse MoCap.txt / FasterLIO.txt:  t(s) x y z qx qy qz qw  (quat scalar-LAST).

    Returns marker(or robot)-to-world poses in the OptiTrack (Y-up) world, on the event
    clock (pose offset is 0 — poses are the reference clock).
    """
    order = cfg["quaternions"]["pose_order"]           # 'xyzw'
    frame = cfg["frames"].get("pose_frame", "marker")
    world_up = cfg["frames"].get("world_up", "y")
    off = _offset(cfg, "pose")                          # normally 0.0

    rows = _read_numeric_rows(path, min_cols=8)
    t = rows[:, 0] + off
    Ts = np.empty((len(rows), 4, 4), dtype=np.float64)
    for i, r in enumerate(rows):
        xyz = r[1:4]
        quat = r[4:8]
        Ts[i] = T_from_quat_trans(quat, xyz, order)
    order_idx = np.argsort(t, kind="stable")
    return PoseTrack(t=t[order_idx], T=Ts[order_idx], frame=frame, world_up=world_up)


@dataclass
class ImuTrack:
    t: np.ndarray        # (N,) seconds, event clock
    omega: np.ndarray    # (N,3) gyro rad/s
    accel: np.ndarray    # (N,3) m/s^2
    quat_R: np.ndarray   # (N,3,3) orientation (from the scalar-FIRST quat)


def parse_vectornav(path: str, cfg: dict) -> ImuTrack:
    """Parse vectornav.txt:  t(us) gx gy gz ax ay az magx magy magz qw qx qy qz.

    Quaternion is scalar-FIRST (TRAP 1); timestamp is microseconds (TRAP 2); apply the
    IMU offset (TRAP 3).
    """
    order = cfg["quaternions"]["imu_order"]     # 'wxyz'
    off = _offset(cfg, "imu")
    rows = _read_numeric_rows(path, min_cols=14)
    t = rows[:, 0] / 1e6 + off
    omega = rows[:, 1:4]
    accel = rows[:, 4:7]
    quatR = np.empty((len(rows), 3, 3))
    for i, r in enumerate(rows):
        quatR[i] = quat_to_R(r[10:14], order)
    idx = np.argsort(t, kind="stable")
    return ImuTrack(t=t[idx], omega=omega[idx], accel=accel[idx], quat_R=quatR[idx])


@dataclass
class FrameRecord:
    t: float           # seconds, event clock (from RGB timestamp + rgb offset)
    rgb_name: str      # filename in the rgb/ (or raw_rgb/) directory
    depth_name: str    # filename in the depth/ directory (aligned to RGB)


def parse_realsense_timestamps(path: str, cfg: dict) -> List[FrameRecord]:
    """Parse realsense_timestamp.txt into per-frame (rgb, aligned-depth) records.

    The published format packs the timestamp into each filename as a leading integer
    (microseconds), with columns for depth-in-rgb, depth-in-event and rgb filenames.
    We read the RGB column's timestamp, convert us->s, and apply the RealSense offset.
    The exact column order is a Phase-1 VERIFY; this parser is tolerant (it finds the
    columns by their filename suffix, falling back to positional order).
    """
    off = _offset(cfg, "rgb")
    records: List[FrameRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            # pure RGB frame: has 'rgb', not 'depth', not 'event'
            rgb_tok = _pick_token(toks, wants=("rgb",), exclude=("depth", "event"),
                                  fallback_idx=-1)
            # depth aligned to RGB: has both 'depth' and 'rgb', not 'event'
            depth_tok = _pick_token(toks, wants=("depth", "rgb"), exclude=("event",),
                                    fallback_idx=0)
            ts_us = _leading_int(rgb_tok)
            records.append(FrameRecord(t=ts_us / 1e6 + off,
                                       rgb_name=rgb_tok, depth_name=depth_tok))
    records.sort(key=lambda r: r.t)
    return records


def parse_joints(path: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Parse mini_cheetah_joint.txt -> (t_seconds_event, joints[N,12])."""
    off = _offset(cfg, "joints")
    rows = _read_numeric_rows(path, min_cols=13)
    t = rows[:, 0] / 1e6 + off
    idx = np.argsort(t, kind="stable")
    return t[idx], rows[idx, 1:13]


# --------------------------------------------------------------------------- #
# Interpolation (plan §5 step 2): LERP translation, SLERP rotation
# --------------------------------------------------------------------------- #

class PoseInterpolator:
    """Interpolate a PoseTrack to arbitrary event-clock timestamps.

    Rotation via SLERP, translation via linear interpolation. NEVER nearest-neighbour
    (plan §5): at 60 Hz with fast motion, NN adds up to ~8 ms of pose error.
    """

    def __init__(self, track: PoseTrack):
        t = track.t
        # de-duplicate identical timestamps (Slerp requires strictly increasing times)
        keep = np.concatenate(([True], np.diff(t) > 0))
        self.t = t[keep]
        T = track.T[keep]
        if len(self.t) < 2:
            raise ValueError("need >= 2 distinct poses to interpolate")
        self.trans = T[:, :3, 3].copy()
        self._slerp = Slerp(self.t, Rotation.from_matrix(T[:, :3, :3]))
        self.frame = track.frame
        self.world_up = track.world_up
        self.tmin, self.tmax = float(self.t[0]), float(self.t[-1])

    def in_range(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=np.float64)
        return (ts >= self.tmin) & (ts <= self.tmax)

    def at_many(self, ts: Sequence[float]) -> np.ndarray:
        """Return (M,4,4) poses. Raises if any timestamp is out of the covered range."""
        ts = np.asarray(ts, dtype=np.float64)
        if not np.all(self.in_range(ts)):
            bad = ts[~self.in_range(ts)]
            raise ValueError(
                f"{len(bad)} timestamp(s) outside pose range "
                f"[{self.tmin:.6f},{self.tmax:.6f}] (e.g. {bad[0]:.6f})")
        R = self._slerp(ts).as_matrix()
        tx = np.empty((len(ts), 3))
        for d in range(3):
            tx[:, d] = np.interp(ts, self.t, self.trans[:, d])
        out = np.repeat(np.eye(4)[None], len(ts), axis=0)
        out[:, :3, :3] = R
        out[:, :3, 3] = tx
        return out

    def at(self, t: float) -> np.ndarray:
        return self.at_many([t])[0]


# --------------------------------------------------------------------------- #
# Extrinsic chain + up-axis (TRAP 4, §5.3-5.4)
# --------------------------------------------------------------------------- #

@dataclass
class Calibration:
    K: np.ndarray                       # 3x3 intrinsics
    dist: np.ndarray                    # distortion coeffs (k1 k2 p1 p2 k3 ...)
    width: int
    height: int
    T_marker_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_robot_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    dist_model: str = "opencv"


def load_calibration(path: str, cfg: dict) -> Calibration:
    """Load calibration (intrinsics + RGB-Marker / RGB-Robot extrinsics).

    The stored extrinsic DIRECTION is read from cfg.frames.extrinsic.*_direction so a
    single flag flip covers the TRAP-4 "T vs T^-1" ambiguity. Everything is normalized
    here to the ``T_marker_cam`` / ``T_robot_cam`` (cam-to-body) convention.
    """
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    K = np.array(c["K"], dtype=np.float64).reshape(3, 3)
    dist = np.array(c.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
    w = int(c.get("width", cfg["intrinsics"].get("image_width", 640)))
    h = int(c.get("height", cfg["intrinsics"].get("image_height", 480)))

    ext_cfg = cfg["frames"].get("extrinsic", {})

    def _norm(mat_key, direction_key, default_dir):
        if mat_key not in c:
            return np.eye(4)
        M = np.array(c[mat_key], dtype=np.float64).reshape(4, 4)
        direction = ext_cfg.get(direction_key, default_dir)
        # normalize to cam_to_body (T_body_cam)
        if direction.startswith("cam_to_"):
            return M
        if direction.endswith("_to_cam"):
            return invert_T(M)
        raise ValueError(f"bad extrinsic direction {direction!r}")

    T_marker_cam = _norm("T_rgb_marker", "rgb_marker_direction", "cam_to_marker")
    T_robot_cam = _norm("T_rgb_robot", "rgb_robot_direction", "cam_to_robot")
    return Calibration(K=K, dist=dist, width=w, height=h,
                       T_marker_cam=T_marker_cam, T_robot_cam=T_robot_cam,
                       dist_model=c.get("dist_model", "opencv"))


def marker_pose_to_world_cam(T_world_body: np.ndarray, calib: Calibration,
                             cfg: dict) -> np.ndarray:
    """Compose the body(marker/robot) pose with the extrinsic to get the camera pose.

    Using the T_a_b convention (p_a = T_a_b @ p_b), the calibration stores the
    cam-to-body transform T_body_cam (it maps camera-frame points into the body frame),
    and the mocap gives T_world_body. The camera-to-world pose is therefore a DIRECT
    product — a spurious inverse here is the classic TRAP-4 failure:

        pose_frame == 'marker' :  T_world_cam = T_world_marker @ T_marker_cam
        pose_frame == 'robot'  :  T_world_cam = T_world_robot  @ T_robot_cam
    """
    frame = cfg["frames"].get("pose_frame", "marker")
    if frame == "marker":
        T_body_cam = calib.T_marker_cam     # T_marker_cam: maps cam-frame pts -> marker
    elif frame == "robot":
        T_body_cam = calib.T_robot_cam
    else:
        raise ValueError(f"unknown pose_frame {frame!r}")
    return T_world_body @ T_body_cam


def build_world_to_cam_track(mocap: PoseTrack, interp: PoseInterpolator,
                             calib: Calibration, cfg: dict,
                             ts_event: Sequence[float]) -> np.ndarray:
    """Full chain for a set of frame timestamps -> (M,4,4) camera-to-world in TARGET up.

    Steps (plan §5): interpolate body pose -> compose extrinsic -> convert up-axis.
    """
    to_up = cfg["frames"].get("target_up", "z")
    from_up = mocap.world_up
    T_world_body = interp.at_many(ts_event)              # (M,4,4) in source-up world
    out = np.empty_like(T_world_body)
    for i in range(len(T_world_body)):
        T_world_cam = marker_pose_to_world_cam(T_world_body[i], calib, cfg)
        out[i] = convert_world_up(T_world_cam, from_up, to_up)
    return out


# --------------------------------------------------------------------------- #
# COLMAP text model I/O (TRAP 1b): world-to-camera, scalar-first quats
# --------------------------------------------------------------------------- #

@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]     # PINHOLE: [fx, fy, cx, cy]


@dataclass
class ColmapImage:
    id: int
    T_world_cam: np.ndarray   # camera-to-world (4x4); we serialize the inverse
    camera_id: int
    name: str


def write_colmap_model(out_dir: str, camera: ColmapCamera,
                       images: List[ColmapImage],
                       points3D: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None) -> None:
    """Write cameras.txt / images.txt / points3D.txt (COLMAP text format).

    images.txt stores WORLD-TO-CAMERA with scalar-FIRST quats (TRAP 1b).
    points3D is optional (seeded in Phase 3); pass None for a placeholder.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{camera.id} {camera.model} {camera.width} {camera.height} "
                + " ".join(f"{p:.10g}" for p in camera.params) + "\n")

    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "# (a second, empty line per image: no 2D points)\n")
        for im in images:
            T_cam_world = invert_T(im.T_world_cam)
            qw, qx, qy, qz = R_to_quat(T_cam_world[:3, :3], "wxyz")
            tx, ty, tz = T_cam_world[:3, 3]
            f.write(f"{im.id} {qw:.10g} {qx:.10g} {qy:.10g} {qz:.10g} "
                    f"{tx:.10g} {ty:.10g} {tz:.10g} {im.camera_id} {im.name}\n\n")

    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        if points3D is not None:
            cols = (colors if colors is not None
                    else np.full((len(points3D), 3), 128, dtype=np.int64))
            for i, (p, c) in enumerate(zip(points3D, cols), start=1):
                f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n")


def write_points3D_txt(model_dir: str, points: np.ndarray,
                       colors: Optional[np.ndarray] = None) -> None:
    """Overwrite ONLY points3D.txt in an existing COLMAP model (Phase 3 seed, §6.2).

    Leaves cameras.txt / images.txt untouched. Track fields are minimal (3DGS
    initializers consume positions + colors only).
    """
    cols = (colors if colors is not None
            else np.full((len(points), 3), 128, dtype=np.int64))
    with open(os.path.join(model_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (p, c) in enumerate(zip(points, cols), start=1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n")


def read_colmap_model(model_dir: str) -> Tuple[Dict[int, ColmapCamera],
                                               List[ColmapImage],
                                               np.ndarray]:
    """Read a COLMAP text model back. Returns (cameras, images (with T_world_cam), pts).

    Reconstructs camera-to-world from the stored world-to-camera pose — the round-trip
    that Gate 2b relies on to catch TRAP 1b.
    """
    cameras: Dict[int, ColmapCamera] = {}
    with open(os.path.join(model_dir, "cameras.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cameras[int(p[0])] = ColmapCamera(
                id=int(p[0]), model=p[1], width=int(p[2]), height=int(p[3]),
                params=[float(x) for x in p[4:]])

    images: List[ColmapImage] = []
    with open(os.path.join(model_dir, "images.txt")) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    # data lines come in pairs (pose line, points2D line); points2D line may be blank
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        p = ln.split()
        img_id = int(p[0])
        qw, qx, qy, qz = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        tx, ty, tz = (float(p[5]), float(p[6]), float(p[7]))
        cam_id = int(p[8])
        name = p[9] if len(p) > 9 else ""
        R_cw = quat_to_R([qw, qx, qy, qz], "wxyz")
        T_cam_world = make_T(R_cw, [tx, ty, tz])
        images.append(ColmapImage(id=img_id, T_world_cam=invert_T(T_cam_world),
                                  camera_id=cam_id, name=name))
        i += 2  # skip the (empty) points2D line

    pts = []
    p3d_path = os.path.join(model_dir, "points3D.txt")
    if os.path.exists(p3d_path):
        with open(p3d_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                pts.append([float(p[1]), float(p[2]), float(p[3])])
    points = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))
    return cameras, images, points


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _read_numeric_rows(path: str, min_cols: int) -> np.ndarray:
    """Read whitespace/comma-separated numeric rows, skipping comments/short lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            try:
                vals = [float(x) for x in parts[:max(min_cols, len(parts))]]
            except ValueError:
                continue
            if len(vals) >= min_cols:
                rows.append(vals)
    if not rows:
        raise ValueError(f"no numeric rows with >= {min_cols} cols in {path}")
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=np.float64)


def _leading_int(token: str) -> int:
    m = re.match(r"\s*(\d+)", os.path.basename(token))
    if not m:
        raise ValueError(f"no leading integer timestamp in token {token!r}")
    return int(m.group(1))


def _pick_token(toks: List[str], wants, exclude, fallback_idx: int) -> str:
    """First token whose basename contains all `wants` and none of `exclude`."""
    wants = (wants,) if isinstance(wants, str) else tuple(wants)
    exclude = (exclude,) if isinstance(exclude, str) else tuple(exclude)
    for t in toks:
        b = os.path.basename(t)
        if all(w in b for w in wants) and not any(e in b for e in exclude):
            return t
    return toks[fallback_idx]


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")
