"""Motion / spatial gating shared by Phase 3 (depth fusion, §6.1 / TRAP 6) and Phase 4
(blur gating, §7.1). Both streams gate on the same angular-velocity criterion; keeping it
in one place guarantees they use the identical ‖ω‖ definition."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

import pose_utils as pu


def omega_magnitude_at(cfg: dict, ts: Sequence[float]) -> np.ndarray:
    """Interpolate IMU gyro onto the given event-clock timestamps; return ‖ω‖ (rad/s)."""
    seq = cfg["sequence"]
    imu = pu.parse_vectornav(os.path.join(seq["data_root"], seq["imu_file"]), cfg)
    ts = np.asarray(ts, dtype=np.float64)
    tc = np.clip(ts, imu.t[0], imu.t[-1])          # clamp to IMU coverage
    w = np.stack([np.interp(tc, imu.t, imu.omega[:, d]) for d in range(3)], axis=1)
    return np.linalg.norm(w, axis=1)


def greedy_spatial_thin(centers: np.ndarray, min_sep: float) -> np.ndarray:
    """Greedy keep-mask enforcing a minimum separation between kept camera centers.

    Prevents 500 near-identical viewpoints from one spot (plan §7.1). Order-preserving.
    """
    centers = np.asarray(centers, dtype=np.float64)
    kept = np.zeros(len(centers), dtype=bool)
    anchors = []
    for i, c in enumerate(centers):
        if not anchors or min(np.linalg.norm(c - a) for a in anchors) >= min_sep:
            kept[i] = True
            anchors.append(c)
    return kept


def apply_preamble_exclusion(cfg: dict, ts: Sequence[float], keep: np.ndarray):
    """Drop frames inside the sync preamble (TRAP 8). Returns (new_keep, n_excluded).

    The preamble (pitch swing + thrown ball) is at the very start; the ball is a dynamic
    object that must not enter training frames or fusion. No-op if disabled or none found.
    """
    import sync_utils as su
    keep = np.asarray(keep, dtype=bool).copy()
    if not cfg.get("gating", {}).get("exclude_preamble", True):
        return keep, 0
    end = su.detect_sync_preamble(cfg)
    if end is None:
        return keep, 0
    excl = np.asarray(ts, dtype=np.float64) <= end
    n = int((keep & excl).sum())
    keep[excl] = False
    return keep, n


def motion_gate(cfg: dict, ts: Sequence[float], centers: np.ndarray,
                omega_percentile_keep: float, min_sep: float,
                laplacian: np.ndarray = None, laplacian_min: float = None):
    """Return (keep_mask, info). Keeps frames with low ‖ω‖ (below the percentile), high
    sharpness (if provided), and adequate spatial spread."""
    omega = omega_magnitude_at(cfg, ts)
    thr = np.percentile(omega, omega_percentile_keep)
    keep = omega <= thr
    if laplacian is not None and laplacian_min is not None:
        keep &= laplacian >= laplacian_min
    # spatial thinning applied among the survivors, in order
    idx = np.where(keep)[0]
    if len(idx):
        sub = greedy_spatial_thin(centers[idx], min_sep)
        keep[idx[~sub]] = False
    info = {"omega": omega, "omega_threshold": float(thr),
            "n_before": len(ts), "n_after": int(keep.sum())}
    return keep, info
