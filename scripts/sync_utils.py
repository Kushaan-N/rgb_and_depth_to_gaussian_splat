"""Time-sync utilities from the paper's sync procedure (docs/PLAN.md TRAP 3 + TRAP 8).

CEAR has NO hardware sync: per-sensor offsets were estimated post-hoc by cross-correlating
a deliberate body-pitch-swing recorded at the START of every sequence (with a thrown ball
for visual verification), using the event camera as reference. Two consequences we handle:

  * TRAP 8 — that preamble (swing + flying ball) must be EXCLUDED from training frames and
    depth fusion (the ball is a dynamic object that would bake into the splat).
  * TRAP 3 — free-running clocks can drift within a sequence; re-estimate the IMU↔pose lag
    on the first vs last window and, if it moved, report/correct a linear drift.

Both use the same signal the authors used: IMU gyro magnitude vs mocap-derived angular
velocity. The signal is in-band, so these checks cost nothing extra.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import pose_utils as pu


# --------------------------------------------------------------------------- #
# angular-velocity signals
# --------------------------------------------------------------------------- #
def _imu_omega_mag(cfg):
    seq = cfg["sequence"]
    import os
    imu = pu.parse_vectornav(os.path.join(seq["data_root"], seq["imu_file"]), cfg)
    return imu.t, np.linalg.norm(imu.omega, axis=1)


def _mocap_omega_mag(cfg, t_grid):
    """Angular-velocity magnitude from the pose stream on t_grid (finite-diff of SLERP).

    ‖ω‖ is invariant to the fixed marker↔camera rotation, so the marker-frame poses give
    the same magnitude the IMU sees — which is why the authors can cross-correlate them.
    """
    import os
    seq = cfg["sequence"]
    mocap = pu.parse_mocap(os.path.join(seq["data_root"], seq["pose_file"]), cfg)
    interp = pu.PoseInterpolator(mocap)
    g = t_grid[(t_grid >= interp.tmin) & (t_grid <= interp.tmax)]
    if len(g) < 3:
        return t_grid, np.zeros_like(t_grid)
    T = interp.at_many(g)
    R = T[:, :3, :3]
    dt = np.diff(g)
    mag = np.zeros(len(g))
    for i in range(len(g) - 1):
        dR = R[i].T @ R[i + 1]
        mag[i] = np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()) / max(dt[i], 1e-6)
    mag[-1] = mag[-2] if len(mag) > 1 else 0.0
    out = np.zeros_like(t_grid)
    out[(t_grid >= interp.tmin) & (t_grid <= interp.tmax)] = mag
    return t_grid, out


# --------------------------------------------------------------------------- #
# TRAP 8 — sync-preamble detection
# --------------------------------------------------------------------------- #
def detect_sync_preamble(cfg) -> Optional[float]:
    """Return the event-clock time the sync preamble ends (everything before is excluded),
    or None if none is detected. A manual override (cfg.gating.preamble_end_s) wins — the
    auto-detector is a heuristic and this is a Phase-1 VERIFY item.

    Auto-detection (robust to a quiet lead-in): the preamble is a large, mostly SINGLE-AXIS
    pitch swing somewhere in the first ~25 s, typically flanked by stationary periods (the
    robot settles, the ball is thrown), after which balanced multi-axis LOCOMOTION begins.
    We find the swing, then return the time sustained locomotion starts (everything before
    it — swing + settle + ball — is excluded). Heuristic: eyeball it and set
    `gating.preamble_end_s` to override.
    """
    import os
    g = cfg.get("gating", {})
    override = g.get("preamble_end_s")
    if override is not None:
        return float(override)

    imu = pu.parse_vectornav(os.path.join(cfg["sequence"]["data_root"],
                                          cfg["sequence"]["imu_file"]), cfg)
    if len(imu.t) < 30:
        return None
    t = imu.t - imu.t[0]
    win = 1.0
    edges = np.arange(0.0, float(t[-1]), win)
    if len(edges) < 4:
        return None
    wmean = np.zeros(len(edges)); dom = np.zeros(len(edges))
    for i, e in enumerate(edges):
        m = (t >= e) & (t < e + win)
        if m.sum() < 3:
            continue
        wmean[i] = np.linalg.norm(imu.omega[m], axis=1).mean()
        s = imu.omega[m].std(axis=0)
        dom[i] = s.max() / (s.sum() + 1e-9)          # 1 => single-axis, ~0.33 => balanced

    loco_level = np.median(wmean[wmean > 1e-6]) if np.any(wmean > 1e-6) else 0.0
    if loco_level <= 1e-6:
        return None
    loco_thr = 0.5 * loco_level
    scan = int(min(len(edges), g.get("preamble_max_scan_s", 25)))

    swings = [i for i in range(scan) if wmean[i] > loco_thr and dom[i] > 0.6]
    if not swings:
        return None                                   # no clear single-axis swing => no preamble

    hold = 3
    for i in range(swings[0] + 1, len(edges) - hold):
        # sustained, balanced motion => locomotion has begun
        if all(wmean[i + k] > loco_thr for k in range(hold)) and dom[i] < 0.55:
            return float(edges[i])
    return float(edges[min(swings[-1] + 1, len(edges) - 1)])   # fallback: just after the swing


# --------------------------------------------------------------------------- #
# TRAP 3 — IMU↔pose lag and clock-drift check
# --------------------------------------------------------------------------- #
def estimate_imu_pose_lag(cfg, t_lo: float, t_hi: float, dt: float = 1.0 / 200.0,
                          max_lag_s: float = 0.05) -> float:
    """Residual IMU↔pose time lag on [t_lo,t_hi] via ‖ω‖ cross-correlation (seconds).

    Near zero once the configured offsets are correct. A nonzero, window-dependent value is
    clock drift (see clock_drift_check).
    """
    grid = np.arange(t_lo, t_hi, dt)
    ti, mi = _imu_omega_mag(cfg)
    imu_mag = np.interp(grid, ti, mi)
    _tg, moc_mag = _mocap_omega_mag(cfg, grid)
    a = imu_mag - imu_mag.mean()
    b = moc_mag - moc_mag.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    max_lag = int(max_lag_s / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(np.roll(a, k), b) for k in lags])
    k = int(np.argmax(corr))
    # parabolic sub-sample peak refinement (removes the ±1-grid quantization on short
    # windows); falls back to the integer peak at the edges
    if 0 < k < len(corr) - 1:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        denom = (c0 - 2 * c1 + c2)
        frac = 0.5 * (c0 - c2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    return float((lags[k] + frac) * dt)


def clock_drift_check(cfg, window_s: float = 10.0) -> dict:
    """Compare IMU↔pose lag on the first vs last window; report drift (ms)."""
    import os
    imu = pu.parse_vectornav(os.path.join(cfg["sequence"]["data_root"],
                                          cfg["sequence"]["imu_file"]), cfg)
    t0, t1 = float(imu.t[0]), float(imu.t[-1])
    start = detect_sync_preamble(cfg) or t0            # skip the preamble
    w = min(window_s, (t1 - start) / 3.0)
    # cross-correlation on a short window is dominated by noise; only trust it with enough
    # signal (real CEAR sequences are tens of seconds -> ~10 s windows). Toy/short clips
    # report insufficient rather than a spurious drift.
    min_w = float(cfg.get("sync", {}).get("min_drift_window_s", 3.0))
    if w < min_w:
        return {"lag_first_ms": None, "lag_last_ms": None, "drift_ms": None,
                "note": f"insufficient duration for drift check (window {w:.1f}s < {min_w}s)"}
    lag_first = estimate_imu_pose_lag(cfg, start, start + w)
    lag_last = estimate_imu_pose_lag(cfg, t1 - w, t1)
    tol = float(cfg.get("sync", {}).get("drift_tolerance_ms", 2.0))
    drift_ms = (lag_last - lag_first) * 1000.0
    return {"lag_first_ms": lag_first * 1000.0, "lag_last_ms": lag_last * 1000.0,
            "drift_ms": drift_ms, "exceeds_tolerance": bool(abs(drift_ms) > tol),
            "tolerance_ms": tol}
