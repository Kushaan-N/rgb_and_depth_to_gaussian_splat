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

    Auto-detection: the preamble is a large, sustained pitch oscillation at the very start.
    In 1 s windows, flag those whose dominant-gyro-axis std exceeds `swing_factor` x the
    whole-sequence median; the preamble is the leading contiguous run of flagged windows.
    """
    g = cfg.get("gating", {})
    override = g.get("preamble_end_s")
    if override is not None:
        return float(override)

    t, _ = _imu_omega_mag(cfg)
    import os
    imu = pu.parse_vectornav(os.path.join(cfg["sequence"]["data_root"],
                                          cfg["sequence"]["imu_file"]), cfg)
    if len(imu.t) < 10:
        return None
    axis = int(np.argmax(imu.omega[: min(len(imu.t), 2000)].std(axis=0)))  # dominant early axis
    sig = imu.omega[:, axis]
    win = 1.0
    t0, t1 = float(imu.t[0]), float(imu.t[-1])
    edges = np.arange(t0, t1, win)
    stds = np.array([sig[(imu.t >= e) & (imu.t < e + win)].std() if
                     ((imu.t >= e) & (imu.t < e + win)).sum() > 3 else 0.0 for e in edges])
    if len(stds) < 3:
        return None
    # Baseline = steady-state locomotion, taken from the LATTER half of the sequence so it
    # is not inflated by the preamble itself (which sits at the very start).
    baseline = np.median(stds[len(stds) // 2:])
    if baseline <= 1e-9:
        baseline = np.median(stds)
    if baseline <= 1e-9:
        return None
    swing_factor = float(g.get("preamble_swing_factor", 2.0))
    flagged = stds > swing_factor * baseline
    if not flagged[0]:
        return None
    end_idx = 0
    while end_idx < len(flagged) and flagged[end_idx]:
        end_idx += 1
    return float(edges[0] + end_idx * win)


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
