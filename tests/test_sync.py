"""Tests for the v3.1 sync/timestamp traps: TRAP 7 (separate depth timestamps),
TRAP 8 (sync-preamble exclusion), TRAP 3 (clock-drift check)."""

import os

import numpy as np

import pose_utils as pu
import gating_utils as gu
import sync_utils as su


# --------------------------------------------------------------------------- #
# TRAP 7 — RGB and depth timestamps differ
# --------------------------------------------------------------------------- #
def test_depth_timestamp_differs_from_rgb(synthetic):
    cfg, seq = synthetic["cfg"], synthetic["seq_dir"]
    frames = pu.parse_realsense_timestamps(
        os.path.join(seq, cfg["sequence"]["timestamp_table"]), cfg)
    gaps = np.array([f.t_depth - f.t for f in frames])
    # generator uses a 4 ms RGB->depth gap; must be parsed as a real, nonzero offset
    assert np.all(gaps > 1e-4)
    assert abs(np.median(gaps) - 0.004) < 1e-3


# --------------------------------------------------------------------------- #
# TRAP 8 — sync-preamble detection + exclusion
# --------------------------------------------------------------------------- #
def test_no_preamble_detected_on_clean_sequence(synthetic):
    assert su.detect_sync_preamble(synthetic["cfg"]) is None


def test_preamble_detected_and_excluded(synthetic_preamble):
    cfg, seq = synthetic_preamble["cfg"], synthetic_preamble["seq_dir"]
    end = su.detect_sync_preamble(cfg)
    assert end is not None
    assert 1.0 < end < 3.5, f"preamble end {end} not near the injected 2.0 s"

    frames = pu.parse_realsense_timestamps(
        os.path.join(seq, cfg["sequence"]["timestamp_table"]), cfg)
    ts = [f.t for f in frames]
    keep = np.ones(len(ts), dtype=bool)
    keep2, n_excl = gu.apply_preamble_exclusion(cfg, ts, keep)
    assert n_excl > 0
    # every kept frame is after the detected preamble end
    assert all(t > end for t, k in zip(ts, keep2) if k)


def test_manual_preamble_override(synthetic):
    import copy
    cfg = copy.deepcopy(synthetic["cfg"])
    cfg["gating"]["preamble_end_s"] = 999.0     # override wins over auto-detect
    assert su.detect_sync_preamble(cfg) == 999.0


# --------------------------------------------------------------------------- #
# TRAP 3 — clock-drift check (synthetic has no drift)
# --------------------------------------------------------------------------- #
def test_imu_pose_lag_near_zero(synthetic):
    cfg = synthetic["cfg"]
    # offsets are applied correctly by the parsers, so residual lag ~ 0 (within the
    # cross-correlation grid resolution of ~5 ms)
    lag = su.estimate_imu_pose_lag(cfg, 0.2, 1.6)
    assert abs(lag) <= 0.006, f"unexpected residual IMU<->pose lag: {lag*1000:.1f} ms"


def test_clock_drift_small(synthetic):
    d = su.clock_drift_check(synthetic["cfg"])
    if d["drift_ms"] is not None:
        assert abs(d["drift_ms"]) < 5.0
