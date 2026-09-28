"""Floor detection must be frame-agnostic: find the floor whether the frame is Z-up or Z-down,
reject tables / walls / ceilings, and return None when there is no floor."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
o3d = pytest.importorskip("open3d")
from floor_plane import find_floor_plane  # noqa: E402

rng = np.random.default_rng(0)


def _plane(n, xr, yr, z, noise=0.005):
    x = rng.uniform(*xr, n); y = rng.uniform(*yr, n)
    return np.stack([x, y, np.full(n, z) + rng.normal(0, noise, n)], 1)


def _wall(n, x, yr, zr):
    y = rng.uniform(*yr, n); z = rng.uniform(*zr, n)
    return np.stack([np.full(n, x) + rng.normal(0, 0.005, n), y, z], 1)


def _room(cam_h=0.4, ceil=2.9, floor=True):
    parts = [_plane(8000, (-2, 2), (-2, 2), 0.75)]                     # table top (smaller than floor)
    parts += [_wall(15000, -4, (-4, 4), (0, ceil)), _wall(15000, 4, (-4, 4), (0, ceil))]
    parts += [_plane(20000, (-4, 4), (-4, 4), ceil)]                   # ceiling
    if floor:
        parts += [_plane(40000, (-4, 4), (-4, 4), 0.0)]
    t = np.linspace(0, 2 * np.pi, 200)
    cams = np.stack([2.5 * np.cos(t), 2.5 * np.sin(t), np.full_like(t, cam_h)], 1)  # robot loop
    return np.concatenate(parts), cams


def test_floor_found_in_z_up_frame():
    P, cams = _room()
    n, d, inl = find_floor_plane(P, cams, verbose=False)
    assert n[2] > 0.99                                  # up = +z
    assert abs(d) < 0.02                                # floor at z = 0, not the table at 0.75


def test_floor_found_in_z_down_frame():
    P, cams = _room()
    P[:, 2] *= -1; cams[:, 2] *= -1                     # CEAR-style Z-down frame
    n, d, inl = find_floor_plane(P, cams, verbose=False)
    assert n[2] < -0.99                                 # up derived from data = -z
    assert abs(d) < 0.02


def test_tall_camera_low_ceiling_still_picks_floor():
    P, cams = _room(cam_h=1.5, ceil=2.4)                # ceiling only 0.9 m above the camera
    n, d, _ = find_floor_plane(P, cams, verbose=False)
    assert n[2] > 0.99 and abs(d) < 0.02


def test_no_floor_returns_none():
    P, cams = _room(floor=False)
    P = P[np.abs(P[:, 2] - 0.75) > 0.1]                 # also drop the table top: walls + ceiling only
    assert find_floor_plane(P, cams, verbose=False) is None
