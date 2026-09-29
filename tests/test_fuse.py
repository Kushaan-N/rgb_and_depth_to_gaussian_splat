"""Fusion pose math: a member camera must land where its Sim3 + ICP correction put the scene."""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fuse_sequences import to_reference  # noqa: E402


def test_camera_follows_its_scene_points():
    rng = np.random.default_rng(0)
    s, R, t = 0.56, Rotation.random(random_state=1).as_matrix(), rng.normal(size=3)
    T = np.eye(4); T[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix(); T[:3, 3] = [0.03, -0.02, 0.01]
    Rwc, C = Rotation.random(random_state=2).as_matrix(), rng.normal(size=3)
    X = rng.normal(size=(20, 3)) * 3                              # scene points, COLMAP frame
    x_cam = (X - C) @ Rwc                                          # in the camera (COLMAP units)
    Rwc_f, C_f = to_reference(Rwc, C, s, R, t, T)
    X_f = (s * X @ R.T + t) @ T[:3, :3].T + T[:3, 3]               # same points, reference frame
    x_cam_f = (X_f - C_f) @ Rwc_f                                  # in the camera (metres)
    assert np.allclose(x_cam_f, s * x_cam)                         # same view, metric scale -> same pixels
    assert np.allclose(Rwc_f.T @ Rwc_f, np.eye(3))
