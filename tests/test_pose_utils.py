"""Unit tests for the trap-prone pose/frame/COLMAP math (docs/PLAN.md §2.3, §5).

Each TRAP gets an explicit test so a regression names the trap it reintroduced.
"""

import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pose_utils as pu


# --------------------------------------------------------------------------- #
# TRAP 1 — quaternion order (scalar-first vs scalar-last) really differs
# --------------------------------------------------------------------------- #
def test_quat_order_matters():
    # a non-symmetric rotation so the two orderings can't coincide
    R = Rotation.from_euler("xyz", [30, 40, 50], degrees=True).as_matrix()
    q_xyzw = Rotation.from_matrix(R).as_quat()          # scipy: scalar-last
    q_wxyz = np.array([q_xyzw[3], *q_xyzw[:3]])

    assert np.allclose(pu.quat_to_R(q_xyzw, "xyzw"), R)
    assert np.allclose(pu.quat_to_R(q_wxyz, "wxyz"), R)
    # feeding a scalar-last vector to a scalar-first parser must NOT recover R
    assert not np.allclose(pu.quat_to_R(q_xyzw, "wxyz"), R)


def test_quat_roundtrip_both_orders():
    R = Rotation.from_euler("zyx", [10, -20, 33], degrees=True).as_matrix()
    for order in ("wxyz", "xyzw"):
        assert np.allclose(pu.quat_to_R(pu.R_to_quat(R, order), order), R)


def test_non_unit_quat_rejected():
    with pytest.raises(AssertionError):
        pu.quat_to_R([0.0, 0.0, 0.0, 0.5], "xyzw")


# --------------------------------------------------------------------------- #
# invert_T / up-axis primitives
# --------------------------------------------------------------------------- #
def test_invert_T():
    T = pu.make_T(Rotation.from_euler("y", 25, degrees=True).as_matrix(), [1, 2, 3])
    assert np.allclose(pu.invert_T(T) @ T, np.eye(4), atol=1e-12)


def test_up_axis_roundtrip_and_floor():
    R_yz = pu.up_axis_R("y", "z")
    R_zy = pu.up_axis_R("z", "y")
    assert np.allclose(R_yz @ R_zy, np.eye(3))
    # a Y-up point on the floor (y==0) must land on the Z-up floor (z==0)
    p_yup = np.array([1.3, 0.0, 2.1])       # y is "up" and zero => on floor
    p_zup = R_yz @ p_yup
    assert abs(p_zup[2]) < 1e-9             # z is "up" and must be zero


# --------------------------------------------------------------------------- #
# TRAP 2 / 3 — timestamp units and per-sensor offsets
# --------------------------------------------------------------------------- #
def test_timestamp_units_and_offsets(synthetic):
    cfg = synthetic["cfg"]
    seq = synthetic["seq_dir"]

    imu = pu.parse_vectornav(os.path.join(seq, "vectornav.txt"), cfg)
    # IMU file is in microseconds; parsed times must be in a small seconds range (~[0,2])
    assert imu.t.min() >= -0.01 and imu.t.max() < 3.0
    # offset actually applied: t_event = t_us/1e6 + imu_offset
    raw0_us = float(open(os.path.join(seq, "vectornav.txt")).readlines()[1].split()[0])
    expected = raw0_us / 1e6 + cfg["timestamps"]["offsets_s"]["imu"]
    assert abs(imu.t[0] - expected) < 1e-9

    frames = pu.parse_realsense_timestamps(
        os.path.join(seq, cfg["sequence"]["timestamp_table"]), cfg)
    assert frames[0].rgb_name.endswith("_rgb.png")
    assert "depth" in frames[0].depth_name        # picked the aligned-depth column, not rgb
    assert frames[0].rgb_name != frames[0].depth_name


# --------------------------------------------------------------------------- #
# TRAP 4 — extrinsic chain direction; full recovery of the TRUE camera poses
# --------------------------------------------------------------------------- #
def _recover_world_cam(synthetic):
    cfg, seq = synthetic["cfg"], synthetic["seq_dir"]
    mocap = pu.parse_mocap(os.path.join(seq, "MoCap.txt"), cfg)
    interp = pu.PoseInterpolator(mocap)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    frames = pu.parse_realsense_timestamps(
        os.path.join(seq, cfg["sequence"]["timestamp_table"]), cfg)
    ts = [f.t for f in frames]
    return pu.build_world_to_cam_track(mocap, interp, calib, cfg, ts), frames


def test_full_chain_recovers_true_poses(synthetic):
    Twc, _frames = _recover_world_cam(synthetic)
    gt = synthetic["gt"]["frames"]
    max_t = max_r = 0.0
    for i, fr in enumerate(gt):
        T_true = np.array(fr["T_world_cam"])
        max_t = max(max_t, np.linalg.norm(T_true[:3, 3] - Twc[i][:3, 3]))
        dR = T_true[:3, :3].T @ Twc[i][:3, :3]
        max_r = max(max_r, np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    # residual is pure SLERP/LERP interpolation error; a chain/convention bug would be
    # orders of magnitude larger (degrees / metres), not sub-milliradian.
    assert max_t < 5e-4, f"translation error too large: {max_t}"
    assert max_r < 0.05, f"rotation error too large: {max_r}"


def test_wrong_pose_frame_breaks_recovery(synthetic):
    """Sanity: dropping the extrinsic (treating marker == cam) must degrade recovery,
    proving the extrinsic is actually doing work (not a no-op that hides a bug)."""
    import copy
    cfg = copy.deepcopy(synthetic["cfg"])
    seq = synthetic["seq_dir"]
    mocap = pu.parse_mocap(os.path.join(seq, "MoCap.txt"), cfg)
    interp = pu.PoseInterpolator(mocap)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    calib.T_marker_cam = np.eye(4)          # pretend there is no extrinsic
    frames = pu.parse_realsense_timestamps(
        os.path.join(seq, cfg["sequence"]["timestamp_table"]), cfg)
    Twc = pu.build_world_to_cam_track(mocap, interp, calib, cfg, [f.t for f in frames])
    gt = synthetic["gt"]["frames"]
    err = max(np.linalg.norm(np.array(gt[i]["T_world_cam"])[:3, 3] - Twc[i][:3, 3])
              for i in range(len(gt)))
    assert err > 1e-3          # ~5 cm marker offset must show up


# --------------------------------------------------------------------------- #
# Interpolation (SLERP/LERP), out-of-range handling
# --------------------------------------------------------------------------- #
def test_interpolator_out_of_range(synthetic):
    cfg, seq = synthetic["cfg"], synthetic["seq_dir"]
    interp = pu.PoseInterpolator(pu.parse_mocap(os.path.join(seq, "MoCap.txt"), cfg))
    with pytest.raises(ValueError):
        interp.at(interp.tmax + 1.0)


# --------------------------------------------------------------------------- #
# TRAP 1b — COLMAP writer/reader round-trip; world-to-camera scalar-first
# --------------------------------------------------------------------------- #
def test_colmap_roundtrip(synthetic, tmp_path):
    Twc, frames = _recover_world_cam(synthetic)
    calib = pu.load_calibration(synthetic["cfg"]["intrinsics"]["calib_file"],
                                synthetic["cfg"])
    cam = pu.ColmapCamera(id=1, model="PINHOLE", width=calib.width, height=calib.height,
                          params=[calib.K[0, 0], calib.K[1, 1], calib.K[0, 2], calib.K[1, 2]])
    imgs = [pu.ColmapImage(id=i + 1, T_world_cam=Twc[i], camera_id=1, name=frames[i].rgb_name)
            for i in range(len(frames))]
    mdir = str(tmp_path / "sparse")
    pu.write_colmap_model(mdir, cam, imgs)

    _cams, imgs2, _pts = pu.read_colmap_model(mdir)
    assert len(imgs2) == len(imgs)
    for a, b in zip(imgs, imgs2):
        assert np.allclose(a.T_world_cam, b.T_world_cam, atol=1e-6)


def test_colmap_stores_world_to_camera(synthetic, tmp_path):
    """images.txt must store the INVERSE (world->cam) of the camera-to-world pose."""
    Twc, frames = _recover_world_cam(synthetic)
    cam = pu.ColmapCamera(id=1, model="PINHOLE", width=320, height=240,
                          params=[190, 190, 160, 120])
    im = pu.ColmapImage(id=1, T_world_cam=Twc[0], camera_id=1, name=frames[0].rgb_name)
    mdir = str(tmp_path / "sparse")
    pu.write_colmap_model(mdir, cam, [im])

    line = [ln for ln in open(os.path.join(mdir, "images.txt"))
            if ln.strip() and not ln.startswith("#")][0].split()
    qw, qx, qy, qz = map(float, line[1:5])
    tx, ty, tz = map(float, line[5:8])
    R_cw = pu.quat_to_R([qw, qx, qy, qz], "wxyz")
    T_cam_world = pu.make_T(R_cw, [tx, ty, tz])
    # stored pose is world->cam == inverse of our camera-to-world
    assert np.allclose(T_cam_world, pu.invert_T(Twc[0]), atol=1e-6)
