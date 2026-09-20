"""Phase 2 — pose/frame alignment -> COLMAP sparse model (docs/PLAN.md §5). HIGHEST RISK.

Emits, under {out_root}/colmap/sparse/0/:
    cameras.txt   one PINHOLE camera (post-undistortion intrinsics)
    images.txt    per-frame world-to-camera pose (scalar-first quats) + filename
    points3D.txt  EMPTY placeholder (seeded by build_depth_map.py in Phase 3)
plus {out_root}/poses_meta.json (canonical event-clock timestamps, the transform chain
written out, convention notes, trajectory stats).

All conventions come from pose_utils / the config. This script only orchestrates.

    python scripts/build_poses.py --config configs/mocap1_well-lit_trot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu


def compute_world_cam(cfg: dict):
    """Rebuild everything Phase 2 needs WITHOUT writing files.

    Returns (kept_frames, Twc[N,4,4] camera-to-world Z-up, calib, interp, mocap,
    n_dropped). Shared by build(), verify_reprojection, and later phases so the pose
    computation is defined in exactly one place.
    """
    seq = cfg["sequence"]
    root = seq["data_root"]
    mocap = pu.parse_mocap(os.path.join(root, seq["pose_file"]), cfg)
    interp = pu.PoseInterpolator(mocap)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    frames = pu.parse_realsense_timestamps(os.path.join(root, seq["timestamp_table"]), cfg)

    # require BOTH the RGB and depth timestamps in range (TRAP 7 — they differ by up to
    # ~8 ms, so an edge frame can have one in range and the other just outside)
    in_rng = interp.in_range(np.array([f.t for f in frames])) & \
        interp.in_range(np.array([f.t_depth for f in frames]))
    kept = [f for f, ok in zip(frames, in_rng) if ok]
    n_dropped = int((~in_rng).sum())
    if not kept:
        raise RuntimeError("no RGB frames fall within the pose time range — check TRAP 2/3 "
                           "(units/offsets); the streams may be misaligned by 1e6 or an offset sign")
    Twc = pu.build_world_to_cam_track(mocap, interp, calib, cfg, [f.t for f in kept])
    return kept, Twc, calib, interp, mocap, n_dropped


def depth_poses(cfg: dict, kept, mocap, interp, calib) -> np.ndarray:
    """Camera-to-world (Z-up) poses for each frame at its DEPTH timestamp (TRAP 7).

    Depth is stored in the RGB optical frame, so the same extrinsic chain applies — only
    the interpolation time differs (t_depth, not t). Used by depth fusion (Phase 3) and
    the Gate-2 warp source so geometry is posed at the instant the depth was captured.
    """
    return pu.build_world_to_cam_track(mocap, interp, calib, cfg, [f.t_depth for f in kept])


def build(cfg: dict) -> dict:
    out_root = cfg["paths"]["out_root"]
    kept, Twc, calib, interp, mocap, n_dropped = compute_world_cam(cfg)
    frames = pu.parse_realsense_timestamps(
        os.path.join(cfg["sequence"]["data_root"], cfg["sequence"]["timestamp_table"]), cfg)

    # --- write COLMAP model ---
    fx, fy, cx, cy = iu.K_params(calib.K)
    camera = pu.ColmapCamera(id=1, model="PINHOLE", width=calib.width,
                             height=calib.height, params=[fx, fy, cx, cy])
    images = [pu.ColmapImage(id=i + 1, T_world_cam=Twc[i], camera_id=1, name=kept[i].rgb_name)
              for i in range(len(kept))]
    model_dir = os.path.join(out_root, "colmap", "sparse", "0")
    pu.write_colmap_model(model_dir, camera, images, points3D=None)  # points seeded in Phase 3

    # --- trajectory stats (§2.6) ---
    C = Twc[:, :3, 3]
    fwd = Twc[:, :3, 2]                                   # camera +z (forward) in world
    extent = (C.max(axis=0) - C.min(axis=0))
    meta = {
        "config": os.path.basename(getattr(cfg, "_path", "")) or None,
        "n_frames_total": len(frames),
        "n_frames_kept": len(kept),
        "n_frames_dropped_out_of_range": n_dropped,
        "pose_time_range_s": [interp.tmin, interp.tmax],
        "transform_chain": ("T_world_cam = up_axis(y->z) @ ( T_world_marker(interp @ t) "
                            "@ T_marker_cam )  ; COLMAP stores inv(T_world_cam) as "
                            "world->cam, scalar-first quats"),
        "conventions": {
            "pose_quat_order": cfg["quaternions"]["pose_order"],
            "imu_quat_order": cfg["quaternions"]["imu_order"],
            "colmap_quat_order": cfg["quaternions"]["colmap_order"],
            "world_up": cfg["frames"]["world_up"],
            "target_up": cfg["frames"]["target_up"],
            "pose_frame": cfg["frames"]["pose_frame"],
            "offsets_s": cfg["timestamps"]["offsets_s"],
        },
        "intrinsics_pinhole": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                               "width": calib.width, "height": calib.height},
        "trajectory": {
            "extent_m": extent.tolist(),
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))),
            "centroid_m": C.mean(axis=0).tolist(),
            "view_dir_std": fwd.std(axis=0).tolist(),
        },
        "frames": [{"id": i + 1, "t_event_s": kept[i].t, "rgb": kept[i].rgb_name,
                    "depth": kept[i].depth_name} for i in range(len(kept))],
        "model_dir": model_dir,
    }
    pu.save_json(os.path.join(out_root, "poses_meta.json"), meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    meta = build(cfg)
    t = meta["trajectory"]
    print(f"[build_poses] wrote COLMAP model: {meta['model_dir']}")
    print(f"[build_poses] frames kept {meta['n_frames_kept']}/{meta['n_frames_total']} "
          f"(dropped out-of-range: {meta['n_frames_dropped_out_of_range']})")
    print(f"[build_poses] trajectory extent (m): "
          f"{[round(x,3) for x in t['extent_m']]}  path_len={t['path_length_m']:.2f} m")
    print(f"[build_poses] view-dir std: {[round(x,3) for x in t['view_dir_std']]}")
    if max(t["extent_m"]) > 0 and min(t["extent_m"]) / max(t["extent_m"]) < 0.05:
        print("[build_poses] WARNING: trajectory is nearly 1-D — see §2.6 (caps novel views)")


if __name__ == "__main__":
    main()
