"""Remove transient returns from the robot's own path in an accumulated LiDAR cloud (CPU).

Accumulating every scan bakes in things that were only there for a moment: the robot's body (its
legs and frame at < 1 m from the sensor) and people following it. Smeared along the trajectory,
they sit right in front of the camera at every other time, and leak into the init, depth targets
and colliders. A ground robot never walks through solid matter, so the corridor it drove through,
above the floor, is free space in a static scene: points there are dropped.

  corridor = within --radius-m of the camera trajectory (measured along the floor plane),
             from --min-height-m to --max-height-m above the floor (floor and its texture kept)

The floor plane and up direction come from the shared detector (floor_plane.py), not the frame's
declared axis.

    python scripts/clear_robot_path.py --cloud <pipeline>/depth_cloud.ply --gt-model <out>/colmap_train/sparse/0 \
        --out <dir>/lidar_static.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_plane import find_floor_plane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="metric cloud (.ply)")
    ap.add_argument("--gt-model", required=True, help="metric-frame camera model (trajectory)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius-m", type=float, default=0.3)
    ap.add_argument("--min-height-m", type=float, default=0.05)
    ap.add_argument("--max-height-m", type=float, default=1.0)
    args = ap.parse_args()
    import open3d as o3d, pycolmap
    from scipy.spatial import cKDTree

    C = np.array([np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(args.gt_model).images.values()])
    pc = o3d.io.read_point_cloud(args.cloud); X = np.asarray(pc.points)
    floor = find_floor_plane(pc, C)
    if floor is None:
        raise SystemExit("[clear_path] no floor plane below the cameras; refusing to guess the corridor")
    n, d, _ = floor
    h, hc = X @ n + d, C @ n + d
    dist, _ = cKDTree(C - np.outer(hc, n)).query(X - np.outer(h, n), workers=-1)   # distance along the floor
    drop = (dist < args.radius_m) & (h > args.min_height_m) & (h < args.max_height_m)
    out = pc.select_by_index(np.nonzero(~drop)[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, out)
    info = {"input_points": int(len(X)), "dropped": int(drop.sum()), "radius_m": args.radius_m,
            "height_band_m": [args.min_height_m, args.max_height_m], "camera_height_m": round(float(np.median(hc)), 3)}
    json.dump(info, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print(f"[clear_path] dropped {drop.sum():,} of {len(X):,} points in the robot's corridor -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
