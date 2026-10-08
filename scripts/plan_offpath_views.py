"""Plan off-path camera views for world-model fill (CPU): where a simulated robot may look from, but the
real robot did not.

From every --every-th recorded frame, four variants: shifted --lateral-m to the left and right (along the
camera's own horizontal axis) and turned --yaw-deg left and right (about the camera's up axis). A view is
dropped if its centre is closer than --clearance-m to the (dynamic-free) LiDAR cloud, so no camera sits
inside an object. Variants are written as four smooth runs in recording order (left run, right run,
turn-left run, turn-right run): ArtiFixer generates autoregressively and drifts over long, jumpy
sequences (docs/experiments/2026-10-08_dense_path.md).

Writes <out>.json (ArtiFixer trajectory) + <out>_dataset (COLMAP text dataset; placeholder images =
the source frame) via afx_trajectory.write_path.

    python scripts/plan_offpath_views.py --views <all-frames dataset> --cloud lidar_static.ply \
        --sim3-json <pipeline>/sim3.json --out <dir>/targets
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afx_trajectory import read_views, write_path
from sim3_utils import load_sim3


def rot_about(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.deg2rad(deg) * axis / np.linalg.norm(axis)).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--cloud", required=True, help="metric LiDAR cloud for the clearance check")
    ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--lateral-m", type=float, default=0.4)
    ap.add_argument("--yaw-deg", type=float, default=30.0)
    ap.add_argument("--clearance-m", type=float, default=0.25)
    args = ap.parse_args()
    import open3d as o3d
    from scipy.spatial import cKDTree

    s, R, t = load_sim3(args.sim3_json)
    tree = cKDTree(np.asarray(o3d.io.read_point_cloud(args.cloud).points))
    cams, views = read_views(args.views)
    cid = views[0][1]
    src = views[::args.every]
    runs = {"left": [], "right": [], "turn_left": [], "turn_right": []}
    for name, _, Rwc, C in src:
        x, up = Rwc[:, 0], -Rwc[:, 1]                       # OpenCV camera: x right, y down
        lat = args.lateral_m / s
        runs["left"].append((name, Rwc, C - lat * x))
        runs["right"].append((name, Rwc, C + lat * x))
        runs["turn_left"].append((name, rot_about(up, args.yaw_deg) @ Rwc, C))
        runs["turn_right"].append((name, rot_about(up, -args.yaw_deg) @ Rwc, C))
    path, kept = [], {}
    for k, run in runs.items():
        ok = [(n, Rw, C) for n, Rw, C in run if tree.query(s * R @ C + t)[0] >= args.clearance_m]
        kept[k] = len(ok); path += ok
    write_path(path, cid, cams[cid], args.views, args.out)
    json.dump({"source_frames": len(src), "views_per_run": kept, "total": len(path), **{k: v for k, v in vars(args).items()
               if k in ("every", "lateral_m", "yaw_deg", "clearance_m")}}, open(args.out + "_plan.json", "w"), indent=2)
    print(f"[plan] {len(src)} source frames -> {len(path)} off-path views {kept}", flush=True)


if __name__ == "__main__":
    main()
