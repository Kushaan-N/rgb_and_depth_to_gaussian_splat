"""Camera path for ArtiFixer from a set of views (CPU), optionally densified.

ArtiFixer generates frame by frame along a camera path; denser, smoother paths give generated views
that overlap more and agree more, so distilling them into 3D blurs less. This takes the views of a
COLMAP text dataset in recording order and inserts --densify interpolated poses (linear position,
slerp rotation) between consecutive views that are closer than --max-gap-m, and writes
  <out>.json       ArtiFixer's transforms-style trajectory (OpenGL camera-to-world, as its
                   prepare_colmap_artifixer_inputs.py writes them: inv(W2C) @ diag(1,-1,-1,1))
  <out>_dataset/   the same poses as a COLMAP text dataset (placeholder images = nearest real
                   view) so our 3DGRUT can render them and LiDAR depth / visibility can be computed

    python scripts/afx_trajectory.py --views <offpath>/test --out <dir>/dense3 --densify 3 --scale <m per unit>
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model
from fuse_sequences import write_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="COLMAP text dataset whose views define the path")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--densify", type=int, default=0, help="poses inserted between consecutive views")
    ap.add_argument("--max-gap-m", type=float, default=0.5)
    ap.add_argument("--scale", type=float, required=True, help="metres per dataset unit")
    args = ap.parse_args()
    from scipy.spatial.transform import Rotation, Slerp

    cams, imgs = read_text_model(os.path.join(args.views, "sparse", "0"))
    imgs = sorted(imgs)                                                   # names are timestamps: recording order
    cid = imgs[0][1]; W, H, (fx, fy, cx, cy) = cams[cid]
    poses = []
    for name, _, q, t in imgs:
        Rcw = quat_to_R(q); Rwc = Rcw.T; C = -Rwc @ t
        poses.append((name, Rwc, C))
    path = []
    for k, (name, Rwc, C) in enumerate(poses):
        path.append((name, Rwc, C))
        if k + 1 < len(poses) and args.densify > 0:
            n2, R2, C2 = poses[k + 1]
            if np.linalg.norm(C2 - C) * args.scale <= args.max_gap_m:
                sl = Slerp([0, 1], Rotation.from_matrix([Rwc, R2]))
                for j in range(1, args.densify + 1):
                    f = j / (args.densify + 1)
                    path.append((name, sl(f).as_matrix(), (1 - f) * C + f * C2))   # placeholder = earlier view
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = []
    for _, Rwc, C in path:
        c2w = np.eye(4); c2w[:3, :3] = Rwc; c2w[:3, 3] = C                 # OpenCV camera-to-world
        frames.append({"transform_matrix": (c2w @ flip).tolist()})        # = inv(W2C) @ flip
    traj = {"camera_model": "OPENCV", "w": W, "h": H, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "frames": frames}               # undistorted frames
    json.dump(traj, open(args.out + ".json", "w"), indent=1)
    ds = args.out + "_dataset"
    if os.path.isdir(ds):
        shutil.rmtree(ds)
    names = [f"traj_{i:05d}.png" for i in range(len(path))]
    write_model(os.path.join(ds, "sparse", "0"), {cid: ("PINHOLE", W, H, [fx, fy, cx, cy])},
                [(n, cid, Rwc, C) for n, (_, Rwc, C) in zip(names, path)])
    os.makedirs(os.path.join(ds, "images"))
    for n, (src, _, _) in zip(names, path):
        os.symlink(os.path.realpath(os.path.join(args.views, "images", src)), os.path.join(ds, "images", n))
    print(f"[trajectory] {len(poses)} views -> {len(path)} path frames (densify {args.densify}) -> {args.out}.json",
          flush=True)


if __name__ == "__main__":
    main()
