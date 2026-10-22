#!/usr/bin/env python3
"""Turn a trained splat into the INIT of another training run (CPU venv): a copy of a COLMAP text dataset whose
points3D.txt holds the splat's surface gaussians (opacity >= --min-opacity, largest axis <= --max-scale m; colour
from the SH DC term), optionally merged with the dataset's own points3D (LiDAR seeds), and whose images/ links every
file of the source images/ (frames, <stem>_depth.npy targets, masks), so train_viewer_gsplat.py / export_viewer.sbatch
(EDGS_DATA=<out>) train on the same frames from that geometry. Used for the Spires hybrid: native-fisheye 3DGUT
geometry (full field of view) as the init of the depth-supervised, viewer-matched pinhole recipe.

    python scripts/datasets/spires_ply_init.py --ply fish.ply --dataset <pinhole train (with depth targets)> --out DIR \
        [--min-opacity 0.3 --max-scale 1.0 --merge-seeds]
"""
from __future__ import annotations
import argparse, json, os, shutil
import numpy as np

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-opacity", type=float, default=0.3); ap.add_argument("--max-scale", type=float, default=1.0)
    ap.add_argument("--merge-seeds", action="store_true", help="keep the dataset's points3D (LiDAR seeds) too")
    a = ap.parse_args()
    from plyfile import PlyData
    v = PlyData.read(a.ply)["vertex"]
    P = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
    o = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    s = np.exp(np.stack([np.asarray(v[f"scale_{i}"], np.float64) for i in range(3)], 1)).max(1)
    rgb = np.clip((np.stack([np.asarray(v[f"f_dc_{i}"], np.float64) for i in range(3)], 1) * SH_C0 + 0.5) * 255, 0, 255)
    k = (o >= a.min_opacity) & (s <= a.max_scale); P, rgb = P[k], rgb[k].astype(np.uint8)
    sp, osp = os.path.join(a.dataset, "sparse", "0"), os.path.join(a.out, "sparse", "0")
    os.makedirs(osp, exist_ok=True); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    for f in ("cameras.txt", "images.txt"):
        shutil.copyfile(os.path.join(sp, f), os.path.join(osp, f))
    src = os.path.join(a.dataset, "images")
    for f in os.listdir(src):
        dst = os.path.join(a.out, "images", f)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(src, f)), dst)
    n = 0
    with open(os.path.join(osp, "points3D.txt"), "w") as fh:
        fh.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for p, c in zip(P, rgb):
            n += 1; fh.write(f"{n} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n")
        if a.merge_seeds and os.path.exists(os.path.join(sp, "points3D.txt")):
            for ln in open(os.path.join(sp, "points3D.txt")):
                q = ln.split()
                if len(q) >= 7 and not ln.startswith("#"):
                    n += 1; fh.write(f"{n} {' '.join(q[1:7])} 0\n")
    for f in ("depth_targets.json",):
        if os.path.exists(os.path.join(a.dataset, f)):
            shutil.copyfile(os.path.join(a.dataset, f), os.path.join(a.out, f))
    info = {"ply": a.ply, "splat_points": int(k.sum()), "of": int(len(k)), "total_points": n, "merge_seeds": a.merge_seeds}
    json.dump(info, open(os.path.join(a.out, "edgs_init.json"), "w"), indent=1)     # export_viewer checks this file
    print("PLY_INIT", json.dumps(info))


if __name__ == "__main__":
    main()
