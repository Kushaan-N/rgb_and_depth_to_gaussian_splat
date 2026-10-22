#!/usr/bin/env python3
"""Remove the 'camera-enveloping' gaussians of an outdoor splat (CPU venv): gaussians wider than --min-scale m whose
--k-sigma ellipsoid contains a training camera centre (tested exactly, in the gaussian's own frame).

Outdoor 3DGS fits the sky with a few hundred to thousands of huge gaussians (10 m - 2 km). Many are centred metres from
the cameras, i.e. every camera sits INSIDE them. SuperSplat never shows them (its screen-radius cap / near cull drops a
splat that covers the whole view), so they cost nothing in the editor, but every renderer without that cap (stock
gsplat, 3DGRUT, Isaac Sim's splat renderer) draws them as a near-camera fog over the whole frame (seen on ROQ-01:
expected depth ~4 m everywhere, washed-out renders). Removing exactly the camera-enveloping ones is a no-op for
SuperSplat by construction - the gate in sbatch/spires_score.sbatch (live SuperSplat on held-out frames) checks it - and
removes the fog elsewhere (eval_ply.py with stock gsplat measures that).

    python scripts/datasets/spires_sky.py --ply in.ply --cams <COLMAP text dataset(s)> --out out.ply [--k-sigma 3] \
        [--near-factor 4]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cams", nargs="+", required=True, help="datasets whose camera centres must not sit inside a splat")
    ap.add_argument("--min-scale", type=float, default=0.5); ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--near-factor", type=float, default=0.0,
                    help=">0: also drop candidates whose nearest camera is closer than this x their largest axis "
                         "(seen from there they span > ~2 atan(1/f): screen-filling haze, not surface)")
    a = ap.parse_args()
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    from rectify_views import read_images, qvec2rot
    C = np.concatenate([np.array([-qvec2rot(q).T @ t for _, q, t, _, _ in read_images(os.path.join(d, "sparse", "0", "images.txt"))])
                        for d in a.cams])
    ply = PlyData.read(a.ply); v = ply["vertex"].data
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    S = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    cand = np.where(S.max(1) > a.min_scale)[0]
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)[cand].astype(np.float64)          # wxyz
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]] / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()   # (n,3,3)
    drop = np.zeros(len(P), bool)
    for i0 in range(0, len(cand), 4096):                                             # Mahalanobis of every camera
        sl = slice(i0, i0 + 4096); d = C[None, :, :] - P[cand[sl], None, :]           # (b, ncam, 3)
        loc = np.einsum("bij,bcj->bci", np.transpose(R[sl], (0, 2, 1)), d) / S[cand[sl], None, :]
        drop[cand[sl]] = (np.linalg.norm(loc, axis=2) < a.k_sigma).any(1)
    if a.near_factor > 0:
        from scipy.spatial import cKDTree
        dn = cKDTree(C).query(P[cand])[0]; drop[cand[dn < a.near_factor * S[cand].max(1)]] = True
    PlyData([PlyElement.describe(v[~drop], "vertex")] + [e for e in ply.elements if e.name != "vertex"],
            text=False, comments=ply.comments).write(a.out)
    info = {"in": a.ply, "out": a.out, "gaussians": int(len(P)), "removed": int(drop.sum()), "candidates": int(len(cand)),
            "removed_median_scale_m": float(np.median(S[drop].max(1))) if drop.any() else 0.0, "cameras": int(len(C))}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_sky.json", "w"), indent=1)
    print("SKY", json.dumps(info))


if __name__ == "__main__":
    main()
