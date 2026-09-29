"""Sim3-align a COLMAP-frame 3DGS splat into the metric (GT/LiDAR) world frame.

COLMAP gives a much sharper splat (poses are photometrically bundle-adjusted) but in an arbitrary
scale/gauge. The LiDAR collider + Isaac need the metric mocap frame. We have BOTH a COLMAP pose and
a GT (OptiTrack) pose for every registered frame, so we solve a similarity transform (Umeyama /
Sim3) from the COLMAP camera centres to the GT camera centres and bake it into the .ply. The result
is the sharp splat, in the same metric frame as everything else.

    python scripts/sim3_align_splat.py --colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0 \
        --in-ply gaussians_colmap.ply --out-ply gaussians_colmap_metric.ply
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim3_utils import solve_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap-model", required=True)
    ap.add_argument("--gt-model", required=True)
    ap.add_argument("--in-ply", default=None, help="splat to transform (omit to only solve the Sim3)")
    ap.add_argument("--out-ply", default=None)
    ap.add_argument("--sim3-json", default=None, help="write {scale, R, t, residual} (COLMAP -> metric)")
    args = ap.parse_args()

    try:
        s, R, t, resid = solve_sim3(args.colmap_model, args.gt_model)
    except ValueError as e:
        print(f"[sim3] {e} — cannot align"); return 2
    print(f"[sim3] {len(resid)} correspondences  scale={s:.4f}  "
          f"residual mean={resid.mean()*1000:.1f}mm p95={np.percentile(resid,95)*1000:.1f}mm", flush=True)
    if args.sim3_json:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.sim3_json)), exist_ok=True)
        json.dump({"scale": s, "R": R.tolist(), "t": t.tolist(), "n_frames": len(resid),
                   "residual_mean_m": float(resid.mean()), "residual_p95_m": float(np.percentile(resid, 95))},
                  open(args.sim3_json, "w"), indent=2)
        print(f"[sim3] wrote {args.sim3_json}", flush=True)
    if not args.in_ply:
        return 0
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation

    ply = PlyData.read(args.in_ply); v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    xyz = (s * (R @ xyz.T).T + t)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # rotate gaussian orientations (quat stored w,x,y,z in 3DGS/3DGRUT ply)
    if all(f"rot_{i}" in v.dtype.names for i in range(4)):
        q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1)  # -> x,y,z,w for scipy
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        qn = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()  # x,y,z,w
        v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    # scale by s (scales are stored in log space)
    for c in ("scale_0", "scale_1", "scale_2"):
        if c in v.dtype.names:
            v[c] = (v[c].astype(np.float64) + np.log(s)).astype(v[c].dtype)
    # rotate normals if present
    if all(k in v.dtype.names for k in ("nx", "ny", "nz")):
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64) @ R.T
        v["nx"], v["ny"], v["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    # NOTE: higher-order SH (f_rest) directional lobes are not rotated (minor for a mostly-diffuse
    # indoor scene); DC colour (f_dc) and opacity are transform-invariant.
    os.makedirs(os.path.dirname(args.out_ply) or ".", exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(args.out_ply)
    print(f"[sim3] wrote {args.out_ply} ({os.path.getsize(args.out_ply)//1024//1024} MB, {len(v):,} gaussians)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
