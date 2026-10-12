"""Sim3-align a COLMAP-frame 3DGS splat into the metric (GT/LiDAR) world frame.

COLMAP gives a much sharper splat (poses are photometrically bundle-adjusted) but in an arbitrary
scale/gauge. The LiDAR collider + Isaac need the metric mocap frame. We have BOTH a COLMAP pose and
a GT (OptiTrack) pose for every registered frame, so we solve a similarity transform (Umeyama /
Sim3) from the COLMAP camera centres to the GT camera centres and bake it into the .ply. The result
is the sharp splat, in the same metric frame as everything else.

    python scripts/sim3_align_splat.py --colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0 \
        --in-ply gaussians_colmap.ply --out-ply gaussians_colmap_metric.ply
Any two COLMAP models with shared image names work (e.g. --gt-model = a reference reconstruction, to put a
splat into that reconstruction's axes). Positions, orientations, scales AND view-dependent colour (SH
bands 1-3) are transformed.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim3_utils import solve_sim3


def sh_basis(d):
    """Real SH basis of bands 1..3 (3DGS / 3DGRUT convention, sh_utils.eval_sh), d: (M,3) unit -> (M,15)."""
    C1 = 0.4886025119029199
    C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
          1.445305721320277, -0.5900435899266435]
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; xx, yy, zz = x * x, y * y, z * z
    return np.stack([-C1 * y, C1 * z, -C1 * x,
                     C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
                     C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                     C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                     C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], 1)


def rotate_sh(F, R):
    """SH coefficients (N,3,15) of a splat rotated by R: new colour(d) = old colour(R^T d), per band."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(512, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    Y, Yr = sh_basis(D), sh_basis(D @ R)                               # rows: d and R^T d
    out = np.empty_like(F)
    for a, b in ((0, 3), (3, 8), (8, 15)):                             # bands 1, 2, 3 rotate independently
        M = np.linalg.lstsq(Y[:, a:b], Yr[:, a:b], rcond=None)[0]       # Y M = Y(R^T d)
        out[:, :, a:b] = F[:, :, a:b] @ M.T
    return out


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
    # rotate the view-dependent colour (SH bands 1..3) with the frame; DC colour and opacity are invariant
    rest = [f"f_rest_{i}" for i in range(45)]
    if all(k in v.dtype.names for k in rest):
        F = np.stack([v[k] for k in rest], 1).astype(np.float64).reshape(len(v), 3, 15)   # channel-major (3DGS)
        F = rotate_sh(F, R)
        for i, k in enumerate(rest):
            v[k] = F.reshape(len(v), 45)[:, i].astype(v[k].dtype)
    os.makedirs(os.path.dirname(args.out_ply) or ".", exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(args.out_ply)
    print(f"[sim3] wrote {args.out_ply} ({os.path.getsize(args.out_ply)//1024//1024} MB, {len(v):,} gaussians)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
