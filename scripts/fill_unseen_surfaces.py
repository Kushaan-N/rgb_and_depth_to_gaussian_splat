"""Fill every surface a metric depth sensor measured but the splat is missing (CPU; optional stage).

Training prunes gaussians no camera supports, so surfaces only the LiDAR saw (floor in the middle of a loop,
backs of objects, walls behind the robot) end up as holes or grey speckle. Measured geometry is kept, and its
appearance comes from the surroundings the cameras did see:

  WHERE   points of the (dynamic-free) depth cloud, voxel-thinned to --cell, with no opaque splat gaussian
          within --covered-m, and that NO training camera sees in front of, or within --min-hidden-m behind, the
          splat's own surface (rendered depth per training view, eval_views.py --save-depth-dir): there the
          camera's version wins (real views untouched), and clutter just behind seen surfaces is not filled. Without depth renders only the coverage test applies.
  COLOUR  for each such point, a blend of the colours (SH DC of the nearest opaque splat gaussian) of the --k
          nearest covered points, weighted by distance and by normal agreement (wall colour does not bleed
          onto floor); view-dependent SH is left at zero (matte).
  SHAPE   one flat, opaque disc per point (in-plane sigma 0.6 x cell, normal from the cloud).

Output keeps the input splat's frame (COLMAP). Additive only; nothing is deleted.

    python scripts/fill_unseen_surfaces.py --splat-ply in.ply --cloud lidar_static.ply --sim3-json sim3.json \
        [--dataset <training frames, COLMAP text> --splat-depth-dir <dir>] --out-ply out.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model

C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat-ply", required=True)
    ap.add_argument("--cloud", required=True, help="metric depth cloud (.ply), ideally dynamic-free")
    ap.add_argument("--sim3-json", required=True, help="COLMAP -> metric Sim3")
    ap.add_argument("--dataset", default=None, help="training frames (COLMAP text) for the visibility test")
    ap.add_argument("--splat-depth-dir", default=None, help="<image>_splatdepth.npy per training frame")
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--cell", type=float, default=0.02, help="fill spacing (m)")
    ap.add_argument("--covered-m", type=float, default=0.04, help="splat gaussian this close = surface present")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--tol-rel", type=float, default=0.05, help="behind the splat surface by more = hidden")
    ap.add_argument("--min-hidden-m", type=float, default=0.15,
                    help="a point hidden by less than this behind a seen surface is clutter around it, not a missing surface")
    ap.add_argument("--opacity", type=float, default=0.95)
    args = ap.parse_args()
    import open3d as o3d
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    ply = PlyData.read(args.splat_ply); v = ply["vertex"].data
    G = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    Gm = s * (G @ R.T) + t                                                  # splat in metres
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    opq = op > args.min_opacity
    gtree = cKDTree(Gm[opq])

    pc = o3d.io.read_point_cloud(args.cloud).voxel_down_sample(args.cell)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * args.cell, max_nn=30))
    P = np.asarray(pc.points); N = np.asarray(pc.normals)
    dist, gi = gtree.query(P)
    covered = dist < args.covered_m
    print(f"[fill] {len(P):,} cloud points at {args.cell} m; {covered.mean():.1%} already on the splat's surface", flush=True)
    cand = np.nonzero(~covered)[0]

    if args.dataset and args.splat_depth_dir:                               # the camera's version wins where seen
        cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
        Xd = ((P[cand] - t) @ R) / s                                         # metric -> dataset frame
        seen = np.zeros(len(cand), bool)
        for name, cid, q, tc in imgs:
            f = os.path.join(args.splat_depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy")
            if not os.path.exists(f):
                continue
            W, H, (fx, fy, cx, cy) = cams[cid]
            Pc = Xd @ quat_to_R(q).T + tc; z = Pc[:, 2]
            ok = z > 0.05
            u = np.full(len(z), -1); vv = np.full(len(z), -1)
            u[ok] = np.floor(fx * Pc[ok, 0] / z[ok] + cx); vv[ok] = np.floor(fy * Pc[ok, 1] / z[ok] + cy)
            ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            ia = np.nonzero(ok & ~seen)[0]
            if not len(ia):
                continue
            D = np.load(f).astype(np.float32)[vv[ia], u[ia]]
            r = np.linalg.norm(Pc[ia], axis=1)
            # visible (nothing rendered / not behind the surface) or only just behind it (thickness, clutter)
            seen[ia[(D <= 0) | (r < D * (1 + args.tol_rel)) | (r < D + args.min_hidden_m / s)]] = True
        print(f"[fill] {len(cand):,} uncovered; {seen.sum():,} visible to a training camera (left to the camera)", flush=True)
        cand = cand[~seen]
    if not len(cand):
        os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
        PlyData([PlyElement.describe(v, "vertex")], text=False).write(args.out_ply)
        print("FILL_UNSEEN", json.dumps({"added": 0})); return

    # colour: covered points carry the colour of their nearest opaque gaussian
    cov = np.nonzero(covered)[0]
    col_cov = dc[np.nonzero(opq)[0][gi[cov]]]
    ctree = cKDTree(P[cov])
    dd, ii = ctree.query(P[cand], k=args.k)
    nagree = np.abs(np.einsum("nkd,nd->nk", N[cov][ii], N[cand]))            # 1 = same surface orientation
    w = np.exp(-dd / max(np.median(dd[:, 0]), args.cell)) * (nagree ** 4 + 1e-3)
    col = (w[..., None] * col_cov[ii]).sum(1) / w.sum(1, keepdims=True)

    # discs: tangent frame from the normal, sigma 0.6 cell in-plane, 1 mm thick (in COLMAP units)
    Nd = N[cand] @ R                                                         # normals into the dataset frame
    tan = np.cross(Nd, np.array([0.0, 0.0, 1.0])); bad = np.linalg.norm(tan, axis=1) < 1e-3
    tan[bad] = np.cross(Nd[bad], np.array([1.0, 0.0, 0.0])); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    bit = np.cross(Nd, tan)
    qq = Rotation.from_matrix(np.stack([tan, bit, Nd], 2)).as_quat()        # x y z w
    m = len(cand); new = np.zeros(m, dtype=v.dtype)
    X = ((P[cand] - t) @ R) / s
    new["x"], new["y"], new["z"] = X[:, 0], X[:, 1], X[:, 2]
    for i, k in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        new[k] = col[:, i]
    for name in v.dtype.names:
        if name.startswith("f_rest_"):
            new[name] = 0.0
    if "nx" in v.dtype.names:
        new["nx"], new["ny"], new["nz"] = Nd[:, 0], Nd[:, 1], Nd[:, 2]
    new["opacity"] = np.log(args.opacity / (1 - args.opacity))
    ls = np.log(np.array([0.6 * args.cell, 0.6 * args.cell, 0.001]) / s)
    new["scale_0"], new["scale_1"], new["scale_2"] = ls[0], ls[1], ls[2]
    new["rot_0"], new["rot_1"], new["rot_2"], new["rot_3"] = qq[:, 3], qq[:, 0], qq[:, 1], qq[:, 2]
    out = np.concatenate([v, new])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(args.out_ply)
    info = {"added": int(m), "cloud_points": int(len(P)), "covered_fraction": round(float(covered.mean()), 4),
            "cell_m": args.cell, "out": args.out_ply}
    print("FILL_UNSEEN", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
