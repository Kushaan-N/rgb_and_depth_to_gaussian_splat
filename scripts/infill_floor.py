"""Fill interior floor holes in a splat with floor a metric depth sensor measured (optional stage).

Floor no camera viewed head-on (e.g. a robot looping around a room facing outward) gets pruned in
training, leaving holes. Where a metric depth source (LiDAR cloud, or the fused RGB-D cloud) DID
measure that floor, this adds flat, matte gaussians there — additive, nothing is deleted:

  WHERE   interior holes only (enclosed by splat floor, not touching the room box edge), on cells
          with a measured floor surface and no tall object -> never paints over furniture/walls
  HEIGHT  the measured surface height of each cell (not an assumed plane)
  COLOUR  median of the real pixels that project onto the cell across every camera view that saw it
          (even grazing); pixel->SH mapping is fit on floor the splat already reconstructed. Cells
          no camera ever saw copy the nearest reconstructed floor gaussian.

Auto-skips (copies the splat through unchanged, prints INFILL {"skipped": ...}) when there is no
clear floor — the largest near-horizontal plane BELOW the cameras must hold >=2% of the points, so
walls (tilted) and ceilings (above the cameras) are never mistaken for floor — or no interior hole.
Assumes the pipeline's metric frame is Z-up. Output stays in the input splat's (COLMAP) frame.

    python scripts/infill_floor.py --depth-cloud <metric cloud .ply> --splat-ply in.ply \
        --colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0 \
        --images <out>/precond/rgb --out-ply out.ply
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_plane import find_floor_plane


def passthrough(args, reason):
    if os.path.abspath(args.splat_ply) != os.path.abspath(args.out_ply):
        os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
        shutil.copyfile(args.splat_ply, args.out_ply)
    print("INFILL", json.dumps({"skipped": reason, "added_gaussians": 0}))
    print(f"[infill] SKIP ({reason}); splat passed through -> {args.out_ply}")


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, d, Vt = np.linalg.svd(((dst - mu_d).T @ (src - mu_s)) / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = np.trace(np.diag(d) @ W) / (((src - mu_s) ** 2).sum() / len(src))
    return float(s), R, mu_d - s * R @ mu_s


def cam_pose(img):
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    M = np.asarray(cfw.matrix())
    return M[:, :3], M[:, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-cloud", "--lidar-cloud", dest="depth_cloud", required=True,
                    help="metric point cloud in the Z-up world frame (LiDAR or fused RGB-D)")
    ap.add_argument("--splat-ply", required=True)
    ap.add_argument("--colmap-model", required=True)
    ap.add_argument("--gt-model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--opacity", type=float, default=0.95)
    args = ap.parse_args()
    import cv2, open3d as o3d, pycolmap
    from plyfile import PlyData, PlyElement
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    Crec = pycolmap.Reconstruction(args.colmap_model); Grec = pycolmap.Reconstruction(args.gt_model)
    C = {i.name: np.asarray(i.projection_center()) for i in Crec.images.values()}
    G = {i.name: np.asarray(i.projection_center()) for i in Grec.images.values()}
    common = sorted(set(C) & set(G))
    s, R, t = umeyama(np.array([C[n] for n in common]), np.array([G[n] for n in common]))

    # floor plane + per-cell surface (metric Z-up world frame)
    pc = o3d.io.read_point_cloud(args.depth_cloud); L = np.asarray(pc.points)
    floor = find_floor_plane(pc, np.array([G[k] for k in G]))
    if floor is None:
        passthrough(args, "no clear floor plane below the cameras"); return
    n, d, finl = floor
    hL = L @ n + d
    fl = L[finl]
    lo = np.percentile(fl[:, :2], 2, axis=0) - 0.3
    hi = np.percentile(fl[:, :2], 98, axis=0) + 0.3
    nx, ny = np.ceil((hi - lo) / args.cell).astype(int)

    def ij_of(P):
        ij = np.floor((P[:, :2] - lo) / args.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        return ij, ok

    def count(P):
        g = np.zeros((nx, ny), int); ij, ok = ij_of(P)
        np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1)
        return g

    low = (hL > -0.05) & (hL < 0.30)
    lidar_low = count(L[low]) >= 3
    tall = count(L[(hL > 0.30) & (hL < 2.0)]) > 2
    # per-cell median surface height
    ij, ok = ij_of(L[low]); hlow = hL[low][ok]; flat = ij[ok, 0] * ny + ij[ok, 1]
    order = np.argsort(flat); flat, hlow = flat[order], hlow[order]
    uniq, start = np.unique(flat, return_index=True)
    hmed = np.zeros(nx * ny)
    for u, a, b in zip(uniq, start, np.r_[start[1:], len(flat)]):
        hmed[u] = np.median(hlow[a:b])
    hmed = hmed.reshape(nx, ny)

    # splat -> GT frame; reconstructed floor coverage
    ply = PlyData.read(args.splat_ply); v = ply["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    Xg = s * (X @ R.T) + t
    op_raw = np.asarray(v["opacity"], dtype=np.float64)
    logit = op_raw.min() < 0 or op_raw.max() > 1
    op = 1.0 / (1.0 + np.exp(-op_raw)) if logit else op_raw
    hS = Xg @ n + d
    fsel = (np.abs(hS) < 0.08) & (op > 0.2)
    cover = count(Xg[fsel]) > 0
    cover_d = ndimage.binary_dilation(cover, iterations=2)

    # interior holes: LiDAR floor, not reconstructed, not under tall objects, not touching the box edge
    holes = lidar_low & ~tall & ~cover_d
    lab, nlab = ndimage.label(holes)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    interior = np.isin(lab, [i for i in range(1, nlab + 1) if i not in edge])
    fill = ndimage.binary_dilation(interior, iterations=2) & lidar_low & ~tall & ~cover
    fi, fj = np.nonzero(fill)
    cx = lo[0] + (fi + 0.5) * args.cell; cy = lo[1] + (fj + 0.5) * args.cell
    z0 = -(n[0] * cx + n[1] * cy + d) / n[2]
    P = np.stack([cx, cy, z0], 1) + hmed[fi, fj][:, None] * n[None, :]     # on the LiDAR surface
    print(f"[infill] grid {nx}x{ny} @ {args.cell}m ; interior hole components={nlab-len(edge)} ; fill cells={len(P)}", flush=True)
    if not len(P):
        passthrough(args, "no interior floor holes"); return

    # colour: sample real pixels for fill cells AND reconstructed-floor cells (for calibration)
    ci, cj = np.nonzero(cover & lidar_low & ~tall)
    ccx = lo[0] + (ci + 0.5) * args.cell; ccy = lo[1] + (cj + 0.5) * args.cell
    Pc = np.stack([ccx, ccy, -(n[0] * ccx + n[1] * ccy + d) / n[2]], 1) + hmed[ci, cj][:, None] * n[None, :]
    Q = np.vstack([P, Pc]); nf = len(P)
    samp_idx, samp_rgb = [], []
    for img in Grec.images.values():
        cam = Grec.cameras[img.camera_id]
        fx, fy, px, py = [float(x) for x in cam.params[:4]]
        Rcw, tcw = cam_pose(img)
        Pcam = Q @ Rcw.T + tcw
        z = Pcam[:, 2]
        u = np.round(fx * Pcam[:, 0] / np.maximum(z, 1e-6) + px).astype(int)
        w = np.round(fy * Pcam[:, 1] / np.maximum(z, 1e-6) + py).astype(int)
        vis = (z > 0.1) & (z < 8.0) & (u >= 0) & (u < cam.width) & (w >= 0) & (w < cam.height)
        if not vis.any():
            continue
        im = cv2.imread(os.path.join(args.images, img.name), cv2.IMREAD_COLOR)
        if im is None:
            continue
        rgb = im[w[vis], u[vis], ::-1].astype(np.float64) / 255.0
        samp_idx.append(np.nonzero(vis)[0]); samp_rgb.append(rgb)
    si = np.concatenate(samp_idx); sr = np.concatenate(samp_rgb)
    order = np.argsort(si); si, sr = si[order], sr[order]
    uq, st = np.unique(si, return_index=True)
    med = np.full((len(Q), 3), np.nan); cnt = np.zeros(len(Q), int)
    for u_, a, b in zip(uq, st, np.r_[st[1:], len(si)]):
        med[u_] = np.median(sr[a:b], axis=0); cnt[u_] = b - a

    # calibrate pixel rgb -> f_dc on reconstructed floor cells (handles SH convention + exposure)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    gsel = fsel & (op > 0.5)
    gij, gok = ij_of(Xg[gsel]); gf = fdc[gsel][gok]
    key = gij[gok, 0] * ny + gij[gok, 1]
    acc = np.zeros((nx * ny, 3)); nacc = np.zeros(nx * ny)
    np.add.at(acc, key, gf); np.add.at(nacc, key, 1)
    ck = ci * ny + cj
    have = (nacc[ck] > 0) & (cnt[nf:] >= 5)
    Xc = med[nf:][have]; Yc = acc[ck][have] / nacc[ck][have][:, None]
    C0 = 0.28209479177387814
    a_ = np.zeros(3); b_ = np.zeros(3); r2 = np.zeros(3)
    for c in range(3):
        A = np.stack([Xc[:, c], np.ones(len(Xc))], 1)
        coef = np.linalg.lstsq(A, Yc[:, c], rcond=None)[0]
        res = np.abs(A @ coef - Yc[:, c]); keep = res < np.percentile(res, 90)
        coef = np.linalg.lstsq(A[keep], Yc[keep, c], rcond=None)[0]
        pred = A[keep] @ coef
        r2[c] = 1 - ((pred - Yc[keep, c]) ** 2).sum() / max(((Yc[keep, c] - Yc[keep, c].mean()) ** 2).sum(), 1e-9)
        a_[c], b_[c] = coef
    use_fit = bool(r2.min() > 0.3 and len(Xc) > 50)
    print(f"[infill] colour calib on {len(Xc)} cells: slope={a_.round(3).tolist()} int={b_.round(3).tolist()} "
          f"R2={r2.round(2).tolist()} (SH-C0 convention would be slope {1/C0:.3f}, int {-0.5/C0:.3f}) "
          f"-> {'pixel colours' if use_fit else 'fit too weak: nearest reconstructed floor colours'}", flush=True)

    # Pixel colours only through a fit that actually holds; the textbook SH-C0 conversion is NOT used
    # as a fallback because trainers disagree on the DC convention (3DGRUT's is ~3x off from C0).
    fill_fdc = np.zeros((nf, 3))
    seen = (cnt[:nf] >= 3) & use_fit
    fill_fdc[seen] = med[:nf][seen] * a_ + b_
    # never-seen cells (or no reliable fit): nearest reconstructed floor gaussian's colour
    if (~seen).any():
        tree = cKDTree(Xg[gsel][:, :2])
        _, nn = tree.query(P[~seen][:, :2])
        fill_fdc[~seen] = fdc[gsel][nn]
    print(f"[infill] colour from camera pixels: {int(seen.sum())} cells ; nearest-floor fallback: {int((~seen).sum())}", flush=True)

    # build flat matte gaussians in the splat (COLMAP) frame
    n_col = R.T @ n
    e1 = np.cross(n_col, [1.0, 0, 0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(n_col, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n_col, e1)
    q = Rotation.from_matrix(np.stack([e1, e2, n_col], 1)).as_quat()          # x,y,z,w
    Pcol = ((P - t) @ R) / s
    new = np.zeros(nf, dtype=v.dtype)
    new["x"], new["y"], new["z"] = Pcol[:, 0], Pcol[:, 1], Pcol[:, 2]
    for kk, val in (("nx", n_col[0]), ("ny", n_col[1]), ("nz", n_col[2])):
        if kk in v.dtype.names:
            new[kk] = val
    new["f_dc_0"], new["f_dc_1"], new["f_dc_2"] = fill_fdc[:, 0], fill_fdc[:, 1], fill_fdc[:, 2]
    new["opacity"] = np.log(args.opacity / (1 - args.opacity)) if logit else args.opacity
    in_plane = np.log(0.6 * args.cell / s); thin = np.log(0.003 / s)
    new["scale_0"], new["scale_1"], new["scale_2"] = in_plane, in_plane, thin
    new["rot_0"], new["rot_1"], new["rot_2"], new["rot_3"] = q[3], q[0], q[1], q[2]

    os.makedirs(os.path.dirname(args.out_ply) or ".", exist_ok=True)
    PlyData([PlyElement.describe(np.concatenate([v, new]), "vertex")], text=False).write(args.out_ply)
    summary = {"added_gaussians": int(nf), "from_pixels": int(seen.sum()), "fallback": int((~seen).sum()),
               "calib_R2": r2.round(3).tolist(), "used_fit": use_fit, "total": int(len(v) + nf)}
    print("INFILL", json.dumps(summary))
    print(f"[infill] wrote {args.out_ply}", flush=True)


if __name__ == "__main__":
    main()
