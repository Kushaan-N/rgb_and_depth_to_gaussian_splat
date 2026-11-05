"""Object priors for the unseen sides of free-standing objects (CPU): geometric primitives, symmetry, appearance.
Post-process on any standard 3DGS .ply (e.g. the delivered <...>_solid3.ply); writes a candidate in
scripts/solidify_unseen.py's build format (base_kept.ply + textured.ply + meta.npz), so its refit / budget / shfit
stages gate it unchanged (seen views within the per-view budget, untouched rows byte-identical).

  fit+build (one command)  per object (find_objects.py; LiDAR points + the splat's camera-constrained gaussians =
           evidence, local frame = floor normal + minimum-area footprint rectangle):
    prim   PRIMITIVES fitted to the evidence, best by inlier share minus "surface claimed where a camera sees through":
           stepbox  = min-area rectangle footprint, top heights quantised to RANSAC'd horizontal levels (stacks, steps)
           plates   = sequential RANSAC tilted planes (15-80 deg: A-frames, ramps), each a rectangle with a thickness
           cylinder = vertical, RANSAC circle on the footprint
           then in the object's zone (only gaussians no training view constrains: support max blend w < --junk-w):
           REMOVE those inside the solid (> --inside-m) or floating off it (> --off-m, above the floor, not on another
           measured surface), SNAP spikes (s_max > --needle-ratio s_mid) near the surface into flat discs on it, ADD
           opaque flat discs on every primitive face sample no camera sees and the kept gaussians do not make solid.
           Disc colour: the object's own seen samples of the same face class (nearest in local coords).
    sym    SYMMETRY: vertical mirror planes (yaw scan) and 90/180 deg rotations about the vertical axis through the
           footprint centre, scored on the seen evidence (geometry within --sym-tol-m, colour agreement); the best one
           (if its inlier share >= --sym-min) copies the seen surface gaussians to where their image is on the
           primitive, unseen and not solid (position, rotation, scales, opacity; SH re-fitted for the reflected
           directions); discs are not added where a copy covers.
    app    APPEARANCE: remaining discs are textured from the same object's seen faces of the same class - walls tiled
           along the perimeter at the same height (brick courses continue), tops / plate sides tiled in 2D - instead of
           a flat nearest colour.
  Methods are cumulative flags: --method prim | prim,sym | prim,sym,app | prim,app.
  learned  --learned-dir GEN (scripts/object_priors/learned_prior.py): per object and seed, the generated gaussians
           (TRELLIS unit cube, +z up) are ALIGNED to the evidence (yaw scan x scale from the height, then trimmed ICP with
           yaw + scale + translation, evidence -> generated surface); the best (seed, pose) by residual + "generated
           surface a camera sees where nothing is"; colours calibrated (per-channel affine) on the seen part; only the
           generated gaussians on needed (unseen, not solid) primitive surface join, before the discs.

    python scripts/object_priors/object_priors.py build --ply in.ply --support S/support.npz --objects objects \
        --cloud lidar_static.ply --sim3-json sim3.json --dataset <train ds> <test ds> --splat-depth-dir D \
        --method prim,sym,app --out-dir OUT
  -> OUT/{base_kept,textured}.ply, OUT/meta.npz (solidify_unseen.py refit/budget input), OUT/priors.json, OUT/prims.png
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import (sim3, read_vertex, write_vertex, gauss_geom, load_objects,  # noqa: E402
                             solidity, disc_rows, C0)
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402

C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


def sh_basis_np(d, K):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]; xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    b = [np.full_like(x, C0), -C1 * y, C1 * z, -C1 * x, C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz,
         C2[4] * (xx - yy), C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
         C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(b[:1 + K], -1)


def sh_transform(rest, A, K):
    """SH bands >= 1 (N,K,3) of gaussians mapped by the orthogonal 3x3 A (dataset frame; reflection or rotation):
    new(d) = old(A^T d), least squares on 200 directions (exact for degree <= 3)."""
    if K == 0:
        return rest
    rng = np.random.default_rng(0); d = rng.normal(size=(200, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    B = sh_basis_np(d, K)[:, 1:]; Bo = sh_basis_np(d @ A, K)[:, 1:]       # (A^T d)_i = d @ A
    T = np.linalg.lstsq(B, Bo, rcond=None)[0]                              # B T = Bo  ->  new coeffs = T^T-mapped
    return np.einsum("jk,nkc->njc", T, rest)


def rect_frame(uv):
    """minimum-area bounding rectangle of 2D points -> angle, (lo, hi) in the rotated frame."""
    from scipy.spatial import ConvexHull
    h = uv[ConvexHull(uv).vertices]; e = np.diff(np.vstack([h, h[:1]]), axis=0)
    best = None
    for th in np.unique(np.round(np.mod(np.arctan2(e[:, 1], e[:, 0]), np.pi / 2), 6)):
        Rm = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]); q = uv @ Rm.T
        a = np.prod(q.max(0) - q.min(0))
        if best is None or a < best[0]:
            best = (a, th, q.min(0), q.max(0))
    return best[1:]


class Obj:
    """object frame: floor point c0, a1, a2 (min-area rectangle axes), n (up); local = (u, v, h above floor)."""
    def __init__(self, X, n, d, lo_pct=3.0, min_h=0.045):
        self.n, self.d = n, d
        e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1); h = X @ n + d; F = X[h > min_h] if (h > min_h).sum() > 20 else X
        base = F - np.outer(F @ n + d, n); uv = np.stack([base @ e1, base @ e2], 1)
        lo_, hi_ = np.percentile(uv, lo_pct, 0), np.percentile(uv, 100 - lo_pct, 0)
        uv = uv[np.all((uv >= lo_ - 0.05) & (uv <= hi_ + 0.05), 1)]
        th, lo, hi = rect_frame(uv)
        self.a1 = np.cos(th) * e1 + np.sin(th) * e2; self.a2 = np.cross(n, self.a1)
        mid = (lo + hi) / 2; Rm = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        cm = Rm.T @ mid; self.c0 = cm[0] * e1 + cm[1] * e2; self.c0 = self.c0 - (self.c0 @ n + d) * n
        self.half = (hi - lo) / 2

    def loc(self, P):
        r = P - self.c0; return np.stack([r @ self.a1, r @ self.a2, P @ self.n + self.d], 1)

    def met(self, L):
        return self.c0 + np.outer(L[:, 0], self.a1) + np.outer(L[:, 1], self.a2) + np.outer(L[:, 2], self.n)

    def vec(self, Vl):                                                      # local direction -> metric
        return np.outer(Vl[:, 0], self.a1) + np.outer(Vl[:, 1], self.a2) + np.outer(Vl[:, 2], self.n)


_CAMS, _DEP = {}, {}


def seen_by_cameras(Pm, datasets, depth_dir, s, R, t, tol_rel, solid_opacity, min_hidden_m, normals=None, max_incidence=90.0, thru=False):
    """solidify_unseen.seen_by_cameras with the per-frame rendered depth / opacity cached in memory (called many times).
    thru=True: only points a camera sees THROUGH (in front of its solid rendered surface by > tol_rel and 3 cm): a
    surface claimed there contradicts the images."""
    seen = np.zeros(len(Pm), bool)
    if not depth_dir or not len(Pm):
        return seen
    Xd = ((Pm - t) @ R) / s; Nd = normals @ R if normals is not None else None; cmin = np.cos(np.radians(max_incidence))
    for ds in datasets:
        if ds not in _CAMS:
            _CAMS[ds] = read_text_model(os.path.join(ds, "sparse", "0"))
        cams, imgs = _CAMS[ds]
        for name, cid, q, tc in imgs:
            f = os.path.join(depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy")
            if f not in _DEP:
                fo = f.replace("_splatdepth.npy", "_splatopacity.npy")
                _DEP[f] = (np.load(f).astype(np.float32), np.load(fo).astype(np.float16) if os.path.exists(fo) else None) if os.path.exists(f) else None
            if _DEP[f] is None:
                continue
            Dm, Am = _DEP[f]
            W, H, (fx, fy, cx, cy) = cams[cid]; Rc = quat_to_R(q)
            Pc = Xd @ Rc.T + tc; z = Pc[:, 2]
            ok = (z > 0.05) & ~seen
            u = np.full(len(z), -1); vv = np.full(len(z), -1)
            u[ok] = np.floor(fx * Pc[ok, 0] / z[ok] + cx); vv[ok] = np.floor(fy * Pc[ok, 1] / z[ok] + cy)
            ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            ia = np.nonzero(ok)[0]
            if not len(ia):
                continue
            D = Dm[vv[ia], u[ia]]; A = Am[vv[ia], u[ia]].astype(np.float32) if Am is not None else np.ones(len(ia), np.float32)
            r = np.linalg.norm(Pc[ia], axis=1)
            if thru:
                vis = (D > 0) & (A >= 0.5) & (r < D * (1 - tol_rel) - 0.03 / s)
            else:
                vis = (D <= 0) | (A < solid_opacity) | (r < D * (1 + tol_rel)) | (r < D + min_hidden_m / s)
            if Nd is not None:
                vis &= np.abs(np.einsum("nd,nd->n", Pc[ia] / r[:, None], Nd[ia] @ Rc.T)) >= cmin
            seen[ia[vis]] = True
    return seen


_IMG = {}


def gt_colours(Pm, datasets, depth_dir, s, R, t, tol_rel, min_hidden_m, min_views=2):
    """median real-image colour (rgb 0..1) at each metric point over the training views that see it (in front of / within
    min_hidden_m behind the input's rendered surface); NaN where fewer than min_views do."""
    from PIL import Image
    Xd = ((Pm - t) @ R) / s; obs = [[] for _ in range(len(Pm))]
    for ds in datasets:
        if ds not in _CAMS:
            _CAMS[ds] = read_text_model(os.path.join(ds, "sparse", "0"))
        cams, imgs = _CAMS[ds]
        for name, cid, q, tc in imgs:
            f = os.path.join(depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy"); fi = os.path.join(ds, "images", name)
            if not (os.path.exists(f) and os.path.exists(fi)):
                continue
            if f not in _DEP:
                fo = f.replace("_splatdepth.npy", "_splatopacity.npy")
                _DEP[f] = (np.load(f).astype(np.float32), np.load(fo).astype(np.float16) if os.path.exists(fo) else None)
            Dm = _DEP[f][0]; W, H, (fx, fy, cx, cy) = cams[cid]
            Pc = Xd @ quat_to_R(q).T + tc; z = Pc[:, 2]; ok = z > 0.05
            u = np.full(len(z), -1); vv = np.full(len(z), -1)
            u[ok] = np.floor(fx * Pc[ok, 0] / z[ok] + cx); vv[ok] = np.floor(fy * Pc[ok, 1] / z[ok] + cy)
            ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H); ia = np.nonzero(ok)[0]
            if not len(ia):
                continue
            D = Dm[vv[ia], u[ia]]; r = np.linalg.norm(Pc[ia], axis=1)
            vis = (D > 0) & ((r < D * (1 + tol_rel)) | (r < D + min_hidden_m / s))
            if not vis.any():
                continue
            if fi not in _IMG:
                _IMG[fi] = np.asarray(Image.open(fi).convert("RGB"))
            c = _IMG[fi][vv[ia[vis]], u[ia[vis]]] / 255.0
            for j, cc in zip(ia[vis], c):
                obs[j].append(cc)
    out = np.full((len(Pm), 3), np.nan)
    for j, o in enumerate(obs):
        if len(o) >= min_views:
            out[j] = np.median(np.array(o), 0)
    return out


# ---------------------------------------------------------------- primitives (local coordinates)
def fit_stepbox(L, half, cell, level_gap=0.03, min_share=0.05):
    """rectangle [-half, half] grid; per cell the 90th-percentile height of the evidence; heights quantised to levels."""
    from scipy import ndimage
    nu, nv = np.maximum(np.ceil(2 * half / cell).astype(int), 1)
    iu = np.clip(((L[:, 0] + half[0]) / cell).astype(int), 0, nu - 1); iv = np.clip(((L[:, 1] + half[1]) / cell).astype(int), 0, nv - 1)
    inside = (np.abs(L[:, 0]) <= half[0] + cell) & (np.abs(L[:, 1]) <= half[1] + cell) & (L[:, 2] > 0.02)
    H = np.zeros((nu, nv)); cnt = np.zeros((nu, nv), int)
    key = iu[inside] * nv + iv[inside]; hh = L[inside, 2]
    for k in np.unique(key):
        v = hh[key == k]
        if len(v) >= 2:
            H.flat[k] = np.percentile(v, 95); cnt.flat[k] = len(v)
    tops = np.sort(H[cnt > 0])
    if not len(tops):
        return None
    levels = top_levels(tops, level_gap, min_share)
    lvl = np.full((nu, nv), -1)
    m = cnt > 0
    lvl[m] = np.abs(H[m][:, None] - levels[None]).argmin(1)
    if (~m).any():                                                         # empty cells: nearest measured cell's level
        idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
        lvl = lvl[idx[0], idx[1]]
    lvl = ndimage.median_filter(lvl, 3, mode="nearest")
    return {"kind": "stepbox", "half": half.tolist(), "cell": cell, "levels": levels.tolist(), "grid": lvl}


def top_levels(tops, min_sep=0.03, min_share=0.05, kmax=4, gain=0.35):
    """discrete top heights from the cells' top heights: 1D k-means for k = 1..kmax, a level more only if it cuts the
    mean absolute deviation by > gain, levels >= min_sep apart and holding >= min_share of the cells."""
    best = np.array([np.median(tops)]); cost = np.abs(tops - best[0]).mean()
    for k in range(2, kmax + 1):
        c = np.quantile(tops, (np.arange(k) + 0.5) / k)
        for _ in range(30):
            a = np.abs(tops[:, None] - c[None]).argmin(1)
            c = np.array([np.median(tops[a == j]) if (a == j).any() else c[j] for j in range(k)])
        a = np.abs(tops[:, None] - c[None]).argmin(1); cc = np.abs(tops - c[a]).mean()
        share = np.bincount(a, minlength=k) / len(tops)
        if cc < (1 - gain) * cost and np.diff(np.sort(c)).min() >= min_sep and share.min() >= min_share:
            best, cost = np.sort(c), cc
        else:
            break
    return best


def stepbox_surface(pr, cell):
    """faces of the quantised height map: tops (class 'top', face 10+level) and walls (class 'wall', face 0..3)."""
    half = np.array(pr["half"]); g = pr["grid"]; lv = np.array(pr["levels"]); c = pr["cell"]
    Hc = lv[g]; nu, nv = g.shape; k = max(1, int(round(c / cell))); sub = (np.arange(k) + 0.5) / k
    pos, nor, face = [], [], []
    for i in range(nu):
        for j in range(nv):
            u0, v0 = -half[0] + i * c, -half[1] + j * c
            for su in sub:
                for sv in sub:
                    pos.append((u0 + su * c, v0 + sv * c, Hc[i, j])); nor.append((0, 0, 1)); face.append(10 + g[i, j])
            for f, (di, dj, vec) in enumerate(((1, 0, (1, 0, 0)), (-1, 0, (-1, 0, 0)), (0, 1, (0, 1, 0)), (0, -1, (0, -1, 0)))):
                ni, nj = i + di, j + dj
                h2 = Hc[ni, nj] if 0 <= ni < nu and 0 <= nj < nv else 0.0
                if Hc[i, j] - h2 < c / 2:
                    continue
                for s in sub:
                    eu = u0 + (1.0 if di == 1 else 0.0 if di == -1 else s) * c
                    ev = v0 + (1.0 if dj == 1 else 0.0 if dj == -1 else s) * c
                    for hh in np.arange(h2 + cell / 2, Hc[i, j], cell):
                        pos.append((eu, ev, hh)); nor.append(vec); face.append(f)
    return np.array(pos, float), np.array(nor, float), np.array(face)


def stepbox_inside(pr, L):
    half = np.array(pr["half"]); g = pr["grid"]; lv = np.array(pr["levels"]); c = pr["cell"]
    i = np.floor((L[:, 0] + half[0]) / c).astype(int); j = np.floor((L[:, 1] + half[1]) / c).astype(int)
    ok = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
    out = np.zeros(len(L), bool); out[ok] = (L[ok, 2] > 0) & (L[ok, 2] < lv[g[i[ok], j[ok]]])
    return out


def fit_plates(L, cell, thr=0.02, min_tilt=15, max_tilt=80, min_share=0.12, max_plates=4, raw_share=0.03,
               merge_deg=12.0, merge_m=0.10):
    """sequential RANSAC tilted planes (noisy LiDAR gives several parallel slices of one panel: slices with normals
    within merge_deg and offsets within merge_m are merged, the panel re-fitted by PCA), panels >= min_share kept."""
    import open3d as o3d
    E = L[L[:, 2] > 0.03]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(E)); rest = pc; raw = []
    for _ in range(16):
        if len(rest.points) < max(50, raw_share * len(E)):
            break
        pl, inl = rest.segment_plane(thr, 3, 3000)
        Q = np.asarray(rest.points)[inl]; m = np.array(pl[:3]); m /= np.linalg.norm(m)
        rest = rest.select_by_index(inl, invert=True)
        if len(inl) < raw_share * len(E):
            break
        if min_tilt <= np.degrees(np.arccos(abs(m[2]))) <= max_tilt:
            raw.append([Q, m])
    groups = []
    for Q, m in raw:
        for g in groups:
            if abs(m @ g[1]) > np.cos(np.radians(merge_deg)) and abs((Q.mean(0) - g[0].mean(0)) @ g[1]) < merge_m:
                g[0] = np.concatenate([g[0], Q]); break
        else:
            groups.append([Q, m])
    plates = []
    for Q, _ in sorted(groups, key=lambda g: -len(g[0])):
        if len(Q) < min_share * len(E) or len(plates) >= max_plates:
            continue
        cen = Q.mean(0); m = np.linalg.svd(Q - cen, full_matrices=False)[2][2]
        tilt = np.degrees(np.arccos(abs(m[2])))
        if not (min_tilt <= tilt <= max_tilt):
            continue
        m = m if m @ (cen - np.r_[0, 0, cen[2]]) >= 0 else -m               # facing away from the vertical axis
        p1 = np.cross([0, 0, 1.0], m); p1 /= np.linalg.norm(p1); p2 = np.cross(m, p1)
        q = np.stack([(Q - cen) @ p1, (Q - cen) @ p2], 1)
        lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        res = (Q - cen) @ m; thick = float(np.clip(2.0 * res.std(), 0.01, 0.04))
        plates.append({"centre": (cen + 0.5 * (lo[0] + hi[0]) * p1 + 0.5 * (lo[1] + hi[1]) * p2).tolist(), "m": m.tolist(),
                       "p1": p1.tolist(), "p2": p2.tolist(), "half": ((hi - lo) / 2).tolist(), "thick": thick,
                       "tilt_deg": round(float(tilt), 1), "inliers": int(len(Q))})
    if not plates:
        return None
    return {"kind": "plates", "plates": plates}


def plates_surface(pr, cell):
    pos, nor, face = [], [], []
    for k, p in enumerate(pr["plates"]):
        c, m, p1, p2, hf, th = (np.array(p[x]) for x in ("centre", "m", "p1", "p2", "half", "thick"))
        a = np.arange(-hf[0] + cell / 2, hf[0], cell); b = np.arange(-hf[1] + cell / 2, hf[1], cell)
        A, B = np.meshgrid(a, b, indexing="ij"); Q = c + np.outer(A.ravel(), p1) + np.outer(B.ravel(), p2)
        for sgn in (1, -1):
            P = Q + sgn * th / 2 * m; ok = P[:, 2] > 0.005
            pos.append(P[ok]); nor.append(np.tile(sgn * m, (ok.sum(), 1))); face.append(np.full(ok.sum(), 20 + 2 * k + (sgn < 0)))
    return np.concatenate(pos), np.concatenate(nor), np.concatenate(face)


def plates_inside(pr, L):
    out = np.zeros(len(L), bool)
    for p in pr["plates"]:
        c, m, p1, p2, hf, th = (np.array(p[x]) for x in ("centre", "m", "p1", "p2", "half", "thick"))
        r = L - c
        out |= (np.abs(r @ m) < th / 2) & (np.abs(r @ p1) <= hf[0]) & (np.abs(r @ p2) <= hf[1])
    return out


def fit_cylinder(L, half, thr=0.015, iters=300):
    E = L[L[:, 2] > 0.03]
    if len(E) < 30:
        return None
    rng = np.random.default_rng(0); best = None; xy = E[:, :2]
    for _ in range(iters):
        a, b, c = xy[rng.choice(len(xy), 3, replace=False)]
        A = 2 * np.array([b - a, c - a]); y = np.array([b @ b - a @ a, c @ c - a @ a])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        ctr = np.linalg.solve(A, y); r = np.linalg.norm(a - ctr)
        if not (0.03 < r < 0.6 * max(half.max(), 0.05)) or np.any(np.abs(ctr) > half):
            continue
        inl = np.abs(np.linalg.norm(xy - ctr, axis=1) - r) < thr
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), ctr, r)
    if best is None:
        return None
    _, ctr, r = best; top = float(np.percentile(E[:, 2], 95))
    return {"kind": "cylinder", "centre": ctr.tolist(), "r": float(r), "top": top}


def cylinder_surface(pr, cell):
    c, r, top = np.array(pr["centre"]), pr["r"], pr["top"]
    na = max(8, int(2 * np.pi * r / cell)); th = 2 * np.pi * (np.arange(na) + 0.5) / na
    hh = np.arange(cell / 2, top, cell); T, Hh = np.meshgrid(th, hh, indexing="ij")
    side = np.stack([c[0] + r * np.cos(T.ravel()), c[1] + r * np.sin(T.ravel()), Hh.ravel()], 1)
    sn = np.stack([np.cos(T.ravel()), np.sin(T.ravel()), 0 * T.ravel()], 1)
    g = np.arange(-r + cell / 2, r, cell); U, V = np.meshgrid(g, g, indexing="ij"); ok = U ** 2 + V ** 2 < r ** 2
    tp = np.stack([c[0] + U[ok], c[1] + V[ok], np.full(ok.sum(), top)], 1)
    return (np.concatenate([side, tp]), np.concatenate([sn, np.tile([0, 0, 1.0], (len(tp), 1))]),
            np.concatenate([np.zeros(len(side), int), np.full(len(tp), 10)]))


def cylinder_inside(pr, L):
    c, r, top = np.array(pr["centre"]), pr["r"], pr["top"]
    return (np.linalg.norm(L[:, :2] - c, axis=1) < r) & (L[:, 2] > 0) & (L[:, 2] < top)


SURF = {"stepbox": stepbox_surface, "plates": plates_surface, "cylinder": cylinder_surface}
INSIDE = {"stepbox": stepbox_inside, "plates": plates_inside, "cylinder": cylinder_inside}


def face_class(face):
    """0 wall, 1 top, 2 plate side (outer), 3 plate side (inner), 4 shadow floor patch under plates"""
    return np.where(face >= 40, 4, np.where(face >= 20, 2 + (face - 20) % 2, np.where(face >= 10, 1, 0)))


def carve(ob, hmax, cell, otree, seen_fn, frac=0.3):
    """grow each side of the footprint rectangle in cell steps while < frac of a strip's samples (3 heights + top) are
    seen by a training camera and none is on another measured surface; total size per axis <= max(other axis, height).
    Updates ob (c0, half) in place; returns the extension per side [[+u, -u], [+v, -v]] (m)."""
    ext = np.zeros((2, 2))
    for ax in (0, 1):
        o_ = 1 - ax; cap = max(2 * ob.half[o_], hmax) - 2 * ob.half[ax]
        if cap <= cell:
            continue
        for si, sg in enumerate((1, -1)):
            while ext[ax, si] + cell <= cap:
                c = ob.half[ax] + ext[ax, si] + cell / 2
                w = np.arange(-ob.half[o_] + cell / 2, ob.half[o_], cell)
                Ls = []
                for h in (0.25 * hmax, 0.5 * hmax, 0.75 * hmax, 0.95 * hmax):
                    L = np.zeros((len(w), 3)); L[:, ax] = sg * c; L[:, o_] = w; L[:, 2] = h; Ls.append(L)
                L = np.concatenate(Ls)
                if otree is not None and np.isfinite(otree.query(L, distance_upper_bound=0.03)[0]).any():
                    break
                if seen_fn(ob.met(L)).mean() > frac:
                    break
                ext[ax, si] += cell
        tot = ext[ax].sum()
        if tot > cap:
            ext[ax] *= cap / tot
    du, dv = (ext[0, 0] - ext[0, 1]) / 2, (ext[1, 0] - ext[1, 1]) / 2
    ob.c0 = ob.c0 + du * ob.a1 + dv * ob.a2; ob.half = ob.half + ext.sum(1) / 2
    return ext.round(3).tolist()


# ---------------------------------------------------------------- symmetry
def sym_candidates(step_deg=3):
    """(name, 3x3 local map A, applied about the footprint centre's vertical axis): mirrors through vertical planes with
    normal at yaw th, and rotations by 90/180/270 deg."""
    out = []
    for th in np.arange(0, 180, step_deg):
        m = np.array([np.cos(np.radians(th)), np.sin(np.radians(th)), 0.0]); out.append((f"mirror{int(th)}", np.eye(3) - 2 * np.outer(m, m)))
    for k in (1, 2, 3):
        a = k * np.pi / 2; out.append((f"rot{90 * k}", np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])))
    return out


def score_sym(Ls, cols, A, tol, ev_r=0.08):
    """inlier share of the seen evidence under the map A (only points whose image lands where there is evidence)."""
    from scipy.spatial import cKDTree
    T = cKDTree(Ls); M = Ls @ A.T
    d, i = T.query(M, distance_upper_bound=ev_r)
    ev = np.isfinite(d)
    if ev.sum() < 0.2 * len(Ls):
        return 0.0, float(ev.mean()), 9.0
    inl = d[ev] < tol
    cd = float(np.abs(cols[ev][inl] - cols[i[ev][inl]]).mean()) if inl.any() else 9.0
    return float(inl.mean()), float(ev.mean()), cd


# ---------------------------------------------------------------- learned prior alignment
def quat_wxyz_to_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.c_[q[:, 1:], q[:, :1]]).as_matrix()


def fit_yaw_sim(Pg, Pe):
    """least squares s Rz(yaw) Pg + t ~ Pe (yaw about +z only) -> (s, Rz, t)."""
    mg, me = Pg.mean(0), Pe.mean(0); A, B = Pg - mg, Pe - me
    c = (A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1]).sum(); sn = (A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum()
    th = np.arctan2(sn, c); Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    sc = float((B * (A @ Rz.T)).sum() / max((A * A).sum(), 1e-12))
    return sc, Rz, me - sc * Rz @ mg


def align_generated(files, Lev, cev, hmax, ob, seen_fn, trim=0.8, iters=25):
    """generated unit-cube gaussians -> object-local metric pose by a yaw scan + trimmed ICP (evidence -> generated)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0); E = Lev[Lev[:, 2] > 0.02]
    E = E[rng.choice(len(E), min(3000, len(E)), replace=False)]
    best, rep = None, []
    for f in files:
        g = np.load(f); X = g["xyz"].astype(np.float64); op = g["opacity"]
        G = X[op > 0.3]; G = G[rng.choice(len(G), min(6000, len(G)), replace=False)]
        z0, z1 = np.percentile(G[:, 2], [0.5, 99.5]); s0 = hmax / max(z1 - z0, 1e-3)
        for yaw in np.radians(np.arange(0, 360, 15)):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
            for sm in (0.85, 1.0, 1.15):
                sc = s0 * sm; Gc = (G - np.r_[np.median(G[:, :2], 0), z0]) @ Rz.T * sc; tt = np.zeros(3)
                Rc = np.eye(3)
                for _ in range(iters):
                    Gt = (sc * Gc @ Rc.T) + tt
                    d, i = cKDTree(Gt).query(E)
                    keep = d <= np.quantile(d, trim)
                    sc2, Rz2, t2 = fit_yaw_sim(Gc[i[keep]], E[keep])
                    sc2 = float(np.clip(sc2 / sc, 0.7, 1.4)) * sc if sc2 > 0 else sc
                    sc, Rc, tt = sc2, Rz2, t2
                Gt = sc * Gc @ Rc.T + tt; d = cKDTree(Gt).query(E)[0]
                res = float(np.mean(np.sort(d)[:int(trim * len(d))]))
                cand = (res, f, Rc @ Rz, sc * sm / sm, tt, np.r_[np.median(G[:, :2], 0), z0], s0, sm)
                if best is None or res < best[0][0]:
                    best = [cand] + ([] if best is None else best[:2])
                elif len(best) < 3 or res < best[-1][0]:
                    best = sorted(best + [cand], key=lambda c: c[0])[:3]
    # final pick among the 3 lowest residuals: + share of generated surface shown in a training view far from evidence
    et = cKDTree(Lev); pick = None
    for res, f, Rt, sc, tt, off, s0, sm in best:
        g = np.load(f); X = g["xyz"].astype(np.float64)
        L = sc * (X - off) @ Rt.T + tt
        sub = np.nonzero(g["opacity"] > 0.3)[0][::5]
        far = et.query(L[sub])[0] > 0.03
        contra = float((seen_fn(ob.met(L[sub[far]])).sum()) / max(len(sub), 1)) if far.any() else 0.0
        score = res / 0.01 + 5 * contra
        rep.append({"file": os.path.basename(f), "resid_m": round(res, 4), "scale_m_per_unit": round(sc, 3), "contradicted": round(contra, 3),
                    "score": round(score, 3)})
        if pick is None or score < pick[0]:
            pick = (score, f, Rt, sc, tt, off, L)
    _, f, Rt, sc, tt, off, L = pick; g = np.load(f)
    Rm = np.stack([ob.a1, ob.a2, ob.n], 1) @ Rt                              # generated frame -> metric
    xyz = ob.met(L); rot = Rm @ quat_wxyz_to_R(g["rot"]); scale = g["scale"] * sc
    dc = g["dc"].astype(np.float64)
    d, i = cKDTree(Lev).query(L)                                             # colour calibration on the seen part
    m = (d < 0.015) & (g["opacity"] > 0.3)
    cal = None
    if m.sum() > 50:
        rg, re_ = 0.5 + C0 * dc[m], 0.5 + C0 * cev[i[m]]
        cal = []
        for c in range(3):
            A_ = np.c_[rg[:, c], np.ones(m.sum())]; gb = np.linalg.lstsq(A_, re_[:, c], rcond=None)[0]
            gb[0] = np.clip(gb[0], 0.5, 2.0); cal.append(gb.tolist())
        rgb = 0.5 + C0 * dc
        rgb = np.stack([cal[c][0] * rgb[:, c] + cal[c][1] for c in range(3)], 1)
        dc = (np.clip(rgb, 0, 1) - 0.5) / C0
    return {"xyz": xyz, "rot": rot, "scale": scale, "opacity": g["opacity"], "dc": dc,
            "report": {"chosen": os.path.basename(f), "candidates": rep, "colour_affine": cal, "calib_pairs": int(m.sum())}}


# ---------------------------------------------------------------- build
def build(args):
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import open3d as o3d
    t0 = time.time(); meth = set(args.method.split(","))
    s, R, t = sim3(args.sim3_json)
    v = read_vertex(args.ply); N = len(v)
    Gm, op, sc = gauss_geom(v, s, R, t)
    ss = np.sort(sc, 1); smin, smid, smax = ss[:, 0], ss[:, 1], ss[:, 2]
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    nrest = len([k for k in v.dtype.names if k.startswith("f_rest_")]); K = nrest // 3
    sup = np.load(args.support); wmax = sup["wmax"]; assert len(wmax) == N, "support was computed for another PLY"
    q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    ometa, opts = load_objects(args.objects)
    assert ometa is not None and len(opts), "no objects"
    n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    cv = o3d.io.read_point_cloud(args.cloud).voxel_down_sample(0.01); Lall = np.asarray(cv.points)
    gtree = cKDTree(Gm)
    keep = np.ones(N, bool); mod = np.zeros(N, bool); vmod = v.copy(); zone_in = np.zeros(N, bool)
    new_rows, meta_rows = [], []                                          # appended rows + (obj, face, local, metric, normal)
    gen_dbg_extra = {}
    rep = {"method": sorted(meth), "objects": []}; gen_dbg = []
    for k, (o, X) in enumerate(zip(ometa["objects"], opts)):
        ob = Obj(X, n_f, d_f); Lx = ob.loc(X)
        hmax = float(np.percentile(Lx[:, 2], 99))
        Lg = ob.loc(Gm); zh = ob.half + args.zone_m                        # 2nd pass: frame from LiDAR + seen gaussians
        z0 = np.nonzero((np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > args.ev_min_h) & (Lg[:, 2] < hmax + 0.05)
                        & (wmax >= args.seen_w) & (op > 0.5) & (smax < args.needle_ratio * smid))[0]
        ob = Obj(np.concatenate([X, Gm[z0]]), n_f, d_f, min_h=args.ev_min_h); Lx = ob.loc(X)
        # depth prior by space carving: a side only one face of which was measured extends while no camera sees through
        # the strip and no other measured surface is there, up to a box no deeper than max(other side, height)
        Ll0 = ob.loc(Lall); nr = (np.abs(Ll0[:, 0]) < 1.5) & (np.abs(Ll0[:, 1]) < 1.5) & (Ll0[:, 2] > 0.03)
        oth0 = Ll0[nr][~np.isfinite(cKDTree(Lx).query(Ll0[nr], distance_upper_bound=0.03)[0])]
        otree0 = cKDTree(oth0) if len(oth0) else None
        ext = carve(ob, hmax, args.cell, otree0, lambda P: seen_by_cameras(P, args.dataset, args.splat_depth_dir, s, R, t,
                    args.tol_rel, args.solid_opacity, args.min_hidden_m, thru=True), args.carve_frac)
        Lx = ob.loc(X)
        zh = ob.half + args.zone_m
        Lg = ob.loc(Gm)
        inz = np.nonzero((np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > -0.02) & (Lg[:, 2] < hmax + args.zone_m))[0]
        zone_in[inz] = True
        strong = inz[(wmax[inz] >= args.seen_w) & (op[inz] > 0.5) & (smax[inz] < args.needle_ratio * smid[inz]) & (Lg[inz, 2] > args.ev_min_h)]
        Lev = np.concatenate([Lx[Lx[:, 2] > args.ev_min_h], Lg[strong]])
        # other measured surfaces near the object (walls, shelves): never cleaned / filled there
        Ll = ob.loc(Lall); near = (np.abs(Ll[:, 0]) <= zh[0] + 0.1) & (np.abs(Ll[:, 1]) <= zh[1] + 0.1) & (Ll[:, 2] > 0.03)
        own = cKDTree(Lx).query(Ll[near], distance_upper_bound=0.03)[0]
        other = Ll[near][~np.isfinite(own)]
        otree = cKDTree(other) if len(other) else None
        # candidates; selection = evidence inlier share - share of claimed surface a camera sees but has no evidence near
        cands = [fit_stepbox(Lev, ob.half, args.cell), fit_plates(Lev, args.cell), fit_cylinder(Lev, ob.half)]
        etree = cKDTree(Lev); best = None; scores = {}
        for pr in cands:
            if pr is None:
                continue
            Ps, Ns, Fs = SURF[pr["kind"]](pr, args.cell)
            if not len(Ps):
                continue
            inl = float(((cKDTree(Ps).query(Lev)[0] < args.inlier_m) | INSIDE[pr["kind"]](pr, Lev)).mean())   # on or in the solid
            noev = etree.query(Ps)[0] > 2 * args.inlier_m
            Pm = ob.met(Ps); Nm = ob.vec(Ns)
            sub_ = np.nonzero(noev)[0][::4]
            seen_ne = seen_by_cameras(Pm[sub_], args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity,
                                      args.min_hidden_m, thru=True)
            fake = float(seen_ne.sum() * 4 / len(Ps))
            sc_ = inl - args.lam_fake * fake - (args.cyl_penalty if pr["kind"] == "cylinder" else 0.0)
            scores[pr["kind"]] = {"inlier": round(inl, 3), "seen_through": round(fake, 3), "score": round(sc_, 3)}
            if best is None or sc_ > best[0]:
                best = (sc_, pr, Ps, Ns, Fs)
        orep = {"id": o["id"], "scores": scores, "zone_gaussians": int(len(inz)), "evidence": int(len(Lev)),
                "carved_extension_m": ext, "half_m": ob.half.round(3).tolist()}
        if best is None:
            rep["objects"].append(orep); continue
        _, pr, Ps, Ns, Fs = best
        orep["model"] = {kk: (vv if kk != "grid" else None) for kk, vv in pr.items()}
        if args.cosmetic:                                                  # shape-aware cleaning zone, floor patch under plates
            mg = args.cos_zone_plates_m if pr["kind"] == "plates" else args.cos_zone_box_m; zb = ob.half + mg
            inz = np.nonzero((np.abs(Lg[:, 0]) <= zb[0]) & (np.abs(Lg[:, 1]) <= zb[1]) & (Lg[:, 2] > -0.02) & (Lg[:, 2] < hmax + mg))[0]
            zone_in[inz] = True
            if pr["kind"] == "plates":
                from scipy.spatial import Delaunay
                cor = []
                for p_ in pr["plates"]:
                    c, p1, p2, hf = (np.array(p_[x]) for x in ("centre", "p1", "p2", "half"))
                    cor += [c + a1_ * hf[0] * p1 + a2_ * hf[1] * p2 for a1_ in (-1, 1) for a2_ in (-1, 1)]
                cor = np.array(cor)[:, :2]; g_ = np.arange(-1.5, 1.5, args.cell)
                U, V = np.meshgrid(g_, g_, indexing="ij"); Q = np.c_[U.ravel(), V.ravel()]
                Q = Q[Delaunay(cor).find_simplex(Q) >= 0]
                Ps = np.concatenate([Ps, np.c_[Q, np.full(len(Q), 0.006)]]); Ns = np.concatenate([Ns, np.tile([0, 0, 1.0], (len(Q), 1))])
                Fs = np.concatenate([Fs, np.full(len(Q), 40)]); orep["floor_patch"] = int(len(Q))
        # zone gaussians vs the primitive (only ones no training view constrains)
        Ltr = cKDTree(Ps); dS, iS = Ltr.query(Lg[inz]); ins = INSIDE[pr["kind"]](pr, Lg[inz])
        weak = wmax[inz] < args.junk_w
        prot = (Lg[inz, 2] < args.floor_m) | (np.isfinite(otree.query(Lg[inz], distance_upper_bound=args.other_m)[0]) if otree else False)
        rm = weak & ~prot & ((ins & (dS > args.inside_m)) | (~ins & (dS > args.off_m)))
        spk = (wmax[inz] < (args.snap_w if args.snap_w is not None else args.junk_w)) & ~prot & ~rm & (smax[inz] > args.needle_ratio * smid[inz]) & (op[inz] > 0.05) & (dS <= args.off_m)
        if args.no_clean:                                                  # add-only candidate (removals / snaps gated separately)
            rm[:] = False; spk[:] = False
        keep[inz[rm]] = False
        si = inz[spk]
        if "prim" in meth and len(si):
            Pn = ob.met(Ps[iS[spk]]); Nn = ob.vec(Ns[iS[spk]])
            disc = disc_rows(v.dtype, ((Pn - t) @ R) / s, Nn @ R, dc[si], args.cell, s, 0.5)
            for kk in ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3") + (("nx", "ny", "nz") if "nx" in v.dtype.names else ()):
                vmod[kk][si] = disc[kk]
            ls = np.log(np.stack([smid[si], smid[si], np.maximum(smin[si], 0.001)], 1) / s)
            for i in range(3):
                vmod[f"scale_{i}"][si] = ls[:, i]
            mod[si] = True
        orep.update({"removed": int(rm.sum()), "removed_inside": int((rm & ins).sum()), "snapped": int(spk.sum()) if "prim" in meth else 0,
                     "weak_in_zone": int(weak.sum()), "protected": int((weak & prot).sum())})
        # face samples: seen?  solid in the kept gaussians?
        Pm = ob.met(Ps); Nm = ob.vec(Ns)
        okf = ~(np.isfinite(otree.query(Ps, distance_upper_bound=args.other_m)[0]) if otree else np.zeros(len(Ps), bool))
        seen = seen_by_cameras(Pm, args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity,
                               args.min_hidden_m, normals=Nm, max_incidence=args.fill_max_incidence)
        kz = inz[keep[inz]]
        Gk, _, sck = gauss_geom(vmod[kz], s, R, t)
        qk = np.stack([vmod["rot_1"][kz], vmod["rot_2"][kz], vmod["rot_3"][kz], vmod["rot_0"][kz]], 1).astype(np.float64)
        Rk = R @ Rotation.from_quat(qk / np.maximum(np.linalg.norm(qk, axis=1, keepdims=True), 1e-12)).as_matrix()
        Sinv = np.einsum("nij,nj,nkj->nik", Rk, 1.0 / np.maximum(sck, 1e-5) ** 2, Rk)
        sol = solidity(Pm, Nm, Gk, op[kz], Sinv, args.solid_radius_m, args.window_m) if len(kz) else np.zeros(len(Pm))
        need = okf & ~seen & (sol < args.solid_alpha)
        if args.cosmetic:                                                  # plate inner faces + shadow floor: always replaced
            need = need | (okf & (face_class(Fs) >= 3))
        if args.replace:                                                   # replacement surfaces also where cameras look in;
            need = okf & ((sol < args.solid_alpha) | (face_class(Fs) == 3))   # plate inner faces: solid only via the plate
        # colours of seen samples: opacity/distance-weighted DC of nearby constrained gaussians of the object
        cons = inz[(wmax[inz] >= args.seen_w) & (op[inz] > 0.3)]
        scol = np.full((len(Ps), 3), np.nan)
        if len(cons):
            ct = cKDTree(Gm[cons]); nb = ct.query_ball_point(Pm[seen], args.tex_radius_m)
            for a_, j in zip(np.nonzero(seen)[0], nb):
                if len(j):
                    j = cons[np.asarray(j, int)]
                    w = op[j] * np.exp(-0.5 * (np.linalg.norm(Gm[j] - Pm[a_], axis=1) / (args.tex_radius_m / 2)) ** 2)
                    scol[a_] = (w[:, None] * dc[j]).sum(0) / max(w.sum(), 1e-9)
        obs = np.isfinite(scol[:, 0]); cls = face_class(Fs)
        orep.update({"face_samples": int(len(Ps)), "seen": int(seen.sum()), "observed_colour": int(obs.sum()),
                     "need": int(need.sum()), "solid_already": int((okf & ~seen & (sol >= args.solid_alpha)).sum())})
        covered = np.zeros(len(Ps), bool)
        # ---- symmetry transfer
        if "sym" in meth and obs.sum() > 30:
            ctr = np.array([0.0, 0.0, 0.0]); Lsym = Lev - ctr
            cev = np.zeros((len(Lev), 3)); ie = cKDTree(Ps[obs]).query(Lev)[1]; cev[:] = scol[obs][ie]
            res = []
            for nm, A in sym_candidates():
                inl, evf, cd = score_sym(Lsym, cev, A, args.sym_tol_m)
                res.append((inl - 0.5 * min(cd, 1.0), nm, A, inl, evf, cd))
            res.sort(key=lambda r: -r[0]); b = res[0]
            orep["symmetry"] = {"best": b[1], "inlier": round(b[3], 3), "evidence_overlap": round(b[4], 3), "colour_diff": round(b[5], 3),
                                "runner_up": [(r[1], round(r[3], 3)) for r in res[1:4]]}
            if b[3] >= args.sym_min:
                A = b[2]
                src = inz[keep[inz] & (wmax[inz] >= args.seen_w) & (op[inz] > 0.3) & (Lg[inz, 2] > args.floor_m) & (dS < args.off_m)]
                Ls_ = ob.loc(Gm[src]) @ A.T
                dT, iT = Ltr.query(Ls_)
                okT = (dT < args.sym_tol_m) & need[iT]
                src, Ls_, iT = src[okT], Ls_[okT], iT[okT]
                if len(src):
                    Pt = ob.met(Ls_)
                    # still hidden from every camera at the copy's own position (normal = face normal there)
                    hid = ~seen_by_cameras(Pt, args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity,
                                           args.min_hidden_m, normals=ob.vec(Ns[iT]), max_incidence=args.max_incidence)
                    src, Pt, iT = src[hid], Pt[hid], iT[hid]
                if len(src):
                    B = np.stack([ob.a1, ob.a2, n_f], 1)                       # local -> metric
                    Ad = R.T @ B @ A @ B.T @ R                                 # map in the dataset frame (orthogonal)
                    rows = vmod[src].copy()
                    Xd = ((Pt - t) @ R) / s
                    rows["x"], rows["y"], rows["z"] = Xd[:, 0], Xd[:, 1], Xd[:, 2]
                    Rl = Rotation.from_quat(q[src]).as_matrix(); Rn = Ad @ Rl
                    Rn[np.linalg.det(Rn) < 0, :, 2] *= -1                      # mirrored frame: flip one axis (same ellipsoid)
                    qq = Rotation.from_matrix(Rn).as_quat()
                    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = qq[:, 3], qq[:, 0], qq[:, 1], qq[:, 2]
                    if "nx" in v.dtype.names:
                        rows["nx"], rows["ny"], rows["nz"] = 0, 0, 0
                    if K:
                        rest = np.stack([rows[f"f_rest_{i}"] for i in range(nrest)], 1).reshape(-1, 3, K).transpose(0, 2, 1)
                        rest = sh_transform(rest.astype(np.float64), Ad, K).transpose(0, 2, 1).reshape(-1, nrest)
                        for i in range(nrest):
                            rows[f"f_rest_{i}"] = rest[:, i]
                    new_rows.append(rows)
                    meta_rows.append((np.full(len(src), k), Fs[iT], Ps[iT], Pt, Nm[iT], np.full(len(src), 3)))
                    cov = Ltr.query_ball_point(Ls_, args.cell)
                    covered[np.unique(np.concatenate([np.asarray(c_, int) for c_ in cov]))] = True
                orep["symmetry"]["copied"] = int(len(src))
        # ---- learned prior: generated object (learned_prior.py) aligned to the evidence; only unseen, needed surface
        gfiles = sorted(glob.glob(os.path.join(args.learned_dir, f"obj{o['id']}_s*.npz"))) if args.learned_dir else []
        if gfiles and obs.any():
            ie = cKDTree(Ps[obs]).query(Lev)[1]
            al = align_generated(gfiles, Lev, scol[obs][ie], hmax, ob, lambda P: seen_by_cameras(
                P, args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity, args.min_hidden_m, thru=True))
            orep["learned"] = al["report"]
            Xg = al["xyz"]; Lgn = ob.loc(Xg); dG, iG = Ltr.query(Lgn)
            okG = (al["opacity"] > 0.05) & (Lgn[:, 2] > args.floor_m) & (dG < args.off_m) & need[iG] & ~covered[iG]
            if okG.any():
                rows = np.zeros(int(okG.sum()), dtype=v.dtype)
                Xd = ((Xg[okG] - t) @ R) / s
                rows["x"], rows["y"], rows["z"] = Xd[:, 0], Xd[:, 1], Xd[:, 2]
                for i in range(3):
                    rows[f"f_dc_{i}"] = al["dc"][okG, i]
                    rows[f"scale_{i}"] = np.log(np.maximum(al["scale"][okG, i], 1e-4) / s)
                Rg = R.T @ al["rot"][okG]; qd = Rotation.from_matrix(Rg).as_quat()
                rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = qd[:, 3], qd[:, 0], qd[:, 1], qd[:, 2]
                og = np.clip(al["opacity"][okG], 0.02, 0.98); rows["opacity"] = np.log(og / (1 - og))
                new_rows.append(rows); idx = iG[okG]
                meta_rows.append((np.full(len(idx), k), Fs[idx], Ps[idx], Xg[okG], Nm[idx], np.full(len(idx), 4)))
                cov = Ltr.query_ball_point(Lgn[okG], args.cell)
                covered[np.unique(np.concatenate([np.asarray(c_, int) for c_ in cov]))] = True
            orep["learned"]["added"] = int(okG.sum())
            gen_dbg_extra[o["id"]] = Lgn[al["opacity"] > 0.3]
        # ---- discs on the remaining needed samples
        add = np.nonzero(need & ~covered)[0]
        if "prim" in meth and len(add):
            col = np.zeros((len(add), 3)); srcm = np.zeros(len(add), int)
            mean_obj = np.nanmedian(scol[obs], 0) if obs.any() else np.median(dc[inz], 0)
            if "app" in meth and obs.any():
                col[:], srcm[:] = np.nan, 0
                for c_ in np.unique(cls[add]):
                    ia = add[cls[add] == c_]; so = np.nonzero(obs & (cls == c_))[0]
                    if len(so) < 10:
                        continue
                    per = 2 * (2 * ob.half).sum() if (c_ == 0 and pr["kind"] == "stepbox") else None   # walls wrap round
                    Cs, Ca = face_coords(pr, Ps[so], Fs[so], ob.half), face_coords(pr, Ps[ia], Fs[ia], ob.half)
                    lo, hi = Cs.min(0), Cs.max(0); Lw = np.maximum(hi - lo, args.cell)
                    Ct = Ca.copy()
                    if per is not None:                                         # wrap the perimeter, then tile the seen run
                        d0 = np.mod(Ca[:, 0] - lo[0], per); Ct[:, 0] = lo[0] + np.mod(d0, Lw[0])
                        Ct[:, 1] = np.clip(Ca[:, 1], lo[1], hi[1])
                    else:
                        Ct = lo + np.mod(Ca - lo, Lw)
                    dd, jj = cKDTree(Cs).query(Ct, k=3)
                    w = 1 / (dd + 1e-3); cc = (w[..., None] * scol[so][jj]).sum(1) / w.sum(1, keepdims=True)
                    pos_ = np.searchsorted(add, ia); col[pos_] = cc; srcm[pos_] = 2
            miss = ~np.isfinite(col[:, 0]) if ("app" in meth and obs.any()) else np.ones(len(add), bool)
            if miss.any():                                                      # prim: same class, nearest seen sample
                for c_ in np.unique(cls[add[miss]]):
                    im = np.nonzero(miss)[0][cls[add[miss]] == c_]; so = np.nonzero(obs & (cls == c_))[0]
                    if not len(so):
                        so = np.nonzero(obs)[0]
                    if not len(so):
                        col[im] = mean_obj; srcm[im] = 0; continue
                    dd, jj = cKDTree(Ps[so]).query(Ps[add[im]], k=min(8, len(so)))
                    dd, jj = np.atleast_2d(dd), np.atleast_2d(jj)
                    w = np.exp(-dd / max(np.median(dd[:, 0]), args.cell)); col[im] = (w[..., None] * scol[so][jj]).sum(1) / w.sum(1, keepdims=True)
                    srcm[im] = 1
            Pa = ob.met(Ps[add]); Na = ob.vec(Ns[add])
            if args.replace:                                               # what the robot saw there (dark interiors, shadows)
                gc = gt_colours(Pa, args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.min_hidden_m)
                okc = np.isfinite(gc[:, 0]); col[okc] = (gc[okc] - 0.5) / C0; srcm[okc] = 5
                orep["gt_coloured"] = int(okc.sum())
            if args.cosmetic:                                              # dark inner material, shadowed floor
                ca = face_class(Fs[add]); rgb_o = 0.5 + C0 * scol[obs] if obs.any() else None
                if (ca == 3).any() and rgb_o is not None:
                    inn = obs & (cls == 3)
                    if inn.sum() >= 10:
                        dark = np.median(0.5 + C0 * scol[inn], 0)
                    else:
                        lum = rgb_o.mean(1); dark = np.median(rgb_o[lum <= np.quantile(lum, 0.2)], 0)
                    col[ca == 3] = (dark - 0.5) / C0; srcm[ca == 3] = 6
                if (ca == 4).any():
                    fl = inz[(Lg[inz, 2] < args.floor_m) & (op[inz] > 0.5)]
                    frgb = np.median(0.5 + C0 * dc[fl], 0) if len(fl) else np.array([0.5, 0.5, 0.5])
                    col[ca == 4] = (np.clip(args.shadow * frgb, 0, 1) - 0.5) / C0; srcm[ca == 4] = 7
            new_rows.append(disc_rows(v.dtype, ((Pa - t) @ R) / s, Na @ R, col, args.cell, s, args.opacity))
            meta_rows.append((np.full(len(add), k), Fs[add], Ps[add], Pa, Na, srcm))
            orep["discs"] = int(len(add)); orep["disc_colour_src"] = {"flat": int((srcm == 0).sum()), "nearest_same_class": int((srcm == 1).sum()),
                                                                     "texture_tiled": int((srcm == 2).sum())}
        orep["copies_covered_samples"] = int(covered.sum())
        rep["objects"].append(orep)
        gen_dbg.append((o["id"], ob, Ps, Fs, seen, need, Lev))
        print(f"[priors] obj {o['id']}: {pr['kind']} {scores} removed {orep['removed']} snapped {orep['snapped']} "
              f"need {orep['need']} sym {orep.get('symmetry', {}).get('best')}/{orep.get('symmetry', {}).get('copied')} "
              f"discs {orep.get('discs', 0)}", flush=True)
    nk = int(keep.sum())
    new = np.concatenate(new_rows) if new_rows else np.zeros(0, v.dtype)
    os.makedirs(args.out_dir, exist_ok=True)
    write_vertex(os.path.join(args.out_dir, "base_kept.ply"), vmod[keep])
    write_vertex(os.path.join(args.out_dir, "textured.ply"), np.concatenate([vmod[keep], new]))
    cat = lambda i, w: np.concatenate([r[i] for r in meta_rows]) if meta_rows else np.zeros((0,) * w)
    np.savez_compressed(os.path.join(args.out_dir, "meta.npz"), keep=keep, mod=mod, removed_near_surface=~keep,
                        disc_obj=cat(0, 1), disc_face=cat(1, 1), disc_local=cat(2, 2), disc_metric=cat(3, 2), disc_normal=cat(4, 2),
                        disc_src=cat(5, 1), cell=args.cell, sim3=np.array([s]), n_in=N)
    np.savez_compressed(os.path.join(args.out_dir, "zone.npz"), zone=zone_in)   # input rows in an object zone (zone_budget.py)
    np.savez_compressed(os.path.join(args.out_dir, "src.npz"), src=np.concatenate([np.nonzero(keep)[0], np.full(len(new), -1)]))
    rep.update({"gaussians_in": N, "kept": nk, "removed": int(N - nk), "snapped": int(mod.sum()), "added": int(len(new)),
                "seconds": round(time.time() - t0, 1)})
    json.dump(rep, open(os.path.join(args.out_dir, "priors.json"), "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    plot(gen_dbg, os.path.join(args.out_dir, "prims.png"), gen_dbg_extra)
    print("PRIORS_BUILD", json.dumps({kk: vv for kk, vv in rep.items() if kk != "objects"}), flush=True)


def face_coords(pr, P, F, half):
    """2D texture coordinates on a face: stepbox walls (perimeter s, 3 x height), tops (u, v), plate sides (in-plate
    p1, p2), cylinder side (arc length, 3 x height)."""
    a, b = half; cls = face_class(F); out = P[:, :2].copy()
    if pr["kind"] == "stepbox":
        u, v = P[:, 0], P[:, 1]
        s_ = np.select([F == 0, F == 2, F == 1, F == 3], [v + b, 2 * b + (a - u), 2 * b + 2 * a + (b - v), 4 * b + 2 * a + (u + a)], 0.0)
        out = np.where((cls == 0)[:, None], np.stack([s_, 3.0 * P[:, 2]], 1), out)
    elif pr["kind"] == "plates":
        for k, p in enumerate(pr["plates"]):
            m = (F - 20) // 2 == k
            c, p1, p2 = np.array(p["centre"]), np.array(p["p1"]), np.array(p["p2"])
            out[m] = np.stack([(P[m] - c) @ p1, (P[m] - c) @ p2], 1)
    elif pr["kind"] == "cylinder":
        c = np.array(pr["centre"]); w = cls == 0
        out[w] = np.stack([pr["r"] * np.arctan2(P[w, 1] - c[1], P[w, 0] - c[0]), 3.0 * P[w, 2]], 1)
    return out


def plot(dbg, path, extra=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    if not dbg:
        return
    fig, ax = plt.subplots(2, len(dbg), figsize=(4 * len(dbg), 8), squeeze=False)
    for c, (oid, ob, Ps, Fs, seen, need, Lev) in enumerate(dbg):
        for r, (i, j, lab) in enumerate(((0, 1, "top u-v"), (0, 2, "side u-h"))):
            a = ax[r, c]; a.scatter(Lev[:, i], Lev[:, j], s=1, c="k", alpha=0.3)
            a.scatter(Ps[seen, i], Ps[seen, j], s=1, c="g"); a.scatter(Ps[need, i], Ps[need, j], s=1, c="r")
            if extra and oid in extra:
                a.scatter(extra[oid][:, i], extra[oid][:, j], s=0.5, c="b", alpha=0.3)
            a.set_title(f"obj {oid} {lab} (black evidence, green seen, red filled)", fontsize=7); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ply", required=True); b.add_argument("--support", required=True); b.add_argument("--objects", required=True)
    b.add_argument("--cloud", required=True); b.add_argument("--sim3-json", required=True)
    b.add_argument("--dataset", nargs="+", required=True); b.add_argument("--splat-depth-dir", required=True)
    b.add_argument("--out-dir", required=True); b.add_argument("--method", default="prim,sym,app")
    b.add_argument("--learned-dir", default=None, help="learned_prior.py generate output (obj<id>_s<seed>.npz, TRELLIS unit cube)")
    b.add_argument("--cell", type=float, default=0.02); b.add_argument("--zone-m", type=float, default=0.12)
    b.add_argument("--seen-w", type=float, default=0.05, help="camera-constrained: support max blend weight >= this (evidence, sources)")
    b.add_argument("--junk-w", type=float, default=0.02, help="only gaussians with max blend weight below this are removed / snapped")
    b.add_argument("--needle-ratio", type=float, default=4.0); b.add_argument("--inlier-m", type=float, default=0.02)
    b.add_argument("--lam-fake", type=float, default=2.5, help="model selection: penalty on claimed surface seen without evidence")
    b.add_argument("--inside-m", type=float, default=0.025); b.add_argument("--off-m", type=float, default=0.04)
    b.add_argument("--floor-m", type=float, default=0.03, help="gaussians below this height are floor (untouched)")
    b.add_argument("--other-m", type=float, default=0.03, help="this close to a non-object LiDAR surface = untouched")
    b.add_argument("--solid-alpha", type=float, default=0.7); b.add_argument("--solid-radius-m", type=float, default=0.10)
    b.add_argument("--window-m", type=float, default=0.06); b.add_argument("--max-incidence", type=float, default=75.0)
    b.add_argument("--tol-rel", type=float, default=0.05); b.add_argument("--solid-opacity", type=float, default=0.98)
    b.add_argument("--min-hidden-m", type=float, default=0.05); b.add_argument("--opacity", type=float, default=0.97)
    b.add_argument("--tex-radius-m", type=float, default=0.03)
    b.add_argument("--fill-max-incidence", type=float, default=90.0,
                   help="a face sample seen by any camera at <= this incidence is not filled (90: grazing views count too)")
    b.add_argument("--replace", action="store_true",
                   help="replacement surfaces: discs on every not-solid primitive face (seen too), coloured by the real images there")
    b.add_argument("--cosmetic", action="store_true", help="shape-aware cleaning zones, A-frame inner faces + shadow floor patch")
    b.add_argument("--cos-zone-plates-m", type=float, default=0.4); b.add_argument("--cos-zone-box-m", type=float, default=0.3)
    b.add_argument("--shadow", type=float, default=0.55, help="floor patch under plates: floor colour x this")
    b.add_argument("--snap-w", type=float, default=None, help="spikes snapped only below this support (default --junk-w)")
    b.add_argument("--no-clean", action="store_true", help="no removals / snaps: only add the prior's surface")
    b.add_argument("--ev-min-h", type=float, default=0.045, help="evidence above this height only (floor band excluded)")
    b.add_argument("--carve-frac", type=float, default=0.3, help="depth prior: stop growing when this share of a strip is seen")
    b.add_argument("--cyl-penalty", type=float, default=0.1, help="model selection: complexity penalty of the cylinder")
    b.add_argument("--sym-tol-m", type=float, default=0.025); b.add_argument("--sym-min", type=float, default=0.55)
    a = ap.parse_args()
    {"build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
