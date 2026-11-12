#!/usr/bin/env python3
"""LiDAR image-based-rendering (IBR) pseudo-views: real pixels re-projected through the LiDAR geometry into virtual
cameras FAR from the training path (gsplat venv, GPU; torch + cv2 + scipy only).

A ground robot's camera rides ~0.4 m above the floor looking forward, so a splat is unconstrained from any other
height / angle (floor needles fitted at grazing angles, SH colour exploding at unseen directions). The LiDAR knows the
geometry from every side, and the training frames hold the (Lambertian) colour of most surfaces: together they give
real supervision at views the robot never took.

  views   virtual cameras + their IBR images, as a pseudo-dataset scripts/train_distill_gsplat.py consumes:
            <out>/sparse/0/{cameras,images,points3D}.txt, images/<stem>.png, images/<stem>_mask.png (255 = supervise),
            images/<stem>_depth.npy (LiDAR ray distance, dataset units, 0 = none; the trainer's --depth-lambda target),
            ignore_masks/<stem>_mask.png (255 = NO loss: the --mask-dir convention of finetune_ply_viewer.py /
            train_viewer_gsplat.py), ibr.json.
          Cameras: --n positions drawn uniformly from the LiDAR free-space grid (scripts/build_freespace_grid.py: free >=
          --min-free scans, no return within 1 voxel), --heights above the fitted floor (--scene-json of
          scripts/offpath_bench_prep.py, or fitted here needs open3d), >= --min-surf m from every LiDAR point, between
          --min-path and --max-path m from the nearest training camera; random yaw, pitch down U(--pitch). Views whose
          supervised share is < --min-valid are dropped. With --poses-from DS the cameras are DS's instead (validation:
          IBR at held-out real poses, output images named as DS's, scored against them: the ceiling of this supervision).
          Per view:
            geometry   the static LiDAR cloud + a dense floor (the fitted plane sampled at 1 cm wherever the cloud has
                       floor returns, holes closed) z-buffered into the view (build_depth_targets.zbuffer_target,
                       front-surface mean, depth edges and isolated returns dropped), then de-staircased: exact
                       ray-plane depth on the floor, 5x5 median elsewhere (refine_depth);
            sources    every TRAINING frame of --dataset whose own LiDAR depth (same cloud, same z-buffer) agrees at the
                       re-projected pixel (--tol-rel, --tol-abs: not occluded) and that sees the FRONT of the surface
                       (Jacobian of target pixel -> source pixel has det > 0);
            quality    q = smaller singular value of that Jacobian = source pixels per target pixel along the worst
                       direction (distance AND grazing foreshortening in one number), mildly penalised above 1;
            colour     the --topk best sources by q, one per group of --group consecutive frames (neighbouring frames
                       agree trivially); the largest photo-consistent set among them (max channel difference < --tau;
                       ties -> the set holding the best q); colour = q^--sharp weighted mean of it;
            mask       >= --min-sources consistent sources of which one views the point from >= --min-parallax deg
                       away from the reference (agreement then tests the geometry too), best q >= --q-min, reliable
                       geometry.
  floor   fit the floor plane once (CPU venv: open3d) -> scene.json for `views --scene-json`.
  merge   <real dataset> + <views dataset> -> one COLMAP dataset (links) + ignore_masks, for trainers without a
          pseudo-view hook (finetune_ply_viewer.py --dataset MERGED --mask-dir MERGED/ignore_masks).

    python scripts/make_lidar_ibr_views.py views --dataset TRAIN --cloud lidar_static.ply --sim3-json sim3.json \
        --scene-json scene.json --freespace freespace.npz --n 200 --out PDS
    python scripts/make_lidar_ibr_views.py views ... --poses-from TEST --out VAL      # validation at real held-out poses
    python scripts/make_lidar_ibr_views.py merge --dataset TRAIN --pseudo PDS --out MERGED
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model, zbuffer_target  # noqa: E402
from make_pseudo_views import rot2quat, write_images  # noqa: E402


def floor_basis(n):
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.3 else np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def dense_floor(L, n, d, cell=0.04, sub=4, thr=0.03, close_m=0.3):
    """Metric points on the floor plane, (cell / sub) m apart, wherever the cloud has floor returns (holes closed and
    filled): the LiDAR samples a floor seen at grazing angles in sparse rings; the plane is exact."""
    from scipy import ndimage
    e1, e2 = floor_basis(n)
    F = L[np.abs(L @ n + d) < thr]
    if len(F) < 100:
        return np.zeros((0, 3))
    x, y = F @ e1, F @ e2; x0, y0 = x.min(), y.min()
    ix, iy = ((x - x0) / cell).astype(int), ((y - y0) / cell).astype(int)
    occ = np.zeros((ix.max() + 1, iy.max() + 1), bool); occ[ix, iy] = True
    k = max(1, int(round(close_m / cell)))
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(occ, k), iterations=k))[k:-k, k:-k]
    ci, cj = np.nonzero(occ)
    o = (np.arange(sub) + 0.5) / sub
    gx = (ci[:, None, None] + o[None, :, None]) * cell + x0; gy = (cj[:, None, None] + o[None, None, :]) * cell + y0
    gx, gy = np.broadcast_arrays(gx, gy)
    return gx.reshape(-1, 1) * e1 + gy.reshape(-1, 1) * e2 - d * n                                   # h = n.x + d = 0


def get_floor(a, L, Cm):
    if a.scene_json and os.path.isfile(a.scene_json):
        f = json.load(open(a.scene_json))["floor"]; return np.asarray(f["n"], float), float(f["d"])
    from floor_plane import find_floor_plane                                                          # needs open3d
    fl = find_floor_plane(L, Cm)
    if fl is None:
        raise SystemExit("[ibr] no floor plane")
    return fl[0], fl[1]


def sample_cameras(a, L, Cm, n, d, rng, count):
    """-> [(c2w rotation (OpenCV), centre)] in the METRIC frame + meta, `count` candidates."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    z = np.load(a.freespace); free, hit, org, vox = z["free"], z["hit"], z["origin"], float(z["voxel"])
    cf = (free >= a.min_free) & ~ndimage.maximum_filter(hit > 0, size=3)
    ijk = np.argwhere(cf); X = org + (ijk + 0.5) * vox
    h = X @ n + d; X = X[(h >= a.heights[0]) & (h <= a.heights[1])]
    dp = cKDTree(Cm).query(X, k=1)[0]; X = X[(dp >= a.min_path) & (dp <= a.max_path)]
    X = X[rng.permutation(len(X))[: 40 * count]]
    ds = cKDTree(L).query(X, k=1, workers=-1)[0]; X = X[ds >= a.min_surf]
    print(f"[ibr] free-space candidates: {int(cf.sum())} free voxels -> {len(X)} positions pass height / path / surface limits", flush=True)
    if len(X) < count:
        print(f"[ibr] WARNING: only {len(X)} candidate positions for {count} requested", flush=True)
    X = X[:count] + rng.uniform(-0.5, 0.5, (min(count, len(X)), 3)) * vox
    e1, e2 = floor_basis(n); out = []
    for C in X:
        yaw = rng.uniform(0, 2 * np.pi); pitch = np.radians(rng.uniform(a.pitch[0], a.pitch[1]))
        hz = np.cos(yaw) * e1 + np.sin(yaw) * e2
        f = np.cos(pitch) * hz - np.sin(pitch) * n; right = np.cross(f, n); right /= np.linalg.norm(right)
        out.append((np.stack([right, np.cross(f, right), f], 1), C,
                    {"height_m": round(float(C @ n + d), 3), "pitch_deg": round(float(np.degrees(pitch)), 1), "yaw_deg": round(float(np.degrees(yaw)), 1)}))
    return out


def refine_depth(D, Rcw, tcw, K, W, H, nc, dc, floor_tol, smooth_rel=0.03):
    """z-buffered ray distance (H,W; 0 = none) -> the same without the point-splat staircase: pixels whose surface point
    lies within floor_tol (dataset units) of the floor plane nc . x + dc = 0 take the exact ray-plane distance; the rest a
    5x5 median (only where it stays within smooth_rel of the z-buffer value, so depth edges are not moved)."""
    import cv2
    import torch
    fx, fy, cx, cy = K; dev = D.device
    uu, vv = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
    ray = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1).double()
    ray = (ray / torch.linalg.norm(ray, dim=-1, keepdim=True)) @ Rcw                                  # world directions
    C = -tcw @ Rcw
    den = ray @ nc; tau = -(C @ nc + dc) / torch.where(den.abs() > 1e-9, den, torch.full_like(den, 1e-9))
    h = (C + ray * D[..., None]) @ nc + dc
    fl = (D > 0) & (h.abs() < floor_tol) & (tau > 0)
    med = torch.from_numpy(cv2.medianBlur(D.float().cpu().numpy(), 5)).to(dev).double()
    sm = (D > 0) & (med > 0) & ((med - D).abs() < smooth_rel * D)
    return torch.where(fl, tau, torch.where(sm, med, D))


def push_pull(img, valid, levels=6):
    """Fill invalid pixels of (H,W,3) float with a smooth extension of the valid ones (keeps the EAS edge map quiet)."""
    import cv2
    acc, w = img * valid[..., None], valid.astype(np.float32)
    out, wo = acc.copy(), w.copy()
    for k in range(1, levels + 1):
        s = 2 * (2 ** k) + 1
        b, bw = cv2.blur(acc, (s, s)), cv2.blur(w, (s, s))
        m = (wo < 1e-3) & (bw > 1e-3)
        out[m] = b[m] / bw[m][:, None]; wo[m] = 1
    return out


def views(a):
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from plyfile import PlyData
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed); t_start = time.time()
    j = json.load(open(a.sim3_json)); s, R, t = float(j["scale"]), np.asarray(j["R"]), np.asarray(j["t"])
    v = PlyData.read(a.cloud)["vertex"]; L = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)   # metric
    cams, imgs = read_text_model(os.path.join(a.dataset, "sparse", "0")); imgs = sorted(imgs, key=lambda i: i[0])
    Cm = s * np.array([-quat_to_R(q).T @ tt for _, _, q, tt in imgs]) @ R.T + t
    n, d = get_floor(a, L, Cm)
    Fl = dense_floor(L, n, d) if a.floor_fill else np.zeros((0, 3))
    G = np.concatenate([L, Fl]); to_ds = lambda Xm: ((Xm - t) @ R) / s                                 # metric -> dataset frame
    X = torch.tensor(to_ds(G), dtype=torch.float64, device=dev)
    vox, rmax = a.voxel_m / s, a.max_range_m / s
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float64), device=dev)
    nc, dc = T(R.T @ n), float(n @ t + d) / s                                                     # floor plane, dataset frame
    zbuf = lambda Rcw, tc, W, H, K: refine_depth(zbuffer_target(X, Rcw, tc, W, H, K, vox, rmax, a.edge_rel, a.min_support, "surface")[0],
                                                 Rcw, tc, K, W, H, nc, dc, a.floor_tol / s) if a.refine else \
        zbuffer_target(X, Rcw, tc, W, H, K, vox, rmax, a.edge_rel, a.min_support, "surface")[0]
    print(f"[ibr] cloud {len(L)} LiDAR + {len(Fl)} floor-plane points; floor n={np.round(n, 3).tolist()} d={d:.3f}; "
          f"{len(imgs)} source frames ({a.dataset})", flush=True)

    # ---- sources: pose, intrinsics, image, LiDAR depth from the same cloud
    src = []
    for name, cid, q, tcw in imgs:
        W, H, K = cams[cid]; Rcw = T(quat_to_R(q)); tc = T(tcw)
        D = zbuf(Rcw, tc, W, H, K)
        im = torch.from_numpy(np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"))).to(dev)
        src.append((name, Rcw, tc, K, W, H, D.float(), im.permute(2, 0, 1)[None]))                    # uint8 (1,3,H,W)
    SR = torch.stack([s_[1] for s_ in src]); ST = torch.stack([s_[2] for s_ in src])
    print(f"[ibr] source depths rendered ({time.time() - t_start:.0f}s); LiDAR-valid share median "
          f"{float(np.median([float((s_[6] > 0).float().mean()) for s_ in src])):.3f}", flush=True)

    # ---- target cameras (dataset frame: Rcw, tcw)
    if a.poses_from:
        tc_, ti_ = read_text_model(os.path.join(a.poses_from, "sparse", "0")); ti_ = sorted(ti_, key=lambda i: i[0])
        tgt = [(nm, tc_[cid], quat_to_R(q), np.asarray(tt), {}) for nm, cid, q, tt in ti_][:: a.every]
    else:
        cid0 = sorted(cams)[0]
        tgt = []
        for k, (Rm, C, meta) in enumerate(sample_cameras(a, L, Cm, n, d, rng, int(a.n * a.oversample))):
            Rc, Cc = R.T @ Rm, to_ds(C)
            tgt.append((f"ibr_{k:05d}.png", cams[cid0], Rc.T, -Rc.T @ Cc, meta))
    im_dir = os.path.join(a.out, "images"); ig_dir = os.path.join(a.out, "ignore_masks"); s0 = os.path.join(a.out, "sparse", "0")
    for d_ in (im_dir, ig_dir, s0):
        os.makedirs(d_, exist_ok=True)
    rows, metas, stats = [], [], []
    k3 = np.ones((3, 3), np.uint8)
    for name, (W, H, K), Rcw_n, tcw_n, meta in tgt:
        if not a.poses_from and len(rows) >= a.n:
            break
        fx, fy, cx, cy = K; Rcw, tcw = T(Rcw_n), T(tcw_n)
        D = zbuf(Rcw, tcw, W, H, K)                                                                    # ray distance
        uu, vv = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
        ray = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1).double()
        ray = ray / torch.linalg.norm(ray, dim=-1, keepdim=True)
        Xw = (ray * D[..., None] - tcw) @ Rcw                                                          # (H,W,3) world points
        val = D > 0
        ok = val.clone(); ok[:, :-1] &= val[:, 1:] & ((D[:, 1:] - D[:, :-1]).abs() < a.edge_rel * D[:, :-1]); ok[:, -1] = False
        ok[:-1] &= val[1:] & ((D[1:] - D[:-1]).abs() < a.edge_rel * D[:-1]); ok[-1] = False
        vi, ui = torch.nonzero(ok, as_tuple=True)
        P = len(vi); stem = os.path.splitext(name)[0]
        col = torch.zeros(H, W, 3, device=dev); msk = torch.zeros(H, W, dtype=torch.bool, device=dev); anyc = torch.zeros_like(msk); sing = torch.zeros_like(msk)
        nsrc = 0
        if P > 0:
            A, B, Cc_ = Xw[vi, ui], Xw[vi, ui + 1], Xw[vi + 1, ui]
            # candidate sources: a sparse subset of the points inside the frustum
            sub = A[:: max(1, P // 600)]
            pc = torch.einsum("sij,nj->sni", SR, sub) + ST[:, None]
            zz = pc[..., 2].clamp_min(1e-6)
            inside = (pc[..., 2] > 0.05) & (pc[..., 0] / zz * fx + cx > 0) & (pc[..., 0] / zz * fx + cx < W) & \
                     (pc[..., 1] / zz * fy + cy > 0) & (pc[..., 1] / zz * fy + cy < H)                  # (same intrinsics model)
            cand = torch.nonzero(inside.float().mean(1) > a.min_overlap)[:, 0].tolist()
            Qb = torch.zeros(P, a.topk, device=dev); Fb = torch.zeros(P, a.topk, 6, device=dev)          # q | colour, direction
            gq = torch.zeros(P, device=dev); gf = torch.zeros(P, 6, device=dev); gcur = -1

            def flush():                                                                               # group best -> top-K
                i_ = torch.nonzero(gq > 0)[:, 0]
                if len(i_):
                    allq = torch.cat([Qb[i_], gq[i_, None]], 1); allf = torch.cat([Fb[i_], gf[i_, None]], 1)
                    tq, ti = torch.topk(allq, a.topk, dim=1)
                    Qb[i_] = tq; Fb[i_] = torch.gather(allf, 1, ti[..., None].expand(-1, -1, 6))
                gq.zero_()

            for si in cand:
                if si // a.group != gcur:                                                              # consecutive frames = one
                    flush(); gcur = si // a.group                                                      # vote (no parallax)
                _, Rs, ts, Ks, Ws, Hs, Ds, Is = src[si]
                fxs, fys, cxs, cys = Ks
                pa = A @ Rs.T + ts; za = pa[:, 2]
                ua, va = fxs * pa[:, 0] / za + cxs, fys * pa[:, 1] / za + cys
                i = torch.nonzero((za > 0.05) & (ua >= 0.5) & (ua < Ws - 0.5) & (va >= 0.5) & (va < Hs - 0.5))[:, 0]
                if len(i) == 0:
                    continue
                ua, va, ra = ua[i], va[i], torch.linalg.norm(pa[i], dim=1)
                ds_ = Ds[va.long(), ua.long()].double()
                okd = (ds_ > 0) & ((ra - ds_).abs() < a.tol_rel * ds_ + a.tol_abs / s)
                pb, pc_ = B[i] @ Rs.T + ts, Cc_[i] @ Rs.T + ts
                okd &= (pb[:, 2] > 0.05) & (pc_[:, 2] > 0.05)
                ja, jc = fxs * pb[:, 0] / pb[:, 2] + cxs - ua, fys * pb[:, 1] / pb[:, 2] + cys - va     # d(u_s, v_s) / d x_t
                jb, jd = fxs * pc_[:, 0] / pc_[:, 2] + cxs - ua, fys * pc_[:, 1] / pc_[:, 2] + cys - va  # d(u_s, v_s) / d y_t
                det = ja * jd - jb * jc
                s1 = ja * ja + jb * jb + jc * jc + jd * jd
                s2 = torch.sqrt((ja * ja + jb * jb - jc * jc - jd * jd) ** 2 + 4 * (ja * jc + jb * jd) ** 2)
                smin = torch.sqrt(((s1 - s2) / 2).clamp_min(0))
                q = torch.where(smin <= 1, smin, smin.clamp_min(1e-6) ** -0.25)
                q = torch.where(okd & (det > 0) & torch.isfinite(q), q, torch.zeros_like(q)).float()
                kq = torch.nonzero(q > 0)[:, 0]
                if len(kq) == 0:
                    continue
                i, q = i[kq], q[kq]
                grid = torch.stack([ua[kq] / Ws * 2 - 1, va[kq] / Hs * 2 - 1], -1).float()[None, None]
                c = F.grid_sample(Is.float() / 255, grid, mode="bilinear", align_corners=False)[0, :, 0].T   # (n,3)
                dv = F.normalize((-ts @ Rs) - A[i], dim=-1).float()                                    # point -> source centre
                b = q > gq[i]; ib = i[b]
                gq[ib] = q[b]; gf[ib] = torch.cat([c[b], dv[b]], 1); nsrc += 1
            flush()
            Cb, Db = Fb[..., :3], Fb[..., 3:]
            vq = Qb > 0
            diff = (Cb[:, :, None] - Cb[:, None]).abs().amax(-1)
            agree = (diff < a.tau) & vq[:, :, None] & vq[:, None]
            cnt = agree.sum(2)
            ref = torch.argmax(cnt.float() + 0.5 * Qb / Qb.amax(1, keepdim=True).clamp_min(1e-6), 1)
            sel = agree[torch.arange(P, device=dev), ref]
            w = (Qb ** a.sharp) * sel
            c = (w[..., None] * Cb).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
            c = torch.where(sel.any(1, keepdim=True), c, Cb[:, 0])                                     # no source: zeros
            qref = (Qb * sel).amax(1); nag = sel.sum(1)
            col[vi, ui] = c; anyc[vi, ui] = vq.any(1)
            ar = torch.arange(P, device=dev)
            par = (((Db * Db[ar, ref][:, None]).sum(-1) < np.cos(np.radians(a.min_parallax))) & sel).any(1)   # a second,
            msk[vi, ui] = (nag >= a.min_sources) & (qref >= a.q_min) & par                             # really different view
            sing[vi, ui] = (qref >= a.q_min) & ((vq.sum(1) == 1) | (nag >= 2))    # not: several sources that disagree
        m = cv2.morphologyEx(cv2.morphologyEx(msk.cpu().numpy().astype(np.uint8), cv2.MORPH_CLOSE, k3), cv2.MORPH_OPEN, k3).astype(bool)
        m &= anyc.cpu().numpy()
        m1 = np.zeros_like(m)
        if a.single_weight > 0:                                 # one source (or no parallax / no agreement): lower weight
            m1 = cv2.morphologyEx(sing.cpu().numpy().astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool) & ~m
        share = float(m.mean())
        st = {"name": name, "lidar_valid": round(float(val.float().mean()), 4), "any_source": round(float(anyc.float().mean()), 4),
              "mask": round(share, 4), "single": round(float(m1.mean()), 4), "sources": nsrc, **meta}
        img = col.cpu().numpy(); anyn = anyc.cpu().numpy()
        gt_f = os.path.join(a.poses_from, "images", name) if a.poses_from else None
        if gt_f and os.path.exists(gt_f) and m.any():                                                  # validation: masked PSNR
            gt = np.asarray(Image.open(gt_f).convert("RGB"), np.float32) / 255
            pr = np.round(np.clip(img, 0, 1) * 255) / 255
            if m1.any():
                st["mse_single"] = float(((pr - gt) ** 2)[m1].mean())
            st["mse_mask"] = float(((pr - gt) ** 2)[m].mean()); st["mse_any"] = float(((pr - gt) ** 2)[anyn].mean())
        stats.append(st)
        if not a.poses_from and share < a.min_valid:
            continue
        out = push_pull(img, anyn) if not a.poses_from else img
        Image.fromarray(np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)).save(os.path.join(im_dir, stem + ".png"))
        mv = (m * 255).astype(np.uint8); mv[m1] = int(round(min(a.single_weight, 0.49) * 255))          # stays below any > 127 test
        Image.fromarray(mv, mode="L").save(os.path.join(im_dir, stem + "_mask.png"))
        Image.fromarray((~m * 255).astype(np.uint8), mode="L").save(os.path.join(ig_dir, stem + "_mask.png"))
        Dn = D.float().cpu().numpy()
        if a.depth_where == "supervised":                       # geometry only where colour is taught too: surfaces no
            Dn = Dn * (mv > 0)                                  # frame saw stay for the fill / inpaint stage
        np.save(os.path.join(im_dir, stem + "_depth.npy"), Dn.astype(np.float16))
        cidw = sorted(cams)[0] if not a.poses_from else None
        rows.append((len(rows) + 1, rot2quat(Rcw_n), tcw_n, cidw or 1, name)); metas.append(st)
        if len(stats) % 25 == 0:
            print(f"[ibr] {len(stats)} views done, {len(rows)} kept ({time.time() - t_start:.0f}s); last: {st}", flush=True)
    src0 = os.path.join(a.poses_from or a.dataset, "sparse", "0")
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(src0, "cameras.txt")))
    if a.poses_from:                                                                                   # keep the camera ids
        ids = {i[0]: i[1] for i in read_text_model(src0)[1]}; rows = [(r[0], r[1], r[2], ids[r[4]], r[4]) for r in rows]
    write_images(os.path.join(s0, "images.txt"), rows)
    if not os.path.exists(os.path.join(s0, "points3D.txt")):
        open(os.path.join(s0, "points3D.txt"), "w").close()
    pct = lambda key: np.round(np.percentile([m_[key] for m_ in metas], [0, 10, 50, 90, 100]), 3).tolist() if metas else None
    info = {"args": {k: v_ for k, v_ in vars(a).items() if k != "func"}, "n_candidates": len(stats), "n_views": len(rows),
            "mask_share_pct_0_10_50_90_100": pct("mask"), "mask_share_mean": round(float(np.mean([m_["mask"] for m_ in metas])), 4) if metas else None,
            "any_source_share_mean": round(float(np.mean([m_["any_source"] for m_ in metas])), 4) if metas else None,
            "lidar_valid_share_mean": round(float(np.mean([m_["lidar_valid"] for m_ in metas])), 4) if metas else None,
            "floor": {"n": n.tolist(), "d": d}, "n_floor_points": len(Fl), "seconds": round(time.time() - t_start, 1)}
    mm = [m_ for m_ in metas if "mse_mask" in m_]
    if mm:      # pixel-pooled over views (weights = masked pixels) and the per-view mean
        wts = np.array([m_["mask"] for m_ in mm])
        info["val_psnr_mask_pooled"] = round(float(-10 * np.log10((np.array([m_["mse_mask"] for m_ in mm]) * wts).sum() / wts.sum())), 3)
        info["val_psnr_mask_per_view"] = round(float(np.mean([-10 * np.log10(max(m_["mse_mask"], 1e-10)) for m_ in mm])), 3)
        wa = np.array([m_["any_source"] for m_ in mm])
        info["val_psnr_any_pooled"] = round(float(-10 * np.log10((np.array([m_["mse_any"] for m_ in mm]) * wa).sum() / wa.sum())), 3)
    ms = [m_ for m_ in metas if "mse_single" in m_]
    info["single_share_mean"] = round(float(np.mean([m_["single"] for m_ in metas])), 4) if metas else None
    if ms:
        w1 = np.array([m_["single"] for m_ in ms])
        info["val_psnr_single_pooled"] = round(float(-10 * np.log10((np.array([m_["mse_single"] for m_ in ms]) * w1).sum() / w1.sum())), 3)
    json.dump({**info, "views": metas}, open(os.path.join(a.out, "ibr.json"), "w"), indent=1)
    print("IBR_VIEWS " + json.dumps(info), flush=True)


def floor(a):
    from plyfile import PlyData
    from floor_plane import find_floor_plane
    j = json.load(open(a.sim3_json)); s, R, t = float(j["scale"]), np.asarray(j["R"]), np.asarray(j["t"])
    v = PlyData.read(a.cloud)["vertex"]; L = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    imgs = read_text_model(os.path.join(a.dataset, "sparse", "0"))[1]
    Cm = s * np.array([-quat_to_R(q).T @ tt for _, _, q, tt in imgs]) @ R.T + t
    fl = find_floor_plane(L, Cm)
    if fl is None:
        raise SystemExit("[ibr] no floor plane")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"floor": {"n": fl[0].tolist(), "d": float(fl[1])}, "sim3": {"scale": s, "R": R.tolist(), "t": t.tolist()}}, open(a.out, "w"), indent=1)
    print(f"[ibr] floor -> {a.out}")


def merge(a):
    """real + pseudo -> one COLMAP text dataset (links); ignore_masks only for the pseudo views."""
    s0 = os.path.join(a.out, "sparse", "0"); im = os.path.join(a.out, "images"); ig = os.path.join(a.out, "ignore_masks")
    for d_ in (s0, im, ig):
        os.makedirs(d_, exist_ok=True)
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(a.dataset, "sparse", "0", "cameras.txt")))
    rows = []
    for ds, base in ((a.dataset, 0), (a.pseudo, 1_000_000)):
        for ln in [x for x in open(os.path.join(ds, "sparse", "0", "images.txt")) if not x.startswith("#")][::2]:
            p = ln.split()
            if len(p) < 10:
                continue
            rows.append((base + int(p[0]), [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), p[9]))
            dst = os.path.join(im, p[9])
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(os.path.join(ds, "images", p[9])), dst)
            mk = os.path.join(ds, "ignore_masks", os.path.splitext(p[9])[0] + "_mask.png")
            if base and os.path.exists(mk) and not os.path.lexists(os.path.join(ig, os.path.basename(mk))):
                os.symlink(os.path.realpath(mk), os.path.join(ig, os.path.basename(mk)))
    write_images(os.path.join(s0, "images.txt"), rows)
    pts = os.path.join(s0, "points3D.txt")
    if not os.path.lexists(pts):
        os.symlink(os.path.realpath(os.path.join(a.dataset, "sparse", "0", "points3D.txt")), pts)
    print(f"[ibr] merged {len(rows)} views ({a.dataset} + {a.pseudo}) -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("views")
    p.add_argument("--dataset", required=True, help="TRAINING frames (COLMAP text dataset): the only colour sources")
    p.add_argument("--cloud", required=True, help="static LiDAR cloud (.ply, metric frame)")
    p.add_argument("--sim3-json", required=True, help="x_metric = s R x_dataset + t")
    p.add_argument("--scene-json", default=None, help="offpath_bench_prep.py scene.json (floor plane); else fitted here (open3d)")
    p.add_argument("--freespace", default=None, help="build_freespace_grid.py .npz (needed unless --poses-from)")
    p.add_argument("--out", required=True)
    p.add_argument("--poses-from", default=None, help="render at this dataset's cameras instead of sampling (validation)")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--n", type=int, default=200); p.add_argument("--oversample", type=float, default=2.5)
    p.add_argument("--heights", type=float, nargs=2, default=[0.3, 1.8], help="m above the floor")
    p.add_argument("--pitch", type=float, nargs=2, default=[0.0, 60.0], help="degrees below the horizon")
    p.add_argument("--min-surf", type=float, default=0.4, help="m to the nearest LiDAR point")
    p.add_argument("--min-path", type=float, default=0.3, help="m to the nearest training camera, at least")
    p.add_argument("--max-path", type=float, default=2.5, help="... and at most (stay in the mapped room)")
    p.add_argument("--min-free", type=int, default=3)
    p.add_argument("--min-valid", type=float, default=0.25, help="drop sampled views with a smaller supervised share")
    p.add_argument("--no-floor-fill", dest="floor_fill", action="store_false")
    p.add_argument("--no-refine", dest="refine", action="store_false", help="keep the raw z-buffer depth (point-splat staircase)")
    p.add_argument("--floor-tol", type=float, default=0.04, help="m: surface points this close to the floor plane take the exact plane depth")
    p.add_argument("--voxel-m", type=float, default=0.02); p.add_argument("--max-range-m", type=float, default=12.0)
    p.add_argument("--edge-rel", type=float, default=0.05); p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--tol-rel", type=float, default=0.04); p.add_argument("--tol-abs", type=float, default=0.02, help="m")
    p.add_argument("--topk", type=int, default=5); p.add_argument("--tau", type=float, default=0.08)
    p.add_argument("--group", type=int, default=8, help="consecutive source frames (name order) that count as ONE candidate")
    p.add_argument("--min-parallax", type=float, default=2.0, help="deg between the reference source and an agreeing one")
    p.add_argument("--sharp", type=float, default=4.0, help="blend weight q^sharp")
    p.add_argument("--min-sources", type=int, default=2); p.add_argument("--q-min", type=float, default=0.15)
    p.add_argument("--single-weight", type=float, default=0.0, help="> 0: pixels with a usable source (q >= --q-min) but no "
                   "verified consensus get this weight in _mask.png (value round(255 w) < 128; train_distill_gsplat.py --pseudo-soft)")
    p.add_argument("--depth-where", choices=("all", "supervised"), default="all", help="_depth.npy target: every LiDAR-valid "
                   "pixel, or only pixels with a colour weight (a depth-only surface gets an arbitrary colour: 'glassy' objects)")
    p.add_argument("--min-overlap", type=float, default=0.01, help="source considered if this share of the view projects into it")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=views)
    fl = sub.add_parser("floor", help="fit the floor plane (CPU venv with open3d) -> scene.json for `views --scene-json`")
    fl.add_argument("--dataset", required=True); fl.add_argument("--cloud", required=True); fl.add_argument("--sim3-json", required=True)
    fl.add_argument("--out", required=True)
    fl.set_defaults(func=floor)
    m = sub.add_parser("merge")
    m.add_argument("--dataset", required=True); m.add_argument("--pseudo", required=True); m.add_argument("--out", required=True)
    m.set_defaults(func=merge)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
