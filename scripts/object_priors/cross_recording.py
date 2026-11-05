"""Unseen object sides from OTHER recordings of the same room, object by object (objects move between recordings).

Recordings of one room share the mocap metric frame (each pipeline's sim3.json: x_metric = s R x_colmap + t), but the
movable objects (brick stacks, blocks, ramps) were rearranged between them, so whole-scene fusion ghosts. Here every
object of the reference recording (find_objects.py) is registered on its own:

  register (CPU venv) per reference object x other recording: the other recording's LiDAR points near the object
           (connected components on a floor grid within --search-m), each candidate aligned by a yaw scan + trimmed ICP
           restricted to the floor plane (rotation about the floor normal + in-plane translation: objects slide / turn on
           the floor), scored both ways (share of points within --inlier-m) -> accepted only if both shares >=
           --min-fit, the inlier residual <= --max-rmse-m and the heights agree (else: moved out of reach, changed, gone).
           Near-ties prefer the smaller rotation (4-fold symmetric stacks).
  views    (CPU venv) surface samples of each object (reference + registered LiDAR, 1 cm, PCA normals pointing out); per
           sample the reference views that see it well (in view, not occluded in that recording's LiDAR z-buffer,
           incidence <= --good-deg) -> faintly seen = fewer than --ref-min good reference views; every accepted other
           recording's frame that sees >= --min-gain such samples becomes an extra view: its pose mapped
           B colmap -> metres -> object transform -> reference colmap frame, and a per-pixel mask of the object's
           pixels (object surface in front of B's LiDAR z-buffer and of the REFERENCE scene's other geometry at the mapped
           pose, eroded) -> OUT/xviews/{images,masks,sparse/0} + OUT/inventory.json (per object: seen / faint share,
           share each recording adds, frames). Every --holdout-th extra view of each (object, recording) is held out.
  xfit     (gsplat venv, GPU) object-local fine-tune: only the input gaussians in the object zones (and rows a build
           added) train; reference training frames with the full viewer-exact L1+SSIM loss (seen pixels stay), extra
           views with the loss on their mask only, through a per-view colour affine (3x3 + bias, regularised to
           identity: exposure / white balance of the other recording), anchor to the input values. Writes the refit
           format (out.ply + out_src.npz) for zone_budget.py / shfit. Before / after scores on the held-out extra views
           (masked object pixels = real ground truth of the faces the reference never saw well) -> out_xfit.json.

    python scripts/object_priors/cross_recording.py register --objects D/objects --ref-pipeline <A>/pipeline \
        --others <B1>/pipeline <B2>/pipeline --out-dir X
    python scripts/object_priors/cross_recording.py views --objects D/objects --ref-pipeline <A>/pipeline \
        --ref-dataset <train ds> <test ds> --out-dir X
    python scripts/object_priors/cross_recording.py xfit --ply in.ply --meta T/meta.npz --zone T/zone.npz \
        --dataset <train ds> <test ds> --xviews X/xviews --out T/textured.ply
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import sim3, load_objects  # noqa: E402


# ---------------------------------------------------------------- shared helpers (CPU)
def floor_frame(meta):
    n = np.array(meta["floor"]["n"], float); n /= np.linalg.norm(n); d = float(meta["floor"]["d"])
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    return n, d, e1, np.cross(n, e1)


def plane_T(theta, tu, tv, n, e1, e2, pivot):
    """4x4 metric rigid map: rotation theta about the floor normal through pivot, then in-plane translation."""
    E = np.stack([e1, e2, n]); c, s_ = np.cos(theta), np.sin(theta)
    M = E.T @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ E
    T = np.eye(4); T[:3, :3] = M; T[:3, 3] = pivot - M @ pivot + tu * e1 + tv * e2
    return T


def apply(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def read_cloud(pipeline, cloud_dir=None):
    """metric LiDAR cloud of a recording: <cloud_dir>/<sequence>/lidar_static.ply (dynamic-free, made by the sbatch),
    else the pipeline's own static / accumulated cloud."""
    from plyfile import PlyData
    seq = os.path.basename(os.path.dirname(os.path.abspath(pipeline)))
    cands = ([os.path.join(cloud_dir, seq, "lidar_static.ply")] if cloud_dir else []) + \
        [os.path.join(pipeline, f) for f in ("offpath/depth/lidar_static.ply", "lidar_cloud.ply", "depth_cloud.ply")]
    for p in cands:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            v = PlyData.read(p)["vertex"].data
            return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), p
    return None, None


def voxel(X, v):
    return X[np.unique(np.floor(X / v).astype(np.int64), axis=0, return_index=True)[1]] if len(X) else X


def cameras(dataset_dirs, sim3_json):
    """[(name, (fx, fy, cx, cy, W, H), c2w metric 4x4, source dataset dir)] from COLMAP models (text or binary)."""
    import pycolmap
    s, R, t = sim3(sim3_json); out = []
    for ds in dataset_dirs:
        rec = pycolmap.Reconstruction(os.path.join(ds, "sparse", "0"))
        for img in sorted(rec.images.values(), key=lambda i: i.name):
            cfw = img.cam_from_world; M = np.asarray((cfw() if callable(cfw) else cfw).matrix())
            Rwc, C = M[:, :3].T, -M[:, :3].T @ M[:, 3]
            cam = rec.cameras[img.camera_id]; p = [float(x) for x in cam.params]
            K = (p[0], p[0], p[1], p[2]) if cam.model.name == "SIMPLE_PINHOLE" else tuple(p[:4])
            c2w = np.eye(4); c2w[:3, :3] = R @ Rwc; c2w[:3, 3] = s * R @ C + t
            out.append((img.name, (*K, cam.width, cam.height), c2w, ds))
    return out


def project(X, c2w, K):
    fx, fy, cx, cy, W, H = K
    Pc = (X - c2w[:3, 3]) @ c2w[:3, :3]; z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Pc[:, 0] / z + cx; v = fy * Pc[:, 1] / z + cy
    ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def zbuffer(X, c2w, K, scale, vox, rmax=8.0):
    """min ray distance per pixel (inf = empty) of a cloud splatted as squares of its voxel size, at 1/scale res."""
    fx, fy, cx, cy, W, H = K; w, h = int(np.ceil(W / scale)), int(np.ceil(H / scale))
    u, v, z, ok = project(X, c2w, K); ok &= z < rmax
    u, v, z = u[ok] / scale, v[ok] / scale, z[ok]
    r = np.clip(np.ceil(0.5 * vox * fx / scale / z).astype(int), 0, 6)
    D = np.full(h * w, np.inf)
    for k in range(int(r.max()) + 1 if len(r) else 0):
        sel = r >= k
        for du in range(-k, k + 1):
            for dv in range(-k, k + 1):
                if max(abs(du), abs(dv)) != k:
                    continue
                uu = np.floor(u[sel]).astype(int) + du; vv = np.floor(v[sel]).astype(int) + dv
                m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                np.minimum.at(D, vv[m] * w + uu[m], z[sel][m])
    return D.reshape(h, w)


def visible(P, N, c2w, K, Dz, scale, good_deg, tol=0.04, tol_rel=0.03, max_dist=4.5):
    """per sample: in view, facing, not behind the z-buffer Dz -> (any view <= 85 deg, good view <= good_deg)."""
    u, v, z, ok = project(P, c2w, K)
    ray = P - c2w[:3, 3]; r = np.linalg.norm(ray, axis=1); cosi = -(ray * N).sum(1) / np.maximum(r, 1e-9)
    ok &= (r < max_dist) & (cosi > np.cos(np.radians(85)))
    i = np.nonzero(ok)[0]
    vis = np.zeros(len(P), bool)
    if len(i):
        d = Dz[np.clip((v[i] / scale).astype(int), 0, Dz.shape[0] - 1), np.clip((u[i] / scale).astype(int), 0, Dz.shape[1] - 1)]
        vis[i] = z[i] <= d + np.maximum(tol, tol_rel * z[i])
    return vis, vis & (cosi > np.cos(np.radians(good_deg)))


def pca_normals(P, k=16, centre=None, up=None):
    from scipy.spatial import cKDTree
    _, nb = cKDTree(P).query(P, k=min(k, len(P)))
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    N = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
    if centre is not None:                                                 # outwards from the vertical axis / upwards
        r = P - centre; r -= np.outer(r @ up, up); flip = (N * r).sum(1) < 0
        topish = np.abs(N @ up) > 0.7; flip[topish] = (N[topish] @ up) < 0
        N[flip] *= -1
    return N


# ---------------------------------------------------------------- register (CPU)
def icp_plane(S, Ttree, Tpts, n, e1, e2, pivot, theta0, t0, iters=30, trim=0.8, max_d=0.08):
    """yaw (about n through pivot) + in-plane translation aligning S onto the target cloud."""
    th, tu, tv = theta0, t0[0], t0[1]
    for it in range(iters):
        T = plane_T(th, tu, tv, n, e1, e2, pivot); Y = apply(T, S)
        dd, jj = Ttree.query(Y, distance_upper_bound=max_d)
        m = np.isfinite(dd)
        if m.sum() < 20:
            break
        m &= dd <= np.quantile(dd[m], trim)
        a = np.stack([S[m] @ e1, S[m] @ e2], 1); b = np.stack([Tpts[jj[m]] @ e1, Tpts[jj[m]] @ e2], 1)
        pc = np.array([pivot @ e1, pivot @ e2]); a0, b0 = a - pc, b - pc
        ma, mb = a0.mean(0), b0.mean(0); Hm = (a0 - ma).T @ (b0 - mb)
        th_n = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        Rm = np.array([[np.cos(th_n), -np.sin(th_n)], [np.sin(th_n), np.cos(th_n)]]); tr = mb - Rm @ ma
        if abs(th_n - th) < 1e-5 and abs(tr[0] - tu) < 1e-5 and abs(tr[1] - tv) < 1e-5:
            th, tu, tv = th_n, tr[0], tr[1]; break
        th, tu, tv = th_n, tr[0], tr[1]
    return th, tu, tv


def components(Lb, n, d, e1, e2, cell, hmin, hmax):
    from scipy import ndimage
    h = Lb @ n + d; sel = (h > hmin) & (h < hmax); X = Lb[sel]
    if len(X) < 30:
        return []
    uv = np.stack([X @ e1, X @ e2], 1); lo = uv.min(0)
    ij = np.floor((uv - lo) / cell).astype(int); occ = np.zeros(ij.max(0) + 1, bool); occ[ij[:, 0], ij[:, 1]] = True
    lab, nl = ndimage.label(ndimage.binary_closing(occ, iterations=1) | occ, structure=np.ones((3, 3)))
    li = lab[ij[:, 0], ij[:, 1]]
    return [X[li == k] for k in range(1, nl + 1) if (li == k).sum() >= 30]


def register(args):
    from scipy.spatial import cKDTree
    meta, opts = load_objects(args.objects); assert meta is not None and len(opts), "no objects"
    n, d, e1, e2 = floor_frame(meta)
    os.makedirs(args.out_dir, exist_ok=True); rep = {"objects": args.objects, "recordings": {}}
    Lr = voxel(read_cloud(args.ref_pipeline, args.cloud_dir)[0], args.voxel)
    def floor_h(L, c, r0, r1):                                             # local floor height in an annulus round c
        r = np.linalg.norm(np.stack([(L - c) @ e1, (L - c) @ e2], 1), axis=1); h = L @ n + d
        f = h[(r > r0) & (r < r1) & (np.abs(h) < 0.08)]
        return float(np.median(f)) if len(f) > 50 else 0.0
    others = [o.rstrip("/") for o in args.others if os.path.realpath(o) != os.path.realpath(args.ref_pipeline)]
    for B in others:
        name = os.path.basename(os.path.dirname(B))
        Lb, src = read_cloud(B, args.cloud_dir)
        if Lb is None or not os.path.exists(os.path.join(B, "sim3.json")):
            print(f"[xrec] {name}: no LiDAR cloud / sim3 - skipped", flush=True); continue
        Lb = voxel(Lb, args.voxel); rr = {"cloud": src, "objects": {}}
        for o, X in zip(meta["objects"], opts):
            X = voxel(X[(X @ n + d) > args.min_h], args.voxel); hX = X @ n + d; hmax = float(np.percentile(hX, 99))
            c = X.mean(0); c -= (c @ n + d) * n; rad = float(np.percentile(np.linalg.norm(np.stack([(X - c) @ e1, (X - c) @ e2], 1), axis=1), 98))
            dz = floor_h(Lb, c, rad + 0.1, rad + 0.8) - floor_h(Lr, c, rad + 0.1, rad + 0.8)   # floor offset (B - ref)
            Tz = np.eye(4); Tz[:3, 3] = -dz * n                                # B -> floor-matched B
            near = apply(Tz, Lb[np.linalg.norm(np.stack([(Lb - c) @ e1, (Lb - c) @ e2], 1), axis=1) < rad + args.search_m])
            rin = np.linalg.norm(np.stack([(near - c) @ e1, (near - c) @ e2], 1), axis=1) < rad
            hin = near[rin] @ n + d; inb = (hin > args.min_h).sum()
            taller = float(((hin > hmax + 0.05) & (hin < hmax + 0.5)).sum() / max(inb, 1))
            comps = components(near, n, d, e1, e2, args.cell, args.min_h, hmax + 0.06)
            here = near[(np.linalg.norm(np.stack([(near - c) @ e1, (near - c) @ e2], 1), axis=1) < rad + 0.05) &
                        ((near @ n + d) > args.min_h) & ((near @ n + d) < hmax + 0.06)]
            cands = [("in_place", here)] if len(here) >= 30 else []
            for k, Cc in enumerate(comps):                                  # similar size + height to the reference object
                hc = float(np.percentile(Cc @ n + d, 95)); cc = Cc.mean(0)
                rc = float(np.percentile(np.linalg.norm(np.stack([(Cc - cc) @ e1, (Cc - cc) @ e2], 1), axis=1), 98))
                if abs(hc - hmax) < max(0.06, 0.25 * hmax) and 0.6 * rad < rc < 1.6 * rad:
                    cands.append((f"comp{k}", Cc))
            best = None
            ok_ = lambda b: (b is not None and min(b["fit_ref"], b["fit_other"]) >= args.min_fit and b["rmse_m"] <= args.max_rmse_m
                             and abs(b["dh_m"]) <= args.max_dh_m)
            for cname, Cc in cands:
                if cname != "in_place" and ok_(best) and best["cand"] == "in_place":
                    break                                                   # still where it was: never swap to a look-alike
                Ct = cKDTree(Cc); Xt = cKDTree(X); cc = Cc.mean(0); cc -= (cc @ n + d) * n
                t0 = np.array([(cc - c) @ e1, (cc - c) @ e2])
                for th0 in np.radians(np.arange(0, 360, args.yaw_step)):
                    # X (reference) moved onto B: T_ref->B; we store its inverse (B -> reference)
                    th, tu, tv = icp_plane(X, Ct, Cc, n, e1, e2, c, th0, t0, max_d=0.15)
                    th, tu, tv = icp_plane(X, Ct, Cc, n, e1, e2, c, th, (tu, tv), max_d=args.inlier_m * 2)
                    T = plane_T(th, tu, tv, n, e1, e2, c); Y = apply(T, X)
                    da = Ct.query(Y)[0]; Ti = np.linalg.inv(T); db = Xt.query(apply(Ti, Cc))[0]
                    fa, fb = float((da < args.inlier_m).mean()), float((db < args.inlier_m).mean())
                    rm = float(np.sqrt(np.mean(da[da < args.inlier_m] ** 2))) if (da < args.inlier_m).any() else 1.0
                    yaw = float(np.degrees(np.arctan2(np.sin(th), np.cos(th))))
                    sc = min(fa, fb) - args.yaw_penalty * abs(yaw) - (0.01 if cname != "in_place" else 0)
                    if best is None or sc > best["score"] + 1e-9:
                        best = {"score": sc, "cand": cname, "fit_ref": round(fa, 3), "fit_other": round(fb, 3), "rmse_m": round(rm, 4),
                                "yaw_deg": round(yaw, 2), "shift_m": round(float(np.hypot(tu, tv)), 3),
                                "dh_m": round(float(np.percentile(Cc @ n + d, 95) - np.percentile(hX, 95)), 3), "floor_dz_m": round(dz, 3),
                                "taller_share": round(taller, 3), "T_other_to_ref": (Ti @ Tz).tolist(), "n_other": len(Cc)}
            if best is None:
                rr["objects"][str(o["id"])] = {"accepted": False, "reason": "nothing of similar size within search radius"}
            else:
                why = [] if min(best["fit_ref"], best["fit_other"]) >= args.min_fit else ["shape differs (fit)"]
                why += [] if best["rmse_m"] <= args.max_rmse_m else ["residual"]
                why += [] if abs(best["dh_m"]) <= args.max_dh_m and taller <= args.max_taller else ["height differs"]
                best.update({"accepted": not why, "reason": ", ".join(why) or "ok"}); best.pop("score")
                rr["objects"][str(o["id"])] = best
            b = rr["objects"][str(o["id"])]
            print(f"[xrec] {name} o{o['id']}: {'ACCEPT' if b['accepted'] else 'reject'} {b.get('cand', '')} fit {b.get('fit_ref')}/"
                  f"{b.get('fit_other')} rmse {b.get('rmse_m')} yaw {b.get('yaw_deg')} shift {b.get('shift_m')} dh {b.get('dh_m')} ({b['reason']})", flush=True)
        rep["recordings"][B] = rr
    json.dump(rep, open(os.path.join(args.out_dir, "register.json"), "w"), indent=1)


# ---------------------------------------------------------------- views (CPU)
def views(args):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from PIL import Image
    meta, opts = load_objects(args.objects); n, d, e1, e2 = floor_frame(meta)
    reg = json.load(open(os.path.join(args.out_dir, "register.json")))
    Lref, _ = read_cloud(args.ref_pipeline, args.cloud_dir); Lref = voxel(Lref, args.zvox)
    refcams = cameras(args.ref_dataset, os.path.join(args.ref_pipeline, "sim3.json"))
    s_a, R_a, t_a = sim3(os.path.join(args.ref_pipeline, "sim3.json"))
    xv = os.path.join(args.out_dir, "xviews"); os.makedirs(os.path.join(xv, "images"), exist_ok=True)
    os.makedirs(os.path.join(xv, "masks"), exist_ok=True)
    inv = {"objects": {}}; ximgs, xcams, held, dbg = [], {}, [], []
    clouds = {}
    for o, X in zip(meta["objects"], opts):
        k = str(o["id"]); X0 = X[(X @ n + d) > args.min_h]
        acc = [(B, r["objects"][k]) for B, r in reg["recordings"].items() if r["objects"].get(k, {}).get("accepted")]
        parts = [X0]
        for B, r in acc:                                                     # B's object points in the reference frame
            if B not in clouds:
                clouds[B] = voxel(read_cloud(B, args.cloud_dir)[0], args.zvox)
            T = np.array(r["T_other_to_ref"]); Y = apply(T, clouds[B])
            c = X0.mean(0); keep = np.linalg.norm(Y - c, axis=1) < 1.5
            Y = Y[keep]; dd = cKDTree(X0).query(Y)[0]; parts.append(Y[(dd < 0.05) & ((Y @ n + d) > args.min_h)])
        P = voxel(np.concatenate(parts), args.sample_m)
        c = P.mean(0); c -= (c @ n + d) * n
        Nn = pca_normals(P, centre=c, up=n)
        near = np.linalg.norm(Lref - c, axis=1) < 6.0
        # reference coverage
        ngood = np.zeros(len(P), int); nany = np.zeros(len(P), int)
        for name, K, c2w, ds in refcams:
            u, v, z, ok = project(c[None], c2w, K)
            if not ok[0] or z[0] > 5:
                continue
            Dz = zbuffer(Lref[near], c2w, K, args.zscale, args.zvox)
            a, g = visible(P, Nn, c2w, K, Dz, args.zscale, args.good_deg); nany += a; ngood += g
        faint = ngood < args.ref_min
        orep = {"samples": int(len(P)), "ref_good_share": round(float((~faint).mean()), 3), "ref_unseen_share": round(float((nany == 0).mean()), 3),
                "faint_share": round(float(faint.mean()), 3), "recordings": {}}
        others_obj = Lref[near & (cKDTree(X0).query(Lref, distance_upper_bound=0.04)[0] > 0.04)]   # reference scene minus this object
        gain_all = np.zeros(len(P), int)
        for B, r in acc:
            bn = os.path.basename(os.path.dirname(B)); T = np.array(r["T_other_to_ref"]); Ti = np.linalg.inv(T)
            Pb = apply(Ti, P); Nb = Nn @ Ti[:3, :3].T                        # samples where the object stood in B
            nearb = np.linalg.norm(clouds[B] - apply(Ti, c[None])[0], axis=1) < 6.0
            LB = clouds[B][nearb]; bcams = cameras([os.path.join(B, "dataset")], os.path.join(B, "sim3.json"))
            gainB = np.zeros(len(P), int); sel = []
            for name, K, c2w, ds in bcams:
                u, v, z, ok = project(apply(Ti, c[None]), c2w, K)
                if not ok[0] or z[0] > 5:
                    continue
                Dz = zbuffer(LB, c2w, K, args.zscale, args.zvox)
                a, g = visible(Pb, Nb, c2w, K, Dz, args.zscale, args.good_deg)
                gm = g & faint; gainB += gm
                if gm.sum() >= args.min_gain:
                    sel.append((name, K, c2w, ds, int(gm.sum())))
            gain_all += gainB
            orep["recordings"][bn] = {"faint_samples_seen": round(float(((gainB > 0) & faint).sum() / max(faint.sum(), 1)), 3),
                                      "frames": len(sel), "yaw_deg": r["yaw_deg"], "shift_m": r["shift_m"]}
            print(f"[xrec] o{k} <- {bn}: covers {orep['recordings'][bn]['faint_samples_seen']:.0%} of the faint samples, {len(sel)} frames", flush=True)
            for j, (name, K, c2w, ds, ng) in enumerate(sel[::args.every]):
                # mask at full res: object surface in front of B's LiDAR and of the reference scene's other geometry
                fx, fy, cx, cy, W, H = K
                Dob = zbuffer(Pb, c2w, K, 1, args.sample_m * 1.6)
                Db = zbuffer(LB, c2w, K, 1, args.zvox)
                c2w_ref = T @ c2w                                            # B camera seeing the object where it is in A
                Dr = zbuffer(others_obj, c2w_ref, K, 1, args.zvox)
                m = np.isfinite(Dob) & (Dob <= Db + 0.04) & (Dob <= Dr - 0.03)
                m = ndimage.binary_erosion(ndimage.binary_closing(m, iterations=2), iterations=args.erode)
                if m.sum() < args.min_px:
                    continue
                img = np.asarray(Image.open(os.path.join(ds, "images", name)).convert("RGB"))
                if (img[m].min(1) >= 245).mean() > args.max_sat or (img[m].max(1) <= 10).mean() > args.max_sat:
                    continue                                                 # blown out / black (blinking lights)
                nm = f"{bn}__o{k}__{name}"
                dst = os.path.join(xv, "images", nm)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(ds, "images", name), dst)
                Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(xv, "masks", os.path.splitext(nm)[0] + ".png"))
                Rwc_d = R_a.T @ c2w_ref[:3, :3]; C_d = R_a.T @ (c2w_ref[:3, 3] - t_a) / s_a     # metres -> reference colmap
                cid = xcams.setdefault(K, len(xcams) + 1)
                ximgs.append((nm, cid, Rwc_d, C_d))
                if j % args.holdout == args.holdout // 2:
                    held.append(nm)
        inv["objects"][k] = orep; dbg.append((k, P, Nn, c, ngood, gain_all))
        print(f"[xrec] o{k}: {len(P)} samples, ref good {orep['ref_good_share']:.0%}, faint {orep['faint_share']:.0%}", flush=True)
    try:                                                                   # per object: 4 side views of the samples
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(dbg), 4, figsize=(16, 3.4 * len(dbg)), squeeze=False)
        for i, (k, P, Nn, c, ng, ga) in enumerate(dbg):
            col = np.where(ng >= args.ref_min, 0, np.where(ga > 0, 1, 2))
            for j, az in enumerate((0, 90, 180, 270)):
                a = np.radians(az); dv = np.cos(a) * e1 + np.sin(a) * e2; rt = np.cross(n, dv)
                face = (Nn @ dv) > -0.2; q = P[face] - c
                ax[i, j].scatter(q @ rt, q @ n, c=np.array(["tab:green", "tab:blue", "tab:red"])[col[face]], s=1)
                ax[i, j].set_aspect("equal"); ax[i, j].set_title(f"o{k} from az {az}: green ref-seen, blue other rec., red faint", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "coverage.png"), dpi=70); plt.close(fig)
    except Exception as ex:  # noqa: BLE001
        print("[xrec] coverage plot failed:", ex)
    from fuse_sequences import write_model
    write_model(os.path.join(xv, "sparse", "0"), {cid: ("PINHOLE", K[4], K[5], list(K[:4])) for K, cid in xcams.items()}, ximgs)
    inv.update({"extra_views": len(ximgs), "held_out": held, "args": vars(args)})
    json.dump(inv, open(os.path.join(args.out_dir, "inventory.json"), "w"), indent=1, default=str)
    print("XREC_VIEWS", json.dumps({k: v for k, v in inv.items() if k not in ("held_out", "args")}), flush=True)


# ---------------------------------------------------------------- base (CPU)
def base(args):
    """the input itself as a build (no rows added / removed): meta / src / priors + zone = input rows in each object's
    box (min-area footprint + --zone-m, up to its top + --zone-m), the rows xfit trains and zone_budget.py relaxes."""
    from solidify_unseen import read_vertex
    from object_priors import Obj
    meta, opts = load_objects(args.objects); n, d, _, _ = floor_frame(meta)
    s, R, t = sim3(args.sim3_json); v = read_vertex(args.ply); N = len(v)
    G = s * (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T) + t
    zone = np.zeros(N, bool); rep = {"method": ["xrec_base"], "objects": []}
    for o, X in zip(meta["objects"], opts):
        ob = Obj(X, n, d); hmax = float(np.percentile(ob.loc(X)[:, 2], 99)); Lg = ob.loc(G); zh = ob.half + args.zone_m
        inz = (np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > -0.02) & (Lg[:, 2] < hmax + args.zone_m)
        zone |= inz; rep["objects"].append({"id": o["id"], "zone_rows": int(inz.sum())})
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "meta.npz"), keep=np.ones(N, bool), mod=np.zeros(N, bool),
                        removed_near_surface=np.zeros(N, bool), cell=0.02, sim3=np.array([s]), n_in=N)
    np.savez_compressed(os.path.join(args.out_dir, "zone.npz"), zone=zone)
    np.savez_compressed(os.path.join(args.out_dir, "src.npz"), src=np.arange(N))
    rep.update({"gaussians_in": N, "kept": N, "removed": 0, "snapped": 0, "added": 0, "zone_rows": int(zone.sum())})
    json.dump(rep, open(os.path.join(args.out_dir, "priors.json"), "w"), indent=1)
    print("XREC_BASE", json.dumps({k: v for k, v in rep.items() if k != "objects"}), flush=True)


# ---------------------------------------------------------------- xfit (GPU)
def xfit(args):
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, ssim, psnr
    from solidify_unseen import all_frames
    from PIL import Image
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"
    vtx, p0 = load_ply(args.ply, dev); deg = sh_deg(p0); N = len(p0["means"])
    M = np.load(args.meta); keep = M["keep"]; nk = int(keep.sum()); m_new = N - nk
    zin = np.load(args.zone)["zone"]
    src = np.concatenate([np.nonzero(keep)[0], np.full(m_new, -1)])
    tm_np = np.concatenate([zin[keep], np.ones(m_new, bool)])                 # trainable rows: object zones + added rows
    tm = torch.tensor(tm_np, device=dev); slow = tm & torch.tensor(src >= 0, device=dev)
    tr = all_frames(args.dataset)
    gt_path = lambda n: next(os.path.join(ds, "images", n) for ds in args.dataset if os.path.exists(os.path.join(ds, "images", n)))
    rgb = [load_rgb(gt_path(n), dev) for n, _, _ in tr]
    inv = json.load(open(os.path.join(os.path.dirname(args.xviews.rstrip("/")), "inventory.json")))
    held = set(inv.get("held_out", []))
    xs = frames(args.xviews, 0, "all")
    xtr = [f for f in xs if f[0] not in held]; xte = [f for f in xs if f[0] in held]
    def xload(f):
        img = load_rgb(os.path.join(args.xviews, "images", f[0]), dev)
        mk = np.asarray(Image.open(os.path.join(args.xviews, "masks", os.path.splitext(f[0])[0] + ".png"))) > 127
        return img, torch.tensor(mk, device=dev)
    xd = [xload(f) for f in xtr]; xh = [xload(f) for f in xte]
    print(f"[xfit] {N:,} gaussians, trainable {int(tm.sum()):,} ({m_new:,} added); {len(tr)} reference frames, "
          f"{len(xtr)} extra views (+{len(xte)} held out)", flush=True)
    centres = np.stack([c[:3, 3] for _, _, c in tr]); scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    s_ = float(M["sim3"][0]) if "sim3" in M.files else 1.0; cell_d = 0.02 / s_
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    lr = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    aff = torch.nn.Parameter(torch.zeros(len(xtr) + len(xh), 3, 4, device=dev))   # per view colour affine (delta to identity)
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params] +
                           [{"params": [aff], "lr": 2e-3, "name": "aff"}], eps=1e-15)
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    I3 = torch.eye(3, 4, device=dev)
    def colour(img, i):
        A = I3 + aff[i]; return (img @ A[:, :3].T + A[:, 3]).clamp(0, 1)
    def mloss(img, gt, mk):
        return (img - gt).abs().mean(-1)[mk].mean()
    @torch.no_grad()
    def held_score(prm, fit_aff=True):
        """held-out extra views: masked PSNR, after a closed-form colour affine fitted on the masked pixels (the colour
        transfer between recordings is not what is being scored)."""
        out = []
        for f, (g, mk) in zip(xte, xh):
            img, _, _ = Rv.render(prm, to_t(np.linalg.inv(f[2]), dev), *cam_args(f[1]), sh_degree=deg)
            a = img[mk]; b = g.float()[mk] / 255
            if fit_aff and len(a) > 50:
                Aa = torch.cat([a, torch.ones(len(a), 1, device=dev)], 1); Wt = torch.linalg.lstsq(Aa, b).solution; a = (Aa @ Wt).clamp(0, 1)
            out.append(float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10))))
        return out
    h0 = held_score(p0)
    t0, log, order, xorder = time.time(), [], [], []
    for step in range(1, args.iters + 1):
        wu = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = (lr[g["name"]] if g["name"] in lr else 2e-3) * wu * (0.1 ** (step / args.iters) if g["name"] == "means" else 1.0)
        if not order:
            order = list(np.random.permutation(len(tr)))
        k = int(order.pop()); name, cam, c2w = tr[k]
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = rgb[k].float() / 255
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        lx = torch.zeros((), device=dev)
        if xtr and step % args.x_every == 0:
            for _ in range(args.x_batch):
                if not xorder:
                    xorder = list(np.random.permutation(len(xtr)))
                j = int(xorder.pop()); f = xtr[j]; g_, mk = xd[j]
                im, _, _ = Rv.render(params, to_t(np.linalg.inv(f[2]), dev), *cam_args(f[1]), sh_degree=deg)
                lx = lx + mloss(colour(im, j), g_.float() / 255, mk) / args.x_batch
            loss = loss + args.lam_x * lx + args.lam_aff * (aff ** 2).mean()
        la = (((params["means"][slow] - p0["means"][slow]) / cell_d) ** 2).mean() + ((params["scales"][slow] - p0["scales"][slow]) ** 2).mean() \
            + ((params["opacities"][slow] - p0["opacities"][slow]) ** 2).mean()
        loss = loss + args.lam_anchor * la
        opt.zero_grad(set_to_none=True); loss.backward()
        for prm in params.values():
            if prm.grad is not None:
                prm.grad[~torch.isfinite(prm.grad)] = 0; prm.grad[~tm] = 0
        opt.step()
        if step % args.log_every == 0 or step == 1:
            e = {"step": step, "psnr": round(psnr(img.detach(), gt), 2), "x_l1": round(float(lx), 4), "anchor": round(float(la), 4)}
            log.append(e); print(f"[xfit] {e} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        for k_ in params:
            params[k_][~tm] = p0[k_][~tm]
        pv = [psnr(Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n, cam, c2w) in list(enumerate(tr))[::10]]
        pv0 = [psnr(Rv.render(p0, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n, cam, c2w) in list(enumerate(tr))[::10]]
    h1 = held_score(params)
    save_ply(args.out, vtx, params)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_src.npz", src=src)
    per = {}
    for f, a, b in zip(xte, h0, h1):
        key = f[0].split("__")[0] + "__" + f[0].split("__")[1]; per.setdefault(key, []).append((a, b))
    res = {"gaussians": N, "trainable": int(tm.sum()), "extra_views": len(xtr), "held_out_views": len(xte), "iters": args.iters,
           "heldout_masked_psnr_before": round(float(np.mean(h0)), 3) if h0 else None,
           "heldout_masked_psnr_after": round(float(np.mean(h1)), 3) if h1 else None,
           "heldout_per_object": {k: {"n": len(v), "before": round(float(np.mean([x[0] for x in v])), 2), "after": round(float(np.mean([x[1] for x in v])), 2)} for k, v in per.items()},
           "train_psnr_every10_before": round(float(np.mean(pv0)), 3), "train_psnr_every10_after": round(float(np.mean(pv)), 3),
           "seconds": round(time.time() - t0), "curve": log, "args": vars(args)}
    json.dump(res, open(os.path.splitext(args.out)[0] + "_xfit.json", "w"), indent=1)
    print("XREC_XFIT", json.dumps({k: v for k, v in res.items() if k not in ("curve", "args")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register")
    r.add_argument("--objects", required=True); r.add_argument("--ref-pipeline", required=True)
    r.add_argument("--others", nargs="+", required=True); r.add_argument("--out-dir", required=True)
    r.add_argument("--voxel", type=float, default=0.01); r.add_argument("--cell", type=float, default=0.02)
    r.add_argument("--min-h", type=float, default=0.03); r.add_argument("--search-m", type=float, default=1.0)
    r.add_argument("--yaw-step", type=float, default=15.0); r.add_argument("--inlier-m", type=float, default=0.02)
    r.add_argument("--min-fit", type=float, default=0.6); r.add_argument("--max-rmse-m", type=float, default=0.015)
    r.add_argument("--max-dh-m", type=float, default=0.03); r.add_argument("--max-taller", type=float, default=0.1)
    r.add_argument("--yaw-penalty", type=float, default=0.0005, help="score per degree of rotation (near-ties keep the pose)")
    r.add_argument("--cloud-dir", default=None, help="<dir>/<sequence>/lidar_static.ply preferred over the pipeline's cloud")
    v = sp.add_parser("views")
    v.add_argument("--objects", required=True); v.add_argument("--ref-pipeline", required=True); v.add_argument("--cloud-dir", default=None)
    v.add_argument("--ref-dataset", nargs="+", required=True); v.add_argument("--out-dir", required=True)
    v.add_argument("--min-h", type=float, default=0.045, help="object samples above this height (floor band excluded)")
    v.add_argument("--sample-m", type=float, default=0.01); v.add_argument("--max-sat", type=float, default=0.2)
    v.add_argument("--zvox", type=float, default=0.02); v.add_argument("--zscale", type=int, default=4)
    v.add_argument("--good-deg", type=float, default=60.0); v.add_argument("--ref-min", type=int, default=2)
    v.add_argument("--min-gain", type=int, default=150, help="faint samples (1 cm) a frame must see well to be used")
    v.add_argument("--min-px", type=int, default=400); v.add_argument("--erode", type=int, default=2)
    v.add_argument("--every", type=int, default=1); v.add_argument("--holdout", type=int, default=5)
    b = sp.add_parser("base")
    b.add_argument("--ply", required=True); b.add_argument("--objects", required=True); b.add_argument("--sim3-json", required=True)
    b.add_argument("--out-dir", required=True); b.add_argument("--zone-m", type=float, default=0.12)
    x = sp.add_parser("xfit")
    x.add_argument("--ply", required=True); x.add_argument("--meta", required=True); x.add_argument("--zone", required=True)
    x.add_argument("--dataset", nargs="+", required=True); x.add_argument("--xviews", required=True); x.add_argument("--out", required=True)
    x.add_argument("--iters", type=int, default=3000); x.add_argument("--x-every", type=int, default=1); x.add_argument("--x-batch", type=int, default=1)
    x.add_argument("--lam-x", type=float, default=1.0); x.add_argument("--lam-aff", type=float, default=1.0)
    x.add_argument("--lam-anchor", type=float, default=0.01); x.add_argument("--display-scale", type=float, default=1.8)
    x.add_argument("--log-every", type=int, default=250)
    a = ap.parse_args()
    {"register": register, "views": views, "base": base, "xfit": xfit}[a.cmd](a)


if __name__ == "__main__":
    main()
