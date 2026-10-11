"""Complete the unseen backs of free-standing objects with structure-conditioned TRELLIS appearance (GPU).

TRELLIS (Microsoft, CVPR 2025, MIT) generates a 3D asset in two stages: a sparse 64^3 voxel structure, then
a structured latent on those voxels that decodes to gaussians. We skip stage 1 and hand stage 2 our MEASURED
geometry, so no registration is needed and the generated shape cannot drift:
  1. Structure: the object's 2.5D surface from its metric LiDAR points (complete_objects.object_surface: flat
     tops, vertical walls, minus samples a camera sees through) plus its footprint as a bottom face, in an
     object frame (PCA axes in the floor plane, +z = floor normal = TRELLIS's up), scaled so the longest
     side spans the unit cube exactly like TRELLIS's training normalisation; voxelised at 64^3.
  2. Conditioning: the --views best real training images of the object (visible surface area x unoccluded
     share, spread in azimuth around it), masked by the projected visible surface samples (z-buffer against
     the base splat's rendered depth), cropped square with TRELLIS's 1.2x padding, black background, 518 px.
  3. TRELLIS sample_slat(cond, coords) with multi-image conditioning (--mode stochastic|multidiffusion),
     decode_slat -> gaussians in the unit cube.
  4. merge (3DGRUT venv): undo the normalisation exactly (cube -> object frame -> metric -> dataset frame via
     sim3: positions, rotations, scales), opacity -> density logit (both sigmoid), colour = SH DC in both codebases
     (rgb = 0.5 + C0 * dc), features_specular = 0. Keep a generated gaussian ONLY if (a) its nearest surface
     sample is measured (LiDAR within --support-m) and seen by no training camera (same test as
     complete_objects.py: LiDAR depth + base-splat depth, incidence), and (b) its centre is hidden from every
     training view (behind the base splat's rendered surface) — seen faces keep the real splat.
     Colour calibration (--color-match, default on): the generated gaussians on SEEN faces (discarded) are
     compared with the real splat there; a per-object, per-channel gain/bias fit maps TRELLIS's colours to the
     splat's exposure and is applied to the kept ones (only generated gaussians change; real ones never do).
     The seen-face error before/after is reported as a measure of TRELLIS's faithfulness.
  5. Gaussians appended to the base checkpoint (same handling as complete_objects.py); <out>_unseen.npz
     (unseen measured samples, dataset frame) for object_back_masks.py; <out>_report.json.
  orbit / preview: novel orbit cameras around each object (no real view shows the backs) as a COLMAP dataset
  with placeholder images for eval_views.py --no-metrics, and a base-vs-completed montage of the views that
  changed most.

    python scripts/trellis_complete.py generate --objects objects --dataset train/ --splat-depth-dir sd/ \
        --sim3-json sim3.json --out-dir gen/                                   # venv-trellis
    python scripts/trellis_complete.py merge --gen-dir gen/ --objects objects --checkpoint ckpt.pt \
        --dataset train/ --splat-depth-dir sd/ --sim3-json sim3.json --out completed.pt   # 3DGRUT venv
    python scripts/trellis_complete.py orbit --objects objects --dataset train/ --sim3-json sim3.json --out orbit/
    python scripts/trellis_complete.py preview --orbit orbit/ --renders base=r0/ trellis=r1/ --out preview.jpg
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covis_masks import cams_of
from complete_objects import object_surface

C0 = 0.28209479177387814
RES = 64


def load_common(a):
    meta = json.load(open(a.objects + ".json")); pts = np.load(a.objects + ".npz")
    n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"])
    j = json.load(open(a.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    return meta, pts, n, d, s, R, t


def obj_axes(P, n, d):
    """object_surface's frame: footprint centroid c0, PCA axis a1, a2 = n x a1 (right-handed [a1 a2 n])."""
    h = P @ n + d; base = P - np.outer(h, n); c0 = base.mean(0)
    e = np.linalg.svd((base - c0) - np.outer((base - c0) @ n, n), full_matrices=False)[2]
    a1 = e[0] - (e[0] @ n) * n; a1 /= np.linalg.norm(a1)
    return c0, np.stack([a1, np.cross(n, a1), n], 1)


class Vis:
    """Observedness against the training cameras (the complete_objects.py test, reusable)."""

    def __init__(self, a, s):
        import torch
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.cams = cams_of(a.dataset)[::a.frame_stride]
        self.depth = [np.load(c[9]).astype(np.float32) if os.path.exists(c[9]) else None for c in self.cams]
        self.sdep = [None] * len(self.cams)
        if a.splat_depth_dir:
            f = [os.path.join(a.splat_depth_dir, os.path.splitext(c[0])[0] + "_splatdepth.npy") for c in self.cams]
            miss = sum(not os.path.exists(x) for x in f)
            assert miss == 0, f"{miss} training frames have no rendered splat depth in {a.splat_depth_dir}"
            self.sdep = [np.load(x).astype(np.float32) for x in f]
        self.s = s; self.cosmax = np.cos(np.deg2rad(a.max_incidence)); self.tol = a.depth_tol_m / s; self.stol = a.splat_tol_rel

    def T(self, x):
        return self.torch.tensor(np.asarray(x), device=self.dev, dtype=self.torch.float64)

    def project(self, X, cam):
        name, W, H, fx, fy, cx, cy, Rcw, tcw, _ = cam
        Rc = self.T(Rcw); tc = self.T(tcw); Pc = X @ Rc.T + tc; z = Pc[:, 2]
        u = self.torch.floor(fx * Pc[:, 0] / z + cx).long(); v = self.torch.floor(fy * Pc[:, 1] / z + cy).long()
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        Cw = -Rc.T @ tc
        return Pc, u, v, ok, Cw

    def observed(self, Xd, Nd):
        """per sample: seen by some camera, and seen THROUGH by some camera (free space). Same rules as
        complete_objects.observed: in view + facing + LiDAR-consistent, or in view + not edge-on + not behind
        the base splat's rendered surface."""
        torch = self.torch
        X = self.T(Xd); Nn = self.T(Nd)
        seen = torch.zeros(len(X), dtype=torch.bool, device=self.dev); thru = torch.zeros_like(seen)
        for cam, D, SD in zip(self.cams, self.depth, self.sdep):
            Pc, u, v, ok, Cw = self.project(X, cam)
            vd = Cw[None] - X; vd = vd / torch.linalg.norm(vd, dim=1, keepdim=True); cs = (vd * Nn).sum(1)
            r = torch.linalg.norm(Pc, dim=1)
            if SD is not None:
                ia = torch.nonzero(ok & (cs.abs() > self.cosmax), as_tuple=True)[0]
                if len(ia):
                    ds_ = self.T(SD)[v[ia], u[ia]]
                    seen[ia[(ds_ <= 0) | (r[ia] < ds_ * (1 + self.stol))]] = True
            if D is not None:
                Dt = self.T(D)
                ia = torch.nonzero(ok, as_tuple=True)[0]
                if len(ia):
                    dt = Dt[v[ia], u[ia]]
                    thru[ia[(dt > 0) & (dt - r[ia] > torch.clamp(0.05 * dt, min=2 * self.tol))]] = True
            idx = torch.nonzero(ok & (cs > self.cosmax), as_tuple=True)[0]
            if D is not None and len(idx):
                dt = Dt[v[idx], u[idx]]
                idx = idx[(dt <= 0) | ((r[idx] - dt).abs() < torch.clamp(0.03 * dt, min=self.tol))]
            seen[idx] = True
        return seen.cpu().numpy(), thru.cpu().numpy()

    def shows(self, Xd):
        """per point: would appear in some training view (in view and not behind the base splat's surface)."""
        torch = self.torch
        X = self.T(Xd); out = torch.zeros(len(X), dtype=torch.bool, device=self.dev)
        for cam, D, SD in zip(self.cams, self.depth, self.sdep):
            ref = SD if SD is not None else D
            if ref is None:
                continue
            Pc, u, v, ok, _ = self.project(X, cam)
            ia = torch.nonzero(ok, as_tuple=True)[0]
            if len(ia):
                dr = self.T(ref)[v[ia], u[ia]]; r = torch.linalg.norm(Pc[ia], dim=1)
                out[ia[(dr <= 0) | (r < dr * (1 + self.stol))]] = True
        return out.cpu().numpy()


def surface(o, pts, n, d, a, to_ds, R, vis):
    """measured surface samples of one object: metric pos/normals, dataset pos/normals, kind, supported, seen."""
    from scipy.spatial import cKDTree
    P = pts[f"obj{o['id']}"]
    pos, nor, kind, _, _, _ = object_surface(P, n, d, a.hcell, a.cell)
    posd, nord = to_ds(pos), nor @ R
    seen, thru = vis.observed(posd, nord)
    sup = cKDTree(P).query(pos)[0] < a.support_m
    k = ~thru
    return dict(P=P, pos=pos[k], nor=nor[k], kind=kind[k], posd=posd[k], nord=nord[k], seen=seen[k], sup=sup[k],
                dropped_free=int(thru.sum()))


# ----------------------------------------------------------------------------------------------- generate
def pick_views(S, vis, a, c_obj, A):
    """score every training view of the object, return the chosen [(score, cam index, mask, azimuth)]."""
    import cv2
    torch = vis.torch
    X = vis.T(S["posd"]); Nn = vis.T(S["nord"]); cands = []
    for ci, (cam, SD, D) in enumerate(zip(vis.cams, vis.sdep, vis.depth)):
        name, W, H, fx, fy, cx, cy, Rcw, tcw, _ = cam
        Pc, u, v, ok, Cw = vis.project(X, cam)
        z = Pc[:, 2]
        front = z > 0.05
        if front.sum() < 10:
            continue
        uf = fx * Pc[front, 0] / z[front] + cx; vf = fy * Pc[front, 1] / z[front] + cy
        trunc = ((uf < 0) | (uf >= W) | (vf < 0) | (vf >= H)).float().mean().item()
        if trunc > a.max_trunc:
            continue
        vd = Cw[None] - X; vd = vd / torch.linalg.norm(vd, dim=1, keepdim=True)
        facing = ok & ((vd * Nn).sum(1) > vis.cosmax)
        ia = torch.nonzero(facing, as_tuple=True)[0]
        if len(ia) < a.min_samples:
            continue
        ref = SD if SD is not None else D
        r = torch.linalg.norm(Pc[ia], dim=1)
        if ref is not None:
            dr = vis.T(ref)[v[ia], u[ia]]
            visb = (dr > 0) & ((r - dr).abs() < torch.clamp(a.mask_tol_rel * dr, min=vis.tol))
        else:
            visb = torch.ones_like(r, dtype=torch.bool)
        iv = ia[visb]
        if len(iv) < a.min_samples:
            continue
        uu, vv, rr = u[iv].cpu().numpy(), v[iv].cpu().numpy(), r[visb].cpu().numpy()
        zmed = float(np.median(z[iv].cpu().numpy()))
        k = int(np.clip(np.ceil(fx * a.cell / vis.s / zmed), 1, 15))                 # px per surface-sample step
        rmap = np.full((H, W), np.inf, np.float32); np.minimum.at(rmap, (vv, uu), rr.astype(np.float32))
        rmap = cv2.erode(rmap, np.ones((2 * k + 1, 2 * k + 1), np.uint8))           # min filter: sample discs
        m = np.isfinite(rmap)
        if ref is not None:                                                         # trim background bleeding
            dref = ref.astype(np.float32)
            m &= (dref > 0) & (np.abs(dref - rmap) < np.maximum(a.mask_tol_rel * rmap, vis.tol))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        ys, xs = np.nonzero(m)
        if len(ys) == 0 or max(np.ptp(xs), np.ptp(ys)) < a.min_crop_px:
            continue
        score = len(iv) * (len(iv) / len(ia))                                       # visible area x unoccluded share
        Cm = Cw.cpu().numpy()
        cl = (Cm - c_obj) @ A                                                       # camera in object frame (dataset units ok: angle)
        cands.append((score, ci, m.astype(bool), float(np.degrees(np.arctan2(cl[1], cl[0])))))
    cands.sort(key=lambda x: -x[0])
    chosen = []
    for sep in (360.0 / (2 * a.views), 360.0 / (4 * a.views), 0.0):                # spread in azimuth, relax if needed
        for c in cands:
            if len(chosen) >= a.views:
                break
            if any(c[1] == q[1] for q in chosen):
                continue
            if all(abs((c[3] - q[3] + 180) % 360 - 180) >= sep for q in chosen):
                chosen.append(c)
    return chosen, len(cands)


def crop(img, m, pad=1.2, size=518):
    """TRELLIS preprocess: object pixels only (black elsewhere), square crop 1.2x the mask bbox, 518 px."""
    from PIL import Image
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = int(np.ceil(max(np.ptp(xs), np.ptp(ys)) * pad / 2)) + 1
    rgb = img.astype(np.float32) * m[..., None]
    H, W = m.shape
    out = np.zeros((2 * half, 2 * half, 3), np.float32)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, W), min(y0 + 2 * half, H)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)


def cmd_generate(a):
    import torch
    from PIL import Image
    os.environ.setdefault("ATTN_BACKEND", "xformers"); os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s)
    sys.path.insert(0, a.trellis_dir)
    from trellis.pipelines import TrellisImageTo3DPipeline
    model = a.model
    if not os.path.exists(os.path.join(model, "pipeline.json")):
        from huggingface_hub import snapshot_download
        model = snapshot_download(model)
    pipe = TrellisImageTo3DPipeline.from_pretrained(model); pipe.cuda()
    steps = a.steps or pipe.slat_sampler_params.get("steps", 25)
    os.makedirs(a.out_dir, exist_ok=True)
    ids = set(int(x) for x in a.ids.split(",")) if a.ids else None
    log = []
    for o in meta["objects"]:
        if ids is not None and o["id"] not in ids:
            continue
        S = surface(o, pts, n, d, a, to_ds, R, vis)
        c0, A = obj_axes(S["P"], n, d)
        top = S["pos"][S["kind"] == 0]
        bottom = top - np.outer(top @ n + d - a.bottom_h, n)                     # footprint as a bottom face
        Q = (np.concatenate([S["pos"], bottom]) - c0) @ A                        # object frame, metres
        lo, hi = Q.min(0), Q.max(0); cb = (lo + hi) / 2; k = (1 - a.margin) / float((hi - lo).max())
        c = c0 + A @ cb                                                          # metric centre of the cube
        q = (Q - cb) * k
        vox = np.unique(np.clip(np.floor((q + 0.5) * RES), 0, RES - 1).astype(np.int32), axis=0)
        # conditioning views (choose in the dataset frame; azimuth from the object frame)
        c_ds = to_ds(c[None])[0]; A_ds = R.T @ A                                   # object axes in the dataset frame
        chosen, ncand = pick_views(S, vis, a, c_ds, A_ds)
        if not chosen:
            print(f"[trellis] object {o['id']}: no usable view, skipped", flush=True); continue
        ims = []
        for sc, ci, m, az in chosen:
            img = np.asarray(Image.open(os.path.join(a.dataset, "images", vis.cams[ci][0])).convert("RGB"))
            ims.append(crop(img, m))
        cond = pipe.get_cond(ims); cond["neg_cond"] = cond["neg_cond"][:1]
        coords = torch.tensor(np.c_[np.zeros(len(vox), np.int32), vox], dtype=torch.int32, device="cuda")
        torch.manual_seed(a.seed)
        prm = {"steps": steps}
        if a.cfg is not None:
            prm["cfg_strength"] = a.cfg
        with torch.no_grad():
            with pipe.inject_sampler_multi_image("slat_sampler", len(ims), steps, mode=a.mode):
                slat = pipe.sample_slat(cond, coords, prm)
            g = pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]
        out = os.path.join(a.out_dir, f"obj{o['id']}")
        np.savez_compressed(out + ".npz", xyz=g.get_xyz.float().cpu().numpy(), scale=g.get_scaling.float().cpu().numpy(),
                            rot=torch.nn.functional.normalize(g.get_rotation.float(), dim=1).cpu().numpy(),
                            opacity=g.get_opacity.float().cpu().numpy()[:, 0], dc=g._features_dc.float().cpu().numpy()[:, 0],
                            centre=c, axes=A, k=k, voxels=vox, id=o["id"])
        tiles = [np.asarray(x.resize((259, 259))) for x in ims]
        Image.fromarray(np.hstack(tiles)).save(out + "_cond.jpg")
        log.append({"id": o["id"], "voxels": int(len(vox)), "gaussians": int(g.get_xyz.shape[0]), "scale_k": k,
                    "extent_m": (hi - lo).round(3).tolist(), "candidate_views": ncand,
                    "views": [{"image": vis.cams[ci][0], "score": round(float(sc), 1), "azimuth_deg": round(az, 1),
                               "mask_px": int(m.sum())} for sc, ci, m, az in chosen]})
        print(f"[trellis] object {o['id']}: {len(vox)} voxels (cube side {1 / k:.2f} m), {len(chosen)}/{ncand} views "
              f"(az {[round(x[3]) for x in chosen]}), {g.get_xyz.shape[0]:,} gaussians", flush=True)
    json.dump({"objects": log, "params": {k_: v for k_, v in vars(a).items() if k_ != "func"}},
              open(os.path.join(a.out_dir, "generate.json"), "w"), indent=2)


# ----------------------------------------------------------------------------------------------- merge
def qmat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def cmd_merge(a):
    import torch
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    meta, pts, n, d, s, R, t = load_common(a)
    assert np.linalg.det(R) > 0
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    G = ck["positions"].detach().double().numpy(); N0 = len(G)
    opac = torch.sigmoid(ck["density"].detach()).numpy().ravel(); alb = ck["features_albedo"].detach().numpy()
    gtree = cKDTree(G)
    objs = {o["id"]: o for o in meta["objects"]}
    add = {k: [] for k in ("pos", "rot", "scale", "dens", "dc")}; rep = []; unseen_all = []
    for f in sorted(glob.glob(os.path.join(a.gen_dir, "obj*.npz"))):
        g = np.load(f); oid = int(g["id"]); o = objs[oid]
        S = surface(o, pts, n, d, a, to_ds, R, vis)
        c, A, k = g["centre"], g["axes"], float(g["k"])
        Xm = c + (g["xyz"].astype(np.float64) / k) @ A.T                         # cube -> object frame -> metric
        Xd = to_ds(Xm)
        M = R.T @ A                                                              # object frame -> dataset rotation
        Rd = M[None] @ qmat(g["rot"].astype(np.float64))
        sc = g["scale"].astype(np.float64) / (k * s)
        op = g["opacity"].astype(np.float64); rgb = 0.5 + C0 * g["dc"].astype(np.float64)
        hgt = Xm @ n + d
        vox_m = 1.0 / (RES * k)                                                  # voxel side, metres
        base_ok = (op > a.min_opacity) & (hgt > a.min_h) & (sc.max(1) * s < a.max_scale_vox * vox_m)
        dist, si = cKDTree(S["pos"]).query(Xm)
        near = dist < max(a.keep_dist_vox * vox_m, a.cell * 2)
        tgt = S["sup"] & ~S["seen"]                                              # measured, never seen
        cand = base_ok & near & tgt[si]
        shows = np.zeros(len(Xm), bool)
        ic = np.nonzero(cand)[0]
        if len(ic):
            shows[ic] = vis.shows(Xd[ic])
        keep = cand & ~shows
        # colour calibration from SEEN faces: generated vs real splat colour there (generated ones are discarded)
        gain, bias = np.ones(3), np.zeros(3); err0 = err1 = None
        on_seen = base_ok & near & S["seen"][si] & (op > 0.5)
        npair = 0
        if on_seen.any():
            dd, gi = gtree.query(Xd[on_seen], k=4)
            src = np.full(len(dd), -1)
            for cc in range(4):
                okc = (src < 0) & (dd[:, cc] < a.match_dist_m / s) & (opac[gi[:, cc]] > 0.5)
                src[okc] = gi[okc, cc]
            v = src >= 0; npair = int(v.sum())
            if npair >= a.min_pairs:
                x = rgb[on_seen][v]; y = (0.5 + C0 * alb[src[v], :3]).clip(0, 1)
                err0 = float(np.sqrt(((x.clip(0, 1) - y) ** 2).mean()))
                if a.color_match:
                    for ch in range(3):
                        cov = np.cov(x[:, ch], y[:, ch]); gch = cov[0, 1] / max(cov[0, 0], 1e-6)
                        gain[ch] = np.clip(gch, 0.5, 2.0); bias[ch] = y[:, ch].mean() - gain[ch] * x[:, ch].mean()
                    err1 = float(np.sqrt((((gain * x + bias).clip(0, 1) - y) ** 2).mean()))
        rgb_k = (gain * rgb[keep] + bias).clip(0.0, 1.0)
        add["pos"].append(Xd[keep]); add["rot"].append(Rotation.from_matrix(Rd[keep]).as_quat()[:, [3, 0, 1, 2]] if keep.any() else np.zeros((0, 4)))
        add["scale"].append(np.log(np.maximum(sc[keep], 1e-7))); add["dens"].append(np.log(op[keep] / (1 - np.minimum(op[keep], 0.999)))[:, None])
        add["dc"].append((rgb_k - 0.5) / C0)
        un = tgt
        unseen_all.append(np.c_[S["posd"][un], S["nord"][un]])
        psnr = lambda e: None if e is None else round(float(-20 * np.log10(max(e, 1e-6))), 2)
        rep.append({"id": oid, "generated": int(len(Xm)), "valid": int(base_ok.sum()), "near_unseen_measured": int(cand.sum()),
                    "dropped_visible_in_training_view": int((cand & shows).sum()), "kept": int(keep.sum()),
                    "samples": int(len(S["pos"])), "unseen_measured_samples": int(un.sum()),
                    "color_pairs_on_seen_faces": npair, "gain": gain.round(3).tolist(), "bias": bias.round(3).tolist(),
                    "seen_face_rmse_raw": err0, "seen_face_rmse_matched": err1,
                    "seen_face_psnr_raw": psnr(err0), "seen_face_psnr_matched": psnr(err1)})
        print(f"[trellis-merge] object {oid}: {len(Xm):,} generated, {base_ok.sum():,} valid, {cand.sum():,} on unseen measured "
              f"surface, {(cand & shows).sum():,} would show in a training view, kept {keep.sum():,}; seen-face colour PSNR "
              f"{psnr(err0)} -> {psnr(err1)} dB (gain {gain.round(2).tolist()}, bias {bias.round(3).tolist()}, {npair} pairs)", flush=True)
    cat0 = lambda key, w: np.concatenate(add[key]) if add[key] else np.zeros((0, w))
    NP, NQ, NS, ND, NC = cat0("pos", 3), cat0("rot", 4), cat0("scale", 3), cat0("dens", 1), cat0("dc", 3)
    m = len(NP)
    fa = ck["features_albedo"]
    if fa.shape[1] > 3:                                                          # (should not happen: DC = 3)
        NC = np.c_[NC, np.zeros((m, fa.shape[1] - 3))]

    def cat(key, extra):
        v = ck[key]; base = v.detach()
        out = torch.cat([base, torch.as_tensor(extra, dtype=base.dtype)], 0)
        ck[key] = torch.nn.Parameter(out, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else out
    cat("positions", NP); cat("rotation", NQ); cat("scale", NS); cat("density", ND); cat("features_albedo", NC)
    cat("features_specular", np.zeros((m, ck["features_specular"].shape[1])))
    for k_, v in list(ck.items()):                                     # any other per-gaussian tensor: pad
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N0 and k_ not in ("positions", "rotation", "scale", "density",
                                                                                "features_albedo", "features_specular"):
            ck[k_] = torch.cat([v.detach(), torch.zeros((m,) + tuple(v.shape[1:]), dtype=v.dtype)], 0)
    ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(ck, a.out)
    base = os.path.splitext(a.out)[0]
    np.savez_compressed(base + "_unseen.npz", unseen=np.concatenate(unseen_all) if unseen_all else np.zeros((0, 6)))
    json.dump({"gaussians_in": N0, "added": m, "objects": rep,
               "params": {k_: v for k_, v in vars(a).items() if k_ != "func"}}, open(base + "_report.json", "w"), indent=2)
    print(f"[trellis-merge] added {m:,} TRELLIS gaussians on unseen measured faces -> {a.out}", flush=True)


# ----------------------------------------------------------------------------------------------- previews
def cmd_orbit(a):
    """orbit cameras around every object (looking at it from all sides, slightly from above) as a dataset."""
    from PIL import Image
    from fuse_sequences import write_model
    meta, pts, n, d, s, R, t = load_common(a)
    name, W, H, fx, fy, cx, cy, _, _, _ = cams_of(a.dataset)[0]
    imgs = []
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    blank = Image.fromarray(np.zeros((H, W, 3), np.uint8))
    for o in meta["objects"]:
        P = pts[f"obj{o['id']}"]; c0, A = obj_axes(P, n, d)
        hgt = float(o["height_m"]); tgt = c0 + 0.5 * hgt * n
        rad = a.radius_m + 0.6 * float(np.hypot(*o["footprint_m"]))
        for az in np.arange(a.azimuths) * 360.0 / a.azimuths:
            ph = np.deg2rad(az)
            C = tgt + rad * (np.cos(ph) * A[:, 0] + np.sin(ph) * A[:, 1]) + (a.height_m + 0.5 * hgt) * n
            z = tgt - C; z /= np.linalg.norm(z); x = np.cross(-n, z); x /= np.linalg.norm(x); y = np.cross(z, x)
            Rwc = R.T @ np.stack([x, y, z], 1); Cd = ((C - t) @ R) / s
            nm = f"obj{o['id']}_az{int(az):03d}.png"; blank.save(os.path.join(a.out, "images", nm))
            imgs.append((nm, 1, Rwc, Cd))
    write_model(os.path.join(a.out, "sparse", "0"), {1: ("PINHOLE", W, H, (fx, fy, cx, cy))}, imgs)
    print(f"[orbit] {len(imgs)} orbit views around {len(meta['objects'])} objects -> {a.out}", flush=True)


def cmd_preview(a):
    import cv2
    runs = [x.split("=", 1) for x in a.renders]
    names = sorted(f for f in os.listdir(os.path.join(a.orbit, "images")) if f.endswith(".png"))
    by = {}
    for f in names:
        by.setdefault(f.split("_az")[0], []).append(f)
    rows = []
    for obj, fs in by.items():
        sc = []
        for f in fs:
            ims = [cv2.imread(os.path.join(dr, f)) for _, dr in runs]
            if any(x is None for x in ims):
                continue
            sc.append((np.abs(ims[1].astype(int) - ims[0].astype(int)).mean() if len(ims) > 1 else 0, f, ims))
        for diff, f, ims in sorted(sc, key=lambda x: -x[0])[:a.per_object]:
            ims = [cv2.resize(x, (a.width, a.width * x.shape[0] // x.shape[1])) for x in ims]
            for x, (lab, _) in zip(ims, runs):
                cv2.putText(x, f"{lab} {f[:-4]}", (4, 16), 0, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            rows.append(np.hstack(ims))
    if rows:
        cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("preview ->", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, cams=True):
        p.add_argument("--objects", required=True, help="find_objects.py output prefix")
        p.add_argument("--sim3-json", required=True)
        p.add_argument("--dataset", required=True, help="training frames (COLMAP text) with <image>_depth.npy")
        if not cams:
            return
        p.add_argument("--splat-depth-dir", default=None, help="base splat's rendered depth per training frame")
        p.add_argument("--frame-stride", type=int, default=1)
        p.add_argument("--cell", type=float, default=0.01); p.add_argument("--hcell", type=float, default=0.02)
        p.add_argument("--max-incidence", type=float, default=85.0); p.add_argument("--depth-tol-m", type=float, default=0.03)
        p.add_argument("--splat-tol-rel", type=float, default=0.05); p.add_argument("--support-m", type=float, default=0.03)
    g = sub.add_parser("generate"); common(g)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--trellis-dir", default=os.path.join(os.environ.get("CEAR_WS", ""), "TRELLIS"))
    g.add_argument("--model", default="microsoft/TRELLIS-image-large")
    g.add_argument("--views", type=int, default=6, help="conditioning crops per object")
    g.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    g.add_argument("--steps", type=int, default=None, help="SLat sampler steps (default: the model's, 25)")
    g.add_argument("--cfg", type=float, default=None, help="SLat cfg strength (default: the model's)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--margin", type=float, default=0.02, help="shrink of the unit-cube fit")
    g.add_argument("--bottom-h", type=float, default=0.005, help="height of the footprint bottom face (m)")
    g.add_argument("--max-trunc", type=float, default=0.05, help="skip views cutting off more of the object")
    g.add_argument("--min-samples", type=int, default=200, help="visible surface samples a view needs")
    g.add_argument("--min-crop-px", type=int, default=40)
    g.add_argument("--mask-tol-rel", type=float, default=0.05)
    g.add_argument("--ids", default=None, help="comma-separated object ids (default: all)")
    g.set_defaults(func=cmd_generate)
    m = sub.add_parser("merge"); common(m)
    m.add_argument("--gen-dir", required=True); m.add_argument("--checkpoint", required=True); m.add_argument("--out", required=True)
    m.add_argument("--min-opacity", type=float, default=0.05)
    m.add_argument("--min-h", type=float, default=0.015, help="drop generated gaussians this close to the floor (m)")
    m.add_argument("--max-scale-vox", type=float, default=2.0, help="drop gaussians larger than this many voxels")
    m.add_argument("--keep-dist-vox", type=float, default=1.5, help="max distance to the measured surface (voxels)")
    m.add_argument("--color-match", type=int, default=1)
    m.add_argument("--match-dist-m", type=float, default=0.015)
    m.add_argument("--min-pairs", type=int, default=50)
    m.set_defaults(func=cmd_merge)
    o = sub.add_parser("orbit"); common(o, cams=False)
    o.add_argument("--out", required=True); o.add_argument("--azimuths", type=int, default=8)
    o.add_argument("--radius-m", type=float, default=0.6); o.add_argument("--height-m", type=float, default=0.45)
    o.set_defaults(func=cmd_orbit)
    p = sub.add_parser("preview")
    p.add_argument("--orbit", required=True); p.add_argument("--renders", nargs="+", required=True, help="label=dir (first = reference)")
    p.add_argument("--out", required=True); p.add_argument("--per-object", type=int, default=2); p.add_argument("--width", type=int, default=400)
    p.set_defaults(func=cmd_preview)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
