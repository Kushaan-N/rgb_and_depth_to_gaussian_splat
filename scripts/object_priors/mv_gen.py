"""Multi-view diffusion prior for the unseen sides of free-standing objects (cosmetic: the model invents the backs).

Per object (find_objects.py) the best real training views (trellis_complete.pick_views: visible area x unoccluded
share, spread in azimuth) are cut out with BiRefNet (masked to the geometric object mask, dilated), the best one
conditions Zero123++ v1.2 (sudo-ai, 6 views at azimuth +30..+330 deg relative to the input, elevation +20/-10 deg
absolute, fov 30 deg), the 6 generated views are cut out (BiRefNet) and get Marigold normal maps. `register` then puts
the generated views into the splat's frame: cameras on the Zero123++ orbit around the object's centre (up = floor
normal, azimuth 0 = the conditioning camera), orbit direction (+-) and the 2D scale/shift of each view picked by
silhouette IoU against the visual hull of the REAL masks; the final hull (real masks AND >= n-1 generated masks) gives
surface seeds (positions, outward normals, colours from the best-facing generated view), minus seeds a training view
would show (in front of the splat's rendered depth). mv_refit.py trains the object-zone gaussians on these views.

  generate (venv-mv, GPU)  -> <out>/obj<id>_{cond,gen,genmask}.png, obj<id>_gen_normal.npy, obj<id>_real.npz, generate.json
  register (venv-mv/gsplat) -> <out>/views/{images,masks,normals,sparse/0} (COLMAP text, PLY frame), <out>/seeds.npz,
                               <out>/register.json, <out>/obj<id>_reg.jpg (generated | hull silhouette overlay)

    python scripts/object_priors/mv_gen.py generate --objects D/objects --sim3-json sim3.json --dataset <train ds> \
        --splat-depth-dir D/depth --out-dir G
    python scripts/object_priors/mv_gen.py register --objects D/objects --sim3-json sim3.json --dataset <train ds> \
        --splat-depth-dir D/depth --out-dir G
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))

AZ = np.array([30, 90, 150, 210, 270, 330], float); EL = np.array([20, -10, 20, -10, 20, -10], float)   # Zero123++ v1.2
GS = 320; FOV = 30.0


def birefnet(model, img, dev):
    """-> soft mask (H,W) in [0,1] of the salient object of an RGB uint8 image."""
    import torch, torch.nn.functional as F
    x = torch.tensor(img, dtype=torch.float32, device=dev).permute(2, 0, 1)[None] / 255
    H, W = img.shape[:2]
    x = F.interpolate(x, (1024, 1024), mode="bilinear", align_corners=False)
    m = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; s = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    with torch.no_grad():
        y = model(((x - m) / s).half())[-1].float().sigmoid()
    return F.interpolate(y, (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def square_box(m, pad):
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = int(np.ceil(max(np.ptp(xs), np.ptp(ys)) * pad / 2)) + 1
    return int(round(cx)) - half, int(round(cy)) - half, 2 * half


def cut(img, x0, y0, S):
    H, W = img.shape[:2]; out = np.zeros((S, S) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + S, W), min(y0 + S, H)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def largest_cc_touching(mb, prior):
    import cv2
    n, lab = cv2.connectedComponents(mb.astype(np.uint8))
    keep = np.zeros_like(mb, bool)
    for i in range(1, n):
        c = lab == i
        if (c & prior).sum() > 0.05 * c.sum() or c.sum() > 0.2 * mb.sum():
            keep |= c
    return keep


def seed_quality(mc, gm):
    """generated views next to the input (+30, +330 deg) vs the conditioning silhouette: object size ratio (Zero123++
    sometimes shrinks the object to a speck) x IoU of the bbox-normalised silhouettes."""
    import cv2

    def norm(m):
        ys, xs = np.nonzero(m)
        if len(ys) < 20:
            return None, 0.0
        c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
        return cv2.resize(c, (64, 64), interpolation=cv2.INTER_NEAREST) > 0, max(np.ptp(xs), np.ptp(ys)) / m.shape[0]
    nc, ec = norm(mc); q = []
    for k in (0, 5):
        ng, eg = norm(gm[k])
        if ng is None:
            q.append(0.0); continue
        iou = (ng & nc).sum() / max((ng | nc).sum(), 1); r = min(eg, ec) / max(eg, ec, 1e-6)
        q.append(iou * r)
    return float(np.mean(q))


def generate(a):
    import torch, cv2
    from PIL import Image
    from trellis_complete import load_common, Vis, surface, pick_views, obj_axes
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, MarigoldNormalsPipeline
    from transformers import AutoModelForImageSegmentation
    dev = "cuda"
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s)
    bir = AutoModelForImageSegmentation.from_pretrained(a.birefnet, trust_remote_code=True).to(dev).half().eval()
    from huggingface_hub import snapshot_download
    pl = a.pipeline if os.path.isdir(a.pipeline) else snapshot_download(a.pipeline)      # local dir: works with HF_HUB_OFFLINE
    zp = DiffusionPipeline.from_pretrained(a.model, custom_pipeline=pl, torch_dtype=torch.float16)
    zp.scheduler = EulerAncestralDiscreteScheduler.from_config(zp.scheduler.config, timestep_spacing="trailing"); zp.to(dev)
    mg = MarigoldNormalsPipeline.from_pretrained(a.normals, variant="fp16", torch_dtype=torch.float16).to(dev) if a.normals else None
    os.makedirs(a.out_dir, exist_ok=True); log = []
    ids = set(a.ids) if a.ids else None
    for o in meta["objects"]:
        if ids is not None and o["id"] not in ids:
            continue
        S = surface(o, pts, n, d, a, to_ds, R, vis)
        c0, A = obj_axes(S["P"], n, d)
        chosen, ncand = pick_views(S, vis, a, to_ds(c0[None])[0], R.T @ A)
        if not chosen:
            print(f"[mv] object {o['id']}: no usable view", flush=True); continue
        real = {"cams": [], "masks": [], "scores": []}; conds = []
        for sc_, ci, m, az in chosen:
            name = vis.cams[ci][0]
            img = np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"))
            x0, y0, Sz = square_box(m, a.pad)
            ci_ = cut(img, x0, y0, Sz); mi = cut(m.astype(np.uint8), x0, y0, Sz) > 0
            k = max(3, int(a.prior_dilate * Sz)) | 1
            ys_, xs_ = np.nonzero(mi); e_ = int(a.prior_box * max(np.ptp(xs_), np.ptp(ys_)))   # the LiDAR mask misses parts:
            prior = np.zeros_like(mi); prior[max(ys_.min() - e_, 0):ys_.max() + e_ + 1, max(xs_.min() - e_, 0):xs_.max() + e_ + 1] = True
            sal = birefnet(bir, ci_, dev) > 0.5
            mb = largest_cc_touching(sal & prior, mi)
            if mb.sum() < 0.3 * mi.sum():                                  # saliency missed the object: geometric mask
                mb = cv2.morphologyEx(mi.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
            full = np.zeros(m.shape, bool)                                 # back to the full frame
            H, W = m.shape; sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + Sz, W), min(y0 + Sz, H)
            full[sy0:sy1, sx0:sx1] = mb[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            real["cams"].append(name); real["masks"].append(full); real["scores"].append(float(sc_))
            conds.append((ci_, mb, name, float(sc_) * min(1.0, Sz / a.min_cond_px)))
        # conditioning: best view, object only (RGBA), re-centred so it fills ~1/pad of the square
        # conditioning: well-seen AND enough pixels first; object only on grey (mask eroded 1 px: no dark fringe), filling
        # 1/pad of the square; seeds scored by seed_quality, other views / pads tried while the best is below --min-quality
        tries, best = [], None
        for ci_, mb, cname, _ in sorted(conds, key=lambda c: -c[3])[:a.cond_tries]:
            for pad in a.cond_pads:
                x0, y0, Sz = square_box(mb, pad)
                al = cv2.erode(cut(mb.astype(np.uint8), x0, y0, Sz), np.ones((3, 3), np.uint8)).astype(np.float32)[..., None]
                rgb = cut(ci_, x0, y0, Sz).astype(np.float32) * al + 127.0 * (1 - al)
                cond = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((GS, GS), Image.Resampling.LANCZOS)
                cmask = np.asarray(Image.fromarray((al[..., 0] * 255).astype(np.uint8)).resize((GS, GS))) > 127
                for seed in a.seeds:
                    g = torch.Generator(device=dev).manual_seed(seed)
                    with torch.no_grad():
                        gen = np.asarray(zp(cond, num_inference_steps=a.steps, generator=g).images[0].convert("RGB"))
                    vw = [gen[(k // 2) * GS:(k // 2 + 1) * GS, (k % 2) * GS:(k % 2 + 1) * GS] for k in range(6)]
                    gm_ = np.stack([birefnet(bir, v, dev) for v in vw])
                    q = seed_quality(cmask, gm_ > 0.5)
                    tries.append({"cond": cname, "pad": pad, "seed": seed, "quality": round(q, 3)}); Image.fromarray(gen).save(
                        os.path.join(a.out_dir, f"obj{o['id']}_gen_try{len(tries) - 1}.png"))
                    print(f"[mv] object {o['id']} {cname} pad {pad} seed {seed}: quality {q:.3f}", flush=True)
                    if best is None or q > best[0]:
                        best = (q, gen, vw, gm_, cond, cname, pad, seed)
            if best[0] >= a.min_quality:
                break
        qbest, gen, views, gm, cond, cname, pbest, sbest = best
        cond.save(os.path.join(a.out_dir, f"obj{o['id']}_cond.png"))
        nrm = None
        if mg is not None:
            with torch.no_grad():
                out = mg([Image.fromarray(v) for v in views], num_inference_steps=4, ensemble_size=a.normal_ensemble,
                         processing_resolution=768, output_type="np")
            nrm = np.asarray(out.prediction, np.float32)                   # (6,H,W,3): x right, y up, z to the viewer
            nrm = np.stack([nrm[..., 0], -nrm[..., 1], -nrm[..., 2]], -1)  # -> OpenCV camera frame
            np.save(os.path.join(a.out_dir, f"obj{o['id']}_gen_normal.npy"), nrm.astype(np.float16))
        Image.fromarray(gen).save(os.path.join(a.out_dir, f"obj{o['id']}_gen.png"))
        Image.fromarray(np.vstack([np.hstack([(gm[k] * 255).astype(np.uint8) for k in (2 * r, 2 * r + 1)]) for r in range(3)])).save(
            os.path.join(a.out_dir, f"obj{o['id']}_genmask.png"))
        np.savez_compressed(os.path.join(a.out_dir, f"obj{o['id']}_real.npz"), masks=np.stack(real["masks"]), gen_masks=gm.astype(np.float16),
                            views=np.stack(views))
        if nrm is not None:
            vis_n = ((np.concatenate([np.concatenate([nrm[2 * r], nrm[2 * r + 1]], 1) for r in range(3)], 0) * [1, -1, -1] + 1) * 127.5)
            Image.fromarray(vis_n.clip(0, 255).astype(np.uint8)).save(os.path.join(a.out_dir, f"obj{o['id']}_gen_normal.png"))
        Image.fromarray(np.hstack([np.asarray(Image.fromarray(c * m_[..., None]).resize((256, 256))) for c, m_, *_ in conds])).save(
            os.path.join(a.out_dir, f"obj{o['id']}_realcuts.jpg"))
        log.append({"id": o["id"], "cond": cname, "real": real["cams"], "candidates": ncand, "seed": sbest, "pad": pbest, "quality": round(float(qbest), 3), "tries": tries,
                    "gen_mask_px": [int((x > 0.5).sum()) for x in gm]})
        print(f"[mv] object {o['id']}: cond {cname}, {len(real['cams'])} real masks, generated 6 views", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "generate.json"), "w"), indent=1)


# ---------------------------------------------------------------- register (CPU numpy, GPU optional for Vis)
def iou_fit(Mg, Mr, sc0=None, search=10, coarse=False):
    """similarity u_gen = sc (u_ref - c) + c + t mapping the reference silhouette onto the generated one -> sc, tx, ty, IoU."""
    import cv2
    c = GS / 2

    def warp(sc, tx, ty):
        M = np.float32([[sc, 0, c - sc * c + tx], [0, sc, c - sc * c + ty]])
        return cv2.warpAffine(Mr.astype(np.uint8), M, (GS, GS), flags=cv2.INTER_NEAREST) > 0

    yg, xg = np.nonzero(Mg); yr, xr = np.nonzero(Mr)
    if len(xg) < 50 or len(xr) < 50:
        return 1.0, 0.0, 0.0, 0.0
    sc = sc0 if sc0 is not None else (np.ptp(xg) + np.ptp(yg)) / max(np.ptp(xr) + np.ptp(yr), 1)
    tx = (xg.min() + xg.max()) / 2 - (sc * ((xr.min() + xr.max()) / 2 - c) + c)
    ty = (yg.min() + yg.max()) / 2 - (sc * ((yr.min() + yr.max()) / 2 - c) + c)
    best = None
    scs = [sc] if sc0 is not None else [sc * f for f in ((0.92, 1.0, 1.08) if coarse else (0.9, 0.95, 1.0, 1.05, 1.1))]
    st = 3 if coarse else 2
    for s_ in scs:
        for dx in range(-search, search + 1, st):
            for dy in range(-search, search + 1, st):
                w = warp(s_, tx + dx, ty + dy); iou = (w & Mg).sum() / max((w | Mg).sum(), 1)
                if best is None or iou > best[3]:
                    best = (s_, tx + dx, ty + dy, float(iou))
    return best


def photo_ncc(img, gm, SP, SG, Rc, tt, f0, fit, px=2):
    """NCC between the generated view's grey and the visible well-seen splat points' grey at the fitted camera."""
    sc_, tx, ty, _ = fit; Pc = SP @ Rc.T + tt; z = Pc[:, 2]
    u = sc_ * f0 * Pc[:, 0] / z + GS / 2 + tx; v = sc_ * f0 * Pc[:, 1] / z + GS / 2 + ty
    ok = (z > 0) & (u >= 0) & (u < GS) & (v >= 0) & (v < GS)
    if ok.sum() < 200:
        return None
    ui, vi, zi, gi = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int), z[ok], SG[ok]
    zb = np.full((GS // px + 1, GS // px + 1), np.inf); np.minimum.at(zb, (vi // px, ui // px), zi)
    vis_ = zi <= zb[vi // px, ui // px] * 1.02; vis_ &= gm[vi, ui]
    if vis_.sum() < 200:
        return None
    g = img[vi[vis_], ui[vis_]].astype(np.float64) @ [0.299, 0.587, 0.114] / 255; p = gi[vis_]
    g, p = g - g.mean(), p - p.mean()
    return float((g * p).sum() / max(np.sqrt((g * g).sum() * (p * p).sum()), 1e-9))


def raster(uv, ok, k=3):
    import cv2
    M = np.zeros((GS, GS), np.uint8); u = np.floor(uv[ok, 0]).astype(int); v = np.floor(uv[ok, 1]).astype(int)
    inn = (u >= 0) & (u < GS) & (v >= 0) & (v < GS); M[v[inn], u[inn]] = 1
    return cv2.morphologyEx(cv2.dilate(M, np.ones((k, k), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def register(a):
    import cv2
    from PIL import Image
    from scipy import ndimage
    from trellis_complete import load_common, obj_axes, Vis
    from covis_masks import cams_of
    from solidify_unseen import look_at
    from build_depth_targets import quat_to_R
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    cams = {c[0]: c for c in cams_of(a.dataset)}
    gl = json.load(open(os.path.join(a.out_dir, "generate.json")))
    V = os.path.join(a.out_dir, "views"); os.makedirs(os.path.join(V, "images"), exist_ok=True)
    os.makedirs(os.path.join(V, "masks"), exist_ok=True); os.makedirs(os.path.join(V, "normals"), exist_ok=True)
    os.makedirs(os.path.join(V, "sparse", "0"), exist_ok=True)
    vis = Vis(a, s) if a.splat_depth_dir else None
    SPLAT = None
    if a.ply and a.support:                                                # (metric centres, grey) of well-seen opaque gaussians
        from plyfile import PlyData
        v = PlyData.read(a.ply)["vertex"].data; wm = np.load(a.support)["wmax"]
        op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64))); k_ = (wm >= a.seen_w) & (op > 0.7)
        Xs = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)[k_]
        col = np.clip(0.5 + 0.28209479 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)[k_], 0, 1)
        SPLAT = (s * (Xs @ R.T) + t, col @ [0.299, 0.587, 0.114])
    crow, irow, rep, seeds = [], [], [], {"X": [], "N": [], "C": [], "obj": []}
    f0 = GS / 2 / np.tan(np.radians(FOV / 2))
    for g in gl:
        oid = g["id"]; X = pts[f"obj{oid}"]
        c0, A = obj_axes(X, n, d); a1, a2 = A[:, 0], A[:, 1]
        L = np.stack([(X - c0) @ a1, (X - c0) @ a2, X @ n + d], 1)
        lo = np.percentile(L, 1, 0) - a.margin_m; hi = np.percentile(L, 99, 0) + a.margin_m; lo[2] = a.floor_m; hi[2] = np.percentile(L[:, 2], 99.5) + a.margin_m
        cm = c0 + n * (0.5 * hi[2]); Rob = 0.5 * float(np.linalg.norm(hi - lo))
        gu = [np.arange(lo[i] + a.cell / 2, hi[i], a.cell) for i in range(3)]
        U, Vv, Hh = np.meshgrid(*gu, indexing="ij"); Lg = np.stack([U.ravel(), Vv.ravel(), Hh.ravel()], 1)
        Pm = c0 + np.outer(Lg[:, 0], a1) + np.outer(Lg[:, 1], a2) + np.outer(Lg[:, 2], n); Pd = to_ds(Pm)
        z_ = np.load(os.path.join(a.out_dir, f"obj{oid}_real.npz")); rmask = z_["masks"]; gmask = z_["gen_masks"].astype(np.float32) > 0.5
        views = z_["views"]; nrm_f = os.path.join(a.out_dir, f"obj{oid}_gen_normal.npy")
        nrm = np.load(nrm_f).astype(np.float32) if os.path.exists(nrm_f) else None
        # visual hull of the real masks (a voxel out of a view's frame does not vote)
        k = a.real_dilate
        inside = np.ones(len(Pd), bool)
        for name, m in zip(g["real"], rmask):
            _, W, H, fx, fy, cx, cy, Rcw, tcw, _ = cams[name]
            md = cv2.dilate(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
            Pc = Pd @ Rcw.T + tcw; u = np.floor(fx * Pc[:, 0] / Pc[:, 2] + cx).astype(int); v = np.floor(fy * Pc[:, 1] / Pc[:, 2] + cy).astype(int)
            inn = (Pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            inside[inn] &= md[v[inn], u[inn]]
        # conditioning camera azimuth in the object frame
        _, W, H, fx, fy, cx, cy, Rcw, tcw, _ = cams[g["cond"]]
        Cm = s * ((-Rcw.T @ tcw) @ R.T) + t; dv = Cm - cm; phi0 = float(np.arctan2(dv @ a2, dv @ a1))
        Pin = Pd[inside][::max(1, int(inside.sum()) // 60000)]
        SP = SG = None
        if SPLAT is not None:                                              # well-seen splat surface in the object box
            Lsp = np.stack([(SPLAT[0] - c0) @ a1, (SPLAT[0] - c0) @ a2, SPLAT[0] @ n + d], 1)
            ks = np.all((Lsp >= lo) & (Lsp <= hi), 1) & (Lsp[:, 2] > 0.02)
            if ks.sum() > 200:
                SP, SG = to_ds(SPLAT[0][ks]), SPLAT[1][ks]

        def trial(sign, dphi, dele, coarse):
            fits, cams_k = [], []
            for kk in range(6):
                phi = phi0 + np.radians(dphi) + sign * np.radians(AZ[kk]); el = np.radians(EL[kk] + dele)
                C = cm + a.radius_k * Rob * (np.cos(el) * (np.cos(phi) * a1 + np.sin(phi) * a2) + np.sin(el) * n)
                q, tt = look_at(to_ds(C[None])[0], to_ds(cm[None])[0], R.T @ n)
                Rc = quat_to_R(q); Pc = Pin @ Rc.T + tt
                uv = np.stack([f0 * Pc[:, 0] / Pc[:, 2] + GS / 2, f0 * Pc[:, 1] / Pc[:, 2] + GS / 2], 1)
                Mr = raster(uv, Pc[:, 2] > 0)
                fits.append(iou_fit(gmask[kk], Mr, search=6 if coarse else 10, coarse=coarse)); cams_k.append((q, tt, Mr))
            sc_ = float(np.mean([f[3] for f in fits]))
            if SP is not None and a.w_photo > 0:                           # texture agreement of the well-seen surface
                nc = [photo_ncc(views[kk], gmask[kk], SP, SG, quat_to_R(cams_k[kk][0]), cams_k[kk][1], f0, fits[kk]) for kk in range(6)]
                nc = [x for x in nc if x is not None]
                sc_ += a.w_photo * (float(np.mean(nc)) if nc else 0.0)
            return sc_, sign, fits, cams_k, (dphi, dele)
        # coarse search: orbit direction, azimuth offset of the conditioning view, elevation offset; then refine
        best = max((trial(sg, dp, de, True) for sg in (1, -1) for dp in np.arange(-a.az_search, a.az_search + 1e-6, 5.0)
                    for de in np.arange(-a.el_search, a.el_search + 1e-6, 5.0)), key=lambda x: x[0])
        sg, (dp0, de0) = best[1], best[4]
        best = max([trial(sg, dp0 + ddp, de0 + dde, False) for ddp in (-2.5, 0, 2.5) for dde in (-2.5, 0, 2.5)], key=lambda x: x[0])
        offs = best[4]
        # final cameras: full hull projection (not subsampled) for the scale / shift fit
        Pin = Pd[inside]; best = trial(best[1], offs[0], offs[1], False)
        tot, sign, fits, cams_k, offs = best
        near = [kk for kk in range(6) if min(AZ[kk], 360 - AZ[kk]) <= 90]           # scale from the views next to the input
        scg = float(np.median([fits[kk][0] for kk in near]))
        regs = [iou_fit(gmask[kk], cams_k[kk][2], sc0=scg, search=16) for kk in range(6)]
        # final hull: real masks AND generated masks (>= n-1 of the generated views; out of frame does not vote)
        gk = a.gen_dilate; votes = np.zeros(len(Pd), int); nin = np.zeros(len(Pd), int); Kv = []; kv_hull = []
        for kk in range(6):
            sc_, tx, ty, iou = regs[kk]; q, tt, _ = cams_k[kk]; Rc = quat_to_R(q)
            fxk, cxk, cyk = f0 * sc_, GS / 2 + tx, GS / 2 + ty; Kv.append((fxk, cxk, cyk, Rc, tt))
            Pc = Pd @ Rc.T + tt; u = np.floor(fxk * Pc[:, 0] / Pc[:, 2] + cxk).astype(int); v = np.floor(fxk * Pc[:, 1] / Pc[:, 2] + cyk).astype(int)
            inn = (Pc[:, 2] > 0) & (u >= 0) & (u < GS) & (v >= 0) & (v < GS)
            md = cv2.dilate(gmask[kk].astype(np.uint8), np.ones((2 * gk + 1, 2 * gk + 1), np.uint8)) > 0
            nin += inn; votes[inn] += md[v[inn], u[inn]]
            nm = f"obj{oid}_v{kk}.png"; kv_hull.append(nm)
            Image.fromarray(views[kk]).save(os.path.join(V, "images", nm))
            Image.fromarray((gmask[kk] * 255).astype(np.uint8)).save(os.path.join(V, "masks", nm.replace(".png", "_mask.png")))
            if nrm is not None:
                np.save(os.path.join(V, "normals", nm.replace(".png", "_normal.npy")), nrm[kk].astype(np.float16))
            cid = len(crow) + 1
            crow.append(f"{cid} PINHOLE {GS} {GS} {fxk:.6f} {fxk:.6f} {cxk:.6f} {cyk:.6f}\n")
            irow.append(f"{len(irow) + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in tt)} {cid} {nm}\n\n")
        hull = inside & (votes >= np.maximum(nin - a.gen_miss, 1))
        occ = hull.reshape(U.shape)
        occ = ndimage.binary_opening(occ, iterations=1) if occ.sum() > 1000 else occ
        surf = occ & ~ndimage.binary_erosion(occ, iterations=1, border_value=0)
        surf &= (Hh >= a.floor_m + a.cell)                                   # no seeds in the floor contact layer
        sm = ndimage.gaussian_filter(occ.astype(np.float32), 1.5)
        gr = np.stack(np.gradient(sm, a.cell), -1)[surf]; idx = np.nonzero(surf.ravel())[0]
        Nl = -gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-9)
        Nm = np.outer(Nl[:, 0], a1) + np.outer(Nl[:, 1], a2) + np.outer(Nl[:, 2], n)
        Ps = Pm[idx]; Psd = Pd[idx]
        shown = vis.shows(Psd) if vis is not None else np.zeros(len(Psd), bool)
        # colour: the generated view facing the seed best (where its pixel is in the mask)
        col = np.full((len(Psd), 3), np.nan); bestc = np.full(len(Psd), -2.0)
        for kk, (fxk, cxk, cyk, Rc, tt) in enumerate(Kv):
            Ccam = -Rc.T @ tt; dvw = Ccam[None] - Psd; dvw /= np.linalg.norm(dvw, axis=1, keepdims=True)
            cs = (dvw * (Nm @ R)).sum(1)
            Pc = Psd @ Rc.T + tt; u = np.floor(fxk * Pc[:, 0] / Pc[:, 2] + cxk).astype(int); v = np.floor(fxk * Pc[:, 1] / Pc[:, 2] + cyk).astype(int)
            inn = (u >= 0) & (u < GS) & (v >= 0) & (v < GS); ok = np.zeros(len(Psd), bool)
            ok[inn] = gmask[kk][v[inn], u[inn]]; ok &= cs > bestc
            col[ok] = views[kk][v[ok], u[ok]] / 255.0; bestc[ok] = cs[ok]
        col[np.isnan(col[:, 0])] = np.nanmedian(col, 0) if np.isfinite(col).any() else 0.5
        keep = ~shown
        seeds["X"].append(Psd[keep]); seeds["N"].append((Nm @ R)[keep]); seeds["C"].append(col[keep]); seeds["obj"].append(np.full(int(keep.sum()), oid))
        # overlay: generated view | its registered hull silhouette
        ov = []
        for kk, (fxk, cxk, cyk, Rc, tt) in enumerate(Kv):
            Pc = Pd[hull] @ Rc.T + tt; uv = np.stack([fxk * Pc[:, 0] / Pc[:, 2] + cxk, fxk * Pc[:, 1] / Pc[:, 2] + cyk], 1)
            Mh = raster(uv, Pc[:, 2] > 0); im = views[kk].copy()
            Image.fromarray((Mh * 255).astype(np.uint8)).save(os.path.join(V, "masks", kv_hull[kk].replace(".png", "_hull.png")))
            e = Mh ^ cv2.erode(Mh.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); im[e] = [255, 0, 0]
            e2 = gmask[kk] ^ cv2.erode(gmask[kk].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); im[e2] = [0, 255, 0]
            ov.append(im)
        Image.fromarray(np.hstack(ov)).save(os.path.join(a.out_dir, f"obj{oid}_reg.jpg"))
        rr = {"id": oid, "sign": sign, "az_el_offset_deg": [float(x) for x in offs], "score_iou_photo": round(tot, 3), "scale": round(scg, 3),
              "iou_final": [round(x[3], 3) for x in regs], "hull_voxels": int(hull.sum()), "real_hull_voxels": int(inside.sum()),
              "surface_seeds": int(len(Psd)), "shown_in_training_views": int(shown.sum()), "seeds": int(keep.sum()),
              "box_lo": lo.round(3).tolist(), "box_hi": hi.round(3).tolist(), "c0": c0.tolist(), "axes": A.tolist()}
        rep.append(rr); print("[mv-reg]", json.dumps({k_: v_ for k_, v_ in rr.items() if k_ not in ("c0", "axes")}), flush=True)
    open(os.path.join(V, "sparse", "0", "cameras.txt"), "w").writelines(crow)
    open(os.path.join(V, "sparse", "0", "images.txt"), "w").writelines(irow)
    open(os.path.join(V, "sparse", "0", "points3D.txt"), "w").write("")
    np.savez_compressed(os.path.join(a.out_dir, "seeds.npz"), **{k_: np.concatenate(v_) if v_ else np.zeros((0, 3)) for k_, v_ in seeds.items()})
    json.dump(rep, open(os.path.join(a.out_dir, "register.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("generate", "register"):
        g = sub.add_parser(nm)
        g.add_argument("--objects", required=True); g.add_argument("--sim3-json", required=True); g.add_argument("--dataset", required=True)
        g.add_argument("--splat-depth-dir", default=None); g.add_argument("--out-dir", required=True)
        g.add_argument("--ids", type=int, nargs="*", default=None)
        # trellis_complete.py's visibility / view-selection parameters
        g.add_argument("--views", type=int, default=4); g.add_argument("--frame-stride", type=int, default=1)
        g.add_argument("--cell", type=float, default=0.01); g.add_argument("--hcell", type=float, default=0.02)
        g.add_argument("--max-incidence", type=float, default=85.0); g.add_argument("--depth-tol-m", type=float, default=0.03)
        g.add_argument("--splat-tol-rel", type=float, default=0.05); g.add_argument("--support-m", type=float, default=0.03)
        g.add_argument("--max-trunc", type=float, default=0.05); g.add_argument("--min-samples", type=int, default=200)
        g.add_argument("--min-crop-px", type=int, default=40); g.add_argument("--mask-tol-rel", type=float, default=0.05)
    g = sub.choices["generate"]
    g.add_argument("--model", default="sudo-ai/zero123plus-v1.2"); g.add_argument("--pipeline", default="sudo-ai/zero123plus-pipeline")
    g.add_argument("--birefnet", default="ZhengPeng7/BiRefNet"); g.add_argument("--normals", default="prs-eth/marigold-normals-v1-1")
    g.add_argument("--normal-ensemble", type=int, default=3)
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); g.add_argument("--steps", type=int, default=75)
    g.add_argument("--pad", type=float, default=1.6, help="real crop: square of pad x the object's mask bbox")
    g.add_argument("--cond-pads", type=float, nargs="+", default=[1.25, 1.6], help="conditioning: object fills 1/pad of the square")
    g.add_argument("--cond-tries", type=int, default=3, help="real views tried as conditioning (best first)")
    g.add_argument("--min-quality", type=float, default=0.45, help="stop trying views / pads once a seed reaches this")
    g.add_argument("--prior-dilate", type=float, default=0.06, help="closing size of the geometric fallback mask (x crop)")
    g.add_argument("--prior-box", type=float, default=0.15, help="BiRefNet kept within the geometric mask's bbox grown by this")
    g.add_argument("--min-cond-px", type=float, default=256, help="conditioning view: crops smaller than this are penalised")
    r = sub.choices["register"]
    r.add_argument("--margin-m", type=float, default=0.06); r.add_argument("--floor-m", type=float, default=0.005)
    r.add_argument("--radius-k", type=float, default=4.0, help="generated-camera distance in object radii (Zero123++: 4)")
    r.add_argument("--real-dilate", type=int, default=4); r.add_argument("--gen-dilate", type=int, default=3)
    r.add_argument("--gen-miss", type=int, default=1, help="a hull voxel may fall outside this many generated masks")
    r.add_argument("--ply", default=None); r.add_argument("--support", default=None)
    r.add_argument("--seen-w", type=float, default=0.2); r.add_argument("--w-photo", type=float, default=0.5)
    r.add_argument("--az-search", type=float, default=30.0); r.add_argument("--el-search", type=float, default=10.0)
    a = ap.parse_args()
    {"generate": generate, "register": register}[a.cmd](a)


if __name__ == "__main__":
    main()
