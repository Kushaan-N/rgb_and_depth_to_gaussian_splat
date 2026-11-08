"""Pose-conditioned novel views of the unseen object sides with Stable Virtual Camera (SEVA, Stability AI 2025), for
mv_refit.py (cosmetic: the model invents the backs). Unlike mv_gen.py (Zero123++, object-only 320 px views on an assumed
orbit that must be registered by silhouette), SEVA takes the REAL training crops with their REAL cameras and renders
576 px views at cameras we choose, so the generated views live in the splat's frame by construction.

  generate (venv-seva, GPU) per object (--ids): the best real training views (trellis_complete.pick_views) cut to
           object-centred squares inside the frame (intrinsics adjusted) -> SEVA img2trajvid conditioned on them, targets
           = orbit cameras around the object (--n-az azimuths x --heights m above the floor, --dist m, focal so the object
           fills --fill of the frame) -> <out>/obj<id>/{inputs,targets}/*.png + cams.json
  hull     (venv-mv, GPU) BiRefNet object masks of every input crop and target (kept within the projected object box),
           visual hull = real masks AND >= n - --gen-miss target masks, surface seeds (minus seeds a training view shows),
           -> mv_refit.py's interface: <out>/views/{images,masks (+ _hull),sparse/0}, seeds.npz, register.json;
           mv_objfit.py also reads <out>/views_real (the masked real crops) and seeds.npz A* (the whole hull surface)

    python scripts/object_priors/mv_seva.py generate --objects D/objects --sim3-json sim3.json --dataset <train ds> \
        --splat-depth-dir D/depth --out-dir S --ids 3 4 5
    python scripts/object_priors/mv_seva.py hull --objects D/objects --sim3-json sim3.json --dataset <train ds> \
        --splat-depth-dir D/depth --out-dir S --ids 3 4 5
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def obj_frame(X, n, d):
    from trellis_complete import obj_axes
    c0, A = obj_axes(X, n, d); a1, a2 = A[:, 0], A[:, 1]
    L = np.stack([(X - c0) @ a1, (X - c0) @ a2, X @ n + d], 1)
    return c0, A, L


def generate(a):
    import torch
    from PIL import Image
    from trellis_complete import load_common, Vis, surface, pick_views
    from solidify_unseen import look_at
    from build_depth_targets import quat_to_R
    import seva.modules.autoencoder as sae
    from diffusers import AutoencoderKL
    _fp = AutoencoderKL.from_pretrained
    # the SD2.1-base VAE moved off stabilityai's HF page: same weights from a mirror
    sae.AutoencoderKL = type("AKL", (), {"from_pretrained": staticmethod(lambda *x, **k: _fp(a.vae_repo, subfolder="vae"))})
    from seva.eval import run_one_scene
    from seva.model import SGMWrapper
    from seva.modules.autoencoder import AutoEncoder
    from seva.modules.conditioner import CLIPConditioner
    from seva.sampling import DiscreteDenoiser
    from seva.utils import load_model
    dev = "cuda:0"
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s)
    model = SGMWrapper(load_model(model_version=a.version, pretrained_model_name_or_path=a.model, device="cpu", verbose=True).eval()).to(dev)
    ae = AutoEncoder(chunk_size=1).to(dev); cond = CLIPConditioner().to(dev); den = DiscreteDenoiser(num_idx=1000, device=dev)
    Z = a.size; log = []
    for o in meta["objects"]:
        if a.ids and o["id"] not in a.ids:
            continue
        S = surface(o, pts, n, d, a, to_ds, R, vis)
        X = pts[f"obj{o['id']}"]; c0, A, L = obj_frame(X, n, d); a1, a2 = A[:, 0], A[:, 1]
        hmax = float(np.percentile(L[:, 2], 99)); Rob = 0.5 * float(np.linalg.norm(np.percentile(L, 99, 0) - np.percentile(L, 1, 0)))
        cm = c0 + n * 0.5 * hmax
        chosen, ncand = pick_views(S, vis, a, to_ds(c0[None])[0], R.T @ A)
        if not chosen:
            print(f"[seva] object {o['id']}: no usable view", flush=True); continue
        O = os.path.join(a.out_dir, f"obj{o['id']}"); os.makedirs(os.path.join(O, "inputs"), exist_ok=True)
        imgs, c2ws, Ks, cams, dists = [], [], [], [], []
        for sc_, ci, m, az in chosen:
            name, W, H, fx, fy, cx, cy, Rcw, tcw, _ = vis.cams[ci]
            img = np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"))
            ys, xs = np.nonzero(m)
            side = int(np.clip(a.pad * max(np.ptp(xs), np.ptp(ys)), a.min_crop, min(W, H)))
            x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, W - side)); y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, H - side))
            crop = np.asarray(Image.fromarray(img[y0:y0 + side, x0:x0 + side]).resize((Z, Z), Image.Resampling.LANCZOS))
            k = Z / side; K = np.array([[fx * k, 0, (cx - x0) * k], [0, fy * k, (cy - y0) * k], [0, 0, 1.0]])
            c2w = np.eye(4); c2w[:3, :3] = Rcw.T; c2w[:3, 3] = -Rcw.T @ tcw
            imgs.append(crop); c2ws.append(c2w); Ks.append(K)
            Image.fromarray(crop).save(os.path.join(O, "inputs", f"in{len(cams):02d}.png"))
            cams.append({"src": name, "crop": [x0, y0, side], "K": K.tolist(), "c2w": c2w.tolist()})
            dists.append(float(np.linalg.norm(s * (c2w[:3, 3] @ R.T) + t - cm)))
        dist = a.dist if a.dist > 0 else float(np.clip(np.median(dists), a.dist_min, a.dist_max))
        f = a.fill * Z / 2 * dist / max(Rob, 0.05)
        # orbit starts at the best input's azimuth so the chunks next to the inputs come first
        Cin = s * (c2ws[0][:3, 3] @ R.T) + t - cm; phi0 = float(np.arctan2(Cin @ a2, Cin @ a1))
        tg = []
        for hh in a.heights:
            for k_ in range(a.n_az):
                phi = phi0 + 2 * np.pi * k_ / a.n_az
                C = c0 + dist * (np.cos(phi) * a1 + np.sin(phi) * a2) + hh * n
                q, tt = look_at(to_ds(C[None])[0], to_ds(cm[None])[0], R.T @ n)
                Rc = quat_to_R(q); c2w = np.eye(4); c2w[:3, :3] = Rc.T; c2w[:3, 3] = -Rc.T @ tt
                K = np.array([[f, 0, Z / 2], [0, f, Z / 2], [0, 0, 1.0]])
                imgs.append(np.zeros((Z, Z, 3), np.uint8)); c2ws.append(c2w); Ks.append(K)
                tg.append({"name": f"obj{o['id']}_t{len(tg):02d}.png", "K": K.tolist(), "c2w": c2w.tolist(), "h": hh, "az_deg": round(np.degrees(phi - phi0), 1)})
        ni = len(cams)
        vd = {"H": Z, "W": Z, "T": a.T, "C": 4, "f": 8,
              "options": {"chunk_strategy": a.chunk_strategy, "video_save_fps": 2.0, "beta_linear_start": 5e-6, "log_snr_shift": 2.4,
                          "guider_types": 1, "cfg": a.cfg, "camera_scale": a.camera_scale, "num_steps": a.steps, "cfg_min": 1.2,
                          "encoding_t": 1, "decoding_t": 1, "save_input": False}}
        sp = os.path.join(O, "seva")
        gen = run_one_scene("img2trajvid", vd, model=model, ae=ae, conditioner=cond, denoiser=den,
                            image_cond={"img": imgs, "input_indices": list(range(ni)), "prior_indices": None},
                            camera_cond={"c2w": torch.tensor(np.stack(c2ws)[:, :3]).float(), "K": torch.tensor(np.stack(Ks)).float(),
                                         "input_indices": list(range(len(imgs)))},
                            save_path=sp, use_traj_prior=False, traj_prior_Ks=None, traj_prior_c2ws=None, seed=a.seed)
        for _ in gen:
            pass
        outs = sorted(glob.glob(os.path.join(sp, "samples-rgb", "*.png")))
        assert len(outs) == len(tg), f"{len(outs)} samples for {len(tg)} targets"
        os.makedirs(os.path.join(O, "targets"), exist_ok=True)
        for p_, g_ in zip(outs, tg):
            Image.open(p_).convert("RGB").save(os.path.join(O, "targets", g_["name"]))
        json.dump({"id": o["id"], "inputs": cams, "targets": tg, "dist_m": dist, "focal": f, "size": Z, "candidates": ncand,
                   "c0": c0.tolist(), "axes": A.tolist(), "hmax": hmax}, open(os.path.join(O, "cams.json"), "w"), indent=1)
        sheet = [np.asarray(Image.open(os.path.join(O, "targets", g_["name"])).resize((192, 192))) for g_ in tg]
        rows = [np.hstack(sheet[i:i + a.n_az]) for i in range(0, len(sheet), a.n_az)]
        ins = np.hstack([np.asarray(Image.fromarray(x).resize((192, 192))) for x in imgs[:ni]])
        ins = np.pad(ins, ((0, 0), (0, rows[0].shape[1] - ins.shape[1]), (0, 0))) if ins.shape[1] < rows[0].shape[1] else ins[:, :rows[0].shape[1]]
        Image.fromarray(np.vstack([ins] + rows)).save(os.path.join(a.out_dir, f"obj{o['id']}_seva.jpg"), quality=88)
        log.append({"id": o["id"], "inputs": [c["src"] for c in cams], "targets": len(tg), "dist_m": round(dist, 3)})
        print(f"[seva] object {o['id']}: {ni} inputs -> {len(tg)} targets at {dist:.2f} m", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "generate.json"), "w"), indent=1)


def proj(P, K, c2w):
    w2c = np.linalg.inv(c2w); Pc = P @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2]
    return K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2], z


def hull(a):
    import cv2, torch
    from PIL import Image
    from scipy import ndimage
    from trellis_complete import load_common, Vis
    from mv_gen import birefnet, largest_cc_touching
    from transformers import AutoModelForImageSegmentation
    from build_depth_targets import quat_to_R  # noqa: F401
    from refine_poses_gsplat import rot2qvec
    dev = "cuda"
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s) if a.splat_depth_dir else None
    bir = AutoModelForImageSegmentation.from_pretrained(a.birefnet, trust_remote_code=True).to(dev).half().eval()
    V = os.path.join(a.out_dir, "views")
    for sd_ in ("images", "masks", "sparse/0"):
        os.makedirs(os.path.join(V, sd_), exist_ok=True)
    crow, irow, rep, seeds = [], [], [], {"X": [], "N": [], "C": [], "obj": []}
    rcrow, rirow = [], []
    for cj in sorted(glob.glob(os.path.join(a.out_dir, "obj*", "cams.json"))):
        J = json.load(open(cj)); oid = J["id"]
        if a.ids and oid not in a.ids:
            continue
        O = os.path.dirname(cj); X = pts[f"obj{oid}"]; c0, A, L = obj_frame(X, n, d); a1, a2 = A[:, 0], A[:, 1]
        lo = np.percentile(L, 1, 0) - a.margin_m; hi = np.percentile(L, 99, 0) + a.margin_m; lo[2] = a.floor_m
        hi[2] = np.percentile(L[:, 2], 99.5) + a.margin_m
        gu = [np.arange(lo[i] + a.cell / 2, hi[i], a.cell) for i in range(3)]
        U, Vv, Hh = np.meshgrid(*gu, indexing="ij"); Lg = np.stack([U.ravel(), Vv.ravel(), Hh.ravel()], 1)
        Pm = c0 + np.outer(Lg[:, 0], a1) + np.outer(Lg[:, 1], a2) + np.outer(Lg[:, 2], n); Pd = to_ds(Pm)
        box = np.array([[lo[0], lo[1], 0], [hi[0], hi[1], hi[2]]]); corners = np.array([[box[i, 0], box[j, 1], box[k, 2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        Cd = to_ds(c0 + np.outer(corners[:, 0], a1) + np.outer(corners[:, 1], a2) + np.outer(corners[:, 2], n))

        def omask(img, K, c2w):
            u, v, _ = proj(Cd, K, c2w); Z = img.shape[0]; e = a.box_grow * max(np.ptp(u), np.ptp(v))
            bx = np.zeros(img.shape[:2], bool)
            bx[int(max(v.min() - e, 0)):int(min(v.max() + e, Z)), int(max(u.min() - e, 0)):int(min(u.max() + e, Z))] = True
            sal = birefnet(bir, img, dev) > 0.5
            core = np.zeros_like(bx); ec = 0.25 * max(np.ptp(u), np.ptp(v)); uc, vc = u.mean(), v.mean()
            core[int(max(vc - ec, 0)):int(max(vc + ec, 0)), int(max(uc - ec, 0)):int(max(uc + ec, 0))] = True
            return largest_cc_touching(sal & bx, core)
        ins = [(np.asarray(Image.open(os.path.join(O, "inputs", f"in{i:02d}.png")).convert("RGB")), np.array(c["K"]), np.array(c["c2w"]))
               for i, c in enumerate(J["inputs"])]
        tgs = [(np.asarray(Image.open(os.path.join(O, "targets", g["name"])).convert("RGB")), np.array(g["K"]), np.array(g["c2w"]), g["name"])
               for g in J["targets"]]
        rm = [omask(*x) for x in ins]; gm = [omask(*x[:3]) for x in tgs]
        inside = np.ones(len(Pd), bool)
        for (img, K, c2w), m in zip(ins, rm):
            md = cv2.dilate(m.astype(np.uint8), np.ones((2 * a.real_dilate + 1,) * 2, np.uint8)) > 0
            u, v, z = proj(Pd, K, c2w); ui, vi = np.floor(u).astype(int), np.floor(v).astype(int); Z = img.shape[0]
            inn = (z > 0) & (ui >= 0) & (ui < Z) & (vi >= 0) & (vi < Z); inside[inn] &= md[vi[inn], ui[inn]]
        votes = np.zeros(len(Pd), int); nin = np.zeros(len(Pd), int)
        for (img, K, c2w, nm), m in zip(tgs, gm):
            md = cv2.dilate(m.astype(np.uint8), np.ones((2 * a.gen_dilate + 1,) * 2, np.uint8)) > 0
            u, v, z = proj(Pd, K, c2w); ui, vi = np.floor(u).astype(int), np.floor(v).astype(int); Z = img.shape[0]
            inn = (z > 0) & (ui >= 0) & (ui < Z) & (vi >= 0) & (vi < Z); nin += inn; votes[inn] += md[vi[inn], ui[inn]]
        hull_ = inside & (votes >= np.maximum(nin - a.gen_miss, 1))
        occ = hull_.reshape(U.shape); occ = ndimage.binary_opening(occ, iterations=1) if occ.sum() > 1000 else occ
        surf = occ & ~ndimage.binary_erosion(occ, iterations=1, border_value=0); surf &= Hh >= a.floor_m + a.cell
        sm = ndimage.gaussian_filter(occ.astype(np.float32), 1.5); gr = np.stack(np.gradient(sm, a.cell), -1)[surf]
        idx = np.nonzero(surf.ravel())[0]; Nl = -gr / np.maximum(np.linalg.norm(gr, axis=1, keepdims=True), 1e-9)
        Nm = np.outer(Nl[:, 0], a1) + np.outer(Nl[:, 1], a2) + np.outer(Nl[:, 2], n); Psd = Pd[idx]; Nd = Nm @ R
        shown = vis.shows(Psd) if vis is not None else np.zeros(len(Psd), bool)
        col = np.full((len(Psd), 3), np.nan); best = np.full(len(Psd), -2.0)
        for (img, K, c2w, nm), m in zip(tgs, gm):
            dv = c2w[:3, 3][None] - Psd; dv /= np.linalg.norm(dv, axis=1, keepdims=True); cs = (dv * Nd).sum(1)
            u, v, z = proj(Psd, K, c2w); ui, vi = np.floor(u).astype(int), np.floor(v).astype(int); Z = img.shape[0]
            ok = (z > 0) & (ui >= 0) & (ui < Z) & (vi >= 0) & (vi < Z); ok[ok] = m[vi[ok], ui[ok]]; ok &= cs > best
            col[ok] = img[vi[ok], ui[ok]] / 255.0; best[ok] = cs[ok]
        col[np.isnan(col[:, 0])] = np.nanmedian(col, 0) if np.isfinite(col).any() else 0.5
        k = ~shown
        for k_, v_ in (("AX", Psd), ("AN", Nd), ("AC", col), ("Aobj", np.full(len(Psd), oid))):   # whole hull surface (mv_objfit.py)
            seeds.setdefault(k_, []).append(v_)
        VR = os.path.join(a.out_dir, "views_real")                         # real input crops + masks (mv_objfit.py)
        for sd_ in ("images", "masks", "sparse/0"):
            os.makedirs(os.path.join(VR, sd_), exist_ok=True)
        for i_, ((img, K, c2w), m) in enumerate(zip(ins, rm)):
            nm = f"obj{oid}_in{i_:02d}.png"; Image.fromarray(img).save(os.path.join(VR, "images", nm))
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(VR, "masks", nm.replace(".png", "_mask.png")))
            Z = img.shape[0]; w2c = np.linalg.inv(c2w); q = rot2qvec(w2c[:3, :3]); cid = len(rcrow) + 1
            rcrow.append(f"{cid} PINHOLE {Z} {Z} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
            rirow.append(f"{len(rirow) + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in w2c[:3, 3])} {cid} {nm}\n\n")
        seeds["X"].append(Psd[k]); seeds["N"].append(Nd[k]); seeds["C"].append(col[k]); seeds["obj"].append(np.full(int(k.sum()), oid))
        ov = []
        for (img, K, c2w, nm), m in zip(tgs, gm):
            Z = img.shape[0]; Image.fromarray(img).save(os.path.join(V, "images", nm))
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(V, "masks", nm.replace(".png", "_mask.png")))
            u, v, z = proj(Pd[hull_], K, c2w); Mh = np.zeros((Z, Z), np.uint8); ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
            inn = (z > 0) & (ui >= 0) & (ui < Z) & (vi >= 0) & (vi < Z); Mh[vi[inn], ui[inn]] = 1
            Mh = cv2.morphologyEx(cv2.dilate(Mh, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
            Image.fromarray((Mh * 255).astype(np.uint8)).save(os.path.join(V, "masks", nm.replace(".png", "_hull.png")))
            cid = len(crow) + 1; w2c = np.linalg.inv(c2w); q = rot2qvec(w2c[:3, :3])
            crow.append(f"{cid} PINHOLE {Z} {Z} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
            irow.append(f"{len(irow) + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in w2c[:3, 3])} {cid} {nm}\n\n")
            im = img.copy(); e = Mh ^ cv2.erode(Mh.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); im[e] = [255, 0, 0]
            e2 = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool); im[e2] = [0, 255, 0]
            ov.append(np.asarray(Image.fromarray(im).resize((192, 192))))
        rws = [np.hstack(ov[i:i + 12]) for i in range(0, len(ov), 12)]
        if len(rws) > 1 and rws[-1].shape[1] < rws[0].shape[1]:
            rws[-1] = np.pad(rws[-1], ((0, 0), (0, rws[0].shape[1] - rws[-1].shape[1]), (0, 0)))
        Image.fromarray(np.vstack(rws)).save(os.path.join(a.out_dir, f"obj{oid}_masks.jpg"), quality=88)
        rr = {"id": oid, "c0": c0.tolist(), "axes": A.tolist(), "box_lo": lo.round(3).tolist(), "box_hi": hi.round(3).tolist(),
              "real_masks_px": [int(x.sum()) for x in rm], "target_masks_px": [int(x.sum()) for x in gm], "hull_voxels": int(hull_.sum()),
              "real_hull_voxels": int(inside.sum()), "surface_seeds": int(len(Psd)), "shown_in_training_views": int(shown.sum()), "seeds": int(k.sum())}
        rep.append(rr); print("[seva-hull]", json.dumps({k_: v_ for k_, v_ in rr.items() if k_ not in ("c0", "axes")}), flush=True)
    open(os.path.join(V, "sparse", "0", "cameras.txt"), "w").writelines(crow)
    open(os.path.join(V, "sparse", "0", "images.txt"), "w").writelines(irow)
    open(os.path.join(V, "sparse", "0", "points3D.txt"), "w").write("")
    if rcrow:
        VR = os.path.join(a.out_dir, "views_real")
        open(os.path.join(VR, "sparse", "0", "cameras.txt"), "w").writelines(rcrow)
        open(os.path.join(VR, "sparse", "0", "images.txt"), "w").writelines(rirow)
        open(os.path.join(VR, "sparse", "0", "points3D.txt"), "w").write("")
    np.savez_compressed(os.path.join(a.out_dir, "seeds.npz"), **{k_: np.concatenate(v_) for k_, v_ in seeds.items()})
    json.dump(rep, open(os.path.join(a.out_dir, "register.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("generate", "hull"):
        g = sub.add_parser(nm)
        g.add_argument("--objects", required=True); g.add_argument("--sim3-json", required=True); g.add_argument("--dataset", required=True)
        g.add_argument("--splat-depth-dir", default=None); g.add_argument("--out-dir", required=True)
        g.add_argument("--ids", type=int, nargs="*", default=None)
        g.add_argument("--views", type=int, default=6); g.add_argument("--frame-stride", type=int, default=1)
        g.add_argument("--cell", type=float, default=0.01); g.add_argument("--hcell", type=float, default=0.02)
        g.add_argument("--max-incidence", type=float, default=85.0); g.add_argument("--depth-tol-m", type=float, default=0.03)
        g.add_argument("--splat-tol-rel", type=float, default=0.05); g.add_argument("--support-m", type=float, default=0.03)
        g.add_argument("--max-trunc", type=float, default=0.05); g.add_argument("--min-samples", type=int, default=200)
        g.add_argument("--min-crop-px", type=int, default=40); g.add_argument("--mask-tol-rel", type=float, default=0.05)
    g = sub.choices["generate"]
    g.add_argument("--model", default="stabilityai/stable-virtual-camera"); g.add_argument("--version", type=float, default=1.1)
    g.add_argument("--vae-repo", default="sd2-community/stable-diffusion-2-1-base")
    g.add_argument("--size", type=int, default=576); g.add_argument("--T", type=int, default=21)
    g.add_argument("--pad", type=float, default=2.5, help="input crop side = pad x the object's mask bbox (inside the frame)")
    g.add_argument("--min-crop", type=int, default=256)
    g.add_argument("--n-az", type=int, default=12); g.add_argument("--heights", type=float, nargs="+", default=[0.45, 0.8])
    g.add_argument("--dist", type=float, default=0.0, help="orbit radius m (0: median input distance, clipped)")
    g.add_argument("--dist-min", type=float, default=0.8); g.add_argument("--dist-max", type=float, default=1.6)
    g.add_argument("--fill", type=float, default=0.45, help="object radius / half frame")
    g.add_argument("--cfg", type=float, default=3.0); g.add_argument("--camera-scale", type=float, default=2.0)
    g.add_argument("--steps", type=int, default=50); g.add_argument("--seed", type=int, default=23)
    g.add_argument("--chunk-strategy", default="nearest-gt")
    h = sub.choices["hull"]
    h.add_argument("--birefnet", default="ZhengPeng7/BiRefNet"); h.add_argument("--box-grow", type=float, default=0.1)
    h.add_argument("--margin-m", type=float, default=0.06); h.add_argument("--floor-m", type=float, default=0.005)
    h.add_argument("--real-dilate", type=int, default=5); h.add_argument("--gen-dilate", type=int, default=4)
    h.add_argument("--gen-miss", type=int, default=3)
    a = ap.parse_args()
    {"generate": generate, "hull": hull}[a.cmd](a)


if __name__ == "__main__":
    main()
