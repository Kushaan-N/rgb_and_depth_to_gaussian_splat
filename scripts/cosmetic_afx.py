#!/usr/bin/env python3
"""Cosmetic unseen sides by IMAGE repair: render camera paths that orbit each free-standing object, repair the renders
with a video world model (NVIDIA ArtiFixer, conditioned on the nearest real training frames), distil the repaired frames
back into ONLY the objects' gaussians. Invents plausible backs / interiors (they need not match real images); every
gaussian outside the object zones is frozen (rows byte-identical), and the real training frames stay in the loss so
seen views stay close.

  paths    (CPU) per object (find_objects.py) and ring height: a path that STARTS at the real training view that best
           frames the object, approaches the ring (--n-approach frames, lerp position / slerp rotation onto a look-at
           pose) and goes once round the object (--n-orbit frames, full turn). Ring 1 at that camera's eye height, one
           more per --heights. Ring radius per azimuth shrunk until the eye and the line of sight are clear of the LiDAR
           cloud, then smoothed. -> OUT/sparse/0 (COLMAP text, the PLY's frame), OUT/paths.json
  prep     (gsplat venv, GPU) the PLY along the paths through the viewer-exact rasteriser (finetune_ply_viewer.py
           ViewerRaster, SuperSplat kernel): OUT/render/<name>.png; per pixel SEEN = its rendered surface point lies on
           the rendered surface of some training view (|z - D| <= max(--tol-m, --tol-rel D)) seen within --max-angle of
           the path ray; OUT/seen/<name>.png; ArtiFixer opacity = 255 seen / 0 unseen / render alpha where empty (what to
           keep vs regenerate); object masks OUT/mask/<name>.png (the object's zone gaussians' alpha + its closed 2.5D
           surface, dilated). ArtiFixer's prepared-scene layout per path under OUT/afx/<seq>_<path> (transforms = path
           targets + every training frame as context, selected / target indices, renders + opacity as 00000.png) and
           one OUT/afx/split.json for all paths (run_inference --render_trajectory trajectory, covisibility context).
  collect  (CPU) ArtiFixer's frames -> OUT/repaired/<name>.png; OUT/strips/<path>.jpg (render | opacity | repaired).
  pathmontage (CPU) per path, ring frames side by side from several render dirs (before | after | repaired).
  distill  (gsplat venv, GPU) fine-tune the gaussians inside the object zones (local min-area box + --zone-m, floor to
           top + --zone-m; colour AND geometry) plus new low-opacity gaussians seeded on each object's closed 2.5D surface
           (complete_objects.object_surface), viewer-exact (--sort dual, display scale --display-scale): real training
           frames that see a zone (full image L1+SSIM, share --real-frac) mixed with repaired path frames (L1+SSIM inside
           the object mask; pixels a training view saw weighted --seen-weight). Trainable gaussians that end below
           --prune-opacity are dropped. Writes <out>.ply + <out>_distill.json (real-frame PSNR before/after).

    python scripts/cosmetic_afx.py paths --objects S/objects --sim3-json sim3.json --dataset DS --cloud lidar.ply --out W
    python scripts/cosmetic_afx.py prep --ply in.ply --dataset DS --paths W --objects S/objects --sim3-json sim3.json \
        --caption caption.h5 --image-root DS --seq mocap2_well-lit_trot
    python scripts/cosmetic_afx.py collect --paths W --afx-run RUN
    python scripts/cosmetic_afx.py distill --ply in.ply --dataset DS --paths W --objects S/objects --sim3-json sim3.json \
        --out out.ply
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402
from solidify_unseen import sim3, load_objects, look_at  # noqa: E402


def obj_frame(X, n, d):
    from object_priors.object_priors import Obj
    return Obj(X, n, d)


def rot_to_q(Rcw):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rcw).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


# ---------------------------------------------------------------- paths (CPU)
def paths(args):
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation, Slerp
    s, R, t = sim3(args.sim3_json)
    ometa, opts = load_objects(args.objects); assert ometa is not None and len(opts), "no objects"
    n_up = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    W, H, (fx, fy, cx, cy) = cams[imgs[0][1]]
    to_m = lambda Xd: s * (Xd @ R.T) + t
    to_d = lambda Xm: ((Xm - t) @ R) / s
    Rtr = [quat_to_R(q) for _, _, q, _ in imgs]; Ttr = [tc for _, _, _, tc in imgs]
    Ctr = np.array([to_m(-Rc.T @ tc) for Rc, tc in zip(Rtr, Ttr)])
    e1 = np.cross(n_up, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n_up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n_up, e1)
    ltree = None
    if args.cloud and os.path.exists(args.cloud):
        from plyfile import PlyData
        cv = PlyData.read(args.cloud)["vertex"].data; L_ = np.stack([cv["x"], cv["y"], cv["z"]], 1).astype(np.float64)
        L_ = L_[np.unique(np.floor(L_ / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
        ltree = cKDTree(L_[(L_ @ n_up + d_f) > 0.05])                     # floor points never block the eye
    up_d = R.T @ n_up
    rows, scenes = [], []
    for o, X in zip(ometa["objects"], opts):
        h = X @ n_up + d_f; foot = X - np.outer(h, n_up); fc = foot.mean(0)
        hmax = float(np.percentile(h, 99)); c = fc + n_up * 0.5 * hmax
        rad = 0.5 * float(np.hypot(*o["footprint_m"])); r_pref = max(args.dist, 1.2 * rad + 0.6)
        cd = to_d(c)
        best = None                                                        # start: training view framing the object
        for j, (Rc, tc) in enumerate(zip(Rtr, Ttr)):
            pc_ = Rc @ cd + tc
            if pc_[2] < 0.1:
                continue
            u, v = fx * pc_[0] / pc_[2] + cx, fy * pc_[1] / pc_[2] + cy
            off = np.hypot((u - cx) / W, (v - cy) / H)
            if off > 0.45:
                continue
            hv = Ctr[j] - c; hv -= (hv @ n_up) * n_up; hd = np.linalg.norm(hv)
            sc_ = abs(hd - r_pref) + args.center_weight * off
            if best is None or sc_ < best[0]:
                best = (sc_, j, hd, hv)
        if best is None:
            print(f"[paths] object {o['id']}: no training view frames it, skipped", flush=True); continue
        _, j0, hd0, hv0 = best
        name0 = imgs[j0][0]; Rs = Rtr[j0]; Cs = Ctr[j0]; hc = float(Cs @ n_up + d_f)
        th0 = float(np.arctan2(hv0 @ e2, hv0 @ e1))
        sign = 1.0 if args.direction >= 0 else -1.0
        r0 = float(np.clip(hd0, max(rad + 0.5, 0.8 * r_pref), 1.3 * r_pref))
        for hh in [hc] + list(args.heights):
            ths = th0 + sign * 2 * np.pi * np.arange(args.n_orbit + 1) / args.n_orbit
            rr = np.full(len(ths), np.nan)
            for i, th in enumerate(ths):                                   # largest clear radius <= r0
                dirh = np.cos(th) * e1 + np.sin(th) * e2
                for r_ in np.arange(r0, rad + 0.35, -0.05):
                    Cm = fc + r_ * dirh + hh * n_up
                    if ltree is None:
                        rr[i] = r_; break
                    if ltree.query(Cm)[0] < args.clear_m:
                        continue
                    Lc = np.linalg.norm(c - Cm); u_ = (c - Cm) / Lc
                    pts = Cm + np.outer(np.arange(0.0, max(Lc - rad - 0.1, 0.0), 0.03), u_)
                    if not len(pts) or ltree.query(pts)[0].min() > args.clear_m:
                        rr[i] = r_; break
            ok = np.isfinite(rr)
            if ok.sum() < 0.5 * len(rr):
                print(f"[paths] object {o['id']} h {hh:.2f}: only {ok.sum()}/{len(rr)} clear azimuths, skipped", flush=True)
                continue
            idx = np.arange(len(rr)); rr = np.interp(idx, idx[ok], rr[ok], period=len(rr))
            k = args.smooth                                                # circular min filter, then box smoothing
            pad = np.concatenate([rr[-k:], rr, rr[:k]])
            rmin = np.array([pad[i:i + 2 * k + 1].min() for i in range(len(rr))])
            pad = np.concatenate([rmin[-k:], rmin, rmin[:k]])
            rr = np.array([pad[i:i + 2 * k + 1].mean() for i in range(len(rr))])
            ring = [fc + r_ * (np.cos(th) * e1 + np.sin(th) * e2) + hh * n_up for r_, th in zip(rr, ths)]
            poses = []
            q1, t1 = look_at(to_d(ring[0]), cd, up_d); R1 = quat_to_R(q1)
            sl = Slerp([0, 1], Rotation.from_matrix([Rs, R1]))
            for a in range(args.n_approach):                               # training pose -> ring start
                f = a / args.n_approach; f = 0.5 - 0.5 * np.cos(np.pi * f)   # ease in / out
                Cd = (1 - f) * to_d(Cs) + f * to_d(ring[0]); Rcw = sl(f).as_matrix()
                poses.append((Rcw, -Rcw @ Cd, "approach"))
            for Cm in ring:
                q_, t_ = look_at(to_d(Cm), cd, up_d); poses.append((quat_to_R(q_), t_, "ring"))
            scene = f"o{o['id']}_h{int(round(hh * 100)):03d}"; names = []
            for i, (Rcw, tt, seg) in enumerate(poses):
                nm = f"{scene}_{i:04d}.png"; names.append(nm); q = rot_to_q(Rcw)
                rows.append(f"{len(rows) + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in tt)} 1 {nm}\n\n")
            scenes.append({"scene": scene, "object": o["id"], "start_image": name0, "eye_h": round(hh, 3),
                           "radius_m": [round(float(x), 3) for x in rr], "n_approach": args.n_approach, "frames": names})
            print(f"[paths] {scene}: start {name0} (eye {hc:.2f} m, {hd0:.2f} m away), ring r {rr.min():.2f}-{rr.max():.2f} m, "
                  f"{len(names)} frames", flush=True)
    sp = os.path.join(args.out, "sparse", "0"); os.makedirs(sp, exist_ok=True)
    open(os.path.join(sp, "cameras.txt"), "w").write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    open(os.path.join(sp, "images.txt"), "w").writelines(rows); open(os.path.join(sp, "points3D.txt"), "w").write("")
    json.dump({"dataset": os.path.abspath(args.dataset), "scenes": scenes}, open(os.path.join(args.out, "paths.json"), "w"), indent=1)
    print(f"[paths] {len(scenes)} paths, {len(rows)} frames -> {args.out}", flush=True)


# ---------------------------------------------------------------- zones (shared by prep / distill)
def zones(Gm, ometa, opts, zone_m):
    """-> per object: (row indices of the gaussians in its zone, closed 2.5D surface samples (metric), Obj)."""
    from complete_objects import object_surface
    n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"]); out = {}
    for o, X in zip(ometa["objects"], opts):
        ob = obj_frame(X, n_f, d_f); Lx = ob.loc(X); hmax = float(np.percentile(Lx[:, 2], 99))
        Lg = ob.loc(Gm); zh = ob.half + zone_m
        inz = np.nonzero((np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > -0.02) &
                         (Lg[:, 2] < hmax + zone_m))[0]
        out[o["id"]] = (inz, object_surface(X, n_f, d_f, 0.03, 0.03)[0], ob)
    return out


def subset(p, idx):
    return {k: v[idx] for k, v in p.items()}


# ---------------------------------------------------------------- prep (GPU)
def prep(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t, save_png
    from build_depth_targets import zbuffer_target
    dev = "cuda"
    s, R, t = sim3(args.sim3_json); ometa, opts = load_objects(args.objects)
    meta = json.load(open(os.path.join(args.paths, "paths.json")))
    _, p = load_ply(args.ply, dev); deg = sh_deg(p)
    Gm = s * (p["means"].double().cpu().numpy() @ R.T) + t
    Z = zones(Gm, ometa, opts, args.zone_m)
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale, sort="depth")
    tr = frames(args.dataset, 0, "all")
    print(f"[prep] {len(tr)} training views: rendered depth", flush=True)
    TD, TW = [], []
    with torch.no_grad():
        for name, cam, c2w in tr:
            fx, fy, cx, cy, W, H = cam_args(cam)
            _, a, dep = Rv.render(p, to_t(np.linalg.inv(c2w), dev), fx, fy, cx, cy, W, H, sh_degree=deg, depth=True)
            TD.append(torch.where(a > 0.5, dep, torch.zeros_like(dep)).half()); TW.append(np.linalg.inv(c2w))
    TD = torch.stack(TD)                                                    # (V,H,W) z-depth, 0 = empty
    TWt = torch.tensor(np.stack(TW), dtype=torch.float32, device=dev)
    TC = torch.tensor(np.stack([c[:3, 3] for _, _, c in tr]), dtype=torch.float32, device=dev)
    K0 = cam_args(tr[0][1])
    pf = {n: (cam, c2w) for n, cam, c2w in frames(args.paths, 0, "all")}
    for d in ("render", "seen", "mask", "opacity"):
        os.makedirs(os.path.join(args.paths, d), exist_ok=True)
    cmax = float(np.cos(np.radians(args.max_angle))); st = args.stride; stats = []
    afx = os.path.join(args.paths, "afx"); os.makedirs(afx, exist_ok=True); split = {}
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    for sc in meta["scenes"]:
        inz, surf, _ = Z[sc["object"]]; pz = subset(p, torch.tensor(inz, device=dev))
        surf_d = torch.tensor(((surf - t) @ R) / s, device=dev)
        sid = f"{args.seq}_{sc['scene']}"; sd = os.path.join(afx, sid)
        for d in ("renders", "opacity"):
            os.makedirs(os.path.join(sd, d), exist_ok=True)
        fr_t = []
        for i, nm in enumerate(sc["frames"]):
            cam, c2w = pf[nm]; fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w)
            with torch.no_grad():
                rgb, a, dep = Rv.render(p, to_t(w2c, dev), fx, fy, cx, cy, W, H, sh_degree=deg, depth=True)
                # seen test on a strided grid
                vv, uu = torch.meshgrid(torch.arange(0, H, st, device=dev), torch.arange(0, W, st, device=dev), indexing="ij")
                z = dep[vv, uu]; ok = a[vv, uu] > 0.5
                Xc = torch.stack([(uu + 0.5 - cx) / fx * z, (vv + 0.5 - cy) / fy * z, z], -1).reshape(-1, 3)
                c2wt = to_t(c2w, dev); Xw = Xc @ c2wt[:3, :3].T + c2wt[:3, 3]
                do = Xw - c2wt[:3, 3]; do = do / do.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                seen = torch.zeros(len(Xw), dtype=torch.bool, device=dev)
                for b in range(0, len(tr), args.batch):
                    Wb = TWt[b:b + args.batch]
                    Pc = torch.einsum("bij,nj->bni", Wb[:, :3, :3], Xw) + Wb[:, None, :3, 3]
                    zj = Pc[..., 2]; zs = zj.clamp_min(1e-6)
                    uj = (K0[0] * Pc[..., 0] / zs + K0[2]).floor().long(); vj = (K0[1] * Pc[..., 1] / zs + K0[3]).floor().long()
                    inb = (zj > 0.05) & (uj >= 0) & (uj < K0[4]) & (vj >= 0) & (vj < K0[5])
                    bi = torch.arange(len(Wb), device=dev)[:, None].expand_as(uj) + b
                    D = TD[bi, vj.clamp(0, K0[5] - 1), uj.clamp(0, K0[4] - 1)].float()
                    on = inb & (D > 0) & ((zj - D).abs() <= torch.clamp(args.tol_rel * D, min=args.tol_m / s))
                    dj = Xw[None] - TC[b:b + args.batch, None]; dj = dj / dj.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    on &= (dj * do[None]).sum(-1) >= cmax
                    seen |= on.any(0)
                seen = (seen & ok.reshape(-1)).reshape(vv.shape).float()
                seen = F.interpolate(seen[None, None], size=(H, W), mode="nearest")[0, 0] > 0.5
                op = torch.where(a > 0.5, torch.where(seen, torch.ones_like(a), torch.zeros_like(a)), a)
                # object mask: zone gaussians' alpha + closed surface footprint, dilated
                _, az, _ = Rv.render(pz, to_t(w2c, dev), fx, fy, cx, cy, W, H, sh_degree=deg)
                Rcw = torch.tensor(w2c[:3, :3], device=dev); tcw = torch.tensor(w2c[:3, 3], device=dev)
                Ds, _ = zbuffer_target(surf_d, Rcw, tcw, W, H, (fx, fy, cx, cy), 0.03 / s, 1e6, min_support=1)
                m = ((az > 0.3) | (Ds > 0)).float()
                k = 2 * args.dilate + 1
                m = F.max_pool2d(m[None, None], k, 1, args.dilate)[0, 0] > 0.5
            save_png(os.path.join(args.paths, "render", nm), rgb)
            Image.fromarray((seen.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.paths, "seen", nm))
            Image.fromarray((m.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.paths, "mask", nm))
            opu = (op.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            Image.fromarray(opu).save(os.path.join(args.paths, "opacity", nm))
            save_png(os.path.join(sd, "renders", f"{i:05d}.png"), rgb)
            Image.fromarray(opu).save(os.path.join(sd, "opacity", f"{i:05d}.png"))
            fr_t.append(c2w @ flip)
            mm = m.cpu().numpy(); sv = seen.cpu().numpy()
            stats.append({"name": nm, "seen_frac": round(float(sv.mean()), 4),
                          "object_px": int(mm.sum()), "object_unseen_frac": round(float((~sv[mm]).mean()) if mm.any() else 0.0, 4)})
        # ArtiFixer prepared scene: targets = this path, context candidates = every training frame
        intr = {"camera_model": "OPENCV", "fl_x": K0[0], "fl_y": K0[1], "cx": K0[2], "cy": K0[3], "w": K0[4], "h": K0[5],
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
        fr = [dict(intr, transform_matrix=m_.tolist()) for m_ in fr_t]
        fr += [dict(intr, file_path=f"images/{n}", transform_matrix=(c2w @ flip).tolist()) for n, _, c2w in tr]
        json.dump(dict(intr, frames=fr), open(os.path.join(sd, "transforms.json"), "w"))
        nt = len(sc["frames"])
        json.dump(list(range(nt)), open(os.path.join(sd, "target_indices.json"), "w"))
        json.dump(list(range(nt, nt + len(tr))), open(os.path.join(sd, "selected_indices.json"), "w"))
        split[sid] = {"camera_scale": s * 0.01, "metric_scale": s, "has_gt": False, "scene_id": sid,
                      "image_root": os.path.abspath(args.image_root), "prompt_path": os.path.abspath(args.caption),
                      "transforms_path": f"{sid}/transforms.json", "render_dir": f"{sid}/renders", "opacity_dir": f"{sid}/opacity",
                      "selected_indices_path": f"{sid}/selected_indices.json", "target_indices_path": f"{sid}/target_indices.json"}
        ss_ = [x for x in stats if x["name"].startswith(sc["scene"] + "_")]
        print(f"[prep] {sid}: {nt} frames, seen {np.mean([x['seen_frac'] for x in ss_]):.3f}, object pixels unseen "
              f"{np.mean([x['object_unseen_frac'] for x in ss_]):.3f}, zone gaussians {len(inz):,}", flush=True)
    json.dump({"test": split}, open(os.path.join(afx, "split.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(args.paths, "prep_stats.json"), "w"), indent=0)
    print(f"[prep] ArtiFixer split with {len(split)} paths -> {afx}/split.json", flush=True)


# ---------------------------------------------------------------- collect (CPU)
def collect(args):
    import glob, shutil
    import cv2
    meta = json.load(open(os.path.join(args.paths, "paths.json")))
    out = os.path.join(args.paths, "repaired"); os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(args.paths, "strips"), exist_ok=True); n_ok = 0
    for sc in meta["scenes"]:
        sid = f"{args.seq}_{sc['scene']}"
        cand = sorted(glob.glob(os.path.join(args.afx_run, "**", sid, "frames", "batch_*", "pred"), recursive=True))
        if not cand:
            print(f"[collect] {sid}: no ArtiFixer frames", flush=True); continue
        pred = cand[0]; fs = sorted(f for f in os.listdir(pred) if f.endswith(".png"))
        assert len(fs) == len(sc["frames"]), (sid, len(fs), len(sc["frames"]))
        for f, nm in zip(fs, sc["frames"]):
            shutil.copyfile(os.path.join(pred, f), os.path.join(out, nm))
        n_ok += len(fs)
        pick = np.linspace(0, len(sc["frames"]) - 1, args.strip_n).round().astype(int); cols = []
        for i in pick:
            nm = sc["frames"][i]
            ims = [cv2.imread(os.path.join(args.paths, d, nm)) for d in ("render", "opacity", "repaired")]
            ims = [cv2.resize(im, (320, 240)) for im in ims]
            cv2.putText(ims[0], nm[:-4], (4, 16), 0, 0.45, (0, 0, 255), 1)
            cols.append(np.vstack(ims))
        cv2.imwrite(os.path.join(args.paths, "strips", sc["scene"] + ".jpg"), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[collect] {n_ok} repaired frames -> {out}", flush=True)


def pathmontage(args):
    """per path: --n frames x columns (label=dir; names = path frame names), e.g. before | after | repaired."""
    import cv2
    meta = json.load(open(os.path.join(args.paths, "paths.json"))); cols = [x.split("=", 1) for x in args.dirs]
    os.makedirs(args.out_dir, exist_ok=True)
    for sc in meta["scenes"]:
        fr = sc["frames"][sc["n_approach"]:]; rows = []
        for i in np.linspace(0, len(fr) - 1, args.n).round().astype(int):
            ims = []
            for lab, d in cols:
                im = cv2.imread(os.path.join(d, fr[i]))
                im = np.zeros((480, 640, 3), np.uint8) if im is None else im
                im = cv2.resize(im, (args.w, int(args.w * im.shape[0] / im.shape[1])))
                cv2.putText(im, f"{lab} {fr[i][:-4]}", (6, 20), 0, 0.55, (0, 0, 255), 2); ims.append(im)
            rows.append(np.hstack(ims))
        half = (len(rows) + 1) // 2
        if len(rows) > 1:
            L, Rr = rows[:half], rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
            rows = [np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b]) for a, b in zip(L, Rr)]
        cv2.imwrite(os.path.join(args.out_dir, sc["scene"] + ".jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[pathmontage] {len(meta['scenes'])} montages -> {args.out_dir}", flush=True)


# ---------------------------------------------------------------- distill (GPU)
def distill(args):
    import torch
    from PIL import Image
    from scipy.spatial import cKDTree
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t
    from refine_poses_gsplat import ssim, load_rgb, psnr
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = "cuda"; t0 = time.time()
    s, R, t = sim3(args.sim3_json); ometa, opts = load_objects(args.objects)
    meta = json.load(open(os.path.join(args.paths, "paths.json")))
    vertex, p0 = load_ply(args.ply, dev); deg = sh_deg(p0); N = len(p0["means"])
    Gm = s * (p0["means"].double().cpu().numpy() @ R.T) + t
    Z = zones(Gm, ometa, opts, args.zone_m)
    zone = np.zeros(N, bool)
    for inz, _, _ in Z.values():
        zone[inz] = True
    zi = np.nonzero(zone)[0]; fi = np.nonzero(~zone)[0]
    # seeds: closed 2.5D surface samples, low opacity, colour of the nearest zone gaussian
    op0 = torch.sigmoid(p0["opacities"]).cpu().numpy()
    seeds = np.concatenate([S_ for _, S_, _ in Z.values()]) if args.seed_cell > 0 else np.zeros((0, 3))
    if len(seeds):
        seeds = seeds[np.unique(np.floor(seeds / args.seed_cell).astype(np.int64), axis=0, return_index=True)[1]]
    ref = zi[op0[zi] > 0.3]
    nn_ = cKDTree(Gm[ref]).query(seeds)[1] if len(seeds) and len(ref) else np.zeros(len(seeds), int)
    M = len(seeds); K = p0["shN"].shape[1]
    new = {"means": torch.tensor(((seeds - t) @ R) / s, dtype=torch.float32, device=dev).reshape(M, 3),
           "scales": torch.full((M, 3), float(np.log(args.seed_scale_m / s)), device=dev),
           "quats": torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(M, 1),
           "opacities": torch.full((M,), float(np.log(args.seed_opacity / (1 - args.seed_opacity))), device=dev),
           "sh0": p0["sh0"][torch.tensor(ref[nn_], device=dev, dtype=torch.long)].clone() if M else torch.zeros(0, 1, 3, device=dev),
           "shN": torch.zeros(M, K, 3, device=dev)}
    zt = torch.tensor(zi, device=dev); ft = torch.tensor(fi, device=dev)
    frozen = subset(p0, ft)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.cat([p0[k][zt], new[k]]).clone()) for k in p0})
    print(f"[distill] {N:,} gaussians: {len(zi):,} in object zones trainable + {M:,} seeded, {len(fi):,} frozen", flush=True)
    full = lambda: {k: torch.cat([frozen[k], params[k]]) for k in params}
    # real frames that see a zone
    tr = frames(args.dataset, 0, "all"); zc = p0["means"][zt][:: max(1, len(zi) // 20000)]
    real = []
    for name, cam, c2w in tr:
        fx, fy, cx, cy, W, H = cam_args(cam); w2c = to_t(np.linalg.inv(c2w), dev)
        pc_ = zc @ w2c[:3, :3].T + w2c[:3, 3]; z = pc_[:, 2].clamp_min(1e-6)
        u, v = fx * pc_[:, 0] / z + cx, fy * pc_[:, 1] / z + cy
        if int(((pc_[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).sum()) >= args.min_zone_px:
            real.append((name, cam, c2w, load_rgb(os.path.join(args.dataset, "images", name), dev)))
    # repaired path frames
    pf = {n: (cam, c2w) for n, cam, c2w in frames(args.paths, 0, "all")}
    orb = []
    for sc in meta["scenes"]:
        for nm in sc["frames"][args.skip_first:]:
            f = os.path.join(args.paths, "repaired", nm)
            if not os.path.exists(f):
                continue
            m = np.asarray(Image.open(os.path.join(args.paths, "mask", nm))) > 127
            sv = np.asarray(Image.open(os.path.join(args.paths, "seen", nm))) > 127
            w = m * np.where(sv, args.seen_weight, 1.0)
            if m.sum() < 200:
                continue
            orb.append((nm, *pf[nm], load_rgb(f, dev), torch.tensor(w, dtype=torch.float32, device=dev)))
    print(f"[distill] {len(real)}/{len(tr)} real frames see a zone, {len(orb)} repaired path frames", flush=True)
    assert real and orb
    centres = np.stack([c[:3, 3] for _, _, c in tr]); scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    lr = {"means": args.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": args.opacity_lr, "sh0": 2.5e-3,
          "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in str(args.display_scale).split(":")]
    Rv = ViewerRaster("ss352", "exact", display_scale=rs[0], sort=args.sort)

    def real_psnr(pp, every):
        out = []
        with torch.no_grad():
            for name, cam, c2w, gt in real[::every]:
                img = Rv.render(pp, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0]
                out.append(psnr(torch.round(img * 255) / 255, gt.float() / 255))
        return float(np.mean(out))
    Rv.sort = "depth"; before = real_psnr(p0, args.eval_every); Rv.sort = args.sort
    print(f"[distill] real frames (zone-visible, every {args.eval_every}) PSNR before {before:.3f}", flush=True)
    log = {"args": vars(args), "n": N, "zone": int(len(zi)), "seeded": M, "real_frames": len(real), "path_frames": len(orb),
           "real_psnr_before": round(before, 3), "curve": []}
    for step in range(1, args.iters + 1):
        frac = step / args.iters; wu = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu * ((args.means_lr_final / args.means_lr) ** frac if g["name"] == "means" else 1.0)
        if len(rs) == 2:
            Rv.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[1]))))
        is_real = np.random.rand() < args.real_frac
        if is_real:
            name, cam, c2w, gt = real[np.random.randint(len(real))]; wm = None
        else:
            name, cam, c2w, gt, wm = orb[np.random.randint(len(orb))]
        img, alpha, _ = Rv.render(full(), to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        g_ = gt.float() / 255
        if wm is None:
            loss = (1 - args.ssim_lambda) * (img - g_).abs().mean() + args.ssim_lambda * (
                1 - ssim(img.permute(2, 0, 1)[None], g_.permute(2, 0, 1)[None]))
        else:
            w3 = wm[..., None]
            l1 = ((img - g_).abs() * w3).sum() / (3 * wm.sum()).clamp_min(1)
            mk = (wm > 0).float()[..., None]
            ls = 1 - ssim((img * mk).permute(2, 0, 1)[None], (g_ * mk).permute(2, 0, 1)[None])
            loss = args.orbit_weight * ((1 - args.ssim_lambda) * l1 + args.ssim_lambda * ls)
            if args.lam_alpha > 0:                                          # no see-through holes on the object
                loss = loss + args.lam_alpha * ((1 - alpha).clamp_min(0) * (wm > 0)).sum() / (wm > 0).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        for prm in params.values():
            if prm.grad is not None:
                prm.grad[~torch.isfinite(prm.grad)] = 0
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"[distill] step {step:6d} {'real' if is_real else 'path'} loss {loss.item():.4f} "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G ({time.time() - t0:.0f}s)", flush=True)
            log["curve"].append({"step": step, "loss": round(loss.item(), 5), "real": bool(is_real)})
    with torch.no_grad():
        opn = torch.sigmoid(params["opacities"]); thr = torch.full_like(opn, args.prune_opacity)
        keep_t = (opn >= thr).cpu().numpy()
        nz = len(zi); keep_z, keep_new = keep_t[:nz], keep_t[nz:]
        p_out = {k: v.clone() for k, v in p0.items()}
        for k in p_out:
            p_out[k][zt] = params[k][:nz].detach()
        keep = np.ones(N, bool); keep[zi[~keep_z]] = False
        pk = torch.tensor(np.nonzero(keep)[0], device=dev); nk = torch.tensor(np.nonzero(keep_new)[0] + nz, device=dev, dtype=torch.long)
        p_out = {k: torch.cat([v[pk], params[k][nk].detach()]) for k, v in p_out.items()}
        vd = vertex.data
        arr = np.concatenate([vd[keep], np.repeat(vd[:1], int(keep_new.sum()), axis=0)])
        Rv.sort = "depth"; Rv.r = rs[0]; after = real_psnr(p_out, args.eval_every)
    save_ply(args.out, arr, p_out)
    log.update({"real_psnr_after": round(after, 3), "pruned_zone": int((~keep_z).sum()), "kept_seeded": int(keep_new.sum()),
                "n_out": int(len(arr)), "seconds": round(time.time() - t0)})
    json.dump(log, open(os.path.splitext(args.out)[0] + "_distill.json", "w"), indent=1)
    print(f"[distill] real-frame PSNR {before:.3f} -> {after:.3f}; pruned {log['pruned_zone']:,} zone, kept {log['kept_seeded']:,}"
          f"/{M:,} seeded; {len(arr):,} gaussians -> {args.out}", flush=True)
    print("DISTILL_DONE", json.dumps({k: v for k, v in log.items() if k not in ("args", "curve")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("paths")
    a.add_argument("--objects", required=True); a.add_argument("--sim3-json", required=True)
    a.add_argument("--dataset", required=True, help="training COLMAP text dataset in the PLY's frame")
    a.add_argument("--cloud", default=None); a.add_argument("--out", required=True)
    a.add_argument("--n-orbit", type=int, default=96); a.add_argument("--n-approach", type=int, default=12)
    a.add_argument("--heights", type=float, nargs="*", default=[0.9], help="extra ring eye heights (m above the floor)")
    a.add_argument("--dist", type=float, default=1.2); a.add_argument("--clear-m", type=float, default=0.15)
    a.add_argument("--smooth", type=int, default=4); a.add_argument("--center-weight", type=float, default=2.0)
    a.add_argument("--direction", type=int, default=1)
    b = sub.add_parser("prep")
    b.add_argument("--ply", required=True); b.add_argument("--dataset", required=True); b.add_argument("--paths", required=True)
    b.add_argument("--objects", required=True); b.add_argument("--sim3-json", required=True)
    b.add_argument("--caption", required=True, help="ArtiFixer caption.h5 of the scene"); b.add_argument("--image-root", required=True)
    b.add_argument("--seq", required=True); b.add_argument("--zone-m", type=float, default=0.15)
    b.add_argument("--display-scale", type=float, default=1.8); b.add_argument("--stride", type=int, default=2)
    b.add_argument("--tol-m", type=float, default=0.03); b.add_argument("--tol-rel", type=float, default=0.03)
    b.add_argument("--max-angle", type=float, default=70.0); b.add_argument("--batch", type=int, default=48)
    b.add_argument("--dilate", type=int, default=6)
    c = sub.add_parser("collect")
    c.add_argument("--paths", required=True); c.add_argument("--afx-run", required=True); c.add_argument("--seq", required=True)
    c.add_argument("--strip-n", type=int, default=8)
    m = sub.add_parser("pathmontage")
    m.add_argument("--paths", required=True); m.add_argument("--dirs", nargs="+", required=True, help="label=dir")
    m.add_argument("--out-dir", required=True); m.add_argument("--n", type=int, default=8); m.add_argument("--w", type=int, default=400)
    d = sub.add_parser("distill")
    d.add_argument("--ply", required=True); d.add_argument("--dataset", required=True); d.add_argument("--paths", required=True)
    d.add_argument("--objects", required=True); d.add_argument("--sim3-json", required=True); d.add_argument("--out", required=True)
    d.add_argument("--zone-m", type=float, default=0.15); d.add_argument("--iters", type=int, default=5000)
    d.add_argument("--real-frac", type=float, default=0.5); d.add_argument("--orbit-weight", type=float, default=1.0)
    d.add_argument("--seen-weight", type=float, default=0.25); d.add_argument("--ssim-lambda", type=float, default=0.2)
    d.add_argument("--lam-alpha", type=float, default=0.05); d.add_argument("--skip-first", type=int, default=0)
    d.add_argument("--seed-cell", type=float, default=0.03); d.add_argument("--seed-scale-m", type=float, default=0.012)
    d.add_argument("--seed-opacity", type=float, default=0.05); d.add_argument("--prune-opacity", type=float, default=1 / 255)
    d.add_argument("--means-lr", type=float, default=4e-5); d.add_argument("--means-lr-final", type=float, default=2e-6)
    d.add_argument("--opacity-lr", type=float, default=0.025)
    d.add_argument("--display-scale", default="1.5:4"); d.add_argument("--sort", default="dual", choices=("depth", "radial", "dual"))
    d.add_argument("--min-zone-px", type=int, default=50); d.add_argument("--eval-every", type=int, default=4)
    d.add_argument("--log-every", type=int, default=250); d.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"paths": paths, "prep": prep, "collect": collect, "distill": distill, "pathmontage": pathmontage}[args.cmd](args)


if __name__ == "__main__":
    main()
