#!/usr/bin/env python3
"""Scene-wide GENERATIVE repair of a splat, distilled back as pseudo-views (the generation step; IBR pseudo-views of
scripts/make_lidar_ibr_views.py only re-project what a camera saw).

A video world model (NVIDIA ArtiFixer; context = the covisible real training frames, opacity = what the training
cameras saw) repairs renders of the current splat along camera trajectories that LEAVE the recorded path progressively
(Difix3D+-style rounds: render -> repair -> distil -> render farther out ...). Invented content is allowed where no
camera looked; where the LiDAR IBR has real pixels they win.

  paths    (CPU) trajectories of exactly --n frames each, all starting ON real training frames (the video model needs
           a clean start), in the training set's frame; --level scales how far they go:
             track   a window of consecutive training frames re-driven with a lateral offset (alternating sides,
                     0.35 m x level) and lift (0.2 m x level), pitched down accordingly; ramped in after --n-real frames
             bridge  across every gap of the recorded trajectory (step > --gap x median step): the pose before the gap
                     interpolated to the pose after it (regions a trajectory skipped)
             orbit   (level >= 2) approach from the nearest training frame, then once round the room centre, 1.0 m
                     (level 3: also 1.5 m) above the floor, looking down at the centre
           Offsets shrink where the eye would come within --clear-m of the LiDAR cloud. -> OUT/sparse/0, OUT/paths.json
  prep     (gsplat venv, GPU) the PLY along the paths (viewer-exact rasteriser) -> OUT/render; SEEN per pixel (its
           rendered surface point lies on the rendered surface of a training view, as scripts/cosmetic_afx.py) ->
           OUT/seen, ArtiFixer opacity (255 seen / 0 unseen / alpha where empty) and prepared scenes OUT/afx/.
  collect  (CPU) ArtiFixer frames -> OUT/repaired/<name>.png, OUT/strips/<path>.jpg (render | opacity | repaired).
  select   (CPU) the frames that become pseudo-views (every --every-th after the real start) -> OUT/sel (COLMAP model)
  compose  (CPU) pseudo-dataset for scripts/train_distill_gsplat.py --pseudo-soft: every --every-th frame after the
           real start; image = LiDAR-IBR pixel where its verified mask holds (--ibr: make_lidar_ibr_views.py views
           --poses-from OUT), the repaired frame elsewhere; _mask.png weight 255 (IBR) / --w-seen (repaired, a camera
           saw it) / --w-unseen (generated); _depth.npy = the IBR run's LiDAR depth (generated surfaces still get the
           measured geometry).
  union    several pseudo-datasets -> one (links).

    python scripts/afx_scene.py paths --dataset TRAIN --cloud lidar.ply --scene-json scene.json --level 1 --out W
    python scripts/afx_scene.py prep --ply s.ply --dataset TRAIN --paths W --scene-json scene.json --caption c.h5 --seq NAME
    (ArtiFixer run_inference on W/afx/split.json);  python scripts/afx_scene.py collect --paths W --afx-run RUN --seq NAME
    python scripts/afx_scene.py compose --paths W --ibr W/ibr --out W/pseudo
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402
from make_pseudo_views import rot2quat, write_images  # noqa: E402


def smooth01(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def look(f, n):
    """c2w rotation (OpenCV: x right, y down, z forward) for a roll-free camera looking along f; n = up."""
    f = f / np.linalg.norm(f); r = np.cross(f, n); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f], 1)


def paths(a):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation, Slerp
    sc = json.load(open(a.scene_json)); n = np.asarray(sc["floor"]["n"], float); d = float(sc["floor"]["d"])
    s, R, t = float(sc["sim3"]["scale"]), np.asarray(sc["sim3"]["R"]), np.asarray(sc["sim3"]["t"])
    cams, imgs = read_text_model(os.path.join(a.dataset, "sparse", "0")); imgs = sorted(imgs, key=lambda i: i[0])
    Rm = np.stack([R @ quat_to_R(q).T for _, _, q, _ in imgs])                                   # c2w rotations, metric frame
    Cm = s * np.array([-quat_to_R(q).T @ tt for _, _, q, tt in imgs]) @ R.T + t
    v = PlyData.read(a.cloud)["vertex"]; L = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); tree = cKDTree(L)
    N = len(imgs); step = np.linalg.norm(np.diff(Cm, axis=0), axis=1); med = float(np.median(step))
    cuts = [0] + [i + 1 for i in range(N - 1) if step[i] > a.gap * med] + [N]
    segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= 16]
    lev = a.level; M = a.n; scenes = []

    def interp(u):                                                                                # fractional frame index -> pose
        i0 = int(np.floor(u)); i1 = min(i0 + 1, N - 1); w = u - i0
        if w < 1e-6 or i1 == i0:
            return Rm[i0], Cm[i0]
        return Slerp([0, 1], Rotation.from_matrix(np.stack([Rm[i0], Rm[i1]])))(w).as_matrix(), (1 - w) * Cm[i0] + w * Cm[i1]

    def clear(C0, C1):                                                                            # per-frame offset factor in [0, 1]
        f = np.ones(len(C0))
        for k in (0.75, 0.5, 0.25, 0.0):
            bad = tree.query(C0 + f[:, None] * (C1 - C0))[0] < a.clear_m
            f[bad] = np.minimum(f[bad], k)
        fm = np.array([f[max(0, i - 4): i + 5].min() for i in range(len(f))])
        return np.convolve(np.pad(fm, 4, mode="edge"), np.ones(9) / 9, mode="valid")

    side = 1
    for s0, s1 in segs:                                                                           # tracks
        win = min(M, s1 - s0)
        for i0 in range(s0, max(s0 + 1, s1 - win + 1), a.stride):
            us = np.linspace(i0, i0 + win - 1, M); P = [interp(u) for u in us]
            r = smooth01((np.arange(M) - a.n_real) / a.ramp)
            C0 = np.stack([c for _, c in P]); rt = np.stack([Rp[:, 0] - (Rp[:, 0] @ n) * n for Rp, _ in P])
            rt /= np.linalg.norm(rt, axis=1, keepdims=True)
            C1 = C0 + side * a.lateral * lev * rt + a.lift * lev * n
            fac = clear(C0, C1) * r; C = C0 + fac[:, None] * (C1 - C0)
            Rs = []
            for k, (Rp, _) in enumerate(P):
                ang = np.radians(a.pitch * lev) * fac[k]
                Rs.append(Rp if fac[k] < 1e-3 else look(Rp[:, 2] - np.tan(ang) * n, n))
            scenes.append((f"t{len(scenes):02d}", "track", Rs, list(C))); side = -side
    for k in range(len(segs) - 1):                                                                # bridges across trajectory gaps
        e, b = segs[k][1] - 1, segs[k + 1][0]
        if np.linalg.norm(Cm[b] - Cm[e]) < 0.3:
            continue
        nr = a.n_real; mid = M - 2 * nr
        Rs = [Rm[i] for i in range(e - nr + 1, e + 1)]; C = [Cm[i] for i in range(e - nr + 1, e + 1)]
        sl = Slerp([0, 1], Rotation.from_matrix(np.stack([Rm[e], Rm[b]])))
        for w in (np.arange(mid) + 1) / (mid + 1):
            Rs.append(sl(w).as_matrix()); C.append((1 - w) * Cm[e] + w * Cm[b])
        Rs += [Rm[i] for i in range(b, b + nr)]; C += [Cm[i] for i in range(b, b + nr)]
        scenes.append((f"b{len(scenes):02d}", "bridge", Rs, C))
    if lev >= 2:                                                                                  # elevated orbits
        h = L @ n + d; body = L[(h > 0.3) & (h < 2.0)]; c0 = np.median(body, 0); c0 -= (c0 @ n + d) * n
        e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        rel = body - c0; rad = np.hypot(rel @ e1, rel @ e2); rr = float(np.clip(0.6 * np.percentile(rad[rad > 0.5], 30), 1.0, 2.5))
        for hk in ([1.0] if lev == 2 else [1.0, 1.5]):
            j = int(np.argmin(np.abs(np.hypot((Cm - c0) @ e1, (Cm - c0) @ e2) - rr)))             # training frame nearest the ring
            th0 = np.arctan2((Cm[j] - c0) @ e2, (Cm[j] - c0) @ e1); na = 12
            ring = []
            for th in th0 + np.linspace(0, 2 * np.pi, M - na):
                Ck = c0 + rr * (np.cos(th) * e1 + np.sin(th) * e2) + hk * n
                ring.append((look((c0 + 0.3 * n) - Ck, n), Ck))
            sl = Slerp([0, 1], Rotation.from_matrix(np.stack([Rm[j], ring[0][0]])))
            Rs = [sl(w).as_matrix() for w in smooth01(np.arange(na) / na)]
            C = [(1 - w) * Cm[j] + w * ring[0][1] for w in smooth01(np.arange(na) / na)]
            Rs += [r_ for r_, _ in ring]; C += [c_ for _, c_ in ring]
            scenes.append((f"o{len(scenes):02d}", "orbit", Rs, C))
    s0_ = os.path.join(a.out, "sparse", "0"); os.makedirs(s0_, exist_ok=True)
    open(os.path.join(s0_, "cameras.txt"), "w").writelines(open(os.path.join(a.dataset, "sparse", "0", "cameras.txt")))
    open(os.path.join(s0_, "points3D.txt"), "w").close()
    cid = sorted(cams)[0]; rows, meta = [], []
    for name, kind, Rs, C in scenes:
        assert len(Rs) == M and len(C) == M, (name, len(Rs), len(C))
        fr = []
        for k in range(M):
            Rc = R.T @ Rs[k]; Cc = R.T @ (C[k] - t) / s; nm = f"{name}_{k:03d}.png"
            rows.append((len(rows) + 1, rot2quat(Rc.T), -Rc.T @ Cc, cid, nm)); fr.append(nm)
        dist = cKDTree(Cm).query(np.stack(C))[0]
        meta.append({"scene": name, "kind": kind, "frames": fr, "n_real": a.n_real if kind != "orbit" else 0,
                     "max_dist_to_train_m": round(float(dist.max()), 3), "mean_dist_to_train_m": round(float(dist.mean()), 3),
                     "max_height_m": round(float((np.stack(C) @ n + d).max()), 3)})
    write_images(os.path.join(s0_, "images.txt"), rows)
    json.dump({"level": lev, "args": {k: v_ for k, v_ in vars(a).items() if k != "func"}, "scenes": meta}, open(os.path.join(a.out, "paths.json"), "w"), indent=1)
    print(f"[afx-scene] level {lev}: {len(meta)} paths x {M} frames ({[m['kind'] for m in meta]}), max distance to a training camera "
          f"{max(m['max_dist_to_train_m'] for m in meta):.2f} m -> {a.out}", flush=True)


def prep(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t, save_png
    dev = "cuda"
    s = float(json.load(open(a.scene_json))["sim3"]["scale"])
    meta = json.load(open(os.path.join(a.paths, "paths.json")))
    _, p = load_ply(a.ply, dev); deg = sh_deg(p)
    Rv = ViewerRaster("ss352", "exact", display_scale=a.display_scale, sort="depth")
    tr = frames(a.dataset, 0, "all")
    TD, TW = [], []
    with torch.no_grad():
        for name, cam, c2w in tr:
            fx, fy, cx, cy, W, H = cam_args(cam)
            _, al, dep = Rv.render(p, to_t(np.linalg.inv(c2w), dev), fx, fy, cx, cy, W, H, sh_degree=deg, depth=True)
            TD.append(torch.where(al > 0.5, dep, torch.zeros_like(dep)).half()); TW.append(np.linalg.inv(c2w))
    TD = torch.stack(TD); TWt = torch.tensor(np.stack(TW), dtype=torch.float32, device=dev)
    TC = torch.tensor(np.stack([c[:3, 3] for _, _, c in tr]), dtype=torch.float32, device=dev)
    K0 = cam_args(tr[0][1])
    pf = {nm: (cam, c2w) for nm, cam, c2w in frames(a.paths, 0, "all")}
    for d_ in ("render", "seen", "opacity"):
        os.makedirs(os.path.join(a.paths, d_), exist_ok=True)
    cmax = float(np.cos(np.radians(a.max_angle))); st = a.stride; stats = []
    afx = os.path.join(a.paths, "afx"); os.makedirs(afx, exist_ok=True); split = {}
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    for sc in meta["scenes"]:
        sid = f"{a.seq}_{sc['scene']}"; sd = os.path.join(afx, sid)
        for d_ in ("renders", "opacity"):
            os.makedirs(os.path.join(sd, d_), exist_ok=True)
        fr_t = []; sf = []
        for i, nm in enumerate(sc["frames"]):
            cam, c2w = pf[nm]; fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w)
            with torch.no_grad():
                rgb, al, dep = Rv.render(p, to_t(w2c, dev), fx, fy, cx, cy, W, H, sh_degree=deg, depth=True)
                vv, uu = torch.meshgrid(torch.arange(0, H, st, device=dev), torch.arange(0, W, st, device=dev), indexing="ij")
                z = dep[vv, uu]; ok = al[vv, uu] > 0.5
                Xc = torch.stack([(uu + 0.5 - cx) / fx * z, (vv + 0.5 - cy) / fy * z, z], -1).reshape(-1, 3)
                c2wt = to_t(c2w, dev); Xw = Xc @ c2wt[:3, :3].T + c2wt[:3, 3]
                do = Xw - c2wt[:3, 3]; do = do / do.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                seen = torch.zeros(len(Xw), dtype=torch.bool, device=dev)
                for b in range(0, len(tr), a.batch):
                    Wb = TWt[b:b + a.batch]
                    Pc = torch.einsum("bij,nj->bni", Wb[:, :3, :3], Xw) + Wb[:, None, :3, 3]
                    zj = Pc[..., 2]; zs = zj.clamp_min(1e-6)
                    uj = (K0[0] * Pc[..., 0] / zs + K0[2]).floor().long(); vj = (K0[1] * Pc[..., 1] / zs + K0[3]).floor().long()
                    inb = (zj > 0.05) & (uj >= 0) & (uj < K0[4]) & (vj >= 0) & (vj < K0[5])
                    bi = torch.arange(len(Wb), device=dev)[:, None].expand_as(uj) + b
                    D = TD[bi, vj.clamp(0, K0[5] - 1), uj.clamp(0, K0[4] - 1)].float()
                    on = inb & (D > 0) & ((zj - D).abs() <= torch.clamp(a.tol_rel * D, min=a.tol_m / s))
                    dj = Xw[None] - TC[b:b + a.batch, None]; dj = dj / dj.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    on &= (dj * do[None]).sum(-1) >= cmax
                    seen |= on.any(0)
                seen = (seen & ok.reshape(-1)).reshape(vv.shape).float()
                seen = F.interpolate(seen[None, None], size=(H, W), mode="nearest")[0, 0] > 0.5
                op = torch.where(al > 0.5, torch.where(seen, torch.ones_like(al), torch.zeros_like(al)), al)
            save_png(os.path.join(a.paths, "render", nm), rgb)
            Image.fromarray((seen.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(a.paths, "seen", nm))
            opu = (op.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
            Image.fromarray(opu).save(os.path.join(a.paths, "opacity", nm))
            save_png(os.path.join(sd, "renders", f"{i:05d}.png"), rgb)
            Image.fromarray(opu).save(os.path.join(sd, "opacity", f"{i:05d}.png"))
            fr_t.append(c2w @ flip); sf.append(float(seen.float().mean()))
        intr = {"camera_model": "OPENCV", "fl_x": K0[0], "fl_y": K0[1], "cx": K0[2], "cy": K0[3], "w": K0[4], "h": K0[5],
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
        fr = [dict(intr, transform_matrix=m_.tolist()) for m_ in fr_t]
        fr += [dict(intr, file_path=f"images/{nm}", transform_matrix=(c2w @ flip).tolist()) for nm, _, c2w in tr]
        json.dump(dict(intr, frames=fr), open(os.path.join(sd, "transforms.json"), "w"))
        nt = len(sc["frames"])
        json.dump(list(range(nt)), open(os.path.join(sd, "target_indices.json"), "w"))
        json.dump(list(range(nt, nt + len(tr))), open(os.path.join(sd, "selected_indices.json"), "w"))
        split[sid] = {"camera_scale": s * 0.01, "metric_scale": s, "has_gt": False, "scene_id": sid,
                      "image_root": os.path.abspath(a.image_root or a.dataset), "prompt_path": os.path.abspath(a.caption),
                      "transforms_path": f"{sid}/transforms.json", "render_dir": f"{sid}/renders", "opacity_dir": f"{sid}/opacity",
                      "selected_indices_path": f"{sid}/selected_indices.json", "target_indices_path": f"{sid}/target_indices.json"}
        stats.append({"scene": sc["scene"], "kind": sc["kind"], "seen_mean": round(float(np.mean(sf)), 4), "seen_min": round(float(np.min(sf)), 4)})
        print(f"[afx-scene] {sid} ({sc['kind']}): {nt} frames, seen share mean {stats[-1]['seen_mean']} min {stats[-1]['seen_min']}", flush=True)
    json.dump({"test": split}, open(os.path.join(afx, "split.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(a.paths, "prep_stats.json"), "w"), indent=1)
    print(f"[afx-scene] ArtiFixer split with {len(split)} paths -> {afx}/split.json", flush=True)


def collect(a):
    import glob, shutil
    import cv2
    meta = json.load(open(os.path.join(a.paths, "paths.json")))
    out = os.path.join(a.paths, "repaired"); os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(a.paths, "strips"), exist_ok=True); n_ok = 0
    for sc in meta["scenes"]:
        sid = f"{a.seq}_{sc['scene']}"
        cand = sorted(glob.glob(os.path.join(a.afx_run, "**", sid, "frames", "batch_*", "pred"), recursive=True))
        if not cand:
            print(f"[afx-scene] {sid}: no ArtiFixer frames", flush=True); continue
        fs = sorted(os.path.join(c, f) for c in cand for f in os.listdir(c) if f.endswith(".png"))
        if len(fs) != len(sc["frames"]):
            print(f"[afx-scene] {sid}: {len(fs)} frames for {len(sc['frames'])} poses, skipped", flush=True); continue
        for f, nm in zip(fs, sc["frames"]):
            shutil.copyfile(f, os.path.join(out, nm))
        n_ok += len(fs)
        pick = np.linspace(0, len(sc["frames"]) - 1, a.strip_n).round().astype(int); cols = []
        for i in pick:
            nm = sc["frames"][i]
            ims = [cv2.resize(cv2.imread(os.path.join(a.paths, d_, nm)), (320, 240)) for d_ in ("render", "opacity", "repaired")]
            cv2.putText(ims[0], nm[:-4], (4, 16), 0, 0.45, (0, 0, 255), 1); cols.append(np.vstack(ims))
        cv2.imwrite(os.path.join(a.paths, "strips", sc["scene"] + ".jpg"), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[afx-scene] {n_ok} repaired frames -> {out}", flush=True)


def select(a):
    """the frames that become pseudo-views (every --every-th after the real start) as their own COLMAP model
    OUT/sel/sparse/0, so the IBR (make_lidar_ibr_views.py views --poses-from OUT/sel) runs on those only."""
    meta = json.load(open(os.path.join(a.paths, "paths.json"))); src0 = os.path.join(a.paths, "sparse", "0")
    lines = {ln.split()[9]: ln.split() for ln in open(os.path.join(src0, "images.txt")) if not ln.startswith("#") and len(ln.split()) >= 10}
    s0 = os.path.join(a.paths, "sel", "sparse", "0"); os.makedirs(s0, exist_ok=True); os.makedirs(os.path.join(a.paths, "sel", "images"), exist_ok=True)
    rows = []
    for sc in meta["scenes"]:
        for nm in sc["frames"][max(sc.get("n_real", 0), a.skip_first):: a.every]:
            p = lines[nm]; rows.append((len(rows) + 1, [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), nm))
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(src0, "cameras.txt")))
    open(os.path.join(s0, "points3D.txt"), "w").close(); write_images(os.path.join(s0, "images.txt"), rows)
    print(f"[afx-scene] {len(rows)} frames selected -> {a.paths}/sel", flush=True)


def compose(a):
    from PIL import Image
    meta = json.load(open(os.path.join(a.paths, "paths.json")))
    sel = os.path.join(a.paths, "sel", "sparse", "0", "images.txt")
    keep = {ln.split()[9] for ln in open(sel) if not ln.startswith("#") and len(ln.split()) >= 10} if os.path.exists(sel) and not a.dense else None
    im = os.path.join(a.out, "images"); s0 = os.path.join(a.out, "sparse", "0"); os.makedirs(im, exist_ok=True); os.makedirs(s0, exist_ok=True)
    src0 = os.path.join(a.paths, "sparse", "0")
    lines = {ln.split()[9]: ln.split() for ln in open(os.path.join(src0, "images.txt")) if not ln.startswith("#") and len(ln.split()) >= 10}
    rows, st = [], {"ibr": [], "seen": [], "gen": []}
    for sc in meta["scenes"]:
        if a.kinds and sc["kind"] not in a.kinds:
            continue
        for nm in ([f_ for f_ in sc["frames"] if f_ in keep] if keep is not None else sc["frames"][max(sc.get("n_real", 0), a.skip_first):: a.every]):
            rp = os.path.join(a.paths, "repaired", nm)
            if not os.path.exists(rp):
                continue
            seen = np.asarray(Image.open(os.path.join(a.paths, "seen", nm)).convert("L")) > 127
            H, W = seen.shape; stem = nm[:-4]
            img = np.asarray(Image.open(rp).convert("RGB").resize((W, H), Image.LANCZOS)).copy()
            wt = np.where(seen, int(round(a.w_seen * 255)), int(round(a.w_unseen * 255))).astype(np.uint8)
            hard = np.zeros_like(seen)
            if a.ibr and os.path.exists(os.path.join(a.ibr, "images", stem + "_mask.png")):
                hard = np.asarray(Image.open(os.path.join(a.ibr, "images", stem + "_mask.png")).convert("L")) > 127
                ib = np.asarray(Image.open(os.path.join(a.ibr, "images", nm)).convert("RGB"))
                img[hard] = ib[hard]; wt[hard] = 255
                dp = os.path.join(a.ibr, "images", stem + "_depth.npy")
                if os.path.exists(dp) and not os.path.lexists(os.path.join(im, stem + "_depth.npy")):
                    os.symlink(os.path.realpath(dp), os.path.join(im, stem + "_depth.npy"))
            Image.fromarray(img).save(os.path.join(im, nm)); Image.fromarray(wt, mode="L").save(os.path.join(im, stem + "_mask.png"))
            p = lines[nm]; rows.append((len(rows) + 1, [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), nm))
            st["ibr"].append(float(hard.mean())); st["seen"].append(float((seen & ~hard).mean())); st["gen"].append(float((~seen & ~hard).mean()))
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(src0, "cameras.txt")))
    open(os.path.join(s0, "points3D.txt"), "w").close(); write_images(os.path.join(s0, "images.txt"), rows)
    info = {"n_views": len(rows), "share_ibr": round(float(np.mean(st["ibr"])), 4), "share_repaired_seen": round(float(np.mean(st["seen"])), 4),
            "share_generated": round(float(np.mean(st["gen"])), 4), "args": {k: v for k, v in vars(a).items() if k != "func"}}
    json.dump(info, open(os.path.join(a.out, "pseudo.json"), "w"), indent=1)
    print("AFX_COMPOSE " + json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)


def union(a):
    im = os.path.join(a.out, "images"); s0 = os.path.join(a.out, "sparse", "0"); os.makedirs(im, exist_ok=True); os.makedirs(s0, exist_ok=True)
    rows, used = [], set()
    for k, ds in enumerate(a.inputs):
        for ln in [x for x in open(os.path.join(ds, "sparse", "0", "images.txt")) if not x.startswith("#")][::2]:
            p = ln.split()
            if len(p) < 10:
                continue
            stem, ext = os.path.splitext(p[9])
            out_stem = stem if stem not in used else f"u{k}_{stem}"      # the same frame name in two sets = two different views
            used.add(out_stem)
            for suf in (ext, "_mask.png", "_depth.npy"):
                src = os.path.join(ds, "images", stem + suf); dst = os.path.join(im, out_stem + suf)
                if os.path.exists(src) and not os.path.lexists(dst):
                    os.symlink(os.path.realpath(src), dst)
            rows.append((len(rows) + 1, [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), out_stem + ext))
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(a.inputs[0], "sparse", "0", "cameras.txt")))
    open(os.path.join(s0, "points3D.txt"), "w").close(); write_images(os.path.join(s0, "images.txt"), rows)
    print(f"[afx-scene] union of {len(a.inputs)} pseudo-datasets: {len(rows)} views -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths")
    p.add_argument("--dataset", required=True); p.add_argument("--cloud", required=True); p.add_argument("--scene-json", required=True)
    p.add_argument("--out", required=True); p.add_argument("--level", type=int, default=1)
    p.add_argument("--n", type=int, default=109); p.add_argument("--n-real", type=int, default=8); p.add_argument("--ramp", type=int, default=30)
    p.add_argument("--stride", type=int, default=45, help="training frames between track starts")
    p.add_argument("--lateral", type=float, default=0.35); p.add_argument("--lift", type=float, default=0.2); p.add_argument("--pitch", type=float, default=8.0)
    p.add_argument("--gap", type=float, default=4.0); p.add_argument("--clear-m", type=float, default=0.25)
    p.set_defaults(func=paths)
    b = sub.add_parser("prep")
    b.add_argument("--ply", required=True); b.add_argument("--dataset", required=True); b.add_argument("--paths", required=True)
    b.add_argument("--scene-json", required=True); b.add_argument("--caption", required=True); b.add_argument("--image-root", default=None)
    b.add_argument("--seq", required=True); b.add_argument("--display-scale", type=float, default=1.8); b.add_argument("--stride", type=int, default=2)
    b.add_argument("--tol-m", type=float, default=0.03); b.add_argument("--tol-rel", type=float, default=0.03)
    b.add_argument("--max-angle", type=float, default=70.0); b.add_argument("--batch", type=int, default=48)
    b.set_defaults(func=prep)
    c = sub.add_parser("collect")
    c.add_argument("--paths", required=True); c.add_argument("--afx-run", required=True); c.add_argument("--seq", required=True)
    c.add_argument("--strip-n", type=int, default=8); c.set_defaults(func=collect)
    e = sub.add_parser("select"); e.add_argument("--paths", required=True); e.add_argument("--every", type=int, default=8)
    e.add_argument("--skip-first", type=int, default=8); e.set_defaults(func=select)
    m = sub.add_parser("compose")
    m.add_argument("--paths", required=True); m.add_argument("--ibr", default=None); m.add_argument("--out", required=True)
    m.add_argument("--every", type=int, default=8); m.add_argument("--skip-first", type=int, default=8)
    m.add_argument("--dense", action="store_true", help="ignore OUT/sel: every --every-th frame of the paths (one consistent video pass)")
    m.add_argument("--kinds", nargs="*", default=[], help="only these path kinds (track / bridge / orbit)")
    m.add_argument("--w-seen", type=float, default=0.3); m.add_argument("--w-unseen", type=float, default=0.6)
    m.set_defaults(func=compose)
    u = sub.add_parser("union"); u.add_argument("--inputs", nargs="+", required=True); u.add_argument("--out", required=True)
    u.set_defaults(func=union)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
