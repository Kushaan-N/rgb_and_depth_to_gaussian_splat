"""Find and remove the gaussians that draw haze in SuperSplat's default/off-path views but no training view constrains
(GPU node; `probe` in venv-viewer, `diagnose` / `clean` in venv-gsplat).

  probe     open the PLY in today's SuperSplat (scripts/viewer_shots.py open_editor), read the editor's DEFAULT camera
            (the view a user sees first) plus an orbit around its focal point, save the editor's own renders and the
            cameras in the PLY's frame (or, with --sim3-json of x_ply = s R x + t, in the target frame x).
  diagnose  per-gaussian statistics: blending-weight contribution to every training view (viewer-exact SuperSplat 3.5.2
            rasteriser, scripts/finetune_ply_viewer.py ViewerRaster, ss352 at --display-scale), distance to the static
            LiDAR cloud, distance to the nearest training camera, frustum count, scale / opacity / SH; and which of them
            carry the probe views' pixels (shares by class, renders with and without each candidate class).
  clean     remove the culprit class (gaussians whose max training blending weight < --max-train-w AND that lie
            > --min-lidar-dist from every LiDAR point, i.e. unconstrained free-space splats; LiDAR-supported fills of
            unseen surfaces are kept), then the measured harm check of scripts/fill_harm_ply.py in reverse: render all
            training frames for the original and the cleaned splat; any removed gaussian that blends into a pixel that
            changed by > --thr is restored; repeat until no training pixel changes.

    python scripts/clean_haze.py probe --ply s_refaxes.ply --out-dir D/probe_ref [--sim3-json ref_from_pipe.json] [--orbit 8]
    python scripts/clean_haze.py diagnose --ply s.ply --dataset <training frames> --cams D/probe_ref/cams.json \
        --cloud lidar_static.ply --sim3-json pipeline/sim3.json --out-dir D/diag
    python scripts/clean_haze.py clean --ply s.ply --dataset <training frames> [<more frames>] --cloud lidar_static.ply \
        --sim3-json pipeline/sim3.json --out s_clean.ply [--stats D/stats.npz (diagnose's cache; else computed)]
"""
from __future__ import annotations
import argparse, base64, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SS_CAM = """() => { const s = window.scene, cm = s.camera, e = cm.mainCamera || cm.entity, cc = (e && e.camera) || cm.entity.camera;
  const v = (x) => (x && x.x !== undefined) ? [x.x, x.y, x.z] : null; const p = e.getPosition(), q = e.getRotation();
  let fp = null; try { fp = v(cm.focalPoint) || v(cm.focalPointTween && cm.focalPointTween.value); } catch (err) {}
  return {p: v(p), q: [q.x, q.y, q.z, q.w], fov: cc.fov, hfov: !!cc.horizontalFov, near: cc.nearClip, far: cc.farClip,
          W: s.canvas.width, H: s.canvas.height, focal: fp, azim: cm.azim, elev: cm.elevation, dist: cm.distance,
          fovProp: cm.fov, bc: v(s.bound.center), bh: v(s.bound.halfExtents)}; }"""
TOP = np.diag([1.0, -1.0, -1.0])                                         # OpenCV <-> PlayCanvas camera axes


# ---------------------------------------------------------------- probe (venv-viewer)
def probe(args):
    from playwright.sync_api import sync_playwright
    from scipy.spatial.transform import Rotation
    from PIL import Image
    import tempfile
    from viewer_shots import SS_FIND, SS_SETUP, SS_SHOT, SS_VIEW, open_editor, serve
    os.makedirs(os.path.join(args.out_dir, "editor"), exist_ok=True)
    root = tempfile.mkdtemp(prefix="haze_")
    os.symlink(os.path.abspath(args.ply), os.path.join(root, "scene.ply"))
    port = serve(root)
    sim = None
    if args.sim3_json:
        j = json.load(open(args.sim3_json)); sim = (float(j["scale"]), np.asarray(j["R"]), np.asarray(j["t"]))
    cams = []

    def add(name, Pw, Rw, fov_deg, hfov, W, H, near, M, rgba):
        """editor-world PlayCanvas camera -> OpenCV c2w in the PLY frame (-> target frame)."""
        sM = np.cbrt(abs(np.linalg.det(M[:3, :3]))); RM = M[:3, :3] / sM
        C = np.linalg.solve(M[:3, :3], np.asarray(Pw) - M[:3, 3]); Rc = RM.T @ Rw @ TOP; nr = near / sM
        if sim:
            s, R, t = sim; C = R.T @ (C - t) / s; Rc = R.T @ Rc; nr = nr / s
        f = (W / 2) / math.tan(math.radians(fov_deg) / 2) if hfov else (H / 2) / math.tan(math.radians(fov_deg) / 2)
        c2w = np.eye(4); c2w[:3, :3] = Rc; c2w[:3, 3] = C
        cams.append({"name": name, "c2w": c2w.tolist(), "fx": f, "fy": f, "cx": W / 2, "cy": H / 2, "W": W, "H": H, "near": nr})
        Image.fromarray(rgba[..., :3]).save(os.path.join(args.out_dir, "editor", name + ".png"))     # premultiplied = over black
        Image.fromarray(rgba[..., 3]).save(os.path.join(args.out_dir, "editor", name + "_alpha.png"))

    with sync_playwright() as pw:
        browser, page, frame, info = open_editor(pw, args.supersplat_url, port)
        M = np.asarray(info["M"], dtype=np.float64).reshape(4, 4).T
        v = frame.evaluate(SS_VIEW)                                       # default view (also resets any override)
        c = frame.evaluate(SS_CAM)
        print("[haze] editor default camera:", json.dumps(c), flush=True)
        W, H = v["W"], v["H"]
        rgba = np.frombuffer(base64.b64decode(v["px"]), np.uint8).reshape(H, W, 4)
        Rw = Rotation.from_quat(c["q"]).as_matrix()
        add("default", c["p"], Rw, c["fov"], c["hfov"] or W > H, W, H, c["near"], M, rgba)
        frame.evaluate(SS_SETUP)
        # orbit around the focal point (what a user does next): same distance/elevation, azimuth steps, world up = +y
        P0 = np.asarray(c["p"]); F = np.asarray(c["focal"] or c["bc"]); up = np.array([0.0, 1.0, 0.0])
        bc, R_b = np.asarray(c["bc"]), float(np.linalg.norm(c["bh"]))
        for k in range(1, args.orbit):
            a = 2 * math.pi * k / args.orbit; Ry = Rotation.from_rotvec(up * a).as_matrix()
            Pk = F + Ry @ (P0 - F)
            z = (Pk - F) / np.linalg.norm(Pk - F); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
            Rk = np.stack([x, y, z], 1)
            fov = float(c["fov"])
            b64 = frame.evaluate(SS_SHOT, [Pk.tolist(), Rotation.from_matrix(Rk).as_quat().tolist(), fov, W, H])
            px = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(H, W, 4)
            d = float((bc - Pk) @ -z); far = max(d + R_b, 1e-2); near = min(1.0, max(d - R_b, far / (1024 * 16)))
            add(f"orbit{k:02d}", Pk, Rk, fov, c["hfov"] or W > H, W, H, near, M, px)
        browser.close()
    json.dump({"ply": args.ply, "sim3_json": args.sim3_json, "editor": c, "cams": cams},
              open(os.path.join(args.out_dir, "cams.json"), "w"), indent=1)
    print(f"[haze] {len(cams)} editor cameras -> {args.out_dir}/cams.json", flush=True)


# ---------------------------------------------------------------- rasteriser with per-pair weights (venv-gsplat)
def make_raster(display_scale, near=None):
    import torch
    from finetune_ply_viewer import ViewerRaster, E4

    class R(ViewerRaster):
        """ViewerRaster (ss352, exact) that keeps (gaussian, pixel, blend weight) of its last render; optional near cull."""
        near_clip = near

        def project(self, p, w2c, fx, fy, cx, cy, W, H):
            md, conics, op, z, radii, keep = super().project(p, w2c, fx, fy, cx, cy, W, H)
            if self.near_clip:
                keep = keep & (z > self.near_clip); radii = radii * keep[:, None].int()
            return md, conics, op, z, radii, keep

        def _exact(self, md, conics, op, feats, W, H, offsets, flatten_ids):
            from gsplat.cuda._wrapper import rasterize_to_indices_in_range
            dev = md.device
            op_idx = (op / (1 - E4)).clamp_max(1.0).detach()
            gid, pid, _ = rasterize_to_indices_in_range(0, 1 << 30, torch.ones(1, H, W, device=dev), md[None].detach().contiguous(),
                                                        conics[None].detach().contiguous(), op_idx[None].contiguous(), W, H,
                                                        self.tile, offsets, flatten_ids)
            pix = torch.stack([pid % W, pid // W], -1).float() + 0.5
            d = pix - md[gid]; cq = conics[gid]
            s2 = cq[:, 0] * d[:, 0] ** 2 + 2 * cq[:, 1] * d[:, 0] * d[:, 1] + cq[:, 2] * d[:, 1] ** 2
            al = op[gid] * (torch.exp(-0.5 * s2) - E4) / (1 - E4)
            al = torch.where((s2 <= 8) & (al > 0), al, torch.zeros_like(al)).clamp(0, 1 - 1e-6)
            lt = torch.log1p(-al.double()); cs = torch.cumsum(lt, 0) - lt
            first = torch.ones_like(pid, dtype=torch.bool); first[1:] = pid[1:] != pid[:-1]
            seg = torch.cumsum(first.long(), 0) - 1
            w = (al.double() * torch.exp(cs - cs[first][seg])).float()
            self.pairs = (gid, pid, w)
            out = torch.zeros(H * W, feats.shape[1], device=dev).index_add_(0, pid, w[:, None] * feats[gid])
            alpha = torch.zeros(H * W, device=dev).index_add_(0, pid, w)
            return out.view(H, W, -1), alpha.view(H, W)

    return R("ss352", "exact", display_scale=display_scale)


def load_cams(path):
    j = json.load(open(path))
    return [(c["name"], np.asarray(c["c2w"]), (c["fx"], c["fy"], c["cx"], c["cy"], c["W"], c["H"]), c.get("near")) for c in j["cams"]]


def ply_count(path):
    with open(path, "rb") as f:
        for ln in f:
            if ln.startswith(b"element vertex"):
                return int(ln.split()[-1])
            if ln.startswith(b"end_header"):
                break
    raise ValueError(path)


def sub(p, sel):
    return {k: v[sel] for k, v in p.items()}


# ---------------------------------------------------------------- diagnose
DEPTH = {}                                                           # image name -> LiDAR ray-distance map (.npy)


def train_stats(p, fr, deg, display_scale, dev, tol_abs=0.0, tol_rel=0.05):
    """Per gaussian over all training frames: max / summed blending weight, #views with weight > 1/255, #frustums;
    where the frame has a LiDAR ray-distance map (<stem>_depth.npy, build_depth_targets.py): #views with valid depth
    at the centre's pixel, #views in which the centre lies behind / in front of the LiDAR surface (beyond tolerance)."""
    import torch
    from finetune_ply_viewer import cam_args, to_t
    R = make_raster(display_scale); N = len(p["means"])
    mx = torch.zeros(N, device=dev); sm = torch.zeros(N, device=dev)
    nv = torch.zeros(N, dtype=torch.int32, device=dev); nf = torch.zeros(N, dtype=torch.int32, device=dev)
    ndv, nbh, nfr = (torch.zeros(N, dtype=torch.int32, device=dev) for _ in range(3))
    t0 = time.time()
    with torch.no_grad():
        for i, (name, cam, c2w) in enumerate(fr):
            w2c = to_t(np.linalg.inv(c2w), dev); fx, fy, cx, cy, W, H = cam_args(cam)
            R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)
            gid, _, w = R.pairs
            mx.scatter_reduce_(0, gid, w, "amax"); sm.index_add_(0, gid, w)
            v = torch.zeros(N, dtype=torch.bool, device=dev); v[gid[w > 1 / 255]] = True; nv += v.int()
            Pc = p["means"] @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2].clamp_min(1e-6)
            u = fx * Pc[:, 0] / z + cx; vv = fy * Pc[:, 1] / z + cy
            inf_ = (Pc[:, 2] > 1e-3) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H); nf += inf_.int()
            if name in DEPTH:
                D = torch.tensor(np.load(DEPTH[name]), dtype=torch.float32, device=dev)
                ui = u.clamp(0, D.shape[1] - 1).long(); vi = vv.clamp(0, D.shape[0] - 1).long()
                L = D[vi, ui]; r = Pc.norm(dim=1); ok = inf_ & (L > 0) & torch.isfinite(L)
                tol = tol_abs + tol_rel * L
                ndv += ok.int(); nbh += (ok & (r > L + tol)).int(); nfr += (ok & (r < L - tol)).int()
            if i % 100 == 0:
                print(f"[haze] train stats {i}/{len(fr)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"train_max_w": mx, "train_sum_w": sm, "train_nviews": nv, "train_nfrustum": nf}
    if DEPTH:
        out.update({"lidar_nviews": ndv, "lidar_behind": nbh, "lidar_front": nfr})
    return {k: v.cpu().numpy() for k, v in out.items()}


def lidar_dist(xyz_ply, cloud, sim3_json):
    """metric distance from each gaussian centre to the nearest LiDAR point (x_metric = s R x + t)."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from sim3_utils import load_sim3
    s, R, t = load_sim3(sim3_json)
    v = PlyData.read(cloud)["vertex"]; L = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    X = s * xyz_ply.astype(np.float64) @ R.T + t
    return cKDTree(L).query(X, k=1, workers=-1)[0].astype(np.float32), s


def load_frames(datasets, every=1):
    """training frames of one or more COLMAP text datasets (same frame), name-deduplicated, name order."""
    from finetune_ply_viewer import frames
    out = {}
    for d in datasets:
        for f in frames(d, 0, "all"):
            out.setdefault(f[0], f)
            dp = os.path.join(d, "images", os.path.splitext(f[0])[0] + "_depth.npy")
            if f[0] not in DEPTH and os.path.exists(dp):
                DEPTH[f[0]] = dp
    return [out[k] for k in sorted(out)][::every]


def get_stats(args, p, fr, deg, dev, sp):
    """per-gaussian stats (cached in sp): training support, frustum count, camera / LiDAR distance (metric if LiDAR)."""
    if os.path.exists(sp):
        st = dict(np.load(sp))
        if len(st["train_max_w"]) == len(p["means"]):
            return st
        print(f"[haze] {sp} is for another splat: recomputing", flush=True)
    xyz = p["means"].cpu().numpy()
    tol = 0.0
    if args.sim3_json:                                               # 5 cm (metric) + 5 % depth tolerance, data units
        from sim3_utils import load_sim3
        tol = 0.05 / load_sim3(args.sim3_json)[0]
    st = train_stats(p, fr, deg, args.display_scale, dev, tol_abs=tol)
    C = np.stack([c2w[:3, 3] for _, _, c2w in fr])
    from scipy.spatial import cKDTree
    st["cam_dist"] = cKDTree(C).query(xyz, k=1, workers=-1)[0].astype(np.float32)
    if args.cloud and os.path.exists(args.cloud) and args.sim3_json:
        st["lidar_dist_m"], s = lidar_dist(xyz, args.cloud, args.sim3_json); st["cam_dist"] *= s   # metric
        st["metric_scale"] = np.float32(s)
    else:
        print("[haze] no LiDAR cloud / sim3: freespace class = unsupported gaussians anywhere", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(sp)), exist_ok=True); np.savez_compressed(sp, **st)
    return st


def diagnose(args):
    import torch
    from finetune_ply_viewer import load_ply, sh_deg, to_t, save_png
    dev = "cuda"; os.makedirs(args.out_dir, exist_ok=True)
    _, p = load_ply(args.ply, dev); N = len(p["means"]); deg = sh_deg(p)
    fr = load_frames(args.dataset, args.every)
    st = get_stats(args, p, fr, deg, dev, args.stats or os.path.join(args.out_dir, "stats.npz"))
    s = float(st.get("metric_scale", 1.0))
    smax = (torch.exp(p["scales"]).max(1).values * s).cpu().numpy(); op = torch.sigmoid(p["opacities"]).cpu().numpy()
    shm = p["shN"].norm(dim=(1, 2)).cpu().numpy()
    nb = ply_count(args.base_ply) if args.base_ply else N
    mw, ld, cd = st["train_max_w"], st.get("lidar_dist_m"), st["cam_dist"]
    classes = {"untrained": mw < 1 / 255, "weak": mw < 0.02,
               "nofrustum": st["train_nfrustum"] == 0, "fill": np.arange(N) >= nb}
    if ld is not None:
        for d in (0.05, 0.10, 0.20):
            classes[f"lidar_gt{int(d * 100)}cm"] = ld > d
            classes[f"untrained_lidar_gt{int(d * 100)}cm"] = (mw < 1 / 255) & (ld > d)
            classes[f"weak_lidar_gt{int(d * 100)}cm"] = (mw < 0.02) & (ld > d)
    classes["nearcam_lt30cm"] = cd < 0.3
    sc = np.sort(np.exp(p["scales"].cpu().numpy()), 1) * s
    classes["needle"] = (sc[:, 2] > 8 * sc[:, 1]) & (sc[:, 2] > 0.03)
    classes["nofrustum_or_big"] = (st["train_nfrustum"] == 0) | (smax > 0.2)
    classes["big_gt20cm"] = smax > 0.2
    classes["freespace"] = (mw < 1 / 255) & ((ld if ld is not None else np.inf) > 0.10)
    classes["giant"] = (smax > 1.0) & ((ld if ld is not None else np.inf) > 1.0)
    dcn = np.linalg.norm(p["sh0"][:, 0].cpu().numpy(), axis=1)
    classes["shhigh_weak"] = (shm > dcn + 0.3) & (mw < 0.1)
    classes["lowsum_offsurf"] = (st["train_sum_w"] < 5) & ((ld if ld is not None else np.inf) > 0.05)
    if "lidar_nviews" in st:
        ndv, nbh = st["lidar_nviews"], st["lidar_behind"]
        classes["occluded"] = (ndv >= 3) & (nbh >= 0.9 * ndv)                # behind the LiDAR surface from (almost) every view
        classes["occluded_weak"] = classes["occluded"] & (mw < 0.1)
        classes["front_weak"] = (st["lidar_front"] >= 0.9 * np.maximum(ndv, 1)) & (ndv >= 3) & (mw < 0.1)
    pre = np.zeros(N, bool)
    for c in filter(None, args.pre_remove.split(",")):
        pre |= classes[c]
    if pre.any():                                                     # analyse what remains after these removals
        print(f"[haze] pre-removed {int(pre.sum()):,} ({args.pre_remove})", flush=True)
        keep_i = np.nonzero(~pre)[0]; kt = torch.tensor(keep_i, device=dev)
        p = sub(p, kt); N = len(keep_i)
        classes = {k: m[keep_i] for k, m in classes.items() if k not in args.pre_remove.split(",")}
        smax, op, shm, mw, cd = smax[keep_i], op[keep_i], shm[keep_i], mw[keep_i], cd[keep_i]
        ld = ld[keep_i] if ld is not None else None
        st = {k: (v[keep_i] if np.ndim(v) else v) for k, v in st.items()}
        nb = int((keep_i < nb).sum())
    rep = {"n": N, "n_base": nb, "classes": {k: int(m.sum()) for k, m in classes.items()}, "views": {}}
    print(f"[haze] {N:,} gaussians ({N - nb:,} appended fill); class sizes:", rep["classes"], flush=True)
    for nm, c2w, (fx, fy, cx, cy, W, H), near in load_cams(args.cams):
        R = make_raster(1.0, near if args.near_cull else None); w2c = to_t(np.linalg.inv(c2w), dev)
        with torch.no_grad():
            img, alpha, _ = R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)
            gid, pid, w = R.pairs
            ew = torch.zeros(N, device=dev).index_add_(0, gid, w).cpu().numpy()
            for bx in args.box:                                       # "<view>:x0,y0,x1,y1" (pixels): attribute a region
                bv, bc = bx.split(":")
                if bv != nm:
                    continue
                x0, y0, x1, y1 = (int(t) for t in bc.split(","))
                inb = ((pid % W) >= x0) & ((pid % W) < x1) & ((pid // W) >= y0) & ((pid // W) < y1)
                bw = torch.zeros(N, device=dev).index_add_(0, gid[inb], w[inb]).cpu().numpy()
                o = np.argsort(-bw); cum = np.cumsum(bw[o]) / max(bw.sum(), 1e-9); k = int(np.searchsorted(cum, 0.8)) + 1
                sel_b = o[:k]
                def pr(a):
                    return [round(float(x), 4) for x in np.percentile(a[sel_b], [10, 50, 90])]
                prof = {"n_80pct": k, "max_w": pr(mw), "sum_w": pr(st["train_sum_w"]), "nviews": pr(st["train_nviews"]),
                        "scale_m": pr(smax), "opacity": pr(op), "sh_rest": pr(shm), "lidar_m": pr(ld) if ld is not None else None,
                        "cam_m": pr(cd)}
                if "lidar_nviews" in st:
                    prof.update({"lidar_nviews": pr(st["lidar_nviews"]), "lidar_behind": pr(st["lidar_behind"]), "lidar_front": pr(st["lidar_front"])})
                print(f"[haze] box {nm} {bc}: " + json.dumps(prof), flush=True)
                kb = torch.zeros(N, dtype=torch.bool, device=dev); kb[torch.tensor(sel_b, device=dev)] = True
                save_png(os.path.join(args.out_dir, nm, "only_box.png"), R.render(sub(p, kb), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
                save_png(os.path.join(args.out_dir, nm, "without_box.png"), R.render(sub(p, ~kb), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
                save_png(os.path.join(args.out_dir, nm, "box_sh0.png"), R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=0)[0])
                np.save(os.path.join(args.out_dir, nm, "box_ids.npy"), sel_b)
            save_png(os.path.join(args.out_dir, nm, "full.png"), img)
            if args.renders:
                save_png(os.path.join(args.out_dir, nm, "sh0.png"), R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=0)[0])
            ref = os.path.join(os.path.dirname(args.cams), "editor", nm + ".png")
            row = {"alpha_hole": float((alpha < 0.98).float().mean())}
            if os.path.exists(ref):
                from PIL import Image
                g = torch.tensor(np.asarray(Image.open(ref).convert("RGB")), device=dev).float() / 255
                row["psnr_vs_editor"] = float(-10 * torch.log10(((img - g) ** 2).mean()))
            tot = ew.sum(); row["share"] = {}
            for k, m in classes.items():
                row["share"][k] = float(ew[m].sum() / tot)
                if args.renders and k in args.renders.split(","):
                    keep = torch.tensor(~m, device=dev)
                    save_png(os.path.join(args.out_dir, nm, f"without_{k}.png"),
                             R.render(sub(p, keep), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
                    save_png(os.path.join(args.out_dir, nm, f"only_{k}.png"),
                             R.render(sub(p, ~keep), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
            # who carries this view: the gaussians holding the top 50% of its weight among the weakly-trained ones
            depth = (p["means"] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2].cpu().numpy()
            sel = ew > 0
            def q(a, m):
                return [float(x) for x in np.percentile(a[m], [5, 50, 95])] if m.any() else None
            weak = sel & (mw < 0.02)
            row["weak_profile"] = {"n": int(weak.sum()), "weight": float(ew[weak].sum() / tot),
                                   "depth": q(depth, weak), "scale_m": q(smax, weak), "opacity": q(op, weak),
                                   "sh_rest_norm": q(shm, weak), "cam_dist_m": q(cd, weak),
                                   "lidar_dist_m": q(ld, weak) if ld is not None else None}
            row["trained_profile"] = {"n": int((sel & ~(mw < 0.02)).sum()), "depth": q(depth, sel & (mw >= 0.02)),
                                      "scale_m": q(smax, sel & (mw >= 0.02)), "lidar_dist_m": q(ld, sel & (mw >= 0.02)) if ld is not None else None}
            # the gaussians carrying most of this view (haze patches are few, large ones): stats of the top 25
            top = np.argsort(-ew)[:25]
            row["top"] = [{"i": int(i), "px": float(ew[i]), "scale_m": [float(x) for x in np.sort(np.exp(p["scales"][i].cpu().numpy()))[::-1] * s],
                           "op": float(op[i]), "depth": float(depth[i]), "max_w": float(mw[i]), "nfr": int(st["train_nfrustum"][i]),
                           "lidar_m": float(ld[i]) if ld is not None else None, "cam_m": float(cd[i]), "fill": bool(i >= nb)} for i in top]
            if args.renders:
                k = torch.zeros(N, dtype=torch.bool, device=dev); k[torch.tensor(top[:25], device=dev)] = True
                save_png(os.path.join(args.out_dir, nm, "only_top25.png"), R.render(sub(p, k), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
                save_png(os.path.join(args.out_dir, nm, "without_top25.png"), R.render(sub(p, ~k), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0])
        rep["views"][nm] = row
        print(f"[haze] {nm}: " + json.dumps({k: v for k, v in row.items() if k != "top"}), flush=True)
    pop = {"scale_m": [float(x) for x in np.percentile(smax, [5, 50, 95])],
           "train_max_w": [float(x) for x in np.percentile(mw, [5, 25, 50, 75, 95])]}
    if ld is not None:
        pop["lidar_dist_m"] = [float(x) for x in np.percentile(ld, [5, 50, 95])]
    rep["population"] = pop
    json.dump(rep, open(os.path.join(args.out_dir, "diagnosis.json"), "w"), indent=1)
    print("[haze] population:", json.dumps(pop), flush=True)


# ---------------------------------------------------------------- clean + measured harm restore
def clean(args):
    import torch
    from plyfile import PlyData, PlyElement
    from finetune_ply_viewer import load_ply, sh_deg, to_t, cam_args
    dev = "cuda"
    vert, p = load_ply(args.ply, dev); N = len(p["means"]); deg = sh_deg(p)
    fr = load_frames(args.dataset, args.every)
    st = get_stats(args, p, fr, deg, dev, args.stats or os.path.splitext(args.out)[0] + "_stats.npz")
    ld = st.get("lidar_dist_m", np.full(N, np.inf, np.float32)); ms = float(st.get("metric_scale", 1.0))
    smax = np.exp(p["scales"].cpu().numpy()).max(1) * ms
    cls = {"freespace": (st["train_max_w"] < args.max_train_w) & (ld > args.min_lidar_dist),   # unconstrained, off every surface
           "giant": (smax > args.giant_scale) & (ld > args.giant_lidar)}                         # huge background splats
    cand = np.zeros(N, bool)
    for c in args.classes.split(","):
        cand |= cls[c]
    if args.protect_from is not None:
        cand &= np.arange(N) < args.protect_from
    exempt = np.zeros(N, bool)                                       # removed without restore (relaxed, informational)
    for c in filter(None, args.no_restore.split(",")):
        exempt |= cls[c] & cand
    rm = torch.tensor(cand, device=dev); ex = torch.tensor(exempt, device=dev)
    print(f"[haze] candidates: {int(rm.sum()):,} of {N:,} (" + ", ".join(f"{c} {int((cls[c] & cand).sum()):,}" for c in cls)
          + f"; no-restore {int(ex.sum()):,})", flush=True)
    R = make_raster(args.display_scale)
    with torch.no_grad():
        base = {name: R.render(p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0].half() for name, cam, c2w in fr}
    hist = []
    for it in range(args.iters):
        pc = sub(p, ~rm)
        restore = torch.zeros(N, dtype=torch.bool, device=dev); changed = 0; total = 0; worst = 0.0
        with torch.no_grad():
            for name, cam, c2w in fr:
                w2c = to_t(np.linalg.inv(c2w), dev); fx, fy, cx, cy, W, H = cam_args(cam)
                diff = (R.render(pc, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0] - base[name].float()).abs().amax(-1)
                d = diff > args.thr; changed += int(d.sum()); total += d.numel(); worst = max(worst, float(diff.max()))
                if not d.any():
                    continue
                R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)        # exact: removed gaussians blending into changed px
                gid, pid, w = R.pairs
                hit = d.view(-1)[pid] & rm[gid] & ~ex[gid] & (w > 0)
                restore[gid[hit]] = True
                # changed pixels no removed gaussian blends into (depth-sort ties, kernel cut-offs): restore every removed
                # gaussian whose 3-sigma footprint covers one (scripts/fill_harm_ply.py's test)
                u = d.view(-1).clone(); u[pid[hit]] = False; u = u.view(H, W)
                if u.any():
                    cand_i = torch.nonzero(rm & ~ex)[:, 0]; X = p["means"][cand_i]
                    ii = torch.nn.functional.pad(u.float().cumsum(0).cumsum(1), (1, 0, 1, 0))
                    Pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2].clamp_min(1e-3)
                    uu = fx * Pc[:, 0] / z + cx; vv = fy * Pc[:, 1] / z + cy
                    r = (3 * torch.exp(p["scales"][cand_i]).max(1).values * fx / z).clamp(1, 4096)
                    u0 = (uu - r).floor().clamp(0, W).long(); u1 = (uu + r).ceil().clamp(0, W).long()
                    v0 = (vv - r).floor().clamp(0, H).long(); v1 = (vv + r).ceil().clamp(0, H).long()
                    cnt = ii[v1, u1] - ii[v0, u1] - ii[v1, u0] + ii[v0, u0]
                    restore[cand_i[(Pc[:, 2] > 1e-3) & (u1 > u0) & (v1 > v0) & (cnt > 0)]] = True
        hist.append({"iter": it, "changed_pixel_share": changed / max(total, 1), "max_abs_diff": worst,
                     "removed": int(rm.sum()), "restored": int(restore.sum())})
        print(f"[haze] iter {it}: changed px {changed / max(total, 1):.2e} (max {worst * 255:.1f}/255), "
              f"removed {int(rm.sum()):,}, restore {int(restore.sum()):,}", flush=True)
        if not restore.any():
            break
        rm &= ~restore
    keep = (~rm).cpu().numpy()
    kept_by_class = {c: {"candidates": int((m & cand).sum()), "removed": int((m & cand & ~keep).sum())} for c, m in cls.items()}
    print("[haze] per class:", kept_by_class, flush=True)
    PlyData([PlyElement.describe(vert.data[keep], "vertex")], text=False).write(args.out)
    json.dump({"in": N, "candidates": int(cand.sum()), "removed": int((~keep).sum()), "iterations": hist, "classes": kept_by_class,
               "args": {k: v for k, v in vars(args).items() if k != "cmd"},
               "max_train_w": args.max_train_w, "min_lidar_dist": args.min_lidar_dist, "thr": args.thr,
               "display_scale": args.display_scale}, open(os.path.splitext(args.out)[0] + "_haze.json", "w"), indent=2)
    print(f"[haze] removed {int((~keep).sum()):,} of {N:,} -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(); s = ap.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("probe"); a.add_argument("--ply", required=True); a.add_argument("--out-dir", required=True)
    a.add_argument("--sim3-json", default=None, help="x_ply = s R x + t: write cameras in the x frame")
    a.add_argument("--orbit", type=int, default=8); a.add_argument("--supersplat-url", default="https://superspl.at/editor")
    a = s.add_parser("diagnose"); a.add_argument("--ply", required=True); a.add_argument("--dataset", required=True, nargs="+")
    a.add_argument("--cams", required=True); a.add_argument("--out-dir", required=True)
    a.add_argument("--cloud", default=None); a.add_argument("--sim3-json", default=None)
    a.add_argument("--base-ply", default=None, help="splat before the appended fill (to tag fill gaussians)")
    a.add_argument("--display-scale", type=float, default=1.8); a.add_argument("--every", type=int, default=1)
    a.add_argument("--near-cull", action="store_true"); a.add_argument("--renders", default="")
    a.add_argument("--stats", default=None, help="per-gaussian stats cache (default <out-dir>/stats.npz)")
    a.add_argument("--box", nargs="*", default=[], help="<view>:x0,y0,x1,y1 regions to attribute (e.g. a haze blob)")
    a.add_argument("--pre-remove", default="", help="classes removed before the view analysis (e.g. freespace,giant)")
    a = s.add_parser("clean"); a.add_argument("--ply", required=True); a.add_argument("--out", required=True)
    a.add_argument("--dataset", required=True, nargs="+", help="training frames (several = union by image name)")
    a.add_argument("--stats", default=None, help="stats cache (diagnose's, or computed here: default <out>_stats.npz)")
    a.add_argument("--cloud", default=None, help="static LiDAR cloud (metric); none -> no LiDAR test")
    a.add_argument("--sim3-json", default=None, help="x_metric = s R x_ply + t")
    a.add_argument("--max-train-w", type=float, default=1 / 255)
    a.add_argument("--min-lidar-dist", type=float, default=0.10, help="metric m; 0 = ignore LiDAR")
    a.add_argument("--classes", default="freespace,giant", help="culprit classes to remove: freespace, giant")
    a.add_argument("--giant-scale", type=float, default=1.0, help="giant: largest axis > this (metric m)")
    a.add_argument("--giant-lidar", type=float, default=1.0, help="giant: centre > this from every LiDAR point (m)")
    a.add_argument("--no-restore", default="", help="classes removed WITHOUT the harm restore (informational runs only)")
    a.add_argument("--protect-from", type=int, default=None, help="never remove gaussians with index >= this")
    a.add_argument("--thr", type=float, default=2.0 / 255); a.add_argument("--iters", type=int, default=6)
    a.add_argument("--display-scale", type=float, default=1.8); a.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    {"probe": probe, "diagnose": diagnose, "clean": clean}[args.cmd](args)


if __name__ == "__main__":
    main()
