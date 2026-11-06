"""Train the object zones on mv_gen.py's generated views (gsplat venv, GPU; cosmetic: invented backs inside the zones).

Rows: the input PLY + mv_gen.py's hull seeds (unseen hull surface: a flat disc per 1 cm voxel, colour from the
best-facing generated view; seeds within --seed-skip-m of a well-seen input gaussian are dropped). Zone = input rows in
an object's box (mv_gen register box + --zone-m). FREE rows = seeds + zone rows no training view constrains (support max
blend weight < --free-w); every other zone row is "seen" (photometric loss only, lr x --lr-seen); outside the zones
everything is frozen (byte-identical).

  per step: viewer-exact L1+SSIM on a training frame (all trainable rows)
          + on a random generated view, the object rows only (zone rows above the floor + seeds), gradients to FREE rows:
            colour L1+SSIM vs the generated image (per-view gain/bias calibrated to the input's seen pixels) inside its
            mask, --lam-in (1 - alpha) inside, --lam-out alpha outside the (dilated) mask, --lam-normal (1 - cos) vs the
            Marigold normal map (rendered normal = blend of each gaussian's thinnest axis, facing the camera)
          + --lam-needle, --lam-opa (opacity 0 or 1) on free rows, scale cap

    python scripts/object_priors/mv_refit.py --ply in.ply --support D/support.npz --gen G --dataset <train> <test> \
        --sim3-json sim3.json --objects D/objects --out T/refit.ply
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import sim3, all_frames  # noqa: E402

C0 = 0.28209479177387814


def quat_from_z(N):
    """unit quats wxyz rotating +z onto N (N,3)."""
    z = np.array([0, 0, 1.0]); v = np.cross(np.broadcast_to(z, N.shape), N); c = N @ z
    q = np.concatenate([(1 + c)[:, None], v], 1); bad = c < -0.9999
    q[bad] = [0, 1.0, 0, 0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def feat_render(Rv, p, w2c, fx, fy, cx, cy, W, H, feats):
    """ViewerRaster.render with arbitrary per-gaussian features instead of SH colour -> (H,W,F), alpha."""
    import torch
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
    md, conics, op, z, radii, keep = Rv.project(p, w2c, fx, fy, cx, cy, W, H)
    z_sort = torch.where(keep, z, torch.full_like(z, 1e10)).detach()
    tw, th = (W + Rv.tile - 1) // Rv.tile, (H + Rv.tile - 1) // Rv.tile
    _, isect_ids, flatten_ids = isect_tiles(md[None].detach().contiguous(), radii[None].contiguous(), z_sort[None].contiguous(), Rv.tile, tw, th)
    offsets = isect_offset_encode(isect_ids, 1, tw, th)
    return Rv._exact(md, conics, op, feats, W, H, offsets, flatten_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--support", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--objects", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--display-scale", default="1.5:4")
    ap.add_argument("--zone-m", type=float, default=0.06); ap.add_argument("--floor-m", type=float, default=0.015)
    ap.add_argument("--free-w", type=float, default=0.2); ap.add_argument("--lr-seen", type=float, default=0.1)
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--seed-skip-m", type=float, default=0.012)
    ap.add_argument("--seed-cell-m", type=float, default=0.01); ap.add_argument("--seed-opacity", type=float, default=0.8)
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--lam-photo", type=float, default=1.0); ap.add_argument("--lam-gen", type=float, default=1.0)
    ap.add_argument("--lam-in", type=float, default=0.5); ap.add_argument("--lam-out", type=float, default=0.5)
    ap.add_argument("--lam-normal", type=float, default=0.0); ap.add_argument("--out-dilate", type=int, default=4)
    ap.add_argument("--lam-needle", type=float, default=0.3); ap.add_argument("--needle-ratio", type=float, default=3.0)
    ap.add_argument("--lam-opa", type=float, default=0.1); ap.add_argument("--max-scale-m", type=float, default=0.03)
    ap.add_argument("--gen-scale", type=float, default=2.0, help="generated views rendered at this x their 320 px")
    ap.add_argument("--low-weight", type=float, default=0.5, help="weight of the -10 deg (below the object centre) views")
    ap.add_argument("--freeze-seen", action="store_true", help="every input row outside FREE frozen (seen faces byte-identical)")
    ap.add_argument("--log-every", type=int, default=250)
    a = ap.parse_args()
    import torch, cv2
    from PIL import Image
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t, quat_to_rot
    from refine_poses_gsplat import load_rgb, ssim, psnr
    from scipy.spatial import cKDTree
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"
    s, R, t = sim3(a.sim3_json)
    vtx, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); N0 = len(p0["means"])
    wmax = np.load(a.support)["wmax"]; assert len(wmax) == N0, "support was computed for another PLY"
    reg = json.load(open(os.path.join(a.gen, "register.json")))
    X = p0["means"].cpu().numpy().astype(np.float64); Xm = s * (X @ R.T) + t
    op0 = torch.sigmoid(p0["opacities"]).cpu().numpy()
    zone = np.zeros(N0, bool); objr = np.zeros(N0, bool); oid_row = np.full(N0, -1)
    for r in reg:
        c0, A = np.array(r["c0"]), np.array(r["axes"]); n = A[:, 2]
        L = np.stack([(Xm - c0) @ A[:, 0], (Xm - c0) @ A[:, 1], (Xm - c0) @ n], 1)
        lo, hi = np.array(r["box_lo"]) - a.zone_m, np.array(r["box_hi"]) + a.zone_m
        iz = (L[:, 0] >= lo[0]) & (L[:, 0] <= hi[0]) & (L[:, 1] >= lo[1]) & (L[:, 1] <= hi[1]) & (L[:, 2] > -0.02) & (L[:, 2] <= hi[2])
        zone |= iz; objr |= iz & (L[:, 2] > a.floor_m); oid_row[iz] = r["id"]
    seen_face = zone & (wmax >= a.face_w) & (op0 > 0.5)
    # seeds -> new rows
    sd = np.load(os.path.join(a.gen, "seeds.npz")); Xs, Ns, Cs = sd["X"], sd["N"], sd["C"]
    if a.no_seeds:
        Xs = Xs[:0]
    if len(Xs) and seen_face.any():
        dd = cKDTree(X[seen_face]).query(Xs)[0] * s; k = dd > a.seed_skip_m
        Xs, Ns, Cs = Xs[k], Ns[k], Cs[k]
    M = len(Xs); print(f"[mvrefit] {N0:,} input rows, zone {zone.sum():,} (object rows {objr.sum():,}), {M:,} seeds", flush=True)
    K = p0["shN"].shape[1]
    sc_in = np.log(0.6 * a.seed_cell_m / s); sc_n = np.log(0.15 * a.seed_cell_m / s)
    newp = {"means": Xs, "scales": np.tile([sc_in, sc_in, sc_n], (M, 1)), "quats": quat_from_z(Ns / np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-9)),
            "opacities": np.full(M, np.log(a.seed_opacity / (1 - a.seed_opacity))), "sh0": ((np.clip(Cs, 0, 1) - 0.5) / C0)[:, None],
            "shN": np.zeros((M, K, 3))}
    P0 = {k: torch.cat([p0[k], torch.tensor(newp[k], dtype=torch.float32, device=dev)]) for k in p0}
    vtx_all = np.concatenate([vtx.data, np.zeros(M, vtx.data.dtype)]) if M else vtx.data
    N = N0 + M
    new = np.r_[np.zeros(N0, bool), np.ones(M, bool)]
    free = new.copy(); free[:N0] = zone & (wmax < a.free_w)
    tr_rows = free.copy() if a.freeze_seen else np.r_[zone, np.ones(M, bool)]
    obj_rows = np.r_[objr, np.ones(M, bool)]
    T = lambda m: torch.tensor(m, device=dev)
    tm, fr_ = T(tr_rows), T(free); seen_rows = tm & ~fr_; oi = torch.nonzero(T(obj_rows))[:, 0]
    print(f"[mvrefit] trainable {int(tm.sum()):,} (free {int(fr_.sum()):,}, seen {int(seen_rows.sum()):,}); generated-view render rows {len(oi):,}", flush=True)
    # generated views
    Rv = ViewerRaster("ss352", "exact", display_scale=1.5)
    gv = []
    V = os.path.join(a.gen, "views")
    sub = lambda p, idx: {k: v[idx] for k, v in p.items()}
    with torch.no_grad():
        for name, cam, c2w in frames(V, 0, "all"):
            fx, fy, cx, cy, W, H = cam_args(cam); g_ = a.gen_scale
            fx, fy, cx, cy, W, H = fx * g_, fy * g_, cx * g_, cy * g_, int(W * g_), int(H * g_)
            im = np.asarray(Image.open(os.path.join(V, "images", name)).convert("RGB").resize((W, H), Image.Resampling.BICUBIC), np.float32) / 255
            mk = np.asarray(Image.open(os.path.join(V, "masks", name.replace(".png", "_mask.png"))).resize((W, H), Image.Resampling.NEAREST)) > 127
            nf = os.path.join(V, "normals", name.replace(".png", "_normal.npy"))
            nm = cv2.resize(np.load(nf).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) if os.path.exists(nf) else None
            w2c = to_t(np.linalg.inv(c2w), dev)
            # calibration: gain/bias per channel from the input's well-seen object rows rendered here
            si = torch.nonzero(T(obj_rows[:N0] & ~free[:N0]))[:, 0]
            img, al, _ = Rv.render(sub(P0, si), w2c, fx, fy, cx, cy, W, H, sh_degree=deg)
            ok = (al.cpu().numpy() > 0.9) & cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            gb = None
            if ok.sum() > 300:
                ren = img.cpu().numpy()[ok]; gen = im[ok]; gb = []
                for c in range(3):
                    A_ = np.c_[gen[:, c], np.ones(len(gen))]; g1, b1 = np.linalg.lstsq(A_, ren[:, c], rcond=None)[0]
                    g1 = float(np.clip(g1, 0.6, 1.6)); gb.append((g1, float(np.median(ren[:, c] - g1 * gen[:, c]))))
            outm = ~(cv2.dilate(mk.astype(np.uint8), np.ones((2 * a.out_dilate + 1,) * 2, np.uint8)).astype(bool))
            low = "_v1" in name or "_v3" in name or "_v5" in name
            gv.append({"name": name, "w2c": w2c, "cam": (fx, fy, cx, cy, W, H), "img": im, "mask": mk, "out": outm, "n": nm, "gb": gb,
                       "w": a.low_weight if low else 1.0, "calib_px": int(ok.sum())})
    gbs = [g["gb"] for g in gv if g["gb"] is not None]
    gmed = [(float(np.median([x[c][0] for x in gbs])), float(np.median([x[c][1] for x in gbs]))) for c in range(3)] if gbs else [(1.0, 0.0)] * 3
    for g in gv:
        gb = g["gb"] or gmed; im = g["img"].copy()
        for c in range(3):
            im[..., c] = np.clip(gb[c][0] * im[..., c] + gb[c][1], 0, 1)
        g["img_t"] = torch.tensor(im, device=dev); g["mask_t"] = T(g["mask"]); g["out_t"] = T(g["out"])
        g["n_t"] = torch.tensor(g["n"], device=dev) if g["n"] is not None else None
        g["bg"] = torch.tensor(np.median(g["img"][g["out"]], 0) if g["out"].any() else [0.5] * 3, device=dev, dtype=torch.float32)
    print(f"[mvrefit] {len(gv)} generated views; calibrated from own pixels: {sum(g['gb'] is not None for g in gv)}", flush=True)
    tr = all_frames(a.dataset)
    gtf = lambda n: next(os.path.join(ds, "images", n) for ds in a.dataset if os.path.exists(os.path.join(ds, "images", n)))
    rgb = [load_rgb(gtf(n), dev) for n, _, _ in tr]
    centres = np.stack([c[:3, 3] for _, _, c in tr]); scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in P0.items()})
    lr = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in a.display_scale.split(":")]
    fi = torch.nonzero(fr_)[:, 0]; ni = torch.nonzero(T(new))[:, 0]; log3 = float(np.log(a.needle_ratio)); keys = list(params.keys())
    order, gorder, log, t0 = [], [], [], time.time()
    for step in range(1, a.iters + 1):
        if not order:
            order = list(np.random.permutation(len(tr)))
        if not gorder:
            gorder = list(np.random.permutation(len(gv)))
        k = int(order.pop()); name, cam, c2w = tr[k]
        Rv.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[-1]))))
        wu = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu * (0.1 ** (step / a.iters) if g["name"] == "means" else 1.0)
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = rgb[k].float() / 255
        lph = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        g1 = torch.autograd.grad(a.lam_photo * lph, [params[x] for x in keys], allow_unused=True)
        # generated view: object rows only
        G_ = gv[int(gorder.pop())]; Rv.r = 1.5
        ps = sub(params, oi)
        io, ao, _ = Rv.render(ps, G_["w2c"], *G_["cam"], sh_degree=0)
        comp = io + (1 - ao)[..., None] * G_["bg"]
        tgt = torch.where(G_["mask_t"][..., None], G_["img_t"], G_["bg"].expand_as(G_["img_t"]))
        mk = G_["mask_t"]
        lc = 0.8 * (comp - tgt).abs()[mk].mean() + 0.2 * (1 - ssim(comp.permute(2, 0, 1)[None], tgt.permute(2, 0, 1)[None]))
        lin = (1 - ao.clamp(0, 1))[mk].mean(); lout = ao[G_["out_t"]].mean() if G_["out_t"].any() else torch.zeros((), device=dev)
        lg = lc + a.lam_in * lin + a.lam_out * lout
        ln_ = torch.zeros((), device=dev)
        if a.lam_normal > 0 and G_["n_t"] is not None:
            w2c = G_["w2c"]; Rm = quat_to_rot(torch.nn.functional.normalize(ps["quats"], dim=-1))
            ax = torch.argmin(ps["scales"], 1)
            nw = torch.gather(Rm, 2, ax[:, None, None].expand(-1, 3, 1))[..., 0]            # thinnest axis, world
            nc = nw @ w2c[:3, :3].T; pc = ps["means"] @ w2c[:3, :3].T + w2c[:3, 3]
            nc = torch.where(((nc * pc).sum(-1, keepdim=True) > 0), -nc, nc)                # facing the camera
            fn, an = feat_render(Rv, ps, w2c, *G_["cam"], nc)
            fn = torch.nn.functional.normalize(fn, dim=-1); vm = mk & (an.detach() > 0.5)
            if vm.any():
                ln_ = (1 - (fn * G_["n_t"]).sum(-1))[vm].mean()
        lg = G_["w"] * a.lam_gen * (lg + a.lam_normal * ln_)
        ssf = torch.sort(params["scales"][fi], 1).values
        lneed = torch.relu(ssf[:, 2] - ssf[:, 1] - log3).mean()
        of = torch.sigmoid(params["opacities"][fi]); lopa = (of * (1 - of)).mean()
        g2 = torch.autograd.grad(lg + a.lam_needle * lneed + a.lam_opa * lopa, [params[x] for x in keys], allow_unused=True)
        opt.zero_grad(set_to_none=True)
        for x, a1, a2 in zip(keys, g1, g2):
            g = torch.zeros_like(params[x]) if a1 is None else a1.clone()
            if a2 is not None:
                g[fr_] += a2[fr_]
            g[~torch.isfinite(g)] = 0; g[~tm] = 0
            params[x].grad = g
        old = {x: params[x].detach()[seen_rows].clone() for x in keys}
        opt.step()
        with torch.no_grad():
            for x in keys:
                params[x][seen_rows] = old[x] + a.lr_seen * (params[x][seen_rows] - old[x])
            params["scales"][ni] = params["scales"][ni].clamp_max(float(np.log(a.max_scale_m / s)))
            params["scales"][fi] = params["scales"][fi].clamp_max(float(np.log(a.max_scale_m * 3 / s)))
        if step % a.log_every == 0 or step == 1:
            e = {"step": step, "psnr": round(psnr(img.detach(), gt), 2), "gen_l1": round(float(lc), 4), "in": round(float(lin), 4),
                 "out": round(float(lout), 4), "normal": round(float(ln_), 4), "needle": round(float(lneed), 4), "opa": round(float(lopa), 4)}
            log.append(e); print(f"[mvrefit] {e} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        Rv.r = rs[0]; subf = list(enumerate(tr))[::10]
        pv = [psnr(Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n_, cam, c2w) in subf]
        p0v = [psnr(Rv.render(p0, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n_, cam, c2w) in subf]
        for x in keys:
            params[x][~tm] = P0[x][~tm]
        # generated-view previews: generated | before (input object rows) | after
        prev = []
        for G_ in gv:
            row = []
            for pp, idx in ((P0, torch.nonzero(T(np.r_[objr, np.zeros(M, bool)]))[:, 0]), (params, oi)):
                io, ao, _ = Rv.render(sub(pp, idx), G_["w2c"], *G_["cam"], sh_degree=0)
                row.append(((io + (1 - ao)[..., None] * G_["bg"]).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            prev.append(np.hstack([(G_["img"] * 255).astype(np.uint8)] + row))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        for j in range(0, len(prev), 6):
            Image.fromarray(np.vstack(prev[j:j + 6])).resize((3 * 320, 320 * len(prev[j:j + 6]))).save(
                os.path.splitext(a.out)[0] + f"_genviews_{j // 6}.jpg", quality=88)
    save_ply(a.out, vtx_all, params)
    src = np.r_[np.arange(N0), -np.ones(M, int)]
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=src)
    r = {"rows": N, "seeds": M, "trainable": int(tm.sum()), "free": int(fr_.sum()), "generated_views": len(gv),
         "train_psnr_every10_before": round(float(np.mean(p0v)), 3), "train_psnr_every10_after": round(float(np.mean(pv)), 3),
         "calib": {g["name"]: g["calib_px"] for g in gv}, "seconds": round(time.time() - t0), "curve": log, "args": vars(a)}
    json.dump(r, open(os.path.splitext(a.out)[0] + "_mvrefit.json", "w"), indent=1)
    print("MV_REFIT", json.dumps({k: v for k, v in r.items() if k not in ("curve", "args", "calib")}), flush=True)


if __name__ == "__main__":
    main()
