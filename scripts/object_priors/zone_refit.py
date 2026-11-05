"""Joint refit that trains THROUGH the object prior (gsplat venv, GPU): object_priors.py's candidate (replacement
surfaces + kept junk) fine-tuned so that the objects look solid from orbit views while the training views stay
faithful. Unlike solidify_unseen.py refit (whose orbit terms pull on every trainable gaussian and cost seen-view
fidelity), the orbit terms here move only FREE rows: the prior's new rows and object-zone input rows no training view
constrains (support max blend weight < --free-w). Camera-constrained zone rows get the photometric loss only (lr x
--lr-seen); everything outside the zones is frozen (byte-identical).

  per step: viewer-exact L1+SSIM on a training frame (all trainable rows)
          + on a random orbit view (no RGB target), object pixels only, gradients to FREE rows only:
            --lam-hole (1 - alpha), --lam-tv (colour total variation), --lam-needle (s_max/s_mid > --needle-ratio),
            --lam-opa (free rows pushed to opacity 0 or 1: no translucent fog)
          + surface prior on new rows (thin, centre within --pos-tol-m of its primitive sample)

    python scripts/object_priors/zone_refit.py --ply T/textured.ply --src T/src.npz --zone T/zone.npz \
        --support D/support.npz --orbit D/orbit --objects D/objects --cloud lidar.ply --sim3-json sim3.json \
        --dataset <train ds> <test ds> --out T/zrefit.ply
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import sim3, load_objects, all_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--src", required=True); ap.add_argument("--zone", required=True)
    ap.add_argument("--support", required=True); ap.add_argument("--orbit", required=True); ap.add_argument("--objects", required=True)
    ap.add_argument("--cloud", required=True); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--display-scale", default="1.5:4")
    ap.add_argument("--free-w", type=float, default=0.2); ap.add_argument("--lr-seen", type=float, default=0.1)
    ap.add_argument("--lam-hole", type=float, default=1.0); ap.add_argument("--lam-tv", type=float, default=0.3)
    ap.add_argument("--lam-needle", type=float, default=0.3); ap.add_argument("--lam-opa", type=float, default=0.1)
    ap.add_argument("--needle-ratio", type=float, default=3.0); ap.add_argument("--pos-tol-m", type=float, default=0.015)
    ap.add_argument("--lam-pos", type=float, default=0.1); ap.add_argument("--lam-flat", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--max-scale-m", type=float, default=0.03, help="new rows: no axis larger than this (no blow-up blobs)")
    ap.add_argument("--new-only", action="store_true", help="cosmetic: only the prior's new rows train (every input row frozen)")
    ap.add_argument("--lam-photo", type=float, default=1.0, help="weight of the training-view photometric loss")
    a = ap.parse_args()
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, ssim, psnr
    from build_depth_targets import zbuffer_target
    from complete_objects import object_surface
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"
    s, R, t = sim3(a.sim3_json)
    vtx, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); N = len(p0["means"])
    src = np.load(a.src)["src"]; assert len(src) == N
    zin = np.load(a.zone)["zone"]; wmax = np.load(a.support)["wmax"]
    new = src < 0; zrow = new.copy(); zrow[~new] = zin[src[~new]]
    free = new.copy(); free[~new] = zin[src[~new]] & (wmax[src[~new]] < a.free_w)
    if a.new_only:
        zrow, free = new.copy(), new.copy()
    T = lambda m: torch.tensor(m, device=dev)
    tm, fr_, nw = T(zrow), T(free), T(new); seen_rows = tm & ~fr_
    print(f"[zrefit] {N:,} rows: trainable {int(tm.sum()):,} (free {int(fr_.sum()):,} incl. {int(nw.sum()):,} new; seen {int(seen_rows.sum()):,})", flush=True)
    # orbit views: object pixels = the object's closed surface is the front surface (LiDAR + closures z-buffer)
    ometa, opts = load_objects(a.objects); n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    osurf = np.concatenate([object_surface(X, n_f, d_f, 0.02, 0.02)[0] for X in opts])
    from plyfile import PlyData
    cv = PlyData.read(a.cloud)["vertex"].data; L = np.stack([cv["x"], cv["y"], cv["z"]], 1).astype(np.float64)
    L = L[np.unique(np.floor(L / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
    to_d = lambda A: torch.tensor(((A - t) @ R) / s, device=dev)
    allX, objX = to_d(np.concatenate([L, osurf])), to_d(osurf); orb = []
    for name, cam, c2w in frames(a.orbit, 0, "all"):
        fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w)
        Rc = torch.tensor(w2c[:3, :3], device=dev); tc = torch.tensor(w2c[:3, 3], device=dev)
        Da, _ = zbuffer_target(allX, Rc, tc, W, H, (fx, fy, cx, cy), 0.02 / s, 1e6)
        Do, _ = zbuffer_target(objX, Rc, tc, W, H, (fx, fy, cx, cy), 0.02 / s, 1e6, min_support=3)
        mk = (Do > 0) & (Da > 0) & ((Do - Da).abs() < 0.03 / s)
        if mk.sum() > 200:
            orb.append((to_t(w2c, dev), (fx, fy, cx, cy, W, H), mk))
    tr = all_frames(a.dataset)
    gtf = lambda n: next(os.path.join(ds, "images", n) for ds in a.dataset if os.path.exists(os.path.join(ds, "images", n)))
    rgb = [load_rgb(gtf(n), dev) for n, _, _ in tr]
    centres = np.stack([c[:3, 3] for _, _, c in tr]); scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    lr = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in a.display_scale.split(":")]; Rv = ViewerRaster("ss352", "exact", display_scale=rs[0])
    ni = torch.nonzero(nw)[:, 0]; x0 = p0["means"][ni].clone(); tol_d = a.pos_tol_m / s
    fi = torch.nonzero(fr_)[:, 0]; log3 = float(np.log(a.needle_ratio)); keys = list(params.keys())
    order, log, t0 = [], [], time.time()
    for step in range(1, a.iters + 1):
        if not order:
            order = list(np.random.permutation(len(tr)))
        k = int(order.pop()); name, cam, c2w = tr[k]
        Rv.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[-1]))))
        wu = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu * (0.1 ** (step / a.iters) if g["name"] == "means" else 1.0)
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = rgb[k].float() / 255
        lph = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        g1 = torch.autograd.grad(a.lam_photo * lph, [params[x] for x in keys], allow_unused=True)
        # prior terms: free rows only
        w2c_o, ca, mk = orb[np.random.randint(len(orb))]
        io, ao, _ = Rv.render(params, w2c_o, *ca, sh_degree=deg)
        lh = (1 - ao.clamp(0, 1))[mk].mean()
        gx = (io[:, 1:] - io[:, :-1]).abs().sum(-1)[mk[:, 1:] & mk[:, :-1]]; gy = (io[1:] - io[:-1]).abs().sum(-1)[mk[1:] & mk[:-1]]
        lt = torch.cat([gx, gy]).mean()
        ssf = torch.sort(params["scales"][fi], 1).values
        lneed = torch.relu(ssf[:, 2] - ssf[:, 1] - log3).mean()
        of = torch.sigmoid(params["opacities"][fi]); lopa = (of * (1 - of)).mean()
        dx = params["means"][ni] - x0; lpos = (torch.relu(dx.norm(dim=-1) - tol_d) ** 2).mean() / tol_d ** 2 if len(ni) else torch.zeros((), device=dev)
        lflat = torch.exp(params["scales"][ni].min(1).values).mean() / (0.02 / s) if len(ni) else torch.zeros((), device=dev)
        lpr = a.lam_hole * lh + a.lam_tv * lt + a.lam_needle * lneed + a.lam_opa * lopa + a.lam_pos * lpos + a.lam_flat * lflat
        g2 = torch.autograd.grad(lpr, [params[x] for x in keys], allow_unused=True)
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
            for x in keys:                                                 # seen zone rows move at lr_seen x the rate
                params[x][seen_rows] = old[x] + a.lr_seen * (params[x][seen_rows] - old[x])
            params["scales"][ni] = params["scales"][ni].clamp_max(float(np.log(a.max_scale_m / s)))   # free rows stay surfels
            params["scales"][fi] = params["scales"][fi].clamp_max(float(np.log(max(a.max_scale_m, 1e-3) * 3 / s)))
        if step % a.log_every == 0 or step == 1:
            e = {"step": step, "psnr": round(psnr(img.detach(), gt), 2), "hole": round(float(lh), 4), "tv": round(float(lt), 4),
                 "needle": round(float(lneed), 4), "opa": round(float(lopa), 4), "pos": round(float(lpos), 4)}
            log.append(e); print(f"[zrefit] {e} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        Rv.r = rs[0]; sub = list(enumerate(tr))[::10]
        pv = [psnr(Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n, cam, c2w) in sub]
        p0v = [psnr(Rv.render(p0, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255) for k, (n, cam, c2w) in sub]
        for x in keys:
            params[x][~tm] = p0[x][~tm]
    save_ply(a.out, vtx, params)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=src)
    r = {"rows": N, "trainable": int(tm.sum()), "free": int(fr_.sum()), "new": int(nw.sum()), "orbit_views": len(orb),
         "train_psnr_every10_before": round(float(np.mean(p0v)), 3), "train_psnr_every10_after": round(float(np.mean(pv)), 3),
         "seconds": round(time.time() - t0), "curve": log, "args": vars(a)}
    json.dump(r, open(os.path.splitext(a.out)[0] + "_zrefit.json", "w"), indent=1)
    print("ZONE_REFIT", json.dumps({k: v for k, v in r.items() if k not in ("curve", "args")}), flush=True)


if __name__ == "__main__":
    main()
