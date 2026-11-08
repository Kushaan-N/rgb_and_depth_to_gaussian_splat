"""Standalone object splats from mv_seva.py's generated views, swapped into the scene's object zones (gsplat venv, GPU;
cosmetic: the backs are invented by the diffusion model).

  1. fit   per object (--ids): gaussians initialised on the WHOLE visual-hull surface (mv_seva.py hull, --cell e.g. 3 mm:
           a flat disc per voxel, colour from the best-facing generated view), trained ALONE (no scene) on the SEVA views
           + the real input crops (weight --real-w), both masked to the object (BiRefNet): L1+SSIM of the render composited
           on grey vs the masked target, (alpha - mask)^2, anti-needle; --densify-at: the top --densify-frac rows by mean
           position-gradient are split in two (brick detail); pruned at opacity < 0.02.
  2. swap  (--wholesale: every base row in the box above --floor-m goes, the whole fitted splat comes in) base rows in the object's box (mv_seva register box + --zone-m, above the floor) that no training view
           constrains (support max blend weight < --del-w) are deleted; well-seen base rows are kept; object rows a
           training view would show (in front of the base's rendered depth: trellis_complete.Vis.shows) are dropped, the
           rest get their opacity ramped down within --blend-m of a kept well-seen base row (seam).
  3. harmonise  colour only (sh0 of the object rows) on the real training frames that see the object, viewer-exact
           renders of the whole scene, --harm-iters.
  -> --out (PLY in the base's frame) + <out>_src.npz (base row index, -1 = new) for solidify_unseen.py budget, previews.

    python scripts/object_priors/mv_objfit.py --ply base.ply --support D/support.npz --gen S --dataset <train> <test> \
        --sim3-json sim3.json --objects D/objects --splat-depth-dir D/depth --ids 3 4 --out T/refit.ply
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from solidify_unseen import sim3, all_frames  # noqa: E402
from mv_refit import quat_from_z  # noqa: E402

C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--support", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--objects", required=True); ap.add_argument("--splat-depth-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--fit-iters", type=int, default=4000); ap.add_argument("--real-w", type=float, default=2.0)
    ap.add_argument("--densify-at", type=int, nargs="*", default=[1000, 2000]); ap.add_argument("--densify-frac", type=float, default=0.15)
    ap.add_argument("--lam-alpha", type=float, default=1.0); ap.add_argument("--lam-needle", type=float, default=0.3)
    ap.add_argument("--max-scale-m", type=float, default=0.012); ap.add_argument("--zone-m", type=float, default=0.03)
    ap.add_argument("--floor-m", type=float, default=0.01); ap.add_argument("--del-w", type=float, default=0.02)
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--blend-m", type=float, default=0.015)
    ap.add_argument("--wholesale", action="store_true", help="delete every base row in the (tight) box above the floor and insert the "
                    "whole fitted splat (no visibility drop, no seam ramp): seen views of the stack change")
    ap.add_argument("--harm-iters", type=int, default=400); ap.add_argument("--display-scale", type=float, default=1.8)
    # trellis_complete.Vis parameters (training-view visibility)
    ap.add_argument("--frame-stride", type=int, default=1); ap.add_argument("--max-incidence", type=float, default=85.0)
    ap.add_argument("--depth-tol-m", type=float, default=0.03); ap.add_argument("--splat-tol-rel", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from scipy.spatial import cKDTree
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, ssim, psnr
    from trellis_complete import Vis
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"
    s, R, t = sim3(a.sim3_json)
    vtx, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); N0 = len(p0["means"]); K = p0["shN"].shape[1]
    wmax = np.load(a.support)["wmax"]; assert len(wmax) == N0, "support was computed for another PLY"
    reg = {r["id"]: r for r in json.load(open(os.path.join(a.gen, "register.json")))}
    ids = [i for i in (a.ids or sorted(reg)) if i in reg]
    sd = np.load(os.path.join(a.gen, "seeds.npz")); AX, AN, AC, Ao = sd["AX"], sd["AN"], sd["AC"], sd["Aobj"].astype(int)
    T_ = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    Rv = ViewerRaster("ss352", "exact", display_scale=1.5)

    def views(V, w):
        out = []
        for name, cam, c2w in frames(V, 0, "all"):
            oid = int(name.split("_")[0][3:])
            if oid not in ids:
                continue
            im = np.asarray(Image.open(os.path.join(V, "images", name)).convert("RGB"), np.float32) / 255
            mk = np.asarray(Image.open(os.path.join(V, "masks", name.replace(".png", "_mask.png")))) > 127
            out.append({"oid": oid, "name": name, "w2c": to_t(np.linalg.inv(c2w), dev), "cam": cam_args(cam), "img": T_(im),
                        "mask": torch.tensor(mk, device=dev), "w": w})
        return out
    gv = views(os.path.join(a.gen, "views"), 1.0) + views(os.path.join(a.gen, "views_real"), a.real_w)
    print(f"[objfit] objects {ids}: {len(gv)} views ({sum(g['w'] > 1 for g in gv)} real crops)", flush=True)
    bg = torch.full((3,), 0.5, device=dev)
    objs = {}
    t0 = time.time()
    # ---------------------------------------------------------------- 1. standalone fits
    for oid in ids:
        k = Ao == oid; X, Nn, C = AX[k], AN[k], AC[k]; cell = float(np.median(cKDTree(X).query(X, 2)[0][:, 1])) * s if len(X) > 2 else 0.005
        M = len(X); sc_in = np.log(0.7 * cell / s); sc_n = np.log(0.2 * cell / s)
        P = {"means": T_(X), "scales": T_(np.tile([sc_in, sc_in, sc_n], (M, 1))),
             "quats": T_(quat_from_z(Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9))),
             "opacities": T_(np.full(M, 2.0)), "sh0": T_(((np.clip(C, 0, 1) - 0.5) / C0)[:, None])}
        P = torch.nn.ParameterDict({kk: torch.nn.Parameter(v) for kk, v in P.items()})
        lr = {"means": 2e-4 * cell / s, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3}
        mk_opt = lambda: torch.optim.Adam([{"params": [P[kk]], "lr": lr[kk], "name": kk} for kk in P], eps=1e-15)
        opt = mk_opt(); mine = [g for g in gv if g["oid"] == oid]; ws = np.array([g["w"] for g in mine]); ws /= ws.sum()
        gacc = torch.zeros(M, device=dev); gcnt = torch.zeros(M, device=dev)
        print(f"[objfit] object {oid}: {M:,} hull-surface rows (cell {cell * 1000:.1f} mm), {len(mine)} views", flush=True)
        for step in range(1, a.fit_iters + 1):
            G = mine[np.random.choice(len(mine), p=ws)]
            full = dict(P); full["shN"] = torch.zeros(len(P["means"]), K, 3, device=dev)
            io, ao, _ = Rv.render(full, G["w2c"], *G["cam"], sh_degree=0)
            comp = io + (1 - ao)[..., None] * bg; tgt = torch.where(G["mask"][..., None], G["img"], bg.expand_as(G["img"]))
            l = 0.8 * (comp - tgt).abs().mean() + 0.2 * (1 - ssim(comp.permute(2, 0, 1)[None], tgt.permute(2, 0, 1)[None]))
            l = l + a.lam_alpha * ((ao - G["mask"].float()) ** 2).mean()
            ss_ = torch.sort(P["scales"], 1).values; l = l + a.lam_needle * torch.relu(ss_[:, 2] - ss_[:, 1] - np.log(3.0)).mean()
            opt.zero_grad(set_to_none=True); l.backward()
            gacc[:len(P["means"])] += P["means"].grad.norm(dim=-1); gcnt[:len(P["means"])] += (P["means"].grad.abs().sum(-1) > 0).float()
            for g in opt.param_groups:
                g["lr"] = lr[g["name"]] * (0.1 ** (step / a.fit_iters) if g["name"] == "means" else 1.0)
            opt.step()
            with torch.no_grad():
                P["scales"].clamp_(max=float(np.log(a.max_scale_m / s)))
            if step in a.densify_at:                                       # split the highest-gradient rows in two
                with torch.no_grad():
                    gm = gacc / gcnt.clamp_min(1); n_ = int(a.densify_frac * len(gm)); top = torch.topk(gm, n_).indices
                    from finetune_ply_viewer import quat_to_rot
                    Rm = quat_to_rot(torch.nn.functional.normalize(P["quats"][top], dim=-1)); sc = torch.exp(P["scales"][top])
                    off = (Rm @ (torch.randn(n_, 3, device=dev) * sc)[..., None])[..., 0]
                    new = {kk: P[kk].detach().clone() for kk in P}
                    new["means"][top] = P["means"][top] - off
                    add = {kk: P[kk][top].detach().clone() for kk in P}; add["means"] = add["means"] + off
                    new["scales"][top] -= np.log(1.6); add["scales"] -= np.log(1.6)
                    P = torch.nn.ParameterDict({kk: torch.nn.Parameter(torch.cat([new[kk], add[kk]])) for kk in P})
                    opt = mk_opt(); gacc = torch.zeros(len(P["means"]), device=dev); gcnt = torch.zeros_like(gacc)
                print(f"[objfit] object {oid} step {step}: densified -> {len(P['means']):,} rows", flush=True)
            if step % a.log_every == 0 or step == 1:
                print(f"[objfit] object {oid} step {step}: loss {float(l):.4f} psnr {psnr(comp.detach(), tgt):.2f} ({time.time() - t0:.0f}s)", flush=True)
        with torch.no_grad():
            keep = torch.sigmoid(P["opacities"]) > 0.02
            Q = {kk: P[kk].detach()[keep] for kk in P}; Q["shN"] = torch.zeros(int(keep.sum()), K, 3, device=dev)
            prev = []
            for G in mine[::max(1, len(mine) // 10)]:
                io, ao, _ = Rv.render(Q, G["w2c"], *G["cam"], sh_degree=0)
                comp = (io + (1 - ao)[..., None] * bg).clamp(0, 1); tgt = torch.where(G["mask"][..., None], G["img"], bg.expand_as(G["img"]))
                prev.append(np.hstack([(tgt.cpu().numpy() * 255).astype(np.uint8), (comp.cpu().numpy() * 255).astype(np.uint8)]))
            Image.fromarray(np.vstack([np.asarray(Image.fromarray(x).resize((512, 256))) for x in prev])).save(
                os.path.splitext(a.out)[0] + f"_obj{oid}_fit.jpg", quality=88)
        objs[oid] = Q; print(f"[objfit] object {oid}: fitted {len(Q['means']):,} rows", flush=True)
    # ---------------------------------------------------------------- 2. swap into the scene
    Xb = p0["means"].cpu().numpy().astype(np.float64); Xm = s * (Xb @ R.T) + t; opb = torch.sigmoid(p0["opacities"]).cpu().numpy()
    dele = np.zeros(N0, bool); kept_seen = np.zeros(N0, bool)
    vis = Vis(argparse.Namespace(dataset=a.dataset[0], frame_stride=a.frame_stride, splat_depth_dir=a.splat_depth_dir,
                                 max_incidence=a.max_incidence, depth_tol_m=a.depth_tol_m, splat_tol_rel=a.splat_tol_rel), s)
    newrows, rep = [], {}
    for oid, Q in objs.items():
        r = reg[oid]; c0, A = np.array(r["c0"]), np.array(r["axes"])
        L = (Xm - c0) @ A; lo, hi = np.array(r["box_lo"]) - a.zone_m, np.array(r["box_hi"]) + a.zone_m
        inz = np.all((L[:, :2] >= lo[:2]) & (L[:, :2] <= hi[:2]), 1) & (L[:, 2] > a.floor_m) & (L[:, 2] <= hi[2])
        if a.wholesale:                                                    # the stack is replaced by its fitted splat
            d_ = inz; ks = np.zeros(N0, bool)
        else:
            d_ = inz & (wmax < a.del_w); ks = inz & (wmax >= a.face_w) & (opb > 0.5)
        dele |= d_; kept_seen |= ks
        Xq = Q["means"].cpu().numpy().astype(np.float64)
        shown = vis.shows(Xq) if not a.wholesale else np.zeros(len(Xq), bool)
        dist = cKDTree(Xb[ks]).query(Xq)[0] * s if ks.any() else np.full(len(Xq), 1e9)
        keep = ~shown
        ramp = np.clip(dist / a.blend_m, 0.05, 1.0)
        with torch.no_grad():
            op = torch.sigmoid(Q["opacities"]) * torch.tensor(ramp, device=dev, dtype=torch.float32)
            Q["opacities"] = torch.logit(op.clamp(1e-4, 1 - 1e-4))
        newrows.append({kk: v[torch.tensor(keep, device=dev)] for kk, v in Q.items()})
        rep[oid] = {"zone_rows": int(inz.sum()), "deleted": int(d_.sum()), "kept_seen": int(ks.sum()), "fit_rows": len(Xq),
                    "shown_dropped": int(shown.sum()), "added": int(keep.sum())}
        print(f"[objfit] swap object {oid}: {rep[oid]}", flush=True)
    kb = torch.tensor(~dele, device=dev)
    Pn = {kk: torch.cat([p0[kk][kb]] + [nr[kk] for nr in newrows]) for kk in p0}
    nb = int(kb.sum()); M = len(Pn["means"]) - nb
    src = np.r_[np.nonzero(~dele)[0], -np.ones(M, int)]
    # ---------------------------------------------------------------- 3. colour-only harmonisation on the real frames
    params = torch.nn.ParameterDict({kk: torch.nn.Parameter(v.clone()) for kk, v in Pn.items()})
    tr = all_frames(a.dataset)
    cen = np.stack([np.array(reg[o]["c0"]) for o in objs]); cd = ((cen - t) @ R) / s
    seen_fr = []
    for i, (n_, cam, c2w) in enumerate(tr):
        fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w); Pc = cd @ w2c[:3, :3].T + w2c[:3, 3]
        u = fx * Pc[:, 0] / Pc[:, 2] + cx; v = fy * Pc[:, 1] / Pc[:, 2] + cy
        if np.any((Pc[:, 2] > 0.05) & (u > 0) & (u < W) & (v > 0) & (v < H)):
            seen_fr.append(i)
    gtf = lambda n_: next(os.path.join(ds, "images", n_) for ds in a.dataset if os.path.exists(os.path.join(ds, "images", n_)))
    Rv.r = a.display_scale; opt = torch.optim.Adam([params["sh0"]], lr=2.5e-3, eps=1e-15); newm = torch.zeros(len(src), dtype=torch.bool, device=dev); newm[nb:] = True
    for step in range(1, a.harm_iters + 1 if seen_fr else 1):
        i = seen_fr[np.random.randint(len(seen_fr))]; n_, cam, c2w = tr[i]
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = load_rgb(gtf(n_), dev).float() / 255
        l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(set_to_none=True); l.backward(); params["sh0"].grad[~newm] = 0; opt.step()
    with torch.no_grad():
        pv, p0v = [], []
        for i in seen_fr[::max(1, len(seen_fr) // 30)]:
            n_, cam, c2w = tr[i]; gt = load_rgb(gtf(n_), dev).float() / 255; w2c = to_t(np.linalg.inv(c2w), dev)
            pv.append(psnr(Rv.render(params, w2c, *cam_args(cam), sh_degree=deg)[0], gt)); p0v.append(psnr(Rv.render(p0, w2c, *cam_args(cam), sh_degree=deg)[0], gt))
    vt = np.concatenate([vtx.data[~dele], np.zeros(M, vtx.data.dtype)])
    save_ply(a.out, vt, params)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=src)
    out = {"objects": rep, "rows": len(src), "new": M, "deleted": int(dele.sum()), "harm_frames": len(seen_fr),
           "obj_frames_psnr_before": round(float(np.mean(p0v)), 3) if p0v else None, "obj_frames_psnr_after": round(float(np.mean(pv)), 3) if pv else None,
           "seconds": round(time.time() - t0), "args": vars(a)}
    json.dump(out, open(os.path.splitext(a.out)[0] + "_mvrefit.json", "w"), indent=1)
    print("MV_OBJFIT", json.dumps({k: v for k, v in out.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
