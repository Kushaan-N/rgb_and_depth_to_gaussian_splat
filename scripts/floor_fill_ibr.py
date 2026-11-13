#!/usr/bin/env python3
"""A proper floor where no camera ever looked (e.g. the middle of a room the robot circled facing outward).

Input: the top-down floor texture of scripts/make_lidar_ibr_views.py `views --floor-map` (real pixels of the training
frames re-projected onto the fitted floor plane; `any` = a frame saw the cell, `occ` = floor exists there, `obj` =
something stands on it). The unseen free floor (occ & ~any & ~obj) is what a splat cannot learn: training leaves it
empty or covered with unconstrained smears.

  inpaint  (inpaint venv; LaMa as scripts/fill_inpaint.py) the texture is continued into the unseen cells (tiles, tape
           lines) -> <map>_inpainted.npz (+ before / after / hole PNGs).
  apply    (gsplat venv, GPU) on a finished PLY:
           1. remove what sits in the unseen floor's column (--hmin..--hmax m around the plane, hole dilated by
              --grow m) and no training view supports (max blending weight < --max-train-w, viewer-exact rasteriser,
              scripts/clean_haze.py train_stats) -- the smears / streaks; plus any unsupported gaussian larger than
              --giant m whose centre is in that column. Measured restore: every training frame is rendered before /
              after, a removed gaussian blending into a changed pixel (> --thr) comes back, until no pixel changes;
           2. append one flat opaque disc per unseen cell (plane-exact, normal = floor normal, colour = inpainted
              texture, SH bands 1-3 zero: no view dependence) as the LAST gaussians -> --out-filled, and the splat
              without them -> --out-base, for the usual measured harm check:
                 python scripts/fill_harm_ply.py --base B --filled F --dataset TRAIN --out OUT
    python scripts/make_lidar_ibr_views.py views --dataset TRAIN --cloud C --sim3-json S --scene-json scene.json --floor-map M.npz
    python scripts/floor_fill_ibr.py inpaint --map M.npz --lama-pt big-lama.pt
    python scripts/floor_fill_ibr.py apply --ply IN.ply --map M_inpainted.npz --dataset TRAIN --out-base B.ply --out-filled F.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C0 = 0.28209479177387814


def hole_of(z, min_cells=400):
    """unseen free floor: floor exists, no frame saw it, nothing stands on it; specks dropped."""
    from scipy import ndimage
    h = z["occ"] & ~z["any"] & ~z["obj"]
    h = ndimage.binary_opening(h, iterations=2)
    lab, n = ndimage.label(h)
    if n:
        size = ndimage.sum(h, lab, np.arange(1, n + 1)); h = np.isin(lab, 1 + np.nonzero(size >= min_cells)[0])
    return h


def inpaint(a):
    import cv2
    from PIL import Image
    from fill_inpaint import Inpainter
    z = dict(np.load(a.map)); rgb = z["rgb"].astype(np.float32); known = z["any"]
    hole = hole_of(z, a.min_cells)
    # context for the network: unknown cells that are NOT to be filled (outside the floor, under objects) get a smooth
    # extension of the known texture, so the hole is surrounded by floor on every side
    acc, w = rgb * known[..., None], known.astype(np.float32); ctx, wo = acc.copy(), w.copy()
    for k in range(1, 9):
        sz = 2 * (2 ** k) + 1; b, bw = cv2.blur(acc, (sz, sz)), cv2.blur(w, (sz, sz)); m = (wo < 1e-3) & (bw > 1e-3)
        ctx[m] = b[m] / bw[m][:, None]; wo[m] = 1
    # LiDAR "objects" inside the unseen floor (>= --enclosed of their rim is unseen floor): if the splat has nothing the
    # training views support there either (apply decides), it was a transient (a person standing still) -> floor too
    from scipy import ndimage
    maybe = np.zeros_like(hole); lab, nl = ndimage.label(z["obj"] & z["occ"] & ~known)
    near_hole = ndimage.binary_dilation(hole, iterations=3)
    for li in range(1, nl + 1):
        comp = lab == li
        if comp.sum() > a.max_object_cells or not (comp & near_hole).any():
            continue
        rim = ndimage.binary_dilation(comp, iterations=3) & ~comp
        if (rim & hole).sum() >= a.enclosed * rim.sum():
            maybe |= comp
    hd = cv2.dilate((hole | maybe).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # overlap the seam
    out = Inpainter(a.method, a.lama_pt)(ctx, hd)
    z["rgb_filled"] = out.astype(np.float32); z["hole"] = hole; z["maybe"] = maybe
    dst = os.path.splitext(a.map)[0] + "_inpainted.npz"; np.savez_compressed(dst, **z)
    u8 = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8)
    Image.fromarray(u8(ctx * ~hd[..., None])).save(os.path.splitext(a.map)[0] + "_before.png")
    Image.fromarray(u8(out)).save(os.path.splitext(a.map)[0] + "_after.png")
    Image.fromarray(u8(hole.astype(np.float32))).save(os.path.splitext(a.map)[0] + "_hole.png")
    print("FLOOR_INPAINT " + json.dumps({"hole_m2": round(float(hole.sum()) * float(z["cell"]) ** 2, 3), "hole_cells": int(hole.sum()), "maybe_object_cells": int(maybe.sum()), "out": dst}), flush=True)


def apply(a):
    import torch
    from scipy import ndimage
    from finetune_ply_viewer import load_ply, save_ply, sh_deg, to_t, cam_args, frames
    from clean_haze import make_raster, train_stats, sub
    from make_pseudo_views import rot2quat
    dev = "cuda"
    z = np.load(a.map); hole = z["hole"]; mc = float(z["cell"]); e1, e2, n, d = z["e1"], z["e2"], z["n"], float(z["d"])
    s, R, t = float(z["sim3_s"]), z["sim3_R"], z["sim3_t"]; x0, y0 = float(z["x0"]), float(z["y0"])
    vert, p = load_ply(a.ply, dev); N = len(p["means"]); deg = sh_deg(p)
    fr = frames(a.dataset, 0, "all")[:: a.every]
    Xm = s * p["means"].cpu().numpy().astype(np.float64) @ R.T + t; h = Xm @ n + d
    ci, cj = ((Xm @ e1 - x0) / mc).astype(int), ((Xm @ e2 - y0) / mc).astype(int)
    inb = (ci >= 0) & (ci < hole.shape[0]) & (cj >= 0) & (cj < hole.shape[1])
    st = None; smax = np.exp(p["scales"].cpu().numpy()).max(1) * s
    op = 1 / (1 + np.exp(-p["opacities"].cpu().numpy()))
    maybe = z["maybe"] if "maybe" in z.files else np.zeros_like(hole)
    if hole.any() or maybe.any():
        st = train_stats(p, fr, deg, a.display_scale, dev)
    conv = 0
    if maybe.any():                                              # LiDAR-only "objects" in the unseen floor: transient?
        lab, nl = ndimage.label(maybe); hole = hole.copy()
        for li in range(1, nl + 1):
            comp = lab == li; inc = np.zeros(N, bool); inc[inb] = comp[ci[inb], cj[inb]]
            sup = inc & (h > 0.05) & (h < 2.2) & (st["train_max_w"] > a.object_w)
            if float(op[sup].sum()) < a.object_mass * comp.sum() * mc * mc:      # supported opacity per m^2 of footprint
                hole |= comp; conv += int(comp.sum())
    grow = ndimage.binary_dilation(hole, iterations=max(1, int(round(a.grow / mc))))
    col = np.zeros(N, bool); col[inb] = grow[ci[inb], cj[inb]]
    col &= (h > a.hmin) & (h < a.hmax)
    info = {"ply": a.ply, "n": N, "hole_m2": round(float(hole.sum()) * mc * mc, 3), "in_column": int(col.sum())}
    rm = torch.zeros(N, dtype=torch.bool, device=dev)
    info["maybe_object_cells_to_floor"] = conv
    ex = torch.zeros(N, dtype=torch.bool, device=dev)
    if col.any():
        near = np.zeros(N, bool); near[inb] = ndimage.binary_dilation(hole, iterations=max(1, int(round(a.giant_reach / mc))))[ci[inb], cj[inb]]
        cand = (col | (near & (h > a.hmin) & (h < a.hmax) & (smax > a.giant))) & (st["train_max_w"] < a.max_train_w)
        info.update(unsupported_in_column=int(cand.sum()), giant_unsupported=int((cand & (smax > a.giant)).sum()))
        # huge, weakly supported splats crossing the patch (needles from below the floor, streaks): removed for good;
        # their small training-view footprint is left to the next fine-tune (QAT) -- measured by the caller's score
        big = near & (h > a.hmin) & (h < a.hmax) & (smax > a.giant) & (st["train_max_w"] < a.giant_w)
        info["giant_weak_removed_no_restore"] = int(big.sum()); cand |= big
        rm = torch.tensor(cand, device=dev); ex = torch.tensor(big, device=dev)
        Rr = make_raster(a.display_scale); hist = []
        with torch.no_grad():
            base = {nm: Rr.render(p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0].half() for nm, cam, c2w in fr}
            for it in range(a.iters):
                pc = sub(p, ~rm); restore = torch.zeros(N, dtype=torch.bool, device=dev); changed = 0; worst = 0.0
                for nm, cam, c2w in fr:
                    w2c = to_t(np.linalg.inv(c2w), dev); fx, fy, cx, cy, W, H = cam_args(cam)
                    diff = (Rr.render(pc, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0] - base[nm].float()).abs().amax(-1)
                    dd = diff > a.thr; changed += int(dd.sum()); worst = max(worst, float(diff.max()))
                    if not dd.any():
                        continue
                    Rr.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)
                    gid, pid, w = Rr.pairs
                    restore[gid[dd.view(-1)[pid] & rm[gid] & ~ex[gid] & (w > 0)]] = True
                hist.append({"iter": it, "changed_px": changed, "max_diff_255": round(worst * 255, 2), "removed": int(rm.sum()), "restore": int(restore.sum())})
                print(f"[floorfill] removal check {hist[-1]}", flush=True)
                if not restore.any():
                    break
                rm &= ~restore
        info["removal"] = hist
    keep = (~rm).cpu().numpy(); info["removed"] = int((~keep).sum())
    pb = {k: v[torch.from_numpy(keep).to(dev)] for k, v in p.items()}
    save_ply(a.out_base, vert.data[keep], pb)
    # ---- floor discs in the unseen cells (dataset frame)
    hi, hj = np.nonzero(hole); M = len(hi)
    Cm = ((hi + 0.5) * mc + x0)[:, None] * e1 + ((hj + 0.5) * mc + y0)[:, None] * e2 - d * n
    Cd = (Cm - t) @ R / s
    Rd = R.T @ np.stack([e1, e2, n], 1); q = rot2quat(Rd)                                    # local z = floor normal
    rgb = z["rgb_filled"][hi, hj].astype(np.float32)
    K = p["shN"].shape[1]
    add = {"means": torch.tensor(Cd, dtype=torch.float32, device=dev),
           "scales": torch.log(torch.tensor([a.sigma * mc / s, a.sigma * mc / s, a.thick / s], dtype=torch.float32, device=dev)).repeat(M, 1),
           "quats": torch.tensor(q, dtype=torch.float32, device=dev).repeat(M, 1),
           "opacities": torch.full((M,), float(np.log(a.opacity / (1 - a.opacity))), device=dev),
           "sh0": torch.tensor((rgb - 0.5) / C0, device=dev)[:, None], "shN": torch.zeros(M, K, 3, device=dev)}
    pf = {k: torch.cat([pb[k], add[k]]) for k in pb}
    arr = np.zeros(int(keep.sum()) + M, dtype=vert.data.dtype); arr[: int(keep.sum())] = vert.data[keep]
    save_ply(a.out_filled, arr, pf)
    info.update(discs=M, out_base=a.out_base, out_filled=a.out_filled, args={k: v for k, v in vars(a).items() if k != "func"})
    json.dump(info, open(os.path.splitext(a.out_filled)[0] + "_floorfill.json", "w"), indent=1)
    print("FLOOR_FILL " + json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    i = sub_.add_parser("inpaint")
    i.add_argument("--map", required=True); i.add_argument("--method", default="lama", choices=["lama", "telea"])
    i.add_argument("--lama-pt", default=None); i.add_argument("--min-cells", type=int, default=400, help="smallest unseen patch filled (cells)")
    i.add_argument("--enclosed", type=float, default=0.6); i.add_argument("--max-object-cells", type=int, default=6000)
    i.set_defaults(func=inpaint)
    p = sub_.add_parser("apply")
    p.add_argument("--ply", required=True); p.add_argument("--map", required=True, help="<map>_inpainted.npz")
    p.add_argument("--dataset", required=True, help="training frames (COLMAP text dataset, the PLY's frame)")
    p.add_argument("--out-base", required=True); p.add_argument("--out-filled", required=True)
    p.add_argument("--hmin", type=float, default=-0.3); p.add_argument("--hmax", type=float, default=1.0, help="m above the floor: the cleaned column")
    p.add_argument("--grow", type=float, default=0.03, help="m the column extends beyond the unseen cells")
    p.add_argument("--max-train-w", type=float, default=0.02, help="max training blending weight below which a gaussian is unsupported")
    p.add_argument("--giant", type=float, default=0.3, help="m: unsupported gaussians larger than this ...")
    p.add_argument("--giant-reach", type=float, default=0.5, help="... are removed up to this far (m) from the unseen cells (streaks reaching in)")
    p.add_argument("--giant-w", type=float, default=0.3, help="... or, with max training weight below this, removed WITHOUT restore")
    p.add_argument("--object-w", type=float, default=0.2); p.add_argument("--object-mass", type=float, default=200.0,
                   help="a LiDAR-only object inside the unseen floor stays if supported opacity per m^2 of footprint exceeds this")
    p.add_argument("--thr", type=float, default=2.0 / 255); p.add_argument("--iters", type=int, default=4)
    p.add_argument("--display-scale", type=float, default=1.8); p.add_argument("--every", type=int, default=1)
    p.add_argument("--sigma", type=float, default=0.7, help="disc sigma in cells"); p.add_argument("--thick", type=float, default=0.0008, help="m")
    p.add_argument("--opacity", type=float, default=0.98)
    p.set_defaults(func=apply)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
