#!/usr/bin/env python3
"""Energy-preserving minimum footprint: make sub-pixel gaussians survive the viewer's size cull (gsplat venv, GPU).

3DGUT models keep many tiny gaussians; a standard viewer drops every splat whose dilated 2D covariance has max
eigenvalue below a threshold (PlayCanvas 2.7.4: 0.125 px^2 incl. its 0.075 px^2 dilation; SuperSplat 3.5.2:
0.5 px^2 incl. 0.3), so their colour disappears (holes / darker fine texture). For each gaussian this script
projects it into every TRAINING camera that sees it (centre in front and inside the image), and if the
viewer culls it in more than --cull-frac of those views it inflates it isotropically in 3D,
S' = S + v I (scales sqrt(s^2 + v), rotation unchanged), with v the median over those views of the variance
needed to pass the cull (x --margin), and rescales its opacity to keep the drawn energy:
o' = o * sqrt(det(S2D_old + dI) / det(S2D_new + dI)) (median over the same views, capped at 1). Optional:
--prune-opacity drops gaussians with opacity below it (after the rescale). Writes a .ply with the input layout.

    python viewer_fix_ply.py --ply s.ply --dataset DS --out fixed.ply [--test-split-interval 8] [--viewer pc274] \
        [--cull-frac 0.5] [--margin 1.05] [--prune-opacity 0]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_ply_viewer import ViewerRaster, load_ply, save_ply, frames, cam_args, to_t  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--test-split-interval", type=int, default=0, help="3DGRUT's rule; held-out frames are not used")
    ap.add_argument("--viewer", default="pc274", choices=("pc274", "ss352"))
    ap.add_argument("--cull-frac", type=float, default=0.5, help="fix gaussians culled in more than this share of views")
    ap.add_argument("--margin", type=float, default=1.05, help="target eigenvalue = threshold x margin")
    ap.add_argument("--prune-opacity", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    import torch
    dev = "cuda"; t0 = time.time()
    vertex, p = load_ply(args.ply, dev); N = len(p["means"])
    fr = frames(args.dataset, args.test_split_interval, "train")
    vr = ViewerRaster(args.viewer, "exact"); dil, lmin = vr.dil, vr.min_lambda(); target = lmin * args.margin
    cams = [(to_t(np.linalg.inv(c2w), dev), cam_args(cam)) for _, cam, c2w in fr]
    newv = torch.zeros(N, device=dev); ratio = torch.ones(N, device=dev)
    nvis = torch.zeros(N, device=dev, dtype=torch.int32); ncull = torch.zeros_like(nvis)
    with torch.no_grad():
        for s0 in range(0, N, args.chunk):
            sl = slice(s0, min(N, s0 + args.chunk)); q = {k: v[sl] for k, v in p.items()}
            n = sl.stop - sl.start
            V = torch.full((len(cams), n), float("nan"), device=dev); RT = torch.full_like(V, float("nan"))
            for i, (w2c, (fx, fy, cx, cy, W, H)) in enumerate(cams):
                xz, yz, z, (a, b, c) = ViewerRaster.cov2d(q, w2c, fx, fy)
                u, v = fx * xz + cx, fy * yz + cy
                vis = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                ad, cd = a + dil, c + dil
                mid = 0.5 * (ad + cd); rad = torch.sqrt((0.5 * (ad - cd)) ** 2 + b * b)
                l1 = mid + rad
                culled = vis & (l1 < lmin)
                nvis[sl] += vis.int(); ncull[sl] += culled.int()
                # isotropic 3D variance v adds ~ (f/z)^2 v to both 2D eigenvalues (Jacobian's off-axis terms ignored)
                g = (0.5 * (fx + fy) / z.clamp_min(1e-3)) ** 2
                need = ((target - l1).clamp_min(0) / g)
                V[i] = torch.where(vis, need, V[i])
                dold = ad * cd - b * b
                dnew = (ad + g * need) * (cd + g * need) - b * b
                RT[i] = torch.where(vis, torch.sqrt(dold / dnew.clamp_min(1e-20)), RT[i])
            fix = (nvis[sl] > 0) & (ncull[sl].float() > args.cull_frac * nvis[sl].float())
            vm = torch.nanmedian(V, 0).values; rm = torch.nanmedian(RT, 0).values
            newv[sl] = torch.where(fix, vm.nan_to_num(0), torch.zeros_like(vm))
            ratio[sl] = torch.where(fix, rm.nan_to_num(1), torch.ones_like(rm))
            del V, RT
        fix = newv > 0
        s_old = torch.exp(p["scales"])
        p["scales"] = torch.where(fix[:, None], 0.5 * torch.log(s_old ** 2 + newv[:, None]), p["scales"])
        o = torch.sigmoid(p["opacities"]); o2 = (o * ratio).clamp(1e-6, 1 - 1e-6)
        p["opacities"] = torch.where(fix, torch.log(o2 / (1 - o2)), p["opacities"])
        keep = torch.ones(N, dtype=torch.bool, device=dev)
        if args.prune_opacity > 0:
            keep = torch.sigmoid(p["opacities"]) >= args.prune_opacity
    rep = {"ply": args.ply, "out": args.out, "viewer": args.viewer, "n": N, "train_views": len(cams),
           "never_seen": int((nvis == 0).sum()), "culled_in_majority": int(fix.sum()),
           "share_fixed": round(float(fix.float().mean()), 4),
           "share_culled_any_view": round(float((ncull > 0).float().mean()), 4),
           "opacity_ratio_median_fixed": float(ratio[fix].median()) if fix.any() else None,
           "scale_gain_median_fixed": float((torch.exp(p["scales"][fix]) / s_old[fix]).max(1).values.median()) if fix.any() else None,
           "pruned": int((~keep).sum()), "seconds": round(time.time() - t0, 1)}
    if not bool(keep.all()):
        idx = keep.nonzero()[:, 0]
        p = {k: v[idx] for k, v in p.items()}
        vertex = vertex.data[idx.cpu().numpy()]
    save_ply(args.out, vertex, p)
    json.dump(rep, open(os.path.splitext(args.out)[0] + "_fix.json", "w"), indent=2)
    print(json.dumps(rep, indent=1)); print("VIEWER_FIX_DONE")


if __name__ == "__main__":
    main()
