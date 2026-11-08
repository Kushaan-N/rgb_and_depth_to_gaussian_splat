"""Sharpness-weighted frames for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --sharp-gamma g.
A trotting quadruped shakes its camera: some frames are motion-blurred (CEAR mocap2: per-frame Laplacian variance
relative to the +-3 neighbouring frames spans 0.77..1.14 (5..95 %)). Every frame pulls the splat equally, so the blurred
ones average the sharp detail away (and densification sees their weak gradients). Here each training frame's photometric
loss is weighted by w_k = clip(r_k ^ g, --sharp-min, --sharp-max) normalised to mean 1, r_k = its sharpness (variance of
the luminance Laplacian) / the median over the --sharp-win frames around it in time (name) order -- relative, so a
textureless view is not mistaken for a blurred one. Implemented as an extra (w_k - 1) x (the trainer's L1 + D-SSIM)
term. Unlike --blur-kernel (learns the blur) or hard-pixel mining (per-pixel error), it decides per frame which
evidence to trust. Weights -> methods.json. The export is a standard 3DGS PLY.
"""
from __future__ import annotations
import os


def add_sharpw_args(g):
    g.add_argument("--sharp-gamma", type=float, default=0.0, help="frame weight = relative sharpness ^ gamma (0 = off; e.g. 2)")
    g.add_argument("--sharp-win", type=int, default=7, help="temporal window (frames) of the sharpness reference median")
    g.add_argument("--sharp-min", type=float, default=0.25)
    g.add_argument("--sharp-max", type=float, default=2.0)


class SharpWeight:
    def __init__(self, args, views):
        self.a, self.w, self.on = args, {}, args.sharp_gamma > 0 and bool(views)
        if not self.on:
            return
        import numpy as np
        import torch
        import torch.nn.functional as F
        lap = torch.tensor([[0.0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=views[0][5].device)[None, None]
        names = [v[0] for v in views]
        s = []
        for v in views:
            y = (v[5].float() @ torch.tensor([0.299, 0.587, 0.114], device=v[5].device))[None, None]
            s.append(float(F.conv2d(y, lap)[0, 0].var()))
        order = sorted(range(len(names)), key=lambda i: names[i])
        s = np.array(s)[order]; h = max(args.sharp_win // 2, 1)
        ref = np.array([np.median(s[max(0, i - h): i + h + 1]) for i in range(len(s))])
        w = np.clip((s / np.maximum(ref, 1e-8)) ** args.sharp_gamma, args.sharp_min, args.sharp_max); w = w / w.mean()
        self.w = {names[order[i]]: float(w[i]) for i in range(len(w))}
        q = np.percentile(w, [0, 5, 50, 95, 100]).round(3).tolist()
        print(f"[methods] sharpness weights (gamma {args.sharp_gamma}): min/5/50/95/max {q}", flush=True)
        self.q = q

    def loss(self, img, gt, name):
        if not self.on or abs(self.w.get(name, 1.0) - 1) < 1e-6:
            return None
        from refine_poses_gsplat import ssim
        a = self.a
        base = (1 - a.ssim_lambda) * (img - gt).abs().mean() + a.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        return (self.w[name] - 1) * base

    def summary(self, out_dir=None):
        if not self.on:
            return {}
        if out_dir:
            import json
            json.dump(self.w, open(os.path.join(out_dir, "sharp_weights.json"), "w"), indent=0)
        return {"weights_pct_0_5_50_95_100": self.q}
