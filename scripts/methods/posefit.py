"""Joint training-camera pose refinement for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --pose-lr.
Every training frame gets an SE(3) correction in its camera frame, viewmat <- [R(w) | t] @ viewmat (w axis-angle rad,
t = --pose units x scene_scale), learned with the splat in the viewer-matched rasteriser (gradients through gsplat's
projection and the SH view directions) from --pose-start x iters, SparseAdam lr --pose-lr decaying 10x to the end, L2
pull --pose-reg toward 0. Per-frame pose noise (time-sync, rolling jitter, BA residuals) otherwise gets baked into
the splat as blur / doubled edges / view-dependent colour that only fits each frame (posefix: +0.3-0.5 dB on FP2
legged, scripts/refine_poses_gsplat.py, but as a separate pre-pass in another rasteriser). Gauge: the mean correction
over all training frames is removed after every step, so the scene stays in the frame of the (unrefined) held-out /
viewer cameras; the export is a standard 3DGS PLY and no camera of the deliverable changes. --pose-hp N (> 0): after every
step the moving average of the corrections over N consecutive training frames (name = time order) is removed too, so only
per-frame JITTER is corrected and the low-frequency trajectory -- which the held-out / viewer cameras between the
training frames share -- stays exactly at the input poses (the mean-only gauge lets the scene drift against them). Corrections ->
methods.json (mm / deg percentiles in dataset units x scene_scale) and <out>/pose_corrections.json.
"""
from __future__ import annotations


def add_posefit_args(g):
    g.add_argument("--pose-lr", type=float, default=0.0, help="per-training-frame pose correction lr (0 = off; e.g. 1e-4)")
    g.add_argument("--pose-start", type=float, default=0.1, help="fraction of iters before which poses are frozen")
    g.add_argument("--pose-reg", type=float, default=1e-3, help="L2 weight on the step's correction")
    g.add_argument("--pose-hp", type=int, default=0, help="high-pass: remove the N-frame moving average of the corrections (0 = mean only)")


def _exp_rot(w):
    import torch
    th = torch.sqrt((w * w).sum() + 1e-20)
    k = w / th; z = torch.zeros((), device=w.device)
    Kx = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]), torch.stack([-k[1], k[0], z])])
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


class PoseFit:
    def __init__(self, args, views, scene_scale, dev):
        self.a, self.on, self.used, self.ss = args, args.pose_lr > 0 and bool(views), None, scene_scale
        if not self.on:
            return
        if args.pose_hp > 0:     # 60k CEAR run (fm_ms0.2_sdual_pose3e-5_poshp9) hit a singular / non-finite viewmat: disabled
            raise SystemExit("[methods] --pose-hp is disabled (non-finite poses at 60k, unfixed; pose refinement measured -0.9 dB anyway)")
        import torch
        self.idx = {v[0]: k for k, v in enumerate(views)}; self.names = [v[0] for v in views]
        self.emb = torch.nn.Embedding(len(views), 6, sparse=True).to(dev)
        torch.nn.init.normal_(self.emb.weight, std=1e-9)                     # ~0, but a defined rotation axis
        self.opt = torch.optim.SparseAdam(self.emb.parameters(), lr=args.pose_lr)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.1 ** (1.0 / max(args.iters, 1)))
        self.start = int(args.pose_start * args.iters)
        print(f"[methods] pose refinement: {len(views)} training frames, lr {args.pose_lr}, from step {self.start}, reg {args.pose_reg}", flush=True)

    def view(self, v, step):
        self.used = None
        if not self.on or step < self.start or v[0] not in self.idx:
            return v
        import torch
        d = self.emb(torch.tensor([self.idx[v[0]]], device=v[4].device))[0]
        E = torch.eye(4, device=d.device)
        E = torch.cat([torch.cat([_exp_rot(d[:3]), (d[3:] * self.ss)[:, None]], 1), E[3:]], 0)
        self.used = d
        return (v[0], v[1], v[2], v[3], E @ v[4]) + tuple(v[5:])

    def loss(self):
        return None if self.used is None or self.a.pose_reg <= 0 else self.a.pose_reg * (self.used * self.used).sum()

    def after_step(self, step):
        if not self.on:
            return
        import torch
        if self.used is not None:
            self.opt.step(); self.opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                W = self.emb.weight; W.nan_to_num_(0.0, 0.0, 0.0); W.sub_(W.mean(0, keepdim=True))
                if self.a.pose_hp > 1:                       # keep only per-frame jitter: remove the trajectory low-pass
                    import torch.nn.functional as F
                    n = self.a.pose_hp | 1
                    lp = F.avg_pool1d(F.pad(W.T[None], (n // 2, n // 2), mode="replicate"), n, 1)[0].T
                    W.sub_(lp)
        if step >= self.start:
            self.sched.step()

    def summary(self, out_dir=None):
        if not self.on:
            return {}
        import json, os
        import numpy as np
        W = self.emb.weight.detach().cpu().numpy()
        deg = np.degrees(np.linalg.norm(W[:, :3], axis=1)); mm = np.linalg.norm(W[:, 3:], axis=1) * self.ss * 1000
        pct = lambda x: [round(float(v), 3) for v in np.percentile(x, [50, 90, 99, 100])]
        if out_dir:
            json.dump({n: [round(float(x), 7) for x in w] for n, w in zip(self.names, W)},
                      open(os.path.join(out_dir, "pose_corrections.json"), "w"))
        return {"pose_deg_p50_90_99_100": pct(deg), "pose_mm_p50_90_99_100": pct(mm)}
