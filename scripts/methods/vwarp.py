"""Virtual-view warp consistency for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --vwarp-lambda.
The held-out frames of a trajectory lie BETWEEN training cameras, and that is where the splat is weakest (CEAR: training
views 40+ dB, held-out 36 dB): view-dependent colour (SH) and semi-transparent stacks are fitted to the exact training
directions and drift in between. Nothing supervises the in-between views directly, but the real images do: with prob
--vwarp-prob per step (from --vwarp-start x iters) a virtual camera is placed on the path between two consecutive
training cameras a, b (lerp centre / slerp rotation at u ~ U(0.15, 0.85), optional random offset up to --vwarp-jitter x
their distance), the splat is rendered there with its depth, and both real frames are backward-warped into it through
that (detached) depth. Where the two warps agree (mean |w_a - w_b| < --vwarp-tau: occlusions, disocclusions, depth
errors and specular glints disagree and are dropped), the render is pulled toward their mean by an L1 of weight
--vwarp-lambda. The target is the photo-consistent (Lambertian) part of real pixels, so it penalises colour that only
exists at the training directions, floaters that only show between views, and blur-free but wrong in-between parallax,
without inventing content. Pairs farther apart than --vwarp-maxgap x the median spacing (trajectory gaps) are skipped.
The exported PLY is untouched (standard 3DGS).
"""
from __future__ import annotations
import math


def add_vwarp_args(g):
    g.add_argument("--vwarp-lambda", type=float, default=0.0, help="virtual-view warp-consistency L1 weight (0 = off)")
    g.add_argument("--vwarp-prob", type=float, default=0.5, help="fraction of steps that add a virtual view")
    g.add_argument("--vwarp-start", type=float, default=0.1, help="fraction of iters before which it is off")
    g.add_argument("--vwarp-tau", type=float, default=0.04, help="max mean |w_a - w_b| (0..1) for a valid pixel")
    g.add_argument("--vwarp-jitter", type=float, default=0.0, help="extra random offset, x the pair's distance")
    g.add_argument("--vwarp-maxgap", type=float, default=3.0, help="skip pairs farther apart than this x median spacing")


def _log_rot(R):
    import torch
    c = ((R.trace() - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7); th = torch.acos(c)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (0.5 / torch.sin(th).clamp_min(1e-9)) * th if float(th) > 1e-6 else 0.5 * w


def _exp_rot(w):
    import torch
    th = w.norm()
    if float(th) < 1e-9:
        return torch.eye(3, device=w.device)
    k = w / th; Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


class VirtualWarp:
    def __init__(self, args, rast, views, params):
        self.a, self.on = args, args.vwarp_lambda > 0 and rast is not None and bool(views)
        self.stats = {"steps": 0, "skipped": 0, "valid_frac": 0.0, "loss": 0.0}
        if not self.on:
            return
        import numpy as np
        import torch
        self.rast, self.views, self.params, self.grid = rast, views, params, {}
        self.c2w = [torch.linalg.inv(v[4]) for v in views]
        C = torch.stack([c[:3, 3] for c in self.c2w]).cpu().numpy()
        gap = np.linalg.norm(np.diff(C, axis=0), axis=1); med = float(np.median(gap))
        self.pairs = [k for k in range(len(views) - 1) if gap[k] <= args.vwarp_maxgap * med and views[k][2:4] == views[k + 1][2:4]]
        print(f"[methods] virtual-view warp consistency: lambda {args.vwarp_lambda}, prob {args.vwarp_prob}, from "
              f"{args.vwarp_start} x iters, tau {args.vwarp_tau}; {len(self.pairs)} camera pairs (median spacing {med:.4f})", flush=True)

    def _pix(self, W, H, dev):
        import torch
        if (W, H) not in self.grid:
            yy, xx = torch.meshgrid(torch.arange(H, device=dev) + 0.5, torch.arange(W, device=dev) + 0.5, indexing="ij")
            self.grid[(W, H)] = (xx, yy)
        return self.grid[(W, H)]

    def loss(self, step, deg):
        import numpy as np
        import torch
        import torch.nn.functional as F
        a = self.a
        if not self.on or step < a.vwarp_start * a.iters or np.random.rand() >= a.vwarp_prob or not self.pairs:
            return None
        k = self.pairs[np.random.randint(len(self.pairs))]
        va, vb = self.views[k], self.views[k + 1]
        ca, cb = self.c2w[k], self.c2w[k + 1]
        u = float(np.random.uniform(0.15, 0.85))
        c = ca.clone()
        c[:3, :3] = ca[:3, :3] @ _exp_rot(u * _log_rot(ca[:3, :3].T @ cb[:3, :3]))
        c[:3, 3] = (1 - u) * ca[:3, 3] + u * cb[:3, 3]
        if a.vwarp_jitter > 0:
            c[:3, 3] += F.normalize(torch.randn(3, device=c.device), dim=0) * a.vwarp_jitter * float(torch.rand(())) * \
                (cb[:3, 3] - ca[:3, 3]).norm()
        K, W, H = va[1], va[2], va[3]
        vv = (va[0] + "#virt", K, W, H, torch.linalg.inv(c), None, None)
        img, alpha, d, _ = self.rast.render(self.params, vv, deg, depth=True)
        with torch.no_grad():
            xx, yy = self._pix(W, H, img.device)
            z = d.detach()
            X = torch.stack([(xx - K[0, 2]) / K[0, 0] * z, (yy - K[1, 2]) / K[1, 1] * z, z], -1)      # virtual cam
            Xw = X @ c[:3, :3].T + c[:3, 3]
            ws, ok = [], alpha.detach() > 0.95
            for v in (va, vb):
                Xs = Xw @ v[4][:3, :3].T + v[4][:3, 3]; zs = Xs[..., 2]
                us = v[1][0, 0] * Xs[..., 0] / zs.clamp_min(1e-6) + v[1][0, 2]
                vs = v[1][1, 1] * Xs[..., 1] / zs.clamp_min(1e-6) + v[1][1, 2]
                ok &= (zs > 1e-3) & (us > 0.5) & (us < v[2] - 0.5) & (vs > 0.5) & (vs < v[3] - 0.5)
                g = torch.stack([2 * us / v[2] - 1, 2 * vs / v[3] - 1], -1)[None]
                ws.append(F.grid_sample(v[5].permute(2, 0, 1)[None].float() / 255, g, mode="bilinear", align_corners=False)[0].permute(1, 2, 0))
            ok &= (ws[0] - ws[1]).abs().mean(-1) < a.vwarp_tau
            tgt = 0.5 * (ws[0] + ws[1])
            m = ok.float()[..., None]; frac = float(ok.float().mean())
        self.stats["steps"] += 1
        if frac < 0.05:
            self.stats["skipped"] += 1
            return None
        l = ((img - tgt).abs() * m).sum() / (m.sum() * 3)
        self.stats["valid_frac"] += frac; self.stats["loss"] += float(l.detach())
        return a.vwarp_lambda * l

    def summary(self):
        s = self.stats; n = max(s["steps"] - s["skipped"], 1)
        return {"vwarp_steps": s["steps"], "vwarp_skipped": s["skipped"], "vwarp_valid_frac": round(s["valid_frac"] / n, 4),
                "vwarp_l1": round(s["loss"] / n, 5)} if s["steps"] else {}
