"""Sharpness-aware training for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --sam-lambda.
FASR (Yun et al., 2025, arXiv 2511.17918) shows 3DGS overfits the training directions into SHARP minima and that a
sharpness-aware objective (SAM, Foret et al. ICLR 2021) generalises better to novel views. Full SAM needs a second
backward per step; here the ascent direction is Adam's own preconditioned momentum (Momentum-SAM, Becker et al. 2022):
eps_k = --sam-rho x lr_k x m_k / (sqrt(v_k) + 1e-15), i.e. rho Adam steps UPHILL, per parameter group, scale-free.
With prob --sam-prob per step from --sam-start x iters (default after densification), the step's view is rendered
again at theta + eps (same SH degree, random background) and --sam-lambda x (L1 + 0.2 D-SSIM) of that render is added,
so the optimiser minimises L(theta) + lambda L(theta + eps): low loss in a neighbourhood, not at a point. FASR's
frequency adaptivity is approximated by --sam-shn-only off/on (on: only view-dependent SH, opacities and scales are
perturbed, not means / sh0, which carry the high-frequency detail). The exported PLY is untouched.
"""
from __future__ import annotations


def add_sam_args(g):
    g.add_argument("--sam-lambda", type=float, default=0.0, help="weight of L(theta + eps) (0 = off)")
    g.add_argument("--sam-rho", type=float, default=5.0, help="perturbation size in Adam steps")
    g.add_argument("--sam-prob", type=float, default=0.5, help="fraction of steps with the perturbed render")
    g.add_argument("--sam-start", type=float, default=0.5, help="fraction of iters before which it is off")
    g.add_argument("--sam-groups", default="all", help="comma list of perturbed groups, or 'all' (e.g. shN,opacities,scales)")


class SharpAware:
    def __init__(self, args, rast, params, opts):
        self.a, self.on = args, args.sam_lambda > 0 and rast is not None and opts is not None
        self.rast, self.params, self.opts, self.n, self.acc = rast, params, opts, 0, 0.0
        self.groups = None if args.sam_groups == "all" else set(args.sam_groups.split(","))
        if self.on:
            print(f"[methods] sharpness-aware (momentum SAM): lambda {args.sam_lambda}, rho {args.sam_rho} Adam steps, prob "
                  f"{args.sam_prob}, from {args.sam_start} x iters, groups {args.sam_groups}", flush=True)

    def loss(self, view, gt, step, deg):
        import numpy as np
        import torch
        a = self.a
        if not self.on or view is None or step < a.sam_start * a.iters or np.random.rand() >= a.sam_prob:
            return None
        from refine_poses_gsplat import ssim
        pe = {}
        with torch.no_grad():
            for k, p in self.params.items():
                o = self.opts[k]; st = o.state.get(o.param_groups[0]["params"][0], {})
                m, v = st.get("exp_avg"), st.get("exp_avg_sq")
                if (self.groups is None or k in self.groups) and m is not None and m.shape == p.shape:
                    pe[k] = a.sam_rho * o.param_groups[0]["lr"] * m / (v.sqrt() + 1e-15)
        if not pe:
            return None
        q = {k: (p + pe[k]) if k in pe else p for k, p in self.params.items()}
        img, alpha, _, _ = self.rast.render(q, view, deg)
        img = img + (1 - alpha[..., None]) * torch.rand(3, device=img.device)
        l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        self.n += 1; self.acc += float(l.detach())
        return a.sam_lambda * l

    def summary(self):
        return {"sam_steps": self.n, "sam_loss": round(self.acc / self.n, 5)} if self.n else {}
