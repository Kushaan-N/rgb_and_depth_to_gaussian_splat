"""DropGaussian for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --drop-rate r.
DropGaussian (Park et al., CVPR 2025, arXiv 2504.00773; also DropoutGS, CVPR 2025): during training every render
drops each gaussian with probability r(step) and divides the opacity of the survivors by (1 - r(step)), so the expected
coverage is unchanged. A pixel can then never rely on one exact stack of splats: gaussians behind / around the front
ones receive gradient too and every subset has to explain the image, which regularises the fit toward configurations
that also hold between the training views (the generalisation gap: CEAR training views 40+ dB vs held-out 36 dB). The
rate ramps linearly 0 -> r over [--drop-start, --drop-start + --drop-ramp] (fractions of iters, the paper's progressive
schedule) and is 0 again from --drop-stop (so the last stretch converges on the full splat that is exported). Applied
only to the gradient-carrying training renders (main view, --vwarp virtual views); evals, EAS edge weights and
contribution counts see the full splat. The exported PLY is untouched (standard 3DGS).
"""
from __future__ import annotations
import types


def add_dropgs_args(g):
    g.add_argument("--drop-rate", type=float, default=0.0, help="DropGaussian max drop probability (0 = off; paper 0.1)")
    g.add_argument("--drop-start", type=float, default=0.0, help="fraction of iters where the ramp starts")
    g.add_argument("--drop-ramp", type=float, default=0.5, help="ramp length (fraction of iters) to --drop-rate")
    g.add_argument("--drop-stop", type=float, default=0.9, help="fraction of iters from which nothing is dropped")


class DropGaussian:
    def __init__(self, args, rast):
        self.a, self.iters, self.step_, self.n, self.dropped = args, max(args.iters, 1), 0, 0, 0.0
        self.on = args.drop_rate > 0 and rast is not None
        if not self.on:
            return
        orig_project, orig_render, me = rast.project, rast.render, self
        state = {"live": False}

        def render(self_, *a, **k):                          # only gradient-carrying renders drop
            import torch
            state["live"] = torch.is_grad_enabled() and me.rate() > 0
            try:
                return orig_render(*a, **k)
            finally:
                state["live"] = False

        def project(self_, *a, **k):
            import torch
            radii, means2d, depths, conics, opr = orig_project(*a, **k)
            if not state["live"]:
                return radii, means2d, depths, conics, opr
            r = me.rate()
            keep = torch.rand(opr.shape[-1], device=opr.device) >= r
            me.n += 1; me.dropped += 1 - float(keep.float().mean())
            radii = radii * keep[None, :, None].to(radii.dtype)          # dropped: not rasterised, no gradient
            return radii, means2d, depths, conics, opr * keep[None].to(opr.dtype) / (1 - r)

        rast.render = types.MethodType(render, rast)
        rast.project = types.MethodType(project, rast)
        print(f"[methods] DropGaussian: rate 0 -> {args.drop_rate} over [{args.drop_start}, {args.drop_start + args.drop_ramp}] "
              f"x iters, off from {args.drop_stop}", flush=True)

    def rate(self):
        a, f = self.a, self.step_ / self.iters
        if f >= a.drop_stop or f < a.drop_start:
            return 0.0
        return a.drop_rate * min(1.0, (f - a.drop_start) / max(a.drop_ramp, 1e-9))

    def step(self, step):
        self.step_ = step

    def summary(self):
        return {"drop_renders": self.n, "drop_mean_frac": round(self.dropped / self.n, 4)} if self.n else {}
