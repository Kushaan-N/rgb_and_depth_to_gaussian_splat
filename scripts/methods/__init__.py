"""Optional fidelity methods for scripts/train_viewer_gsplat.py, all default-off (the exported PLY stays a standard 3DGS
splat, SH degree 3; nothing here changes the image formation the viewer uses). The trainer calls, in order:
    methods.add_args(ap)                                   # before parse_args
    fm = methods.Hooks(args, params, opts, strat, scene_scale)   # after the strategy is built (may patch it)
    loss = loss + fm.loss(img, gt, step)                   # extra loss terms on the step's render
    fm.after_step(step, stepping)                          # after the optimizer step (EMA, LR anneal)
    fm.finalize(out_dir)                                   # before the held-out eval / export (EMA swap; methods.json)
Methods (module, flag):
  losses.py     --freq-lambda   FreGS (Zhang et al., CVPR 2024): Fourier amplitude + phase L1 under a radial low-pass
                                whose cut-off opens from --freq-c0 to the full band over --freq-open (coarse -> fine)
                --hard-lambda   hard-pixel mining: extra L1 weighted by the detached, 3x3-blurred, mean-normalised
                                per-pixel error ^ --hard-gamma (capped at --hard-cap): more gradient where it is wrong
                --grad-lambda   image-gradient (Sobel, per channel) L1: edge / structure sharpness
  average.py    --ema-decay     parameter EMA (Polyak) over the optimizer steps from --ema-start (fraction of iters,
                                after densification so N is fixed; re-seeded if N changes), swapped in for eval/export
                --anneal-start  cosine anneal of ALL learning rates (not only means') from that fraction of iters to
                                x --anneal-final at the end (chained, so means' exponential decay still applies)
  densify.py    --pixelgs       Pixel-GS (Zhang et al., ECCV 2024) for the improved strategy: per-view gradient norms
                                averaged with the gaussian's pixel footprint as weight (sum_k pix_k |g_k| / sum_k pix_k)
                                instead of the visible-view count, so large splats seen big in few views can split;
                --pixelgs-depth x: gradient scaled by min(1, z / (x scene_scale)) (near-camera floater damping)
  (here)        --shn-reg       L2 pull of the view-dependent SH bands (shN) of the step's visible gaussians toward 0:
                                less view-dependent overfitting to the training trajectory (train PSNR >> held-out)
"""
from __future__ import annotations
import json, os

from .losses import ExtraLosses, add_loss_args
from .average import ParamAverager, add_average_args
from .densify import add_densify_args, patch_strategy


def add_args(ap):
    g = ap.add_argument_group("fidelity methods (scripts/methods, default off)")
    add_loss_args(g); add_average_args(g); add_densify_args(g)
    g.add_argument("--shn-reg", type=float, default=0.0, help="L2 weight on the visible gaussians' shN (0 = off)")


class Hooks:
    def __init__(self, args, params, opts, strat, scene_scale):
        self.args, self.params = args, params
        self.losses = ExtraLosses(args)
        self.avg = ParamAverager(args, params, opts)
        self.densify = patch_strategy(strat, args, scene_scale)
        on = [k for k in ("freq_lambda", "hard_lambda", "grad_lambda", "ema_decay", "anneal_start", "pixelgs", "shn_reg")
              if getattr(args, k)]
        if on:
            print(f"[methods] on: " + ", ".join(f"{k}={getattr(args, k)}" for k in on), flush=True)

    def loss(self, img, gt, step, info=None):
        l = self.losses(img, gt, step)
        if self.args.shn_reg > 0 and info is not None:
            vis = (info["radii"][0] > 0).all(-1)
            if bool(vis.any()):
                l = l + self.args.shn_reg * self.params["shN"][vis].pow(2).sum((1, 2)).mean()
        return l

    def after_step(self, step, stepping):
        self.avg.after_step(step, stepping)

    def finalize(self, out_dir):
        rec = {"losses": self.losses.summary(), "average": self.avg.finalize(), "densify": self.densify}
        if out_dir and any(v for v in rec.values()):
            json.dump(rec, open(os.path.join(out_dir, "methods.json"), "w"), indent=1)
            print(f"[methods] {json.dumps(rec)}", flush=True)
        return rec
