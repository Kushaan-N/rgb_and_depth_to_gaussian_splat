"""End-of-training averaging for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py).
  --ema-decay d: exponential moving average (Polyak averaging) of every gaussian parameter, updated after each optimizer
    step from --ema-start x iters (default 0.6: after Improved-GS densification stops at 0.5, so N is fixed; if N changes
    anyway -- clean-up pruning -- the average is re-seeded from the live parameters). At finalize the averaged values
    replace the live ones (quaternions are re-normalised by the export), so eval and splat.ply use the average.
    Note: Improved-GS' MU steps the optimizer only every 5th / 20th iteration late in training; d is per optimizer step.
  --anneal-start s: every learning rate (sh0, shN, opacities, scales, quats and means on top of its exponential decay)
    multiplied by a cosine factor going 1 -> --anneal-final from s x iters to the last step (chained multiplicatively).
"""
from __future__ import annotations
import math


def add_average_args(g):
    g.add_argument("--ema-decay", type=float, default=0.0, help="parameter EMA decay per optimizer step (0 = off)")
    g.add_argument("--ema-start", type=float, default=0.6, help="fraction of iters where the EMA starts")
    g.add_argument("--anneal-start", type=float, default=0.0, help="fraction of iters where the cosine LR anneal starts (0 = off)")
    g.add_argument("--anneal-final", type=float, default=0.1, help="LR multiplier reached at the last step")


class ParamAverager:
    def __init__(self, args, params, opts):
        self.a, self.p, self.opts, self.iters = args, params, opts, max(args.iters, 1)
        self.shadow, self.n_upd, self.reseeds, self.prev_f = None, 0, 0, 1.0

    def _f(self, step):
        s0 = self.a.anneal_start * self.iters
        if step < s0:
            return 1.0
        t = min(1.0, (step - s0) / max(self.iters - 1 - s0, 1))
        return self.a.anneal_final + (1 - self.a.anneal_final) * 0.5 * (1 + math.cos(math.pi * t))

    def after_step(self, step, stepping):
        import torch
        if self.a.anneal_start > 0:                         # applies to the NEXT step's update
            f = self._f(step + 1); r = f / self.prev_f; self.prev_f = f
            if r != 1.0:
                for o in self.opts.values():
                    for g in o.param_groups:
                        g["lr"] *= r
        if self.a.ema_decay > 0 and stepping and step >= self.a.ema_start * self.iters:
            with torch.no_grad():
                n = len(self.p["means"])
                if self.shadow is None or len(self.shadow["means"]) != n:
                    self.reseeds += self.shadow is not None
                    self.shadow = {k: v.detach().clone() for k, v in self.p.items()}; self.n_upd = 0
                else:
                    d = min(self.a.ema_decay, (1 + self.n_upd) / (10 + self.n_upd))     # warm-up (TF-style)
                    for k, v in self.p.items():
                        self.shadow[k].lerp_(v.detach(), 1 - d)
                self.n_upd += 1

    def finalize(self):
        import torch
        out = {}
        if self.a.anneal_start > 0:
            out["anneal_final_factor"] = round(self.prev_f, 4)
        if self.shadow is not None and len(self.shadow["means"]) == len(self.p["means"]):
            with torch.no_grad():
                drift = {k: round(float((self.p[k] - self.shadow[k]).abs().mean()), 6) for k in self.p}
                for k in self.p:
                    self.p[k].data.copy_(self.shadow[k])
            out.update({"ema_updates": self.n_upd, "ema_reseeds": self.reseeds, "live_minus_ema_mean_abs": drift})
        return out
