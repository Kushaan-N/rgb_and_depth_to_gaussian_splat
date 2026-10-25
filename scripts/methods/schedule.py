"""Coarse-to-fine resolution schedule for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py).
  --c2f a: the training view is rendered and compared at 1/4 resolution for the first a/3 x iters, 1/2 until 2a/3 x iters,
    then at full resolution (integer factor: K scaled, real image area-averaged, LiDAR depth nearest-subsampled; frames
    with a transient mask stay at full resolution). Low frequencies / coarse geometry settle first (as in FreGS' annealing
    and the coarse-to-fine schedules of e.g. EAGLES / Mini-Splatting), densification starts from a settled scene, and the
    early iterations are cheaper. Held-out eval and export are unaffected.
"""
from __future__ import annotations


def add_schedule_args(g):
    g.add_argument("--c2f", type=float, default=0.0, help="coarse-to-fine: fraction of iters at reduced resolution (0 = off)")


class ResSchedule:
    def __init__(self, args):
        self.a, self.iters, self._cache, self.used = args.c2f, max(args.iters, 1), {}, {}

    def factor(self, step):
        if self.a <= 0:
            return 1
        f = step / (self.a * self.iters)
        return 4 if f < 1 / 3 else 2 if f < 2 / 3 else 1

    def view(self, v, step, masked=False):
        """(name, K, W, H, viewmat, rgb u8, depth) -> the same tuple at the step's resolution."""
        import torch
        import torch.nn.functional as F
        d = 1 if masked else self.factor(step)
        if d == 1:
            return v
        key = (v[0], d)
        if key not in self._cache:
            if len(self._cache) > 4096:
                self._cache.clear()
            W, H = v[2] // d, v[3] // d
            K = v[1].clone(); K[:2] /= d
            rgb = F.avg_pool2d(v[5][: H * d, : W * d].permute(2, 0, 1)[None].float(), d)[0].permute(1, 2, 0)
            rgb = torch.round(rgb).to(v[5].dtype)
            dep = v[6][d // 2:: d, d // 2:: d][:H, :W].contiguous() if v[6] is not None else None
            self._cache[key] = (v[0], K, W, H, v[4], rgb, dep) + tuple(v[7:])
        self.used[d] = self.used.get(d, 0) + 1
        return self._cache[key]

    def summary(self):
        return {"c2f_steps_per_factor": self.used} if self.used else {}
