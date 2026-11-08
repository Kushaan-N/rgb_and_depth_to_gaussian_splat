"""Sub-pixel camera jitter for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --spj-px j.
Every training camera is always rendered at exactly its own pixel grid, so the splat may encode the scene only at those
sample positions (pixel-locked high-frequency splats / aliasing), while any other camera -- the held-out frames, and
the live viewer, whose centred canvas already shifts the principal point by ~0.2 px (scripts/register_gt.py: worth
~1.4 dB on CEAR) -- samples it in between. With probability --spj-prob a step shifts the camera's principal point by
d ~ U(-j, j)^2 px and the target by the same d with a separable Lanczos-3 resampler (near-ideal passband, so the target
is not blurred the way a bilinear shift would blur it): the splat has to reproduce the band-limited continuous image,
not just its samples. Off from --spj-stop x iters (the last stretch converges on the exact grid) and on frames with
transient masks. Depth targets are dropped on jittered steps. The export is a standard 3DGS PLY.
"""
from __future__ import annotations


def add_subpix_args(g):
    g.add_argument("--spj-px", type=float, default=0.0, help="sub-pixel principal-point jitter amplitude in px (0 = off; e.g. 0.5)")
    g.add_argument("--spj-prob", type=float, default=0.5, help="fraction of steps that are jittered")
    g.add_argument("--spj-stop", type=float, default=0.9, help="fraction of iters from which no step is jittered")


def _lanczos_taps(t, a=3):
    """1D taps for sampling at offset -t (t in px): out[x] = sum_k w_k in[x + k], k = -a+1..a (t in (-1, 1))."""
    import math
    ks = list(range(-a, a + 1))
    L = lambda x: 1.0 if abs(x) < 1e-8 else (a * math.sin(math.pi * x) * math.sin(math.pi * x / a) / (math.pi * x) ** 2 if abs(x) < a else 0.0)
    w = [L(k + t) for k in ks]
    s = sum(w)
    return [x / s for x in w]


def shift_image(img, dx, dy):
    """img (H,W,C) float; out(u) = img(u - d) (content moves by +d), reflect padding."""
    import torch
    import torch.nn.functional as F
    x = img.permute(2, 0, 1)[None]
    C = x.shape[1]
    for t, dim in ((dx, 3), (dy, 2)):
        w = torch.tensor(_lanczos_taps(t), device=img.device, dtype=img.dtype)
        k = w.view(1, 1, 1, -1) if dim == 3 else w.view(1, 1, -1, 1)
        pad = (3, 3, 0, 0) if dim == 3 else (0, 0, 3, 3)
        x = F.conv2d(F.pad(x, pad, mode="reflect"), k.repeat(C, 1, 1, 1), groups=C)
    return x[0].permute(1, 2, 0)


class SubPixJitter:
    def __init__(self, args):
        self.a, self.iters, self.n, self.on = args, max(args.iters, 1), 0, args.spj_px > 0
        if self.on:
            print(f"[methods] sub-pixel jitter: +-{args.spj_px} px, prob {args.spj_prob}, off from {args.spj_stop} x iters", flush=True)

    def view(self, v, step, masked=False):
        import random
        a = self.a
        if not self.on or masked or step >= a.spj_stop * self.iters or random.random() >= a.spj_prob:
            return v
        dx, dy = random.uniform(-a.spj_px, a.spj_px), random.uniform(-a.spj_px, a.spj_px)
        K = v[1].clone(); K[0, 2] += dx; K[1, 2] += dy
        gt = shift_image(v[5].float(), dx, dy).clamp(0, 255)
        self.n += 1
        return (v[0], K, v[2], v[3], v[4], gt, None) + tuple(v[7:])

    def summary(self):
        return {"jittered_steps": self.n} if self.n else {}
