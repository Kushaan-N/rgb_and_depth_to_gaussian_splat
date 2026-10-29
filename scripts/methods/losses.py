"""Extra photometric loss terms for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py). img / gt: (H,W,3)
in [0,1], the render after background / appearance / mask handling, i.e. exactly what the L1 + D-SSIM terms see.
  FreGS frequency loss (Zhang et al., CVPR 2024, arXiv 2403.06908): amplitude L1 + complex-spectrum L1 (amplitude AND phase
    mismatch, smooth where a plain angle() L1 is singular at zero amplitude) of the orthonormal 2D FFT, under a radial
    low-pass |f| <= c(step) (normalised so 1 = the corner frequency) opening linearly from --freq-c0 at step 0 to 1 at
    --freq-open x iters: low frequencies first, then the full band (FreGS' progressive annealing).
  hard-pixel mining: lambda x mean(w |img - gt|), w = (e / mean e) ^ gamma capped, e = detached 3x3-blurred error.
  image-gradient loss: lambda x L1 between the Sobel x/y gradients of render and target (per channel).
  multi-scale D-SSIM: lambda x mean over --msssim-scales dyadic scales (2x2 area pooling) of (1 - SSIM): structure at
    coarser scales too (an additive, MS-SSIM-like term; the base L1 + D-SSIM stays).
"""
from __future__ import annotations


def add_loss_args(g):
    g.add_argument("--freq-lambda", type=float, default=0.0, help="FreGS frequency loss weight (0 = off)")
    g.add_argument("--freq-c0", type=float, default=0.15, help="initial low-pass cut-off (fraction of the corner frequency)")
    g.add_argument("--freq-open", type=float, default=0.5, help="fraction of iters at which the full band is open")
    g.add_argument("--hard-lambda", type=float, default=0.0, help="hard-pixel (error-weighted) L1 weight (0 = off)")
    g.add_argument("--hard-gamma", type=float, default=1.0)
    g.add_argument("--hard-cap", type=float, default=8.0)
    g.add_argument("--grad-lambda", type=float, default=0.0, help="image-gradient (Sobel) L1 weight (0 = off)")
    g.add_argument("--msssim-lambda", type=float, default=0.0, help="multi-scale D-SSIM weight (0 = off)")
    g.add_argument("--msssim-scales", type=int, default=3)


class ExtraLosses:
    def __init__(self, args):
        self.a = args; self.iters = max(args.iters, 1); self._rad = {}; self._sob = None
        self.acc = {"freq": [0.0, 0], "hard": [0.0, 0], "grad": [0.0, 0], "msssim": [0.0, 0]}

    def _radius(self, H, W, dev):
        import torch
        if (H, W) not in self._rad:
            fy = torch.fft.fftfreq(H, device=dev).abs() / 0.5; fx = torch.fft.rfftfreq(W, device=dev) / 0.5
            self._rad[(H, W)] = torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2) / 2 ** 0.5        # [0, 1]
        return self._rad[(H, W)]

    def _rec(self, k, v):
        self.acc[k][0] += float(v); self.acc[k][1] += 1

    def __call__(self, img, gt, step):
        import torch
        import torch.nn.functional as F
        a, tot = self.a, img.new_zeros(())
        if a.freq_lambda > 0:
            c = min(1.0, a.freq_c0 + (1 - a.freq_c0) * step / max(a.freq_open * self.iters, 1))
            Fr = torch.fft.rfft2(img.permute(2, 0, 1), norm="ortho"); Fg = torch.fft.rfft2(gt.permute(2, 0, 1), norm="ortho")
            m = (self._radius(img.shape[0], img.shape[1], img.device) <= c).float()
            n = m.sum() * 3
            amp = ((Fr.abs() - Fg.abs()).abs() * m).sum() / n
            cpx = (torch.sqrt((Fr - Fg).real ** 2 + (Fr - Fg).imag ** 2 + 1e-12) * m).sum() / n
            lf = amp + cpx; tot = tot + a.freq_lambda * lf
            if step % 100 == 0: self._rec("freq", lf.detach())
        if a.hard_lambda > 0:
            e = (img - gt).abs().mean(-1)
            with torch.no_grad():
                eb = F.avg_pool2d(e[None, None], 3, 1, 1)[0, 0]
                w = (eb / eb.mean().clamp_min(1e-8)).pow(a.hard_gamma).clamp_max(a.hard_cap)
            lh = (w * e).mean(); tot = tot + a.hard_lambda * lh
            if step % 100 == 0: self._rec("hard", lh.detach())
        if a.grad_lambda > 0:
            if self._sob is None:
                k = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device) / 8
                self._sob = torch.stack([k, k.T])[:, None].repeat(3, 1, 1, 1)                 # (6,1,3,3), groups=3
            gi = F.conv2d(img.permute(2, 0, 1)[None], self._sob, padding=1, groups=3)
            gg = F.conv2d(gt.permute(2, 0, 1)[None], self._sob, padding=1, groups=3)
            lg = (gi - gg).abs().mean(); tot = tot + a.grad_lambda * lg
            if step % 100 == 0: self._rec("grad", lg.detach())
        if a.msssim_lambda > 0:
            from refine_poses_gsplat import ssim
            x, y, ls = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], []
            for s in range(a.msssim_scales):
                if s:
                    x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
                if min(x.shape[-2:]) < 16:
                    break
                ls.append(1 - ssim(x, y))
            lm = torch.stack(ls).mean(); tot = tot + a.msssim_lambda * lm
            if step % 100 == 0: self._rec("msssim", lm.detach())
        return tot

    def summary(self):
        return {k: round(s / n, 6) for k, (s, n) in self.acc.items() if n}
