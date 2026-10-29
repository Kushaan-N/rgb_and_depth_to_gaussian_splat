"""Blur-aware training for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py): --blur-kernel K.
Real frames of a moving robot carry motion / defocus blur that differs per frame; fitting a sharp-capable splat to them
directly averages the blur into the scene (soft textures everywhere). In the spirit of Deblurring 3DGS (Lee et al.,
ECCV 2024) and BAD-Gaussians / Deblur-GS (2024), but kept lightweight and standard-PLY compatible: every training frame
gets a learned K x K blur kernel (softmax over K^2 logits, initialised to a near-delta), applied to the RENDER before the
loss only. The splat is pushed to explain the sharp scene; the kernel explains the frame's blur. The exported PLY is
unchanged in format and is rendered without any kernel. Regularisers keep it a blur, not a shift or a sharpening:
--blur-centroid (L2 on the kernel's centre of mass, px: no image translation / pose drift into the kernel) and
--blur-reg (L2 pull of the logits toward the delta initialisation). Kernels start at --blur-start x iters.
Note: held-out frames are blurry too, so a sharper splat can score LOWER PSNR there while looking better; judge with
LPIPS / DISTS and renders (accept_gate --perceptual), and off-path views.
"""
from __future__ import annotations


def add_blur_args(g):
    g.add_argument("--blur-kernel", type=int, default=0, help="per-frame learned blur kernel size (odd; 0 = off)")
    g.add_argument("--blur-lr", type=float, default=5e-3,
                   help="Adam lr of the logits; each frame's kernel is only stepped when that frame is drawn (~iters / n_frames "
                        "times), so with hundreds of frames use ~5e-2 (5e-3 left the kernels at the delta on CEAR)")
    g.add_argument("--blur-centroid", type=float, default=0.1, help="L2 weight on the kernel centroid offset (px)")
    g.add_argument("--blur-reg", type=float, default=1e-4, help="L2 pull of the logits toward the delta init")
    g.add_argument("--blur-start", type=float, default=0.1, help="fraction of iters where kernels start being used")


class FrameBlur:
    def __init__(self, args, names, dev):
        import torch
        self.on = args.blur_kernel > 0
        if not self.on:
            return
        K = args.blur_kernel | 1; self.K, self.a, self.iters = K, args, max(args.iters, 1)
        self.idx = {n: i for i, n in enumerate(names)}
        init = torch.full((len(names), K * K), -4.0, device=dev); init[:, (K * K) // 2] = 4.0
        self.init = init.clone()
        self.logits = torch.nn.Parameter(init)
        self.opt = torch.optim.Adam([self.logits], lr=args.blur_lr)
        r = torch.arange(K, device=dev, dtype=torch.float32) - K // 2
        self.yy, self.xx = torch.meshgrid(r, r, indexing="ij")
        self.used, self.last = 0, None

    def kernel(self, i):
        import torch
        return torch.softmax(self.logits[i], 0).reshape(self.K, self.K)

    def __call__(self, img, name, step):
        """(H,W,3) render -> blurred render (training only) and the regulariser term."""
        import torch
        import torch.nn.functional as F
        if not self.on or step < self.a.blur_start * self.iters or name not in self.idx:
            self.last = None
            return img, None
        i = self.idx[name]; k = self.kernel(i)
        x = img.permute(2, 0, 1)[None]
        x = F.conv2d(F.pad(x, [self.K // 2] * 4, mode="replicate"), k[None, None].repeat(3, 1, 1, 1), groups=3)
        cy, cx = (k * self.yy).sum(), (k * self.xx).sum()
        reg = self.a.blur_centroid * (cx * cx + cy * cy) + self.a.blur_reg * ((self.logits[i] - self.init[i]) ** 2).sum()
        self.used += 1; self.last = i
        return x[0].permute(1, 2, 0), reg

    def step(self):
        if self.on and self.last is not None:
            self.opt.step(); self.opt.zero_grad(set_to_none=True)

    def summary(self):
        import torch
        if not self.on:
            return {}
        with torch.no_grad():
            c = torch.softmax(self.logits, 1)[:, (self.K * self.K) // 2]
        q = [round(float(v), 4) for v in torch.quantile(c, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=c.device))]
        return {"blur_kernel": self.K, "steps_used": self.used, "centre_weight_pct_0_10_50_90_100": q}
