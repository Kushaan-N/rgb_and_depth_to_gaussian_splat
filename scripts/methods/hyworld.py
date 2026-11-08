"""HY-World 2.0 "World Composition" 3DGS objective for scripts/train_viewer_gsplat.py (see scripts/methods/__init__.py).
HY-World 2.0 (Tencent Hunyuan, arXiv 2604.14268; code github.com/Tencent-Hunyuan/HY-World-2.0,
hyworld2/worldgen/world_gs_trainer.py) optimises its final splat with L1 + 0.2 D-SSIM plus terms whose weights are
linearly interpolated over training, w(step) = l2 + (1 - step / iters) (l1 - l2):
  --hy-lpips l1:l2   LPIPS-VGG on the step's render (official 0.2:0.1, i.e. strong early, half at the end). Inputs
                     are scaled to [-1, 1] (the official code feeds [0, 1] to torchmetrics with normalize=False, which
                     LPIPS treats as a low-contrast image; --hy-lpips-raw reproduces that quirk exactly).
  --hy-normal l1:l2  depth-to-normal loss against WorldMirror 2.0 normals (official 0.1:0.5, i.e. it grows): the
                     render's expected depth is back-projected with the step's K, normals = cross product of the
                     central differences (official Depth2Normal), loss = mean (1 - n_render . n_prior) over pixels where
                     both exist (prior |n|^2 > 0.1, as official). Priors: <--hy-normal-dir>/<image stem>.npy, (H,W,3)
                     float16 unit normals in the OpenCV camera frame (x right, y down, z forward), written by
                     scripts/hyworld_normals.py (WorldMirror 2.0 fed our poses, intrinsics and LiDAR depth as priors);
                     frames without a file are skipped. Signs are made camera-facing on both sides.
Not ported (and why): its disparity L1 to WorldMirror depth (our LiDAR depth priors were measured and rejected; use
--depth-lambda), sky TV (indoor), MaskGaussian (a pruning device, budgets are not the CEAR bottleneck; needs a patched
rasteriser), bilateral grid (an appearance model, rejected). The export stays a standard 3DGS PLY.
"""
from __future__ import annotations
import os


def add_hyworld_args(g):
    g.add_argument("--hy-lpips", default="", help="HY-World LPIPS-VGG weight schedule start:end (official 0.2:0.1; '' = off)")
    g.add_argument("--hy-lpips-raw", action="store_true", help="feed [0,1] images to LPIPS like the official code")
    g.add_argument("--hy-lpips-every", type=int, default=1, help="apply the LPIPS term every N steps (x N weight)")
    g.add_argument("--hy-normal", default="", help="depth-to-normal loss weight schedule start:end (official 0.1:0.5; '' = off)")
    g.add_argument("--hy-normal-dir", default="", help="WorldMirror 2.0 normals (scripts/hyworld_normals.py)")


def _sched(s):
    a, b = (float(x) for x in s.split(":")) if ":" in s else (float(s), float(s))
    return a, b


def depth_to_normal(d, K):
    """d (H,W) z-depth, K (3,3) -> camera-frame normals (H,W,3) (zeros at the border / invalid), valid (H,W)."""
    import torch
    import torch.nn.functional as F
    H, W = d.shape
    v, u = torch.meshgrid(torch.arange(H, device=d.device, dtype=d.dtype) + 0.5,
                          torch.arange(W, device=d.device, dtype=d.dtype) + 0.5, indexing="ij")
    X = torch.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    dx = torch.zeros_like(X); dy = torch.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]; dy[1:-1] = X[2:] - X[:-2]
    n = torch.cross(dx, dy, dim=-1)
    ok = (d > 0); ok = ok & torch.roll(ok, 1, 0) & torch.roll(ok, -1, 0) & torch.roll(ok, 1, 1) & torch.roll(ok, -1, 1)
    ok[0] = ok[-1] = False; ok[:, 0] = ok[:, -1] = False
    n = F.normalize(n, dim=-1)
    n = torch.where((n * X).sum(-1, keepdim=True) > 0, -n, n)          # face the camera
    return n * ok[..., None], ok & (n.norm(dim=-1) > 0.5)


class HYWorld:
    def __init__(self, args, dev="cuda"):
        self.a, self.iters = args, max(args.iters, 1)
        self.lp = _sched(args.hy_lpips) if args.hy_lpips else None
        self.nm = _sched(args.hy_normal) if args.hy_normal else None
        self.net, self.prior, self.acc = None, {}, {"lpips": [0.0, 0], "normal": [0.0, 0], "normal_frames": 0}
        self.need_depth = self.nm is not None
        if self.lp:
            from torchmetrics.functional.image.lpips import _NoTrainLpips
            self.net = _NoTrainLpips(net="vgg").to(dev).eval()
            for q in self.net.parameters():
                q.requires_grad_(False)
        if self.nm:
            if not os.path.isdir(args.hy_normal_dir):
                raise SystemExit(f"[methods] --hy-normal needs --hy-normal-dir (scripts/hyworld_normals.py): {args.hy_normal_dir!r}")
            self.files = {os.path.splitext(f)[0]: os.path.join(args.hy_normal_dir, f)
                          for f in os.listdir(args.hy_normal_dir) if f.endswith(".npy")}
            print(f"[methods] HY-World normal priors: {len(self.files)} frames in {args.hy_normal_dir}", flush=True)
        if self.lp or self.nm:
            print(f"[methods] HY-World objective: lpips {self.lp} (raw {args.hy_lpips_raw}), normal {self.nm}", flush=True)

    def w(self, s, step):
        return s[1] + (1 - step / self.iters) * (s[0] - s[1])

    def _prior(self, name, H, W, dev):
        import numpy as np
        import torch
        import torch.nn.functional as F
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem not in self.prior:
            f = self.files.get(stem)
            self.prior[stem] = None if f is None else torch.from_numpy(np.load(f).astype(np.float16)).to(dev)
        n = self.prior[stem]
        if n is None:
            return None
        n = n.float()
        if n.shape[:2] != (H, W):
            n = F.interpolate(n.permute(2, 0, 1)[None], size=(H, W), mode="nearest")[0].permute(1, 2, 0)
        return n

    def loss(self, img, gt, step, depth=None, view=None):
        import torch
        tot = img.new_zeros(())
        if self.lp and step % max(self.a.hy_lpips_every, 1) == 0:
            x, y = img.clamp(0, 1).permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
            if not self.a.hy_lpips_raw:
                x, y = x * 2 - 1, y * 2 - 1
            l = self.net(x, y).mean()
            tot = tot + self.w(self.lp, step) * max(self.a.hy_lpips_every, 1) * l
            if step % 100 == 0:
                self.acc["lpips"][0] += float(l); self.acc["lpips"][1] += 1
        if self.nm and depth is not None and view is not None:
            n0 = self._prior(view[0], depth.shape[0], depth.shape[1], depth.device)
            if n0 is not None:
                K = view[1] * torch.tensor([[depth.shape[1] / view[2]], [depth.shape[0] / view[3]], [1.0]], device=depth.device)
                n, ok = depth_to_normal(depth, K)
                ok = ok & ((n0 * n0).sum(-1) > 0.1)
                if bool(ok.any()):
                    n0 = torch.nn.functional.normalize(n0, dim=-1)
                    u, v = torch.meshgrid(torch.arange(depth.shape[1], device=depth.device) + 0.5,
                                          torch.arange(depth.shape[0], device=depth.device) + 0.5, indexing="xy")
                    ray = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
                    n0 = torch.where((n0 * ray).sum(-1, keepdim=True) > 0, -n0, n0)   # face the camera, like n
                    l = (1 - (n * n0).sum(-1))[ok].mean()
                    tot = tot + self.w(self.nm, step) * l
                    if step % 100 == 0:
                        self.acc["normal"][0] += float(l); self.acc["normal"][1] += 1
                    self.acc["normal_frames"] += 1
        return tot

    def summary(self):
        r = {k: round(s / n, 5) for k, v in self.acc.items() if isinstance(v, list) for s, n in [v] if n}
        if self.nm:
            r["normal_steps"] = self.acc["normal_frames"]
        return r
