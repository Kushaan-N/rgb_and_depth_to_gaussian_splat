#!/usr/bin/env python3
"""scripts/train_viewer_gsplat.py with a per-frame BILATERAL GRID appearance model instead of its global gain/affine
(gsplat venv + viewer kernels, GPU). Wrapper only: train_viewer_gsplat.py is imported unchanged and its `Appearance`
class is swapped for a bilateral-grid one with the same interface, so every other flag/behaviour is identical.

Why: outdoor auto-exposure + ISP (tone curve, local adaptation, vignetting) vary across the image, which one 3x4 affine
per frame cannot absorb (Wang et al., "Bilateral Guided Radiance Field Processing", SIGGRAPH 2024; gsplat's
examples/lib_bilagrid.py). A 16x16x8 grid of affine colour transforms per training frame, sliced by pixel position and
luminance, is applied to the render before the loss (render side only, like --appearance affine), with gsplat's
total-variation regulariser (--bilagrid-tv, default 10). It is training-only: the exported PLY is a standard splat
(held-out frames and the viewer never see the grids). Grids -> <out>/bilagrid.pt.

    PYTHONPATH=$CEAR_WS/gsplat-viewer-<prof> python scripts/datasets/spires_train_bilagrid.py <train_viewer_gsplat.py args> \
        --appearance bilagrid [--bilagrid-tv 10 --bilagrid-lr 2e-3 --bilagrid-shape 16 16 8]
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
BILAGRID_LIB = os.environ.get("BILAGRID_LIB", os.path.join(os.environ.get("CEAR_WS", ""), "gsplat", "examples"))


def _pop(argv, flag, n=1, default=None):
    if flag in argv:
        i = argv.index(flag); v = argv[i + 1:i + 1 + n]; del argv[i:i + 1 + n]
        return v if n > 1 else v[0]
    return default


def main():
    argv = sys.argv[1:]
    tv = float(_pop(argv, "--bilagrid-tv", default=10.0)); lr = float(_pop(argv, "--bilagrid-lr", default=2e-3))
    shape = [int(x) for x in _pop(argv, "--bilagrid-shape", 3, ["16", "16", "8"])]
    use = False
    if "--appearance" in argv and argv[argv.index("--appearance") + 1] == "bilagrid":
        argv[argv.index("--appearance") + 1] = "affine"; use = True       # trainer's choices; class swapped below
    import train_viewer_gsplat as T
    if use:
        sys.path.insert(0, BILAGRID_LIB)
        from lib_bilagrid import BilateralGrid
        out_dir = _pop(list(argv), "--out")

        class BilagridAppearance:
            def __init__(self, mode, n, app_lr, dev):
                import torch
                self.grid = BilateralGrid(n, grid_X=shape[0], grid_Y=shape[1], grid_W=shape[2]).to(dev)
                self.opt = torch.optim.Adam(self.grid.parameters(), lr=lr, eps=1e-15); self._xy = {}

            def __call__(self, img, k):
                import torch
                H, W = img.shape[:2]
                if (H, W) not in self._xy:
                    yy, xx = torch.meshgrid((torch.arange(H, device=img.device) + 0.5) / H,
                                            (torch.arange(W, device=img.device) + 0.5) / W, indexing="ij")
                    self._xy[(H, W)] = torch.stack([xx, yy], -1)[None]
                out = self.grid(self._xy[(H, W)], img[None], torch.tensor([k], device=img.device))[0]
                # the trainer adds app_reg * (w*w).sum(): make that equal to tv x total-variation of the grids
                reg = (tv * self.grid.tv_loss() / max(T._APP_REG, 1e-12)).clamp_min(0).sqrt()
                return out, reg[None]

            def recentre(self):
                pass

            def summary(self, names):
                import torch
                if out_dir:
                    torch.save(self.grid.state_dict(), os.path.join(out_dir, "bilagrid.pt"))
                z = [0.0] * 7
                return {"mode": f"bilagrid{shape}", "gain_pct_0_5_25_50_75_95_100": {"lum": z}, "gain_std": [0, 0, 0],
                        "bias_std": [0, 0, 0], "tv": float(self.grid.tv_loss())}

        T.Appearance = BilagridAppearance
        T._APP_REG = float(_pop(list(argv), "--app-reg", default=0.01))
    sys.argv = [os.path.join(HERE, "..", "train_viewer_gsplat.py")] + argv
    T.main()


if __name__ == "__main__":
    main()
