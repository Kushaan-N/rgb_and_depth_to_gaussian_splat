#!/usr/bin/env python3
"""Surface depth of a standard 3DGS PLY at the PINHOLE cameras of a COLMAP text dataset, for geometry scoring (gsplat
venv, GPU) -> <out-dir>/<stem>_splatdepth.npy (ray distance, float16, 0 where the surface opacity < --min-alpha).

Unlike scripts/render_ply_depth.py (expected depth of ALL gaussians), gaussians whose largest axis exceeds --max-scale
(metres; default 1) are left out: outdoor splats model the sky with a few hundred huge (10-1800 m) gaussians, some
centred within metres of the cameras. SuperSplat caps their screen radius, so they are invisible there, but every
uncapped rasteriser (stock gsplat, 3DGRUT, Isaac) draws them as near-camera fog, and their depth swamps the
expected depth (ROQ-01: ~4 m everywhere). The TLS has no sky, so those are not surface; the fraction removed is
printed. Depth = expected depth of the remaining gaussians, kept where their accumulated alpha >= --min-alpha.

    python scripts/datasets/spires_render_depth.py --ply S.ply --dataset <pinhole set> --out-dir D [--max-scale 1]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402
from eval_ply import load_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-scale", type=float, default=1.0); ap.add_argument("--min-alpha", type=float, default=0.5)
    a = ap.parse_args()
    import torch
    from gsplat import rasterization
    G = load_ply(a.ply, "cuda"); keep = G["scales"].max(1).values <= a.max_scale
    print(f"[sdepth] {int((~keep).sum())} of {len(keep)} gaussians wider than {a.max_scale} m left out", flush=True)
    G = {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == keep.shape else v) for k, v in G.items()}
    cams, imgs = read_text_model(os.path.join(a.dataset, "sparse", "0")); os.makedirs(a.out_dir, exist_ok=True)
    for name, cid, q, t in imgs:
        W, H, (fx, fy, cx, cy) = cams[cid]
        V = np.eye(4); V[:3, :3] = quat_to_R(q); V[:3, 3] = t
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with torch.no_grad():
            ed, alpha, _ = rasterization(G["means"], G["quats"], G["scales"], G["opacities"], G["sh"][:, :1],
                                         torch.tensor(V, dtype=torch.float32, device="cuda")[None],
                                         torch.tensor(K, dtype=torch.float32, device="cuda")[None], W, H,
                                         sh_degree=0, render_mode="ED", rasterize_mode="classic")
        z = ed[0, ..., 0]; al = alpha[0, ..., 0]
        u = (torch.arange(W, device="cuda") + 0.5 - cx) / fx; v = (torch.arange(H, device="cuda") + 0.5 - cy) / fy
        d = torch.where(al >= a.min_alpha, z * torch.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2), torch.zeros_like(z))
        np.save(os.path.join(a.out_dir, os.path.splitext(name)[0] + "_splatdepth.npy"), d.half().cpu().numpy())
    print(f"[sdepth] {len(imgs)} views -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
