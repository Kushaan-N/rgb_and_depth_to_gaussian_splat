"""Rendered depth + accumulated opacity of a standard 3DGS PLY at every camera of a COLMAP text dataset (gsplat venv,
GPU) -> <out-dir>/<image stem>_splatdepth.npy (ray distance, dataset units, 0 where opacity < 0.5) and
_splatopacity.npy: the same files eval_views.py --save-depth-dir writes for 3DGRUT checkpoints, so
fill_unseen_surfaces.py --splat-depth-dir works for splats trained in any framework.

    python scripts/render_ply_depth.py --ply splat.ply --dataset <COLMAP text dataset, PLY's frame> --out-dir DIR
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model
from eval_ply import load_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import torch
    from gsplat import rasterization
    dev = "cuda"
    G = load_ply(args.ply, dev)
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, cid, q, t in sorted(imgs, key=lambda i: i[0]):
        W, H, (fx, fy, cx, cy) = cams[cid]
        V = np.eye(4); V[:3, :3] = quat_to_R(q); V[:3, 3] = t
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with torch.no_grad():
            ed, alpha, _ = rasterization(G["means"], G["quats"], G["scales"], G["opacities"], G["sh"][:, :1],
                                         torch.tensor(V, dtype=torch.float32, device=dev)[None],
                                         torch.tensor(K, dtype=torch.float32, device=dev)[None], W, H,
                                         sh_degree=0, render_mode="ED", rasterize_mode="classic")
        z = ed[0, ..., 0]; a = alpha[0, ..., 0]
        u = (torch.arange(W, device=dev) + 0.5 - cx) / fx; v = (torch.arange(H, device=dev) + 0.5 - cy) / fy
        ray = torch.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)            # z-depth -> distance along the ray
        d = torch.where(a > 0.5, z * ray, torch.zeros_like(z))
        stem = os.path.join(args.out_dir, os.path.splitext(name)[0])
        np.save(stem + "_splatdepth.npy", d.half().cpu().numpy())
        np.save(stem + "_splatopacity.npy", a.clamp(0, 1).half().cpu().numpy())
    print(f"[ply-depth] {len(imgs)} views -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
