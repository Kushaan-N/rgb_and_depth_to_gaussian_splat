#!/usr/bin/env python3
"""Option A, stage 1 — render DENSE depth from the trained splat at camera poses (3DGRUT).

The recorded RealSense depth is sparse/grazing, so the fused collider is thin/holey. The splat
is a novel-view renderer with a native ray distance output (`pred_dist`), so we render clean,
dense depth from it — at the recorded poses AND extra rotated ("look-around") poses so obstacle
sides the forward camera missed get covered. Saves per-frame (z-depth, camera-to-world, K) for
CPU TSDF fusion (fuse_splat_collider.py). Runs in the 3DGRUT venv on a GPU.

    python render_splat_depth.py --checkpoint ckpt_last.pt --dataset <dataset_3dgrut> \
        --out <dir> --stride 4 --yaw 25 --limit 0
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch


def rot_yaw_pitch(yaw_deg, pitch_deg):
    """Small camera-frame rotation (about the camera's up=Y then right=X) to look around."""
    y = np.deg2rad(yaw_deg); p = np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4, help="use every Nth training camera (base views)")
    ap.add_argument("--yaw", type=float, default=25.0, help="extra look-around yaw magnitude (deg); 0=off")
    ap.add_argument("--limit", type=int, default=0, help="cap total frames (0=all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from threedgrut.render import Renderer
    from threedgrut.datasets.protocols import Batch
    r = Renderer.from_checkpoint(checkpoint_path=args.checkpoint, path=args.dataset,
                                 out_dir=args.out, save_gt=False, computes_extra_metrics=False)
    model, dataset, loader = r.model, r.dataset, r.dataloader
    dev = "cuda"

    # extra look-around offsets (camera-frame rotations) applied on top of each base pose
    offs = [(0.0, 0.0)]
    if args.yaw > 0:
        offs += [(args.yaw, 0), (-args.yaw, 0), (0, args.yaw*0.6), (0, -args.yaw*0.6)]

    n = 0
    for it, batch in enumerate(loader):
        if it % args.stride != 0:
            continue
        gb = dataset.get_gpu_batch_with_intrinsics(batch)
        rays_dir0 = gb.rays_dir                      # (1,H,W,3) camera-frame, normalized
        T0 = gb.T_to_world.clone()                   # (1,4,4) cam->world
        K = list(gb.intrinsics)
        H, W = rays_dir0.shape[1], rays_dir0.shape[2]
        for (yaw, pit) in offs:
            gb.T_to_world = T0.clone()
            gb.rays_dir = rays_dir0
            if yaw != 0 or pit != 0:                 # rotate the camera in place to look around
                Rc = torch.tensor(rot_yaw_pitch(yaw, pit), dtype=T0.dtype, device=T0.device)
                Tn = T0.clone()
                Tn[0, :3, :3] = T0[0, :3, :3] @ Rc   # rotate cam->world basis
                gb.T_to_world = Tn
                gb.rays_dir = torch.einsum("ij,bhwj->bhwi", Rc.T.to(rays_dir0.dtype), rays_dir0)
            with torch.no_grad():
                out = model(gb)
            dist = out["pred_dist"].squeeze().float().cpu().numpy()        # (H,W) ray distance
            rz = gb.rays_dir.squeeze()[..., 2].float().cpu().numpy()       # cos angle to +Z
            zdepth = (dist * rz).astype(np.float32)                        # (H,W) z-depth
            zdepth[~np.isfinite(zdepth)] = 0.0
            zdepth[zdepth < 0] = 0.0
            c2w = gb.T_to_world.squeeze().float().cpu().numpy()            # (4,4)
            np.savez_compressed(os.path.join(args.out, f"f_{n:05d}.npz"),
                                depth=zdepth, c2w=c2w, K=np.array(K, np.float32))
            if n == 0:
                print(f"[render] first frame: depth {zdepth.shape} range "
                      f"[{zdepth[zdepth>0].min():.2f},{zdepth.max():.2f}]m  K={K}", flush=True)
            n += 1
            if args.limit and n >= args.limit:
                print(f"[render] wrote {n} depth frames to {args.out}", flush=True); return
    print(f"[render] wrote {n} depth frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
