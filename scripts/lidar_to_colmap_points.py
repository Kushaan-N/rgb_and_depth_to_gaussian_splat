"""Convert the metric LiDAR cloud into a COLMAP points3D.txt in the COLMAP splat's frame, to SEED
3DGRUT's init with dense, correct geometry (especially the under-observed room center).

COLMAP inits gaussians from its sparse SfM points (~28k, and almost none in the room center the
camera drove through). The LiDAR cloud is dense and covers the center. We map it into the COLMAP
frame via the inverse of the Sim3 (Umeyama on shared camera centres) and write it as points3D.txt,
so the splat starts with gaussians already on the real surfaces where the camera gave no signal.

    python scripts/lidar_to_colmap_points.py --cloud <out>/lidar_cloud.ply \
        --colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0 \
        --out-points3d <ds_lidarinit>/sparse/0/points3D.txt [--max-points 400000]
"""
from __future__ import annotations
import argparse, os
import numpy as np


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, d, Vt = np.linalg.svd(((dst - mu_d).T @ (src - mu_s)) / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = np.trace(np.diag(d) @ W) / (((src - mu_s) ** 2).sum() / len(src))
    return float(s), R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--colmap-model", required=True)
    ap.add_argument("--gt-model", required=True)
    ap.add_argument("--out-points3d", required=True)
    ap.add_argument("--max-points", type=int, default=400000)
    args = ap.parse_args()
    import open3d as o3d, pycolmap

    # Sim3 colmap->gt from shared camera centres, then invert for gt->colmap
    C = {i.name: np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(args.colmap_model).images.values()}
    G = {i.name: np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(args.gt_model).images.values()}
    common = sorted(set(C) & set(G))
    s, R, t = umeyama(np.array([C[n] for n in common]), np.array([G[n] for n in common]))
    print(f"[l2c] Sim3 colmap->gt scale={s:.4f} from {len(common)} frames", flush=True)

    pc = o3d.io.read_point_cloud(args.cloud)
    X_gt = np.asarray(pc.points)
    X_col = ((X_gt - t) @ R) / s                                  # gt->colmap: (1/s) R^T (x-t)
    col = (np.asarray(pc.colors) * 255).astype(int) if pc.has_colors() else np.full((len(X_col), 3), 160, int)
    if len(X_col) > args.max_points:
        idx = np.random.default_rng(0).choice(len(X_col), args.max_points, replace=False)
        X_col, col = X_col[idx], col[idx]
    os.makedirs(os.path.dirname(args.out_points3d), exist_ok=True)
    with open(args.out_points3d, "w") as f:
        f.write("# 3D point list from LiDAR (COLMAP frame)\n")
        for i, (p, c) in enumerate(zip(X_col, col), 1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")
    # drop any binary that would shadow the txt
    b = os.path.join(os.path.dirname(args.out_points3d), "points3D.bin")
    if os.path.exists(b):
        os.remove(b)
    print(f"[l2c] wrote {len(X_col):,} LiDAR init points -> {args.out_points3d}", flush=True)


if __name__ == "__main__":
    main()
