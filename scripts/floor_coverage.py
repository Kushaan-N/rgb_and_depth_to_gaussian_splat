"""Diagnose floor holes in a splat: compare floor coverage of the LiDAR cloud, the exported splat,
and the training cameras on a top-down grid, all in the metric GT frame.

Classifies every room-floor cell the splat is missing, which tells you WHY it's missing:
  B  LiDAR saw floor + cameras saw it well  -> training lost it (optimizer failed / pruned)
  C  LiDAR saw floor, cameras only grazing / never -> RGB can't justify it, LiDAR init gets pruned
  D  cameras saw it, no LiDAR floor          -> RGB-only, too weak
  E  nothing saw it                          -> true blind spot (needs a floor prior or recapture)

    python scripts/floor_coverage.py --lidar-cloud <out>/lidar_cloud.ply \
        --splat-ply <out>/gaussians_colmap_lidarinit.ply \
        --colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0 --out fig.png
"""
from __future__ import annotations
import argparse, json, os
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


def cam_pose(img):
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    M = np.asarray(cfw.matrix())
    return M[:, :3], M[:, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar-cloud", required=True)
    ap.add_argument("--splat-ply", required=True)
    ap.add_argument("--colmap-model", required=True)
    ap.add_argument("--gt-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.10)
    ap.add_argument("--min-opacity", type=float, default=0.2)
    args = ap.parse_args()
    import open3d as o3d, pycolmap
    from plyfile import PlyData
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    Crec = pycolmap.Reconstruction(args.colmap_model); Grec = pycolmap.Reconstruction(args.gt_model)
    C = {i.name: np.asarray(i.projection_center()) for i in Crec.images.values()}
    G = {i.name: np.asarray(i.projection_center()) for i in Grec.images.values()}
    common = sorted(set(C) & set(G))
    s, R, t = umeyama(np.array([C[n] for n in common]), np.array([G[n] for n in common]))

    # LiDAR floor plane (metric GT frame)
    pc = o3d.io.read_point_cloud(args.lidar_cloud); L = np.asarray(pc.points)
    model, inl = pc.segment_plane(0.03, 3, 1000)
    n = np.array(model[:3]); d = model[3]; k = np.linalg.norm(n); n, d = n / k, d / k
    if n[2] < 0:
        n, d = -n, -d
    hL = L @ n + d
    fl = L[np.asarray(inl)]
    lo = np.percentile(fl[:, :2], 2, axis=0) - 0.3
    hi = np.percentile(fl[:, :2], 98, axis=0) + 0.3
    nx, ny = np.ceil((hi - lo) / args.cell).astype(int)

    def cells(P):
        ij = np.floor((P[:, :2] - lo) / args.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        return ij[ok]

    def grid(P):
        g = np.zeros((nx, ny), int)
        ij = cells(P)
        np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
        return g

    lidar_floor = grid(L[np.abs(hL) < 0.05]) > 0
    lidar_obj = grid(L[(hL > 0.10) & (hL < 1.2)]) > 2          # something standing on the floor there

    # splat gaussians -> GT frame, keep visible ones in the floor band
    v = PlyData.read(args.splat_ply)["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    Xg = s * (X @ R.T) + t
    op = np.asarray(v["opacity"], dtype=np.float64)
    if op.min() < 0 or op.max() > 1:
        op = 1.0 / (1.0 + np.exp(-op))
    hS = Xg @ n + d
    splat_cnt = grid(Xg[(np.abs(hS) < 0.08) & (op > args.min_opacity)])
    splat_floor = splat_cnt > 0

    # camera coverage of each floor cell: any view, and "good" views (not grazing, not far)
    cx_ = lo[0] + (np.arange(nx) + 0.5) * args.cell
    cy_ = lo[1] + (np.arange(ny) + 0.5) * args.cell
    XX, YY = np.meshgrid(cx_, cy_, indexing="ij")
    ZZ = -(n[0] * XX + n[1] * YY + d) / n[2]
    Pf = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1)
    any_v = np.zeros(len(Pf), int); good_v = np.zeros(len(Pf), int)
    traj = []
    for img in Grec.images.values():
        cam = Grec.cameras[img.camera_id]
        fx, fy, cxx, cyy = [float(x) for x in cam.params[:4]]
        Rcw, tcw = cam_pose(img)
        Pc = Pf @ Rcw.T + tcw
        z = Pc[:, 2]
        u = fx * Pc[:, 0] / np.maximum(z, 1e-6) + cxx
        vv = fy * Pc[:, 1] / np.maximum(z, 1e-6) + cyy
        vis = (z > 0.1) & (u >= 0) & (u < cam.width) & (vv >= 0) & (vv < cam.height)
        ctr = np.asarray(img.projection_center()); traj.append(ctr)
        ray = Pf - ctr; dist = np.linalg.norm(ray, axis=1)
        cosang = np.abs(ray @ n) / np.maximum(dist, 1e-6)       # 1 = looking straight down at the cell
        any_v += vis
        good_v += vis & (cosang > np.cos(np.deg2rad(70))) & (dist < 4.0)
    any_v = any_v.reshape(nx, ny); good_v = good_v.reshape(nx, ny); traj = np.array(traj)

    # classify missing floor cells (skip cells under furniture)
    cls = np.full((nx, ny), 0)                         # 0 = splat has floor
    miss = ~splat_floor
    cls[miss & lidar_obj] = 1                          # under / at an object
    free = miss & ~lidar_obj
    cls[free & lidar_floor & (good_v > 0)] = 2         # B: training lost it
    cls[free & lidar_floor & (good_v == 0)] = 3        # C: camera grazing/blind, LiDAR has it
    cls[free & ~lidar_floor & (any_v > 0)] = 4         # D: camera only
    cls[free & ~lidar_floor & (any_v == 0)] = 5        # E: nothing
    room = lidar_floor | splat_floor | lidar_obj
    tot = int(room.sum())
    frac = {name: round(float(((cls == c) & room).sum()) / tot, 3) for c, name in
            [(0, "splat_has_floor"), (1, "under_object"), (2, "B_lost_despite_rgb+lidar"),
             (3, "C_lidar_only_camera_grazing_or_blind"), (4, "D_camera_only"), (5, "E_nothing")]}
    frac["lidar_floor_cover"] = round(float(lidar_floor[room].mean()), 3)
    frac["cells_room"] = tot
    frac["cell_m"] = args.cell

    fig, ax = plt.subplots(1, 4, figsize=(22, 6.2))
    ext = [lo[1], hi[1], hi[0], lo[0]]
    ax[0].imshow(lidar_floor * 1.0 + lidar_obj * 0.5, cmap="Greys", extent=ext); ax[0].set_title("LiDAR: floor (black) / objects (gray)")
    ax[1].imshow(np.log1p(splat_cnt), cmap="viridis", extent=ext); ax[1].set_title("Splat: visible floor gaussians (log count)")
    im = ax[2].imshow(np.minimum(good_v, 30), cmap="magma", extent=ext); ax[2].set_title("Camera: non-grazing views per floor cell")
    ax[2].plot(traj[:, 1], traj[:, 0], "c-", lw=1); plt.colorbar(im, ax=ax[2], fraction=0.04)
    cmap = ListedColormap(["#2e8b57", "#9a9a9a", "#d62728", "#ff7f0e", "#ffd700", "#111111"])
    shown = np.where(room, cls, 1)
    ax[3].imshow(shown, cmap=cmap, vmin=0, vmax=5, extent=ext)
    ax[3].plot(traj[:, 1], traj[:, 0], "c-", lw=1)
    ax[3].set_title("green=splat floor  red=B lost  orange=C LiDAR-only\nyellow=D cam-only  black=E nothing  gray=object")
    for a in ax:
        a.set_xlabel("y (m)"); a.set_ylabel("x (m)")
    fig.tight_layout(); fig.savefig(args.out, dpi=100)
    print("FLOOR_COVERAGE", json.dumps(frac))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
