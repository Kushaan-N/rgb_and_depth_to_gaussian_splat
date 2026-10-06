"""Per-frame LiDAR depth targets for depth-supervised 3DGRUT training (3DGRUT venv; GPU if present).

For every camera of a COLMAP text dataset, the metric LiDAR cloud (all scans, voxelised) is
z-buffered into the image: each point is splatted as a small square whose size matches its 3D
spacing at that distance, and each pixel keeps the NEAREST point. Then a pixel is dropped if
  - it is farther than the nearest depth in its neighbourhood by > --edge-rel (a background point
    showing through a gap in a sparsely sampled foreground surface, or a depth edge where a small
    LiDAR-camera misalignment would teach the wrong surface), or
  - fewer than --min-support of its neighbours are valid (isolated returns).
The target is the ray distance along the unit pixel-centre ray, exactly as 3DGRUT renders it
(pred_dist), so no z-to-range conversion is needed downstream.

Writes <images>/<stem>_depth.npy (float16, dataset units, 0 = no target) next to each image — the
patched 3DGRUT loads it (patches/3dgrut-depth-loss.patch) — plus <dataset>/depth_targets.json.

    python scripts/build_depth_targets.py --dataset <dir with sparse/0/*.txt + images> \
        --cloud <pipeline>/depth_cloud.ply --sim3-json <pipeline>/sim3.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def read_text_model(d):
    cams, imgs = {}, []
    for ln in open(os.path.join(d, "cameras.txt")):
        if ln.strip() and not ln.startswith("#"):
            p = ln.split(); cams[int(p[0])] = (int(p[2]), int(p[3]), [float(x) for x in p[4:8]])
    lines = [ln for ln in open(os.path.join(d, "images.txt")) if not ln.startswith("#")]
    for ln in lines[::2]:
        p = ln.split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, p[1:8])
        imgs.append((p[9], int(p[8]), np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])))
    return cams, imgs


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cloud", required=True, help="metric LiDAR cloud (.ply)")
    ap.add_argument("--sim3-json", required=True, help="COLMAP -> metric Sim3")
    ap.add_argument("--voxel-m", type=float, default=0.02, help="3D spacing of the cloud (splat size)")
    ap.add_argument("--edge-rel", type=float, default=0.05)
    ap.add_argument("--min-support", type=int, default=5, help="valid neighbours required in a 3x3 window")
    ap.add_argument("--max-range-m", type=float, default=12.0)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from plyfile import PlyData

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    v = PlyData.read(args.cloud)["vertex"].data
    Xm = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    X = torch.tensor(((Xm - t) @ R) / s, dtype=torch.float64, device=dev)      # cloud in dataset units
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    vox = args.voxel_m / s; rmax = args.max_range_m / s

    stats = []
    for name, cid, q, tcw in sorted(imgs):
        W, H, (fx, fy, cx, cy) = cams[cid]
        Rcw = torch.tensor(quat_to_R(q), dtype=torch.float64, device=dev)
        Pc = X @ Rcw.T + torch.tensor(tcw, dtype=torch.float64, device=dev)
        z = Pc[:, 2]
        u = fx * Pc[:, 0] / z + cx; w = fy * Pc[:, 1] / z + cy                  # continuous pixel coords
        keep = (z > 0.05) & (u > -8) & (u < W + 8) & (w > -8) & (w < H + 8)
        Pc, z, u, w = Pc[keep], z[keep], u[keep], w[keep]
        rng = torch.linalg.norm(Pc, dim=1)
        ok = rng < rmax
        Pc, z, u, w, rng = Pc[ok], z[ok], u[ok], w[ok], rng[ok]
        rad = torch.clamp(torch.ceil(0.5 * fx * vox / z), 1, 6).long()           # splat half-size in px
        D = torch.full((H * W,), float("inf"), dtype=torch.float64, device=dev)
        ui, wi = torch.floor(u).long(), torch.floor(w).long()
        for r in range(1, 7):
            sel = rad == r
            if not sel.any():
                continue
            us, ws, ds = ui[sel], wi[sel], z[sel]
            for dy in range(-r + 1, r):
                for dx in range(-r + 1, r):
                    uu, ww = us + dx, ws + dy
                    m = (uu >= 0) & (uu < W) & (ww >= 0) & (ww < H)
                    D.scatter_reduce_(0, (ww[m] * W + uu[m]), ds[m], reduce="amin")
        Z = D.view(H, W)                                                         # nearest z-depth per pixel
        valid = torch.isfinite(Z)
        Zf = torch.where(valid, Z, torch.full_like(Z, 1e9))
        zmin = -F.max_pool2d(-Zf[None, None], 5, 1, 2)[0, 0]                     # nearest depth nearby
        valid &= Z <= zmin * (1 + args.edge_rel)
        support = F.avg_pool2d(valid[None, None].double(), 3, 1, 1)[0, 0] * 9
        valid &= support >= args.min_support + 1
        # z-depth -> distance along the unit pixel-centre ray (3DGRUT's pred_dist)
        uu, ww = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
        ray = torch.sqrt(((uu - cx) / fx) ** 2 + ((ww - cy) / fy) ** 2 + 1)
        out = torch.where(valid, Z * ray, torch.zeros_like(Z)).float().cpu().numpy()
        dst = os.path.splitext(os.path.join(args.dataset, "images", name))[0] + "_depth.npy"
        np.save(dst, out.astype(np.float16))
        stats.append(float(valid.float().mean()))
    info = {"frames": len(stats), "valid_fraction": {"median": round(float(np.median(stats)), 3),
                                                    "min": round(float(np.min(stats)), 3)},
            "cloud": args.cloud, "edge_rel": args.edge_rel, "min_support": args.min_support,
            "units": "ray distance, dataset (COLMAP) units"}
    json.dump(info, open(os.path.join(args.dataset, "depth_targets.json"), "w"), indent=2)
    print(f"[depth] {len(stats)} frames; pixels with a LiDAR target: median "
          f"{info['valid_fraction']['median']:.1%} (min {info['valid_fraction']['min']:.1%})", flush=True)


if __name__ == "__main__":
    main()
