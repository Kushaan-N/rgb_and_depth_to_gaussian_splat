"""Per-frame LiDAR depth targets for depth-supervised 3DGRUT training (3DGRUT venv; GPU if present).

For every camera of a COLMAP text dataset, the metric LiDAR cloud (all scans, voxelised) is
z-buffered into the image: each point is splatted as a small square whose size matches its 3D
spacing at that distance, and each pixel keeps the NEAREST point. Then a pixel is dropped if
  - it is farther than the nearest depth in its neighbourhood by > --edge-rel (a background point
    showing through a gap in a sparsely sampled foreground surface, or a depth edge where a small
    LiDAR-camera misalignment would teach the wrong surface), or
  - fewer than --min-support of its neighbours are valid (isolated returns).
The target is the ray distance along the unit pixel-centre ray, exactly as 3DGRUT renders it
(pred_dist), so no z-to-range conversion is needed downstream. Per-pixel depth (--reduce): the nearest
splat only decides occlusion; the value is the centre-weighted mean inverse depth of the splats on that
front surface ("surface", default). The original "min" rule (nearest splat in the window) is biased
towards the camera on slanted surfaces: ~2% on CEAR, where EDGS triangulation sat +2.3% behind it.

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


def zbuffer_target(X, Rcw, tcw, W, H, K, vox, rmax, edge_rel=0.05, min_support=5, reduce="surface", attr=None):
    """One camera: z-buffer the cloud X (N,3 torch float64, dataset units) -> (ray distance HxW, 0 = no target;
    per-pixel mean of `attr` (N,) over the front-surface points, or None). reduce="min": every pixel of a point's
    square splat takes the NEAREST point (the original rule; on slanted surfaces such as the floor seen at a grazing
    angle the nearest of a k x k window is systematically closer than the surface at the pixel centre — a
    ~1-3% near bias). reduce="surface": the nearest depth only decides occlusion; the target is the Gaussian-
    weighted (by distance to the pixel centre) mean INVERSE depth of the splats on that front surface (within
    edge_rel of the nearest), which is exact for planes and unbiased for smooth surfaces."""
    import torch
    import torch.nn.functional as F
    fx, fy, cx, cy = K
    dev = X.device
    Pc = X @ Rcw.T + tcw
    z = Pc[:, 2]
    u = fx * Pc[:, 0] / z + cx; w = fy * Pc[:, 1] / z + cy                          # continuous pixel coords
    keep = (z > 0.05) & (u > -8) & (u < W + 8) & (w > -8) & (w < H + 8)
    keep &= torch.linalg.norm(Pc, dim=1) < rmax
    z, u, w = z[keep], u[keep], w[keep]
    a = attr[keep] if attr is not None else None
    rad = torch.clamp(torch.ceil(0.5 * fx * vox / z), 1, 6).long()                   # splat half-size in px
    ui, wi = torch.floor(u).long(), torch.floor(w).long()
    D = torch.full((H * W,), float("inf"), dtype=torch.float64, device=dev)
    offs = []                                                                         # (point idx, pixel idx) per splat
    for r in range(1, 7):
        sel = torch.nonzero(rad == r)[:, 0]
        if not len(sel):
            continue
        for dy in range(-r + 1, r):
            for dx in range(-r + 1, r):
                uu, ww = ui[sel] + dx, wi[sel] + dy
                m = (uu >= 0) & (uu < W) & (ww >= 0) & (ww < H)
                pix = ww[m] * W + uu[m]
                D.scatter_reduce_(0, pix, z[sel[m]], reduce="amin")
                offs.append((sel[m], pix))
    if reduce == "surface" or a is not None:
        num = torch.zeros(H * W, dtype=torch.float64, device=dev); den = torch.zeros_like(num)
        anum = torch.zeros_like(num)
        for idx, pix in offs:
            zi = z[idx]
            front = zi <= D[pix] * (1 + edge_rel)                                     # same surface as the nearest
            du = (pix % W).double() + 0.5 - u[idx]; dv = (pix // W).double() + 0.5 - w[idx]
            sig = 0.5 * rad[idx].double()
            wt = torch.exp(-(du * du + dv * dv) / (2 * sig * sig)) * front
            num.index_add_(0, pix, wt / zi); den.index_add_(0, pix, wt)
            if a is not None:
                anum.index_add_(0, pix, wt * a[idx])
        if reduce == "surface":
            D = torch.where(den > 0, den / num.clamp_min(1e-30), D)
        A = torch.where(den > 0, anum / den.clamp_min(1e-30), torch.zeros_like(den)).view(H, W) if a is not None else None
    else:
        A = None
    Z = D.view(H, W)                                                                  # z-depth per pixel
    valid = torch.isfinite(Z)
    Zf = torch.where(valid, Z, torch.full_like(Z, 1e9))
    zmin = -F.max_pool2d(-Zf[None, None], 5, 1, 2)[0, 0]                              # nearest depth nearby
    valid &= Z <= zmin * (1 + edge_rel)
    support = F.avg_pool2d(valid[None, None].double(), 3, 1, 1)[0, 0] * 9
    valid &= support >= min_support + 1
    # z-depth -> distance along the unit pixel-centre ray (3DGRUT's pred_dist)
    uu, ww = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
    ray = torch.sqrt(((uu - cx) / fx) ** 2 + ((ww - cy) / fy) ** 2 + 1)
    return torch.where(valid, Z * ray, torch.zeros_like(Z)), A


def load_cloud(cloud, sim3_json, dev):
    """Metric .ply -> torch (N,3) float64 in dataset units, plus the Sim3 scale (metres per dataset unit)."""
    import torch
    from plyfile import PlyData
    j = json.load(open(sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    v = PlyData.read(cloud)["vertex"].data
    Xm = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return torch.tensor(((Xm - t) @ R) / s, dtype=torch.float64, device=dev), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cloud", required=True, help="metric LiDAR cloud (.ply)")
    ap.add_argument("--sim3-json", required=True, help="COLMAP -> metric Sim3")
    ap.add_argument("--voxel-m", type=float, default=0.02, help="3D spacing of the cloud (splat size)")
    ap.add_argument("--edge-rel", type=float, default=0.05)
    ap.add_argument("--min-support", type=int, default=5, help="valid neighbours required in a 3x3 window")
    ap.add_argument("--max-range-m", type=float, default=12.0)
    ap.add_argument("--reduce", choices=("surface", "min"), default="surface",
                    help="per-pixel depth: front-surface weighted mean (unbiased) or nearest splat (original, near-biased)")
    args = ap.parse_args()
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, s = load_cloud(args.cloud, args.sim3_json, dev)
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    vox = args.voxel_m / s; rmax = args.max_range_m / s

    stats = []
    for name, cid, q, tcw in sorted(imgs):
        W, H, K = cams[cid]
        Rcw = torch.tensor(quat_to_R(q), dtype=torch.float64, device=dev)
        D, _ = zbuffer_target(X, Rcw, torch.tensor(tcw, dtype=torch.float64, device=dev), W, H, K, vox, rmax,
                              args.edge_rel, args.min_support, args.reduce)
        dst = os.path.splitext(os.path.join(args.dataset, "images", name))[0] + "_depth.npy"
        if os.path.islink(dst):
            os.remove(dst)                                                            # never write through a link
        np.save(dst, D.float().cpu().numpy().astype(np.float16))
        stats.append(float((D > 0).float().mean()))
    info = {"frames": len(stats), "valid_fraction": {"median": round(float(np.median(stats)), 3),
                                                    "min": round(float(np.min(stats)), 3)},
            "cloud": args.cloud, "edge_rel": args.edge_rel, "min_support": args.min_support, "reduce": args.reduce,
            "units": "ray distance, dataset (COLMAP) units"}
    json.dump(info, open(os.path.join(args.dataset, "depth_targets.json"), "w"), indent=2)
    print(f"[depth] {len(stats)} frames; pixels with a LiDAR target: median "
          f"{info['valid_fraction']['median']:.1%} (min {info['valid_fraction']['min']:.1%})", flush=True)


if __name__ == "__main__":
    main()
