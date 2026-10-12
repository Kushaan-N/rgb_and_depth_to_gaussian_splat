"""Reliability mask for per-frame depth targets (<image>_depth.npy) -> <image>_depthmask.png (255 = trust, 0 = drop).

A LiDAR (or any) depth target is wrong where the sensor is: glossy / dark / transparent surfaces (black counter,
cabinets, windows — a VLP-16 returns through glass or not at all), and at depth discontinuities (a small
LiDAR-camera misalignment teaches the wrong surface). Supervising there blurs reflections and fights the
photometric loss. A pixel's target is dropped if
  - there is no target (0), or the LiDAR returns that produced it are weak: mean intensity of the front-surface
    points at or below the --min-intensity-q quantile of the cloud (needs --intensity, see below),
  - it lies within --edge-px of a depth discontinuity of the target (a valid neighbour differs by > --edge-rel),
  - a photometric depth (ray distance rendered from a splat trained WITHOUT depth loss, eval_views.py
    --save-depth-dir) disagrees by > --disagree-rel on a textured pixel (local gradient >= --texture-min; on
    textureless pixels the photometric depth is unconstrained, so the target stays).
Every criterion is skipped when its input is missing (no intensity, no photometric depth).

Optional: --rebuild-targets --cloud --sim3-json rewrites the targets first with build_depth_targets.zbuffer_target
(--reduce surface, the near-bias fix) and keeps per-pixel LiDAR intensity from the same splats; --edgs-points
(points3D.txt of scripts/edgs_init.py; its LiDAR seeds come first) measures the triangulation-vs-target bias with
the old and the rebuilt targets. Per-point intensity comes from a CPU-venv pass over the bag (rosbags):
  python scripts/depth_reliability.py --intensity-from-config configs/<seq>.yaml --cloud lidar_static.ply --intensity I.npy
Then (3DGRUT venv, GPU):
  python scripts/depth_reliability.py --dataset <ds> [--photo-depth-dir <renders>] [--intensity I.npy --cloud ...
      --sim3-json ... --rebuild-targets] [--edgs-points <edgs ds>/sparse/0/points3D.txt] [--viz-dir viz/]
Never writes through a symlink (datasets here are link copies). Stats -> <dataset>/depth_reliability.json.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def write_intensity(config, cloud, out, voxel=0.02):
    """Per-point mean LiDAR intensity for `cloud` (metric .ply) from every scan of the config's bag (CPU venv)."""
    import build_lidar_cloud as blc
    import pose_utils as pu
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    base_read = blc.read_xyz

    def read_xyzi(msg, min_r, max_r):                     # same parsing, plus the intensity column
        f = {x.name: x for x in msg.fields}
        if "intensity" not in f:
            raise ValueError("PointCloud2 has no intensity field")
        end = ">" if msg.is_bigendian else "<"
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        buf = np.ascontiguousarray(raw.reshape(msg.height, msg.row_step)[:, : msg.width * msg.point_step]).reshape(-1, msg.point_step)
        col = lambda n: buf[:, f[n].offset: f[n].offset + np.dtype(end + blc._PF_DTYPE[f[n].datatype]).itemsize].copy() \
            .view(np.dtype(end + blc._PF_DTYPE[f[n].datatype])).ravel().astype(np.float64)
        x, y, z, i = col("x"), col("y"), col("z"), col("intensity")
        r = np.sqrt(x * x + y * y + z * z); ok = np.isfinite(r) & (r > min_r) & (r < max_r)
        return np.stack([x[ok], y[ok], z[ok], i[ok]], 1)
    blc.read_xyz = read_xyzi
    try:
        scans = blc.registered_scans(pu.load_config(config), log=lambda m: print(m, flush=True))
    finally:
        blc.read_xyz = base_read
    W = np.concatenate([p[:, :3] @ T[:3, :3].T + T[:3, 3] for T, p in scans]); I = np.concatenate([p[:, 3] for _, p in scans])
    key = np.floor(W / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    cen = np.stack([np.bincount(inv, W[:, k]) for k in range(3)], 1) / cnt[:, None]
    imean = np.bincount(inv, I) / cnt
    v = PlyData.read(cloud)["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    d, j = cKDTree(cen).query(X, distance_upper_bound=1.5 * voxel, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    out_i = np.where(np.isfinite(d), imean[np.minimum(j, len(imean) - 1)], np.nan).astype(np.float32)
    hits = np.where(np.isfinite(d), cnt[np.minimum(j, len(cnt) - 1)], 0).astype(np.int32)
    np.savez(out, intensity=out_i, hits=hits)
    print(f"[reliab] intensity for {len(X):,} cloud points from {len(W):,} returns ({np.isfinite(out_i).mean():.1%} matched); "
          f"quantiles 5/50/95: {np.nanpercentile(out_i, [5, 50, 95]).round(1)}", flush=True)


def texture(img):
    """Local gradient magnitude of the grey image in [0,1] (Sobel / 8, 7x7 box mean)."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8
    return cv2.blur(m, (7, 7))


def depth_edges(D, rel, px):
    """Pixels within `px` of a discontinuity of D (a valid 4-neighbour differing by > rel)."""
    import cv2
    e = np.zeros(D.shape, bool); v = D > 0
    for a, b in ((np.s_[:, 1:], np.s_[:, :-1]), (np.s_[1:, :], np.s_[:-1, :])):
        da, db = D[a], D[b]
        jump = (v[a] & v[b]) & (np.abs(da - db) > rel * np.minimum(da, db))       # holes are not edges
        e[a] |= jump; e[b] |= jump
    if px > 0:
        e = cv2.dilate(e.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
    return e & v


def normals_cos(D, K):
    """|cos| between the pixel ray and the surface normal of the target (NaN where undefined)."""
    H, W = D.shape; fx, fy, cx, cy = K
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    r = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1); r /= np.linalg.norm(r, axis=-1, keepdims=True)
    P = r * D[..., None]
    n = np.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1])
    nn = np.linalg.norm(n, axis=-1); ok = (nn > 0) & (D[1:-1, 1:-1] > 0) & (D[1:-1, 2:] > 0) & (D[1:-1, :-2] > 0) \
        & (D[2:, 1:-1] > 0) & (D[:-2, 1:-1] > 0)
    c = np.full(D.shape, np.nan, np.float32)
    c[1:-1, 1:-1] = np.where(ok, np.abs((n * r[1:-1, 1:-1]).sum(-1)) / np.maximum(nn, 1e-12), np.nan)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--photo-depth-dir", default=None, help="<stem>_splatdepth.npy from a splat without depth loss")
    ap.add_argument("--target-suffix", default="_depth.npy")
    ap.add_argument("--intensity", default=None, help=".npz from --intensity-from-config (per point of --cloud)")
    ap.add_argument("--intensity-from-config", default=None, help="CPU pass: write --intensity from the config's bag")
    ap.add_argument("--cloud", default=None); ap.add_argument("--sim3-json", default=None)
    ap.add_argument("--rebuild-targets", action="store_true", help="rewrite targets with the unbiased z-buffer first")
    ap.add_argument("--voxel-m", type=float, default=0.02); ap.add_argument("--max-range-m", type=float, default=12.0)
    ap.add_argument("--min-intensity-q", type=float, default=0.10)
    ap.add_argument("--edge-rel", type=float, default=0.05); ap.add_argument("--edge-px", type=int, default=1)
    ap.add_argument("--disagree-rel", type=float, default=0.10)
    ap.add_argument("--texture-min", type=float, default=0.01)
    ap.add_argument("--edgs-points", default=None, help="EDGS points3D.txt: triangulation vs target bias diagnosis")
    ap.add_argument("--diag-every", type=int, default=8)
    ap.add_argument("--viz-dir", default=None); ap.add_argument("--viz-n", type=int, default=6)
    args = ap.parse_args()
    if args.intensity_from_config:
        return write_intensity(args.intensity_from_config, args.cloud, args.intensity, args.voxel_m)

    import cv2, torch
    from PIL import Image
    from build_depth_targets import read_text_model, quat_to_R, zbuffer_target, load_cloud
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    imgs = sorted(imgs)
    stem = lambda n: os.path.splitext(os.path.join(args.dataset, "images", n))[0]
    X = s = A = None
    if args.cloud and args.sim3_json and (args.rebuild_targets or args.intensity):
        X, s = load_cloud(args.cloud, args.sim3_json, dev)
    if args.intensity and X is not None:
        z = np.load(args.intensity); ia = z["intensity"]
        thr = float(np.nanquantile(ia, args.min_intensity_q))
        A = torch.tensor(np.nan_to_num(ia, nan=-1.0), dtype=torch.float64, device=dev)   # unmatched -> "weak"
        print(f"[reliab] intensity threshold {thr:.2f} ({args.min_intensity_q:.0%} quantile)", flush=True)
    E = None
    if args.edgs_points:                                              # EDGS triangulated points (skip the seeds)
        ei = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.edgs_points))), "edgs_init.json")
        nseed = json.load(open(ei))["seeds"] if os.path.isfile(ei) else 0
        E = np.loadtxt(args.edgs_points, comments="#", usecols=(1, 2, 3))[nseed:]
        E = E[np.random.default_rng(0).choice(len(E), min(len(E), 600000), replace=False)]

    tot = {k: 0 for k in ("pixels", "target", "low_intensity", "edge", "disagree", "kept", "photo_checked")}
    diag = {"old": {"facing": [], "grazing": []}, "new": {"facing": [], "grazing": []}, "edgs_old": [], "edgs_new": []}
    diag_frame = lambda k: k % args.diag_every == args.diag_every // 2       # off the trainer's every-8th test frames
    viz_keep = set(np.linspace(0, len(imgs) - 1, args.viz_n).astype(int).tolist()) if args.viz_dir else set()
    for k, (name, cid, q, tcw) in enumerate(imgs):
        W, H, K = cams[cid]
        tp = stem(name) + args.target_suffix
        D_old = np.load(tp).astype(np.float32) if os.path.isfile(tp) else np.zeros((H, W), np.float32)
        D, I = D_old, None
        if X is not None:
            Rcw = torch.tensor(quat_to_R(q), dtype=torch.float64, device=dev); t = torch.tensor(tcw, dtype=torch.float64, device=dev)
            Dn, In = zbuffer_target(X, Rcw, t, W, H, K, args.voxel_m / s, args.max_range_m / s, args.edge_rel, 5,
                                    "surface" if args.rebuild_targets else "min", A)
            if args.rebuild_targets:
                D = Dn.float().cpu().numpy()
                if os.path.islink(tp): os.remove(tp)
                np.save(tp, D.astype(np.float16))
            I = In.float().cpu().numpy() if In is not None else None
        valid = D > 0
        keep = valid.copy()
        low = np.zeros_like(valid)
        if I is not None:
            low = valid & (I <= thr); keep &= ~low
        edge = depth_edges(D, args.edge_rel, args.edge_px); keep &= ~edge
        img = np.asarray(Image.open(os.path.join(args.dataset, "images", name)).convert("RGB"))
        dis = np.zeros_like(valid); Dp = None
        pp = os.path.join(args.photo_depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy") if args.photo_depth_dir else ""
        if os.path.isfile(pp):
            Dp = np.load(pp).astype(np.float32)
            tex = texture(img) >= args.texture_min
            chk = valid & (Dp > 0) & tex
            dis = chk & (np.abs(Dp - D) > args.disagree_rel * D); keep &= ~dis
            tot["photo_checked"] += int(chk.sum())
            if diag_frame(k):                                             # bias: photometric vs old / new target
                cosn = normals_cos(D, K)
                for tag, T in (("old", D_old), ("new", D)):
                    m = chk & (T > 0) & (np.abs(Dp - T) < 0.1 * np.maximum(T, 1e-6)) & ~edge
                    rel = (Dp - T) / np.maximum(T, 1e-6)
                    diag[tag]["facing"].append(rel[m & (cosn >= 0.5)]); diag[tag]["grazing"].append(rel[m & (cosn < 0.5)])
        if E is not None and diag_frame(k):                            # EDGS points seen from this camera
            R = quat_to_R(q); Pc = E @ R.T + tcw; zc = Pc[:, 2]; ok = zc > 0.05
            u = K[0] * Pc[ok, 0] / zc[ok] + K[2]; v = K[1] * Pc[ok, 1] / zc[ok] + K[3]; dist = np.linalg.norm(Pc[ok], axis=1)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ui, vi, dist = u[inb].astype(int), v[inb].astype(int), dist[inb]
            for tag, T in (("edgs_old", D_old), ("edgs_new", D)):
                t_ = T[vi, ui]; vis = (t_ > 0) & ~edge[vi, ui]
                if Dp is not None:                                     # visible = on the photometric surface
                    vis &= (Dp[vi, ui] > 0) & (np.abs(dist - Dp[vi, ui]) < 0.03 * Dp[vi, ui])
                rel = (dist[vis] - t_[vis]) / t_[vis]; diag[tag].append(rel[np.abs(rel) < 0.2])
        mp = stem(name) + "_depthmask.png"
        if os.path.islink(mp): os.remove(mp)
        cv2.imwrite(mp, (keep * 255).astype(np.uint8))
        for key, m in (("target", valid), ("low_intensity", low), ("edge", edge), ("disagree", dis), ("kept", keep)):
            tot[key] += int(m.sum())
        tot["pixels"] += valid.size
        if k in viz_keep:                                             # image | target | photometric | reasons
            os.makedirs(args.viz_dir, exist_ok=True)
            hi = np.percentile(D[valid], 98) if valid.any() else 1.0
            col = lambda d: cv2.applyColorMap((np.clip(d / hi, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * (d > 0)[..., None]
            rs = np.zeros((H, W, 3), np.uint8); rs[keep] = (0, 160, 0); rs[edge] = (255, 0, 0)
            rs[low] = (0, 200, 255); rs[dis] = (0, 0, 255)                    # BGR: kept green, edge blue, weak yellow, disagree red
            tiles = [img[..., ::-1].copy(), col(D), col(Dp) if Dp is not None else np.zeros_like(img), rs]
            for tl, lab in zip(tiles, ("image", "target", "photometric", "kept/edge/weak/disagree")):
                cv2.putText(tl, lab, (6, 22), 0, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(args.viz_dir, os.path.splitext(name)[0] + "_reliab.jpg"), np.hstack(tiles))
    pct = lambda a: round(a / max(1, tot["target"]), 4)
    info = {"frames": len(imgs), "target_fraction": round(tot["target"] / max(1, tot["pixels"]), 4),
            "of_target": {"dropped_low_intensity": pct(tot["low_intensity"]), "dropped_edge": pct(tot["edge"]),
                          "dropped_disagree": pct(tot["disagree"]), "kept": pct(tot["kept"]),
                          "photo_checked": pct(tot["photo_checked"])},
            "args": vars(args)}
    med = lambda a: (round(float(np.median(np.concatenate(a))), 4), int(sum(len(x) for x in a))) if a and sum(len(x) for x in a) else None
    info["bias_signed_median"] = {
        "photometric_minus_target_old": {kk: med(v) for kk, v in diag["old"].items()},
        "photometric_minus_target_new": {kk: med(v) for kk, v in diag["new"].items()},
        "edgs_minus_target_old": med(diag["edgs_old"]), "edgs_minus_target_new": med(diag["edgs_new"]),
        "note": "(signed median of (other - target) / target, n); facing = |cos(ray, normal)| >= 0.5"}
    json.dump(info, open(os.path.join(args.dataset, "depth_reliability.json"), "w"), indent=2)
    print("[reliab]", json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
