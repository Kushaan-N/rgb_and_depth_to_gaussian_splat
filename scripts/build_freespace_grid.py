"""LiDAR free-space grid for floater suppression during splat training (CPU). Dataset-agnostic.

Every LiDAR beam proves the space between the sensor and its return is empty. Gaussians sitting there
are floaters: they explain some training view but show up as haze/blobs from novel viewpoints. This
stage carves that measured free space into a voxel grid (metric frame of the LiDAR) of per-scan counts:
  free[v]  number of scans with a beam passing through v, from the sensor up to return - margin
           (margin = --margin-m + --margin-rel * range: range noise, pose / sim3 error)
  hit[v]   number of scans with a return inside v
The trainer (3DGRUT + patches/3dgrut-freespace.patch, freespace.grid_path) treats a voxel as confidently
free when free >= freespace.min_free and no hit lies within freespace.dilate voxels; voxels no beam
crossed are unknown and never touched. The COLMAP->metric sim3 (sim3.json) is stored alongside, so the
trainer maps gaussian centres (COLMAP frame) into the grid itself.

Held-out views must not leak through the LiDAR: with --cameras only scans whose sensor origin is within
--near-m of a camera of that COLMAP model (the training split) are carved, so a spatial hold-out
(offpath) gets no free space from scans recorded inside the held-out region. With the full dataset as
--cameras (production) every scan is used. No lidar section / no bag: prints SKIP, writes nothing.

    python scripts/build_freespace_grid.py --config configs/<seq>.yaml --sim3-json <pipeline>/sim3.json \
        --cameras <train dataset>/sparse/0 --out <dir>/freespace.npz [--check-ply splat.ply ...]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from build_lidar_cloud import LidarSkip, registered_scans
from sim3_utils import load_sim3


def camera_centres(model_dir):
    """COLMAP text model -> (N,3) camera centres in its frame (images.txt only, no pycolmap)."""
    C = []
    lines = [l for l in open(os.path.join(model_dir, "images.txt")) if l.strip() and not l.startswith("#")]
    for l in lines[::2]:
        q = np.array(l.split()[1:5], float); t = np.array(l.split()[5:8], float)
        w, x, y, z = q / np.linalg.norm(q)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        C.append(-R.T @ t)
    return np.array(C)


def read_ply_xyz_opacity(path):
    """Binary little-endian float PLY (3DGS export) -> xyz (N,3), opacity logit (N,) or None."""
    with open(path, "rb") as f:
        props, n = [], 0
        while (line := f.readline().strip()) != b"end_header":
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append((line.split()[2].decode(), line.split()[1].decode()))
        dt = np.dtype([(p, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]) for p, t in props])
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64), (a["opacity"] if "opacity" in a.dtype.names else None)


def confident_free(free, hit, min_free, dilate):
    """Same rule as the trainer: free >= min_free and no hit within `dilate` voxels."""
    from scipy.ndimage import maximum_filter
    occ = maximum_filter(hit > 0, size=2 * dilate + 1) if dilate > 0 else hit > 0
    return (free >= min_free) & ~occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim3-json", required=True, help="COLMAP -> metric similarity (x_m = s R x_c + t)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", default=None, help="COLMAP text model: carve only scans near its cameras")
    ap.add_argument("--near-m", type=float, default=0.3, help="max sensor-to-nearest-camera distance (m)")
    ap.add_argument("--voxel", type=float, default=0.04, help="voxel size (m)")
    ap.add_argument("--margin-m", type=float, default=0.05, help="free carving stops this far before a return")
    ap.add_argument("--margin-rel", type=float, default=0.01, help="... plus this fraction of the range")
    ap.add_argument("--step", type=float, default=0.75, help="ray sample step, in voxels")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan (smoke tests)")
    ap.add_argument("--ray-stride", type=int, default=1, help="use every Nth beam of a scan")
    ap.add_argument("--max-voxels", type=float, default=4e8, help="coarsen the voxel above this grid size")
    ap.add_argument("--min-free", type=int, default=3, help="for the printed statistics / --check-ply only")
    ap.add_argument("--dilate", type=int, default=1, help="for the printed statistics / --check-ply only")
    ap.add_argument("--check-ply", nargs="*", default=[], help="splat PLYs / points3D.txt (COLMAP frame): report share in free space")
    args = ap.parse_args()

    cfg = pu.load_config(args.config)
    log = lambda m: print(m, flush=True)
    try:
        scans = registered_scans(cfg, stride=args.stride, log=log)
    except LidarSkip as e:
        print(f"[freespace] SKIP: {e}"); return 0
    s, R, t = load_sim3(args.sim3_json)
    if args.cameras:
        C = camera_centres(args.cameras) * s @ R.T + t
        d = np.array([np.linalg.norm(C - T[:3, 3], axis=1).min() for T, _ in scans])
        keep = d <= args.near_m
        log(f"[freespace] {int(keep.sum())}/{len(scans)} scans within {args.near_m} m of the {len(C)} cameras "
            f"of {args.cameras}")
        scans = [sc for sc, k in zip(scans, keep) if k]
    if not scans:
        print("[freespace] SKIP: no scan near the cameras"); return 0

    # grid bounds: all returns and sensor origins (robust: drop 0.05% outliers per axis), padded
    W = np.concatenate([p[:: max(len(p) // 2000, 1)] @ T[:3, :3].T + T[:3, 3] for T, p in scans])
    O = np.array([T[:3, 3] for T, _ in scans])
    lo = np.minimum(np.percentile(W, 0.05, axis=0), O.min(0)) - 0.2
    hi = np.maximum(np.percentile(W, 99.95, axis=0), O.max(0)) + 0.2
    vox = args.voxel
    while np.prod(np.ceil((hi - lo) / vox)) > args.max_voxels:
        vox *= 1.25
    dims = np.ceil((hi - lo) / vox).astype(np.int64)
    log(f"[freespace] grid {dims.tolist()} ({np.prod(dims) / 1e6:.0f} M voxels) of {vox:.3f} m, "
        f"bbox {lo.round(2).tolist()}..{hi.round(2).tolist()}")
    free = np.zeros(int(np.prod(dims)), np.uint16); hit = np.zeros_like(free)

    def flat(X):  # metric points -> flat voxel index, -1 outside
        ijk = np.floor((X - lo) / vox).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
        return np.where(ok, (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2], -1)

    step = args.step * vox; t0 = time.time(); n_rays = 0
    for k, (T, p) in enumerate(scans):
        p = p[:: args.ray_stride]
        Rw, o = T[:3, :3], T[:3, 3]
        r = np.linalg.norm(p, axis=1)
        u = (p / r[:, None]) @ Rw.T                                    # unit beam directions, world frame
        end = r - (args.margin_m + args.margin_rel * r)                # carve up to here
        h = flat(p @ Rw.T + o); h = np.unique(h[h >= 0]); hit[h] += 1  # one count per scan
        n_rays += len(p)
        ns = np.floor(np.clip(end, 0, None) / step).astype(np.int64)
        idx = []
        for c in range(0, len(p), 4096):                              # chunks keep memory bounded
            nc, uc = ns[c: c + 4096], u[c: c + 4096]
            ray = np.repeat(np.arange(len(nc)), nc)
            j = np.arange(int(nc.sum())) - np.repeat(np.cumsum(nc) - nc, nc)
            X = o + uc[ray] * ((j + 0.5) * step)[:, None]
            f = flat(X); idx.append(np.unique(f[f >= 0]))
        if idx:
            f = np.unique(np.concatenate(idx)); free[f] += 1
        if k % 100 == 0:
            log(f"[freespace] scan {k}/{len(scans)}  ({time.time() - t0:.0f} s)")
    free, hit = free.reshape(*dims), hit.reshape(*dims)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"scans": len(scans), "rays": int(n_rays), "voxel_m": float(vox), "dims": dims.tolist(),
            "margin_m": args.margin_m, "margin_rel": args.margin_rel, "near_m": args.near_m,
            "cameras": args.cameras, "stride": args.stride, "ray_stride": args.ray_stride}
    np.savez_compressed(args.out, free=free, hit=hit, origin=lo.astype(np.float64), voxel=np.float64(vox),
                        sim3_s=np.float64(s), sim3_R=R, sim3_t=t, meta=json.dumps(meta))
    cf = confident_free(free, hit, args.min_free, args.dilate)
    meta.update({"free_voxels": int((free > 0).sum()), "hit_voxels": int((hit > 0).sum()),
                 "confident_free_voxels": int(cf.sum()), "min_free": args.min_free, "dilate": args.dilate,
                 "minutes": round((time.time() - t0) / 60, 1)})
    for ply in args.check_ply:      # how much of an existing splat sits in measured free space
        if ply.endswith(".txt"):  # COLMAP points3D.txt (e.g. the LiDAR init cloud: should be ~0% in free space)
            X = np.array([l.split()[1:4] for l in open(ply) if l.strip() and not l.startswith("#")], float); op = None
        else:
            X, op = read_ply_xyz_opacity(ply)
        ijk = np.floor((X * s @ R.T + t - lo) / vox).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < dims), axis=1)
        inf = np.zeros(len(X), bool); inf[ok] = cf[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        a = 1 / (1 + np.exp(-op)) if op is not None else np.ones(len(X))
        meta.setdefault("check", {})[os.path.basename(os.path.dirname(ply)) + "/" + os.path.basename(ply)] = {
            "gaussians": int(len(X)), "in_free": int(inf.sum()), "share": round(float(inf.mean()), 4),
            "opacity_share": round(float(a[inf].sum() / a.sum()), 4), "in_free_opaque": int((inf & (a > 0.5)).sum())}
    json.dump(meta, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print("FREESPACE " + json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
