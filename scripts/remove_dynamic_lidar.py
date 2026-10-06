"""Remove moving things (people, the robot's own body) from an accumulated LiDAR cloud by visibility (CPU).

Accumulating every scan bakes in anything that was there for a moment: the operator walking around,
the robot's legs. Those points later sit in front of the camera, in the splat init, the depth
targets and the collider. Static geometry is hit again and again from different places; a passer-by
is "seen through" by every later sweep. So, Removert/OctoMap-style, for every scan:
  1. build its range image (rings x azimuth bins, nearest return per bin) in the sensor frame,
  2. for every point of the cloud in range, look up the scan's measured range in that direction:
       measured > point range + margin   -> the beam passed through the point: FREE vote
       |measured - point range| <= margin -> the beam hit it: HIT vote
A point is removed when it collects at least --min-free FREE votes and more than --ratio times as
many FREE as HIT votes. Points no beam ever passed (occluded, out of range) are kept.
Ring elevations are read off the data, so any spinning multi-beam LiDAR works.

    python scripts/remove_dynamic_lidar.py --config configs/<seq>.yaml --cloud <pipeline>/depth_cloud.ply \
        --out <dir>/lidar_static.ply
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from build_lidar_cloud import registered_scans


def ring_centres(scans, n=20):
    """Beam elevations (deg) from the data: peaks of the elevation histogram of a few scans."""
    el = np.concatenate([np.degrees(np.arcsin(p[:, 2] / np.linalg.norm(p, axis=1))) for _, p in scans[:n]])
    h, e = np.histogram(el, bins=np.arange(-90, 90.05, 0.1))
    peak = (h > 0.2 * h.max() / 4) & (h >= np.roll(h, 1)) & (h >= np.roll(h, -1))
    c = (e[:-1] + 0.05)[peak]
    out = []
    for x in c:                                                   # merge neighbouring bins of one ring
        if out and x - out[-1][-1] < 0.5:
            out[-1].append(x)
        else:
            out.append([x])
    return np.array([np.mean(g) for g in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cloud", required=True, help="accumulated metric cloud to clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--az-deg", type=float, default=0.4, help="azimuth bin of the range image")
    ap.add_argument("--margin-m", type=float, default=0.10, help="+ 1%% of range")
    ap.add_argument("--min-free", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan")
    args = ap.parse_args()
    import open3d as o3d

    cfg = pu.load_config(args.config)
    scans = registered_scans(cfg, stride=args.stride, log=lambda m: print(m, flush=True))
    rings = ring_centres(scans)
    tol = 0.4 * (np.min(np.diff(rings)) if len(rings) > 1 else 1.0)
    max_r = float(cfg["lidar"].get("max_range_m", 12.0))
    print(f"[dynamic] {len(rings)} rings at {np.round(rings, 1).tolist()} deg; {len(scans)} scans", flush=True)

    pc = o3d.io.read_point_cloud(args.cloud)
    X = np.asarray(pc.points).astype(np.float32)
    free = np.zeros(len(X), np.uint16); hit = np.zeros(len(X), np.uint16)
    naz = int(round(360 / args.az_deg)); t0 = time.time()
    for k, (T, p) in enumerate(scans):
        # range image of this scan
        r = np.linalg.norm(p, axis=1)
        el = np.degrees(np.arcsin(p[:, 2] / r)); az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360
        ri = np.abs(el[:, None] - rings[None]).argmin(1); ok = np.abs(el - rings[ri]) < tol
        img = np.full((len(rings), naz), np.inf, np.float32)
        np.minimum.at(img, (ri[ok], (az[ok] / args.az_deg).astype(int) % naz), r[ok].astype(np.float32))
        # every cloud point in range, in this scan's sensor frame
        R, t = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
        Q = (X - t) @ R
        rq = np.linalg.norm(Q, axis=1)
        idx = np.nonzero((rq > 0.3) & (rq < max_r))[0]
        Q, rq = Q[idx], rq[idx]
        elq = np.degrees(np.arcsin(Q[:, 2] / rq))
        rr = np.searchsorted(rings, elq).clip(1, len(rings) - 1)
        rr = np.where(np.abs(elq - rings[rr - 1]) < np.abs(elq - rings[rr]), rr - 1, rr)
        okq = np.abs(elq - rings[rr]) < tol
        idx, rq, rr, Q = idx[okq], rq[okq], rr[okq], Q[okq]
        azq = (np.degrees(np.arctan2(Q[:, 1], Q[:, 0])) % 360 / args.az_deg).astype(int) % naz
        m = img[rr, azq]
        seen = np.isfinite(m); marg = args.margin_m + 0.01 * rq
        free[idx[seen & (m > rq + marg)]] += 1
        hit[idx[seen & (np.abs(m - rq) <= marg)]] += 1
        if k % 100 == 0:
            print(f"[dynamic] scan {k}/{len(scans)}  ({time.time() - t0:.0f} s)", flush=True)
    dyn = (free >= args.min_free) & (free > args.ratio * hit)
    out = pc.select_by_index(np.nonzero(~dyn)[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, out)
    info = {"input_points": int(len(X)), "removed_dynamic": int(dyn.sum()), "never_tested": int(((free + hit) == 0).sum()),
            "rings": len(rings), "scans": len(scans), "min_free": args.min_free, "ratio": args.ratio,
            "margin_m": args.margin_m, "az_deg": args.az_deg}
    json.dump(info, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print(f"[dynamic] removed {dyn.sum():,} of {len(X):,} points seen through by later scans -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
