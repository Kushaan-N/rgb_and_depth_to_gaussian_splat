"""Deterministic, frame-agnostic floor-plane detection shared by the floor tools (fill + coverage).

Never trust the frame's declared up axis — on CEAR the pipeline's "Z-up" metric frame is actually
Z-DOWN (cameras at z=-0.43 above a floor at z=0), which silently inverted every "above the floor"
test. Instead derive "up" from the data:

  floor  = the LARGEST plane that the cameras ride consistently above:
           >=95% of cameras on one side, median camera height in [0.05, 2.0] m, and a small height
           spread (a ground robot's camera height barely changes; walls fail this, the robot moves
           toward and away from them). Tables pass the height test but lose on size; ceilings are
           excluded by the 2 m cap for ground robots.
  up     = the floor normal pointing toward the cameras (heights h = P @ n + d are then + upward)

Seeded RANSAC proposes planes; the winner is refined by least squares on its inliers, because a
1-degree tilt moves floor height by ~14 cm across an 8 m room.
"""
from __future__ import annotations
import numpy as np


def _refine(P, n, d, thr, iters=3):
    for _ in range(iters):
        Q = P[np.abs(P @ n + d) < thr]
        if len(Q) < 3:
            break
        c = Q.mean(0)
        n2 = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
        n = n2 if n2 @ n >= 0 else -n2
        d = -float(n @ c)
    return n, d, np.nonzero(np.abs(P @ n + d) < thr)[0]


def find_floor_plane(pc, cam_centers, thr=0.03, seed=0, min_frac=0.02,
                     cam_height=(0.05, 2.0), max_spread=0.30, declared_up=(0.0, 0.0, 1.0), verbose=True):
    """Return (n_up, d, inlier_idx) of the floor with n_up pointing toward the cameras, or None."""
    import open3d as o3d
    if isinstance(pc, np.ndarray):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    o3d.utility.random.seed(seed)
    P = np.asarray(pc.points)
    cams = np.asarray(cam_centers, dtype=np.float64)
    work = o3d.geometry.PointCloud(pc); best = None

    def cam_stats(n, d):
        h = cams @ n + d
        if np.median(h) < 0:
            n, d, h = -n, -d, -h
        q1, q3 = np.percentile(h, [25, 75])
        return n, d, float((h > 0).mean()), float(np.median(h)), float(q3 - q1)

    for _ in range(10):
        if len(work.points) < 1000:
            break
        model, inl = work.segment_plane(thr, 3, 2000)
        n = np.array(model[:3], dtype=np.float64); d = float(model[3]); k = np.linalg.norm(n)
        n, d, side, med, spread = cam_stats(n / k, d / k)
        ok = side >= 0.95 and cam_height[0] <= med <= cam_height[1] and spread <= max_spread
        if ok and (best is None or len(inl) > best[2]):
            best = (n, d, len(inl))
        work = work.select_by_index(inl, invert=True)
    if best is None:
        return None
    n, d, inl = _refine(P, best[0], best[1], thr)
    n, d, side, med, spread = cam_stats(n, d)
    inl = np.nonzero(np.abs(P @ n + d) < thr)[0]
    if len(inl) < min_frac * len(P):
        return None
    if verbose:
        print(f"[floor] {len(inl):,} inliers ({len(inl)/len(P):.1%}); camera height {med:.3f} m "
              f"(spread {spread:.3f}); up = {np.round(n, 3).tolist()}", flush=True)
        if declared_up is not None and n @ np.asarray(declared_up, float) < 0:
            print(f"[floor] WARNING: data says up = {np.round(n, 3).tolist()}, OPPOSITE to the frame's "
                  f"declared up {list(declared_up)} — floor tools use the data; check other consumers.", flush=True)
    return n, d, inl
