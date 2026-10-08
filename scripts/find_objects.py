"""Find free-standing objects (blocks, boxes, ramps, ...) in a static metric LiDAR cloud (CPU).

Objects for completion are compact things standing on the floor that the robot drove around: the
cloud's points between --min-h and --max-h above the floor are binned on a --cell grid in the floor
plane, occupied cells are grouped into connected components, and a component is an object if
  - its footprint diagonal is within [--min-size, --max-size] and it is at least --min-height tall,
  - (almost) nothing in its footprint rises above --max-h (shelves, desks, walls are excluded),
  - it lies within --max-dist of the camera path (inside the area that was recorded).
Uses the dynamic-free cloud (remove_dynamic_lidar.py) so people and the robot are not objects.

Writes <out>.npz (per object: its points, metric) and <out>.json (floor plane, per-object footprint,
height, point count) and a top-down preview <out>.png.

    python scripts/find_objects.py --cloud lidar_static.ply --gt-model <out>/colmap_train/sparse/0 --out objects
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floor_plane import find_floor_plane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--gt-model", required=True, help="metric camera model (floor detection, path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--min-h", type=float, default=0.03)
    ap.add_argument("--max-h", type=float, default=1.2)
    ap.add_argument("--min-size", type=float, default=0.08)
    ap.add_argument("--max-size", type=float, default=2.0)
    ap.add_argument("--min-height", type=float, default=0.08, help="lower things are floor tape / mats")
    ap.add_argument("--max-dist", type=float, default=3.0)
    ap.add_argument("--min-points", type=int, default=150)
    args = ap.parse_args()
    import open3d as o3d, pycolmap
    from scipy import ndimage
    from scipy.spatial import cKDTree

    C = np.array([np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(args.gt_model).images.values()])
    pc = o3d.io.read_point_cloud(args.cloud); X = np.asarray(pc.points)
    n, d, _ = find_floor_plane(pc, C, verbose=False)
    h = X @ n + d
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    uv = np.stack([X @ e1, X @ e2], 1)
    band = (h > args.min_h) & (h < args.max_h)
    tall = (h >= args.max_h) & (h < 2.5)
    lo = uv[band].min(0) - 0.1; shape = np.ceil((uv[band].max(0) + 0.1 - lo) / args.cell).astype(int) + 1
    ij = lambda m: np.floor((uv[m] - lo) / args.cell).astype(int)
    occ = np.zeros(shape, bool); bi = ij(band); ok = (bi >= 0).all(1) & (bi < shape).all(1); occ[bi[ok, 0], bi[ok, 1]] = True
    hi = np.zeros(shape, bool); ti = ij(tall); ok = (ti >= 0).all(1) & (ti < shape).all(1); hi[ti[ok, 0], ti[ok, 1]] = True
    lab, nlab = ndimage.label(ndimage.binary_closing(occ, iterations=1))
    path_uv = np.stack([C @ e1, C @ e2], 1); ptree = cKDTree(path_uv)
    cell_of = np.full(len(X), -1); bidx = np.nonzero(band)[0]; bi = ij(band)
    ok = (bi >= 0).all(1) & (bi < shape).all(1); cell_lab = np.zeros(len(bidx), int); cell_lab[ok] = lab[bi[ok, 0], bi[ok, 1]]
    objs, pts = [], {}
    for k in range(1, nlab + 1):
        cells = lab == k
        if (cells & hi).sum() > 0.02 * cells.sum():
            continue                                                  # part of something tall
        ii, jj = np.nonzero(cells)
        span = np.array([ii.max() - ii.min() + 1, jj.max() - jj.min() + 1]) * args.cell
        diag = float(np.hypot(*span))
        if not (args.min_size <= diag <= args.max_size):
            continue
        cuv = lo + (np.array([ii.mean(), jj.mean()]) + 0.5) * args.cell
        if ptree.query(cuv)[0] > args.max_dist:
            continue
        idx = bidx[cell_lab == k]
        if len(idx) < args.min_points or h[idx].max() < args.min_height:
            continue
        oid = len(objs)
        pts[f"obj{oid}"] = X[idx]
        objs.append({"id": oid, "points": int(len(idx)), "footprint_m": span.round(3).tolist(),
                     "height_m": round(float(h[idx].max()), 3), "centre_m": (X[idx].mean(0)).round(4).tolist()})
    np.savez_compressed(args.out + ".npz", **pts)
    json.dump({"floor": {"n": n.tolist(), "d": float(d)}, "axes": {"e1": e1.tolist(), "e2": e2.tolist()},
               "params": {k: v for k, v in vars(args).items() if k not in ("cloud", "gt_model", "out")}, "objects": objs},
              open(args.out + ".json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(occ.T, origin="lower", cmap="Greys", extent=[lo[0], lo[0] + shape[0] * args.cell, lo[1], lo[1] + shape[1] * args.cell])
        ax.plot(path_uv[:, 0], path_uv[:, 1], "c-", lw=1)
        for o in objs:
            c = np.array(o["centre_m"]); ax.annotate(str(o["id"]), (c @ e1, c @ e2), color="r", fontsize=11, weight="bold")
        ax.set_title(f"{len(objs)} objects (red ids), robot path (cyan)"); fig.savefig(args.out + ".png", dpi=90)
    except Exception as ex:  # noqa: BLE001
        print("[objects] preview skipped:", ex)
    print(f"[objects] {len(objs)} free-standing objects: " +
          ", ".join(f"#{o['id']} {o['footprint_m'][0]:.2f}x{o['footprint_m'][1]:.2f}x{o['height_m']:.2f}m" for o in objs), flush=True)


if __name__ == "__main__":
    main()
