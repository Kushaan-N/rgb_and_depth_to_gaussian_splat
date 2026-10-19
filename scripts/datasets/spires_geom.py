#!/usr/bin/env python3
"""Geometry of an exported splat vs a ground-truth laser scan (TLS), metric frame (CPU venv).

points  Spires reconstruction-benchmark style (Tao et al. 2025, Sec. 6.2): splat centres with opacity >= --min-opacity
        (the surface the splat draws) vs the TLS cloud: precision / recall / F-score at 5 and 10 cm (a point is a
        true positive if its nearest neighbour in the other cloud is within the threshold), accuracy (mean
        splat->TLS distance) and completeness (mean TLS->splat distance). Both clouds are restricted to the region
        the rig observed: within --max-cam-dist of a training camera centre and inside the TLS crop (their protocol
        instead filters by the TLS occupancy octree; the LiDAR cloud scored the same way is the reference row).
        --ref-cloud also scores a plain point cloud (e.g. the merged LiDAR) the same way.
depth   rendered depth at held-out cameras vs the TLS z-buffered into the same cameras: per-pixel ray distance from
        render_ply_depth.py (<stem>_splatdepth.npy) vs build_depth_targets.py --cloud tls.ply (<stem>_depth.npy in
        the dataset's images/): AbsRel, median |err|, share within 5 / 10 cm, coverage (pixels with a target where the
        splat is opaque).

    python scripts/datasets/spires_geom.py points --ply S.ply --tls tls.ply --cams <fisheye/train> [--ref-cloud lidar.ply]
    python scripts/datasets/spires_geom.py depth --pred-dir D --dataset <pinhole test set> --json-out J
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def read_ply_xyz_opacity(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]; names = [p.name for p in v.properties]
    P = np.stack([np.asarray(v[k], np.float64) for k in "xyz"], 1)
    o = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64))) if "opacity" in names else np.ones(len(P))
    return P, o


def cam_centres(ds):
    from rectify_views import read_images, qvec2rot
    return np.array([-qvec2rot(q).T @ t for _, q, t, _, _ in read_images(os.path.join(ds, "sparse", "0", "images.txt"))])


def pr_scores(rec, gt, ths=(0.05, 0.10)):
    from scipy.spatial import cKDTree
    d_rg = cKDTree(gt).query(rec, workers=-1)[0]; d_gr = cKDTree(rec).query(gt, workers=-1)[0]
    out = {"n_rec": int(len(rec)), "n_gt": int(len(gt)), "accuracy_m": float(d_rg.mean()), "completeness_m": float(d_gr.mean()),
           "accuracy_median_m": float(np.median(d_rg)), "completeness_median_m": float(np.median(d_gr))}
    for th in ths:
        p, r = float((d_rg < th).mean()), float((d_gr < th).mean())
        k = f"{int(round(th * 100))}cm"
        out[f"precision_{k}"], out[f"recall_{k}"], out[f"fscore_{k}"] = p, r, (2 * p * r / (p + r) if p + r else 0.0)
    return out


def cmd_points(a):
    from scipy.spatial import cKDTree
    gt = read_ply_xyz_opacity(a.tls)[0]; C = cam_centres(a.cams)
    keep = lambda X: (cKDTree(C).query(X, workers=-1)[0] < a.max_cam_dist) & np.all((X >= gt.min(0)) & (X <= gt.max(0)), 1)
    gt = gt[keep(gt)]
    if a.max_gt_points and len(gt) > a.max_gt_points:
        gt = gt[np.random.default_rng(0).choice(len(gt), a.max_gt_points, replace=False)]
    res = {"region": {"max_cam_dist_m": a.max_cam_dist, "n_cams": int(len(C))}}
    if a.ply:
        P, o = read_ply_xyz_opacity(a.ply); P = P[o >= a.min_opacity]; P = P[keep(P)]
        res["splat"] = {"ply": a.ply, "min_opacity": a.min_opacity, **pr_scores(P, gt)}
    if a.ref_cloud:
        R = read_ply_xyz_opacity(a.ref_cloud)[0]; R = R[keep(R)]
        res["ref_cloud"] = {"ply": a.ref_cloud, **pr_scores(R, gt)}
    for k in ("splat", "ref_cloud"):
        if k in res:
            r = res[k]
            print(f"[geom] {k:9s} acc {r['accuracy_m']:.3f} m  comp {r['completeness_m']:.3f} m  "
                  f"P/R/F@5cm {r['precision_5cm']:.3f}/{r['recall_5cm']:.3f}/{r['fscore_5cm']:.3f}  "
                  f"P/R/F@10cm {r['precision_10cm']:.3f}/{r['recall_10cm']:.3f}/{r['fscore_10cm']:.3f}  (n {r['n_rec']})")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)


def cmd_depth(a):
    errs, rel, cov_n, cov_d = [], [], 0, 0
    for f in sorted(glob.glob(os.path.join(a.dataset, "images", "*_depth.npy"))):
        stem = os.path.basename(f)[:-len("_depth.npy")]
        pf = os.path.join(a.pred_dir, stem + "_splatdepth.npy")
        if not os.path.exists(pf):
            continue
        gt = np.load(f).astype(np.float32); pr = np.load(pf).astype(np.float32)
        m = gt > 0; cov_d += int(m.sum()); m &= pr > 0; cov_n += int(m.sum())
        e = np.abs(pr[m] - gt[m]); errs.append(e); rel.append(e / gt[m])
    if not errs:
        raise SystemExit("no matching depth files")
    e, r = np.concatenate(errs), np.concatenate(rel)
    res = {"n_px": int(len(e)), "coverage": cov_n / max(cov_d, 1), "absrel": float(r.mean()), "median_abs_m": float(np.median(e)),
           "within_5cm": float((e < 0.05).mean()), "within_10cm": float((e < 0.10).mean()), "within_5pct": float((r < 0.05).mean())}
    print(f"[geom] depth vs TLS: AbsRel {res['absrel']:.4f}  median {res['median_abs_m'] * 100:.1f} cm  <5cm {res['within_5cm']:.3f}  "
          f"<10cm {res['within_10cm']:.3f}  coverage {res['coverage']:.3f}")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    p = s.add_parser("points"); p.add_argument("--ply"); p.add_argument("--tls", required=True); p.add_argument("--cams", required=True)
    p.add_argument("--ref-cloud"); p.add_argument("--min-opacity", type=float, default=0.5)
    p.add_argument("--max-cam-dist", type=float, default=20.0); p.add_argument("--max-gt-points", type=int, default=8_000_000)
    p.add_argument("--json-out")
    d = s.add_parser("depth"); d.add_argument("--pred-dir", required=True); d.add_argument("--dataset", required=True)
    d.add_argument("--json-out")
    a = ap.parse_args()
    {"points": cmd_points, "depth": cmd_depth}[a.cmd](a)


if __name__ == "__main__":
    main()
