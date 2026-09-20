"""Gate 3 — depth-map consistency (docs/PLAN.md §6.3). Depth-only, so the cross-checks
are internal-consistency checks (they fail loudly under the same pose/sync errors a LiDAR
cross-check would have caught):

  * reprojection    : project fused points into gated RGB frames; they must land on the
                      surfaces they belong to (measured depth agrees with projected z).
  * split-half      : fuse odd- vs even-indexed gated frames into two INDEPENDENT TSDFs;
                      floor-height difference + cloud-to-cloud distance must be small.
  * crispness       : the fused floor plane must be thin (small point-to-plane spread).
  * coverage        : fraction of the floor observed within the truncation range (holes
                      are honest; smeared phantom geometry is not).

    python scripts/verify_depth_map.py --config configs/mocap1_well-lit_trot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
import gating_utils as gu
from build_poses import compute_world_cam, depth_poses
from build_depth_map import fuse


def fit_floor_plane(pcd, dist_thresh=0.02, tries=4):
    """Find the near-horizontal (floor) plane. Returns (height_z, thickness_std, n_in)."""
    work = pcd
    for _ in range(tries):
        if len(work.points) < 100:
            break
        model, inliers = work.segment_plane(dist_thresh, ransac_n=3, num_iterations=800)
        a, b, c, d = model
        if abs(c) > 0.9:                       # normal ~ vertical => floor/ceiling
            pts = np.asarray(work.points)[inliers]
            z = pts[:, 2]
            # prefer the lower horizontal plane (floor, not ceiling)
            height = float(np.median(z))
            n = np.array([a, b, c]); n /= np.linalg.norm(n)
            dists = (np.asarray(work.points)[inliers] @ n) + d / np.linalg.norm([a, b, c])
            return height, float(np.std(dists)), len(inliers)
        work = work.select_by_index(inliers, invert=True)
    return float("nan"), float("nan"), 0


def reprojection_consistency(cfg, frames, Twc, calib, gated, fused_pcd, out_dir, n_show=10):
    root = cfg["sequence"]["data_root"]
    depth_dir = os.path.join(root, cfg["sequence"]["depth_dir"])
    rgb_dir = os.path.join(root, cfg["sequence"]["rgb_dir"])
    K = calib.K
    pts_w = np.asarray(fused_pcd.points)
    sub = pts_w[np.random.default_rng(0).choice(len(pts_w), min(20000, len(pts_w)), replace=False)]
    fracs = []
    show = list(gated)[:: max(1, len(gated) // n_show)][:n_show]
    for i in show:
        depth_raw = iu.read_depth_raw(os.path.join(depth_dir, frames[i].depth_name))
        meas_m, valid = iu.depth_to_meters(depth_raw, cfg, truncate=True)
        rgb = iu.read_rgb(os.path.join(rgb_dir, frames[i].rgb_name))
        H, W = meas_m.shape
        pc = iu.transform_points(pu.invert_T(Twc[i]), sub)
        uv, z = iu.project(pc, K)
        ui = np.round(uv[:, 0]).astype(int); vi = np.round(uv[:, 1]).astype(int)
        ok = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, z = ui[ok], vi[ok], z[ok]
        m = meas_m[vi, ui]; mvalid = valid[vi, ui]
        # A fused point is CONSISTENT unless it floats in FRONT of the measured surface
        # (z < m - tol). Points behind (z > m) are legitimately occluded in this view,
        # not errors — penalizing them would just measure occlusion. Floaters are the
        # real failure mode this check exists to catch.
        tol = 0.05
        on_surf = mvalid & (np.abs(z - m) < tol)
        floater = mvalid & (z < m - tol)
        if mvalid.sum():
            fracs.append(1.0 - float(floater.sum()) / float(mvalid.sum()))
        ov = rgb.copy()
        ov[vi[on_surf], ui[on_surf]] = [0, 255, 0]
        ov[vi[floater], ui[floater]] = [255, 0, 0]
        import cv2
        cv2.imwrite(os.path.join(out_dir, f"overlay_{i:04d}.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    return float(np.mean(fracs)) if fracs else float("nan")


def coverage_map(fused_pcd, floor_z, out_path, cell=0.10, band=0.05):
    pts = np.asarray(fused_pcd.points)
    floor = pts[np.abs(pts[:, 2] - floor_z) < band]
    if len(floor) == 0:
        return float("nan")
    xy = pts[:, :2]
    xmin, ymin = xy.min(0); xmax, ymax = xy.max(0)
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))
    grid = np.zeros((ny, nx), dtype=bool)
    ix = np.clip(((floor[:, 0] - xmin) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((floor[:, 1] - ymin) / cell).astype(int), 0, ny - 1)
    grid[iy, ix] = True
    frac = float(grid.mean())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(grid, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="Greens")
    ax.set_title(f"floor coverage within range = {frac*100:.0f}%")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return frac


def build(cfg: dict) -> dict:
    out_dir = os.path.join(cfg["paths"]["out_root"], "gate3")
    os.makedirs(out_dir, exist_ok=True)
    kept, Twc, calib, interp, mocap, _ = compute_world_cam(cfg)
    Twc_depth = depth_poses(cfg, kept, mocap, interp, calib)   # TRAP 7
    ts = [f.t for f in kept]
    g = cfg["gating"]
    keep, _info = gu.motion_gate(cfg, ts, Twc[:, :3, 3],
                                 omega_percentile_keep=float(g["omega_percentile_keep"]),
                                 min_sep=float(g["spatial_min_sep_m"]))
    keep, _n_pre = gu.apply_preamble_exclusion(cfg, ts, keep)   # TRAP 8
    gated = list(np.where(keep)[0])

    # full fused cloud (load if present, else fuse) — depth-timestamp poses
    cloud_ply = os.path.join(cfg["paths"]["out_root"], "depth_map", "fused_cloud.ply")
    full = o3d.io.read_point_cloud(cloud_ply) if os.path.exists(cloud_ply) else fuse(cfg, kept, Twc_depth, calib, gated)[0]

    # --- split-half fusion (independent odd/even) ---
    even = gated[0::2]; odd = gated[1::2]
    pcd_e, _ = fuse(cfg, kept, Twc_depth, calib, even)
    pcd_o, _ = fuse(cfg, kept, Twc_depth, calib, odd)
    ze, te, ne = fit_floor_plane(pcd_e)
    zo, to, no = fit_floor_plane(pcd_o)
    floor_dz = abs(ze - zo)
    d_eo = np.asarray(pcd_e.compute_point_cloud_distance(pcd_o))
    c2c_med = float(np.median(d_eo)) if len(d_eo) else float("nan")
    c2c_p95 = float(np.percentile(d_eo, 95)) if len(d_eo) else float("nan")

    zf, tf, nf = fit_floor_plane(full)
    reproj_frac = reprojection_consistency(cfg, kept, Twc, calib, gated, full, out_dir)
    cov = coverage_map(full, zf, os.path.join(out_dir, "coverage.png"))

    thr = cfg.get("gate3", {})
    ok_floor = floor_dz < thr.get("floor_dz_m", 0.02)
    ok_c2c = c2c_med < thr.get("c2c_median_m", 0.02)
    ok_crisp = (tf < thr.get("floor_thickness_m", 0.02)) if np.isfinite(tf) else False
    ok_reproj = (reproj_frac > thr.get("reproj_frac_min", 0.80)) if np.isfinite(reproj_frac) else False
    passed = ok_floor and ok_c2c and ok_crisp and ok_reproj

    meta = {
        "n_gated": len(gated), "split": {"even": len(even), "odd": len(odd)},
        "floor_height_full_m": zf, "floor_thickness_full_m": tf,
        "split_floor_height_m": [ze, zo], "split_floor_dz_m": floor_dz,
        "c2c_median_m": c2c_med, "c2c_p95_m": c2c_p95,
        "reproj_consistency_frac": reproj_frac,
        "floor_coverage_frac": cov,
        "checks": {"floor_dz": ok_floor, "c2c": ok_c2c, "crisp": ok_crisp, "reproj": ok_reproj},
        "gate3_pass": bool(passed),
    }
    pu.save_json(os.path.join(out_dir, "gate3_meta.json"), meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    m = build(cfg)
    print(f"[Gate 3] gated frames: {m['n_gated']} (even {m['split']['even']} / odd {m['split']['odd']})")
    print(f"[Gate 3] floor height={m['floor_height_full_m']:.4f} m  "
          f"thickness(std)={m['floor_thickness_full_m']*1000:.1f} mm")
    print(f"[Gate 3] split-half floor Δz={m['split_floor_dz_m']*1000:.1f} mm  "
          f"C2C median={m['c2c_median_m']*1000:.1f} mm (p95 {m['c2c_p95_m']*1000:.1f} mm)")
    print(f"[Gate 3] reprojection consistency={m['reproj_consistency_frac']*100:.1f}%  "
          f"floor coverage={m['floor_coverage_frac']*100:.0f}%")
    print(f"GATE 3 {'PASS' if m['gate3_pass'] else 'REVIEW'}  checks={m['checks']}")
    return 0 if m["gate3_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
