"""Phase 3 — fuse RealSense depth into one metric map; seed points3D (docs/PLAN.md §6).

Depth is the SOLE geometry source in v3 (no LiDAR). This script:
  1. motion-gates depth frames (‖ω‖ criterion, TRAP 6) + enforces spatial spread;
  2. truncates depth to [min,max] range (§2.7) to bound D455 noise growth;
  3. TSDF-fuses the gated frames (Open3D ScalableTSDFVolume), posed by pose_utils;
  4. extracts a point cloud (-> points3D.txt seed) and a mesh (-> Phase 5 head start).

    python scripts/build_depth_map.py --config configs/mocap1_well-lit_trot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
import gating_utils as gu
from build_poses import compute_world_cam, depth_poses


def fuse(cfg, frames, Twc, calib, indices):
    """TSDF-fuse the frames at `indices` and return legacy (pcd, mesh).

    Uses Open3D's tensor VoxelBlockGrid (marching cubes) — the maintained TSDF path;
    the legacy ScalableTSDFVolume mis-scales depth in current builds. Depth is fed as the
    native 16-bit image with depth_scale = units_per_meter, so the config's Phase-1
    VERIFY of the depth units flows straight through.
    """
    seq = cfg["sequence"]
    root = seq["data_root"]
    depth_dir = os.path.join(root, seq["depth_dir"])
    rgb_dir = os.path.join(root, seq["rgb_dir"])
    f = cfg["fusion"]
    dcfg = cfg["depth"]
    depth_scale = float(dcfg["units_per_meter"])
    depth_max = float(dcfg["max_range_m"])
    invalid = dcfg.get("invalid_value", 0)
    dev = o3c.Device("CPU:0")
    fx, fy, cx, cy = iu.K_params(calib.K)
    intr_t = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=float(f["voxel_size_m"]),
        block_resolution=int(f.get("block_resolution", 16)),
        block_count=int(f.get("block_count", 100000)),
        device=dev)

    for i in indices:
        depth_raw = iu.read_depth_raw(os.path.join(depth_dir, frames[i].depth_name))
        _dm, valid = iu.depth_to_meters(depth_raw, cfg, truncate=True)
        du16 = np.where(valid, depth_raw, invalid).astype(np.uint16)
        rgb = iu.undistort_rgb(iu.read_rgb(os.path.join(rgb_dir, frames[i].rgb_name)),
                               calib.K, calib.dist)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(du16), device=dev))
        color_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb), device=dev))
        ext_t = o3c.Tensor(np.ascontiguousarray(pu.invert_T(Twc[i])), o3c.float64)
        coords = vbg.compute_unique_block_coordinates(depth_t, intr_t, ext_t,
                                                      depth_scale, depth_max)
        vbg.integrate(coords, depth_t, color_t, intr_t, intr_t, ext_t,
                      depth_scale, depth_max)

    pcd = vbg.extract_point_cloud().to_legacy()
    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    return pcd, mesh


def build(cfg: dict) -> dict:
    out_root = cfg["paths"]["out_root"]
    dmap_dir = os.path.join(out_root, "depth_map")
    os.makedirs(dmap_dir, exist_ok=True)

    kept, Twc, calib, interp, mocap, _ = compute_world_cam(cfg)
    Twc_depth = depth_poses(cfg, kept, mocap, interp, calib)   # TRAP 7: pose at depth time
    ts = [f.t for f in kept]
    centers = Twc[:, :3, 3]

    g = cfg["gating"]
    keep, info = gu.motion_gate(cfg, ts, centers,
                                omega_percentile_keep=float(g["omega_percentile_keep"]),
                                min_sep=float(g["spatial_min_sep_m"]))
    keep, n_preamble = gu.apply_preamble_exclusion(cfg, ts, keep)   # TRAP 8
    gated = list(np.where(keep)[0])
    if len(gated) < 3:
        raise RuntimeError(f"only {len(gated)} depth frames survived gating — loosen "
                           "gating.omega_percentile_keep / spatial_min_sep_m")

    pcd, mesh = fuse(cfg, kept, Twc_depth, calib, gated)
    o3d.io.write_point_cloud(os.path.join(dmap_dir, "fused_cloud.ply"), pcd)
    o3d.io.write_triangle_mesh(os.path.join(dmap_dir, "fused_mesh.ply"), mesh)

    # --- seed points3D.txt (§6.2): voxel-downsample the fused cloud ---
    seed = pcd.voxel_down_sample(float(cfg["fusion"]["seed_downsample_m"]))
    pts = np.asarray(seed.points)
    if seed.has_colors():
        c = np.asarray(seed.colors)
        if c.max() <= 1.001:                       # legacy convention is [0,1]
            c = c * 255.0
        cols = np.clip(c, 0, 255).astype(np.int64)
    else:
        cols = np.full((len(pts), 3), 128, dtype=np.int64)
    model_dir = os.path.join(out_root, "colmap", "sparse", "0")
    if os.path.exists(os.path.join(model_dir, "images.txt")):
        pu.write_points3D_txt(model_dir, pts, cols)

    meta = {
        "n_depth_frames_total": len(kept),
        "n_depth_frames_gated": len(gated),
        "n_excluded_preamble": int(n_preamble),
        "omega_threshold_rad_s": info["omega_threshold"],
        "truncation_range_m": [cfg["depth"]["min_range_m"], cfg["depth"]["max_range_m"]],
        "voxel_size_m": cfg["fusion"]["voxel_size_m"],
        "n_fused_points": int(len(pcd.points)),
        "n_seed_points": int(len(pts)),
        "n_mesh_triangles": int(len(mesh.triangles)),
        "cloud_ply": os.path.join(dmap_dir, "fused_cloud.ply"),
        "mesh_ply": os.path.join(dmap_dir, "fused_mesh.ply"),
        "seeded_points3D": os.path.join(model_dir, "points3D.txt"),
    }
    pu.save_json(os.path.join(dmap_dir, "fusion_meta.json"), meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    m = build(cfg)
    print(f"[build_depth_map] gated {m['n_depth_frames_gated']}/{m['n_depth_frames_total']} "
          f"depth frames (‖ω‖ ≤ {m['omega_threshold_rad_s']:.3f} rad/s; "
          f"{m['n_excluded_preamble']} excluded as sync preamble)")
    print(f"[build_depth_map] fused points={m['n_fused_points']}  "
          f"mesh tris={m['n_mesh_triangles']}  seed points={m['n_seed_points']}")
    print(f"[build_depth_map] truncation range {m['truncation_range_m']} m, "
          f"voxel {m['voxel_size_m']} m")
    print(f"[build_depth_map] seeded {m['seeded_points3D']}")


if __name__ == "__main__":
    main()
