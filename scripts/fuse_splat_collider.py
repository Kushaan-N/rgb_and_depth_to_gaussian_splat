#!/usr/bin/env python3
"""Option A, stage 2 — TSDF-fuse splat-rendered depth into a watertight collider (CPU/Open3D).

Reads the per-frame (z-depth, camera-to-world, K) written by render_splat_depth.py, TSDF-fuses
them with Open3D's tensor VoxelBlockGrid (same path as build_depth_map), then gravity-aligns the
mesh (largest planar surface = floor -> +Z, scene above, floor at z=0) so it drops into Isaac
physics. Fully automatic; no hand-placed geometry.

    python fuse_splat_collider.py --depth-dir <splat_depth> --out <collider_dir> [--voxel 0.03]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import open3d as o3d
import open3d.core as o3c


def align_gravity(pcd, mesh):
    """Rotate so the dominant plane (floor) normal -> +Z with the scene above; return floor_z."""
    work = pcd
    best = None                                  # (n_inliers, normal, height_pts)
    allpts = np.asarray(pcd.points)
    for _ in range(6):
        if len(work.points) < 200:
            break
        model, inl = work.segment_plane(0.03, ransac_n=3, num_iterations=1200)
        if best is None or len(inl) > best[0]:
            a, b, c, d = model
            n = np.array([a, b, c], float); n /= (np.linalg.norm(n) + 1e-9)
            best = (len(inl), n, d / (np.linalg.norm([a, b, c]) + 1e-9))
        work = work.select_by_index(inl, invert=True)
    if best is None:
        return mesh, pcd, 0.0
    n = best[1]
    # orient normal so the scene centroid is on the +normal side (floor points up)
    centroid = allpts.mean(0)
    if np.dot(centroid, n) + best[2] < 0:
        n = -n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); cth = float(np.dot(n, z))
    if s < 1e-6:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s * s))
    mesh.rotate(R, center=(0, 0, 0)); pcd.rotate(R, center=(0, 0, 0))
    zr = np.asarray(pcd.points)[:, 2]
    floor_z = float(np.percentile(zr, 2))        # robust floor height (2nd percentile)
    mesh.translate((0, 0, -floor_z)); pcd.translate((0, 0, -floor_z))
    return mesh, pcd, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--depth-max", type=float, default=8.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(args.depth_dir, "*.npz")))
    if not frames:
        print("no depth frames"); return 2
    dev = o3c.Device("CPU:0")
    K = np.load(frames[0])["K"]
    fx, fy, cx, cy = [float(v) for v in K]
    intr = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"), attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)), voxel_size=args.voxel, block_resolution=16,
        block_count=200000, device=dev)
    used = 0
    for fp in frames:
        d = np.load(fp)
        depth_m = d["depth"].astype(np.float32)
        du16 = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)      # mm, depth_scale=1000
        c2w = d["c2w"].astype(np.float64)
        ext = o3c.Tensor(np.ascontiguousarray(np.linalg.inv(c2w)), o3c.float64)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(du16), device=dev))
        try:
            coords = vbg.compute_unique_block_coordinates(depth_t, intr, ext, 1000.0, args.depth_max)
            vbg.integrate(coords, depth_t, intr, ext, 1000.0, args.depth_max)
            used += 1
        except Exception as e:  # noqa: BLE001
            print(f"  integrate {os.path.basename(fp)} failed: {e}")
    mesh = vbg.extract_triangle_mesh().to_legacy()
    pcd = vbg.extract_point_cloud().to_legacy()
    mesh.compute_vertex_normals()
    print(f"[fuse] {used}/{len(frames)} frames -> mesh V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
    mesh, pcd, floor_z = align_gravity(pcd, mesh)
    V = np.asarray(mesh.vertices)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "collider.obj"), mesh)
    meta = {"source": "splat_novel_view_depth_fusion", "frames_used": used,
            "voxel_m": args.voxel, "floor_z_m": float(floor_z),
            "bbox_min": V.min(0).round(3).tolist(), "bbox_max": V.max(0).round(3).tolist(),
            "n_vertices": len(V), "n_faces": len(mesh.triangles)}
    json.dump(meta, open(os.path.join(args.out, "collider_meta.json"), "w"), indent=2)
    print(f"[fuse] wrote collider.obj (floor_z={floor_z:.3f}, "
          f"bbox z [{V[:,2].min():.2f},{V[:,2].max():.2f}]) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
