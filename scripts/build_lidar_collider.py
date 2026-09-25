"""LiDAR world cloud -> Isaac collider (CPU/Open3D).

The Velodyne cloud (build_lidar_cloud.py) is dense, metric, and gravity-consistent, so it makes a
far better collider than the ground-level RGB-D depth. Steps:

  1. gravity-align (RANSAC dominant plane -> +Z, floor -> z=0)   [floor is already ~level, small fix]
  2. crop to the room box from the near-floor footprint (drop the 360deg long-range returns)
  3. Poisson surface reconstruction, trimmed by density (removes the empty-space extrapolation)
  4. clip the mesh to the cropped cloud's extent, drop small components, quadric-decimate

Writes collider.obj + collider_meta.json in the same Z-up frame as the splat / other colliders.

    python scripts/build_lidar_collider.py --cloud <out>/lidar_cloud.ply --out <out>/collider_lidar
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import open3d as o3d


def gravity_align(pcd):
    work = o3d.geometry.PointCloud(pcd); best = None; allpts = np.asarray(pcd.points)
    for _ in range(6):
        if len(work.points) < 200:
            break
        model, inl = work.segment_plane(0.03, 3, 1000)
        if best is None or len(inl) > best[0]:
            a, b, c, d = model; nn = np.linalg.norm([a, b, c]) + 1e-9
            best = (len(inl), np.array([a, b, c]) / nn, d / nn)
        work = work.select_by_index(inl, invert=True)
    n = best[1]
    if np.dot(allpts.mean(0), n) + best[2] < 0:
        n = -n
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); c = float(np.dot(n, z))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    pcd.rotate(R, center=(0, 0, 0))
    fz = float(np.percentile(np.asarray(pcd.points)[:, 2], 2))
    pcd.translate((0, 0, -fz))
    return pcd


def crop_room(pcd, floor_band, pct, pad, ceil):
    V = np.asarray(pcd.points)
    near = V[(V[:, 2] > -0.1) & (V[:, 2] < floor_band)]
    if len(near) < 500:
        near = V
    lo = np.percentile(near[:, :2], pct, axis=0) - pad
    hi = np.percentile(near[:, :2], 100 - pct, axis=0) + pad
    aabb = o3d.geometry.AxisAlignedBoundingBox((lo[0], lo[1], -0.15), (hi[0], hi[1], ceil))
    return pcd.crop(aabb), (lo.tolist(), hi.tolist()), aabb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-band", type=float, default=0.2)
    ap.add_argument("--crop-pct", type=float, default=5.0)
    ap.add_argument("--pad", type=float, default=0.4)
    ap.add_argument("--ceil", type=float, default=2.8)
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--density-quantile", type=float, default=0.04, help="drop Poisson verts below this density quantile")
    ap.add_argument("--target-faces", type=int, default=300000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.cloud)
    print(f"[lidar-col] loaded {len(pcd.points):,} pts", flush=True)
    pcd = gravity_align(pcd)
    pcd, box, aabb = crop_room(pcd, args.floor_band, args.crop_pct, args.pad, args.ceil)
    print(f"[lidar-col] cropped to room {box}: {len(pcd.points):,} pts", flush=True)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([box[0][0], box[0][1], args.ceil + 2.0]))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
    dens = np.asarray(dens)
    keep = dens >= np.quantile(dens, args.density_quantile)
    mesh.remove_vertices_by_mask(~keep)
    mesh = mesh.crop(aabb)                       # Poisson extrapolates; clip to the room box
    print(f"[lidar-col] poisson mesh: V={len(mesh.vertices):,} F={len(mesh.triangles):,}", flush=True)

    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl = np.asarray(tl); cnt = np.asarray(cnt)
    if len(cnt):
        big = np.where(cnt >= max(1, int(5e-4 * cnt.sum())))[0]
        mesh.remove_triangles_by_index(np.where(~np.isin(tl, big))[0].tolist())
        mesh.remove_unreferenced_vertices()
    if args.target_faces and len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    V = np.asarray(mesh.vertices)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "collider.obj"), mesh)
    meta = {"source": "lidar", "floor_z_m": 0.0, "n_vertices": len(V), "n_faces": len(mesh.triangles),
            "bbox_min": V.min(0).round(3).tolist(), "bbox_max": V.max(0).round(3).tolist(), "room_box": box}
    json.dump(meta, open(os.path.join(args.out, "collider_meta.json"), "w"), indent=2)
    print(f"[lidar-col] wrote collider.obj  V={len(V):,} F={len(mesh.triangles):,} "
          f"z[{V[:,2].min():.2f},{V[:,2].max():.2f}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
