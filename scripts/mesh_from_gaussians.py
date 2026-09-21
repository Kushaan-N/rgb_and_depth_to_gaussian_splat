"""Option B, stage 2 — reconstruct a surface mesh directly from the gaussian centres (CPU).

Reads the 3DGS .ply (gaussian centres), drops floaters (statistical outlier removal), estimates
normals, and runs Poisson surface reconstruction -> a watertight collider. Geometry straight from
the splat's own representation (no rendering, no hand placement). Gravity-aligns like Option A.

    python mesh_from_gaussians.py --ply gaussians.ply --out <collider_dir> [--poisson-depth 9]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import open3d as o3d


def align_gravity(pcd, mesh):
    """Rotate so the dominant plane (floor) normal -> +Z, scene above, floor at z=0. Returns floor_z."""
    work = o3d.geometry.PointCloud(pcd)
    best = None
    allpts = np.asarray(pcd.points)
    for _ in range(6):
        if len(work.points) < 200:
            break
        model, inl = work.segment_plane(0.03, ransac_n=3, num_iterations=1200)
        if best is None or len(inl) > best[0]:
            a, b, c, d = model
            nrm = np.linalg.norm([a, b, c]) + 1e-9
            best = (len(inl), np.array([a, b, c]) / nrm, d / nrm)
        work = work.select_by_index(inl, invert=True)
    if best is None:
        return mesh, 0.0
    n = best[1]
    if np.dot(allpts.mean(0), n) + best[2] < 0:
        n = -n
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); cth = float(np.dot(n, z))
    if s < 1e-6:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s * s))
    mesh.rotate(R, center=(0, 0, 0)); pcd.rotate(R, center=(0, 0, 0))
    floor_z = float(np.percentile(np.asarray(pcd.points)[:, 2], 2))
    mesh.translate((0, 0, -floor_z)); pcd.translate((0, 0, -floor_z))
    return mesh, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--density-quantile", type=float, default=0.05, help="trim lowest-density Poisson verts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"[mesh_gauss] loaded {len(pcd.points)} gaussian centres", flush=True)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # drop floaters
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(40)
    print(f"[mesh_gauss] {len(pcd.points)} pts after outlier removal; Poisson depth={args.poisson_depth}", flush=True)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.density_quantile))  # trim Poisson overshoot
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center()))
    mesh.compute_vertex_normals()
    print(f"[mesh_gauss] mesh V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
    mesh, floor_z = align_gravity(pcd, mesh)
    V = np.asarray(mesh.vertices)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "collider.obj"), mesh)
    meta = {"source": "gaussian_poisson_surface", "poisson_depth": args.poisson_depth,
            "floor_z_m": float(floor_z), "n_vertices": len(V), "n_faces": len(mesh.triangles),
            "bbox_min": V.min(0).round(3).tolist(), "bbox_max": V.max(0).round(3).tolist()}
    json.dump(meta, open(os.path.join(args.out, "collider_meta.json"), "w"), indent=2)
    print(f"[mesh_gauss] wrote collider.obj (floor_z={floor_z:.3f}, "
          f"z [{V[:,2].min():.2f},{V[:,2].max():.2f}]) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
