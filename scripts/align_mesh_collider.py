"""Gravity-align an arbitrary reconstructed mesh into a Z-up Isaac collider (CPU/Open3D).

Takes any mesh (e.g. the 2DGS TSDF surface), finds the dominant plane (floor) -> +Z with the
scene above, sets floor to z=0, and writes collider.obj + collider_meta.json. Automatic; used
to turn the 2DGS mesh into a physics collider for the Isaac collision test.

    python align_mesh_collider.py --mesh fuse_post.ply --out <collider_dir> --source 2dgs
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import open3d as o3d


def align_gravity(pcd, mesh):
    work = o3d.geometry.PointCloud(pcd); best = None; allpts = np.asarray(pcd.points)
    for _ in range(6):
        if len(work.points) < 200:
            break
        model, inl = work.segment_plane(0.03, ransac_n=3, num_iterations=1200)
        if best is None or len(inl) > best[0]:
            a, b, c, d = model; nrm = np.linalg.norm([a, b, c]) + 1e-9
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
    fz = float(np.percentile(np.asarray(pcd.points)[:, 2], 2))
    mesh.translate((0, 0, -fz)); pcd.translate((0, 0, -fz))
    return mesh, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="mesh")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.vertices) == 0:
        print(f"empty mesh: {args.mesh}"); return 2
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    print(f"[align] {args.mesh}: V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
    mesh, floor_z = align_gravity(pcd, mesh)
    V = np.asarray(mesh.vertices)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "collider.obj"), mesh)
    meta = {"source": args.source, "floor_z_m": float(floor_z), "n_vertices": len(V),
            "n_faces": len(mesh.triangles), "bbox_min": V.min(0).round(3).tolist(),
            "bbox_max": V.max(0).round(3).tolist()}
    json.dump(meta, open(os.path.join(args.out, "collider_meta.json"), "w"), indent=2)
    print(f"[align] wrote collider.obj (floor_z={floor_z:.3f}, z [{V[:,2].min():.2f},{V[:,2].max():.2f}]) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
