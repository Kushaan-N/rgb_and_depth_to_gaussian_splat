"""Gravity-align + clean an arbitrary reconstructed mesh into a Z-up Isaac collider (CPU/Open3D).

Takes any mesh (e.g. the 2DGS unbounded TSDF surface), finds the dominant plane (floor) -> +Z
with the scene above, sets floor to z=0, then makes it usable as a physics collider:

  1. gravity-align (RANSAC floor plane -> +Z, floor -> z=0)
  2. crop to the room box, derived automatically from the near-floor footprint (drops the
     far-field background the 2DGS *unbounded* contraction reconstructs beyond the room)
  3. drop small disconnected components (floater noise), keeping the floor, walls AND the
     scattered objects on the floor (each may be its own component)
  4. quadric-decimate to a target face count so PhysX can load/simulate it

All thresholds are data-driven percentiles, not per-scene constants, so this stays generalizable.
Writes collider.obj + collider_meta.json.

    python align_mesh_collider.py --mesh fuse_unbounded_post.ply --out <collider_dir> --source 2dgs
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


def crop_to_room(mesh, floor_band, crop_pct, pad, ceil_h):
    """Crop to a box derived from the near-floor footprint (the real room extent).

    The unbounded TSDF reconstructs a contracted 'background' floor far beyond the room; the
    genuine floor is the dense central plateau. We take robust percentiles of the near-floor xy
    to bound the room, pad a little to keep the walls, and cap height at the ceiling band.
    """
    V = np.asarray(mesh.vertices)
    near = V[(V[:, 2] > -0.1) & (V[:, 2] < floor_band)]
    if len(near) < 500:
        near = V
    lo = np.percentile(near[:, :2], crop_pct, axis=0) - pad
    hi = np.percentile(near[:, :2], 100 - crop_pct, axis=0) + pad
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(lo[0], lo[1], -0.1), max_bound=(hi[0], hi[1], ceil_h))
    out = mesh.crop(aabb)
    return out, (lo.tolist(), hi.tolist())


def drop_small_components(mesh, min_frac):
    """Remove connected components smaller than min_frac of the total tri count (floater noise),
    keeping floor + walls + each scattered object (objects are often their own component)."""
    tl, cnt, _ = mesh.cluster_connected_triangles()
    tl = np.asarray(tl); cnt = np.asarray(cnt)
    if len(cnt) == 0:
        return mesh, 0, 0
    keep = np.where(cnt >= max(1, int(min_frac * cnt.sum())))[0]
    drop_tris = np.where(~np.isin(tl, keep))[0]
    mesh.remove_triangles_by_index(drop_tris.tolist())
    mesh.remove_unreferenced_vertices()
    return mesh, len(keep), len(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="mesh")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip crop/component/decimate cleanup (for already-small meshes)")
    ap.add_argument("--floor-band", type=float, default=0.15, help="z window above floor used to find footprint")
    ap.add_argument("--crop-pct", type=float, default=6.0, help="footprint percentile clip per side")
    ap.add_argument("--pad", type=float, default=0.3, help="metres of padding added around the room box")
    ap.add_argument("--ceil", type=float, default=2.6, help="max z (ceiling) kept, metres above floor")
    ap.add_argument("--min-comp-frac", type=float, default=5e-4, help="drop components below this frac of tris")
    ap.add_argument("--target-faces", type=int, default=200000, help="quadric-decimation target (0=off)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.vertices) == 0:
        print(f"empty mesh: {args.mesh}"); return 2
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    print(f"[align] {args.mesh}: V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
    mesh, floor_z = align_gravity(pcd, mesh)

    room_box = None
    if not args.no_clean:
        mesh, room_box = crop_to_room(mesh, args.floor_band, args.crop_pct, args.pad, args.ceil)
        print(f"[align] cropped to room {room_box}: V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
        mesh, kept, total = drop_small_components(mesh, args.min_comp_frac)
        print(f"[align] kept {kept}/{total} components: V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
        if args.target_faces and len(mesh.triangles) > args.target_faces:
            mesh = mesh.simplify_quadric_decimation(args.target_faces)
            print(f"[align] decimated -> V={len(mesh.vertices)} F={len(mesh.triangles)}", flush=True)
        mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "collider.obj"), mesh)
    meta = {"source": args.source, "floor_z_m": float(floor_z), "n_vertices": len(V),
            "n_faces": len(mesh.triangles), "bbox_min": V.min(0).round(3).tolist(),
            "bbox_max": V.max(0).round(3).tolist(), "room_box": room_box}
    json.dump(meta, open(os.path.join(args.out, "collider_meta.json"), "w"), indent=2)
    print(f"[align] wrote collider.obj (floor_z={floor_z:.3f}, z [{V[:,2].min():.2f},{V[:,2].max():.2f}], "
          f"F={len(mesh.triangles)}) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
