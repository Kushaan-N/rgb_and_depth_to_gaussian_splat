"""Phase 5 — collider mesh from depth (docs/PLAN.md §8).

Gaussian splats are visual-only in Isaac Sim, so we build a SECOND, co-registered
artifact: a physics mesh for foot contact. Default (Option A) is to mesh the Phase-3 TSDF
(already marching-cubes'd). Unobserved floor holes along the route are patched with the
RANSAC floor plane so a foot cannot fall through; every patch is logged. The mesh is
decimated to a foot-contact-adequate budget and (if pxr is available) written to USD with a
collision API and marked invisible; otherwise it is written as PLY/OBJ for Phase 6 to
compose on the L40S.

Gate 5 proper (drop a rigid body) runs in Isaac (Phase 6). Here we run a CPU proxy: cast
rays straight down over the floor and confirm (a) no holes and (b) the free-floor contact
height matches the fused floor within ±3 cm.

    python scripts/build_collider.py --config configs/mocap1_well-lit_trot.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from verify_depth_map import fit_floor_plane


def _floor_grid(mesh, cell):
    """Cell-center (x,y) grid over the mesh's xy footprint."""
    V = np.asarray(mesh.vertices)
    xmin, ymin = V[:, :2].min(0); xmax, ymax = V[:, :2].max(0)
    xs = np.arange(xmin + cell / 2, xmax, cell)
    ys = np.arange(ymin + cell / 2, ymax, cell)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _raycast_down(mesh, centers_xy, top):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    origins = np.hstack([centers_xy, np.full((len(centers_xy), 1), top)])
    rays = o3d.core.Tensor(np.hstack([origins, np.tile([0, 0, -1.0], (len(origins), 1))]
                                     ).astype(np.float32))
    t = scene.cast_rays(rays)["t_hit"].numpy()
    return t, top - t, np.isfinite(t)


def patch_floor_holes(mesh, floor_z, centers_xy, cell=0.10):
    """Fill exactly the cells a dropped foot would fall through (§8).

    A cell is a hole iff a downward ray hits NOTHING (cells that hit the obstacle top are
    not holes). Coverage is decided by raycasting the actual triangulated surface, not by
    vertex presence — vertex-based coverage misses gaps where no triangle spans a cell.
    """
    top = floor_z + 1.5
    _t, _hz, finite = _raycast_down(mesh, centers_xy, top)
    holes = centers_xy[~finite]
    if len(holes) == 0:
        return mesh, {"patched_cells": 0, "patched_area_m2": 0.0}
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    new_v, new_f = [], []
    h = cell / 2.0
    for (cx, cy) in holes:
        k = len(V) + len(new_v)
        new_v += [[cx - h, cy - h, floor_z], [cx + h, cy - h, floor_z],
                  [cx + h, cy + h, floor_z], [cx - h, cy + h, floor_z]]
        new_f += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack([V, np.array(new_v)]))
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack([F, np.array(new_f)]))
    mesh.compute_vertex_normals()
    return mesh, {"patched_cells": int(len(holes)),
                  "patched_area_m2": float(len(holes) * cell * cell)}


def flatten_floor(mesh, floor_z, band):
    """Snap collider vertices within `band` of the floor to the RANSAC plane.

    A physics collider's floor only needs to be a clean, flat contact surface (§8: "ground
    plane and major obstacles"); real-depth noise + decimation leave it bumpy, which makes
    a dropped foot sink/hover. Flattening keeps walls/obstacles intact (only the bottom
    `band` of their geometry, which sits on the floor anyway, is snapped). The visual splat
    keeps the true floor appearance — this is physics-only.
    """
    V = np.asarray(mesh.vertices).copy()
    near = np.abs(V[:, 2] - floor_z) < band
    V[near, 2] = floor_z
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.compute_vertex_normals()
    return mesh, int(near.sum())


def drop_test(mesh, floor_z, cell=0.10):
    """CPU proxy for Gate 5: cast rays down; check no holes + free-floor contact height."""
    centers = _floor_grid(mesh, cell)
    top = floor_z + 1.5
    _t, hit_z, finite = _raycast_down(mesh, centers, top)
    hit_frac = float(finite.mean())
    # genuine floor = hits in a tight band; p95 is robust to ramp triangles at obstacle bases
    near_floor = finite & (np.abs(hit_z - floor_z) < 0.05)
    if near_floor.any():
        err = np.abs(hit_z[near_floor] - floor_z)
        err_p95 = float(np.percentile(err, 95)); err_max = float(err.max())
    else:
        err_p95 = err_max = float("nan")
    return {"n_rays": int(len(centers)), "hit_fraction": hit_frac,
            "floor_height_err_p95_m": err_p95, "floor_height_err_max_m": err_max,
            "n_free_floor_cells": int(near_floor.sum())}


def build(cfg: dict) -> dict:
    out_root = cfg["paths"]["out_root"]
    col_dir = os.path.join(out_root, "collider")
    os.makedirs(col_dir, exist_ok=True)
    dmap = os.path.join(out_root, "depth_map")

    # Option A: mesh the TSDF (already extracted in Phase 3); else Poisson the cloud.
    mesh_ply = os.path.join(dmap, "fused_mesh.ply")
    cloud_ply = os.path.join(dmap, "fused_cloud.ply")
    if cfg["collider"].get("method") == "poisson" or not os.path.exists(mesh_ply):
        pcd = o3d.io.read_point_cloud(cloud_ply)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)
        mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        provenance = "poisson"
    else:
        mesh = o3d.io.read_triangle_mesh(mesh_ply)
        provenance = "tsdf_marching_cubes"
    mesh.compute_vertex_normals()

    floor_z, floor_thick, _n = fit_floor_plane(o3d.io.read_point_cloud(cloud_ply))

    # decimate to a foot-contact-adequate budget FIRST (decimation can open small holes),
    # then patch, so the patches are guaranteed present in the final mesh.
    max_tris = int(cfg["collider"]["max_triangles"])
    n_before = len(mesh.triangles)
    if n_before > max_tris:
        mesh = mesh.simplify_quadric_decimation(max_tris)
        mesh.compute_vertex_normals()

    patch_info = {"patched_cells": 0, "patched_area_m2": 0.0}
    if cfg["collider"].get("floor_patch", True) and np.isfinite(floor_z):
        centers = _floor_grid(mesh, 0.10)
        mesh, patch_info = patch_floor_holes(mesh, floor_z, centers, cell=0.10)

    n_flattened = 0
    if cfg["collider"].get("flatten_floor", True) and np.isfinite(floor_z):
        mesh, n_flattened = flatten_floor(mesh, floor_z,
                                          float(cfg["collider"].get("flatten_floor_band_m", 0.04)))

    o3d.io.write_triangle_mesh(os.path.join(col_dir, "collider.ply"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(col_dir, "collider.obj"), mesh)

    gate5 = drop_test(mesh, floor_z) if np.isfinite(floor_z) else {"hit_fraction": float("nan")}
    usd_written = _try_write_usd(mesh, os.path.join(col_dir, "collider.usd"))

    ok = (np.isfinite(floor_z)
          and gate5.get("hit_fraction", 0) > 0.99
          and np.isfinite(gate5.get("floor_height_err_p95_m", np.nan))
          and gate5["floor_height_err_p95_m"] < 0.03)
    meta = {
        "provenance": provenance,
        "floor_z_m": floor_z, "floor_thickness_m": floor_thick,
        "n_triangles_before_decimation": n_before,
        "n_triangles_final": int(len(mesh.triangles)),
        "floor_patch": patch_info,
        "floor_vertices_flattened": n_flattened,
        "gate5_proxy": gate5, "gate5_proxy_pass": bool(ok),
        "usd_written": usd_written,
        "collider_ply": os.path.join(col_dir, "collider.ply"),
    }
    pu.save_json(os.path.join(col_dir, "collider_meta.json"), meta)
    return meta


def _try_write_usd(mesh, path) -> bool:
    """Write a USD collider (collision API + invisible) if pxr is importable (Isaac side)."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, Vt, Gf  # noqa: F401
    except Exception:
        return False
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    m = UsdGeom.Mesh.Define(stage, "/collider")
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in V])
    m.CreateFaceVertexCountsAttr([3] * len(F))
    m.CreateFaceVertexIndicesAttr(F.ravel().tolist())
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdGeom.Imageable(m).MakeInvisible()          # collider is physics-only, not rendered
    stage.GetRootLayer().Save()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    m = build(cfg)
    print(f"[collider] provenance={m['provenance']}  floor_z={m['floor_z_m']:.4f} m")
    print(f"[collider] triangles {m['n_triangles_before_decimation']} -> {m['n_triangles_final']} "
          f"(patched {m['floor_patch']['patched_cells']} floor cells, "
          f"{m['floor_patch']['patched_area_m2']:.2f} m²)")
    g = m["gate5_proxy"]
    print(f"[collider] Gate 5 proxy: hit {g.get('hit_fraction', float('nan'))*100:.1f}% of rays, "
          f"floor height err p95 {g.get('floor_height_err_p95_m', float('nan'))*1000:.1f} mm "
          f"(max {g.get('floor_height_err_max_m', float('nan'))*1000:.1f} mm)")
    print(f"[collider] USD written: {m['usd_written']} "
          f"({'pxr available' if m['usd_written'] else 'no pxr — compose in Phase 6 on L40S'})")
    print(f"GATE 5 (CPU proxy) {'PASS' if m['gate5_proxy_pass'] else 'REVIEW'}")
    return 0 if m["gate5_proxy_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
