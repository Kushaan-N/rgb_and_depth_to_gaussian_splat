"""Viewer-friendly copy of a 3DGS PLY: recentred and gravity-aligned for SuperSplat, metric scale kept (any dataset).

Exports land in whatever frame the data came in (a GT/total-station world with its origin metres away, Z-up LiDAR
frames, COLMAP gauges...), so the SuperSplat editor opens on an arbitrary, often tilted view. This writes a rigid copy
x' = R (x - c):
  c   robust centre: median of the training camera centres (--center cameras, default; needs --dataset), the
      LiDAR cloud (--center cloud --cloud C: 2-98 % box middle) or the opaque splats (--center splat)
  R   rotates the scene's up direction to the PLY axis SuperSplat shows as up. The editor imports a PLY with a 180 deg
      rotation about Z (its transform panel shows rotation 0 0 180; viewer_shots.py reads the same matrix), so PLY -Y
      is screen-up (--target-up -y). Up comes from --up (+x|-x|+y|-y|+z|-z) or `auto`: the mean of the cameras'
      up vectors (-y of every OpenCV camera, i.e. cameras held roughly level - true for robots / handheld rigs), and
      the heading is chosen so the mean viewing direction maps to PLY +Z (the editor's default view then looks the
      way the rig mostly looked).
Positions, gaussian rotations, view-dependent colour (SH bands 1-3, rotate_sh from sim3_align_splat.py) and normals are
transformed; scales/opacity/DC colour are invariant. Writes <out>.json with the 4x4 (new <- old) so poses / clouds
can follow (--dataset-out also writes the COLMAP text model in the new frame, for viewer_shots.py checks).

    python scripts/view_frame.py --in-ply exports/x.ply --out-ply exports/x_view.ply --dataset <P>/onpath/train \
        [--up auto] [--center cameras] [--dataset-out <dir>]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim3_align_splat import rotate_sh  # noqa: E402

AX = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def qR(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_cams(ds):
    """[(line fields, R_cw, t)] of a COLMAP text images.txt."""
    out = []
    for ln in open(os.path.join(ds, "sparse", "0", "images.txt")):
        p = ln.split()
        if len(p) >= 10 and not ln.startswith("#") and p[9].lower().endswith((".png", ".jpg", ".jpeg")):
            out.append((p, qR([float(x) for x in p[1:5]]), np.array([float(x) for x in p[5:8]])))
    return out


def frame_from(up, fwd, target_up, target_fwd):
    """Rotation mapping up -> target_up and the horizontal part of fwd -> target_fwd."""
    up = up / np.linalg.norm(up); f = fwd - up * (fwd @ up)
    if np.linalg.norm(f) < 1e-6:
        f = np.cross(up, [1, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1, 0])
    f /= np.linalg.norm(f); A = np.stack([up, f, np.cross(up, f)], 1)
    tu, tf = np.asarray(target_up, float), np.asarray(target_fwd, float); B = np.stack([tu, tf, np.cross(tu, tf)], 1)
    return B @ A.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-ply", required=True); ap.add_argument("--out-ply", required=True)
    ap.add_argument("--dataset", default=None, help="COLMAP text dataset in the PLY's frame (camera centres / up)")
    ap.add_argument("--cloud", default=None, help="--center cloud: a point cloud in the PLY's frame")
    ap.add_argument("--up", default="auto", help="auto | +x|-x|+y|-y|+z|-z (the scene's up in the input frame)")
    ap.add_argument("--center", default="cameras", choices=("cameras", "cloud", "splat"))
    ap.add_argument("--target-up", default="-y", choices=list(AX), help="PLY axis the viewer shows as up")
    ap.add_argument("--target-fwd", default="+z", choices=list(AX), help="PLY axis the mean viewing direction maps to")
    ap.add_argument("--dataset-out", default=None, help="also write the dataset's model in the new frame here")
    a = ap.parse_args()
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    ply = PlyData.read(a.in_ply); v = ply["vertex"].data
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    cams = read_cams(a.dataset) if a.dataset else []
    C = np.array([-R.T @ t for _, R, t in cams]) if cams else None
    # up and mean viewing direction
    if a.up == "auto":
        if not cams:
            raise SystemExit("[view] --up auto needs --dataset")
        ups = np.array([-R.T[:, 1] for _, R, _ in cams]); up = ups.mean(0); up /= np.linalg.norm(up)
        spread = np.degrees(np.arccos(np.clip(ups @ up, -1, 1)))
        near = min(AX, key=lambda k: np.degrees(np.arccos(np.clip(np.dot(AX[k], up), -1, 1))))
        print(f"[view] up from {len(cams)} cameras: {np.round(up, 3).tolist()} (camera tilt median {np.median(spread):.1f} deg, "
              f"p90 {np.percentile(spread, 90):.1f}; nearest input axis {near}, "
              f"{np.degrees(np.arccos(np.clip(np.dot(AX[near], up), -1, 1))):.1f} deg off)", flush=True)
    else:
        up = np.asarray(AX[a.up], float)
    fwd = np.array([R.T[:, 2] for _, R, _ in cams]).mean(0) if cams else np.cross(up, [1, 0, 0])
    R = frame_from(up, fwd, AX[a.target_up], AX[a.target_fwd])
    # centre
    if a.center == "cameras":
        if C is None:
            raise SystemExit("[view] --center cameras needs --dataset")
        c = np.median(C, 0)
    elif a.center == "cloud":
        import open3d as o3d
        P = np.asarray(o3d.io.read_point_cloud(a.cloud).points); lo, hi = np.percentile(P, [2, 98], axis=0); c = (lo + hi) / 2
    else:
        op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64))); lo, hi = np.percentile(X[op > 0.5], [2, 98], axis=0); c = (lo + hi) / 2
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ c
    Y = X @ R.T + T[:3, 3]
    v["x"], v["y"], v["z"] = Y[:, 0], Y[:, 1], Y[:, 2]
    if all(f"rot_{i}" in v.dtype.names for i in range(4)):
        q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1).astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        qn = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()
        v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = qn[:, 3], qn[:, 0], qn[:, 1], qn[:, 2]
    if all(k in v.dtype.names for k in ("nx", "ny", "nz")):
        n = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64) @ R.T
        v["nx"], v["ny"], v["nz"] = n[:, 0], n[:, 1], n[:, 2]
    rest = [f"f_rest_{i}" for i in range(45)]
    if all(k in v.dtype.names for k in rest):
        F = rotate_sh(np.stack([v[k] for k in rest], 1).astype(np.float64).reshape(len(v), 3, 15), R)
        for i, k in enumerate(rest):
            v[k] = F.reshape(len(v), 45)[:, i].astype(v[k].dtype)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False, comments=list(ply.comments)).write(a.out_ply)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    lo, hi = np.percentile(Y[op > 0.5], [1, 99], axis=0)
    info = {"in": a.in_ply, "out": a.out_ply, "T_new_old": T.tolist(), "center_old": c.tolist(), "up_old": up.tolist(),
            "target_up": a.target_up, "target_fwd": a.target_fwd, "opaque_1_99_box_new": [lo.tolist(), hi.tolist()]}
    json.dump(info, open(os.path.splitext(a.out_ply)[0] + ".json", "w"), indent=1)
    print(f"[view] {len(v):,} gaussians -> {a.out_ply}; centre {np.round(c, 2).tolist()} -> origin; opaque 1-99 % box "
          f"{np.round(lo, 1).tolist()} .. {np.round(hi, 1).tolist()}", flush=True)
    if a.dataset_out and cams:                        # same cameras in the new frame (R_cw' = R_cw R^T, C' = T C)
        s0 = os.path.join(a.dataset_out, "sparse", "0"); os.makedirs(s0, exist_ok=True)
        for f in ("cameras.txt", "points3D.txt"):
            src = os.path.join(a.dataset, "sparse", "0", f)
            if f == "cameras.txt":
                shutil.copy(src, s0)
            else:
                open(os.path.join(s0, f), "w").close()
        with open(os.path.join(s0, "images.txt"), "w") as f:
            for p, Rcw, t in cams:
                Rn = Rcw @ R.T; Cn = T[:3, :3] @ (-Rcw.T @ t) + T[:3, 3]; tn = -Rn @ Cn
                w, x, y, z = (lambda q: (q[3], q[0], q[1], q[2]))(Rotation.from_matrix(Rn).as_quat())
                f.write(" ".join([p[0], *map(str, (w, x, y, z, *tn)), p[8], p[9]]) + "\n\n")
        im = os.path.join(a.dataset_out, "images")
        if not os.path.lexists(im):
            os.symlink(os.path.realpath(os.path.join(a.dataset, "images")), im)
        print(f"[view] cameras in the new frame -> {a.dataset_out}")
    print("VIEW_FRAME_DONE")


if __name__ == "__main__":
    main()
