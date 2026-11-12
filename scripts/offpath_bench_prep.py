#!/usr/bin/env python3
"""Datasets of the viewer-era off-path benchmark (sbatch/offpath_bench.sbatch; CPU venv, needs open3d).

The spatial hold-out of scripts/offpath_split.py (<offpath>/train_data, <offpath>/test) measures views ~1 m off the
training path; the usual on-path hold-out (every 8th frame, <onpath>/test) measures views on it. One splat should be
scored on both, so the on-path test frames that fall inside the off-path TRAINING split are held out here too:
  <out>/train        COLMAP text dataset = --train-ds (e.g. the EDGS init of the off-path training split, its points3D.txt)
                     minus those frames; images/ = links to the frames (+ their _depth.npy LiDAR targets)
  <out>/onpath_test  the held-out on-path frames (links; poses of --train-ds)
  <out>/eyeball      --orbit "user-like" elevated cameras (no ground truth; for looking at): on a ring around the room
                     centre, 1.2-1.8 m above the floor, pitched down 30-45 deg towards the centre; the ring radius per
                     azimuth is --ring-frac of the free distance to the LiDAR walls (clipped to [1, 3.5] m). Up comes from
                     the fitted floor plane (scripts/floor_plane.py: the CEAR metric frame is Z-DOWN).
  <out>/scene.json   floor plane (metric frame: h = n . x + d, n up), room centre, sim3, the eyeball cameras
Everything is derived from the data (no per-scene numbers).

    python scripts/offpath_bench_prep.py --train-ds $P/edgs/off/data --onpath-test $P/onpath/test \
        --cloud $P/offpath/depth/lidar_static.ply --sim3-json $P/sim3.json --out $P/offpath_bench
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402
from make_pseudo_views import rot2quat, write_images  # noqa: E402


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(src), dst)


def subset(src_ds, names, out, points=True):
    """COLMAP text dataset with only `names` (image lines copied verbatim), images linked (+ _depth.npy, _covis.png)."""
    s0 = os.path.join(out, "sparse", "0"); os.makedirs(s0, exist_ok=True); os.makedirs(os.path.join(out, "images"), exist_ok=True)
    src0 = os.path.join(src_ds, "sparse", "0")
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(src0, "cameras.txt")))
    lines = [ln for ln in open(os.path.join(src0, "images.txt")) if not ln.startswith("#")][::2]
    n = 0
    with open(os.path.join(s0, "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for ln in lines:
            p = ln.split()
            if len(p) >= 10 and p[9] in names:
                f.write(" ".join(p[:10]) + "\n\n"); n += 1
                stem = os.path.splitext(p[9])[0]
                for suf in (os.path.splitext(p[9])[1], "_depth.npy", "_covis.png"):
                    a = os.path.join(src_ds, "images", stem + suf)
                    if os.path.exists(a):
                        link(a, os.path.join(out, "images", stem + suf))
    pts = os.path.join(s0, "points3D.txt")
    if points and os.path.exists(os.path.join(src0, "points3D.txt")):
        link(os.path.join(src0, "points3D.txt"), pts)
    elif not os.path.exists(pts):
        open(pts, "w").close()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-ds", required=True); ap.add_argument("--onpath-test", required=True)
    ap.add_argument("--cloud", required=True, help="static LiDAR cloud, metric frame"); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orbit", type=int, default=8); ap.add_argument("--ring-frac", type=float, default=0.7)
    ap.add_argument("--heights", type=float, nargs="+", default=[1.2, 1.5, 1.8]); ap.add_argument("--pitches", type=float, nargs="+", default=[30, 37.5, 45])
    a = ap.parse_args()
    from plyfile import PlyData
    from floor_plane import find_floor_plane
    cams, imgs = read_text_model(os.path.join(a.train_ds, "sparse", "0"))
    all_names = {i[0] for i in imgs}
    on = {i[0] for i in read_text_model(os.path.join(a.onpath_test, "sparse", "0"))[1]} & all_names
    ntr = subset(a.train_ds, all_names - on, os.path.join(a.out, "train"))
    nte = subset(a.train_ds, on, os.path.join(a.out, "onpath_test"), points=False)
    print(f"[obprep] {len(all_names)} frames of {a.train_ds}: {ntr} train, {nte} on-path held out -> {a.out}", flush=True)

    j = json.load(open(a.sim3_json)); s, R, t = float(j["scale"]), np.asarray(j["R"]), np.asarray(j["t"])
    v = PlyData.read(a.cloud)["vertex"]; L = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    Cc = np.array([-quat_to_R(q).T @ tt for _, _, q, tt in imgs]); Cm = s * Cc @ R.T + t           # camera centres, metric
    fl = find_floor_plane(L, Cm)
    if fl is None:
        raise SystemExit("[obprep] no floor plane found")
    n, d, _ = fl
    h = L @ n + d
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.3 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    body = L[(h > 0.3) & (h < 2.0)]
    c0 = np.median(body, 0); c0 = c0 - (c0 @ n + d) * n                                              # room centre on the floor
    rel = body - c0; x, y = rel @ e1, rel @ e2; rad, az = np.hypot(x, y), np.arctan2(y, x)
    views, rows = [], []
    cid = sorted(cams)[0]
    for k in range(a.orbit):
        th = 2 * np.pi * k / a.orbit
        m = (np.abs(np.angle(np.exp(1j * (az - th)))) < np.radians(12)) & (rad > 0.5)
        wall = float(np.percentile(rad[m], 10)) if m.sum() > 50 else 2.0
        r = float(np.clip(a.ring_frac * wall, 1.0, 3.5)); hk = a.heights[k % len(a.heights)]; pk = np.radians(a.pitches[k % len(a.pitches)])
        C = c0 + r * (np.cos(th) * e1 + np.sin(th) * e2) + hk * n
        hz = -(np.cos(th) * e1 + np.sin(th) * e2)                                                    # horizontal, towards the centre
        f = np.cos(pk) * hz - np.sin(pk) * n; right = np.cross(f, n); right /= np.linalg.norm(right); down = np.cross(f, right)
        Rm = np.stack([right, down, f], 1)                                                           # c2w (OpenCV), metric frame
        Rc, Ccol = R.T @ Rm, R.T @ (C - t) / s                                                       # dataset (COLMAP) frame
        name = f"eyeball_{k:02d}.png"
        rows.append((k + 1, rot2quat(Rc.T), -Rc.T @ Ccol, cid, name))
        views.append({"name": name, "azimuth_deg": round(float(np.degrees(th)), 1), "radius_m": round(r, 2), "height_m": hk,
                      "pitch_deg": float(np.degrees(pk)), "wall_m": round(wall, 2)})
    eb = os.path.join(a.out, "eyeball"); s0 = os.path.join(eb, "sparse", "0"); os.makedirs(s0, exist_ok=True); os.makedirs(os.path.join(eb, "images"), exist_ok=True)
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(a.train_ds, "sparse", "0", "cameras.txt")))
    write_images(os.path.join(s0, "images.txt"), rows); open(os.path.join(s0, "points3D.txt"), "w").close()
    from PIL import Image
    W, H, _ = cams[cid]
    for r_ in rows:
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(os.path.join(eb, "images", r_[4]))
    json.dump({"floor": {"n": n.tolist(), "d": float(d)}, "room_centre_m": c0.tolist(), "sim3": {"scale": s, "R": R.tolist(), "t": t.tolist()},
               "n_train": ntr, "n_onpath_test": nte, "eyeball": views, "args": vars(a)}, open(os.path.join(a.out, "scene.json"), "w"), indent=1)
    print(f"[obprep] floor n={np.round(n, 3).tolist()} d={d:.3f}; room centre {np.round(c0, 2).tolist()}; {len(rows)} eyeball views "
          f"(radius {[v_['radius_m'] for v_ in views]} m) -> {eb}", flush=True)
    print("OBPREP_DONE")


if __name__ == "__main__":
    main()
