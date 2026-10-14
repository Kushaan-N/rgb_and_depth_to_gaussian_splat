#!/usr/bin/env python3
"""Pseudo-training views for self-distillation: virtual cameras NEAR (not on) the training path, rendered by a
teacher splat, as a COLMAP text dataset with per-pixel loss masks (CPU; the render itself is eval_views.py).

  plan    interpolate between consecutive TRAINING frames (name order; frame k is held out iff k % --test-split-interval
          == 0, 3DGRUT's rule): --per-pair cameras per pair at t ~ U(--tmin, --tmax) (rotation slerp, centre lerp), each
          moved by a random camera-frame offset (right/down uniform in +-[0.5, 1] x --offset-m metres, yaw/pitch uniform
          in +-[0.5, 1] x --rot-deg; 0 = interpolated only). Held-out frames stay held out: a camera is dropped when its
          fractional frame index is within --min-frame-gap of a held-out frame or its centre is within --min-dist m of a
          held-out camera; pairs across pose jumps (> --max-step x the median step) are skipped. Metres -> dataset units
          via --sim3-json (x_metric = s R x + t; without it the dataset is taken as metric). Writes <out>/sparse/0/
          {cameras,images,points3D}.txt (points linked from the source), black placeholder images/pseudo_NNNNN.png (so
          3DGRUT's loader can render the poses) and <out>/pseudo.json (per-view source pair, t, offsets).
  finish  teacher renders (eval_views.py --save-dir / --save-opacity-dir / --save-depth-dir) -> images/<stem>.png and
          images/<stem>_mask.png (255 = supervise): accumulated opacity >= --min-opacity, rendered depth present, and not
          within --edge-dilate px of a depth edge (5x5 max/min of the rendered depth ratio > 1 + --edge-rel), and (with
          --lidar-points) not > --lidar-tol nearer than every LiDAR return of its block (teacher floaters). Views whose
          valid share is below --min-valid are dropped from images.txt. Stats -> pseudo.json["finish"].

    python scripts/make_pseudo_views.py plan --dataset $P/edgs/on/data --out PDS --test-split-interval 8 \
        --sim3-json $P/sim3.json [--per-pair 1 --offset-m 0.15 --rot-deg 8]
    python scripts/eval_views.py --checkpoint CK --dataset PDS --no-metrics --save-dir R --save-opacity-dir O --save-depth-dir D
    python scripts/make_pseudo_views.py finish --out PDS --renders R --opacity O --depth D
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses_gsplat import read_model, intrinsics  # noqa: E402


def rot2quat(R):
    """3x3 -> (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0); return q / np.linalg.norm(q)
    a = np.arccos(d)
    return (np.sin((1 - t) * a) * q0 + np.sin(t * a) * q1) / np.sin(a)


def axis_rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]


def plan(a):
    cams, imgs = read_model(os.path.join(a.dataset, "sparse", "0"))
    unit = 1.0                                                   # metres per dataset unit
    if a.sim3_json:
        unit = float(json.load(open(a.sim3_json))["scale"])
    n, iv = len(imgs), a.test_split_interval
    test = [k for k in range(n) if iv > 0 and k % iv == 0]
    C = np.array([c2w[:3, 3] for _, _, _, c2w in imgs])
    Ct = C[test] if test else np.zeros((0, 3))
    step = np.linalg.norm(np.diff(C, axis=0), axis=1) * unit
    med = float(np.median(step)) if len(step) else 0.0
    rng = np.random.default_rng(a.seed)
    rows, meta, drop = [], [], {"near_heldout_index": 0, "near_heldout_dist": 0, "pose_jump": 0}
    for i in range(n - 1):
        if i in test or (i + 1) in test:
            continue
        if med > 0 and step[i] > a.max_step * med:
            drop["pose_jump"] += 1; continue
        (_, _, cid, A), (_, _, _, B) = imgs[i], imgs[i + 1]
        qa, qb = rot2quat(A[:3, :3]), rot2quat(B[:3, :3])
        for _ in range(a.per_pair):
            t = float(rng.uniform(a.tmin, a.tmax))
            if test and min(abs(i + t - j) for j in test) < a.min_frame_gap:
                drop["near_heldout_index"] += 1; continue
            R = quat2rot(slerp(qa, qb, t)); c = (1 - t) * A[:3, 3] + t * B[:3, 3]
            mag = lambda m: float(rng.uniform(0.5, 1.0) * m * rng.choice([-1, 1])) if m > 0 else 0.0
            dx, dy, yaw, pitch = mag(a.offset_m), mag(a.offset_m), mag(a.rot_deg), mag(a.rot_deg)
            c = c + R @ np.array([dx, dy, 0.0]) / unit           # camera right / down (OpenCV axes), metres
            R = R @ axis_rot("y", yaw) @ axis_rot("x", pitch)    # yaw about camera down, pitch about camera right
            if len(Ct) and float(np.linalg.norm(Ct - c, axis=1).min()) * unit < a.min_dist:
                drop["near_heldout_dist"] += 1; continue
            k = len(rows); name = f"pseudo_{k:05d}.png"
            Rw2c = R.T; tw2c = -R.T @ c
            rows.append((k + 1, rot2quat(Rw2c), tw2c, cid, name))
            meta.append({"name": name, "pair": [imgs[i][0], imgs[i + 1][0]], "t": round(t, 4), "right_m": round(dx, 4),
                         "down_m": round(dy, 4), "yaw_deg": round(yaw, 3), "pitch_deg": round(pitch, 3),
                         "dist_heldout_m": round(float(np.linalg.norm(Ct - c, axis=1).min()) * unit, 4) if len(Ct) else None})
    s0 = os.path.join(a.out, "sparse", "0"); im = os.path.join(a.out, "images")
    os.makedirs(s0, exist_ok=True); os.makedirs(im, exist_ok=True)
    src = os.path.join(a.dataset, "sparse", "0")
    with open(os.path.join(s0, "cameras.txt"), "w") as f:
        f.writelines(ln for ln in open(os.path.join(src, "cameras.txt")))
    write_images(os.path.join(s0, "images.txt"), rows)
    pts = os.path.join(s0, "points3D.txt")
    if not os.path.exists(pts):
        os.symlink(os.path.abspath(os.path.join(src, "points3D.txt")), pts)
    from PIL import Image
    for _, _, _, cid, name in rows:
        W, H = cams[cid][1], cams[cid][2]
        p = os.path.join(im, name)
        if not os.path.exists(p):
            Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(p)
    info = {"args": {k: v for k, v in vars(a).items() if k != "func"}, "n_source": n, "n_heldout": len(test), "median_step_m": round(med, 4), "metres_per_unit": unit,
            "n_views": len(rows), "dropped": drop, "views": meta}
    json.dump(info, open(os.path.join(a.out, "pseudo.json"), "w"), indent=1)
    dh = [m["dist_heldout_m"] for m in meta if m["dist_heldout_m"] is not None]
    print(f"[pseudo] {len(rows)} views from {n} frames ({len(test)} held out; median step {med * 100:.1f} cm), dropped {drop}; "
          f"distance to nearest held-out camera pct 0/50/100 {np.round(np.percentile(dh, [0, 50, 100]), 3).tolist() if dh else '-'} m -> {a.out}")


def write_images(path, rows):
    with open(path, "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# (no 2D points)\n")
        for iid, q, t, cid, name in rows:
            f.write(f"{iid} {' '.join(f'{x:.12g}' for x in q)} {' '.join(f'{x:.12g}' for x in t)} {cid} {name}\n\n")


def finish(a):
    import cv2
    from PIL import Image
    s0 = os.path.join(a.out, "sparse", "0"); im = os.path.join(a.out, "images")
    info = json.load(open(os.path.join(a.out, "pseudo.json")))
    lines = [ln for ln in open(os.path.join(s0, "images.txt")) if not ln.startswith("#")][::2]
    rows, kept, fr, k5 = [], [], [], np.ones((5, 5), np.uint8)
    X, nfl = None, []
    if a.lidar_points and os.path.isfile(a.lidar_points):          # LiDAR (dataset frame) for the floater check
        X = np.loadtxt(a.lidar_points, comments="#", usecols=(1, 2, 3), dtype=np.float64)
        cams, views = read_model(s0); views = {n: (cams[c], c2w) for n, _, c, c2w in views}
        print(f"[pseudo] floater check vs {len(X)} LiDAR points ({a.lidar_points})")
    for ln in lines:
        p = ln.split()
        if len(p) < 10:
            continue
        name = p[9]; stem = os.path.splitext(name)[0]
        rgb = os.path.join(a.renders, stem + ".png"); op = os.path.join(a.opacity, stem + ".png")
        dp = os.path.join(a.depth, stem + "_splatdepth.npy")
        if not (os.path.exists(rgb) and os.path.exists(op)):
            print(f"[pseudo] {name}: no teacher render, dropped"); continue
        alpha = np.asarray(Image.open(op), np.float32) / 255.0
        m = alpha >= a.min_opacity
        if os.path.exists(dp):
            d = np.load(dp).astype(np.float32)
            m &= d > 0
            ld = np.log(np.where(d > 0, d, 1e-3))
            rel = cv2.dilate(ld, k5) - cv2.erode(ld, k5)              # log of the 5x5 max/min depth ratio
            edge = (rel > np.log1p(a.edge_rel)).astype(np.uint8)
            if a.edge_dilate > 0:
                edge = cv2.dilate(edge, np.ones((2 * a.edge_dilate + 1,) * 2, np.uint8))
            m &= edge == 0
            if X is not None:                                        # teacher surface well in front of all LiDAR returns
                zb = lidar_zbuf(X, *views[name], a.lidar_cell, a.lidar_min_returns)
                fl = (d > 0) & np.isfinite(zb) & (d < (1 - a.lidar_tol) * zb)
                nfl.append(float(fl.mean())); m &= ~fl
        f = float(m.mean()); fr.append(f)
        if f < a.min_valid:
            continue
        Image.open(rgb).convert("RGB").save(os.path.join(im, name))
        Image.fromarray((m * 255).astype(np.uint8), mode="L").save(os.path.join(im, stem + "_mask.png"))
        rows.append((int(p[0]), [float(x) for x in p[1:5]], [float(x) for x in p[5:8]], int(p[8]), name)); kept.append(name)
    for ln in lines:                                                 # placeholders of dropped views go
        p = ln.split()
        if len(p) >= 10 and p[9] not in kept and os.path.exists(os.path.join(im, p[9])):
            os.remove(os.path.join(im, p[9]))
    write_images(os.path.join(s0, "images.txt"), rows)
    fr = np.array(fr) if fr else np.zeros(1)
    info["finish"] = {"args": {k: v for k, v in vars(a).items() if k != "func"}, "n_rendered": len(lines), "n_kept": len(rows),
                      "valid_share_pct_0_10_50_90_100": np.round(np.percentile(fr, [0, 10, 50, 90, 100]), 3).tolist(),
                      "valid_share_mean": round(float(fr.mean()), 4),
                      "lidar_floater_share_mean": round(float(np.mean(nfl)), 4) if nfl else None}
    json.dump(info, open(os.path.join(a.out, "pseudo.json"), "w"), indent=1)
    print(f"[pseudo] finish: {len(rows)}/{len(lines)} views kept (valid share >= {a.min_valid}); supervised pixel share "
          f"pct 0/10/50/90/100 {info['finish']['valid_share_pct_0_10_50_90_100']}; LiDAR floater share "
          f"{info['finish']['lidar_floater_share_mean']} -> {a.out}")


def lidar_zbuf(X, cam, c2w, cell, min_returns=3):
    """(H, W) nearest LiDAR ray distance per cell x cell block (inf where fewer than min_returns), pixel-upsampled.
    Several returns are required: a glossy floor mirrors returns below itself, single strays are common."""
    K, W, H = intrinsics(cam)
    w2c = np.linalg.inv(c2w); x = X @ w2c[:3, :3].T + w2c[:3, 3]; z = x[:, 2]
    ok = z > 1e-3; x, z = x[ok], z[ok]
    u, v = K[0, 0] * x[:, 0] / z + K[0, 2], K[1, 1] * x[:, 1] / z + K[1, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    gw, gh = -(-W // cell), -(-H // cell)
    zb = np.full(gw * gh, np.inf)
    ci = (v[ok] // cell).astype(int) * gw + (u[ok] // cell).astype(int)
    np.minimum.at(zb, ci, np.linalg.norm(x[ok], axis=1))
    zb[np.bincount(ci, minlength=gw * gh) < min_returns] = np.inf
    return np.repeat(np.repeat(zb.reshape(gh, gw), cell, 0), cell, 1)[:H, :W]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dataset", required=True, help="training COLMAP text dataset (sparse/0, images/)")
    p.add_argument("--out", required=True)
    p.add_argument("--test-split-interval", type=int, default=8)
    p.add_argument("--sim3-json", default=None, help="x_metric = s R x + t: metres -> dataset units (default: metric)")
    p.add_argument("--per-pair", type=int, default=1)
    p.add_argument("--tmin", type=float, default=0.2); p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument("--offset-m", type=float, default=0.0, help="right/down offset magnitude, metres (0 = on the interpolated path)")
    p.add_argument("--rot-deg", type=float, default=0.0, help="yaw/pitch offset magnitude, degrees")
    p.add_argument("--min-frame-gap", type=float, default=2.0, help="min |fractional index - held-out index|")
    p.add_argument("--min-dist", type=float, default=0.08, help="min distance (m) to any held-out camera centre")
    p.add_argument("--max-step", type=float, default=4.0, help="skip pairs whose step exceeds this x the median step")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=plan)
    f = sub.add_parser("finish")
    f.add_argument("--out", required=True, help="the plan's output dataset")
    f.add_argument("--renders", required=True); f.add_argument("--opacity", required=True); f.add_argument("--depth", required=True)
    f.add_argument("--min-opacity", type=float, default=0.98)
    f.add_argument("--edge-rel", type=float, default=0.08, help="depth edge: 5x5 max/min rendered depth ratio > 1 + this")
    f.add_argument("--edge-dilate", type=int, default=2, help="px")
    f.add_argument("--lidar-points", default=None, help="LiDAR points3D.txt in the dataset frame: mask teacher pixels "
                   "more than --lidar-tol nearer than every LiDAR return of their --lidar-cell block (floaters); skipped if absent")
    f.add_argument("--lidar-tol", type=float, default=0.2); f.add_argument("--lidar-cell", type=int, default=8)
    f.add_argument("--lidar-min-returns", type=int, default=3)
    f.add_argument("--min-valid", type=float, default=0.3, help="drop views with fewer supervised pixels than this share")
    f.set_defaults(func=finish)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
