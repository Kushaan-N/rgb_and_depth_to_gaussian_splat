#!/usr/bin/env python3
"""Composite a driving robot into the SHARP 3DGRUT renders (GPU-free).

The Isaac/NuRec render of the .usdz is soft; 3DGRUT's own renders are sharp. So instead of
rendering the robot in Isaac, we take a sharp 3DGRUT frame as the background, recover its
camera pose (NCC-match to the COLMAP ground-truth frames), and project a shaded robot cube
onto it with the true intrinsics. The robot is placed in CAMERA-frame coordinates (a floor
point in front of the camera, swept laterally) so it's always in view and grounded, then
animated across frames -> a robot driving through the sharp reconstructed lab.

    python scripts/robot_composite.py --seq mocap1_well-lit_trot --frames 60
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import cv2


def qwxyz_R(qw, qx, qy, qz):
    n = (qw*qw+qx*qx+qy*qy+qz*qz) ** 0.5 or 1.0
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])


def read_colmap(images_txt, cameras_txt):
    cam = None
    for ln in open(cameras_txt):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
        cam = (int(p[2]), int(p[3]), fx, fy, cx, cy)
        break
    poses = {}
    for ln in open(images_txt):
        p = ln.split()
        if len(p) >= 10 and p[9].endswith(".png"):
            R = qwxyz_R(*map(float, p[1:5]))
            t = np.array(list(map(float, p[5:8])))
            poses[p[9]] = (R, t)
    return cam, poses


def ncc(a, b):
    a = a.astype(np.float32) - a.mean(); b = b.astype(np.float32) - b.mean()
    return float((a*b).sum() / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-9))


def match_pose(bg_gray, poses, rgb_dir):
    """Find the COLMAP image whose GT best matches the background render (NCC on small gray)."""
    best, bestname = -2, None
    small = cv2.resize(bg_gray, (80, 60))
    for name in poses:
        gp = os.path.join(rgb_dir, name)
        im = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        s = ncc(small, cv2.resize(im, (80, 60)))
        if s > best:
            best, bestname = s, name
    return bestname, best


CUBE = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], float)
FACES = [(0,1,3,2),(4,6,7,5),(0,2,6,4),(1,5,7,3),(2,3,7,6),(0,4,5,1)]


def draw_cube(img, Xw, size, R, t, K, base=(210,120,30)):
    """Project + paint a shaded cube at world center Xw onto img (OpenCV cam R,t; K intrinsics)."""
    V = Xw[None, :] + CUBE * size
    Vc = (R @ V.T).T + t                                   # world->camera
    if np.any(Vc[:, 2] <= 0.05):
        return
    uv = np.stack([K[0, 0]*Vc[:, 0]/Vc[:, 2] + K[0, 2],
                   K[1, 1]*Vc[:, 1]/Vc[:, 2] + K[1, 2]], 1)
    order = sorted(range(len(FACES)), key=lambda f: -Vc[list(FACES[f]), 2].mean())  # far->near
    for f in order:
        idx = list(FACES[f])
        c0, c1 = Vc[idx[0]], Vc[idx[1]]; c2 = Vc[idx[2]]
        n = np.cross(c1-c0, c2-c0); n = n/(np.linalg.norm(n)+1e-9)
        shade = 0.45 + 0.55*max(0.0, -n[2])                # brighter facing camera
        col = tuple(int(min(255, c*shade)) for c in base)  # BGR-ish
        cv2.fillConvexPoly(img, uv[idx].astype(np.int32), col, cv2.LINE_AA)
    # outline
    for f in order:
        cv2.polylines(img, [uv[list(FACES[f])].astype(np.int32)], True, (20, 20, 20), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="mocap1_well-lit_trot")
    ap.add_argument("--bg-index", type=int, default=25, help="which 3DGRUT eval render to use as background")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fwd", type=float, default=2.6, help="robot distance in front of camera (m)")
    ap.add_argument("--height", type=float, default=1.0, help="robot drop below camera toward floor (m)")
    ap.add_argument("--sweep", type=float, default=1.4, help="lateral sweep half-width (m)")
    ap.add_argument("--size", type=float, default=0.35)
    args = ap.parse_args()
    ws = os.environ.get("CEAR_WS", "/scratch4/workspace/%s-cear" % os.environ.get("USER", ""))
    nav = f"{ws}/outputs/{args.seq}"
    ct = f"{nav}/colmap_train/sparse/0"
    rgb_dir = f"{nav}/precond/rgb"
    renders = sorted(glob.glob(f"{nav}/train_3dgrut/*/*/ours_*/renders/*.png"))
    out_dir = f"{nav}/robot_composite"; os.makedirs(out_dir, exist_ok=True)

    bg_path = renders[min(args.bg_index, len(renders)-1)]
    bg = cv2.imread(bg_path); H, W = bg.shape[:2]
    cam, poses = read_colmap(f"{ct}/images.txt", f"{ct}/cameras.txt")
    _, _, fx, fy, cx, cy = cam
    # intrinsics scaled to the render resolution
    K = np.array([[fx*W/cam[0], 0, cx*W/cam[0]], [0, fy*H/cam[1], cy*H/cam[1]], [0, 0, 1]])
    name, score = match_pose(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), poses, rgb_dir)
    R, t = poses[name]
    print(f"[composite] bg={os.path.basename(bg_path)} matched pose {name} (NCC {score:.2f})", flush=True)

    vw = cv2.VideoWriter(f"{out_dir}/robot_in_lab.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))
    n_ok = 0
    for k in range(args.frames):
        s = (k/max(args.frames-1, 1))*2 - 1                # -1..+1 lateral sweep
        # robot position in CAMERA frame (OpenCV: +x right, +y down, +z forward) -> world
        p_cam = np.array([s*args.sweep, args.height, args.fwd])
        Xw = R.T @ (p_cam - t)
        frame = bg.copy()
        draw_cube(frame, Xw, args.size, R, t, K)
        cv2.imwrite(f"{out_dir}/comp_{k:04d}.png", frame); vw.write(frame); n_ok += 1
    vw.release()
    print(f"[composite] wrote {n_ok} frames + robot_in_lab.mp4 to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
