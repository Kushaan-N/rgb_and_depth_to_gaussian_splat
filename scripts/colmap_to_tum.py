#!/usr/bin/env python3
"""COLMAP images.txt -> TUM trajectory (camera-to-world) for nurec_render.py `poses` mode.

COLMAP stores world->camera (R_cw, t_cw) with a scalar-first quaternion. TUM wants
camera-to-world: position = camera centre C = -R_cw^T t_cw, orientation = R_wc = R_cw^T,
emitted as `timestamp tx ty tz qx qy qz qw`. COLMAP and TUM share the OpenCV optical
convention (+Z forward, +Y down), so no basis flip is needed. Picks N evenly-spaced views.
"""
import sys

def qwxyz_to_R(qw, qx, qy, qz):
    n = (qw*qw+qx*qx+qy*qy+qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return [
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ]

def R_to_qxyzw(R):
    t = R[0][0]+R[1][1]+R[2][2]
    if t > 0:
        s = (t+1.0) ** 0.5 * 2; w=0.25*s
        x=(R[2][1]-R[1][2])/s; y=(R[0][2]-R[2][0])/s; z=(R[1][0]-R[0][1])/s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s=(1+R[0][0]-R[1][1]-R[2][2])**0.5*2; w=(R[2][1]-R[1][2])/s
        x=0.25*s; y=(R[0][1]+R[1][0])/s; z=(R[0][2]+R[2][0])/s
    elif R[1][1] > R[2][2]:
        s=(1+R[1][1]-R[0][0]-R[2][2])**0.5*2; w=(R[0][2]-R[2][0])/s
        x=(R[0][1]+R[1][0])/s; y=0.25*s; z=(R[1][2]+R[2][1])/s
    else:
        s=(1+R[2][2]-R[0][0]-R[1][1])**0.5*2; w=(R[1][0]-R[0][1])/s
        x=(R[0][2]+R[2][0])/s; y=(R[1][2]+R[2][1])/s; z=0.25*s
    return x, y, z, w

def transpose(R): return [[R[j][i] for j in range(3)] for i in range(3)]
def matvec(R, v): return [sum(R[i][k]*v[k] for k in range(3)) for i in range(3)]

def main():
    # args: images_txt out_tum [n] [stride] [start]
    #   stride>0 -> CONSECUTIVE sampling (smooth walkthrough): poses start, start+stride, ...
    #   stride<=0 (default) -> n evenly-spaced poses across the whole trajectory
    images_txt, out_tum = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    start = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    poses = []
    for ln in open(images_txt):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 10:            # skip the empty points2D lines
            continue
        try:
            qw, qx, qy, qz = map(float, p[1:5]); tx, ty, tz = map(float, p[5:8])
        except ValueError:
            continue
        name = p[9]
        ts = "".join(c for c in name.split("_")[0] if c.isdigit()) or str(len(poses))
        poses.append((ts, qw, qx, qy, qz, tx, ty, tz))
    if not poses:
        print("no poses parsed", file=sys.stderr); sys.exit(1)
    if stride > 0:                                  # consecutive walkthrough
        if start < 0:
            start = max(0, (len(poses) - n * stride) // 2)   # centre the window
        idxs = [start + i * stride for i in range(n) if start + i * stride < len(poses)]
    else:
        idxs = [int(round(i*(len(poses)-1)/(n-1))) for i in range(n)] if n > 1 else [len(poses)//2]
    with open(out_tum, "w") as f:
        for i in idxs:
            ts, qw, qx, qy, qz, tx, ty, tz = poses[i]
            Rcw = qwxyz_to_R(qw, qx, qy, qz); Rwc = transpose(Rcw)
            C = matvec(Rwc, [tx, ty, tz]); C = [-c for c in C]
            ox, oy, oz, ow = R_to_qxyzw(Rwc)
            f.write(f"{ts} {C[0]:.6f} {C[1]:.6f} {C[2]:.6f} {ox:.6f} {oy:.6f} {oz:.6f} {ow:.6f}\n")
    print(f"wrote {len(idxs)} poses to {out_tum} (of {len(poses)} images)")

if __name__ == "__main__":
    main()
