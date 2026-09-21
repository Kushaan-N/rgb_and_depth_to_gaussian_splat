#!/usr/bin/env python3
"""Emit a TUM file that renders ONE COLMAP pose under several camera conventions.

We know the Isaac/NuRec render is rotated+smeared vs the ground truth, i.e. the camera basis
we feed is wrong. This writes the same COLMAP pose transformed by a set of candidate basis
changes (applied to the OpenCV cam-to-world rotation R_wc). Rendering these and comparing each
to the GT frame identifies the correct convention. Frame i (timestamp = 100000+i) = candidate i.

    python scripts/convention_test.py <images.txt> <ts> <out.tum>
"""
import sys
import numpy as np


def qwxyz_R(qw, qx, qy, qz):
    n = (qw*qw+qx*qx+qy*qy+qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def R_qxyzw(R):
    t = np.trace(R)
    if t > 0:
        s = (t+1) ** 0.5 * 2; w = .25*s
        x = (R[2, 1]-R[1, 2])/s; y = (R[0, 2]-R[2, 0])/s; z = (R[1, 0]-R[0, 1])/s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1+R[0, 0]-R[1, 1]-R[2, 2]) ** 0.5*2; w = (R[2, 1]-R[1, 2])/s
        x = .25*s; y = (R[0, 1]+R[1, 0])/s; z = (R[0, 2]+R[2, 0])/s
    elif R[1, 1] > R[2, 2]:
        s = (1+R[1, 1]-R[0, 0]-R[2, 2]) ** 0.5*2; w = (R[0, 2]-R[2, 0])/s
        x = (R[0, 1]+R[1, 0])/s; y = .25*s; z = (R[1, 2]+R[2, 1])/s
    else:
        s = (1+R[2, 2]-R[0, 0]-R[1, 1]) ** 0.5*2; w = (R[1, 0]-R[0, 1])/s
        x = (R[0, 2]+R[2, 0])/s; y = (R[1, 2]+R[2, 1])/s; z = .25*s
    return x, y, z, w


def Rz(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def Rx(d):
    r = np.radians(d); c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


CANDS = [
    ("identity(rawOpenCV)", np.eye(3)),
    ("flipYZ(cv2usd)",      np.diag([1., -1, -1])),
    ("Rz+90",               Rz(90)),
    ("Rz-90",               Rz(-90)),
    ("flipYZ*Rz+90",        np.diag([1., -1, -1]) @ Rz(90)),
    ("flipYZ*Rz-90",        np.diag([1., -1, -1]) @ Rz(-90)),
    ("flipXY",              np.diag([-1., -1, 1])),
    ("flipXZ",              np.diag([-1., 1, -1])),
    ("Rx-90",               Rx(-90)),
    ("flipYZ*Rx-90",        np.diag([1., -1, -1]) @ Rx(-90)),
]


def main():
    images_txt, ts, out = sys.argv[1], sys.argv[2], sys.argv[3]
    row = None
    for ln in open(images_txt):
        p = ln.split()
        if len(p) >= 10 and p[9].startswith(ts):
            row = p; break
    if row is None:
        print(f"ts {ts} not found", file=sys.stderr); sys.exit(1)
    qw, qx, qy, qz = map(float, row[1:5]); tx, ty, tz = map(float, row[5:8])
    Rcw = qwxyz_R(qw, qx, qy, qz); Rwc = Rcw.T
    C = -Rwc @ np.array([tx, ty, tz])
    with open(out, "w") as f:
        for i, (name, T) in enumerate(CANDS):
            x, y, z, w = R_qxyzw(Rwc @ T)
            f.write(f"{100000+i} {C[0]:.6f} {C[1]:.6f} {C[2]:.6f} {x:.6f} {y:.6f} {z:.6f} {w:.6f}\n")
            print(f"frame {100000+i}.png (index {i}) = {name}")
    print(f"wrote {len(CANDS)} candidate poses to {out}")


if __name__ == "__main__":
    main()
