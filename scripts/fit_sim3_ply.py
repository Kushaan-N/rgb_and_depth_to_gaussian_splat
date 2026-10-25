#!/usr/bin/env python3
"""Recover the similarity transform between two copies of the same 3DGS splat (e.g. a pipeline-frame PLY and its
delivered *_refaxes.ply from sim3_align_splat.py, or a *_view.ply from view_frame.py): Umeyama fit on a subsample of the
gaussian centres (same rows, same order), written as {"T_new_old": 4x4 (s R | t), "scale", "resid_max", "in", "out"} --
the json format view_frame.py writes, so sbatch/qat_view.sbatch can move COLMAP datasets into the delivered frame.
Fails (exit 1) when the two files have different row counts or the fit residual exceeds --max-resid (not a copy).

    python scripts/fit_sim3_ply.py --src exports/x.ply --dst exports/x_refaxes.ply [--out exports/x_refaxes.json]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def centres(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    return np.stack([np.asarray(v[c], dtype=np.float64) for c in "xyz"], 1)


def umeyama(X, Y):
    """Y ~ s R X + t (least squares) -> s, R, t."""
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean())
    return s, R, my - s * R @ mx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="PLY in the old frame"); ap.add_argument("--dst", required=True, help="the same splat, new frame")
    ap.add_argument("--out", default=None, help="json (default <dst stem>.json)")
    ap.add_argument("--samples", type=int, default=200000); ap.add_argument("--max-resid", type=float, default=1e-3,
                                                                           help="max |s R x + t - y| (dst units)")
    a = ap.parse_args()
    X, Y = centres(a.src), centres(a.dst)
    if len(X) != len(Y):
        sys.exit(f"[fit_sim3] row counts differ: {len(X)} vs {len(Y)}")
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1); idx = np.flatnonzero(ok)
    idx = np.random.default_rng(0).choice(idx, min(a.samples, len(idx)), replace=False)
    s, R, t = umeyama(X[idx], Y[idx])
    res = float(np.linalg.norm(s * X[idx] @ R.T + t - Y[idx], axis=1).max())
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    out = a.out or os.path.splitext(a.dst)[0] + ".json"
    json.dump({"in": os.path.abspath(a.src), "out": os.path.abspath(a.dst), "T_new_old": T.tolist(), "scale": s, "resid_max": res},
              open(out, "w"), indent=1)
    print(f"[fit_sim3] scale {s:.6f}, max residual {res:.2e} -> {out}")
    if res > a.max_resid:
        sys.exit(f"[fit_sim3] residual {res:.3g} > {a.max_resid}: {a.dst} is not a similarity copy of {a.src}")


if __name__ == "__main__":
    main()
