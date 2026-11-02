#!/usr/bin/env python3
"""Splat-level ensemble: merge several splats of the same scene into ONE standard 3DGS PLY, and prune it back
(see scripts/methods/__init__.py; sbatch/fidelity_methods_bench.sbatch arm fm_ensm).

CEAR: the pixel mean of three fine-tuned seeds' live renders beats each seed by ~+0.25 dB, but distilling it into one
seed recovers only ~+0.03. Here the seeds are not distilled, they are UNIONED:
  merge  concatenate the vertex rows of --plys (identical layout; same frame) and rescale each opacity so that n
         overlapping copies of the same surface composite like one: alpha' = 1 - (1 - alpha)^(1/n) (exact for n
         identical, co-located, perfectly sorted copies; elsewhere a first guess the joint fine-tune corrects:
         ftv_distill.py --init merged.ply toward the seeds' render mean and the real frames)
  prune  keep the --keep gaussians with the largest blending weight sum_px T alpha over the TRAINING views of --dataset
         (held-out frames excluded by --test-split-interval) in train_viewer_gsplat.py's viewer-matched rasteriser
         (PYTHONPATH=$CEAR_WS/gsplat-viewer-pc274); --keep < 1 = fraction, else a count. Rows keep the input layout.
    python scripts/methods/ens_merge.py merge --plys a.ply b.ply c.ply --out m.ply
    python scripts/methods/ens_merge.py prune --ply m.ply --dataset DS --test-split-interval 8 --keep 2000000 --out p.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def merge(a):
    from plyfile import PlyData, PlyElement
    V = [PlyData.read(f)["vertex"].data for f in a.plys]
    if len({v.dtype for v in V}) != 1:
        raise SystemExit("[ens-merge] PLY layouts differ")
    m = np.concatenate(V); n = len(V)
    o = 1 / (1 + np.exp(-m["opacity"].astype(np.float64)))
    o2 = np.clip(1 - (1 - o) ** (1.0 / n), 1e-6, 1 - 1e-6)
    m["opacity"] = np.log(o2 / (1 - o2)).astype(np.float32)
    PlyData([PlyElement.describe(m, "vertex")], comments=[f"ens_merge of {n} splats, alpha' = 1-(1-alpha)^(1/{n})"]).write(a.out)
    r = {"inputs": a.plys, "counts": [len(v) for v in V], "merged": len(m), "out": a.out}
    print("ENS_MERGE " + json.dumps(r), flush=True)


def prune(a):
    import torch
    from plyfile import PlyData, PlyElement
    import train_viewer_gsplat as tvg
    dev = "cuda"
    rast = tvg.ViewerGS(argparse.Namespace(viewer=a.viewer, eps2d=None, cull_l1=None, aa=False, sh_band2=None, min_px=0.0,
                                           near=1e-4, cap_px=1024.0, sort="depth"))
    p = tvg.load_ply_params(a.ply, dev)
    tr = tvg.load_views(a.dataset, a.test_split_interval, "train", dev)
    w = tvg.contributions(rast, p, tr).cpu().numpy()
    n = len(w); k = int(a.keep * n) if a.keep < 1 else min(int(a.keep), n)
    keep = np.sort(np.argsort(-w)[:k])
    v = PlyData.read(a.ply)["vertex"].data[keep]
    PlyData([PlyElement.describe(v, "vertex")], comments=[f"ens_merge prune: top {k} of {n} by training-view blending weight"]).write(a.out)
    r = {"in": n, "kept": k, "weight_kept_frac": round(float(w[keep].sum() / max(w.sum(), 1e-12)), 5),
         "weight_pct_dropped_max": round(float(np.sort(w)[n - k - 1]) if k < n else 0.0, 4), "out": a.out}
    print("ENS_PRUNE " + json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("merge"); g.add_argument("--plys", nargs="+", required=True); g.add_argument("--out", required=True)
    q = sp.add_parser("prune"); q.add_argument("--ply", required=True); q.add_argument("--dataset", required=True)
    q.add_argument("--test-split-interval", type=int, default=8); q.add_argument("--keep", type=float, required=True)
    q.add_argument("--viewer", default="pc274"); q.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a) if a.cmd == "merge" else prune(a)


if __name__ == "__main__":
    main()
