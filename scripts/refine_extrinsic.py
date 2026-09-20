"""§5.3 — constant-extrinsic ΔT refinement fallback (docs/PLAN.md §5.3).

The marker→RGB chain passes through CAD-derived (RGB↔Robot) and factory-default
(intrinsics, RGB↔Depth) links, so a small CONSTANT extrinsic error (a few mm / fractions
of a degree from mounting tolerance) is plausible even after Gate 2 passes visually — and
it yields a splat that is "almost right" but soft.

This optimizes a single rigid ΔT (6 DoF) applied as
    T_world_cam = T_world_marker @ T_marker_cam @ ΔT
to minimize the Gate-2 photometric warp error over frame pairs. Because ΔT is ONE
transform for the whole sequence, it can only absorb the constant error mode — not
per-frame pose noise. Run it ONLY if Gate 2 passes but Gates 3-4 show systematic softness;
a large ΔT means something else is wrong (go back to TRAP 4).

    python scripts/refine_extrinsic.py --config configs/mocap1_well-lit_trot.yaml
    # then paste the printed delta_T into the config's frames.extrinsic.delta_T and re-run.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
from build_poses import compute_world_cam
from verify_reprojection import warp, photo_error, _pairs


def _delta_from_params(p):
    return pu.make_T(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])


def _cache_pairs(cfg, kept, k, n_pairs):
    """Precompute pose-independent data per pair (backprojected cam-frame points, images)."""
    root = cfg["sequence"]["data_root"]
    depth_dir = os.path.join(root, cfg["sequence"]["depth_dir"])
    rgb_dir = os.path.join(root, cfg["sequence"]["rgb_dir"])
    cache = []
    for (i, j) in _pairs(len(kept), k, n_pairs):
        di_m, vi = iu.depth_to_meters(iu.read_depth_raw(
            os.path.join(depth_dir, kept[i].depth_name)), cfg)
        rgb_i = iu.read_rgb(os.path.join(rgb_dir, kept[i].rgb_name))
        rgb_j = iu.read_rgb(os.path.join(rgb_dir, kept[j].rgb_name))
        cache.append((i, j, di_m, vi, rgb_i, rgb_j))
    return cache


def refine(cfg: dict, k: int = 8, n_pairs: int = 8) -> dict:
    # base poses with ΔT = identity (A_i); final pose(ΔT) = A_i @ ΔT
    base = copy.deepcopy(cfg)
    base.setdefault("frames", {}).setdefault("extrinsic", {})["delta_T"] = None
    kept, A, calib, *_ = compute_world_cam(base)
    K = calib.K
    cache = _cache_pairs(cfg, kept, k, n_pairs)

    def cost(p):
        dT = _delta_from_params(p)
        poses = A @ dT
        errs = []
        for (i, j, di_m, vi, rgb_i, rgb_j) in cache:
            warped, mask = warp(di_m, vi, rgb_i, poses[i], poses[j], K)
            errs.append(photo_error(warped, mask, rgb_j))
        return float(np.nanmean(errs))

    e0 = cost(np.zeros(6))
    res = minimize(cost, np.zeros(6), method="Powell",
                   options={"xtol": 1e-4, "ftol": 1e-3, "maxiter": 200})
    dT = _delta_from_params(res.x)
    rot_deg = float(np.degrees(np.linalg.norm(res.x[:3])))
    trans_mm = float(np.linalg.norm(res.x[3:]) * 1000)
    rel_gain = (e0 - float(res.fun)) / e0 if e0 > 0 else 0.0
    return {"delta_T": dT.tolist(), "cost_before": e0, "cost_after": float(res.fun),
            "rotation_deg": rot_deg, "translation_mm": trans_mm,
            "relative_gain": float(rel_gain),
            # only worth applying if it meaningfully lowers the warp error (>10%); on a
            # correct extrinsic the optimizer just wanders the noise floor. A genuine
            # 1-2 deg extrinsic error clears this bar easily.
            "significant": bool(rel_gain > 0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    r = refine(cfg, k=args.k, n_pairs=args.pairs)
    out = os.path.join(cfg["paths"]["out_root"], "extrinsic_refine.json")
    pu.save_json(out, r)
    print(f"[refine] warp err {r['cost_before']:.3f} -> {r['cost_after']:.3f} "
          f"({r['relative_gain']*100:.1f}% gain)  ΔT = {r['rotation_deg']:.3f} deg / "
          f"{r['translation_mm']:.1f} mm")
    if not r["significant"]:
        print("[refine] gain < 10% — the extrinsic is already fine; do NOT apply ΔT.")
        return
    if r["rotation_deg"] > 2.0 or r["translation_mm"] > 50.0:
        print("[refine] WARNING: ΔT is large — this is not a mounting-tolerance correction. "
              "Re-check TRAP 4 (pose frame / extrinsic direction) before trusting it.")
    print(f"[refine] wrote {out}. To apply, set frames.extrinsic.delta_T in the config to:")
    print("  " + str(r["delta_T"]))


if __name__ == "__main__":
    main()
