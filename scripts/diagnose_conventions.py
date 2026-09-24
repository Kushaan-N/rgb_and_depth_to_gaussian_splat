"""Diagnose the pose-chain convention (TRAP-4) before trusting a large ΔT.

refine_extrinsic.py fits ONE constant ΔT to reduce the Gate-2 photometric warp error. If
that ΔT comes out large (>~2°/50mm) it usually means the *convention* is wrong (extrinsic
direction / pose frame), not that calibration needs a nudge. This sweeps the conventions and
reports the BASELINE warp error (ΔT = identity) for each: the lowest one is the correct
convention. If a different convention is materially lower than the current config, that's the
real fix (a flag flip) — apply it instead of a bolted-on ΔT.

    python scripts/diagnose_conventions.py --config configs/mocap2_well-lit_trot.yaml
"""
from __future__ import annotations
import argparse, copy, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
from build_poses import compute_world_cam
from verify_reprojection import warp, photo_error, _pairs


def baseline_cost(cfg: dict, k: int = 8, n_pairs: int = 8) -> float:
    """Mean Gate-2 photometric warp error over frame pairs with ΔT = identity."""
    c = copy.deepcopy(cfg)
    c.setdefault("frames", {}).setdefault("extrinsic", {})["delta_T"] = None
    kept, A, calib, *_ = compute_world_cam(c)
    K = calib.K
    root = c["sequence"]["data_root"]
    depth_dir = os.path.join(root, c["sequence"]["depth_dir"])
    rgb_dir = os.path.join(root, c["sequence"]["rgb_dir"])
    errs = []
    for (i, j) in _pairs(len(kept), k, n_pairs):
        di_m, vi = iu.depth_to_meters(iu.read_depth_raw(os.path.join(depth_dir, kept[i].depth_name)), c)
        rgb_i = iu.read_rgb(os.path.join(rgb_dir, kept[i].rgb_name))
        rgb_j = iu.read_rgb(os.path.join(rgb_dir, kept[j].rgb_name))
        warped, mask = warp(di_m, vi, rgb_i, A[i], A[j], K)
        errs.append(photo_error(warped, mask, rgb_j))
    return float(np.nanmean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)

    combos = [("marker", "marker_to_cam"), ("marker", "cam_to_marker"),
              ("robot", "robot_to_cam"), ("robot", "cam_to_robot")]
    print(f"{'pose_frame':<10}{'direction':<16}{'baseline warp err':>18}")
    results = []
    for frame, direction in combos:
        c = copy.deepcopy(cfg)
        c["frames"]["pose_frame"] = frame
        c["frames"].setdefault("extrinsic", {})
        key = "rgb_marker_direction" if frame == "marker" else "rgb_robot_direction"
        c["frames"]["extrinsic"][key] = direction
        try:
            e = baseline_cost(c, args.k, args.pairs)
        except Exception as ex:
            e = float("nan"); print(f"  ({frame}/{direction} failed: {ex})")
        results.append((frame, direction, e))
        print(f"{frame:<10}{direction:<16}{e:>18.4f}")

    ok = [r for r in results if not np.isnan(r[2])]
    if not ok:
        print("no valid convention evaluated"); return
    best = min(ok, key=lambda r: r[2])
    cur = next(r for r in results if r[0] == "marker" and r[1] == "marker_to_cam")
    print(f"\nLOWEST baseline : {best[0]}/{best[1]} = {best[2]:.4f}")
    print(f"current config  : {cur[0]}/{cur[1]} = {cur[2]:.4f}")
    if best[2] < cur[2] * 0.85:
        print(">> A DIFFERENT convention is materially better — the config's TRAP-4 setting is "
              "likely wrong. Fix the convention (flag flip), then re-run refine (ΔT should shrink).")
    else:
        print(">> current convention is best/near-best — the large ΔT is NOT a direction bug; it is "
              "real physical/temporal error a constant ΔT can't legitimately absorb. Do not apply it.")


if __name__ == "__main__":
    main()
