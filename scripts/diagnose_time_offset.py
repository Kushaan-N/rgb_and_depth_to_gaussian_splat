"""Sweep the RGB->event time offset to test whether a timing error is the residual pose error.

If refine's ΔT is large but the pose-chain convention is already correct (see
diagnose_conventions.py), the residual is per-frame, not a constant extrinsic. The most likely
cause on a *moving* robot is a wrong sensor time offset: a few ms of desync shifts every
interpolated pose by a motion-dependent amount -> blur that a single constant ΔT can't fix.

This re-uses the Gate-2 warp error and sweeps timestamps.offsets_s.rgb. The offset with the
lowest error is the correct sync; if it differs meaningfully from the config, re-syncing (then
retraining) is a real, cheap fidelity fix.

    python scripts/diagnose_time_offset.py --config configs/mocap2_well-lit_trot.yaml
"""
from __future__ import annotations
import argparse, copy, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from diagnose_conventions import baseline_cost


def sweep(cfg: dict, offsets):
    res = []
    for off in offsets:
        c = copy.deepcopy(cfg)
        c.setdefault("timestamps", {}).setdefault("offsets_s", {})["rgb"] = float(off)
        try:
            e = baseline_cost(c)
        except Exception as ex:
            e = float("nan"); print(f"  (offset {off*1000:.2f} ms failed: {ex})")
        res.append((float(off), e))
        print(f"  rgb offset {off*1000:8.2f} ms  ->  warp err {e:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--span-ms", type=float, default=12.0)
    ap.add_argument("--step-ms", type=float, default=2.0)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    cur = float(cfg.get("timestamps", {}).get("offsets_s", {}).get("rgb", 0.0))
    curerr = baseline_cost(cfg)
    print(f"current rgb offset: {cur*1000:.3f} ms  (warp err {curerr:.4f})\n--- coarse sweep ---")

    span, step = args.span_ms / 1000.0, args.step_ms / 1000.0
    coarse = np.arange(cur - span, cur + span + 1e-9, step)
    r1 = sweep(cfg, coarse)
    ok1 = [r for r in r1 if not np.isnan(r[1])]
    best1 = min(ok1, key=lambda r: r[1])

    print("--- fine sweep around coarse best ---")
    fine = np.arange(best1[0] - step, best1[0] + step + 1e-9, step / 4.0)
    r2 = sweep(cfg, fine)
    ok2 = [r for r in r2 if not np.isnan(r[1])]
    best = min(ok1 + ok2, key=lambda r: r[1])

    gain = (curerr - best[1]) / curerr if curerr > 0 else 0.0
    print(f"\nBEST offset : {best[0]*1000:.3f} ms  (warp err {best[1]:.4f})")
    print(f"current     : {cur*1000:.3f} ms  (warp err {curerr:.4f})")
    print(f"shift        : {(best[0]-cur)*1000:+.3f} ms   relative gain: {gain*100:.1f}%")
    if gain > 0.10 and abs(best[0] - cur) > 0.0005:
        print(f">> TIMING is a real culprit. Set timestamps.offsets_s.rgb = {best[0]:.6f}, then "
              f"regenerate poses and retrain.")
    else:
        print(">> offset already ~optimal — timing is NOT the main residual. Remaining softness is "
              "the 640x480 sensor ceiling + per-frame jitter, not a cheap constant fix.")


if __name__ == "__main__":
    main()
