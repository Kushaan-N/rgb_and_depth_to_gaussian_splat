"""Acceptance gate: a generated / completed splat ships only if it is NOT worse than the splat without
that stage, scored against real images at their real camera poses.

Takes pairs of eval_views.py results (baseline, candidate) computed on the same real-image sets (e.g.
held-out frames of this recording, frames of another recording of the same scene) and fails if, on any
set, any metric gets worse beyond a small tolerance:
  psnr, psnr_observed, psnr_unobserved   may drop by at most --psnr-tol dB
  ssim                                   may drop by at most --ssim-tol
  lpips                                  may rise by at most --lpips-tol
Writes a decision json; exit code 0 = accept, 3 = reject (callers then keep the baseline).

    python scripts/accept_gate.py --pair heldout base.json cand.json --pair other base2.json cand2.json --out decision.json
"""
from __future__ import annotations
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "BASE_JSON", "CAND_JSON"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--psnr-tol", type=float, default=0.05)
    ap.add_argument("--ssim-tol", type=float, default=0.002)
    ap.add_argument("--lpips-tol", type=float, default=0.002)
    args = ap.parse_args()
    fails, table = [], []
    for name, bj, cj in args.pair:
        b, c = json.load(open(bj)), json.load(open(cj))
        for k in ("psnr", "psnr_observed", "psnr_unobserved", "ssim", "lpips"):
            if k not in b or k not in c or b[k] is None or c[k] is None:
                continue
            delta = c[k] - b[k]
            worse = (delta > args.lpips_tol) if k == "lpips" else (-delta > (args.ssim_tol if k == "ssim" else args.psnr_tol))
            table.append({"set": name, "metric": k, "baseline": b[k], "candidate": c[k], "delta": round(delta, 4), "worse": worse})
            if worse:
                fails.append(f"{name}: {k} {b[k]} -> {c[k]}")
    decision = {"accept": not fails, "failures": fails, "checks": table,
                "tolerances": {"psnr_db": args.psnr_tol, "ssim": args.ssim_tol, "lpips": args.lpips_tol}}
    json.dump(decision, open(args.out, "w"), indent=2)
    for r in table:
        print(f"[gate] {r['set']:12s} {r['metric']:16s} {r['baseline']:9.4f} -> {r['candidate']:9.4f}  {'WORSE' if r['worse'] else 'ok'}")
    print("[gate] ACCEPT" if not fails else f"[gate] REJECT — keep the baseline ({len(fails)} regressions)", flush=True)
    return 0 if not fails else 3


if __name__ == "__main__":
    sys.exit(main())
