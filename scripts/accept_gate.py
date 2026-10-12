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
    ap.add_argument("--perceptual", action="store_true",
                    help="perceptual trade-off: allow a PSNR drop up to --perceptual-psnr-tol on a set when LPIPS improves "
                         "by >= --perceptual-lpips-gain there (and DISTS, if both jsons carry it, does not get worse)")
    ap.add_argument("--perceptual-psnr-tol", type=float, default=0.3)
    ap.add_argument("--perceptual-ssim-tol", type=float, default=0.005)
    ap.add_argument("--perceptual-lpips-gain", type=float, default=0.005)
    args = ap.parse_args()
    fails, table = [], []
    for name, bj, cj in args.pair:
        b, c = json.load(open(bj)), json.load(open(cj))
        # an invisible pixel-accuracy loss is acceptable for a clear perceptual gain (user rule, 2026-10-12)
        sharper = (args.perceptual and b.get("lpips") is not None and c.get("lpips") is not None
                   and b["lpips"] - c["lpips"] >= args.perceptual_lpips_gain
                   and (b.get("dists") is None or c.get("dists") is None or c["dists"] <= b["dists"]))
        ptol = args.perceptual_psnr_tol if sharper else args.psnr_tol
        stol = args.perceptual_ssim_tol if sharper else args.ssim_tol
        for k in ("psnr", "psnr_observed", "psnr_unobserved", "ssim", "lpips", "dists"):
            if k not in b or k not in c or b[k] is None or c[k] is None:
                continue
            delta = c[k] - b[k]
            worse = (delta > args.lpips_tol) if k in ("lpips", "dists") else (-delta > (stol if k == "ssim" else ptol))
            table.append({"set": name, "metric": k, "baseline": b[k], "candidate": c[k], "delta": round(delta, 4), "worse": worse})
            if worse:
                fails.append(f"{name}: {k} {b[k]} -> {c[k]}")
    decision = {"accept": not fails, "failures": fails, "checks": table, "perceptual_mode": args.perceptual,
                "tolerances": {"psnr_db": args.psnr_tol, "ssim": args.ssim_tol, "lpips": args.lpips_tol,
                               "perceptual_psnr_db": args.perceptual_psnr_tol, "perceptual_ssim": args.perceptual_ssim_tol}}
    json.dump(decision, open(args.out, "w"), indent=2)
    for r in table:
        print(f"[gate] {r['set']:12s} {r['metric']:16s} {r['baseline']:9.4f} -> {r['candidate']:9.4f}  {'WORSE' if r['worse'] else 'ok'}")
    print("[gate] ACCEPT" if not fails else f"[gate] REJECT — keep the baseline ({len(fails)} regressions)", flush=True)
    return 0 if not fails else 3


if __name__ == "__main__":
    sys.exit(main())
