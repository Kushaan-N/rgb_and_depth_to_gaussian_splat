"""Hyper-parameter overrides for scripts/train_viewer_gsplat.py that have no trainer flag (see scripts/methods/__init__.py),
used by the random search sbatch/hp_search.sbatch:
  --lr-mult means=1.5,shN=0.5,...   multiplies the named parameter groups' Adam learning rates at the start (means keeps
                                    its exponential decay; groups: means scales quats opacities sh0 shN)
  --reset-mult x                    Improved-GS / default strategy opacity-reset interval x x (strategy attribute reset_every)
The exported PLY is untouched.
"""
from __future__ import annotations


def add_hparam_args(g):
    g.add_argument("--lr-mult", default="", help="comma list group=mult of learning-rate multipliers ('' = off)")
    g.add_argument("--reset-mult", type=float, default=1.0, help="opacity-reset interval multiplier (1 = off)")


def apply_hparams(args, opts, strat):
    rec = {}
    for kv in filter(None, args.lr_mult.split(",")):
        k, m = kv.split("=")
        if k not in opts:
            raise SystemExit(f"[methods] --lr-mult: unknown group {k} (have {list(opts)})")
        for g in opts[k].param_groups:
            g["lr"] *= float(m)
        rec[k] = float(m)
    if args.reset_mult != 1.0 and hasattr(strat, "reset_every"):
        strat.reset_every = max(int(strat.reset_every * args.reset_mult), 1); rec["reset_every"] = strat.reset_every
    if rec:
        print(f"[methods] hparams: {rec}", flush=True)
    return rec
