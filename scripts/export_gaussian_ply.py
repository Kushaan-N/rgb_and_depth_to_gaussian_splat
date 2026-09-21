#!/usr/bin/env python3
"""Option B, stage 1 — export the trained gaussians to a standard 3DGS .ply (3DGRUT).

Just dumps the gaussian representation (centres + opacity + scale + SH). mesh_from_gaussians.py
then reconstructs a surface directly from it (Poisson) — geometry straight from the splat, no
rendering, no hand placement. Runs in the 3DGRUT venv.

    python export_gaussian_ply.py --checkpoint ckpt_last.pt --out gaussians.ply
"""
from __future__ import annotations
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from threedgrut.export.scripts.export_usd import load_model_from_checkpoint
    from threedgrut.export.formats.ply import PLYExporter
    model, conf, _bg, _pp = load_model_from_checkpoint(args.checkpoint)
    n = model.get_positions().shape[0]
    print(f"[export_ply] {n} gaussians -> {args.out}", flush=True)
    PLYExporter().export(model, args.out, conf=conf)
    print(f"[export_ply] wrote {args.out} ({os.path.getsize(args.out)//1024//1024} MB)", flush=True)


if __name__ == "__main__":
    main()
