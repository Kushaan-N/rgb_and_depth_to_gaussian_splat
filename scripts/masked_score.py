#!/usr/bin/env python3
"""Score saved renders against real frames ON A MASK (3DGRUT venv: torchmetrics; GPU).

For off-path benchmarks most of a held-out view may show content no training frame saw; a method can only invent it,
so the fair number is the one restricted to what the training data determines. Per frame, <pred-dir>/<stem>.png vs
<dataset>/images/<name> over the pixels where the mask file equals --mask-value (default: > 127):
  --mask-dir D --mask-suffix _covis.png --mask-value 255     covis_masks.py: observed by a training camera
  --mask-dir D --mask-suffix _mask.png                       e.g. make_lidar_ibr_views.py: photo-consistent IBR pixels
(--mask-dir defaults to <dataset>/images). Reports psnr_mask (pixel-pooled), psnr_mask_per_view (mean over frames with
>= --min-pixels masked pixels), lpips_mask / ssim_mask (prediction composited over the real frame outside the mask, so
only masked pixels can differ: comparable between methods on the SAME masks only; LPIPS-VGG as eval_views.py), the mask coverage, and the plain full-image psnr.

    python scripts/masked_score.py --pred-dir shots/ --dataset TEST --mask-suffix _covis.png --mask-value 255 --json-out r.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--mask-dir", default=None); ap.add_argument("--mask-suffix", default="_mask.png")
    ap.add_argument("--mask-value", type=int, default=None, help="exact value that counts as inside (default: > 127)")
    ap.add_argument("--min-pixels", type=int, default=500)
    ap.add_argument("--tag", default="masked"); ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from torchmetrics import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    gdir = os.path.join(a.dataset, "images"); mdir = a.mask_dir or gdir
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).cuda()
    names = sorted(n for n in os.listdir(gdir) if n.lower().endswith((".png", ".jpg", ".jpeg"))
                   and not n.endswith(("_mask.png", "_covis.png")))
    se, npx, pv, L, S, cov, full, used = 0.0, 0, [], [], [], [], [], 0
    for n in names:
        stem = os.path.splitext(n)[0]
        pf, mf = os.path.join(a.pred_dir, stem + ".png"), os.path.join(mdir, stem + a.mask_suffix)
        if not (os.path.exists(pf) and os.path.exists(mf)):
            continue
        pr, gt = load(pf), load(os.path.join(gdir, n))
        mk = torch.from_numpy(np.asarray(Image.open(mf).convert("L"))).cuda()
        m = (mk == a.mask_value) if a.mask_value is not None else (mk > 127)
        e = ((pr - gt) ** 2).mean(-1)
        full.append(float(-10 * torch.log10(e.mean().clamp_min(1e-10)))); cov.append(float(m.float().mean())); used += 1
        k = int(m.sum())
        if k < a.min_pixels:
            continue
        se += float(e[m].sum()); npx += k; pv.append(float(-10 * torch.log10(e[m].mean().clamp_min(1e-10))))
        comp = torch.where(m[..., None], pr, gt).permute(2, 0, 1)[None]; g = gt.permute(2, 0, 1)[None]
        with torch.no_grad():
            L.append(float(lp(comp, g))); S.append(float(ssim_m(comp, g)))
    res = {"tag": a.tag, "pred_dir": a.pred_dir, "mask": f"{mdir}/*{a.mask_suffix}" + (f"=={a.mask_value}" if a.mask_value is not None else ">127"),
           "n": used, "n_masked": len(pv), "coverage": round(float(np.mean(cov)), 4) if cov else None,
           "psnr": round(float(np.mean(full)), 3) if full else None,
           "psnr_mask": round(float(-10 * np.log10(se / npx)), 3) if npx else None,
           "psnr_mask_per_view": round(float(np.mean(pv)), 3) if pv else None,
           "lpips_mask": round(float(np.mean(L)), 4) if L else None,
           "ssim_mask": round(float(np.mean(S)), 4) if S else None}
    print("MASKED_SCORE", json.dumps(res), flush=True)
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        json.dump(res, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
