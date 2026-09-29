"""Render a trained 3DGRUT checkpoint at every camera of a COLMAP-format dataset and score it against
that dataset's images (3DGRUT venv, GPU). The dataset's poses must be in the checkpoint's frame.

    python scripts/eval_views.py --checkpoint ckpt.pt --dataset <dir with sparse/0 + images> \
        [--tag name] [--json-out result.json]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def score_dataset(checkpoint, ds):
    """{psnr, ssim, lpips, n} over all frames of `ds` (no train/test split)."""
    import torch
    from threedgrut.render import Renderer
    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    r = Renderer.from_checkpoint(checkpoint_path=checkpoint, path=ds, out_dir=ds,
                                 save_gt=False, computes_extra_metrics=False)
    model, conf = r.model, r.conf
    conf.dataset.test_split_interval = 0            # use every frame
    dataset, _ = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(dataset, **configure_dataloader_for_platform(
        {"num_workers": 4, "batch_size": 1, "shuffle": False, "collate_fn": None}))
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    P, S, L = [], [], []
    for batch in loader:
        gb = dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            out = model(gb)
        pred = out["pred_features"][..., :3].clip(0, 1)      # (1,H,W,3)
        gt = gb.rgb_gt[..., :3]
        P.append(psnr_m(pred, gt).item())
        pchw = pred.permute(0, 3, 1, 2); gchw = gt.permute(0, 3, 1, 2)
        S.append(ssim_m(pchw, gchw).item())
        L.append(lpips_m(pchw, gchw).item())
    return {"psnr": round(float(np.mean(P)), 3), "ssim": round(float(np.mean(S)), 4),
            "lpips": round(float(np.mean(L)), 4), "n": len(P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    res = {"tag": args.tag, "checkpoint": args.checkpoint, **score_dataset(args.checkpoint, args.dataset)}
    print("EVAL_VIEWS", json.dumps(res), flush=True)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(res, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
