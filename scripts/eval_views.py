"""Render a trained 3DGRUT checkpoint at every camera of a COLMAP-format dataset and score it against
that dataset's images (3DGRUT venv, GPU). The dataset's poses must be in the checkpoint's frame.
`--save-dir` also writes each render as <save-dir>/<image name>; `--no-metrics` renders poses whose
images are only placeholders (novel views).

    python scripts/eval_views.py --checkpoint ckpt.pt --dataset <dir with sparse/0 + images> \
        [--tag name] [--json-out result.json] [--save-dir renders/] [--no-metrics]
    python scripts/eval_views.py --pred-dir fixed/ --dataset <dir> ...   # score saved images instead
    python scripts/eval_views.py --pred-dir pred/ --gt-dir gt/ ...       # ... against a folder of real images
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def score_dataset(checkpoint, ds, save_dir=None, metrics=True):
    """{psnr, ssim, lpips, n} over all frames of `ds` (no train/test split); optionally save renders."""
    import torch, torchvision
    from threedgrut.render import Renderer
    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform

    r = Renderer.from_checkpoint(checkpoint_path=checkpoint, path=ds, out_dir=ds,
                                 save_gt=False, computes_extra_metrics=False)
    model, conf = r.model, r.conf
    conf.dataset.test_split_interval = 0            # use every frame
    dataset, _ = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(dataset, **configure_dataloader_for_platform(
        {"num_workers": 4, "batch_size": 1, "shuffle": False, "collate_fn": None}))
    psnr_m, ssim_m, lpips_m = _metrics()
    P, S, L = [], [], []
    names = [os.path.relpath(p, os.path.join(ds, "images")) for p in dataset.image_paths]
    for k, batch in enumerate(loader):
        gb = dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            out = model(gb)
        pred = out["pred_features"][..., :3].clip(0, 1)      # (1,H,W,3)
        if save_dir:
            dst = os.path.join(save_dir, os.path.splitext(names[k])[0] + ".png")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torchvision.utils.save_image(pred[0].permute(2, 0, 1), dst)
        if not metrics:
            continue
        gt = gb.rgb_gt[..., :3]
        P.append(psnr_m(pred, gt).item())
        pchw = pred.permute(0, 3, 1, 2); gchw = gt.permute(0, 3, 1, 2)
        S.append(ssim_m(pchw, gchw).item())
        L.append(lpips_m(pchw, gchw).item())
    if not metrics:
        return {"n": len(names)}
    return {"psnr": round(float(np.mean(P)), 3), "ssim": round(float(np.mean(S)), 4),
            "lpips": round(float(np.mean(L)), 4), "n": len(P)}


def _metrics():
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    return (PeakSignalNoiseRatio(data_range=1.0).cuda(), StructuralSimilarityIndexMeasure(data_range=1.0).cuda(),
            LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda())


def score_images(pred_dir, ds=None, gt_dir=None):
    """Same metrics as score_dataset, for images already on disk: <pred_dir>/<name> vs <ds>/images/<name>
    (or vs <gt_dir>/<name>)."""
    import torch
    from PIL import Image
    psnr_m, ssim_m, lpips_m = _metrics()
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0) \
        .permute(2, 0, 1)[None].cuda()
    gdir = gt_dir or os.path.join(ds, "images")
    names = sorted(os.listdir(gdir))
    P, S, L = [], [], []
    for n in names:
        pr, gt = load(os.path.join(pred_dir, os.path.splitext(n)[0] + ".png")), load(os.path.join(gdir, n))
        P.append(psnr_m(pr, gt).item()); S.append(ssim_m(pr, gt).item()); L.append(lpips_m(pr, gt).item())
    return {"psnr": round(float(np.mean(P)), 3), "ssim": round(float(np.mean(S)), 4),
            "lpips": round(float(np.mean(L)), 4), "n": len(P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pred-dir", default=None, help="score these saved images instead of rendering")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--gt-dir", default=None, help="with --pred-dir: real images to score against")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()
    if args.pred_dir:
        res = {"tag": args.tag, "pred_dir": args.pred_dir, **score_images(args.pred_dir, args.dataset, args.gt_dir)}
    else:
        res = {"tag": args.tag, "checkpoint": args.checkpoint,
               **score_dataset(args.checkpoint, args.dataset, args.save_dir, not args.no_metrics)}
    print("EVAL_VIEWS", json.dumps(res), flush=True)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(res, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
