"""Render a trained 3DGRUT checkpoint at every camera of a COLMAP-format dataset and score it against
that dataset's images (3DGRUT venv, GPU). The dataset's poses must be in the checkpoint's frame.
`--save-dir` also writes each render as <save-dir>/<image name>; `--no-metrics` renders poses whose
images are only placeholders (novel views). Where <image>_depth.npy (LiDAR target, build_depth_targets.py)
exists, the rendered depth is scored too (AbsRel, share within 5%); where <image>_covis.png
(covis_masks.py) exists, PSNR is also split into pixels a training camera observed / never observed.

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
    ext = {"depth_absrel": [], "depth_within5": [], "sq_obs": [0.0, 0], "sq_unobs": [0.0, 0]}
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
        stem = os.path.splitext(dataset.image_paths[k])[0]
        if os.path.exists(stem + "_depth.npy"):
            dgt = torch.from_numpy(np.load(stem + "_depth.npy").astype(np.float32)).to(pred.device)
            dpr = (out["pred_dist"] / out["pred_opacity"].clamp_min(1e-3))[0, ..., 0]
            vm = dgt > 0
            if vm.any():
                rel = (torch.abs(dpr[vm] - dgt[vm]) / dgt[vm])
                ext["depth_absrel"].append(rel.median().item()); ext["depth_within5"].append((rel < 0.05).float().mean().item())
        if os.path.exists(stem + "_covis.png"):
            from PIL import Image
            cm = torch.from_numpy(np.asarray(Image.open(stem + "_covis.png"))).to(pred.device)
            se = ((pred - gt) ** 2).mean(-1)[0]
            for key, val in (("sq_obs", 255), ("sq_unobs", 128)):
                sel = cm == val
                ext[key][0] += se[sel].sum().item(); ext[key][1] += int(sel.sum().item())
        P.append(psnr_m(pred, gt).item())
        pchw = pred.permute(0, 3, 1, 2); gchw = gt.permute(0, 3, 1, 2)
        S.append(ssim_m(pchw, gchw).item())
        L.append(lpips_m(pchw, gchw).item())
    if not metrics:
        return {"n": len(names)}
    res = {"psnr": round(float(np.mean(P)), 3), "ssim": round(float(np.mean(S)), 4),
           "lpips": round(float(np.mean(L)), 4), "n": len(P)}
    if ext["depth_absrel"]:
        res["depth_absrel"] = round(float(np.mean(ext["depth_absrel"])), 4)
        res["depth_within5pct"] = round(float(np.mean(ext["depth_within5"])), 4)
    for key, tag in (("sq_obs", "psnr_observed"), ("sq_unobs", "psnr_unobserved")):
        if ext[key][1]:
            res[tag] = round(float(-10 * np.log10(ext[key][0] / ext[key][1])), 3)
    return res


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
    names = sorted(n for n in os.listdir(gdir) if n.lower().endswith((".png", ".jpg", ".jpeg"))
                   and not n.endswith(("_mask.png", "_covis.png")))
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
