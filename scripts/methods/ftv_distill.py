#!/usr/bin/env python3
"""Ensemble distillation INSIDE the viewer-exact fine-tune (gsplat venv, GPU; see scripts/methods/__init__.py).

Measured on CEAR (held-out, live SuperSplat): after the deliverable's viewer-exact fine-tune (finetune_ply_viewer.py,
SuperSplat profile, WD) three seeds of the same recipe score 36.39 / 36.18 / 36.30 dB (Depth), while the pixel MEAN of
their three live renders scores 36.58 -- the seeds disagree where the data under-constrain them, and the mean is the
better novel-view prediction. Distilling that mean in train_viewer_gsplat.py's rasteriser (ensemble_distill.py) gained
+0.19 raw but only +0.03 once the fine-tune ran after it (most of the raw gain was trainer-vs-viewer mismatch). Here
the distillation IS the fine-tune: finetune_ply_viewer.ViewerRaster (the 52-dB-agreement copy of SuperSplat's renderer,
random on-screen scale in --display-scale, Depth / Radial order drawn per step, identical for teachers and student),
and per step either
  real   (1 - --pseudo-frac): a training frame vs its real image: (1-s) L1 + s D-SSIM + --lambda-wd x WD (as the fine-tune)
  pseudo (--pseudo-frac):     a camera on the path between consecutive training frames (lerp / slerp, u ~ U(0,1)) jittered
                              by up to --jitter-t x the median spacing / --jitter-r deg, vs the mean of the teachers' renders
                              there ((1-s) L1 + s D-SSIM; no WD -- the mean is slightly soft, WD would chase its blur)
from --init (default: the first teacher), no densification, LRs of the fine-tune with log-linear means decay. Held-out
frames (--test-split-interval) are never used; --eval-dataset is reported (viewer camera, matched renderer) for the
student and each teacher at the start and end. Output: --out PLY (layout of --init) + <out>_ftvd.json.

    PYTHONPATH=... python scripts/methods/ftv_distill.py --teachers a_ft.ply b_ft.ply c_ft.ply --dataset DS \
        --test-split-interval 8 --eval-dataset TEST --out OUT.ply [--iters 3000] [--pseudo-frac 0.5]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from finetune_ply_viewer import ViewerRaster, load_ply, save_ply, frames, cam_args, to_t, sh_deg, evaluate  # noqa: E402
from refine_poses_gsplat import ssim, load_rgb  # noqa: E402
from vwarp import _exp_rot, _log_rot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teachers", nargs="+", required=True); ap.add_argument("--init", default=None)
    ap.add_argument("--dataset", required=True); ap.add_argument("--test-split-interval", type=int, default=8)
    ap.add_argument("--eval-dataset", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", default="ss352", choices=("pc274", "ss352"))
    ap.add_argument("--display-scale", default="1.5:4"); ap.add_argument("--eval-display-scale", type=float, default=1.5)
    ap.add_argument("--sort", choices=("depth", "radial", "dual"), default="dual")
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--pseudo-frac", type=float, default=0.5)
    ap.add_argument("--jitter-t", type=float, default=1.0); ap.add_argument("--jitter-r", type=float, default=2.0)
    ap.add_argument("--lr-mult", type=float, default=1.0); ap.add_argument("--means-lr", type=float, default=1.6e-5)
    ap.add_argument("--means-lr-final", type=float, default=1.6e-6); ap.add_argument("--opacity-lr", type=float, default=0.025)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--lambda-wd", type=float, default=0.025); ap.add_argument("--wd-log2-sigma", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = "cuda"; t0 = time.time()
    vertex, p0 = load_ply(args.init or args.teachers[0], dev); deg = sh_deg(p0)
    T = [{k: v.detach() for k, v in load_ply(f, dev)[1].items()} for f in args.teachers]
    tr = frames(args.dataset, args.test_split_interval, "train")
    rgb = [load_rgb(os.path.join(args.dataset, "images", n), dev) for n, _, _ in tr]
    C = np.stack([c[:3, 3] for _, _, c in tr]); gaps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    spacing = float(np.median(gaps)); scene_scale = float(np.linalg.norm(C - C.mean(0), axis=1).max()) * 1.1
    pairs = [k for k in range(len(tr) - 1) if gaps[k] <= 3 * spacing]
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    m = args.lr_mult
    lr = {"means": args.means_lr * scene_scale * m, "scales": 5e-3 * m, "quats": 1e-3 * m, "opacities": args.opacity_lr * m,
          "sh0": 2.5e-3 * m, "shN": 2.5e-3 / 20 * m}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in args.display_scale.split(":")]
    raster = ViewerRaster(args.viewer, "exact", display_scale=args.eval_display_scale, sort="depth")
    wdnet = None
    if args.lambda_wd > 0:
        from wasserstein_distortion import VGG16WassersteinDistortion
        wdnet = VGG16WassersteinDistortion().to(dev).eval()
        for q in wdnet.parameters():
            q.requires_grad_(False)
    ev = [(n, cam, c2w, load_rgb(os.path.join(args.eval_dataset, "images", n), dev).float() / 255)
          for n, cam, c2w in frames(args.eval_dataset, 0, "all")] if args.eval_dataset else []
    print(f"[ftvd] {len(T)} teachers, student {len(p0['means'])} gaussians, {len(tr)} train frames ({len(pairs)} pairs, spacing "
          f"{spacing:.4f}), pseudo {args.pseudo_frac}, jitter {args.jitter_t} x spacing / {args.jitter_r} deg, sort {args.sort}", flush=True)

    def held(p):
        res = {}
        for o in (("depth", "radial") if args.sort == "dual" else (args.sort,)):
            raster.sort = o; raster.r = args.eval_display_scale
            res[o] = round(evaluate(raster, p, ev, dev, deg, viewer_cam=args.viewer), 3)
        return res
    log = {"args": vars(args), "curve": []}
    if ev:
        log["teachers_heldout"] = [held(t) for t in T]
        print(f"[ftvd] held-out (matched renderer, viewer camera) teachers: {log['teachers_heldout']}", flush=True)

    def pseudo_cam():
        k = pairs[np.random.randint(len(pairs))]; ca, cb = tr[k][2], tr[k + 1][2]; u = float(np.random.rand())
        Ra, Rb = torch.tensor(ca[:3, :3], dtype=torch.float32), torch.tensor(cb[:3, :3], dtype=torch.float32)
        R = Ra @ _exp_rot(u * _log_rot(Ra.T @ Rb))
        ax = torch.nn.functional.normalize(torch.randn(3), dim=0) * math.radians(args.jitter_r) * float(torch.rand(()))
        R = R @ _exp_rot(ax)
        d = np.random.randn(3)
        t = (1 - u) * ca[:3, 3] + u * cb[:3, 3] + d / np.linalg.norm(d) * args.jitter_t * spacing * np.random.rand()
        c2w = np.eye(4); c2w[:3, :3] = R.numpy(); c2w[:3, 3] = t
        return tr[k][1], c2w

    nps = 0
    for step in range(1, args.iters + 1):
        wu = min(1.0, step / args.warmup) if args.warmup > 0 else 1.0
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu
        next(g for g in opt.param_groups if g["name"] == "means")["lr"] = \
            lr["means"] * (args.means_lr_final / args.means_lr) ** (step / args.iters) * wu
        if len(rs) == 2:
            raster.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[1]))))
        raster.sort = args.sort if args.sort != "dual" else ("radial" if np.random.rand() < 0.5 else "depth")
        pseudo = np.random.rand() < args.pseudo_frac
        if pseudo:
            cam, c2w = pseudo_cam(); nps += 1
        else:
            k = np.random.randint(len(tr)); _, cam, c2w = tr[k]
        w2c = to_t(np.linalg.inv(c2w), dev); ca_ = cam_args(cam)
        if pseudo:
            with torch.no_grad():
                gt = torch.stack([raster.render(t, w2c, *ca_, sh_degree=deg)[0] for t in T]).mean(0)
        else:
            gt = rgb[k].float() / 255
        img = raster.render(params, w2c, *ca_, sh_degree=deg)[0]
        loss = (1 - args.ssim_lambda) * (img - gt).abs().mean() + \
            args.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        if wdnet is not None and not pseudo:
            pc_, gc_ = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
            loss = loss + args.lambda_wd * wdnet(pc_, gc_, torch.full_like(gc_[:, :1], args.wd_log2_sigma), num_scales=3)
        opt.zero_grad(set_to_none=True); loss.backward()
        for prm in params.values():
            if prm.grad is not None:
                prm.grad[~torch.isfinite(prm.grad)] = 0
        opt.step()
        if step % args.log_every == 0 or step == 1:
            e = {"step": step, "loss": round(loss.item(), 5), "pseudo": bool(pseudo), "sec": round(time.time() - t0, 1)}
            log["curve"].append(e); print(f"[ftvd] {e}", flush=True)
    pf = {k: v.detach() for k, v in params.items()}
    if ev:
        log["student_heldout_init"] = held({k: v.detach() for k, v in p0.items()})
        log["student_heldout"] = held(pf)
        print(f"[ftvd] held-out student: init {log['student_heldout_init']} -> {log['student_heldout']}", flush=True)
    save_ply(args.out, vertex, params)
    log.update(pseudo_steps=nps, seconds=round(time.time() - t0, 1))
    json.dump(log, open(os.path.splitext(args.out)[0] + "_ftvd.json", "w"), indent=2)
    print(f"[ftvd] wrote {args.out}\nFTVD_DONE", flush=True)


if __name__ == "__main__":
    main()
