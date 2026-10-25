#!/usr/bin/env python3
"""Multi-seed ensemble distillation into ONE standard 3DGS PLY (gsplat viewer venv, GPU; see scripts/methods/__init__.py).

Splats trained with different seeds (same recipe) disagree mostly where the data under-constrain them: between and off
the training views (floaters, view-dependent colour, needles), while all fit the training frames. The pixel-wise MEAN of
their renders is a better novel-view predictor than any one of them (ensemble), but a mean of renders is not a splat.
Here a student (initialised from --init, default the first teacher) is fine-tuned in the viewer-matched rasteriser of
scripts/train_viewer_gsplat.py on
  real steps   (1 - --pseudo-frac): a training frame vs its real image (L1 + 0.2 D-SSIM, random background), and
  pseudo steps (--pseudo-frac):     a jittered training camera (translation up to --jitter-t x the median spacing of
                                    consecutive training cameras, rotation up to --jitter-r deg) vs the mean of the
                                    teachers' renders there (each composited on the same random background),
with no densification (gaussian count fixed), all learning rates x --lr-mult with a cosine decay to 0.1x. Held-out
frames (--test-split-interval) are never used; --eval-dataset is scored at the end for the student, every teacher and the
teachers' render mean (the ensemble upper bound) at the viewer camera. Output: <out>/splat.ply, <out>/distill.json.

    PYTHONPATH=$CEAR_WS/gsplat-viewer-pc274 python scripts/methods/ensemble_distill.py --teachers A.ply B.ply C.ply \
        --dataset DS --test-split-interval 8 --eval-dataset TEST --out OUT [--viewer pc274] [--iters 15000]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_viewer_gsplat as tvg  # noqa: E402
from refine_poses_gsplat import ssim, psnr  # noqa: E402


def rot(axis_angle):
    import torch
    th = axis_angle.norm()
    if float(th) < 1e-12:
        return torch.eye(3, device=axis_angle.device)
    k = axis_angle / th; Kx = torch.zeros(3, 3, device=k.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=k.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--init", default=None, help="student initialisation PLY (default: the first teacher)")
    ap.add_argument("--dataset", required=True); ap.add_argument("--test-split-interval", type=int, default=8)
    ap.add_argument("--eval-dataset", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", choices=tuple(tvg.PROFILES), default="pc274")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--pseudo-frac", type=float, default=0.5)
    ap.add_argument("--jitter-t", type=float, default=3.0, help="max translation, x median consecutive camera spacing")
    ap.add_argument("--jitter-r", type=float, default=5.0, help="max rotation, degrees")
    ap.add_argument("--lr-mult", type=float, default=0.3)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import gsplat
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = "cuda"; t0 = time.time()
    pf = os.path.join(os.path.dirname(gsplat.__file__), "VIEWER_PROFILE")
    cut = open(pf).read().strip() if os.path.exists(pf) else "stock"
    if cut != args.viewer:
        raise SystemExit(f"[distill] --viewer {args.viewer} needs PYTHONPATH=$CEAR_WS/gsplat-viewer-{args.viewer}; loaded {cut}")
    rast = tvg.ViewerGS(argparse.Namespace(viewer=args.viewer, eps2d=None, cull_l1=None, aa=False, sh_band2=None,
                                           min_px=0.0, near=1e-4, cap_px=1024.0))
    tr = tvg.load_views(args.dataset, args.test_split_interval, "train", dev)
    ev = tvg.load_views(args.eval_dataset, 0, "all", dev) if args.eval_dataset else []
    T = [tvg.load_ply_params(f, dev) for f in args.teachers]
    deg = int(round(math.sqrt(T[0]["shN"].shape[1] + 1))) - 1
    p0 = tvg.load_ply_params(args.init or args.teachers[0], dev)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    C = torch.stack([torch.linalg.inv(v[4])[:3, 3] for v in tr]).cpu().numpy()
    spacing = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    scene_scale = float(np.linalg.norm(C - C.mean(0), axis=1).max()) * 1.1
    print(f"[distill] {len(T)} teachers, student {len(params['means'])} gaussians, {len(tr)} train / {len(ev)} eval views, "
          f"camera spacing {spacing:.4f}, jitter {args.jitter_t} x spacing / {args.jitter_r} deg", flush=True)
    m = args.lr_mult
    lr = {"means": 1.6e-5 * scene_scale * m, "scales": 5e-3 * m, "quats": 1e-3 * m, "opacities": 5e-2 * m,
          "sh0": 2.5e-3 * m, "shN": 2.5e-3 / 20 * m}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k]} for k in params], eps=1e-15)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / args.iters, 1))))

    def jitter(v):
        c = torch.linalg.inv(v[4])
        r = rot(torch.nn.functional.normalize(torch.randn(3, device=dev), dim=0) * math.radians(args.jitter_r) * float(torch.rand(())))
        t = torch.nn.functional.normalize(torch.randn(3, device=dev), dim=0) * args.jitter_t * spacing * float(torch.rand(()))
        c2 = c.clone(); c2[:3, :3] = c[:3, :3] @ r; c2[:3, 3] = c[:3, 3] + t
        return (v[0], v[1], v[2], v[3], torch.linalg.inv(c2), None, None)

    curve, nps = [], 0
    for step in range(args.iters):
        v = tr[np.random.randint(len(tr))]
        bg = torch.rand(3, device=dev)
        pseudo = np.random.rand() < args.pseudo_frac
        if pseudo:
            v = jitter(v); nps += 1
            with torch.no_grad():
                tgt = torch.stack([(lambda o: o[0] + (1 - o[1][..., None]) * bg)(rast.render(t, v, deg)) for t in T]).mean(0).clamp(0, 1)
        else:
            tgt = v[5].float() / 255
        img, alpha, _, _ = rast.render(params, v, deg)
        img = img + (1 - alpha[..., None]) * bg
        loss = (1 - args.ssim_lambda) * (img - tgt).abs().mean() + \
            args.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], tgt.permute(2, 0, 1)[None]))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if step % 1000 == 0 or step == args.iters - 1:
            e = {"step": step, "loss": round(loss.item(), 5), "pseudo": pseudo, "sec": round(time.time() - t0, 1)}
            curve.append(e); print(f"[distill] {e}", flush=True)
    res = {"args": vars(args), "n_gaussians": len(params["means"]), "pseudo_steps": nps, "curve": curve,
           "train_seconds": round(time.time() - t0, 1)}
    if ev:
        res["student"] = tvg.evaluate(rast, params, ev, deg, os.path.join(args.out, "renders_trainer"), args.viewer)
        res["teachers"] = [tvg.evaluate(rast, t, ev, deg, viewer=args.viewer) for t in T]
        ps, ss_ = [], []
        with torch.no_grad():                               # the ensemble upper bound: mean of the teachers' renders
            for v in ev:
                vv = (v[0], tvg.viewer_K(v[1], v[2], v[3], args.viewer)) + tuple(v[2:])
                img = torch.round(torch.stack([rast.render(t, vv, deg)[0] for t in T]).mean(0).clamp(0, 1) * 255) / 255
                gt = v[5].float() / 255
                ps.append(psnr(img, gt)); ss_.append(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        res["ensemble_mean"] = (round(float(np.mean(ps)), 3), round(float(np.mean(ss_)), 4))
        print(f"[distill] held-out (viewer camera, this rasteriser) PSNR/SSIM: student {res['student']}, teachers "
              f"{res['teachers']}, teachers' render mean {res['ensemble_mean']}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    tvg.save_ply(os.path.join(args.out, "splat.ply"), params, rast, [f"ensemble_distill teachers={len(T)} viewer={args.viewer}"])
    json.dump(res, open(os.path.join(args.out, "distill.json"), "w"), indent=2)
    print("DISTILL_DONE", flush=True)


if __name__ == "__main__":
    main()
