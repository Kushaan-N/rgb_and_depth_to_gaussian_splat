#!/usr/bin/env python3
"""Difix3D+-style progressive refinement of an exported splat .ply for SuperSplat (gsplat venv, GPU).

NVIDIA Difix (single-step SD-Turbo artifact remover, conditioned on a real reference image; nvidia/difix_ref) cleans
renders of the splat at virtual cameras near the training path; the cleaned renders are distilled back into the
splat with a viewer-exact fine-tune (scripts/finetune_ply_viewer.py's ViewerRaster), real frames dominating, and the
virtual cameras move further from the path every round (fix -> distill -> move further). Unlike a softer 3DGS teacher,
the fixer's targets are sharp; the risk is hallucinated texture, so each pseudo pixel is supervised only where
  - the render is opaque (alpha >= --min-alpha), and
  - the fixer's LOW-frequency change is small (|blur(fixed) - blur(render)| <= --max-lowfreq-change, Gaussian sigma
    --blur-px): Difix may sharpen edges / remove floaters' fringes, not repaint content or shift colour,
and pixels outside the mask take the (detached) render as target, so they contribute nothing.

Subcommands:
  refine  rounds over (offset m, rot deg[, per-pair]) of --rounds; per round: make_pseudo_views.py `plan` (between
          consecutive training frames; held-out frames of --test-split-interval stay held out), render the CURRENT
          splat there (viewer-exact, true intrinsics), Difix them in a subprocess (--difix-python running
          scripts/difix_images.py, ref = the nearer real frame of the view's training pair), masks, then
          --iters-per-round fine-tune steps (real frame each step; a pseudo view with probability --pseudo-prob, its
          loss x --pseudo-weight). --pseudo-prob 0 = the matching control (same schedule, real frames only).
  fixtest Difix3D+ post-render diagnostic: Difix renders of held-out views (refs = nearest training camera of the
          training dataset looking the same way) and, as the ceiling, the real images themselves -> two PNG dirs
          (score with eval_views.py --pred-dir)

    python scripts/diffix_refine.py refine --ply s.ply --dataset DS --out o.ply --work W --test-split-interval 8 \
        --rounds 0:0,0.1:5,0.2:10 --iters-per-round 2000 --difix-python $DIFIX_VENV/bin/python [finetune flags]
    python scripts/diffix_refine.py fixtest --renders shots/black --test TEST --train DS --test-split-interval 8 --work W ...
Difix code + weights: NVIDIA non-commercial research licence.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finetune_ply_viewer import (ViewerRaster, load_ply, save_ply, frames, cam_args, to_t, sh_deg, u8, evaluate,  # noqa: E402
                                 save_png)
from refine_poses_gsplat import ssim, load_rgb, psnr  # noqa: E402


def run_difix(a, in_dir, out_dir, refs):
    """Difix in its own venv (old diffusers pins) -> out_dir/<same names>."""
    rj = os.path.join(os.path.dirname(out_dir.rstrip("/")), os.path.basename(out_dir.rstrip("/")) + "_refs.json")
    json.dump(refs, open(rj, "w"), indent=0)
    cmd = [a.difix_python, os.path.join(HERE, "difix_images.py"), "--in-dir", in_dir, "--out-dir", out_dir, "--refs", rj,
           "--timestep", str(a.timestep)]
    t0 = time.time(); subprocess.run(cmd, check=True)
    print(f"[dfx] difix {len(refs)} views {time.time() - t0:.0f}s", flush=True)


def nearest_train(c2w, train):
    """Training frame (name, c2w) nearest in position among those looking the same way (else nearest overall)."""
    C = np.array([t[1][:3, 3] for t in train]); Z = np.array([t[1][:3, 2] for t in train])
    d = np.linalg.norm(C - c2w[:3, 3], axis=1); same = Z @ c2w[:3, 2] > 0.5
    return train[int(np.argmin(np.where(same, d, d + 1e6)))][0]


def plan_round(a, rnd, off, rot, per_pair, out):
    from make_pseudo_views import plan
    ns = argparse.Namespace(dataset=a.dataset, out=out, test_split_interval=a.test_split_interval, sim3_json=a.sim3_json,
                            per_pair=per_pair, tmin=0.2, tmax=0.8, offset_m=off, rot_deg=rot, min_frame_gap=a.min_frame_gap,
                            min_dist=a.min_dist, max_step=4.0, seed=a.seed + 101 * rnd)
    plan(ns)
    info = json.load(open(os.path.join(out, "pseudo.json")))
    refs = {}
    for v in info["views"]:                                        # the nearer real frame of the view's pair
        refs[v["name"]] = v["pair"][0] if v["t"] < 0.5 else v["pair"][1]
    return refs


def make_masks(a, render, fixed, alpha):
    """-> (H,W) bool mask, stats. render/fixed (H,W,3) float [0,1] torch, alpha (H,W)."""
    import torch, torch.nn.functional as F
    s = a.blur_px; k = int(2 * round(3 * s) + 1)
    g = torch.exp(-((torch.arange(k, device=render.device) - k // 2) ** 2) / (2 * s * s)); g = g / g.sum()
    blur = lambda x: F.conv2d(F.conv2d(x.permute(2, 0, 1)[None], g.view(1, 1, 1, k).expand(3, 1, 1, k), padding=(0, k // 2), groups=3),
                              g.view(1, 1, k, 1).expand(3, 1, k, 1), padding=(k // 2, 0), groups=3)[0].permute(1, 2, 0)
    low = (blur(fixed) - blur(render)).abs().amax(-1)
    m = (alpha >= a.min_alpha) & (low <= a.max_lowfreq_change)
    return m, {"opaque": float((alpha >= a.min_alpha).float().mean()), "kept": float(m.float().mean()),
               "hf_change": float(((fixed - render).abs().mean(-1))[m].mean()) if m.any() else 0.0}


def refine(a):
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    vertex, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); N = len(p0["means"])
    tr = frames(a.dataset, a.test_split_interval, "train")
    img_dir = os.path.join(a.dataset, "images")
    rgb = [load_rgb(os.path.join(img_dir, n), dev) for n, _, _ in tr]
    centres = np.stack([c[:3, 3] for _, _, c in tr])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    lr = {"means": a.means_lr * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": a.opacity_lr, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    lr = {k: v * a.lr_mult for k, v in lr.items()}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in str(a.display_scale).split(":")]; r_eval = a.eval_display_scale or rs[0]
    raster = ViewerRaster("ss352", "exact", display_scale=r_eval)
    ev = [(n, cam, c2w, load_rgb(os.path.join(a.eval_dataset, "images", n), dev).float() / 255)
          for n, cam, c2w in frames(a.eval_dataset, 0, "all")[::2]] if a.eval_dataset else []
    trs = [(n, cam, c2w, rgb[k].float() / 255) for k, (n, cam, c2w) in enumerate(tr)][::max(1, len(tr) // 30)]
    wdnet = None
    if a.lambda_wd > 0:
        from wasserstein_distortion import VGG16WassersteinDistortion
        wdnet = VGG16WassersteinDistortion().to(dev).eval()
        for q in wdnet.parameters():
            q.requires_grad_(False)
    rounds = [[float(x) for x in r.split(":")] for r in a.rounds.split(",")]
    total = len(rounds) * a.iters_per_round
    log = {"args": vars(a), "n_gaussians": N, "n_train": len(tr), "rounds": [], "curve": []}
    print(f"[dfx] {a.ply}: {N} gaussians; {len(tr)} training frames (interval {a.test_split_interval}); rounds {rounds}; "
          f"pseudo prob {a.pseudo_prob} weight {a.pseudo_weight}", flush=True)

    def report(step):
        p = {k: v.detach() for k, v in params.items()}
        raster.r = r_eval
        e = {"step": step, "train_psnr": round(evaluate(raster, p, trs, dev, deg), 3)}
        if ev:
            e["heldout_psnr"] = round(evaluate(raster, p, ev, dev, deg, viewer_cam="ss352"), 3)
        log["curve"].append(e); print(f"[dfx] eval {e}", flush=True)

    def loss_of(img, gt):
        l = (1 - a.ssim_lambda) * (img - gt).abs().mean() + a.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        if wdnet is not None:
            pc_, gc_ = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
            l = l + a.lambda_wd * wdnet(pc_, gc_, torch.full_like(gc_[:, :1], a.wd_log2_sigma), num_scales=3)
        return l

    report(0)
    step, t0 = 0, time.time()
    for rnd, rr in enumerate(rounds):
        off, rot = rr[0], rr[1]; per_pair = int(rr[2]) if len(rr) > 2 else a.per_pair
        pseudo = []
        if a.pseudo_prob > 0:
            W = os.path.join(a.work, f"round{rnd}"); PD = os.path.join(W, "views")
            refs = plan_round(a, rnd, off, rot, per_pair, PD)
            pv = frames(PD, 0, "all")
            raster.r = r_eval
            with torch.no_grad():                                 # the current splat at the virtual cameras
                for n, cam, c2w in pv:
                    img, al, _ = raster.render({k: v.detach() for k, v in params.items()}, to_t(np.linalg.inv(c2w), dev),
                                               *cam_args(cam), sh_degree=deg)
                    save_png(os.path.join(W, "render", n), img)
                    np.save(os.path.join(W, "render", os.path.splitext(n)[0] + "_alpha.npy"), al.half().cpu().numpy())
            run_difix(a, os.path.join(W, "render"), os.path.join(W, "fixed"), {n: os.path.join(img_dir, refs[n]) for n, _, _ in pv})
            st = []
            for n, cam, c2w in pv:
                stem = os.path.splitext(n)[0]
                ren = load_rgb(os.path.join(W, "render", n), dev).float() / 255
                fix = load_rgb(os.path.join(W, "fixed", n), dev).float() / 255
                al = torch.from_numpy(np.load(os.path.join(W, "render", stem + "_alpha.npy")).astype(np.float32)).to(dev)
                m, s = make_masks(a, ren, fix, al); st.append(s)
                save_png(os.path.join(W, "mask", n), m.float()[..., None].expand(-1, -1, 3))
                if s["kept"] >= a.min_valid:
                    pseudo.append((n, cam, c2w, fix.half(), m))
            rs_ = {"round": rnd, "offset_m": off, "rot_deg": rot, "per_pair": per_pair, "n_views": len(pv), "n_used": len(pseudo),
                   "opaque_mean": round(float(np.mean([s["opaque"] for s in st])), 4) if st else None,
                   "kept_mean": round(float(np.mean([s["kept"] for s in st])), 4) if st else None,
                   "hf_change_mean": round(float(np.mean([s["hf_change"] for s in st])), 5) if st else None}
            log["rounds"].append(rs_); print(f"[dfx] round {rs_}", flush=True)
        order = []
        for _ in range(a.iters_per_round):
            step += 1; frac = step / total
            wu = min(1.0, step / a.warmup) if a.warmup > 0 else 1.0
            for g in opt.param_groups:
                g["lr"] = lr[g["name"]] * wu * ((a.means_lr_final / a.means_lr) ** frac if g["name"] == "means" else 1.0)
            if len(rs) == 2:
                raster.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[1]))))
            is_p = bool(pseudo) and np.random.rand() < a.pseudo_prob
            if is_p:
                n, cam, c2w, fix, m = pseudo[np.random.randint(len(pseudo))]
            else:
                if not order:
                    order = list(np.random.permutation(len(tr)))
                k = int(order.pop()); n, cam, c2w = tr[k]
            img, _, _ = raster.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
            if is_p:                                             # unmasked pixels: target = own render (no gradient signal)
                gt = torch.where(m[..., None], fix.float(), img.detach())
                loss = a.pseudo_weight * loss_of(img, gt)
            else:
                gt = rgb[k].float() / 255; loss = loss_of(img, gt)
            opt.zero_grad(set_to_none=True); loss.backward()
            for prm in params.values():
                if prm.grad is not None:
                    prm.grad[~torch.isfinite(prm.grad)] = 0
            opt.step()
            if step % a.log_every == 0 or step == 1:
                print(f"[dfx] step {step:6d} {'P' if is_p else 'R'} loss {loss.item():.4f} psnr {psnr(img.detach(), gt):.2f} "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G ({time.time() - t0:.0f}s)", flush=True)
        report(step)
    save_ply(a.out, vertex, params)
    log["seconds"] = time.time() - t0
    json.dump(log, open(os.path.splitext(a.out)[0] + "_dfx.json", "w"), indent=2)
    print(f"[dfx] wrote {a.out}"); print("DFX_REFINE_DONE")


def fixtest(a):
    """Difix the held-out renders + the real held-out images (ceiling: what the fixer does to a perfect render)."""
    te = frames(a.test, 0, "all")
    tr = [(n, c) for n, _, c in frames(a.train, a.test_split_interval, "train")]
    refs = {}
    for n, _, c2w in te:
        refs[n] = os.path.join(a.train, "images", nearest_train(c2w, tr))
    os.makedirs(a.work, exist_ok=True)
    run_difix(a, a.renders, os.path.join(a.work, "fixed_render"), refs)
    run_difix(a, os.path.join(a.test, "images"), os.path.join(a.work, "fixed_gt"), refs)
    print("DFX_FIXTEST_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("refine", "fixtest"):
        s = sub.add_parser(nm)
        s.add_argument("--work", required=True)
        s.add_argument("--test-split-interval", type=int, default=0)
        s.add_argument("--difix-python", default=os.path.join(os.environ.get("DIFIX_VENV", ""), "bin", "python"))
        s.add_argument("--timestep", type=int, default=199)
        s.add_argument("--seed", type=int, default=0)
    f = sub.choices["fixtest"]
    f.add_argument("--renders", required=True); f.add_argument("--test", required=True); f.add_argument("--train", required=True)
    r = sub.choices["refine"]
    r.add_argument("--ply", required=True); r.add_argument("--dataset", required=True); r.add_argument("--out", required=True)
    r.add_argument("--sim3-json", default=None, help="metres -> dataset units for the view offsets")
    r.add_argument("--rounds", default="0:0,0.1:5,0.2:10", help="offset_m:rot_deg[:per_pair] per round, comma-separated")
    r.add_argument("--per-pair", type=int, default=1)
    r.add_argument("--min-frame-gap", type=float, default=2.0); r.add_argument("--min-dist", type=float, default=0.08)
    r.add_argument("--iters-per-round", type=int, default=2000)
    r.add_argument("--pseudo-prob", type=float, default=0.3, help="0 = control (real frames only)")
    r.add_argument("--pseudo-weight", type=float, default=0.3)
    r.add_argument("--min-alpha", type=float, default=0.95)
    r.add_argument("--max-lowfreq-change", type=float, default=0.04, help="max |blur(fixed)-blur(render)| per pixel (0-1)")
    r.add_argument("--blur-px", type=float, default=4.0)
    r.add_argument("--min-valid", type=float, default=0.3, help="drop views with a smaller supervised share")
    r.add_argument("--display-scale", default="1.5:4"); r.add_argument("--eval-display-scale", type=float, default=1.8)
    r.add_argument("--lr-mult", type=float, default=1.0)
    r.add_argument("--means-lr", type=float, default=1.6e-5); r.add_argument("--means-lr-final", type=float, default=1.6e-6)
    r.add_argument("--opacity-lr", type=float, default=0.025)
    r.add_argument("--ssim-lambda", type=float, default=0.2)
    r.add_argument("--lambda-wd", type=float, default=0.025); r.add_argument("--wd-log2-sigma", type=float, default=4.0)
    r.add_argument("--warmup", type=int, default=300)
    r.add_argument("--eval-dataset", default=None)
    r.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()
    {"refine": refine, "fixtest": fixtest}[a.cmd](a)


if __name__ == "__main__":
    main()
