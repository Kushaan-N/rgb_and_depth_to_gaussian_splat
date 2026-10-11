"""Fine-tune ONLY the added gaussians of an edited splat so it renders like the base splat wherever the base
is trusted (3DGRUT venv, GPU). Soft counterpart of harm_filter.py; run harm_filter.py afterwards for the
hard, measured guarantee.

Loss over the same poses and rules as harm_filter.py (training views + jittered copies: strict; render-only
--extra-poses: changes that put a surface clearly in front of the base surface are allowed):
    L = mean |render(edited) - render(base)| over harmful-rule pixels
Trainable: density (opacity) and SH DC colour of the added gaussians (the last --added rows); with
--move-normal also an offset along each added disc's normal (|offset| <= --max-offset-m). Real gaussians
get no update (asserted bit-identical at the end). Added gaussians that change trusted pixels fade or blend
into the base appearance; those no pose shows (the unseen backs) get no gradient and keep their values.
Added gaussians whose opacity falls below --min-opacity are pruned (never set to zero: the tracer dislikes
zero-opacity particles). Only poses whose render differs from the base are optimised (re-checked per epoch).

Writes <out>, <out stem>_finetune.json (loss / harm-PSNR history, pruned count) and <out stem>_kept.npy (bool per
input added gaussian).

    python scripts/preserve_finetune.py --base base.pt --edited completed.pt --added 15907 --dataset train/ --out tuned.pt \
        [--extra-poses heldout/ xviews/] [--jitter 1 --sim3-json sim3.json] [--move-normal]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harm_filter import Gauss, add_pose_args, build_views, harm_map, load_model, psnr_of, render, subset_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--added", type=int, required=True, help="number of added gaussians (the last entries)")
    ap.add_argument("--out", required=True)
    add_pose_args(ap)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-density", type=float, default=0.2)
    ap.add_argument("--lr-color", type=float, default=0.05, help="SH DC units (rgb = 0.282 * dc + 0.5)")
    ap.add_argument("--move-normal", action="store_true")
    ap.add_argument("--max-offset-m", type=float, default=0.01)
    ap.add_argument("--lr-offset-m", type=float, default=0.001)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    s = float(json.load(open(args.sim3_json))["scale"]) if args.sim3_json else 1.0
    views = build_views(args)
    bm, _ = load_model(args.base); gb = Gauss(bm)
    base = []
    for v in views:
        o = render(gb, v); base.append({"rgb": o["rgb"].half(), "depth": o["depth"].half()})
    del gb, bm; torch.cuda.empty_cache()
    em, ck = load_model(args.edited)
    N = em.positions.shape[0]; n0 = N - args.added; A = args.added
    pos_r, dens_r, alb_r = em.positions.detach()[:n0], em.density.detach()[:n0], em.features_albedo.detach()[:n0]
    pos_a = em.positions.detach()[n0:]
    dens_a = em.density.detach()[n0:].clone().requires_grad_(True)
    alb_a = em.features_albedo.detach()[n0:].clone().requires_grad_(True)
    assert 0 < A < N, "need >= 1 added gaussian"
    off = torch.zeros(A, 1, device="cuda", dtype=pos_a.dtype, requires_grad=args.move_normal)
    # disc normal = axis of the smallest scale
    q = torch.nn.functional.normalize(em.rotation.detach()[n0:], dim=1); w, x, y, z = q.unbind(1)
    Rm = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)
    nrm = Rm[torch.arange(A), :, em.scale.detach()[n0:].argmin(1)]
    groups = [{"params": [dens_a], "lr": args.lr_density}, {"params": [alb_a], "lr": args.lr_color}]
    if args.move_normal:
        groups.append({"params": [off], "lr": args.lr_offset_m / s})
    opt = torch.optim.Adam(groups, eps=1e-15)

    def current():
        return Gauss(em, positions=torch.cat([pos_r, pos_a + off * nrm]), density=torch.cat([dens_r, dens_a]),
                     albedo=torch.cat([alb_r, alb_a]))

    def survey():
        """no-grad pass: per-view harm PSNR and whether the view differs from the base at all."""
        with torch.no_grad():
            g = current(); ps, act = [], []
            for vi, v in enumerate(views):
                d, hm, se = harm_map(render(g, v), base[vi], v["rule"], args.thr, args.fill_margin_rel)
                ps.append(psnr_of(se.mean().item())); act.append(bool(hm.any()) or se.max().item() > args.thr ** 2)
        return ps, act

    hist = []
    for ep in range(args.epochs + 1):
        ps, act = survey()
        idx = [i for i, a in enumerate(act) if a]
        rec = {"epoch": ep, "active_views": len(idx), "min_psnr": round(min(ps), 2), "median_psnr": round(float(np.median(ps)), 2),
               "mean_opacity_added": round(torch.sigmoid(dens_a).mean().item(), 4)}
        if ep == args.epochs or not idx:
            hist.append(rec); print(f"[finetune] epoch {ep}: {rec}", flush=True)
            break
        tot = 0.0
        for vi in rng.permutation(idx):
            v = views[vi]
            o = render(current(), v, grad=True)
            diff = (o["rgb"].clamp(0, 1) - base[vi]["rgb"].float()).abs().mean(-1)
            if v["rule"] == "fill":
                Db, Dc = base[vi]["depth"].float(), o["depth"].float()
                diff = torch.where((Dc > 0) & ((Db <= 0) | (Dc < Db * (1 - args.fill_margin_rel))), torch.zeros_like(diff), diff)
            loss = diff.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
            with torch.no_grad():
                off.clamp_(-args.max_offset_m / s, args.max_offset_m / s)
        rec["loss"] = tot / len(idx); hist.append(rec)
        print(f"[finetune] epoch {ep}: {rec}", flush=True)
    op = torch.sigmoid(dens_a.detach()).squeeze(1)
    keep_a = (op >= args.min_opacity).cpu().numpy()
    out = dict(ck)
    with torch.no_grad():
        for k_, val in (("positions", torch.cat([pos_r, pos_a + off * nrm])), ("density", torch.cat([dens_r, dens_a])),
                        ("features_albedo", torch.cat([alb_r, alb_a]))):
            v0 = ck[k_]; t = val.detach().to(v0.dtype)
            out[k_] = torch.nn.Parameter(t, requires_grad=v0.requires_grad) if isinstance(v0, torch.nn.Parameter) else t
    keep = np.concatenate([np.ones(n0, bool), keep_a])
    out = subset_ckpt(out, keep)
    for k_ in ("positions", "density", "features_albedo", "rotation", "scale", "features_specular"):
        assert torch.equal(out[k_][:n0].detach().cuda(), ck[k_][:n0].detach().cuda()), k_
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    stem = os.path.splitext(args.out)[0]
    np.save(stem + "_kept.npy", keep_a)
    info = {"added_in": A, "added_kept": int(keep_a.sum()), "pruned_low_opacity": int((~keep_a).sum()),
            "faded_below_0.5": int((op < 0.5).sum()), "history": hist,
            "params": {k: v for k, v in vars(args).items() if k not in ("base", "edited", "out")}}
    json.dump(info, open(stem + "_finetune.json", "w"), indent=2)
    print(f"[finetune] kept {keep_a.sum():,} of {A:,} added gaussians (pruned {(~keep_a).sum():,} below opacity "
          f"{args.min_opacity}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
