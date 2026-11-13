#!/usr/bin/env python3
"""Viewer-matched training (scripts/train_viewer_gsplat.py, unchanged) + teacher pseudo-views at a lower, masked loss weight.

Self-distillation: a teacher splat (e.g. 3DGRUT EDGS + Improved-GS, accurate near the camera path) is rendered at
virtual cameras near the training path (scripts/make_pseudo_views.py: COLMAP text dataset, images = teacher renders,
images/<stem>_mask.png = pixels to supervise). This wrapper runs train_viewer_gsplat.main() with two hooks, so every
option, the recipe and the outputs stay exactly those of the trainer:
  load_views  the trainer's TRAINING set (subset "train" of --dataset) gets every view of --pseudo-dataset appended
              (held-out selection on the real frames is untouched: pseudo-views never shift the k % interval split);
  render      for a pseudo-view while gradients are on (training step): masked pixels are replaced by the target
              (zero L1, alpha 1 so the random background cannot leak in) and the gradient of the rest is scaled by
              --pseudo-weight (value unchanged: x w + x.detach() (1 - w)), so the loss term, the densification
              gradients and Improved-GS's statistics all see the pseudo-view at that weight. Evaluation renders
              (no grad) are untouched. EAS edge maps of pseudo-views come from the teacher render.
--pseudo-until f: pseudo-views are sampled only during the first f x --iters steps (the trainer's epoch permutation is
restricted to the real frames afterwards). Sharing the steps costs in-sample fidelity (CEAR all-frames 60k: -0.5 dB at the
training cameras with 22 % pseudo steps); the last stretch, at decayed learning rates, wins it back.
--pseudo-depth: pseudo-views carry a depth target (<stem>_depth.npy, ray distance) for the trainer's --depth-lambda
term (alpha-normalised expected depth, full weight); the real frames' own targets are dropped, so the on-path recipe is
unchanged. The pseudo dataset may also be LiDAR image-based renders (scripts/make_lidar_ibr_views.py) instead of teacher renders.
Pseudo-views are drawn uniformly with the real frames (share = N_pseudo / (N_pseudo + N_train)).
--tvg: the trainer module file to wrap (default: the repo's; the sbatch snapshots git HEAD so concurrent edits to
the trainer cannot change a running comparison). Everything after the wrapper's own flags goes to the trainer.

    PYTHONPATH=$CEAR_WS/gsplat-viewer-pc274 python scripts/train_distill_gsplat.py --pseudo-dataset PDS \
        [--pseudo-weight 0.3] [--tvg snapshot/train_viewer_gsplat.py] -- --dataset DS --out OUT --viewer pc274 ...
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--pseudo-dataset", required=True, help="make_pseudo_views.py dataset (teacher renders + _mask.png)")
    ap.add_argument("--pseudo-weight", type=float, default=0.3, help="gradient weight of pseudo-view pixels")
    ap.add_argument("--pseudo-limit", type=int, default=0, help="use at most this many pseudo-views (0 = all)")
    ap.add_argument("--pseudo-soft", action="store_true", help="_mask.png values are per-pixel weights (v / 255, x --pseudo-weight; "
                    "0 = unsupervised) instead of a > 127 threshold (make_lidar_ibr_views.py --single-weight)")
    ap.add_argument("--pseudo-until", type=float, default=1.0, help="fraction of --iters after which no pseudo-view is drawn "
                    "any more (the last stretch fits the real frames alone: in-sample / on-path fidelity back, see docstring)")
    ap.add_argument("--pseudo-depth", action="store_true", help="the trainer's --depth-lambda term at the PSEUDO-views only "
                    "(their images/<stem>_depth.npy, e.g. LiDAR depth of make_lidar_ibr_views.py); real frames get none")
    ap.add_argument("--tvg", default=os.path.join(HERE, "train_viewer_gsplat.py"), help="trainer module file to wrap")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.path.insert(0, HERE)                                    # trainer's own imports (refine_poses_gsplat, ...)
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.tvg)))
    spec = importlib.util.spec_from_file_location("train_viewer_gsplat", a.tvg)
    tvg = importlib.util.module_from_spec(spec); sys.modules["train_viewer_gsplat"] = tvg; spec.loader.exec_module(tvg)
    import numpy as np
    import torch
    from PIL import Image

    masks, stats = {}, {"pseudo_dataset": a.pseudo_dataset, "pseudo_weight": a.pseudo_weight, "tvg": os.path.abspath(a.tvg)}
    orig_load, orig_render = tvg.load_views, tvg.ViewerGS.render

    def load_views(dataset, interval, subset, dev, *args, **kw):
        out = orig_load(dataset, interval, subset, dev, *args, **kw)
        if subset != "train" or "n_pseudo" in stats:             # first training-set load only
            return out
        ps = orig_load(a.pseudo_dataset, 0, "all", dev, depth=a.pseudo_depth)
        if a.pseudo_depth:                                       # depth supervision where the camera never was, only
            out = [tuple(v[:6]) + (None,) + tuple(v[7:]) for v in out]
            stats["pseudo_depth_views"] = sum(v[6] is not None for v in ps)
        if a.pseudo_limit and len(ps) > a.pseudo_limit:
            ps = [ps[i] for i in np.linspace(0, len(ps) - 1, a.pseudo_limit).astype(int)]
        names = {v[0] for v in out}
        for v in ps:
            if v[0] in names:
                raise SystemExit(f"[distill] pseudo-view name {v[0]} collides with a real frame")
            f = os.path.join(a.pseudo_dataset, "images", os.path.splitext(v[0])[0] + "_mask.png")
            m = np.asarray(Image.open(f).convert("L")) if os.path.exists(f) else np.full((v[3], v[2]), 255, np.uint8)
            m = m.astype(np.float32) / 255 if a.pseudo_soft else (m > 127).astype(np.float32)   # per-pixel weight in [0, 1]
            masks[v[0]] = torch.from_numpy(m).to(dev)
        share = [float((m > 0).float().mean()) for m in masks.values()]
        stats.update(n_train_real=len(out), n_pseudo=len(ps), pseudo_step_share=round(len(ps) / max(1, len(out) + len(ps)), 4),
                     supervised_pixel_share=round(float(np.mean(share)), 4) if share else None)
        print(f"[distill] {len(out)} real training frames + {len(ps)} pseudo-views ({a.pseudo_dataset}), weight "
              f"{a.pseudo_weight}, supervised pixel share {stats['supervised_pixel_share']}, pseudo share of steps "
              f"{stats['pseudo_step_share']}", flush=True)
        return out + ps

    iters = int(rest[rest.index("--iters") + 1]) if "--iters" in rest else 30000
    cut = int(round(a.pseudo_until * iters)); stats["pseudo_until_step"] = cut; cnt = {"steps": 0}
    orig_perm = np.random.permutation

    def permutation(x, *args, **kw):                             # the trainer's epoch order: real frames only after the cut
        if isinstance(x, (int, np.integer)) and "n_pseudo" in stats and x == stats["n_train_real"] + stats["n_pseudo"] \
                and cnt["steps"] >= cut and stats["n_pseudo"] > 0:
            if "pseudo_off_at" not in stats:
                stats["pseudo_off_at"] = cnt["steps"]; print(f"[distill] step {cnt['steps']}: pseudo-views off (real frames only from here)", flush=True)
            return orig_perm(stats["n_train_real"])
        return orig_perm(x, *args, **kw)

    def render(self, p, view, deg, *args, **kw):
        out = orig_render(self, p, view, deg, *args, **kw)
        if torch.is_grad_enabled():
            cnt["steps"] += 1
        m = masks.get(view[0])
        if m is None or not torch.is_grad_enabled():
            return out
        img, alpha, d, info = out[0], out[1], out[2], out[3]
        w = a.pseudo_weight * m                                  # (H, W): 0 = unsupervised
        img = img * w[..., None] + img.detach() * (1 - w[..., None]); alpha = alpha * w + alpha.detach() * (1 - w)
        img = torch.where((m > 0)[..., None], img, view[5].float() / 255)
        alpha = torch.where(m > 0, alpha, torch.ones_like(alpha))
        return (img, alpha, d, info) + tuple(out[4:])

    tvg.load_views, tvg.ViewerGS.render = load_views, render
    if a.pseudo_until < 1.0:
        np.random.permutation = permutation
    sys.argv = [a.tvg] + rest
    tvg.main()
    o = rest[rest.index("--out") + 1] if "--out" in rest else None
    if o and os.path.isfile(os.path.join(o, "train.json")):      # record the distillation setup next to the trainer log
        j = json.load(open(os.path.join(o, "train.json"))); j["distill"] = stats
        pj = os.path.join(a.pseudo_dataset, "pseudo.json")
        if os.path.isfile(pj):
            pi = json.load(open(pj)); j["distill"]["pseudo"] = {k: v for k, v in pi.items() if k != "views"}
        json.dump(j, open(os.path.join(o, "train.json"), "w"), indent=2)
    print("DISTILL_TRAIN_DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
