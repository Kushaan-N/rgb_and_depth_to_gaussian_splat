"""Gate for depth-supervised training: does 3DGRUT's rendered depth (pred_dist) carry usable gradients?
(3DGRUT venv, GPU.) 3DGRUT's own source marks the depth backward as unfinished, so check before training.

Renders a few training frames of a trained checkpoint, computes the same depth loss as the patched
trainer (L1 between the opacity-normalised rendered ray distance and the LiDAR target), backpropagates,
and passes only if (1) gradients reach the gaussian parameters, finite and non-zero, and (2) a small
gradient step on positions/densities lowers the depth loss. Exit code 0 = pass, 1 = fail.

    python scripts/check_depth_grad.py --checkpoint ckpt.pt --dataset <dir with *_depth.npy>
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    import torch
    from threedgrut.render import Renderer
    import threedgrut.datasets as datasets

    r = Renderer.from_checkpoint(checkpoint_path=args.checkpoint, path=args.dataset, out_dir=args.dataset,
                                 save_gt=False, computes_extra_metrics=False)
    model, conf = r.model, r.conf
    conf.dataset.test_split_interval = 0
    ds, _ = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad or True}
    for p in params.values():
        p.requires_grad_(True)

    def depth_loss(idx):
        tot = 0.0
        for i in idx:
            item = ds[i]; batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v])) for k, v in item.items()}
            gb = ds.get_gpu_batch_with_intrinsics(batch)
            stem = os.path.splitext(ds.image_paths[i])[0]
            d = torch.from_numpy(np.load(stem + "_depth.npy").astype(np.float32)).cuda()[None, ..., None]
            out = model(gb)
            dp = out["pred_dist"] / out["pred_opacity"].clamp_min(1e-3)
            m = d > 0
            tot = tot + torch.abs(dp - d)[m].mean()
        return tot / len(idx)

    idx = list(np.linspace(0, len(ds) - 1, args.frames).astype(int))
    model.zero_grad(set_to_none=True)
    L0 = depth_loss(idx); L0.backward()
    grads = {n: p.grad for n, p in params.items() if p.grad is not None}
    norms = {n: float(g.norm()) for n, g in grads.items()}
    print("[gradcheck] depth loss", round(float(L0), 5), "| grad norms:", {k: f"{v:.3e}" for k, v in norms.items()})
    ok_grad = any(v > 0 for v in norms.values()) and all(np.isfinite(v) for v in norms.values())
    ok_step = False
    if ok_grad:
        with torch.no_grad():
            saved = {n: p.detach().clone() for n, p in params.items() if n in grads}
            for eps in (1e-1, 1e-2, 1e-3, 1e-4):
                for n, p in params.items():
                    if n in grads and norms[n] > 0:
                        p.copy_(saved[n] - eps * grads[n] / norms[n] * saved[n].abs().mean().clamp_min(1e-6))
                if hasattr(model, "build_acc"):
                    model.build_acc()
                L1 = float(depth_loss(idx))
                print(f"[gradcheck] step {eps:g}: depth loss {float(L0):.5f} -> {L1:.5f}")
                if L1 < float(L0):
                    ok_step = True; break
            for n, p in params.items():
                if n in saved:
                    p.copy_(saved[n])
    print("GRADCHECK", "PASS" if ok_grad and ok_step else "FAIL", f"(gradients {'ok' if ok_grad else 'missing'}, "
          f"descent {'ok' if ok_step else 'not observed'})", flush=True)
    return 0 if ok_grad and ok_step else 1


if __name__ == "__main__":
    sys.exit(main())
