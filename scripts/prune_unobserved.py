"""Remove the gaussians no training camera observed from a 3DGRUT checkpoint (3DGRUT venv; GPU if present).

A LiDAR-seeded splat puts gaussians everywhere the LiDAR saw, including places no camera looked; there
they keep the init's grey and whatever training smeared into them, and they are opaque. A gaussian is
OBSERVED if some training camera sees its centre in front of (or on) the measured surface:
  - the centre projects inside the training image, in front of the camera, and
  - its distance along the pixel ray is <= the camera's LiDAR depth target there x (1 + --tol-rel),
    or that pixel has no LiDAR target (unknown: kept, to stay conservative).
Unobserved gaussians are removed (every per-gaussian tensor filtered; zeroing their opacity instead
crashes 3DGUT's tracer); nothing observed changes. The output is for rendering, not for resuming
training (optimizer state is dropped).
Rendering the result leaves never-seen regions genuinely empty, which is the cue a generative filler
(ArtiFixer) uses to decide where to generate.

    python scripts/prune_unobserved.py --checkpoint ckpt.pt --train <dataset with *_depth.npy> --out pruned.pt
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covis_masks import cams_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train", required=True, help="training dataset (COLMAP text model + <image>_depth.npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-rel", type=float, default=0.05)
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    X = ck["positions"].detach().to(dev, torch.float64)
    seen = torch.zeros(len(X), dtype=torch.bool, device=dev)
    for (name, W, H, fx, fy, cx, cy, Rcw, tcw, dep) in cams_of(args.train):
        Pc = X @ torch.tensor(Rcw, device=dev).T + torch.tensor(tcw, device=dev)
        z = Pc[:, 2]
        u = torch.floor(fx * Pc[:, 0] / z + cx).long(); v = torch.floor(fy * Pc[:, 1] / z + cy).long()
        inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = torch.nonzero(inside, as_tuple=True)[0]
        if os.path.exists(dep):
            D = torch.tensor(np.load(dep).astype(np.float32), device=dev, dtype=torch.float64)
            d = D[v[idx], u[idx]]
            rng = torch.linalg.norm(Pc[idx], dim=1)
            ok = (d <= 0) | (rng <= d * (1 + args.tol_rel))                   # no target: unknown -> keep
            seen[idx[ok]] = True
        else:
            seen[idx] = True
    n_un = int((~seen).sum()); keep = seen.cpu(); N = len(X)
    for k, v in list(ck.items()):
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
            f = v.detach()[keep].clone()
            ck[k] = torch.nn.Parameter(f, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else f
    ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    info = {"gaussians": int(len(X)), "unobserved_removed": n_un, "fraction": round(n_un / len(X), 4),
            "tol_rel": args.tol_rel, "checkpoint": args.checkpoint}
    json.dump(info, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print(f"[prune] {n_un:,} of {len(X):,} gaussians ({n_un / len(X):.1%}) never observed by a training camera "
          f"-> removed; {args.out}", flush=True)


if __name__ == "__main__":
    main()
