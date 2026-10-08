"""Score masks for completed object surfaces in views from another recording (3DGRUT venv; GPU if present).

Projects the unseen-surface samples of complete_objects.py (<out>_unseen.npz, dataset frame) into each
view of a dataset (with <image>_depth.npy LiDAR targets and <image>_covis.png from covis_masks.py),
keeping samples that face the view and are not occluded, and rewrites <image>_covis.png as
  128 = pixel shows an object surface the source recording never saw (the completed backs)
  255 = pixel the source recording did observe (regression check)
    0 = anything else
so eval_views.py reports psnr_unobserved on the completed backs and psnr_observed on seen content.
Frames where the backs cover at least --min-frac of the image are copied to <out> as a dataset.

    python scripts/object_back_masks.py --views <xviews> --unseen completed_unseen.npz --out <xviews_backs>
"""
from __future__ import annotations
import argparse, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model
from fuse_sequences import write_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frac", type=float, default=0.003)
    ap.add_argument("--max-incidence", type=float, default=80.0)
    ap.add_argument("--tol-rel", type=float, default=0.05)
    ap.add_argument("--cell-units", type=float, default=None, help="sample spacing in dataset units (splat radius)")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U = np.load(args.unseen)["unseen"]
    X = torch.tensor(U[:, :3], device=dev); Nn = torch.tensor(U[:, 3:], device=dev)
    cams, imgs = read_text_model(os.path.join(args.views, "sparse", "0"))
    cosmax = np.cos(np.deg2rad(args.max_incidence)); kept = []
    for name, cid, q, tcw in sorted(imgs):
        W, H, (fx, fy, cx, cy) = cams[cid]
        stem = os.path.splitext(os.path.join(args.views, "images", name))[0]
        Rc = torch.tensor(quat_to_R(q), device=dev); tc = torch.tensor(tcw, device=dev)
        Pc = X @ Rc.T + tc; z = Pc[:, 2]
        u = torch.floor(fx * Pc[:, 0] / z + cx).long(); v = torch.floor(fy * Pc[:, 1] / z + cy).long()
        Cw = -Rc.T @ tc; vd = Cw[None] - X; vd = vd / torch.linalg.norm(vd, dim=1, keepdim=True)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & ((vd * Nn).sum(1) > cosmax)
        idx = torch.nonzero(ok, as_tuple=True)[0]
        back = torch.zeros(H, W, device=dev)
        if len(idx) and os.path.exists(stem + "_depth.npy"):
            D = torch.tensor(np.load(stem + "_depth.npy").astype(np.float32), device=dev).double()
            dt = D[v[idx], u[idx]]; r = torch.linalg.norm(Pc[idx], dim=1)
            vis = (dt > 0) & ((r - dt).abs() < args.tol_rel * dt)
            back[v[idx[vis]], u[idx[vis]]] = 1
            back = F.max_pool2d(back[None, None], 5, 1, 2)[0, 0]          # close gaps between samples
        cv = np.asarray(Image.open(stem + "_covis.png")) if os.path.exists(stem + "_covis.png") else np.zeros((H, W), np.uint8)
        m = np.where(back.cpu().numpy() > 0, 128, np.where(cv == 255, 255, 0)).astype(np.uint8)
        frac = float((m == 128).mean())
        if frac >= args.min_frac:
            kept.append((name, cid, q, tcw, m, frac))
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(os.path.join(args.out, "images"))
    sel = []
    for name, cid, q, tcw, m, frac in kept:
        Rcw = quat_to_R(q); sel.append((name, cid, Rcw.T, -Rcw.T @ tcw))
        src = os.path.join(args.views, "images", name)
        os.symlink(os.path.realpath(src), os.path.join(args.out, "images", name))
        Image.fromarray(m, mode="L").save(os.path.join(args.out, "images", os.path.splitext(name)[0] + "_covis.png"))
    write_model(os.path.join(args.out, "sparse", "0"), {c: ("PINHOLE", W, H, p) for c, (W, H, p) in cams.items()}, sel)
    fr = [k[5] for k in kept]
    print(f"[backs] {len(kept)} of {len(imgs)} views show completed surfaces (>= {args.min_frac:.1%} of the image; "
          f"median {np.median(fr) if fr else 0:.1%}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
