"""Learned 3D object prior (TRELLIS venv, GPU): free image-to-3D generation of every free-standing object from its best
real training views, for object_priors.py build --learned-dir (which aligns, colour-calibrates and keeps only the
unseen parts).

Unlike scripts/trellis_complete.py (stage 2 only, on our 2.5D LiDAR structure: the voxel structure was ours, so the
generated geometry could not be better than the height map), here TRELLIS's own stage 1 (sparse structure flow,
image-conditioned) decides the shape, i.e. the model's object prior completes the backs. Model choice (2026-11):
microsoft/TRELLIS-image-large (MIT, weights already cached, runs in the existing venv-trellis on A100/A40/L40S);
SAM 3D Objects (gated HF weights, SAM licence), TRELLIS.2 (new CUDA extensions to compile) and Hunyuan3D 2.1 (mesh +
paint stack) need new environments - drop-in alternatives behind the same npz interface.

  generate  per object: conditioning views = trellis_complete.pick_views (visible surface area x unoccluded share,
            spread in azimuth), masked square crops (black background); TRELLIS run_multi_image (stage 1 + 2,
            --mode stochastic) for each --seeds; gaussians in TRELLIS's unit cube (+z up) -> <out>/obj<id>_s<seed>.npz
            (xyz, scale, rot wxyz, opacity, dc) + the conditioning crops (<out>/obj<id>_cond.jpg).

    python scripts/object_priors/learned_prior.py generate --objects objects --dataset <train ds> \
        --splat-depth-dir D --sim3-json sim3.json --out-dir GEN       # venv-trellis
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def generate(a):
    import torch
    from PIL import Image
    from trellis_complete import load_common, Vis, surface, pick_views, crop, obj_axes
    os.environ.setdefault("ATTN_BACKEND", "xformers"); os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    meta, pts, n, d, s, R, t = load_common(a)
    to_ds = lambda X: ((X - t) @ R) / s
    vis = Vis(a, s)
    sys.path.insert(0, a.trellis_dir)
    from trellis.pipelines import TrellisImageTo3DPipeline
    model = a.model
    if not os.path.exists(os.path.join(model, "pipeline.json")):
        from huggingface_hub import snapshot_download
        model = snapshot_download(model)
    pipe = TrellisImageTo3DPipeline.from_pretrained(model); pipe.cuda()
    os.makedirs(a.out_dir, exist_ok=True); log = []
    for o in meta["objects"]:
        S = surface(o, pts, n, d, a, to_ds, R, vis)
        c0, A = obj_axes(S["P"], n, d)
        chosen, ncand = pick_views(S, vis, a, to_ds(c0[None])[0], R.T @ A)
        if not chosen:
            print(f"[learned] object {o['id']}: no usable view", flush=True); continue
        ims = [crop(np.asarray(Image.open(os.path.join(a.dataset, "images", vis.cams[ci][0])).convert("RGB")), m)
               for _, ci, m, _ in chosen]
        Image.fromarray(np.hstack([np.asarray(x.resize((259, 259))) for x in ims])).save(os.path.join(a.out_dir, f"obj{o['id']}_cond.jpg"))
        for seed in a.seeds:
            with torch.no_grad():
                out = pipe.run_multi_image(ims, seed=seed, formats=["gaussian"], preprocess_image=False, mode=a.mode)
            g = out["gaussian"][0]
            np.savez_compressed(os.path.join(a.out_dir, f"obj{o['id']}_s{seed}.npz"), xyz=g.get_xyz.float().cpu().numpy(),
                                scale=g.get_scaling.float().cpu().numpy(),
                                rot=torch.nn.functional.normalize(g.get_rotation.float(), dim=1).cpu().numpy(),
                                opacity=g.get_opacity.float().cpu().numpy()[:, 0], dc=g._features_dc.float().cpu().numpy()[:, 0],
                                id=o["id"], seed=seed)
            log.append({"id": o["id"], "seed": seed, "gaussians": int(g.get_xyz.shape[0]), "views": len(chosen), "candidates": ncand,
                        "images": [vis.cams[ci][0] for _, ci, _, _ in chosen]})
            print(f"[learned] object {o['id']} seed {seed}: {g.get_xyz.shape[0]:,} gaussians from {len(chosen)}/{ncand} views", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "generate.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--objects", required=True); g.add_argument("--sim3-json", required=True); g.add_argument("--dataset", required=True)
    g.add_argument("--splat-depth-dir", default=None); g.add_argument("--out-dir", required=True)
    g.add_argument("--trellis-dir", default=os.path.join(os.environ.get("CEAR_WS", ""), "TRELLIS"))
    g.add_argument("--model", default="microsoft/TRELLIS-image-large")
    g.add_argument("--views", type=int, default=4); g.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # trellis_complete.py's visibility / view-selection parameters
    g.add_argument("--frame-stride", type=int, default=1); g.add_argument("--cell", type=float, default=0.01)
    g.add_argument("--hcell", type=float, default=0.02); g.add_argument("--max-incidence", type=float, default=85.0)
    g.add_argument("--depth-tol-m", type=float, default=0.03); g.add_argument("--splat-tol-rel", type=float, default=0.05)
    g.add_argument("--support-m", type=float, default=0.03); g.add_argument("--max-trunc", type=float, default=0.05)
    g.add_argument("--min-samples", type=int, default=200); g.add_argument("--min-crop-px", type=int, default=40)
    g.add_argument("--mask-tol-rel", type=float, default=0.05)
    a = ap.parse_args()
    {"generate": generate}[a.cmd](a)


if __name__ == "__main__":
    main()
