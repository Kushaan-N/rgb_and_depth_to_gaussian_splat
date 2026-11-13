"""Complete generated 3D assets of every free-standing object, for asset_replace.py (no per-object or per-scene constants).

  crops    (venv-mv, GPU) per object of find_objects.py (--ids only as a debug override): the best real training views
           (trellis_complete.pick_views: visible area x unoccluded share, spread in azimuth), square crops, object cut out
           with BiRefNet (salient object within the geometric mask's box; geometric mask when saliency misses it), black
           background, 518 px -> <out>/crops/obj<id>_<k>.png + crops.json (views, mask sizes).
  generate (venv-trellis, GPU) TRELLIS image-to-3D, multi-image conditioned on those crops (stage 1 + 2), --seeds ->
           <out>/obj<id>_s<seed>.npz (xyz, scale, rot wxyz, opacity, dc in the unit cube, +z up: learned_prior.py's format)
           + <out>/obj<id>_cond.jpg. Objects with no usable view are skipped (asset_replace.py then leaves them alone).

    python scripts/object_priors/mv_asset_gen.py crops --objects D/objects --sim3-json sim3.json --dataset <train ds> \
        --splat-depth-dir D/depth --out-dir G                                   # venv-mv
    python scripts/object_priors/mv_asset_gen.py generate --out-dir G           # venv-trellis
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def crops(a):
    import cv2
    from PIL import Image
    from transformers import AutoModelForImageSegmentation
    from trellis_complete import load_common, Vis, surface, pick_views, obj_axes
    from mv_gen import birefnet, square_box, cut, largest_cc_touching
    dev = "cuda"
    meta, pts, n, d, s, R, t = load_common(a); to_ds = lambda X: ((X - t) @ R) / s; vis = Vis(a, s)
    bir = AutoModelForImageSegmentation.from_pretrained(a.birefnet, trust_remote_code=True).to(dev).half().eval()
    C = os.path.join(a.out_dir, "crops"); os.makedirs(C, exist_ok=True); log = []
    for o in meta["objects"]:
        if a.ids and o["id"] not in a.ids:
            continue
        S = surface(o, pts, n, d, a, to_ds, R, vis); c0, A = obj_axes(S["P"], n, d)
        chosen, ncand = pick_views(S, vis, a, to_ds(c0[None])[0], R.T @ A)
        if not chosen:
            print(f"[asset-gen] object {o['id']}: no usable view", flush=True); continue
        names, sheet = [], []
        for k, (sc_, ci, m, az) in enumerate(chosen):
            name = vis.cams[ci][0]; img = np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"))
            x0, y0, Sz = square_box(m, a.pad); ci_ = cut(img, x0, y0, Sz); mi = cut(m.astype(np.uint8), x0, y0, Sz) > 0
            ys, xs = np.nonzero(mi); e_ = int(a.prior_box * max(np.ptp(xs), np.ptp(ys)))
            prior = np.zeros_like(mi); prior[max(ys.min() - e_, 0):ys.max() + e_ + 1, max(xs.min() - e_, 0):xs.max() + e_ + 1] = True
            mb = largest_cc_touching((birefnet(bir, ci_, dev) > 0.5) & prior, mi)
            if mb.sum() < 0.3 * mi.sum():
                kk = max(3, int(0.06 * Sz)) | 1; mb = cv2.morphologyEx(mi.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8)) > 0
            x1, y1, S2 = square_box(mb, 1.2)                               # TRELLIS: object fills 1/1.2 of the square
            out = Image.fromarray(cut(ci_ * mb[..., None], x1, y1, S2)).resize((518, 518), Image.Resampling.LANCZOS)
            out.save(os.path.join(C, f"obj{o['id']}_{k}.png")); names.append(name); sheet.append(np.asarray(out.resize((259, 259))))
        Image.fromarray(np.hstack(sheet)).save(os.path.join(a.out_dir, f"obj{o['id']}_cond.jpg"))
        log.append({"id": o["id"], "views": names, "candidates": ncand})
        print(f"[asset-gen] object {o['id']}: {len(names)} crops of {ncand} candidate views", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "crops.json"), "w"), indent=1)


def generate(a):
    import torch
    from PIL import Image
    os.environ.setdefault("ATTN_BACKEND", "xformers"); os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers"); os.environ.setdefault("SPCONV_ALGO", "native")
    sys.path.insert(0, a.trellis_dir)
    from trellis.pipelines import TrellisImageTo3DPipeline
    model = a.model
    if not os.path.exists(os.path.join(model, "pipeline.json")):
        from huggingface_hub import snapshot_download
        model = snapshot_download(model)
    pipe = TrellisImageTo3DPipeline.from_pretrained(model); pipe.cuda(); log = []
    for c in json.load(open(os.path.join(a.out_dir, "crops.json"))):
        ims = [Image.open(f).convert("RGB") for f in sorted(glob.glob(os.path.join(a.out_dir, "crops", f"obj{c['id']}_*.png")))]
        for seed in a.seeds:
            if os.path.exists(os.path.join(a.out_dir, f"obj{c['id']}_s{seed}.npz")):
                log.append({"id": c["id"], "seed": seed, "reused": True}); continue
            try:
                with torch.no_grad():
                    out = (pipe.run_multi_image(ims, seed=seed, formats=["gaussian"], preprocess_image=False, mode=a.mode) if len(ims) > 1 else
                           pipe.run(ims[0], seed=seed, formats=["gaussian"], preprocess_image=False))
            except RuntimeError as e:                                       # e.g. an empty sparse structure for this seed
                print(f"[asset-gen] object {c['id']} seed {seed}: generation failed ({str(e)[:80]}), skipped", flush=True)
                log.append({"id": c["id"], "seed": seed, "failed": str(e)[:200]})
                del pipe; torch.cuda.empty_cache()                          # a failed multi-image run leaves the sampler patched: reload
                pipe = TrellisImageTo3DPipeline.from_pretrained(model); pipe.cuda(); continue
            g = out["gaussian"][0]
            np.savez_compressed(os.path.join(a.out_dir, f"obj{c['id']}_s{seed}.npz"), xyz=g.get_xyz.float().cpu().numpy(),
                                scale=g.get_scaling.float().cpu().numpy(), rot=torch.nn.functional.normalize(g.get_rotation.float(), dim=1).cpu().numpy(),
                                opacity=g.get_opacity.float().cpu().numpy()[:, 0], dc=g._features_dc.float().cpu().numpy()[:, 0], id=c["id"], seed=seed)
            log.append({"id": c["id"], "seed": seed, "gaussians": int(g.get_xyz.shape[0]), "views": len(ims)})
            print(f"[asset-gen] object {c['id']} seed {seed}: {g.get_xyz.shape[0]:,} gaussians from {len(ims)} crops", flush=True)
    json.dump(log, open(os.path.join(a.out_dir, "generate.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--objects", required=True); c.add_argument("--sim3-json", required=True); c.add_argument("--dataset", required=True)
    c.add_argument("--splat-depth-dir", default=None); c.add_argument("--out-dir", required=True)
    c.add_argument("--ids", type=int, nargs="*", default=None, help="debug override: only these objects")
    c.add_argument("--birefnet", default="ZhengPeng7/BiRefNet"); c.add_argument("--pad", type=float, default=1.6); c.add_argument("--prior-box", type=float, default=0.15)
    c.add_argument("--views", type=int, default=4); c.add_argument("--frame-stride", type=int, default=1)
    c.add_argument("--cell", type=float, default=0.01); c.add_argument("--hcell", type=float, default=0.02)
    c.add_argument("--max-incidence", type=float, default=85.0); c.add_argument("--depth-tol-m", type=float, default=0.03)
    c.add_argument("--splat-tol-rel", type=float, default=0.05); c.add_argument("--support-m", type=float, default=0.03)
    c.add_argument("--max-trunc", type=float, default=0.05); c.add_argument("--min-samples", type=int, default=200)
    c.add_argument("--min-crop-px", type=int, default=40); c.add_argument("--mask-tol-rel", type=float, default=0.05)
    g = sub.add_parser("generate")
    g.add_argument("--out-dir", required=True); g.add_argument("--trellis-dir", default=os.path.join(os.environ.get("CEAR_WS", ""), "TRELLIS"))
    g.add_argument("--model", default="microsoft/TRELLIS-image-large"); g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    a = ap.parse_args(); {"crops": crops, "generate": generate}[a.cmd](a)


if __name__ == "__main__":
    main()
