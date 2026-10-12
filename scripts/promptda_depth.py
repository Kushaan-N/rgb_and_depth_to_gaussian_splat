"""Dense metric depth targets from Prompt Depth Anything (venv-promptda: env/make_promptda_venv.sh; GPU).

PromptDA (Lin et al., CVPR 2025) predicts full-resolution metric depth from an RGB image and a low-resolution depth
"prompt" (it was trained with iPhone-LiDAR prompts at 192x256). Here the prompt is the projected LiDAR target of
the frame (<image>_depth.npy, ray distance in dataset units, build_depth_targets.py): converted to z-depth in metres
(Sim3 scale), pixels the reliability mask drops (<image>_depthmask.png, depth_reliability.py; --no-mask to ignore)
and pixels without a return filled from the nearest trusted pixel, then area-downsampled to --prompt-short rows.
The prediction (z, metres) goes back to ray distance in dataset units. With --align (default) each frame is rescaled
by the median ratio LiDAR / prediction over trusted pixels (PromptDA is metric already; this only removes a residual
per-frame scale, reported as `scale`). Writes <out-dataset>/images/<stem>_depth.npy (float16, dense; never through a
symlink) and <out-dataset>/promptda.json with AbsRel vs LiDAR on trusted / dropped pixels.

    python scripts/promptda_depth.py --dataset <ds with _depth.npy [+ _depthmask.png]> --sim3-json sim3.json \
        --out-dataset <link copy of ds> [--model depth-anything/prompt-depth-anything-vitl-hf] [--viz-dir viz/]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import read_text_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dataset", required=True)
    ap.add_argument("--sim3-json", required=True, help="metres per dataset unit (scale)")
    ap.add_argument("--model", default="depth-anything/prompt-depth-anything-vitl-hf")
    ap.add_argument("--prompt-short", type=int, default=192)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--viz-dir", default=None); ap.add_argument("--viz-n", type=int, default=6)
    args = ap.parse_args()
    import cv2, torch
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = float(json.load(open(args.sim3_json))["scale"])
    proc = PromptDepthAnythingImageProcessor.from_pretrained(args.model)
    model = PromptDepthAnythingForDepthEstimation.from_pretrained(args.model).to(dev).eval()
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    imgs = sorted(imgs)
    t0 = time.time(); stats = []
    viz = set(np.linspace(0, len(imgs) - 1, args.viz_n).astype(int).tolist()) if args.viz_dir else set()
    for k, (name, cid, *_r) in enumerate(imgs):
        W, H, (fx, fy, cx, cy) = cams[cid]
        st = os.path.splitext(name)[0]
        D = np.load(os.path.join(args.dataset, "images", st + "_depth.npy")).astype(np.float32)
        mp = os.path.join(args.dataset, "images", st + "_depthmask.png")
        trust = D > 0
        if not args.no_mask and os.path.isfile(mp):
            trust &= np.asarray(Image.open(mp).convert("L")) > 127
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        ray = np.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1).astype(np.float32)
        z = D / ray * s                                                         # metres, z-depth
        if trust.sum() < 100:
            print(f"[pda] {name}: too few trusted LiDAR pixels, skipped"); continue
        _, (iy, ix) = distance_transform_edt(~trust, return_indices=True)        # nearest trusted pixel
        zf = z[iy, ix]
        hp = args.prompt_short; wp = int(round(hp * W / H))
        prompt = cv2.resize(zf, (wp, hp), interpolation=cv2.INTER_AREA)
        img = Image.open(os.path.join(args.dataset, "images", name)).convert("RGB")
        Hm, Wm = 14 * math.ceil(H / 14), 14 * math.ceil(W / 14)                  # native resolution, ViT patch 14
        inp = proc(images=img, prompt_depth=prompt * 1000.0, return_tensors="pt",    # mm (prompt_scale_to_meter 1e-3)
                   size={"height": Hm, "width": Wm}, keep_aspect_ratio=False, ensure_multiple_of=14).to(dev)
        with torch.no_grad():
            pred = proc.post_process_depth_estimation(model(**inp), target_sizes=[(H, W)])[0]["predicted_depth"]
        zp = pred.float().cpu().numpy().reshape(H, W)
        ratio = float(np.median(z[trust] / np.maximum(zp[trust], 1e-6)))
        sc = 1.0 if args.no_align else ratio
        Dp = (zp * sc * ray / s).astype(np.float32)                              # ray distance, dataset units
        Dp[~np.isfinite(Dp) | (Dp <= 0)] = 0
        rel = lambda m: float(np.median(np.abs(Dp[m] - D[m]) / D[m])) if m.any() else None
        stats.append({"name": name, "scale": round(ratio, 4), "absrel_trusted": rel(trust), "absrel_dropped": rel((D > 0) & ~trust),
                      "in_hm": [Hm, Wm], "shape_in": list(inp["pixel_values"].shape)})
        dst = os.path.join(args.out_dataset, "images", st + "_depth.npy")
        if os.path.islink(dst): os.remove(dst)
        np.save(dst, Dp.astype(np.float16))
        if k in viz:
            os.makedirs(args.viz_dir, exist_ok=True)
            hi = np.percentile(D[D > 0], 98) if (D > 0).any() else 1
            col = lambda d: cv2.applyColorMap((np.clip(d / hi, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * (d > 0)[..., None]
            pv = cv2.resize(prompt, (W, H), interpolation=cv2.INTER_NEAREST) * ray / s
            err = np.where(D > 0, np.clip(np.abs(Dp - D) / np.maximum(D, 1e-6) / 0.1, 0, 1), 0)
            tiles = [np.asarray(img)[..., ::-1].copy(), col(D), col(pv), col(Dp), cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)]
            for tl, lab in zip(tiles, ("image", "LiDAR target", "prompt", "PromptDA", "|err| vs LiDAR (0-10%)")):
                cv2.putText(tl, lab, (6, 22), 0, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(args.viz_dir, st + "_promptda.jpg"), np.hstack(tiles))
        if k % 50 == 0:
            print(f"[pda] {k + 1}/{len(imgs)} {name}: scale {ratio:.3f} absrel trusted {stats[-1]['absrel_trusted']:.4f} "
                  f"dropped {stats[-1]['absrel_dropped']}  in {tuple(inp['pixel_values'].shape)} ({time.time() - t0:.0f}s)", flush=True)
    agg = lambda key: round(float(np.median([x[key] for x in stats if x[key] is not None])), 4) if stats else None
    info = {"frames": len(stats), "model": args.model, "align": not args.no_align, "mask": not args.no_mask,
            "scale_median": agg("scale"), "scale_p5_p95": [round(float(np.percentile([x["scale"] for x in stats], q)), 4) for q in (5, 95)],
            "absrel_trusted_median": agg("absrel_trusted"), "absrel_dropped_median": agg("absrel_dropped"), "per_frame": stats}
    json.dump(info, open(os.path.join(args.out_dataset, "promptda.json"), "w"), indent=2)
    print("[pda]", json.dumps({k: v for k, v in info.items() if k != "per_frame"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
