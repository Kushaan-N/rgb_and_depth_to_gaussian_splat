#!/usr/bin/env python3
"""Real held-out images resampled into the camera that scripts/viewer_shots.py actually renders (CPU).

viewer_shots renders a canvas CENTRED on the principal point and crops W x H at an INTEGER offset, with ONE focal length
(--engine current: SuperSplat's horizontal fov from fx when the canvas is wider than tall, else fy; pc274: fy). The
viewer camera therefore has principal point (CW/2 - x0, CH/2 - y0) and square pixels, not the real (cx, cy, fx, fy):
on CEAR (cy = 249.29 -> 249.5, fy -> fx) every viewer render is misregistered by ~0.2 px vertically against its real
frame, which costs ~1.4 dB PSNR on EVERY splat alike (sharp splats more than blurry ones) -- a property of the
measurement, not of the splat (in interactive SuperSplat there is no "real" camera to miss).
This writes <out>/<image> = the real image remapped (Lanczos-4, reflect border) so pixel (j, i) shows what the real camera
saw along the viewer camera's ray: u = (j + .5 - cxv) fx / f + cx, v = (i + .5 - cyv) fy / f + cy. _covis.png masks are
remapped with nearest. Score viewer renders against it with `eval_views.py --pred-dir R --gt-dir <out>` ("registered"
metric; report next to the raw one, which stays the gate's metric unless the user decides otherwise).

    python scripts/register_gt.py --dataset $P/onpath/test --out $P/onpath/test_viewreg_current [--engine current]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses_gsplat import read_model, intrinsics  # noqa: E402


def viewer_camera(K, W, H, engine):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    CW, CH = int(np.ceil(2 * max(cx, W - cx))), int(np.ceil(2 * max(cy, H - cy)))
    x0, y0 = int(round(CW / 2 - cx)), int(round(CH / 2 - cy))
    f = (fx if CW > CH else fy) if engine == "current" else fy
    return f, CW / 2 - x0, CH / 2 - y0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=("current", "pc274"), default="current")
    args = ap.parse_args()
    import cv2
    cams, imgs = read_model(os.path.join(args.dataset, "sparse", "0"))
    os.makedirs(args.out, exist_ok=True); rec, maps = {}, {}
    for name, _iid, cid, _c2w in imgs:
        K, W, H = intrinsics(cams[cid]); W, H = int(W), int(H)
        if cid not in maps:
            f, cxv, cyv = viewer_camera(K, W, H, args.engine)
            jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            mx = ((jj - cxv) * K[0, 0] / f + K[0, 2] - 0.5).astype(np.float32)
            my = ((ii - cyv) * K[1, 1] / f + K[1, 2] - 0.5).astype(np.float32)
            maps[cid] = (mx, my)
            rec[cid] = {"real": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "viewer_f_cx_cy": [f, cxv, cyv],
                        "max_offset_px": round(float(np.hypot(mx - jj + 0.5, my - ii + 0.5).max()), 4),
                        "centre_offset_px": [round(K[0, 2] - cxv, 4), round(K[1, 2] - cyv, 4)]}
        mx, my = maps[cid]; stem = os.path.splitext(name)[0]
        src = os.path.join(args.dataset, "images", name)
        im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(args.out, stem + ".png"), cv2.remap(im, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT))
        cov = os.path.join(args.dataset, "images", stem + "_covis.png")
        if os.path.exists(cov):
            cv2.imwrite(os.path.join(args.out, stem + "_covis.png"),
                        cv2.remap(cv2.imread(cov, cv2.IMREAD_UNCHANGED), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE))
    json.dump({"engine": args.engine, "dataset": args.dataset, "n": len(imgs), "cameras": {str(k): v for k, v in rec.items()}},
              open(os.path.join(args.out, "_register.json"), "w"), indent=1)
    print(f"[register] {len(imgs)} images -> {args.out}: {json.dumps(rec)}")


if __name__ == "__main__":
    main()
