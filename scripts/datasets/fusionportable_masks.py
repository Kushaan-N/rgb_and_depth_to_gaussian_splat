"""Transient-object masks (people, by default) for a COLMAP text dataset, and person-excluded scoring of viewer renders.

FusionPortable room sequences are "dynamic": the operator walks in front of the legged robot and office workers move.
A static splat correctly does NOT contain them, so a held-out frame showing a person scores low for a reason that is
not reconstruction quality. This marks them and scores the rest:

  mask      torchvision Mask R-CNN (COCO, weights cached in $TORCH_HOME) on every image of --dataset: instances of
            --classes (COCO ids; 1 = person) with score > --thr, dilated by --dilate px -> <stem>_mask.png next to
            the image (255 = transient). Skips images that already have one.
  composite for scoring: <pred-dir>/<stem>.png with masked pixels replaced by the real image -> --out-dir, so
            eval_views.py --pred-dir <out-dir> gives PSNR/SSIM/LPIPS on the static scene only (the mask fraction is
            printed: the share of pixels excluded).

    python scripts/datasets/fusionportable_masks.py mask --dataset <P>/onpath/test           # GPU venv with torchvision
    python scripts/datasets/fusionportable_masks.py composite --dataset <P>/onpath/test --pred-dir R --out-dir R_static
"""
from __future__ import annotations
import argparse, os
import numpy as np


def images(ds):
    d = os.path.join(ds, "images")
    return d, sorted(n for n in os.listdir(d) if n.endswith((".png", ".jpg")) and not n.endswith(("_mask.png", "_covis.png")))


def cmd_mask(a):
    import cv2, torch, torchvision
    d, names = images(a.dataset)
    w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w).eval().cuda()
    k = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8); frac = []
    for n in names:
        out = os.path.join(d, os.path.splitext(n)[0] + "_mask.png")
        if os.path.exists(out) and not a.force:
            continue
        im = cv2.imread(os.path.join(d, n))[:, :, ::-1].copy()
        with torch.no_grad():
            r = m([torch.from_numpy(im).permute(2, 0, 1).float().cuda() / 255])[0]
        sel = (r["scores"] > a.thr) & torch.isin(r["labels"], torch.tensor(a.classes, device=r["labels"].device))
        mk = (r["masks"][sel, 0] > 0.5).any(0).cpu().numpy() if sel.any() else np.zeros(im.shape[:2], bool)
        mk = cv2.dilate(mk.astype(np.uint8), k) > 0
        cv2.imwrite(out, mk.astype(np.uint8) * 255); frac.append(mk.mean())
    if frac:
        f = np.array(frac)
        print(f"[masks] {len(f)} images: masked fraction mean {f.mean():.4f}, images with a transient {np.mean(f > 0):.2%}")


def cmd_composite(a):
    import cv2
    d, names = images(a.dataset); os.makedirs(a.out_dir, exist_ok=True); fr = []
    for n in names:
        s = os.path.splitext(n)[0]; pr = cv2.imread(os.path.join(a.pred_dir, s + ".png")); gt = cv2.imread(os.path.join(d, n))
        mp = os.path.join(d, s + "_mask.png")
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127 if os.path.exists(mp) else np.zeros(gt.shape[:2], bool)
        pr[mk] = gt[mk]; cv2.imwrite(os.path.join(a.out_dir, s + ".png"), pr); fr.append(mk.mean())
    print(f"[masks] composited {len(names)} renders; excluded pixel fraction {np.mean(fr):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mask", "composite"]); ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir"); ap.add_argument("--out-dir")
    ap.add_argument("--classes", type=int, nargs="+", default=[1]); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=8); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
