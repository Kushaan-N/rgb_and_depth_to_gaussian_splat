"""Training set that adds world-model views to real frames WITHOUT letting them override real data (CPU).

ArtiFixer3D retrains from scratch with its own recipe and gives generated views full weight everywhere,
which degrades areas the real cameras covered well. Here the generated views only teach pixels no
real camera observed: each gets <image>_mask.png (3DGRUT applies it to the photometric loss) = 255 where
its co-visibility mask (covis_masks.py) says never observed, 0 elsewhere (observed, or no LiDAR to tell).
Real frames are unmasked. LiDAR depth targets (<image>_depth.npy) are linked for both, and the real
dataset's init points are kept, so training uses our normal recipe (LiDAR init + depth loss).

  --gen-mode unobserved   generated views supervise only never-observed pixels (default)
  --gen-mode full         generated views supervise every pixel (what ArtiFixer3D does; for comparison)
  --gen-mode none         real frames only
  --holdout-every N       also hold out every Nth real frame (written to <out>_onpath_test) to score
                          on-path quality against real images

    python scripts/build_distill_set.py --real <dataset> --gen-views <dataset of generated poses> \
        --gen-images <dir: one image per generated pose, same names> --out <dir> [--holdout-every 8]
"""
from __future__ import annotations
import argparse, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afx_trajectory import read_views
from fuse_sequences import write_model


def pinhole(cams):
    return {k: ("PINHOLE", W, H, list(p)) for k, (W, H, p) in cams.items()}


def link(src, dst):
    if os.path.exists(src) and not os.path.lexists(dst):
        os.symlink(os.path.realpath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--gen-views", default=None)
    ap.add_argument("--gen-images", default=None)
    ap.add_argument("--gen-mode", choices=["unobserved", "full", "none"], default="unobserved")
    ap.add_argument("--holdout-every", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from PIL import Image

    for d in (args.out, args.out + "_onpath_test"):
        if os.path.isdir(d):
            shutil.rmtree(d)
    rcams, real = read_views(args.real); rcams = pinhole(rcams)
    held = real[::args.holdout_every] if args.holdout_every else []
    hset = {n for n, *_ in held}
    train = [v for v in real if v[0] not in hset]
    cams = dict(rcams); imgs = []
    os.makedirs(os.path.join(args.out, "images", "real"));
    for n, cid, Rwc, C in train:
        imgs.append((f"real/{n}", cid, Rwc, C))
        link(os.path.join(args.real, "images", n), os.path.join(args.out, "images", "real", n))
        link(os.path.join(args.real, "images", n[:-4] + "_depth.npy"), os.path.join(args.out, "images", "real", n[:-4] + "_depth.npy"))
    n_gen = 0; frac = []
    if args.gen_mode != "none":
        gcams, gen = read_views(args.gen_views); gcams = pinhole(gcams)
        off = max(cams) if cams else 0
        cams.update({off + k: v for k, v in gcams.items()})
        os.makedirs(os.path.join(args.out, "images", "gen"))
        for n, cid, Rwc, C in gen:
            imgs.append((f"gen/{n}", off + cid, Rwc, C))
            link(os.path.join(args.gen_images, n), os.path.join(args.out, "images", "gen", n))
            link(os.path.join(args.gen_views, "images", n[:-4] + "_depth.npy"), os.path.join(args.out, "images", "gen", n[:-4] + "_depth.npy"))
            if args.gen_mode == "unobserved":
                cv = np.asarray(Image.open(os.path.join(args.gen_views, "images", n[:-4] + "_covis.png")))
                m = np.where(cv == 128, 255, 0).astype(np.uint8)            # only never-observed pixels
                Image.fromarray(m, mode="L").save(os.path.join(args.out, "images", "gen", n[:-4] + "_mask.png"))
                frac.append(float((m > 0).mean()))
            n_gen += 1
    write_model(os.path.join(args.out, "sparse", "0"), cams, imgs, points3d=True)
    shutil.copyfile(os.path.join(args.real, "sparse", "0", "points3D.txt"), os.path.join(args.out, "sparse", "0", "points3D.txt"))
    if held:
        t = args.out + "_onpath_test"
        write_model(os.path.join(t, "sparse", "0"), rcams, held)
        os.makedirs(os.path.join(t, "images"))
        for n, *_ in held:
            link(os.path.join(args.real, "images", n), os.path.join(t, "images", n))
            link(os.path.join(args.real, "images", n[:-4] + "_depth.npy"), os.path.join(t, "images", n[:-4] + "_depth.npy"))
    extra = f"; generated pixels supervised: {np.mean(frac):.1%}" if frac else ""
    print(f"[distill_set] {len(train)} real + {n_gen} generated ({args.gen_mode}) frames, {len(held)} real held out"
          f"{extra} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
