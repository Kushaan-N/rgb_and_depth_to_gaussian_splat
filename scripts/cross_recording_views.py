"""Another recording's frames as a COLMAP text dataset in THIS recording's splat frame (CPU).

Recordings of the same place share the mocap world frame, so frames of recording B can score
recording A's splat on what A never saw (e.g. the backs of objects B looked at):
  B's COLMAP pose -> metres (B's Sim3) -> A's COLMAP frame (inverse of A's Sim3).
Only valid where the scene did not change between the recordings (check the LiDAR clouds first).

    python scripts/cross_recording_views.py --src-dataset <B>/pipeline/dataset --src-sim3 <B>/pipeline/sim3.json \
        --dst-sim3 <A>/pipeline/sim3.json --out <A>/pipeline/objects/xviews --every 4
"""
from __future__ import annotations
import argparse, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_sequences import cam_to_world, link_images, write_model
from sim3_utils import load_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dataset", required=True)
    ap.add_argument("--src-sim3", required=True)
    ap.add_argument("--dst-sim3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    import pycolmap
    s1, R1, t1 = load_sim3(args.src_sim3); s2, R2, t2 = load_sim3(args.dst_sim3)
    rec = pycolmap.Reconstruction(os.path.join(args.src_dataset, "sparse", "0"))
    cams = {cid: (c.model.name, c.width, c.height, [float(x) for x in c.params]) for cid, c in rec.cameras.items()}
    imgs = []
    for img in sorted(rec.images.values(), key=lambda i: i.name)[::args.every]:
        Rwc, C = cam_to_world(img)
        Rm, Cm = R1 @ Rwc, s1 * R1 @ C + t1                              # B -> metres
        imgs.append((img.name, img.camera_id, R2.T @ Rm, R2.T @ (Cm - t2) / s2))   # metres -> A
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    write_model(os.path.join(args.out, "sparse", "0"), cams, imgs)
    link_images(os.path.join(args.out, "images"), os.path.join(args.src_dataset, "images"), [x[0] for x in imgs])
    print(f"[xviews] {len(imgs)} frames of {args.src_dataset} in the frame of {args.dst_sim3} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
