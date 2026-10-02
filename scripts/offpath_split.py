"""Off-path benchmark split (CPU): hold out one spatial region of the trajectory.

The usual every-8th held-out frame sits a few centimetres from a training frame, so it only measures
quality ON the recorded path. A robot in simulation leaves that path. Following Difix3D+ (Wu et al.,
CVPR 2025), cluster the camera positions and hold out a whole cluster: its frames are views the
splat never saw from nearby. Clusters come from seeded k-means on the camera centres; the held-out
one is the cluster whose size is closest to `--holdout` of the frames.

Writes, from the prepped <out>/pipeline (COLMAP poses + depth-seeded init):
  offpath/train_data/  COLMAP text model of the training region + the same init points
  offpath/test/        COLMAP text model of the held-out region (poses in the splat's frame)
  offpath/split.json   names, cluster sizes, distance from each test camera to the nearest train camera

    python scripts/offpath_split.py --config configs/<seq>.yaml [--holdout 0.25 --clusters 4]
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from fuse_sequences import cam_to_world, link_images, write_model
from sim3_utils import load_sim3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--holdout", type=float, default=0.25, help="target fraction of frames held out")
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import pycolmap
    from scipy.cluster.vq import kmeans2

    cfg = pu.load_config(args.config)
    P = os.path.join(cfg["paths"]["out_root"], "pipeline"); O = os.path.join(P, "offpath")
    rec = pycolmap.Reconstruction(os.path.join(P, "dataset", "sparse", "0"))
    s = load_sim3(os.path.join(P, "sim3.json"))[0]                 # COLMAP units -> metres
    cams = {cid: (c.model.name, c.width, c.height, [float(x) for x in c.params]) for cid, c in rec.cameras.items()}
    imgs = sorted((i.name, i.camera_id, *cam_to_world(i)) for i in rec.images.values())
    C = np.array([x[3] for x in imgs])

    _, lab = kmeans2(C, args.clusters, minit="++", seed=args.seed)
    sizes = np.bincount(lab, minlength=args.clusters)
    hold = int(np.argmin(np.abs(sizes - args.holdout * len(imgs))))
    test = lab == hold
    d = np.linalg.norm(C[test][:, None] - C[~test][None], axis=2).min(1) * s   # metres to nearest train camera

    if os.path.isdir(O):
        shutil.rmtree(O)
    for kind, sel in (("train_data", ~test), ("test", test)):
        part = [x for x, k in zip(imgs, sel) if k]
        write_model(os.path.join(O, kind, "sparse", "0"), cams, part, points3d=(kind == "train_data"))
        link_images(os.path.join(O, kind, "images"), os.path.join(P, "dataset", "images"), [x[0] for x in part])
    init = os.path.join(P, "train_data", "sparse", "0", "points3D.txt")
    if os.path.exists(init):
        shutil.copyfile(init, os.path.join(O, "train_data", "sparse", "0", "points3D.txt"))
    else:   # no depth init: start from COLMAP's own points, like the main pipeline
        os.makedirs(os.path.join(O, "_colmap_txt")); rec.write_text(os.path.join(O, "_colmap_txt"))
        shutil.move(os.path.join(O, "_colmap_txt", "points3D.txt"), os.path.join(O, "train_data", "sparse", "0", "points3D.txt"))
        shutil.rmtree(os.path.join(O, "_colmap_txt"))

    info = {"frames": len(imgs), "train": int((~test).sum()), "test": int(test.sum()),
            "cluster_sizes": sizes.tolist(), "held_out_cluster": hold, "seed": args.seed,
            "test_to_nearest_train_m": {"median": round(float(np.median(d)), 3),
                                        "p90": round(float(np.percentile(d, 90)), 3), "max": round(float(d.max()), 3)},
            "test_names": [x[0] for x, k in zip(imgs, test) if k]}
    json.dump(info, open(os.path.join(O, "split.json"), "w"), indent=2)
    print(f"[offpath] {info['train']} train / {info['test']} held-out frames (cluster {hold} of {sizes.tolist()}); "
          f"held-out camera to nearest train camera: {info['test_to_nearest_train_m']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
