"""Difix3D progressive novel views for 3DGRUT (CPU). Two steps per round, driven by
sbatch/offpath_difix.sbatch:

  novel     move each novel camera one step toward its target view and write it as a COLMAP text
            dataset to render (images are placeholders), plus refs.json = the real training image
            nearest to each novel camera (Difix's reference)
  trainset  the next round's training set: the real training frames (repeated `--real-repeat` times)
            + this round's Difix-fixed novel views

Mirrors the authors' gsplat trainer (examples/gsplat/simple_trainer_difix3d.py): novel cameras start
at the training camera nearest each target and shift a fixed distance toward it every round; only
the newest round's fixed views are trained on. Their trainer samples fixed views 30% of the time at
loss weight 0.3 (real 1.5); 3DGRUT samples frames uniformly, so repeating the real frames keeps the
fixed views to a similar ~10% share of the gradient.

    python scripts/difix3d_views.py novel --train <offpath>/train_data --targets <offpath>/test \\
        --state <work>/state.json --out <work>/round_1/novel --step-m 0.3 --scale <metres per unit>
    python scripts/difix3d_views.py trainset --train <offpath>/train_data --fixed <work>/round_1/fixed \\
        --novel <work>/round_1/novel --out <work>/round_1/train_data --real-repeat 3
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_sequences import cam_to_world, link_images, write_model


def load(model_dir):
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    cams = {cid: (c.model.name, c.width, c.height, [float(x) for x in c.params]) for cid, c in rec.cameras.items()}
    return cams, sorted((i.name, i.camera_id, *cam_to_world(i)) for i in rec.images.values())


def nearest_ref(Rwc, C, train, min_cos=0.5):
    """Training frame nearest in position among those looking the same way (else nearest overall)."""
    TC = np.array([t[3] for t in train]); TZ = np.array([t[2][:, 2] for t in train])
    d = np.linalg.norm(TC - C, axis=1)
    same = TZ @ Rwc[:, 2] > min_cos
    return train[int(np.argmin(np.where(same, d, d + 1e6)) if same.any() else np.argmin(d))]


def cmd_novel(a):
    from scipy.spatial.transform import Rotation, Slerp
    cams, train = load(os.path.join(a.train, "sparse", "0"))
    tcams, targets = load(os.path.join(a.targets, "sparse", "0"))
    step = a.step_m / a.scale                                   # metres -> model units
    if os.path.exists(a.state):
        cur = {k: (np.array(v["R"]), np.array(v["C"])) for k, v in json.load(open(a.state)).items()}
    else:                                                       # round 1 starts at the nearest training camera
        cur = {}
        for n, _, R, C in targets:
            near = train[int(np.argmin(np.linalg.norm(np.array([t[3] for t in train]) - C, axis=1)))]
            cur[n] = (near[2], near[3])
    new, state, reached, rest = [], {}, 0, []
    for n, cid, Rt, Ct in targets:
        R0, C0 = cur[n]; dist = float(np.linalg.norm(Ct - C0))
        f = 1.0 if dist <= step else step / dist
        C1 = C0 + f * (Ct - C0)
        R1 = Slerp([0, 1], Rotation.from_matrix([R0, Rt]))(f).as_matrix()
        reached += f >= 1.0; rest.append(max(dist - step, 0.0) * a.scale)
        state[n] = {"R": R1.tolist(), "C": C1.tolist()}
        new.append((n, cid, R1, C1))
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    write_model(os.path.join(a.out, "sparse", "0"), tcams, new)
    refs = {}
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    for n, cid, R1, C1 in new:
        ref = nearest_ref(R1, C1, train)
        refs[os.path.splitext(n)[0] + ".png"] = os.path.realpath(os.path.join(a.train, "images", ref[0]))
        os.symlink(refs[os.path.splitext(n)[0] + ".png"], os.path.join(a.out, "images", n))   # placeholder
    json.dump(refs, open(os.path.join(a.out, "refs.json"), "w"), indent=1)
    json.dump(state, open(a.state, "w"))
    print(f"[difix3d] {len(new)} novel views; {reached} at their target; remaining distance "
          f"median {np.median(rest):.2f} m, max {max(rest):.2f} m", flush=True)
    print("DIFIX3D_ALL_REACHED" if reached == len(new) else "DIFIX3D_MOVING", flush=True)


def cmd_trainset(a):
    cams, train = load(os.path.join(a.train, "sparse", "0"))
    ncams, novel = load(os.path.join(a.novel, "sparse", "0"))
    offset = max(cams) if cams else 0
    allcams = {**cams, **{offset + k: v for k, v in ncams.items()}}
    imgs = []
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for k in range(a.real_repeat):                              # real frames, repeated (sampling weight)
        imgs += [(f"real{k}/{n}", cid, R, C) for n, cid, R, C in train]
        link_images(os.path.join(a.out, "images", f"real{k}"), os.path.realpath(os.path.join(a.train, "images")),
                    [n for n, *_ in train])
    imgs += [(f"novel/{n}", offset + cid, R, C) for n, cid, R, C in novel]
    link_images(os.path.join(a.out, "images", "novel"), os.path.realpath(a.fixed), [n for n, *_ in novel])
    write_model(os.path.join(a.out, "sparse", "0"), allcams, imgs, points3d=True)
    shutil.copyfile(os.path.join(a.train, "sparse", "0", "points3D.txt"), os.path.join(a.out, "sparse", "0", "points3D.txt"))
    print(f"[difix3d] training set: {len(train)} real x{a.real_repeat} + {len(novel)} fixed novel views "
          f"({len(novel) / len(imgs):.0%} of frames) -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("novel")
    n.add_argument("--train", required=True); n.add_argument("--targets", required=True)
    n.add_argument("--state", required=True); n.add_argument("--out", required=True)
    n.add_argument("--step-m", type=float, default=0.3); n.add_argument("--scale", type=float, required=True)
    t = sub.add_parser("trainset")
    t.add_argument("--train", required=True); t.add_argument("--fixed", required=True)
    t.add_argument("--novel", required=True); t.add_argument("--out", required=True)
    t.add_argument("--real-repeat", type=int, default=3)
    a = ap.parse_args()
    {"novel": cmd_novel, "trainset": cmd_trainset}[a.cmd](a)


if __name__ == "__main__":
    main()
