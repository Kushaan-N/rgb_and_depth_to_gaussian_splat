"""Leave-arc-out split for testing object completion against real images (3DGRUT venv; GPU if present).

Ground truth for "unseen" object faces: drop a contiguous arc of the recording (--frac of the frames) such
that faces the full recording observed become unobserved by the remaining frames. The dropped frames are
real images at exact poses of exactly those faces. The arc is chosen automatically to maximise the object
surface (complete_objects.py's surface samples, LiDAR-supported) that only it observes.

Writes <out>/train (remaining frames) and <out>/heldout (the arc) as COLMAP text datasets with images,
LiDAR depth targets and the init points, plus <out>/arc.json.

    python scripts/choose_arc.py --objects objects --dataset <all frames with *_depth.npy> --sim3-json sim3.json --out arc
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from complete_objects import object_surface
from covis_masks import cams_of
from afx_trajectory import read_views
from fuse_sequences import write_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--max-incidence", type=float, default=85.0)
    args = ap.parse_args()
    import torch
    from scipy.spatial import cKDTree
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(args.objects + ".json")); pts = np.load(args.objects + ".npz")
    n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"])
    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    allpos, allnor = [], []
    for o in meta["objects"]:
        P = pts[f"obj{o['id']}"]
        pos, nor, *_ = object_surface(P, n, d, 0.02, 0.01)
        sup = cKDTree(P).query(pos)[0] < 0.03
        allpos.append(pos[sup]); allnor.append(nor[sup])
    Xd = ((np.concatenate(allpos) - t) @ R) / s; Nd = np.concatenate(allnor) @ R
    cams = cams_of(args.dataset); F = len(cams)
    X = torch.tensor(Xd, device=dev); Nn = torch.tensor(Nd, device=dev); cosmax = np.cos(np.deg2rad(args.max_incidence))
    vis = torch.zeros(F, len(X), dtype=torch.bool, device=dev)          # frame x sample: observed by that frame
    for f, (name, W, H, fx, fy, cx, cy, Rcw, tcw, dep) in enumerate(cams):
        Rc = torch.tensor(Rcw, device=dev); tc = torch.tensor(tcw, device=dev)
        Pc = X @ Rc.T + tc; z = Pc[:, 2]
        u = torch.floor(fx * Pc[:, 0] / z + cx).long(); v = torch.floor(fy * Pc[:, 1] / z + cy).long()
        Cw = -Rc.T @ tc; vd = Cw[None] - X; vd = vd / torch.linalg.norm(vd, dim=1, keepdim=True)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & ((vd * Nn).sum(1) > cosmax)
        idx = torch.nonzero(ok, as_tuple=True)[0]
        if os.path.exists(dep) and len(idx):
            D = torch.tensor(np.load(dep).astype(np.float32), device=dev).double()
            dt = D[v[idx], u[idx]]; r = torch.linalg.norm(Pc[idx], dim=1)
            idx = idx[(dt <= 0) | ((r - dt).abs() < torch.clamp(0.03 * dt, min=0.03 / s))]
        vis[f, idx] = True
    total = vis.sum(0)                                                   # views per sample over all frames
    L = max(1, int(round(args.frac * F))); best = (-1, 0)
    cs = torch.cat([torch.zeros(1, len(X), dtype=torch.long, device=dev), torch.cumsum(vis.long(), 0)])
    for a in range(0, F - L + 1):
        inarc = cs[a + L] - cs[a]
        lost = int(((total > 0) & (inarc == total)).sum())              # observed ONLY by the arc
        if lost > best[0]:
            best = (lost, a)
    lost, a = best; arc = set(range(a, a + L))
    cams_all, views = read_views(args.dataset)
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    pin = {k: ("PINHOLE", W, H, list(p)) for k, (W, H, p) in cams_all.items()}
    for kind, sel in (("train", [v for i, v in enumerate(views) if i not in arc]), ("heldout", [v for i, v in enumerate(views) if i in arc])):
        dd = os.path.join(args.out, kind)
        write_model(os.path.join(dd, "sparse", "0"), pin, sel, points3d=True if kind == "train" else None)
        os.makedirs(os.path.join(dd, "images"))
        for nme, *_ in sel:
            for suf in ("", "_depth.npy"):
                src = os.path.join(args.dataset, "images", nme if not suf else nme[:-4] + suf)
                if os.path.exists(src):
                    os.symlink(os.path.realpath(src), os.path.join(dd, "images", nme if not suf else nme[:-4] + suf))
    shutil.copyfile(os.path.join(args.dataset, "sparse", "0", "points3D.txt"), os.path.join(args.out, "train", "sparse", "0", "points3D.txt"))
    dj = os.path.join(args.dataset, "depth_targets.json")
    for kind in ("train", "heldout"):
        if os.path.exists(dj):
            shutil.copyfile(dj, os.path.join(args.out, kind, "depth_targets.json"))
    info = {"frames": F, "arc_start": a, "arc_len": L, "arc_names": [views[i][0] for i in sorted(arc)],
            "surface_samples": int(len(X)), "observed_by_all": int((total > 0).sum()), "observed_only_by_arc": lost}
    json.dump(info, open(os.path.join(args.out, "arc.json"), "w"), indent=2)
    print(f"[arc] frames {a}..{a + L - 1} of {F} held out: {lost} of {int((total > 0).sum())} observed object-surface "
          f"samples become unseen ({lost / max(int((total > 0).sum()), 1):.1%}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
