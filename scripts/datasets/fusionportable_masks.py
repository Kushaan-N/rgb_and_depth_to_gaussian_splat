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
    python scripts/datasets/fusionportable_masks.py persist --dataset <train ds> --out-dir <train ds>/masks_persist

  persist   keep only TRANSIENT people: a person who stays put (a seated office worker) is masked in nearly every
            frame, so masking him leaves that region unsupervised (a garbled blob in the splat). Every masked pixel
            with a LiDAR target (<stem>_depth.npy, ray distance) is lifted to a --voxel grid in the dataset frame; a
            voxel is PERSISTENT when it is masked in > --persist-frac of the frames that see it (projects inside the
            image and agrees with that frame's LiDAR depth) and is seen by >= --min-seen frames. A mask component
            (connected region) whose depth-valid pixels are mostly (> --comp-frac) persistent is dropped: that person
            is reconstructed like the rest of the static scene. Writes <out-dir>/<stem>_mask.png (only for frames
            that keep a mask) + persist.json; use --mask-dir <out-dir> in training.
"""
from __future__ import annotations
import argparse, json, os
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


def read_model(ds):
    """cams {id: (W, H, fx, fy, cx, cy)}, [(name, cam id, R_cw, t)] of a PINHOLE COLMAP text model."""
    s0 = os.path.join(ds, "sparse", "0"); cams = {}
    for ln in open(os.path.join(s0, "cameras.txt")):
        p = ln.split()
        if p and not ln.startswith("#"):
            cams[int(p[0])] = (int(p[2]), int(p[3]), *map(float, p[4:8]))
    imgs = []
    for ln in open(os.path.join(s0, "images.txt")):
        p = ln.split()
        if len(p) >= 10 and not ln.startswith("#") and p[9].lower().endswith((".png", ".jpg")):
            w, x, y, z = map(float, p[1:5]); n = np.linalg.norm([w, x, y, z]); w, x, y, z = w / n, x / n, y / n, z / n
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            imgs.append((p[9], int(p[8]), R, np.array(list(map(float, p[5:8])))))
    return cams, imgs


def cmd_persist(a):
    import cv2
    cams, imgs = read_model(a.dataset); d = os.path.join(a.dataset, "images"); os.makedirs(a.out_dir, exist_ok=True)
    stem = lambda n: os.path.splitext(n)[0]
    load = lambda n, suf: os.path.join(d, stem(n) + suf)
    rays = {}

    def ray(cid):
        if cid not in rays:
            W, H, fx, fy, cx, cy = cams[cid]; u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
            r = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1); rays[cid] = r / np.linalg.norm(r, axis=-1, keepdims=True)
        return rays[cid]

    key = lambda X: tuple(np.floor(X / a.voxel).astype(np.int64).T)
    masked, per_frame = {}, {}                          # voxel -> #frames masked; frame -> (mask, voxel ids per pixel)
    for n, cid, R, t in imgs:
        if not (os.path.exists(load(n, "_mask.png")) and os.path.exists(load(n, "_depth.npy"))):
            continue
        m = cv2.imread(load(n, "_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
        if not m.any():
            continue
        dep = np.load(load(n, "_depth.npy")).astype(np.float32); ok = m & (dep > 0)
        X = (ray(cid)[ok] * dep[ok, None]) @ R + (-R.T @ t)          # camera -> world (R_cw^T)
        vk = np.floor(X / a.voxel).astype(np.int64); u, inv = np.unique(vk, axis=0, return_inverse=True)
        for k in map(tuple, u):
            masked[k] = masked.get(k, 0) + 1
        per_frame[n] = (m, ok, vk)
    if not masked:
        print("[masks] persist: no masked pixels with depth"); return
    V = np.array(list(masked.keys()), np.float64); Vc = (V + 0.5) * a.voxel; mc = np.array(list(masked.values()))
    seen = np.zeros(len(V), np.int64)
    for n, cid, R, t in imgs:                           # how many frames see each voxel's surface
        if not os.path.exists(load(n, "_depth.npy")):
            continue
        dep = np.load(load(n, "_depth.npy")).astype(np.float32); W, H, fx, fy, cx, cy = cams[cid]
        Q = Vc @ R.T + t; z = Q[:, 2]; ok = z > 0.3
        uu = np.full(len(Q), -1); vv = np.full(len(Q), -1)
        uu[ok] = (fx * Q[ok, 0] / z[ok] + cx).astype(int); vv[ok] = (fy * Q[ok, 1] / z[ok] + cy).astype(int)
        ok &= (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        r = np.linalg.norm(Q, axis=1); dd = np.zeros(len(Q), np.float32); dd[ok] = dep[vv[ok], uu[ok]]
        seen += ok & (dd > 0) & (np.abs(dd - r) < 0.1 + 0.05 * r)
    frac = mc / np.maximum(seen, 1); pers = (frac > a.persist_frac) & (seen >= a.min_seen)
    P = {tuple(v) for v in V[pers].astype(np.int64)}
    print(f"[masks] persist: {len(V)} masked voxels ({a.voxel} m), {pers.sum()} persistent (masked in > {a.persist_frac:.0%} "
          f"of >= {a.min_seen} seeing frames)", flush=True)
    stats = {"frames_with_mask": len(per_frame), "dropped_components": 0, "kept_components": 0, "frames_written": 0,
             "masked_frac_before": [], "masked_frac_after": []}
    for n, (m, ok, vk) in per_frame.items():
        pv = np.zeros(m.shape, bool); pv[ok] = np.array([tuple(v) in P for v in vk])
        ncomp, lab = cv2.connectedComponents(m.astype(np.uint8))
        out = m.copy()
        for c in range(1, ncomp):
            cm = lab == c; nv = int((cm & ok).sum())
            if nv and (cm & pv).sum() > a.comp_frac * nv:
                out[cm] = False; stats["dropped_components"] += 1
            else:
                stats["kept_components"] += 1
        stats["masked_frac_before"].append(float(m.mean())); stats["masked_frac_after"].append(float(out.mean()))
        if out.any():
            cv2.imwrite(os.path.join(a.out_dir, stem(n) + "_mask.png"), out.astype(np.uint8) * 255); stats["frames_written"] += 1
    for k in ("masked_frac_before", "masked_frac_after"):
        stats[k] = float(np.mean(stats[k]))
    stats.update({"voxels": int(len(V)), "persistent_voxels": int(pers.sum()), "voxel": a.voxel, "persist_frac": a.persist_frac,
                  "min_seen": a.min_seen, "comp_frac": a.comp_frac})
    json.dump(stats, open(os.path.join(a.out_dir, "persist.json"), "w"), indent=1)
    print(f"[masks] persist: components dropped {stats['dropped_components']}, kept {stats['kept_components']}; masked "
          f"fraction {stats['masked_frac_before']:.4f} -> {stats['masked_frac_after']:.4f}; {stats['frames_written']} masks -> {a.out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mask", "composite", "persist"]); ap.add_argument("--dataset", required=True)
    ap.add_argument("--voxel", type=float, default=0.10); ap.add_argument("--persist-frac", type=float, default=0.6)
    ap.add_argument("--min-seen", type=int, default=5); ap.add_argument("--comp-frac", type=float, default=0.5)
    ap.add_argument("--pred-dir"); ap.add_argument("--out-dir")
    ap.add_argument("--classes", type=int, nargs="+", default=[1]); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=8); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
