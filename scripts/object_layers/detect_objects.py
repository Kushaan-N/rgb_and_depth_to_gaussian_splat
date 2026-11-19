"""Image-based detection of every free-standing static object, lifted to 3D and merged with the LiDAR clusters.

The LiDAR detector (scripts/find_objects.py) only finds compact clusters with enough returns and nothing tall above them:
small stacks, things next to furniture, low LiDAR coverage are missed. This adds what the CAMERA saw:

  segment  (venv-seg, GPU) on every --every-th training frame: Grounding-DINO (open, text-prompted boxes; the prompt
           lists come from the config: `objects` = things to complete, `exclude` = furniture / structure the boxes must not
           be confused with, `people` = dynamic, never objects) -> SAM 2.1 box-prompted masks -> <out>/seg/<stem>.npz
  lift     (CPU) every object mask is eroded and back-projected with the base splat's rendered ray depth (the depth the
           splat itself shows at that pixel) into the metric frame; points above the floor vote into a floor-plane grid.
           Multi-view consistency: a cell is an object cell when >= min_views frames segmented an object there AND those
           are >= min_ratio of the frames that saw something standing in that cell (a person / a wrong box in a few frames
           does not survive). Cells people were segmented on more often than objects are dropped. Connected components
           -> objects (size / height / "nothing tall above" / distance-to-path filters, as find_objects.py), each with the
           static LiDAR points in its footprint (the image lift where the LiDAR is too sparse).
           Merge: an image component overlapping a LiDAR cluster confirms it (the cluster keeps its id and points); the
           rest become new ids. -> <out>.json / .npz in find_objects.py's format (+ source, label per object),
           <out>.png (detection map: every id, LiDAR-only / image-only / both), <out>_frames.jpg (mask overlays)

    python detect_objects.py segment --dataset <frames> --out-dir <dir> --cfg-json cfg.json
    python detect_objects.py lift --dataset <frames> --out-dir <dir> --splat-depth-dir <depth> --sim3-json sim3.json \
        --lidar-objects <objects prefix> --cloud lidar_static.ply --out <objects_v2 prefix> --cfg-json cfg.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

DEF = dict(every=2, objects=["box", "brick", "stack of blocks", "crate", "chair", "stool", "ramp", "board", "cone", "bin", "bucket", "bag", "toy", "cart"],
           exclude=["table", "desk", "shelf", "cabinet", "door", "wall", "window", "monitor", "floor", "ceiling", "whiteboard"],
           people=["person"], box_thr=0.3, text_thr=0.25, max_box_share=0.35, nms_iou=0.7, erode_px=3, cell_m=0.02, min_h=0.03, max_h=1.2,
           min_height=0.08, min_views=3, min_ratio=0.3, max_depth_m=5.0, min_size=0.08, max_size=2.0, max_dist=3.0, tall_share=0.02,
           min_points=60, merge_iou=0.05, depth_band_m=0.2, frag_gap_m=0.12, grow_m=0.12, height_slack=1.15, frag_area_m2=0.01, gdino="IDEA-Research/grounding-dino-base", sam="facebook/sam2.1-hiera-large")


def load_cfg(path):
    c = dict(DEF)
    if path and os.path.exists(path):
        j = json.load(open(path)); c.update({k: v for k, v in (j.get("detect") or {}).items() if k in DEF})
    return c


def frame_list(ds, every):
    from build_depth_targets import read_text_model
    _, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
    return sorted(n for n, *_ in imgs)[::max(1, every)]


# ------------------------------------------------------------------ segment (GPU)
def segment(a):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
    cfg = load_cfg(a.cfg_json); dev = "cuda" if torch.cuda.is_available() else "cpu"; od = os.path.join(a.out_dir, "seg"); os.makedirs(od, exist_ok=True)
    gp = AutoProcessor.from_pretrained(cfg["gdino"]); gm = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["gdino"]).to(dev).eval()
    sp = Sam2Processor.from_pretrained(cfg["sam"]); sm = Sam2Model.from_pretrained(cfg["sam"]).to(dev).eval()
    vocab = list(cfg["objects"]) + list(cfg["exclude"]) + list(cfg["people"]); text = ". ".join(vocab) + "."
    kind = {w: ("object" if w in cfg["objects"] else "exclude" if w in cfg["exclude"] else "person") for w in vocab}
    names = frame_list(a.dataset, cfg["every"]); t0 = time.time(); stats = dict(frames=0, object=0, person=0, exclude=0)
    for k, nm in enumerate(names):
        f = os.path.join(od, os.path.splitext(nm)[0] + ".npz")
        if os.path.exists(f):
            continue
        im = Image.open(os.path.join(a.dataset, "images", nm)).convert("RGB"); W, H = im.size
        with torch.no_grad():
            inp = gp(images=im, text=text, return_tensors="pt").to(dev); out = gm(**inp)
            res = gp.post_process_grounded_object_detection(out, inp["input_ids"], threshold=cfg["box_thr"], text_threshold=cfg["text_thr"], target_sizes=[(H, W)])[0]
        labs = res.get("text_labels", res.get("labels")); B = res["boxes"].cpu().numpy(); S = res["scores"].cpu().numpy()
        # map the (possibly partial / merged) phrase back to one vocabulary word: longest vocabulary entry contained in it
        L = []
        for l_ in labs:
            l_ = str(l_).strip().lower(); hit = [w for w in vocab if w in l_] or [w for w in vocab if l_ and l_ in w]
            L.append(max(hit, key=len) if hit else "")
        keep = [i for i in range(len(B)) if L[i] and (B[i, 2] - B[i, 0]) * (B[i, 3] - B[i, 1]) < cfg["max_box_share"] * W * H]
        keep.sort(key=lambda i: -S[i]); sel = []
        for i in keep:                                                      # class-agnostic NMS: one box per thing
            if all(_iou(B[i], B[j]) < cfg["nms_iou"] for j in sel):
                sel.append(i)
        M = np.zeros((0, H, W), bool)
        if sel:
            with torch.no_grad():
                si = sp(images=im, input_boxes=[[B[i].tolist() for i in sel]], return_tensors="pt").to(dev)
                so = sm(**si, multimask_output=False)
                M = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0][:, 0].numpy().astype(bool)
        kinds = [kind[L[i]] for i in sel]
        np.savez_compressed(f, masks=np.packbits(M, axis=-1), shape=np.array([len(sel), H, W]), boxes=B[sel].astype(np.float32) if sel else np.zeros((0, 4), np.float32),
                            scores=S[sel].astype(np.float32) if sel else np.zeros(0, np.float32), labels=np.array([L[i] for i in sel]), kinds=np.array(kinds))
        stats["frames"] += 1
        for kk in kinds:
            stats[kk] += 1
        if k % 50 == 0:
            print(f"[detect] segment {k}/{len(names)} ({time.time() - t0:.0f}s) {json.dumps(stats)}", flush=True)
    json.dump(dict(stats, prompt=text, cfg=cfg), open(os.path.join(a.out_dir, "segment.json"), "w"), indent=1)
    print("DETECT_SEGMENT " + json.dumps(stats), flush=True)


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def Pimg_h_cells(Pi, n, d, q=97):
    """height (m above the floor) the segmented pixels reach: the image-lifted points' q-th percentile (None: too few)."""
    if len(Pi) < 50:
        return None
    return float(np.percentile(Pi @ n + d, q))


def load_seg(f):
    z = np.load(f); n, H, W = z["shape"]
    M = np.unpackbits(z["masks"], axis=-1, count=W).astype(bool) if n else np.zeros((0, H, W), bool)
    return M, z["boxes"], z["scores"], [str(x) for x in z["labels"]], [str(x) for x in z["kinds"]]


# ------------------------------------------------------------------ lift (CPU)
def lift(a):
    import cv2
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from object_layers import load_cams, sim3, depth_of
    cfg = load_cfg(a.cfg_json); s, R, t = sim3(a.sim3_json)
    meta = json.load(open(a.lidar_objects + ".json")); lp = np.load(a.lidar_objects + ".npz")
    n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"]); e1, e2 = np.array(meta["axes"]["e1"]), np.array(meta["axes"]["e2"])
    cams = {c["name"]: c for c in load_cams(a.dataset)}; cell = cfg["cell_m"]
    from plyfile import PlyData
    v = PlyData.read(a.cloud)["vertex"].data; X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); hX = X @ n + d
    uvX = np.stack([X @ e1, X @ e2], 1); lo = uvX.min(0) - 0.5; shape = tuple(np.ceil((uvX.max(0) + 0.5 - lo) / cell).astype(int) + 1)
    gidx = lambda uv: np.floor((uv - lo) / cell).astype(int)
    obj_v = np.zeros(shape, np.int32); per_v = np.zeros(shape, np.int32); occ_v = np.zeros(shape, np.int32)
    lab_v = {}; pts_img = []; cell_tag = {}
    files = sorted(glob.glob(os.path.join(a.out_dir, "seg", "*.npz"))); t0 = time.time(); nmask = dict(object=0, person=0, used=0)
    ys = xs = None
    for k, f in enumerate(files):
        nm = os.path.basename(f)[:-4] + ".png"; c = cams.get(nm); SD = depth_of(a.splat_depth_dir, nm) if c else None
        if c is None or SD is None:
            continue
        M, B, S, L, K = load_seg(f); H, W = SD.shape
        if ys is None or ys.shape != (H, W):
            ys, xs = np.mgrid[0:H, 0:W]
        ray = np.stack([(xs + 0.5 - c["cx"]) / c["fx"], (ys + 0.5 - c["cy"]) / c["fy"], np.ones((H, W))], -1); ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        Pd = (ray * SD[..., None]).reshape(-1, 3); Pm = s * ((Pd - c["t"]) @ c["R"]) @ R.T + t; h = (Pm @ n + d).reshape(H, W)
        val = (SD > 0) & (SD * s < cfg["max_depth_m"]); uv = np.stack([Pm @ e1, Pm @ e2], 1); ij = gidx(uv).reshape(H, W, 2)
        ok = val & (ij[..., 0] >= 0) & (ij[..., 0] < shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < shape[1])
        stand = ok & (h > cfg["min_h"]) & (h < cfg["max_h"])
        seen = np.zeros(shape, bool); seen[ij[stand][:, 0], ij[stand][:, 1]] = True; occ_v += seen
        fo = np.zeros(shape, bool); fp = np.zeros(shape, bool)
        er = np.ones((2 * cfg["erode_px"] + 1,) * 2, np.uint8)
        for m, sc, l_, kd in zip(M, S, L, K):
            if kd == "exclude":
                continue
            m = cv2.erode(m.astype(np.uint8), er).astype(bool) & stand
            if m.sum() < 30:
                continue
            # one object = one depth band: pixels whose splat depth jumps to the background / foreground behind the mask edge
            # are dropped (else they lift into radial streaks); band = max(depth_band_m, its own width at that range)
            rm = SD[m] * s; med = float(np.median(rm)); yy_, xx_ = np.nonzero(m); wid = (np.ptp(xx_) + 1) * med / c["fx"]
            mm = np.abs(SD * s - med) <= max(cfg["depth_band_m"], wid); m &= mm
            if m.sum() < 30:
                continue
            q = ij[m]
            if kd == "person":
                fp[q[:, 0], q[:, 1]] = True; nmask["person"] += 1; continue
            nmask["object"] += 1; g = np.zeros(shape, bool); g[q[:, 0], q[:, 1]] = True; fo |= g
            for key in map(tuple, np.unique(q, axis=0)[::4]):
                lab_v.setdefault(key, {}).setdefault(l_, 0.0); lab_v[key][l_] += float(sc)
            P_ = Pm.reshape(H, W, 3)[m]; pts_img.append(P_[np.random.default_rng(k).choice(len(P_), min(400, len(P_)), replace=False)])
        obj_v += fo; per_v += fp
        if k % 100 == 0:
            print(f"[detect] lift {k}/{len(files)} ({time.time() - t0:.0f}s) masks {json.dumps(nmask)}", flush=True)
    ratio = obj_v / np.maximum(occ_v, 1)
    objc = (obj_v >= cfg["min_views"]) & (ratio >= cfg["min_ratio"]) & (per_v < obj_v)
    objc = ndimage.binary_opening(objc, iterations=1) | (objc & (obj_v >= 3 * cfg["min_views"]))
    lab, nl = ndimage.label(ndimage.binary_closing(objc, iterations=1), structure=np.ones((3, 3)))
    # fragments (a chair's legs / a partly segmented stack) join the larger component within frag_gap_m
    sz = ndimage.sum(np.ones(shape), lab, np.arange(1, nl + 1)) if nl else []; gap = max(1, int(round(cfg["frag_gap_m"] / cell)))
    for k_ in np.argsort(sz):
        k_ += 1; cells = lab == k_
        if not cells.any() or cells.sum() * cell * cell > cfg["frag_area_m2"]:
            continue
        nb = np.unique(lab[ndimage.binary_dilation(cells, iterations=gap) & (lab > 0) & ~cells])
        if len(nb):
            big = max(nb, key=lambda j: (lab == j).sum()); lab[cells] = big
    # things that rise above max_h in the footprint (shelves, desks) are structure, not objects
    tall = np.zeros(shape, bool); tb = (hX >= cfg["max_h"]) & (hX < 2.5); ti = gidx(uvX[tb]); okt = (ti >= 0).all(1) & (ti < np.array(shape)).all(1); tall[ti[okt, 0], ti[okt, 1]] = True
    band = (hX > cfg["min_h"]) & (hX < cfg["max_h"]); bi = gidx(uvX[band]); okb = (bi >= 0).all(1) & (bi < np.array(shape)).all(1); bidx = np.nonzero(band)[0][okb]; bi = bi[okb]
    occ0 = np.zeros(shape, bool); occ0[bi[:, 0], bi[:, 1]] = True; olab, _ = ndimage.label(ndimage.binary_closing(occ0, iterations=1) & ~tall, structure=np.ones((3, 3)))
    Pimg = np.concatenate(pts_img) if pts_img else np.zeros((0, 3)); Pimg_ij = gidx(np.stack([Pimg @ e1, Pimg @ e2], 1)) if len(Pimg) else np.zeros((0, 2), int)
    C = np.stack([s * (c["C"] @ R.T) + t for c in cams.values()]); ptree = cKDTree(np.stack([C @ e1, C @ e2], 1))
    # LiDAR clusters: footprint cells
    lid = {}
    for o in meta["objects"]:
        P_ = lp[f"obj{o['id']}"]; q = gidx(np.stack([P_ @ e1, P_ @ e2], 1)); okq = (q >= 0).all(1) & (q < np.array(shape)).all(1); g = np.zeros(shape, bool); g[q[okq, 0], q[okq, 1]] = True
        lid[o["id"]] = ndimage.binary_dilation(g, iterations=1)
    comps, rej = [], {}
    for kk in range(1, nl + 1):
        cells = lab == kk; ii, jj = np.nonzero(cells)
        if not len(ii):
            continue                                                        # merged into a neighbour
        span = np.array([np.ptp(ii) + 1, np.ptp(jj) + 1]) * cell; diag = float(np.hypot(*span))
        cuv = lo + (np.array([ii.mean(), jj.mean()]) + 0.5) * cell; why = None
        if not (cfg["min_size"] <= diag <= cfg["max_size"]):
            why = f"size {diag:.2f} m"
        elif ptree.query(cuv)[0] > cfg["max_dist"]:
            why = "far from the path"
        elif (ndimage.binary_dilation(cells, iterations=1) & tall).sum() > cfg["tall_share"] * cells.sum():
            why = "under something tall (furniture / structure)"
        # amodal extent: the image votes cover what the cameras segmented (edges below the vote ratio are cut); the LiDAR
        # occupancy components touching the votes (within grow_m of them, not larger than max_size) give the full footprint
        fd = ndimage.binary_dilation(cells, iterations=1); near = ndimage.binary_dilation(cells, iterations=max(1, int(round(cfg["grow_m"] / cell))))
        for lj in np.unique(olab[fd & (olab > 0)]):
            cj = (olab == lj) & near
            if cj.any():
                fd |= cj
        li = bidx[fd[bi[:, 0], bi[:, 1]]]; Pl = X[li]
        Pi = Pimg[fd[Pimg_ij[:, 0].clip(0, shape[0] - 1), Pimg_ij[:, 1].clip(0, shape[1] - 1)]] if len(Pimg) else np.zeros((0, 3))
        # nothing above what the cameras segmented: the grown LiDAR components may touch furniture (a desk leg next to a stack)
        hi_ = Pimg_h_cells(Pimg[cells[Pimg_ij[:, 0].clip(0, shape[0] - 1), Pimg_ij[:, 1].clip(0, shape[1] - 1)]] if len(Pimg) else Pimg, n, d)
        if hi_ is not None:
            Pl = Pl[Pl @ n + d <= hi_ * cfg["height_slack"] + 0.03]
        P_ = Pl if len(Pl) >= cfg["min_points"] else np.concatenate([Pl, Pi])
        hh = float(np.percentile(P_ @ n + d, 99)) if len(P_) else 0.0
        if why is None and (len(P_) < cfg["min_points"] or hh < cfg["min_height"]):
            why = f"too low / too few points (h {hh:.2f} m, {len(P_)} pts)"
        votes = {}
        for i_, j_ in zip(ii, jj):
            for l_, w in lab_v.get((i_, j_), {}).items():
                votes[l_] = votes.get(l_, 0) + w
        label = max(votes, key=votes.get) if votes else ""
        if why:
            rej[kk] = dict(why=why, centre_uv=np.round(cuv, 2).tolist(), label=label); continue
        ov = {oid: float((g & cells).sum() / max((g | cells).sum(), 1)) for oid, g in lid.items()}
        comps.append(dict(cells=cells, P=P_, lidar_pts=int(len(Pl)), label=label, overlap=ov, views=int(obj_v[cells].max()), ratio=round(float(ratio[cells].mean()), 2), span=span, h=hh))
    # merge
    objs, pts, used = [], {}, {}
    for o in meta["objects"]:
        hit = [ci for ci, cp in enumerate(comps) if cp["overlap"].get(o["id"], 0) > cfg["merge_iou"]]
        src = "lidar+image" if hit else "lidar"
        for ci in hit:
            used[ci] = o["id"]
        pts[f"obj{o['id']}"] = lp[f"obj{o['id']}"]
        objs.append(dict(o, source=src, label=comps[hit[0]]["label"] if hit else "", image_views=max([comps[ci]["views"] for ci in hit], default=0)))
    nid = 1 + max([o["id"] for o in meta["objects"]], default=-1)
    for ci, cp in enumerate(comps):
        if ci in used:
            continue
        P_ = cp["P"]; pts[f"obj{nid}"] = P_.astype(np.float64)
        objs.append(dict(id=nid, points=int(len(P_)), footprint_m=np.round(cp["span"], 3).tolist(), height_m=round(cp["h"], 3), centre_m=np.round(P_.mean(0), 4).tolist(),
                         source="image", label=cp["label"], image_views=cp["views"], vote_ratio=cp["ratio"], lidar_points=cp["lidar_pts"])); nid += 1
    np.savez_compressed(a.out + ".npz", **pts)
    json.dump(dict(floor=meta["floor"], axes=meta["axes"], params=dict(meta.get("params", {}), detect=cfg), objects=objs,
                   rejected={str(k): v for k, v in rej.items()}, masks=nmask, frames=len(files)), open(a.out + ".json", "w"), indent=2)
    # detection map
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11)); ext = [lo[0], lo[0] + shape[0] * cell, lo[1], lo[1] + shape[1] * cell]
    occ = np.zeros(shape, bool); occ[bi[:, 0], bi[:, 1]] = True
    ax.imshow(occ.T, origin="lower", cmap="Greys", extent=ext, alpha=0.5); ax.plot(ptree.data[:, 0], ptree.data[:, 1], "c-", lw=0.6, alpha=0.6)
    vis = np.ma.masked_where(obj_v == 0, np.minimum(obj_v, 30)); ax.imshow(vis.T, origin="lower", cmap="autumn_r", extent=ext, alpha=0.35)
    colr = {"lidar": "red", "image": "green", "lidar+image": "blue"}
    for o in objs:
        P_ = pts[f"obj{o['id']}"]; u_, v_ = P_ @ e1, P_ @ e2
        ax.plot([u_.min(), u_.max(), u_.max(), u_.min(), u_.min()], [v_.min(), v_.min(), v_.max(), v_.max(), v_.min()], "-", color=colr[o["source"]], lw=1.2)
        ax.annotate(f"{o['id']}", (u_.max(), v_.max()), color=colr[o["source"]], fontsize=9, weight="bold")
    for r_ in rej.values():
        ax.plot(*r_["centre_uv"], "x", color="grey", ms=4)
    sz = np.ptp(ptree.data, 0); cc_ = ptree.data.mean(0); half = 0.5 * max(sz) + cfg["max_dist"]
    ax.set_xlim(cc_[0] - half, cc_[0] + half); ax.set_ylim(cc_[1] - half, cc_[1] + half)
    ns = {k_: sum(o["source"] == k_ for o in objs) for k_ in colr}
    ax.set_title(f"{len(objs)} objects: blue LiDAR+image {ns['lidar+image']}, green image only {ns['image']}, red LiDAR only {ns['lidar']}; "
                 f"x rejected {len(rej)}; orange = image votes; cyan path", fontsize=9)
    fig.savefig(a.out + ".png", dpi=110, bbox_inches="tight")
    overlay_sheet(a, files, cams)
    print("DETECT_LIFT " + json.dumps(dict(objects=len(objs), **ns, rejected=len(rej), masks=nmask,
                                           ids={o["id"]: f"{o['source']}:{o['label']}" for o in objs})), flush=True)


def overlay_sheet(a, files, cams, n=12):
    import cv2
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]; tiles = []
    col = {"object": (0, 255, 0), "person": (0, 0, 255), "exclude": (128, 128, 128)}
    for f in pick:
        nm = os.path.basename(f)[:-4] + ".png"; im = cv2.imread(os.path.join(a.dataset, "images", nm)).astype(np.float32); M, B, S, L, K = load_seg(f)
        for m, kd in zip(M, K):
            im[m] = 0.55 * im[m] + 0.45 * np.array(col[kd])
        im = np.ascontiguousarray(np.clip(im, 0, 255).astype(np.uint8))
        for b, l_, kd in zip(B, L, K):
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col[kd], 1)
            cv2.putText(im, l_, (int(b[0]), max(10, int(b[1]) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col[kd], 1)
        tiles.append(cv2.resize(im, (480, 360)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out + "_frames.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["segment", "lift"]); ap.add_argument("--dataset", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cfg-json", default=None, help="the config's object_layers block as JSON (uses its `detect` sub-block)")
    ap.add_argument("--splat-depth-dir"); ap.add_argument("--sim3-json"); ap.add_argument("--lidar-objects", help="find_objects.py output prefix")
    ap.add_argument("--cloud"); ap.add_argument("--out", help="merged objects prefix")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    {"segment": segment, "lift": lift}[a.cmd](a)


if __name__ == "__main__":
    main()
