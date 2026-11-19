"""Layered, object-centric scene: every free-standing object becomes ONE closed textured mesh; the rest stays a splat.

CPU only (numpy / scipy / open3d raycasting / skimage / trimesh: $CEAR_WS/venv-trellis python). No per-object or per-scene
constants: everything comes from the detector output, the cameras, the LiDAR cloud and the `object_layers` config block.

  evidence    per object of find_objects.py: LiDAR cluster cleaned of floor-noise tails (cause of the old o1 failure),
              object frame, the training views that see it, and a voxel grid of what those views MEASURED: free space (>= k
              views looked through the voxel: rendered depth of the base splat is farther), seen surface, unknown.
              Also writes the cut-out + context crops for the vision-language stage -> <dir>/ev, <dir>/vlm_in, evidence.json
  candidates  per object up to two complete shapes, both fitted to the evidence and carved by measured free space:
                prim  the VLM shape recipe (<dir>/vlm/recipe_obj<id>.json: cuboids / cylinders / wedges / plates, stacking)
                      -> similarity + per-primitive dimensions fitted to seen surface + LiDAR + free space
                gen   the image-to-3D asset (<gen-dir>/obj<id>_s<seed>.npz) -> yaw + per-axis scale + shift registration,
                      fused with a crust behind the seen surface (hull of the evidence box when there is no asset)
              texture: real photos on every face a training camera saw (visibility by ray casting, view-angle / distance /
              sharpness weights, per-frame exposure gains), unseen faces from the SAME-MATERIAL seen faces (prim) or the
              colour-matched generated texture (gen), feathered seam. Held-out numeric check + a comparison sheet.
  finalize    pick per object (VLM choice <dir>/vlm/select_obj<id>.json, else numeric), apply reviewed edits, write
              obj<id>_mesh.npz / .glb, scene.glb, scene.usda (metric scene frame), review renders, objects_layers.json
  floormap    floor-texture map (make_lidar_ibr_views.py --floor-map) + object footprints -> the cells to complete: under
              every object and its occlusion shadow (hole_override for floor_fill_ibr.py)
  compose     background PLY minus the object zones + one flat opaque gaussian per mesh vertex -> scene PLY
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402

C0 = 0.28209479177387814
DEF = dict(voxel_m=0.005, max_voxels=6e6, margin_m=0.08, cam_max_m=4.0, max_views=160, tol_abs_m=0.012, tol_rel=0.01, min_free_views=3,
           min_surf_views=3, texel_m=0.003, crust_m=0.03, carve_tol_m=0.015, max_resid_m=0.03, cos_min=0.2, q_min_px_m=60.0, top_k=4,
           sharp_pow=4.0, mv_floor_z=3.0, mv_black=0.03, seed_m=0.005, lidar_contain=0.0, yaw_free=1, lidar_solid=0, mv_n=24, mv_dist=(0.5, 1.5), mv_h=(0.3, 1.2), mv_floor_m=1.5, mv_clear_m=0.05,
           mv_min_share=0.02, mv_feather_px=3, min_face_psnr=20.0, solid_margin_m=0.015, floor_pixel_m=-1.0, backing_m=0.008, backing_sigma=0.7, backing_inset_m=0.0015, shadow_cams=1000, colour_reach_m=0.15, refit_iters=600, max_object_loss_db=2.0, gate_frames=20, base_m=0.02, prefer_prim_iou_tol=0.03, foot_close_m=0.03, foot_grow_m=0.01, photo_tol_m=0.015, rest_m=0.06, auto_box_resid_m=0.012, seam_m=0.02, val_every=6, zone_m=0.03, floor_m=0.012, gauss_sigma=0.9, gauss_thick_m=0.0005, gauss_opacity=0.985,
           floor_close_m=0.25, shadow_reach_m=1.0, strict_gates=False, fallback_object_loss_db=0.0, unseen_fill='synth', synth_patch_m=0.06, synth_z=3.0, synth_q_mult=1.5, synth_contact_m=0.06, shell_m=0.02, extent_cap_m=0.05, zone_mode='sdf', seen_from='splat', inner_keep_m=10.0, local_floor=False, unseen_clear_m=0.10, unseen_clear_min_h=0.05, min_vis_views=2, keep_train_w=0.05, match_unseen_colour=True, vd_refit_iters=400, vd_cone_deg=50.0, vd_lambda=2.0, min_prim_confidence=0.3, plane_snap=True, max_edit_m=0.05, vlm_images=4)


# ------------------------------------------------------------------ small helpers
def sim3(path):
    j = json.load(open(path)); return float(j["scale"]), np.array(j["R"]), np.array(j["t"])


def load_cfg(a):
    c = dict(DEF)
    if a.cfg_json and os.path.exists(a.cfg_json):
        c.update({k: v for k, v in json.load(open(a.cfg_json)).items() if k in DEF})
    return c


def load_cams(ds):
    cams, imgs = read_text_model(os.path.join(ds, "sparse", "0")); out = []
    for name, cid, q, tcw in sorted(imgs, key=lambda x: x[0]):
        W, H, (fx, fy, cx, cy) = cams[cid]; Rc = quat_to_R(q)
        out.append(dict(name=name, W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, R=Rc, t=np.asarray(tcw), C=-Rc.T @ np.asarray(tcw), ds=ds))
    return out


class Frame:
    """object frame: floor point c0, a1, a2 (min-area footprint rectangle), n up; local = (u, v, h above the floor)."""
    def __init__(self, c0, a1, a2, n, d):
        self.c0, self.a1, self.a2, self.n, self.d = (np.asarray(x, float) for x in (c0, a1, a2, n, [d])); self.d = float(self.d[0])

    def loc(self, P):
        r = P - self.c0; return np.stack([r @ self.a1, r @ self.a2, P @ self.n + self.d], 1)

    def met(self, L):
        return self.c0 + np.outer(L[:, 0], self.a1) + np.outer(L[:, 1], self.a2) + np.outer(L[:, 2], self.n)

    def vec(self, V):
        return np.outer(V[:, 0], self.a1) + np.outer(V[:, 1], self.a2) + np.outer(V[:, 2], self.n)

    def dump(self):
        return dict(c0=self.c0.tolist(), a1=self.a1.tolist(), a2=self.a2.tolist(), n=self.n.tolist(), d=self.d)


def floor_basis(n):
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def clean_points(X, n, d, cell=0.02):
    """drop the floor-noise tails of a LiDAR cluster: keep points over the (4 cm dilated) footprint of its tall core."""
    from scipy import ndimage
    h = X @ n + d; hmax = np.percentile(h, 99.5); core = h > max(0.06, 0.3 * hmax)
    if core.sum() < 30:
        return X, 1
    e1, e2 = floor_basis(n); uv = np.stack([X @ e1, X @ e2], 1); lo = uv.min(0) - 0.1
    ij = np.floor((uv - lo) / cell).astype(int); shape = ij.max(0) + 6; g = np.zeros(shape, bool); g[ij[core, 0], ij[core, 1]] = True
    lab, nl = ndimage.label(ndimage.binary_dilation(g, iterations=1)); sizes = ndimage.sum(g, lab, np.arange(1, nl + 1)) if nl else []
    ncomp = int((np.asarray(sizes) >= 0.15 * max(np.max(sizes), 1)).sum()) if nl else 1
    keep = ndimage.binary_dilation(g, iterations=2)[ij[:, 0], ij[:, 1]]
    return X[keep], ncomp


def split_points(X, n, d, cell=0.02, min_share=0.1):
    """physically separate things in one LiDAR cluster: connected components of the tall core's footprint."""
    from scipy import ndimage
    h = X @ n + d; core = h > max(0.06, 0.3 * np.percentile(h, 99.5)); e1, e2 = floor_basis(n); uv = np.stack([X @ e1, X @ e2], 1); lo = uv.min(0) - 0.1
    ij = np.floor((uv - lo) / cell).astype(int); g = np.zeros(ij.max(0) + 6, bool); g[ij[core, 0], ij[core, 1]] = True
    lab, nl = ndimage.label(g, structure=np.ones((3, 3)))
    if nl < 2:
        return [X]
    sz = ndimage.sum(g, lab, np.arange(1, nl + 1)); keep = 1 + np.nonzero(sz >= min_share * sz.max())[0]
    if len(keep) < 2:
        return [X]
    idx = ndimage.distance_transform_edt(~np.isin(lab, keep), return_distances=False, return_indices=True); near = lab[idx[0], idx[1]][ij[:, 0], ij[:, 1]]
    return [X[near == k] for k in keep]


def apply_detection(a, work, n, d):
    """vision-language detection check (recipe_obj<id>.json: detection.separate_objects / partial) -> split / merge the clusters."""
    notes = {}; out = []; rec = {}
    for oid, X in work:
        f = os.path.join(a.dir, "vlm", f"recipe_obj{oid}.json")
        if os.path.exists(f):
            rec[oid] = (json.load(open(f)).get("detection") or {})
    e1, e2 = floor_basis(n); box = {oid: (np.percentile(np.stack([clean_points(X, n, d)[0] @ e1, clean_points(X, n, d)[0] @ e2], 1), [1, 99], 0)) for oid, X in work}
    gone = set()
    for oid, X in work:
        if oid in gone:
            continue
        det = rec.get(oid, {})
        if det.get("partial"):                                              # the rest of it may be a neighbouring cluster
            for o2, X2 in work:
                if o2 == oid or o2 in gone:
                    continue
                gap = np.maximum(np.maximum(box[oid][0] - box[o2][1], box[o2][0] - box[oid][1]), 0)
                if float(np.hypot(*gap)) < 0.06:
                    X = np.concatenate([X, X2]); gone.add(o2); notes[oid] = f"merged with o{o2} (cutout partial, clusters {float(np.hypot(*gap)):.2f} m apart)"
        if int(det.get("separate_objects", 1) or 1) >= 2:
            parts = split_points(clean_points(X, n, d)[0], n, d)
            if len(parts) >= 2:
                notes[oid] = f"split into {len(parts)} (vision-language: {det.get('separate_objects')} separate objects)"
                out += [(100 * (oid + 1) + k, P_) for k, P_ in enumerate(parts)]; continue
            notes[oid] = f"vision-language sees {det.get('separate_objects')} separate objects but the LiDAR core is one piece: kept as one"
        out.append((oid, X))
    return [w for w in out if w[0] not in gone], notes


def make_frame(X, n, d):
    e1, e2 = floor_basis(n); h = X @ n + d; F = X[h > 0.045] if (h > 0.045).sum() > 20 else X
    uv = np.stack([F @ e1, F @ e2], 1); best = None
    for th in np.radians(np.arange(0, 90, 1.0)):
        c, s_ = np.cos(th), np.sin(th); q = np.stack([c * uv[:, 0] + s_ * uv[:, 1], -s_ * uv[:, 0] + c * uv[:, 1]], 1)
        lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0); ar = np.prod(hi - lo)
        if best is None or ar < best[0]:
            best = (ar, th, lo, hi)
    th = best[1]; a1 = np.cos(th) * e1 + np.sin(th) * e2; fr = Frame(-d * n, a1, np.cross(n, a1), n, d); L = fr.loc(F)
    p1, p99 = np.percentile(L[:, :2], 1, 0), np.percentile(L[:, :2], 99, 0); mid = (p1 + p99) / 2; half = (p99 - p1) / 2
    fr.c0 = fr.c0 + mid[0] * fr.a1 + mid[1] * fr.a2
    return fr, half


def depth_of(depth_dir, name):
    f = os.path.join(depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy")
    return np.load(f).astype(np.float32) if os.path.exists(f) else None


def load_img(cam):
    import cv2
    im = cv2.imread(os.path.join(cam["ds"], "images", cam["name"]))
    return im[..., ::-1].astype(np.float32) / 255


def project(cam, Xd):
    P = Xd @ cam["R"].T + cam["t"]; z = P[:, 2]; ok = z > 0.05; zz = np.where(ok, z, 1.0)
    return cam["fx"] * P[:, 0] / zz + cam["cx"], cam["fy"] * P[:, 1] / zz + cam["cy"], np.linalg.norm(P, axis=1), ok


class Grid:
    def __init__(self, lo, vox, shape):
        self.lo, self.vox, self.shape = np.asarray(lo, float), float(vox), tuple(int(x) for x in shape)

    def centres(self):
        ax = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * self.vox for i in range(3)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)

    def index(self, L):
        ij = np.floor((L - self.lo) / self.vox).astype(int); ok = np.all((ij >= 0) & (ij < np.array(self.shape)), 1)
        return np.clip(ij, 0, np.array(self.shape) - 1), ok

    def look(self, A, L, fill=0):
        ij, ok = self.index(L); v = A[ij[:, 0], ij[:, 1], ij[:, 2]]
        return np.where(ok, v, fill)


# ------------------------------------------------------------------ evidence
def evidence(a):
    import cv2
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); to_ds = lambda X: ((X - t) @ R) / s
    meta = json.load(open(a.objects + ".json")); pts = np.load(a.objects + ".npz")
    n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"]); cams = load_cams(a.dataset)
    os.makedirs(os.path.join(a.dir, "ev"), exist_ok=True); os.makedirs(os.path.join(a.dir, "vlm_in"), exist_ok=True); rep = []
    agc = {}
    if a.gen_dir and os.path.exists(os.path.join(a.gen_dir, "crops.json")):
        agc = {c["id"]: c["views"] for c in json.load(open(os.path.join(a.gen_dir, "crops.json")))}
    work = [(o["id"], pts[f"obj{o['id']}"]) for o in meta["objects"] if not a.ids or o["id"] in a.ids]; notes = {}; prev = {}
    if a.apply_detection:
        work, notes = apply_detection(a, work, n, d); ef = os.path.join(a.dir, "evidence.json")
        prev = {r_["id"]: r_ for r_ in json.load(open(ef))["objects"]} if os.path.exists(ef) else {}
        print("[layers] detection check: " + (json.dumps(notes) if notes else "no split / merge"), flush=True)
    fpm = {o["id"]: o["footprint_m"] for o in meta["objects"]}
    for oid, X0 in work:
        if a.apply_detection and oid in prev and oid not in notes and os.path.exists(os.path.join(a.dir, "ev", f"obj{oid}.npz")):
            rep.append(prev[oid]); continue
        t0 = time.time(); X, ncomp = clean_points(X0, n, d); fr, half = make_frame(X, n, d); L = fr.loc(X)
        hmax = float(np.percentile(L[:, 2], 99.5)) + 0.01; m = cfg["margin_m"]
        lo = np.array([-half[0] - m, -half[1] - m, 0.0]); hi = np.array([half[0] + m, half[1] + m, hmax + m])
        vox = max(cfg["voxel_m"], float(np.cbrt(np.prod(hi - lo) / cfg["max_voxels"]))); lo[2] = -vox
        G = Grid(lo, vox, np.ceil((hi - lo) / vox).astype(int)); Lc = G.centres().astype(np.float32); Xd = to_ds(fr.met(Lc)).astype(np.float32)
        cm = fr.met(np.array([[0, 0, hmax / 2]]))[0]; cd = to_ds(cm[None])[0]; rob = float(np.linalg.norm(np.r_[half, hmax / 2])); sel = []
        for ci, c in enumerate(cams):
            pc = c["R"] @ cd + c["t"]; r = np.linalg.norm(pc)
            if pc[2] < 0.05 or r * s > cfg["cam_max_m"] or r * s < rob + 0.05:
                continue
            u = c["fx"] * pc[0] / pc[2] + c["cx"]; v = c["fy"] * pc[1] / pc[2] + c["cy"]
            if not (-0.1 * c["W"] <= u < 1.1 * c["W"] and -0.1 * c["H"] <= v < 1.1 * c["H"]):
                continue
            SD = depth_of(a.splat_depth_dir, c["name"])
            if SD is None:
                continue
            if 0 <= u < c["W"] and 0 <= v < c["H"] and SD[int(v), int(u)] > 0 and (r - SD[int(v), int(u)]) * s > rob + 0.1:
                continue                                                   # something else stands in front of the whole object
            sel.append(ci)
        if len(sel) < 3:
            print(f"[layers] evidence o{oid}: {len(sel)} training views see it: skipped (keeps its splat)", flush=True); continue
        if len(sel) > cfg["max_views"]:
            sel = [sel[i] for i in np.linspace(0, len(sel) - 1, cfg["max_views"]).round().astype(int)]
        free = np.zeros(len(Xd), np.uint16); surf = np.zeros(len(Xd), np.uint16); seen = np.zeros(len(Xd), np.uint16)
        for ci in sel:
            c = cams[ci]; SD = depth_of(a.splat_depth_dir, c["name"]); u, v, r, ok = project(c, Xd)
            ui, vi = np.floor(u).astype(int), np.floor(v).astype(int); ok &= (ui >= 0) & (ui < c["W"]) & (vi >= 0) & (vi < c["H"])
            i = np.nonzero(ok)[0]; sd = SD[vi[i], ui[i]]; val = sd > 0; tol = np.maximum(cfg["tol_abs_m"] / s, cfg["tol_rel"] * sd)
            free[i] += val & (r[i] < sd - tol); surf[i] += val & (np.abs(r[i] - sd) <= tol); seen[i] += val
        sh = G.shape; free = free.reshape(sh); surf = surf.reshape(sh); seen = seen.reshape(sh)
        az = [float(np.arctan2(*(fr.loc((s * (cams[ci]["C"] @ R.T) + t)[None])[0][[1, 0]]))) for ci in sel]
        np.savez_compressed(os.path.join(a.dir, "ev", f"obj{oid}.npz"), free=free, surf=surf, seen=seen, lo=G.lo, vox=G.vox, shape=np.array(G.shape),
                            lidar=L.astype(np.float32), half=half, hmax=hmax, views=np.array([cams[ci]["name"] for ci in sel]), az=np.array(az),
                            **{f"fr_{k}": np.asarray(v) for k, v in fr.dump().items()})
        # crops for the vision-language stage: the asset stage's BiRefNet cut-outs when present, else depth-masked ones
        box = fr.met(np.array([[sx * (half[0] + 0.03), sy * (half[1] + 0.03), z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, hmax + 0.03)]))
        boxd = to_ds(box); names = [nm for nm in agc.get(oid, []) if any(cams[ci]["name"] == nm for ci in range(len(cams)))]
        names = names[:1]                                                   # one of the asset stage's views + views spread over azimuth / distance
        if sel:
            bins = {}
            for ci, az_ in zip(sel, az):
                b = int((az_ + np.pi) / (2 * np.pi) * 16) % 16; r = np.linalg.norm(cams[ci]["C"] - cd)
                if b not in bins or r < bins[b][0]:
                    bins[b] = (r, cams[ci]["name"])
            spread = [v[1] for _, v in sorted(bins.items())]; byd = [cams[ci]["name"] for ci in sorted(sel, key=lambda ci: np.linalg.norm(cams[ci]["C"] - cd))]
            for nm in spread + [byd[len(byd) // 2], byd[-1], byd[len(byd) // 4]]:
                if nm not in names:
                    names.append(nm)
        byname = {c["name"]: c for c in cams}; pm_ = {}
        if a.seg_dir and sel:                                               # more candidates: the person filter drops some
            names += [nm for nm in byd[::max(1, len(byd) // 24)] if nm not in names]
        if a.seg_dir:                                                       # only frames the segmenter checked, and none with a person
            keep_ = []                                                       # anywhere in the crop (people never go to the API)
            for nm in names:
                pm = person_mask(a.seg_dir, nm)
                if pm is None:
                    continue
                u, v, _, ok = project(byname[nm], boxd)
                if ok.all():
                    x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max(); S = 1.5 * max(x1 - x0, y1 - y0); cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
                    if pm[max(0, int(cy_ - S / 2)):max(0, int(cy_ + S / 2)), max(0, int(cx_ - S / 2)):max(0, int(cx_ + S / 2))].any():
                        continue
                keep_.append(nm); pm_[nm] = pm
            names = keep_
        names = names[:cfg["vlm_images"]]
        for k, nm in enumerate(names):
            c = byname[nm]; img = load_img(c); u, v, _, ok = project(c, boxd)
            if not ok.all():
                continue
            x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max(); S = 1.5 * max(x1 - x0, y1 - y0); cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
            M = np.array([[512 / S, 0, 256 - cx_ * 512 / S], [0, 512 / S, 256 - cy_ * 512 / S]])
            cf = os.path.join(a.gen_dir or "", "crops", f"obj{oid}_{k}.png")
            if a.gen_dir and os.path.exists(cf) and nm in agc.get(oid, []) and not a.seg_dir:
                cut = cv2.imread(cf)
            else:                                                           # depth mask: pixels whose rendered surface lies in the object's box
                SD = depth_of(a.splat_depth_dir, nm)
                if SD is None:
                    continue
                ys, xs = np.mgrid[0:c["H"], 0:c["W"]]
                ray = np.stack([(xs + 0.5 - c["cx"]) / c["fx"], (ys + 0.5 - c["cy"]) / c["fy"], np.ones_like(xs, float)], -1); ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
                Pl = fr.loc(s * (((ray * SD[..., None]).reshape(-1, 3) - c["t"]) @ c["R"]) @ R.T + t).reshape(c["H"], c["W"], 3)
                mk = (np.abs(Pl[..., 0]) < half[0] + 0.03) & (np.abs(Pl[..., 1]) < half[1] + 0.03) & (Pl[..., 2] > 0.015) & (Pl[..., 2] < hmax + 0.03) & (SD > 0)
                if a.seg_dir and mk.sum() < 0.25 * (x1 - x0) * (y1 - y0):   # mostly hidden behind something in this frame: no use
                    continue
                cut = cv2.warpAffine(img * mk[..., None], M, (512, 512), flags=cv2.INTER_CUBIC, borderValue=0)
                cut = np.clip(cut[..., ::-1] * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.dir, "vlm_in", f"obj{oid}_cut{k}.jpg"), cut, [cv2.IMWRITE_JPEG_QUALITY, 90])
        fm = free >= cfg["min_free_views"]
        rep.append(dict(id=oid, lidar_points=int(len(X0)), lidar_points_kept=int(len(X)), core_components=ncomp, box_m=[round(2 * float(half[0]), 3), round(2 * float(half[1]), 3), round(hmax, 3)],
                        detector_footprint_m=fpm.get(oid), detection=notes.get(oid), views=len(sel), voxel_m=round(vox, 4), grid=list(G.shape), free_share=round(float(fm.mean()), 3),
                        surface_voxels=int(((surf >= cfg["min_surf_views"]) & ~fm).sum()), azimuth_coverage=round(float(len({int((x + np.pi) / (2 * np.pi) * 16) % 16 for x in az}) / 16), 2),
                        vlm_crops=len(names), seconds=round(time.time() - t0)))
        print("[layers] evidence " + json.dumps(rep[-1]), flush=True)
    json.dump(dict(objects=rep, floor=meta["floor"], cfg=cfg, detection_notes={str(k): v for k, v in notes.items()}), open(os.path.join(a.dir, "evidence.json"), "w"), indent=1)


def ndimage_dilate(m, it):
    from scipy import ndimage
    return ndimage.binary_dilation(m, iterations=it)


def person_mask(seg_dir, name):
    """union of the segmenter's person masks in that frame (detect_objects.py segment), None when the frame was not segmented."""
    f = os.path.join(seg_dir, os.path.splitext(name)[0] + ".npz")
    if not os.path.exists(f):
        return None
    z = np.load(f); n, H, W = z["shape"]; kinds = [str(x) for x in z["kinds"]]
    M = np.unpackbits(z["masks"], axis=-1, count=W).astype(bool) if n else np.zeros((0, H, W), bool)
    out = np.zeros((H, W), bool)
    for m, k in zip(M, kinds):
        if k == "person":
            out |= m
    return out


def load_ev(a, oid, cfg):
    z = np.load(os.path.join(a.dir, "ev", f"obj{oid}.npz")); fr = Frame(z["fr_c0"], z["fr_a1"], z["fr_a2"], z["fr_n"], float(z["fr_d"]))
    G = Grid(z["lo"], float(z["vox"]), z["shape"]); free = z["free"] >= cfg["min_free_views"]
    Lc = G.centres().reshape(G.shape + (3,)); surf = (z["surf"] >= cfg["min_surf_views"]) & ~free & (Lc[..., 2] > max(0.025, 3 * float(z["vox"])))
    # the object's amodal footprint from its own LiDAR cluster (seen by the spinning LiDAR from far more sides than the camera):
    # nothing of this object lies outside it (neighbouring clutter, background in the grid, an asset wider than the object)
    from scipy import ndimage
    lid = z["lidar"].astype(float); hmax = float(z["hmax"]); P = lid[(lid[:, 2] > 0.03) & (lid[:, 2] < hmax + 0.01)]
    f2 = np.zeros(G.shape[:2], bool); ij, ok = G.index(np.c_[P[:, :2], np.zeros(len(P))]); f2[ij[ok, 0], ij[ok, 1]] = True
    r_ = max(1, int(round(cfg["foot_close_m"] / G.vox))); f2 = ndimage.binary_closing(np.pad(f2, r_ + 1), iterations=r_)[r_ + 1:-r_ - 1, r_ + 1:-r_ - 1]
    f2 = ndimage.binary_fill_holes(f2); f2 = ndimage.binary_dilation(f2, iterations=max(1, int(round(cfg["foot_grow_m"] / G.vox))))
    foot = f2[:, :, None] & (Lc[..., 2] <= hmax + 0.015); surf &= foot
    # LiDAR solid: under the LiDAR's top surface of each footprint cell the object is solid, whatever the splat's rendered depth
    # says (a semi-transparent splat renders depths BEHIND its surface, which would read as free space inside the object)
    Ht = np.full(G.shape[:2], -1.0); ij2, ok2 = G.index(np.c_[P[:, :2], np.zeros(len(P))]); np.maximum.at(Ht, (ij2[ok2, 0], ij2[ok2, 1]), P[ok2, 2])
    Ht = ndimage.grey_closing(ndimage.maximum_filter(Ht, size=max(3, 2 * int(round(0.015 / G.vox)) + 1)), size=max(3, 2 * r_ + 1))
    solid = f2[:, :, None] & (Lc[..., 2] <= Ht[:, :, None] - cfg["solid_margin_m"])
    if cfg["lidar_solid"]:
        free = free & ~solid
    return dict(fr=fr, G=G, free=free, surf=surf, lidar=lid, half=z["half"], hmax=hmax, views=[str(x) for x in z["views"]], az=z["az"], Lc=Lc, foot=foot)


# ------------------------------------------------------------------ primitives (VLM recipe)
def box_sdf(x, y, z, hx, hy, hz):
    qx, qy, qz = np.abs(x) - hx, np.abs(y) - hy, np.abs(z) - hz
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)


def prim_local(p, P):
    x, y = P[:, 0] - p["cx"], P[:, 1] - p["cy"]; c, s_ = np.cos(p["yaw"]), np.sin(p["yaw"])
    return c * x + s_ * y, -s_ * x + c * y, P[:, 2] - p["z0"]


def prim_sdf(p, P):
    xp, yp, z = prim_local(p, P); sx, sy, sz = p["sx"], p["sy"], p["sz"]; ty = p["type"]
    if ty == "cuboid":
        return box_sdf(xp, yp, z - sz / 2, sx / 2, sy / 2, sz / 2)
    if ty == "wedge":                                                       # slope descends toward +x'
        return np.maximum(box_sdf(xp, yp, z - sz / 2, sx / 2, sy / 2, sz / 2), (z / sz + xp / sx - 0.5) * (sx * sz / np.hypot(sx, sz)))
    if ty == "cylinder":
        if p.get("axis", "z") == "z":
            dr = np.hypot(xp, yp) - sx / 2; dz = np.abs(z - sz / 2) - sz / 2
        else:                                                               # lying along x'
            dr = np.hypot(yp, z - sz / 2) - sz / 2; dz = np.abs(xp) - sx / 2
        return np.minimum(np.maximum(dr, dz), 0) + np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))
    if ty == "plate":                                                       # thin board on the floor leaning toward +y' by tilt
        tl = p["tilt"]; l = sz / np.cos(tl); y0 = -sz * np.tan(tl) / 2
        ql = (yp - y0) * np.sin(tl) + z * np.cos(tl); qn = (yp - y0) * np.cos(tl) - z * np.sin(tl)
        return box_sdf(xp, qn, ql - l / 2, sx / 2, p["thick"] / 2, l / 2)
    raise ValueError(ty)


def prims_sdf(prims, P):
    D = np.stack([prim_sdf(p, P) for p in prims], 1)
    return D.min(1), D


DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def prims_init(rec, half, hmax, fx, fy, swap, vox):
    """recipe (box fractions) -> primitives in the object frame for one mirror / axis assignment."""
    bu, bv = 2 * half[0], 2 * half[1]; long_u = (bu >= bv) != swap; out = []

    def vec(x, y):                                                          # recipe (x, y) direction / offset -> local (u, v)
        x, y = fx * x, fy * y
        return (x, y) if long_u else (y, x)
    bx, by = (bu, bv) if long_u else (bv, bu)
    for q in rec["primitives"]:
        ty = q.get("type")
        if ty not in ("cuboid", "cylinder", "wedge", "plate"):
            return None
        cx, cy = vec((float(q["center"][0]) - 0.5) * bx, (float(q["center"][1]) - 0.5) * by)
        sx, sy = abs(float(q["size"][0])) * bx, abs(float(q["size"][1])) * by; sz = abs(float(q["size"][2])) * hmax
        if not long_u:
            sx, sy = sy, sx
        p = dict(id=str(q.get("id", len(out))), type=ty, cx=cx, cy=cy, yaw=np.radians(float(q.get("yaw_deg", 0) or 0)) * fx * fy * (1 if long_u else -1),
                 sx=max(sx, 2 * vox), sy=max(sy, 2 * vox), sz=max(sz, 2 * vox), on=str(q.get("on", "floor")), material=str(q.get("material", "m0")), z0=0.0)
        if ty in ("wedge", "plate"):
            dx, dy = vec(*DIRS.get(q.get("low_side" if ty == "wedge" else "lean", "+x"), (1, 0))); th = np.arctan2(dy, dx)
            if ty == "plate":
                th -= np.pi / 2                                             # y' = lean direction
            if abs(np.cos(th)) < 0.5:                                       # prim x' runs along local v: sizes follow
                p["sx"], p["sy"] = p["sy"], p["sx"]
            p["yaw"] += th
            if ty == "plate":
                p["tilt"] = np.radians(float(q.get("tilt_deg", 0) or 0)); p["thick"] = max(0.02, 3 * vox)
        if ty == "cylinder":
            p["axis"] = q.get("axis", "z") or "z"
            if p["axis"] == "z":
                p["sx"] = p["sy"] = (p["sx"] + p["sy"]) / 2
            else:
                ax = vec(*DIRS["+x" if p["axis"] == "x" else "+y"])
                if abs(ax[1]) > 0.5:
                    p["yaw"] += np.pi / 2; p["sx"], p["sy"] = p["sy"], p["sx"]
                p["axis"] = "x"
        out.append(p)
    return out


def prims_pack(prims):
    v = [0.0]
    for p in prims:
        v += [p["cx"], p["cy"], np.log(p["sx"]), np.log(p["sy"]), np.log(p["sz"])] + ([p["tilt"]] if p["type"] == "plate" else [])
    return np.array(v)


def prims_unpack(prims0, v, lim=None, lid=None, contain=0.9, yaw_free=True):
    """lim: (max |centre| m, max size m) -- an unconstrained optimiser must not grow a primitive without bound.
    lid: the object's cleaned LiDAR points (object frame): every box level contains the central --contain share of the points
    in its own height band along each of its axes (the LiDAR saw the object from far more sides than the camera: its plan
    extent is a lower bound; the camera / free-space terms may only grow a level beyond it)."""
    out = []; k = 1; ids = {p["id"]: i for i, p in enumerate(prims0)}; lc, ls = lim or (1.0, 2.0)
    for p in prims0:
        q = dict(p); q["cx"], q["cy"] = np.clip(v[k:k + 2], -lc, lc); q["sx"], q["sy"], q["sz"] = np.exp(np.clip(v[k + 2:k + 5], -7.0, np.log(ls))); k += 5
        if p["type"] == "plate":
            q["tilt"] = float(np.clip(v[k], -1.2, 1.2)); k += 1
        if p["type"] == "cylinder" and p.get("axis", "z") == "z":
            q["sy"] = q["sx"]
        q["yaw"] = p["yaw"] + (v[0] if yaw_free else 0.0); out.append(q)    # yaw_free 0: the LiDAR min-area footprint rectangle's
    for _ in range(len(out)):                                               # stacking: bottoms follow the supporter's top
        for q in out:
            j = ids.get(q["on"]); q["z0"] = 0.0 if j is None or out[j] is q else out[j]["z0"] + out[j]["sz"]
    if lid is not None:
        tl = (1 - contain) / 2 * 100
        for q in out:
            if q["type"] not in ("cuboid", "wedge"):
                continue
            B = lid[(lid[:, 2] > q["z0"] + 0.2 * q["sz"]) & (lid[:, 2] <= q["z0"] + q["sz"] + 0.01)]
            if len(B) < 30:
                continue
            c, s_ = np.cos(q["yaw"]), np.sin(q["yaw"]); x, y = B[:, 0] - q["cx"], B[:, 1] - q["cy"]; xp, yp = c * x + s_ * y, -s_ * x + c * y
            lo = np.minimum([-q["sx"] / 2, -q["sy"] / 2], [np.percentile(xp, tl), np.percentile(yp, tl)])
            hi = np.maximum([q["sx"] / 2, q["sy"] / 2], [np.percentile(xp, 100 - tl), np.percentile(yp, 100 - tl)])
            m_ = (lo + hi) / 2; q["sx"], q["sy"] = float(hi[0] - lo[0]), float(hi[1] - lo[1])
            q["cx"], q["cy"] = q["cx"] + c * m_[0] - s_ * m_[1], q["cy"] + s_ * m_[0] + c * m_[1]
    return out


def auto_prims(ev, vox):
    """stack of cuboids from the evidence alone: discrete top levels (object_priors.top_levels) of the per-cell top heights,
    each level a footprint rectangle in the object frame standing on the level below."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "object_priors")); from object_priors import top_levels
    E = np.concatenate([ev["Lc"][ev["surf"]], ev["lidar"][ev["lidar"][:, 2] > 0.03]]); cell = 0.02; hmax = ev["hmax"]
    ij = np.floor(E[:, :2] / cell).astype(int); key = ij[:, 0] * 100000 + ij[:, 1]; order = np.argsort(key); ks, st = np.unique(key[order], return_index=True)
    tops = np.array([np.percentile(E[order][a:b, 2], 95) for a, b in zip(st, list(st[1:]) + [len(E)]) if b - a >= 3])
    if len(tops) < 10:
        return None
    lv = top_levels(tops, 0.04, 0.08); lv = lv[lv >= max(0.06, 0.15 * hmax)]; prims = []; z0 = 0.0; rects = []
    for h in lv:                                                            # footprint of each level: points in the upper 3/4 of its own band
        zb = rects[-1][2] if rects else 0.0; P = E[(E[:, 2] > zb + 0.25 * (h - zb)) & (E[:, 2] <= h + 0.01)]
        if len(P) < 30:
            continue
        lo, hi = np.percentile(P[:, :2], 1, 0), np.percentile(P[:, :2], 99, 0)
        if rects and np.abs(np.r_[lo - rects[-1][0], hi - rects[-1][1]]).max() < 0.025:
            rects[-1] = (np.minimum(lo, rects[-1][0]), np.maximum(hi, rects[-1][1]), float(h)); continue   # same box: one level
        rects.append((lo, hi, float(h)))
    for k, (lo, hi, h) in enumerate(rects):
        prims.append(dict(id=f"p{k}", type="cuboid", cx=float((lo[0] + hi[0]) / 2), cy=float((lo[1] + hi[1]) / 2), yaw=0.0, sx=float(max(hi[0] - lo[0], 2 * vox)),
                          sy=float(max(hi[1] - lo[1], 2 * vox)), sz=float(h - z0), on="floor" if not k else f"p{k - 1}", material="m0", z0=z0))
        z0 = float(h)
    return prims or None


def fit_prims(rec, ev, cfg, rng, inits=None):
    from scipy import ndimage
    from scipy.optimize import minimize
    G = ev["G"]; E = ev["Lc"][ev["surf"]]; E = E[rng.choice(len(E), min(4000, len(E)), replace=False)] if len(E) else np.zeros((0, 3))
    Lp = ev["lidar"]; Lp = Lp[Lp[:, 2] > 0.03]; Lp = Lp[rng.choice(len(Lp), min(2000, len(Lp)), replace=False)]
    occ = ev["surf"].copy(); ij, ok = G.index(ev["lidar"]); occ[ij[ok, 0], ij[ok, 1], ij[ok, 2]] = True
    near = ndimage.distance_transform_edt(~occ) * G.vox < 0.10; Fp = ev["Lc"][ev["free"] & near & (ev["Lc"][..., 2] > 0.01)]
    Fp = Fp[rng.choice(len(Fp), min(6000, len(Fp)), replace=False)] if len(Fp) else np.zeros((0, 3))
    hmax = float(np.percentile(np.concatenate([E[:, 2], Lp[:, 2]]), 99.5)) if len(E) + len(Lp) else ev["hmax"]

    Lb = ev["lidar"][ev["lidar"][:, 2] > 0.03] if cfg["lidar_contain"] > 0 else None
    lim = (float(np.max(ev["half"])) + 0.05, 1.3 * float(max(2 * np.max(ev["half"]), ev["hmax"])) + 0.02)

    def cost(v, p0, v0):
        pr = prims_unpack(p0, v, lim, Lb, cfg["lidar_contain"] or 0.9, bool(cfg["yaw_free"])); c = 0.0
        if len(E):
            c += 0.7 * np.minimum(np.abs(prims_sdf(pr, E)[0]), 0.05).mean()
        if len(Lp):
            c += 0.3 * np.minimum(np.abs(prims_sdf(pr, Lp)[0]), 0.05).mean()
        if len(Fp):
            c += 2.0 * np.minimum(np.maximum(-prims_sdf(pr, Fp)[0], 0), 0.05).mean()
        return c + 0.003 * float(((v[1:] - v0[1:]) ** 2).mean())             # unobservable dimensions stay at the recipe's
    asp = min(ev["half"]) / max(ev["half"]); inits = [(cost(prims_pack(p0), p0, prims_pack(p0)), p0, prims_pack(p0)) for p0 in (inits or [])]
    for swap in ((False, True) if asp > 0.7 else (False,)) if rec is not None else ():
        for fx in (1, -1):
            for fy in (1, -1):
                p0 = prims_init(rec, ev["half"], hmax, fx, fy, swap, G.vox)
                if p0 is None:
                    return None
                v0 = prims_pack(p0); inits.append((cost(v0, p0, v0), p0, v0))
    inits.sort(key=lambda x: x[0]); best = None
    for c0, p0, v0 in inits[:3]:
        r = minimize(cost, v0, args=(p0, v0), method="Powell", options=dict(maxfev=400 * len(v0), xtol=1e-3, ftol=1e-5))
        if best is None or r.fun < best[0]:
            best = (float(r.fun), prims_unpack(p0, r.x, lim, Lb, cfg["lidar_contain"] or 0.9, bool(cfg["yaw_free"])), float(c0))
    pr = best[1]; res = float(np.minimum(np.abs(prims_sdf(pr, E)[0]), 0.05).mean()) if len(E) else None
    return dict(prims=pr, cost=round(best[0], 4), init_cost=round(best[2], 4), resid_m=None if res is None else round(res, 4),
                free_pen_m=round(float(np.maximum(-prims_sdf(pr, Fp)[0], 0).mean()), 4) if len(Fp) else 0.0)


def prim_field(prims, ev, cfg):
    from scipy import ndimage
    G = ev["G"]; P = ev["Lc"].reshape(-1, 3); f = np.concatenate([prims_sdf(prims, P[i:i + 2000000])[0] for i in range(0, len(P), 2000000)]).reshape(G.shape)
    inside = f < 0; deep = ev["free"] & (f < -cfg["carve_tol_m"]); carved = np.zeros_like(inside)
    if deep.any():                                                          # the recipe claims space the cameras saw through: cut it out
        carved = ndimage.binary_propagation(deep, mask=ev["free"] & inside); f = np.where(carved, G.vox, f)
    f = np.where(ev["foot"], f, np.maximum(f, G.vox)); f[:, :, 0] = np.maximum(f[:, :, 0], G.vox)
    return f, float(carved.sum() / max(inside.sum(), 1))


# ------------------------------------------------------------------ generated asset
def fit_asset(files, ev, cfg, rng):
    from scipy import ndimage
    from scipy.optimize import minimize
    G = ev["G"]; E = ev["Lc"][ev["surf"]]; E = E[rng.choice(len(E), min(4000, len(E)), replace=False)] if len(E) else np.zeros((0, 3))
    Lp = ev["lidar"]; Lp = Lp[Lp[:, 2] > 0.03]; Lp = Lp[rng.choice(len(Lp), min(2000, len(Lp)), replace=False)]
    occ = ev["surf"].copy(); ij, ok = G.index(ev["lidar"]); occ[ij[ok, 0], ij[ok, 1], ij[ok, 2]] = True
    dE = np.minimum(ndimage.distance_transform_edt(~occ) * G.vox, 0.05).astype(np.float32); cap = 0.05
    hev = float(np.percentile(np.concatenate([E[:, 2], Lp[:, 2]]), 99.5)); best = None; cands = []
    for f in files:
        g = np.load(f); X = g["xyz"].astype(np.float64)[g["opacity"] > 0.3]
        if len(X) < 200:
            continue
        z0, z1 = np.percentile(X[:, 2], [0.5, 99.5]); off = np.r_[(np.percentile(X[:, :2], 1, 0) + np.percentile(X[:, :2], 99, 0)) / 2, z0]
        res = 128; glo = -0.64; cell = 1.28 / res; ai = np.clip(np.floor((X - glo) / cell).astype(int), 0, res - 1)
        sh = np.zeros((res,) * 3, bool); sh[ai[:, 0], ai[:, 1], ai[:, 2]] = True; dA = (ndimage.distance_transform_edt(~sh) * cell).astype(np.float32)
        Gs = X[rng.choice(len(X), min(5000, len(X)), replace=False)]; sc0 = hev / max(z1 - z0, 1e-3)

        def tf(p):
            S = sc0 * np.exp(p[1:4]); c, s_ = np.cos(p[0]), np.sin(p[0]); return S, c, s_, np.array([p[4], p[5], 0.0])

        def cost(p):
            S, c, s_, tt = tf(p); sm = float(np.exp(np.log(S).mean())); tot = 0.0
            for Q, w in ((E, 0.7), (Lp, 0.3)):
                if not len(Q):
                    continue
                q = Q - tt; gq = np.stack([c * q[:, 0] + s_ * q[:, 1], -s_ * q[:, 0] + c * q[:, 1], q[:, 2]], 1) / S + off
                gi = np.floor((gq - glo) / cell).astype(int); out = np.any((gi < 0) | (gi >= res), 1); gi = np.clip(gi, 0, res - 1)
                tot += w * np.minimum(np.where(out, cap, dA[gi[:, 0], gi[:, 1], gi[:, 2]] * sm), cap).mean()
            q = (Gs - off) * S; Lq = np.stack([c * q[:, 0] - s_ * q[:, 1], s_ * q[:, 0] + c * q[:, 1], q[:, 2]], 1) + tt
            ij, ok = G.index(Lq); pen = np.where(ok, ev["free"][ij[:, 0], ij[:, 1], ij[:, 2]] * dE[ij[:, 0], ij[:, 1], ij[:, 2]], cap)
            return tot + 2.0 * pen.mean() + 0.01 * np.abs(p[1:4] - p[1:4].mean()).sum()
        starts = sorted(((cost(np.array([y, 0, 0, 0, 0, 0.0])), y) for y in np.radians(np.arange(0, 360, 10))))[:3]; fb = None
        for _, y in starts:
            r = minimize(cost, np.array([y, 0, 0, 0, 0, 0.0]), method="Powell", bounds=[(y - 0.3, y + 0.3)] + [(-0.5, 0.5)] * 3 + [(-0.15, 0.15)] * 2,
                         options=dict(maxfev=1500, xtol=1e-3, ftol=1e-5))
            if fb is None or r.fun < fb[0]:
                fb = (float(r.fun), r.x)
        S, c, s_, tt = tf(fb[1]); cands.append(dict(file=os.path.basename(f), cost=round(fb[0], 4), scale_xyz=[round(float(x), 3) for x in S / sc0], yaw_deg=round(float(np.degrees(fb[1][0])) % 360, 1)))
        if best is None or fb[0] < best[0]:
            best = (fb[0], f, fb[1], off, sc0)
    if best is None:
        return None
    _, f, p, off, sc0 = best; g = np.load(f); k = g["opacity"] > 0.3; S = sc0 * np.exp(p[1:4]); c, s_ = np.cos(p[0]), np.sin(p[0])
    q = (g["xyz"].astype(np.float64)[k] - off) * S; Lq = np.stack([c * q[:, 0] - s_ * q[:, 1], s_ * q[:, 0] + c * q[:, 1], q[:, 2]], 1) + np.array([p[4], p[5], 0.0])
    from scipy.spatial import cKDTree
    res = float(np.minimum(cKDTree(Lq).query(E)[0], 0.05).mean()) if len(E) else None
    return dict(L=Lq, rgb=np.clip(0.5 + C0 * g["dc"].astype(np.float64)[k], 0, 1), spacing=float(np.exp(np.log(S).mean())) / 64, chosen=os.path.basename(f), cands=cands,
                cost=round(best[0], 4), resid_m=None if res is None else round(res, 4), scale_xyz=[round(float(x), 3) for x in S / sc0])


def ball(r):
    k = int(np.ceil(r)); g = np.mgrid[-k:k + 1, -k:k + 1, -k:k + 1]
    return (g ** 2).sum(0) <= r * r + 1e-6


def solid_fill(o):
    from scipy import ndimage
    p = np.pad(o, 1); p[:, :, 0] = True                                     # the floor closes the bottom
    return ndimage.binary_fill_holes(p)[1:-1, 1:-1, 1:-1]


def gen_field(al, ev, cfg):
    """occupancy of the registered asset fused with the measurements -> field (0.5 - smoothed occupancy)."""
    from scipy import ndimage
    G = ev["G"]; free = ev["free"]; z_ok = ev["Lc"][..., 2] > 0; info = {}
    if al is not None:
        pr = np.zeros(G.shape, bool); ij, ok = G.index(al["L"]); pr[ij[ok, 0], ij[ok, 1], ij[ok, 2]] = True
        r = max(1.0, np.ceil(1.2 * al["spacing"] / G.vox)); pr = ndimage.binary_closing(np.pad(pr, 4), structure=ball(r))[4:-4, 4:-4, 4:-4]
        pr = solid_fill(pr); info["prior_in_free"] = round(float((pr & free).sum() / max(pr.sum(), 1)), 3)
        ds = ndimage.distance_transform_edt(~ev["surf"]) * G.vox; dp = ndimage.distance_transform_edt(~pr) * G.vox
        occ = ((pr | ((ds <= cfg["crust_m"]) & (dp <= cfg["crust_m"]))) & ~free)
    else:                                                                   # no usable prior: what is not free inside the evidence box
        ev_occ = ev["surf"].copy(); ij, ok = G.index(ev["lidar"][ev["lidar"][:, 2] > 0.03]); ev_occ[ij[ok, 0], ij[ok, 1], ij[ok, 2]] = True
        ii = np.argwhere(ev_occ); lo, hi = np.percentile(ii, 1, 0).astype(int), np.percentile(ii, 99, 0).astype(int) + 1
        occ = np.zeros(G.shape, bool); occ[lo[0]:hi[0], lo[1]:hi[1], 0:hi[2]] = True; occ &= ~free; info["hull"] = True
    occ &= z_ok & ev["foot"]
    low = np.argmax(occ, axis=2); has = occ.any(2) & (low * G.vox + G.lo[2] < cfg["rest_m"]); zi = np.arange(G.shape[2])[None, None, :]
    occ |= has[..., None] & (zi <= low[..., None]) & z_ok                    # no gap under the object (the surface evidence starts above the contact line)
    occ = ndimage.binary_closing(np.pad(occ, 2), structure=ball(1.5))[2:-2, 2:-2, 2:-2] & ~free & z_ok
    occ = solid_fill(occ) & ~free & z_ok & ev["foot"]; occ = ndimage.binary_opening(occ, structure=ball(1.0))
    lab, nl = ndimage.label(occ)
    if nl > 1:
        sz = ndimage.sum(occ, lab, np.arange(1, nl + 1)); occ = np.isin(lab, 1 + np.nonzero(sz >= 0.05 * sz.max())[0])
    info["volume_l"] = round(float(occ.sum()) * G.vox ** 3 * 1000, 2)
    return 0.5 - ndimage.gaussian_filter(occ.astype(np.float32), 0.8), info


# ------------------------------------------------------------------ mesh
def plane_snap(V, N, vox, rng, max_planes=14, min_frac=0.05):
    """flat faces stay flat: dominant planes (1-point RANSAC on vertex normals), vertices near one are projected onto it."""
    V = V.copy(); rem = np.ones(len(V), bool); n_ = 0
    for _ in range(max_planes):
        idx = np.nonzero(rem)[0]
        if len(idx) < min_frac * len(V):
            break
        best = None
        for i in rng.choice(idx, min(150, len(idx)), replace=False):
            m = rem & (np.abs((V - V[i]) @ N[i]) < 1.5 * vox) & (N @ N[i] > 0.9); c = int(m.sum())
            if best is None or c > best[0]:
                best = (c, m)
        if best[0] < min_frac * len(V):
            break
        m = best[1]; c = V[m].mean(0); nn = np.linalg.svd(V[m] - c, full_matrices=False)[2][2]; nn = nn if nn @ N[m].mean(0) > 0 else -nn
        m = rem & (np.abs((V - c) @ nn) < 1.5 * vox) & (N @ nn > 0.85); V[m] -= np.outer((V[m] - c) @ nn, nn); rem &= ~m; n_ += 1
    return V, n_


def field_mesh(field, ev, cfg, rng, snap):
    import trimesh
    from skimage import measure
    G = ev["G"]; f = np.pad(field, 1, constant_values=float(np.abs(field).max()) or 1.0)
    if not ((f < 0).any() and (f > 0).any()):
        return None
    v, fc, _, _ = measure.marching_cubes(f, 0.0); V = G.lo + (v - 1 + 0.5) * G.vox
    m = trimesh.Trimesh(V, fc, process=True); parts = m.split(only_watertight=False)
    if len(parts) > 1:
        ar = [p.area for p in parts]; m = trimesh.util.concatenate([p for p, a_ in zip(parts, ar) if a_ >= 0.05 * max(ar)])
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    nsnap = 0
    if snap:
        trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=8)
        Vs, nsnap = plane_snap(np.asarray(m.vertices), np.asarray(m.vertex_normals), G.vox, rng); m = trimesh.Trimesh(Vs, m.faces, process=False)
    for _ in range(int(max(0, round(np.log2(max(G.vox / cfg["texel_m"], 1)) + 0.25)))):
        if len(m.faces) > 600000:                                            # memory / texture budget
            break
        m = m.subdivide()
    V = np.asarray(m.vertices); V[:, 2] = np.maximum(V[:, 2], 0.0); m = trimesh.Trimesh(V, m.faces, process=False)
    return m, nsnap


# ------------------------------------------------------------------ rendering + texture
def ray_scene(Vd, F):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(Vd.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return sc


def cast(sc, O, Dn):
    import open3d as o3d
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([O, Dn], 1).astype(np.float32)))
    return r["t_hit"].numpy(), r["primitive_ids"].numpy(), r["primitive_uvs"].numpy()


def render(sc, F, Cv, cam, bg=None, floor=None):
    """vertex-colour render of the ray scene -> rgb (H,W,3), mask. floor = (n_d, d_d, colour): the floor plane where no mesh is hit."""
    W, H = cam["W"], cam["H"]; xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(xs - cam["cx"]) / cam["fx"], (ys - cam["cy"]) / cam["fy"], np.ones_like(xs)], -1).reshape(-1, 3)
    dw = dc @ cam["R"]; dw /= np.linalg.norm(dw, axis=1, keepdims=True); O = np.broadcast_to(cam["C"], dw.shape)
    th, pid, uv = cast(sc, O, dw); hit = np.isfinite(th); img = np.zeros((H * W, 3), np.float32) + (0.2 if bg is None else 0)
    if floor is not None:
        nd, dd, col, cell = floor; den = dw @ nd; tt = -(cam["C"] @ nd + dd) / np.where(np.abs(den) > 1e-9, den, 1e-9); fl = (tt > 0) & (np.abs(den) > 1e-9)
        P = O[fl] + dw[fl] * tt[fl, None]; chk = ((np.floor(P[:, 0] / cell) + np.floor(P[:, 1] / cell) + np.floor(P[:, 2] / cell)) % 2) * 0.04
        img[fl] = np.asarray(col) - chk[:, None]
    if bg is not None:
        img = bg.reshape(-1, 3).copy()
    f = F[pid[hit]]; u, v = uv[hit, 0], uv[hit, 1]
    img[hit] = Cv[f[:, 0]] * (1 - u - v)[:, None] + Cv[f[:, 1]] * u[:, None] + Cv[f[:, 2]] * v[:, None]
    return img.reshape(H, W, 3), hit.reshape(H, W), np.where(hit, th, 0).reshape(H, W)


def bilinear(img, x, y):
    H, W = img.shape[:2]; x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001); x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy).astype(np.float32)


def bake(Vd, Nd, F, cams, depth_dir, s, cfg, height=None):
    """real photos onto the vertices the cameras saw -> colour (V,3), seen (V,), best quality (px / m)."""
    import cv2
    sc = ray_scene(Vd, F); nv = len(Vd); K = cfg["top_k"]; tw = np.zeros((nv, K), np.float32); tc = np.zeros((nv, K, 3), np.float32); tf_ = -np.ones((nv, K), np.int32)
    eps = 0.004 / s; sharp = []
    for fi, c in enumerate(cams):
        img = load_img(c); g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        u, v, r, ok = project(c, Vd); ok &= (u >= 2) & (u < c["W"] - 2) & (v >= 2) & (v < c["H"] - 2)
        dv = c["C"] - Vd; cos = (Nd * dv).sum(1) / np.maximum(r, 1e-9); ok &= cos > cfg["cos_min"]; i = np.nonzero(ok)[0]
        if not len(i):
            continue
        th, _, _ = cast(sc, np.broadcast_to(c["C"], (len(i), 3)), -dv[i] / r[i, None]); vis = th >= r[i] - eps
        SD = depth_of(depth_dir, c["name"])
        if SD is not None:                                                  # the photo shows THIS surface there (not floor / background through a too-wide mesh)
            sd = SD[np.floor(v[i]).astype(int), np.floor(u[i]).astype(int)]; vis &= (sd > 0) & (np.abs(sd - r[i]) < np.maximum(cfg["photo_tol_m"] / s, 0.03 * r[i]))
            if height is not None and cfg["floor_pixel_m"] > 0:              # the pixel's own surface point is floor: not this object's colour
                Pp = c["C"] + (Vd[i] - c["C"]) * (sd / np.maximum(r[i], 1e-9))[:, None]; vis &= height(Pp) > cfg["floor_pixel_m"]
        i = i[vis]
        if not len(i):
            continue
        col = bilinear(img, u[i] - 0.5, v[i] - 0.5)
        w = (c["fx"] * cos[i] / (r[i] * s)).astype(np.float32)
        j = tw[i].argmin(1); b = w > tw[i, j]; ii, jj = i[b], j[b]; tw[ii, jj] = w[b]; tc[ii, jj] = col[b]; tf_[ii, jj] = fi
    q = tw.max(1); seen = q >= cfg["q_min_px_m"]
    sh = np.asarray(sharp); sh = np.clip(sh / max(np.median(sh), 1e-9), 0.5, 1.5) if len(sh) else sh
    val = tf_ >= 0; w = tw * np.where(val, sh[np.clip(tf_, 0, None)], 0); lum = tc.mean(2) + 1e-3; gain = np.ones(len(cams))
    for _ in range(3):                                                      # per-frame exposure gains from the overlaps
        gl = lum * gain[np.clip(tf_, 0, None)]; ref = (w * gl).sum(1) / np.maximum(w.sum(1), 1e-9)
        for fi in range(len(cams)):
            m = val & (tf_ == fi) & seen[:, None]
            if m.sum() > 50:
                gain[fi] = float(np.clip(np.exp(np.median(np.log(np.broadcast_to(ref[:, None], lum.shape)[m] / lum[m]))), 0.7, 1.4))
        gain /= np.exp(np.log(gain).mean())
    tcg = tc * gain[np.clip(tf_, 0, None)][..., None]; jb = w.argmax(1); cb = tcg[np.arange(nv), jb]
    cons = np.abs(tcg - cb[:, None]).max(2) < 0.2; ww = np.where(val & cons, w, 0) ** cfg["sharp_pow"]
    col = (ww[..., None] * tcg).sum(1) / np.maximum(ww.sum(1), 1e-30)[:, None]
    return np.clip(col, 0, 1), seen, q, gain


def cam_visible(Vd, Nd, F, cams, cfg, s):
    """per vertex: number of training views that see it unoccluded by the mesh itself (inside the image, facing the camera).
    Geometric only (no photo-consistency): what any camera COULD see is the splat's job; the rest is completed."""
    sc = ray_scene(Vd, F); cnt = np.zeros(len(Vd), np.int32); eps = 0.004 / s
    for c in cams:
        u, v, r, ok = project(c, Vd); ok &= (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"])
        dv = c["C"] - Vd; ok &= (Nd * dv).sum(1) / np.maximum(r, 1e-9) > cfg["cos_min"]; i = np.nonzero(ok)[0]
        if not len(i):
            continue
        th, _, _ = cast(sc, np.broadcast_to(c["C"], (len(i), 3)), -dv[i] / r[i, None]); cnt[i[th >= r[i] - eps]] += 1
    return cnt


def charts_of(prims, L, Nl):
    """per vertex: chart id (primitive face), (u, v) in metres on it, orientation class, material."""
    _, D = prims_sdf(prims, L); pi = np.abs(D).argmin(1); cid = np.zeros(len(L), int); uv = np.zeros((len(L), 2)); cls = np.zeros(len(L), int); mats = {}
    for k, p in enumerate(prims):
        m = pi == k
        if not m.any():
            continue
        c, s_ = np.cos(p["yaw"]), np.sin(p["yaw"]); ex, ey, ez = np.array([c, s_, 0]), np.array([-s_, c, 0]), np.array([0, 0, 1.0])
        if p["type"] == "cuboid":
            nf = [ex, -ex, ey, -ey, ez]
        elif p["type"] == "wedge":
            sl = (p["sz"] * ex + p["sx"] * ez); nf = [sl / np.linalg.norm(sl), -ex, ey, -ey]
        elif p["type"] == "plate":
            en = np.cos(p["tilt"]) * ey - np.sin(p["tilt"]) * ez; el = np.sin(p["tilt"]) * ey + np.cos(p["tilt"]) * ez; nf = [en, -en, ex, -ex, el]
        else:
            nf = [ez] if p.get("axis", "z") == "z" else [ex, -ex]
        nf = np.array(nf); sc_ = Nl[m] @ nf.T; j = sc_.argmax(1); side = np.zeros(m.sum(), bool)
        if p["type"] == "cylinder":
            side = sc_.max(1) < 0.7; j = np.where(side, len(nf), j)
        idx = np.nonzero(m)[0]
        for jj in np.unique(j):
            q = idx[j == jj]; key = 100 * k + int(jj); cid[q] = key; mats[key] = p["material"]
            if jj == len(nf):                                               # cylinder side: arc length x axial coordinate
                xp, yp, z = prim_local(p, L[q])
                if p.get("axis", "z") == "z":
                    uv[q] = np.stack([p["sx"] / 2 * np.arctan2(yp, xp), L[q, 2]], 1); cls[q] = 1
                else:
                    uv[q] = np.stack([xp, p["sz"] / 2 * np.arctan2(z - p["sz"] / 2, yp)], 1); cls[q] = 2
                continue
            n_ = nf[jj]
            if abs(n_[2]) < 0.9:
                a_ = np.cross(ez, n_); a_ /= np.linalg.norm(a_)
            else:
                a_ = ex
            b_ = np.cross(n_, a_); uv[q] = np.stack([L[q] @ a_, L[q] @ b_], 1); cls[q] = 0 if abs(n_[2]) > 0.9 else (1 if abs(n_[2]) < 0.3 else 2)
    return cid, uv, cls, mats


def fill_same_material(col, seen, cid, uv, cls, mats, texel):
    """unseen vertices take the real texture of a seen face of the same material (same height on vertical faces, tiled)."""
    from scipy import ndimage
    col = col.copy(); T = {}; info = {}
    for c in np.unique(cid):
        m = cid == c; lo = uv[m].min(0); hi = uv[m].max(0); sh = np.clip(np.ceil((hi - lo) / texel).astype(int) + 1, 1, 1500)
        ms = m & seen; img = np.zeros(tuple(sh) + (3,), np.float32); cnt = np.zeros(tuple(sh), np.float32)
        if ms.any():
            ij = np.minimum(np.floor((uv[ms] - lo) / texel).astype(int), sh - 1); np.add.at(img, (ij[:, 0], ij[:, 1]), col[ms]); np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
            ok = cnt > 0; img[ok] /= cnt[ok][:, None]; ind = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True); img = img[ind[0], ind[1]]
        T[c] = dict(lo=lo, hi=hi, sh=sh, img=img, cov=float(seen[m].mean()), n=int(m.sum()), cls=int(np.bincount(cls[m]).argmax()), ok=cnt > 0)
    for c, t in T.items():
        m = (cid == c) & ~seen
        if not m.any():
            continue
        cand = [(10 * (T[k]["cls"] == t["cls"]) + 2 * (k // 100 == c // 100) + (k != c) + T[k]["cov"], k) for k in T
                if mats[k] == mats[c] and T[k]["cov"] >= 0.5 and T[k]["n"] >= 50]
        if not cand:
            sm = seen & np.isin(cid, [k for k in T if mats[k] == mats[c]]); sm = sm if sm.any() else seen
            col[m] = col[sm].mean(0) if sm.any() else 0.5; info[int(c)] = "mean colour of the material"; continue
        k = max(cand)[1]; D = T[k]; du = uv[m] - t["lo"]; ext = (D["sh"] - 1) * texel
        if k == c:
            du = np.stack([(t["hi"][0] - t["lo"][0]) - du[:, 0], du[:, 1]], 1)   # own seen part, mirrored
        ij = np.floor(np.mod(du, np.maximum(ext, texel)) / texel).astype(int); ij = np.minimum(ij, D["sh"] - 1)
        col[m] = D["img"][ij[:, 0], ij[:, 1]]; info[int(c)] = f"face {k} (coverage {D['cov']:.2f})"
    return col, info


def fill_synth(col, seen, q, cid, uv, cls, mats, texel, cfg, rng, hgt):
    """generic texture completion for primitive candidates (replaces copying one seen face, which carried its shadows /
    grazing-view smears onto the back): per material,
      1. reliable seen texels = photo quality >= synth_q_mult x q_min AND colour within synth_z robust sigmas of the
         material's seen colours (dark smears, floor pixels at the contact line, neighbours' colours are outliers);
         everything else is completed as if unseen (returned `seen` flag shrinks accordingly)
      2. exemplar per chart (primitive face) from its reliable texels; target texels are filled patch by patch
         (synth_patch_m) from a random offset in the best exemplar of the same material and orientation class
         (vertical faces keep their height: brick rows stay aligned), so the texture keeps its structure without
         repeating one copied face
      3. each completed chart's brightness follows the reliable faces of the same material + orientation class."""
    from scipy import ndimage
    col = col.copy(); info = {}; good = seen & (q >= cfg["synth_q_mult"] * cfg["q_min_px_m"]); keep = seen.copy()
    mat_v = np.array([mats.get(int(c), "m0") for c in cid])
    low = hgt < cfg["synth_contact_m"]                                     # contact band: photographs as floor / shadow
    for mt in np.unique(mat_v):
        m = (mat_v == mt) & good; m = m & ~low if (m & ~low).sum() >= 50 else m
        if m.sum() < 50:
            continue
        med = np.median(col[m], 0); mad = 1.4826 * np.median(np.abs(col[m] - med), 0) + 0.02; z_ = (np.abs(col - med) / mad).max(1)
        out = (mat_v == mt) & seen & ((z_ > cfg["synth_z"]) | (low & (z_ > cfg["synth_z"] / 2)))
        keep &= ~out; good &= ~out
    T = {}
    for c in np.unique(cid):
        m = cid == c; lo = uv[m].min(0); hi = uv[m].max(0); sh = np.clip(np.ceil((hi - lo) / texel).astype(int) + 1, 1, 1500)
        mg = m & good; img = np.zeros(tuple(sh) + (3,), np.float32); cnt = np.zeros(tuple(sh), np.float32)
        if mg.any():
            ij = np.minimum(np.floor((uv[mg] - lo) / texel).astype(int), sh - 1); np.add.at(img, (ij[:, 0], ij[:, 1]), col[mg]); np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
        ok = cnt > 0; img[ok] /= cnt[ok][:, None]
        if ok.any():
            ind = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True); img = img[ind[0], ind[1]]
        T[int(c)] = dict(lo=lo, hi=hi, sh=sh, img=img, ok=ok, cov=float(good[m].mean()), n=int(m.sum()), cls=int(np.bincount(cls[m]).argmax()),
                         lum=float(np.median(col[mg].mean(1))) if mg.sum() > 20 else None)
    P = max(2, int(round(cfg["synth_patch_m"] / texel)))
    for c, t in T.items():
        tm = (cid == c) & ~keep
        if not tm.any():
            continue
        mc = mats.get(c, "m0")
        cand = [(5 * (T[k]["cls"] == t["cls"]) + T[k]["cov"], k) for k in T if mats.get(k, "m0") == mc and T[k]["cov"] >= 0.3 and T[k]["n"] >= 50 and T[k]["ok"].mean() > 0.2]
        if not cand:
            sm = good & (mat_v == mc); sm = sm if sm.any() else keep
            col[tm] = np.median(col[sm], 0) if sm.any() else 0.5; info[c] = "median colour of the material"; continue
        k = max(cand)[1]; S = T[k]; tij = np.floor((uv[tm] - t["lo"]) / texel).astype(int); pid = tij // P
        src = np.zeros_like(tij)
        for pp in np.unique(pid, axis=0):                                   # one random source offset per target patch
            sel = np.all(pid == pp, 1); best = None
            for _ in range(8):
                ou = rng.integers(0, max(1, S["sh"][0] - P)); ov = rng.integers(0, max(1, S["sh"][1] - P))
                if t["cls"] in (1, 2):                                      # vertical / slanted: keep the height (rows stay aligned)
                    ov = int(np.clip(pp[1] * P + int(round((t["lo"][1] - S["lo"][1]) / texel)), 0, max(0, S["sh"][1] - P)))
                fr_ = S["ok"][ou:ou + P, ov:ov + P].mean() if S["ok"].size else 0
                if best is None or fr_ > best[0]:
                    best = (fr_, ou, ov)
            src[sel] = np.stack([best[1] + tij[sel, 0] - pp[0] * P, best[2] + tij[sel, 1] - pp[1] * P], 1)
        src = np.clip(src, 0, S["sh"] - 1); fill = S["img"][src[:, 0], src[:, 1]]
        ref = [T[j]["lum"] for j in T if mats.get(j, "m0") == mc and T[j]["cls"] == t["cls"] and T[j]["lum"] is not None]
        if ref and S["lum"]:
            fill = np.clip(fill * float(np.clip(np.median(ref) / max(S["lum"], 1e-3), 0.7, 1.4)), 0, 1)
        col[tm] = fill; info[c] = f"synth from face {k} (reliable coverage {S['cov']:.2f}, patch {P * texel:.2f} m)"
    return col, info, keep


def seam_blend(V, col, seen, seam):
    from scipy.spatial import cKDTree
    if seen.all() or not seen.any():
        return col
    d, i = cKDTree(V[seen]).query(V[~seen]); a_ = np.clip(d / seam, 0, 1)[:, None]; out = col.copy()
    out[~seen] = (1 - a_) * col[seen][i] + a_ * col[~seen]
    return out


def gen_colours(V, col, seen, al):
    """unseen vertices: the generated asset's colour, matched to the real colours where both exist; else the nearest seen colour."""
    from scipy.spatial import cKDTree
    out = col.copy(); un = ~seen
    if not un.any():
        return out, "all seen"
    near = col[seen][cKDTree(V[seen]).query(V[un])[1]] if seen.any() else np.full((un.sum(), 3), 0.5)
    if al is None:
        out[un] = near; return out, "nearest seen colour"
    d, i = cKDTree(al["L"]).query(V, k=4); g = al["rgb"][i].mean(1); dg = d[:, 0]; m = seen & (dg < 0.015)
    if m.sum() > 100:
        gn = np.clip(col[m].std(0) / np.maximum(g[m].std(0), 1e-3), 0.5, 2.0); g = np.clip((g - g[m].mean(0)) * gn + col[m].mean(0), 0, 1)
    w = np.clip((0.04 - dg[un]) / 0.02, 0, 1)[:, None]; out[un] = w * g[un] + (1 - w) * near
    return out, f"generated texture colour-matched on {int(m.sum())} seen vertices"


def obj_mask(cam, SD, ev, s, R, t):
    ys, xs = np.mgrid[0:cam["H"], 0:cam["W"]]
    ray = np.stack([(xs + 0.5 - cam["cx"]) / cam["fx"], (ys + 0.5 - cam["cy"]) / cam["fy"], np.ones_like(xs, float)], -1); ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    Pl = ev["fr"].loc(s * (((ray * SD[..., None]).reshape(-1, 3) - cam["t"]) @ cam["R"]) @ R.T + t).reshape(cam["H"], cam["W"], 3)
    return (np.abs(Pl[..., 0]) < ev["half"][0] + 0.04) & (np.abs(Pl[..., 1]) < ev["half"][1] + 0.04) & (Pl[..., 2] > 0.015) & (Pl[..., 2] < ev["hmax"] + 0.04) & (SD > 0)


def numeric(sc, F, C, cams, depth_dir, ev, s, R, t):
    iou, ps = [], []
    for c in cams:
        SD = depth_of(depth_dir, c["name"])
        if SD is None:
            continue
        img, mk, _ = render(sc, F, C, c); real = load_img(c); rm = obj_mask(c, SD, ev, s, R, t)
        if rm.sum() < 200:
            continue
        iou.append(float((mk & rm).sum() / max((mk | rm).sum(), 1))); b = mk & rm
        if b.sum() > 100:
            g = real[b].mean() / max(img[b].mean(), 1e-3); ps.append(float(-10 * np.log10(max(((np.clip(img[b] * g, 0, 1) - real[b]) ** 2).mean(), 1e-8))))
    return dict(iou=round(float(np.mean(iou)), 3) if iou else None, psnr=round(float(np.mean(ps)), 2) if ps else None, frames=len(iou))


def novel_cams(ev, cam0, s, R, t, hs=(0.45, 0.45, 0.45, 1.3)):
    """cameras on the side the robot did not see (opposite the mean training azimuth), +-70 deg, and from above."""
    fr = ev["fr"]; back = float(np.arctan2(np.sin(ev["az"]).mean(), np.cos(ev["az"]).mean())) + np.pi if len(ev["az"]) else 0.0
    rad = float(np.linalg.norm(np.r_[ev["half"], ev["hmax"] / 2])); out = []
    for da, h in zip((0.0, 1.2, -1.2, 0.4), hs):
        th = back + da; dist = 1.3 * rad + 0.7; eye = fr.met(np.array([[dist * np.cos(th), dist * np.sin(th), h]]))[0]; tgt = fr.met(np.array([[0, 0, ev["hmax"] / 2]]))[0]
        ed, td = ((eye - t) @ R) / s, ((tgt - t) @ R) / s; up = R.T @ fr.n; z = td - ed; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
        Rc = np.stack([x, y, z]); out.append(dict(cam0, R=Rc, t=-Rc @ ed, C=ed, name=f"novel{len(out)}"))
    return out


def crop_to(img, mk, pad=1.5, size=320):
    import cv2
    ys, xs = np.nonzero(mk)
    if not len(xs):
        return cv2.resize(img, (size, size))
    S = pad * max(np.ptp(xs), np.ptp(ys), 20); cx, cy = xs.mean(), ys.mean(); M = np.array([[size / S, 0, size / 2 - cx * size / S], [0, size / S, size / 2 - cy * size / S]])
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderValue=(0.2, 0.2, 0.2))


def label(img, txt):
    import cv2
    im = np.ascontiguousarray(np.clip(img * 255, 0, 255).astype(np.uint8)); cv2.putText(im, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return im


def save_cand(path, m, C, seen, fr, extra):
    np.savez_compressed(path, V=np.asarray(m.vertices, np.float32), F=np.asarray(m.faces, np.int32), C=C.astype(np.float32), N=np.asarray(m.vertex_normals, np.float32),
                        seen=seen, meta=json.dumps(extra, default=str), **{f"fr_{k}": np.asarray(v) for k, v in fr.dump().items()})


def candidates(a):
    import cv2
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); to_ds = lambda X: ((X - t) @ R) / s; cams = {c["name"]: c for c in load_cams(a.dataset)}
    evj = json.load(open(os.path.join(a.dir, "evidence.json"))); os.makedirs(os.path.join(a.dir, "cand"), exist_ok=True); rep = {}
    nf = np.array(evj["floor"]["n"]); dfl = float(evj["floor"]["d"]); nd = R.T @ nf; dd = (nf @ t + dfl) / s
    for o in evj["objects"]:
        oid = o["id"]
        if a.ids and oid not in a.ids:
            continue
        t0 = time.time(); rng = np.random.default_rng(0); ev = load_ev(a, oid, cfg); fr = ev["fr"]; r = rep[oid] = {"box_m": o["box_m"]}
        vc = [cams[n_] for n_ in ev["views"] if n_ in cams]; val = vc[cfg["val_every"] // 2::cfg["val_every"]]; vn = {c["name"] for c in val}; trn = [c for c in vc if c["name"] not in vn] or vc
        fields = {}
        rf = os.path.join(a.dir, "vlm", f"recipe_obj{oid}.json")
        if os.path.exists(rf):
            rec = json.load(open(rf)); r["recipe"] = {k: rec.get(k) for k in ("summary", "detection", "confidence", "symmetry")}
            r["recipe"]["primitives"] = [f"{p.get('type')}:{p.get('material')}" for p in rec.get("primitives", [])]
            if rec.get("primitives") and float(rec.get("confidence", 0) or 0) >= cfg["min_prim_confidence"]:
                try:
                    pf = fit_prims(rec, ev, cfg, rng)
                except (KeyError, ValueError, TypeError, IndexError) as e:
                    pf = None; r["prim_error"] = f"recipe not usable: {type(e).__name__} {e}"
                if pf is not None:
                    f, carved = prim_field(pf["prims"], ev, cfg); fields["prim"] = (f, pf)
                    r["prim"] = dict(cost=pf["cost"], init_cost=pf["init_cost"], resid_m=pf["resid_m"], free_pen_m=pf["free_pen_m"], carved_share=round(carved, 3),
                                     prims=[{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()} for p in pf["prims"]])
        if "prim" not in fields:                                            # no (usable) VLM recipe: stacked boxes from the evidence alone, kept only if box-like
            ap0 = auto_prims(ev, ev["G"].vox)
            pf = fit_prims(None, ev, cfg, rng, inits=[ap0]) if ap0 else None
            if pf is not None:
                r["prim"] = dict(source="auto stacked boxes", resid_m=pf["resid_m"], free_pen_m=pf["free_pen_m"], levels=len(pf["prims"]))
                al_ = fit_asset(sorted(glob.glob(os.path.join(a.gen_dir, f"obj{oid}_s*.npz"))), ev, cfg, np.random.default_rng(0)) if a.gen_dir else None
                lim = max(cfg["auto_box_resid_m"], 1.2 * al_["resid_m"]) if al_ is not None and al_["resid_m"] is not None else cfg["auto_box_resid_m"]
                if pf["resid_m"] is not None and pf["resid_m"] <= lim:
                    f, carved = prim_field(pf["prims"], ev, cfg); fields["prim"] = (f, pf); r["prim"].update(carved_share=round(carved, 3),
                        prims=[{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()} for p in pf["prims"]])
                else:
                    r["prim"]["rejected"] = f"not box-like: residual {pf['resid_m']} m > {lim:.4f} (max of {cfg['auto_box_resid_m']} and 1.2 x the generated asset's)"
        else:
            r["prim"]["source"] = "vision-language recipe"
        files = sorted(f_ for g_ in [a.gen_dir] + (a.extra_gen_dir or []) if g_ for f_ in glob.glob(os.path.join(g_, f"obj{oid}_s*.npz")))
        al = fit_asset(files, ev, cfg, rng) if files else None
        if al is not None:
            r["gen"] = dict(asset=al["chosen"], cost=al["cost"], resid_m=al["resid_m"], scale_xyz=al["scale_xyz"], seeds=al["cands"])
            if al["resid_m"] is not None and al["resid_m"] > cfg["max_resid_m"]:
                r["gen"]["rejected"] = f"registration residual {al['resid_m']} m > {cfg['max_resid_m']}: evidence hull used instead"; al = None
        f, gi = gen_field(al, ev, cfg); fields["gen"] = (f, al); r.setdefault("gen", {}).update(gi)
        sheets = {}
        for name, (f, aux) in fields.items():
            mm = field_mesh(f, ev, cfg, rng, snap=cfg["plane_snap"] and name == "gen")
            if mm is None:
                r[name]["failed"] = "empty mesh"; continue
            m, nsnap = mm; L = np.asarray(m.vertices); Nl = np.asarray(m.vertex_normals); Vd = to_ds(fr.met(L)); Nd = fr.vec(Nl) @ R; F = np.asarray(m.faces)
            col, seen, q, gain = bake(Vd, Nd, F, trn, a.splat_depth_dir, s, cfg, height=lambda Pd_: fr.loc(s * (Pd_ @ R.T) + t)[:, 2])
            seen &= L[:, 2] > cfg["base_m"]                                  # the contact line photographs as floor: completed from above instead
            if name == "prim":
                cid, uv, cls, mats = charts_of(aux["prims"], L, Nl)
                if cfg["unseen_fill"] == "synth":
                    col, how, seen = fill_synth(col, seen, q, cid, uv, cls, mats, cfg["texel_m"], cfg, np.random.default_rng(oid), L[:, 2])
                else:
                    col, how = fill_same_material(col, seen, cid, uv, cls, mats, cfg["texel_m"])
            else:
                col, how = gen_colours(L, col, seen, aux)
            col = seam_blend(L, col, seen, cfg["seam_m"]); sc = ray_scene(Vd, F)
            num = numeric(sc, F, col, val[:12], a.splat_depth_dir, ev, s, R, t)
            r[name].update(vertices=int(len(L)), planes_snapped=nsnap, seen_share=round(float(seen.mean()), 3), unseen_fill=how, numeric=num,
                           watertight=bool(m.is_watertight), size_m=[round(float(x), 3) for x in np.ptp(L, 0)])
            save_cand(os.path.join(a.dir, "cand", f"obj{oid}_{name}.npz"), m, col, seen, fr, r[name])
            reals = []; vs_ = val
            if a.seg_dir:                                                   # sheet photos go to the API: segmented frames without people only
                vs_ = []
                for c in vc[::max(1, len(vc) // 40)]:
                    pm = person_mask(a.seg_dir, c["name"]); SD_ = depth_of(a.splat_depth_dir, c["name"])
                    if pm is not None and SD_ is not None:
                        om = obj_mask(c, SD_, ev, s, R, t)
                        if not (pm & ndimage_dilate(om, 25)).any():
                            vs_.append((int(om.sum()), c))
                vs_ = [c for _, c in sorted(vs_, key=lambda x: -x[0])[:12]]   # most of the object visible first
            for c in vs_[:: max(1, len(vs_) // 2)][:2]:
                SD_ = depth_of(a.splat_depth_dir, c["name"]); real = load_img(c) * (obj_mask(c, SD_, ev, s, R, t)[..., None] if SD_ is not None else 0)
                img, mk, _ = render(sc, F, col, c, bg=np.zeros_like(real)); reals.append((real, img, mk))
            nov = [render(sc, F, col, c, floor=(nd, dd, (0.62, 0.62, 0.6), 0.25 / s)) for c in novel_cams(ev, vc[0], s, R, t)] if vc else []
            sheets[name] = (reals, nov)
        # comparison sheet (A = prim, B = gen) for the vision-language choice and for the report
        names = [n_ for n_ in ("prim", "gen") if n_ in sheets]; rows = []
        if names:
            for k in range(len(sheets[names[0]][0])):
                real, _, mk0 = sheets[names[0]][0][k]; mk = mk0 if mk0.any() else np.ones_like(mk0)
                row = [label(crop_to(real, mk), "REAL photo (object only)")] + [label(crop_to(sheets[n_][0][k][1], mk), "AB"[("prim", "gen").index(n_)] + " same camera") for n_ in names]
                rows.append(np.hstack(row + [np.zeros_like(row[0])] * (3 - len(row))))
            for k in range(len(sheets[names[0]][1])):
                row = [label(crop_to(sheets[n_][1][k][0], sheets[n_][1][k][1]), "AB"[("prim", "gen").index(n_)] + f" NOVEL {('behind', 'behind-left', 'behind-right', 'above')[k]}") for n_ in names]
                rows.append(np.hstack(row + [np.zeros_like(row[0])] * (3 - len(row))))
            cv2.imwrite(os.path.join(a.dir, "cand", f"obj{oid}_sheet.jpg"), np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        json.dump({n_: r[n_].get("numeric", {}) for n_ in ("prim", "gen") if n_ in r}, open(os.path.join(a.dir, "cand", f"obj{oid}_numeric.json"), "w"))
        r["seconds"] = round(time.time() - t0)
        print(f"[layers] candidates o{oid}: " + json.dumps({k: v for k, v in r.items() if k != "recipe"}, default=str)[:1500], flush=True)
    json.dump(rep, open(os.path.join(a.dir, "candidates.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------ finalize
def load_cand(path):
    z = np.load(path, allow_pickle=False); fr = Frame(z["fr_c0"], z["fr_a1"], z["fr_a2"], z["fr_n"], float(z["fr_d"]))
    return dict(V=z["V"].astype(float), F=z["F"], C=z["C"].astype(float), N=z["N"].astype(float), seen=z["seen"], fr=fr, meta=json.loads(str(z["meta"])))


def finalize(a):
    import cv2, trimesh
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); to_ds = lambda X: ((X - t) @ R) / s; cams = {c["name"]: c for c in load_cams(a.dataset)}
    cj = json.load(open(os.path.join(a.dir, "candidates.json"))); evj = json.load(open(os.path.join(a.dir, "evidence.json")))
    nf = np.array(evj["floor"]["n"]); dfl = float(evj["floor"]["d"]); nd = R.T @ nf; dd = (nf @ t + dfl) / s
    edits = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "vlm", "review_round*.json"))):
        for e in json.load(open(f)).get("edits", []):
            edits.setdefault(str(e.get("object")), []).append(dict(e, source=os.path.basename(f)))
    out = {}; meshes = []
    for oid, r in cj.items():
        have = [n_ for n_ in ("prim", "gen") if os.path.exists(os.path.join(a.dir, "cand", f"obj{oid}_{n_}.npz"))]
        if not have:
            out[oid] = dict(decision="no mesh (object keeps its splat)"); continue
        sf = os.path.join(a.dir, "vlm", f"select_obj{oid}.json"); sel = json.load(open(sf)) if os.path.exists(sf) else None
        if sel and sel.get("candidate") == "none":                          # the vision-language check: neither shape is this object
            out[oid] = dict(decision="no mesh (vision-language: neither candidate matches the real object: " + str(sel.get("reason"))[:300] + ")"); continue
        num = {n_: (r[n_].get("numeric") or {}) for n_ in have}; best_num = max(have, key=lambda n_: ((num[n_].get("iou") or 0), (num[n_].get("psnr") or 0)))
        if "prim" in have and (num["prim"].get("iou") or 0) >= max((num[n_].get("iou") or 0) for n_ in have) - cfg["prefer_prim_iou_tol"]:
            best_num = "prim"                                               # as good a silhouette: the primitive shape is clean and closed by construction
        if len(have) == 1:
            ch, why = have[0], "only candidate"
        elif sel and sel.get("candidate") in have:
            ch, why = sel["candidate"], "vision-language choice: " + str(sel.get("reason"))[:300]
        else:
            ch, why = best_num, "numeric check (no vision-language answer)"
        c = load_cand(os.path.join(a.dir, "cand", f"obj{oid}_{ch}.npz")); ev = load_ev(a, int(oid), cfg); L = c["V"].copy(); C = c["C"].copy(); app = []
        vc = [cams[n_] for n_ in ev["views"] if n_ in cams]; val = vc[cfg["val_every"] // 2::cfg["val_every"]][:12]
        if ch == "gen" and cfg["match_unseen_colour"] and c["seen"].any() and (~c["seen"]).any():
            # generated texture on the unseen side: colour statistics matched to the RELIABLE seen colours (above the contact band,
            # robust inliers) -- a pale generated texture / floor pixels at the contact line made backs look white
            rel = c["seen"] & (L[:, 2] > cfg["synth_contact_m"]); rel = rel if rel.sum() > 50 else c["seen"]
            med = np.median(C[rel], 0); mad = 1.4826 * np.median(np.abs(C[rel] - med), 0) + 0.02; rel &= (np.abs(C - med) / mad).max(1) <= cfg["synth_z"]
            un = ~c["seen"]; mu_u, sd_u = np.median(C[un], 0), C[un].std(0) + 1e-3
            C[un] = np.clip((C[un] - mu_u) * np.clip(C[rel].std(0) / sd_u, 0.5, 2.0) + np.median(C[rel], 0), 0, 1); C = seam_blend(L, C, c["seen"], cfg["seam_m"])
        base_num = numeric(ray_scene(to_ds(c["fr"].met(L)), c["F"]), c["F"], C, val, a.splat_depth_dir, ev, s, R, t)
        for e in edits.get(str(oid), []):                                   # reviewed edits: bounded, kept only if the real frames do not get worse
            L2, C2 = L.copy(), C.copy(); mx = cfg["max_edit_m"]
            if e.get("edit") == "offset":
                L2 += np.clip([float(e.get("dx", 0) or 0), float(e.get("dy", 0) or 0), float(e.get("dz", 0) or 0)], -mx, mx)
            elif e.get("edit") == "scale":
                L2 *= np.clip([float(e.get("sx", 1) or 1), float(e.get("sy", 1) or 1), float(e.get("sz", 1) or 1)], 0.8, 1.25)
            elif e.get("edit") == "material" and (~c["seen"]).any() and c["seen"].any():
                if e.get("fix") == "mean_colour":
                    C2[~c["seen"]] = C[c["seen"]].mean(0)
                else:                                                       # colour statistics of the completed faces matched to the
                    un, se = ~c["seen"], c["seen"]                          # seen ones (keeps the synthesised structure; nearest-seen
                    mu_s, sd_s = np.median(C[se], 0), C[se].std(0) + 1e-3   # copying made faceted patches)
                    mu_u, sd_u = np.median(C[un], 0), C[un].std(0) + 1e-3
                    C2[un] = np.clip((C[un] - mu_u) * np.clip(sd_s / sd_u, 0.5, 2.0) + mu_s, 0, 1)
                C2 = seam_blend(L, C2, c["seen"], cfg["seam_m"])
            else:
                continue
            L2[:, 2] = np.maximum(L2[:, 2], 0); n2 = numeric(ray_scene(to_ds(c["fr"].met(L2)), c["F"]), c["F"], C2, val, a.splat_depth_dir, ev, s, R, t)
            ok = (n2["iou"] or 0) >= (base_num["iou"] or 0) - 0.01
            app.append(dict(e, applied=bool(ok), iou_before=base_num["iou"], iou_after=n2["iou"]))
            if ok:
                L, C, base_num = L2, C2, n2
        Vm = c["fr"].met(L); m = trimesh.Trimesh(Vm, c["F"], process=False); Nm = np.asarray(m.vertex_normals); area = np.zeros(len(Vm)); np.add.at(area, c["F"].reshape(-1), np.repeat(m.area_faces / 3, 3))
        vis = cam_visible(to_ds(Vm), Nm @ R, c["F"], list(cams.values()), cfg, s)   # every training camera, near or far
        # completed faces that stand in measured free space (cameras looked through there: the mesh is bigger than the object,
        # LiDAR misregistration) would hide what the cameras saw: counted as seen, never drawn
        vis = np.where(ev["G"].look(ev["free"], L, fill=False), np.maximum(vis, 99), vis)
        np.savez_compressed(os.path.join(a.dir, f"obj{oid}_mesh.npz"), V=Vm.astype(np.float32), F=c["F"], C=C.astype(np.float32), N=Nm.astype(np.float32), area=area.astype(np.float32),
                            L=L.astype(np.float32), seen=c["seen"], vis=vis, **{f"fr_{k}": np.asarray(v) for k, v in c["fr"].dump().items()})
        tm = trimesh.Trimesh(Vm, c["F"], vertex_colors=np.clip(C * 255, 0, 255).astype(np.uint8), process=False); tm.export(os.path.join(a.dir, f"obj{oid}.glb")); meshes.append((oid, tm))
        out[oid] = dict(decision=f"mesh = {ch} ({why})", candidate=ch, numeric_choice=best_num, vlm_choice=None if not sel else sel.get("candidate"), numeric=num, final_numeric=base_num,
                        edits=app, size_m=[round(float(x), 3) for x in np.ptp(L, 0)], vertices=int(len(L)), seen_share=round(float(c["seen"].mean()), 3),
                        watertight=bool(tm.is_watertight), volume_l=round(float(abs(tm.volume)) * 1000, 2), unseen_fill=c["meta"].get("unseen_fill"),
                        glb=os.path.join(a.dir, f"obj{oid}.glb"))
        print(f"[layers] final o{oid}: " + json.dumps(out[oid], default=str)[:900], flush=True)
    if meshes:
        import trimesh
        sc_ = trimesh.Scene()
        for oid, tm in meshes:
            sc_.add_geometry(tm, node_name=f"obj{oid}")
        sc_.export(os.path.join(a.dir, "scene.glb"))
        try:                                                                # USD for Isaac (metric scene frame, per-vertex displayColor)
            from pxr import Usd, UsdGeom, Gf, Vt
            st = Usd.Stage.CreateNew(os.path.join(a.dir, "scene.usda")); UsdGeom.SetStageMetersPerUnit(st, 1.0); UsdGeom.Xform.Define(st, "/World")
            for oid, tm in meshes:
                me = UsdGeom.Mesh.Define(st, f"/World/obj{oid}"); V = np.asarray(tm.vertices); F = np.asarray(tm.faces)
                me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32))); me.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
                me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32))); me.CreateSubdivisionSchemeAttr("none")
                pv = UsdGeom.PrimvarsAPI(me).CreatePrimvar("displayColor", me.GetDisplayColorAttr().GetTypeName(), UsdGeom.Tokens.vertex)
                pv.Set(Vt.Vec3fArray.FromNumpy((np.asarray(tm.visual.vertex_colors)[:, :3] / 255).astype(np.float32)))
            st.SetDefaultPrim(st.GetPrimAtPath("/World")); st.GetRootLayer().Save()
        except Exception as e:  # noqa: BLE001
            print(f"[layers] USD export skipped: {type(e).__name__} {e}", flush=True)
        # review renders: every object from behind / side / above in the assembled mesh scene
        rd = os.path.join(a.dir, "review", f"round{a.round}"); os.makedirs(rd, exist_ok=True)
        Vall, Fall, Call, off = [], [], [], 0
        for oid, tm in meshes:
            Vall.append(to_ds(np.asarray(tm.vertices))); Fall.append(np.asarray(tm.faces) + off); Call.append(np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0); off += len(tm.vertices)
        Vall, Fall, Call = np.concatenate(Vall), np.concatenate(Fall), np.concatenate(Call); sc = ray_scene(Vall, Fall); c0 = next(iter(cams.values()))
        for oid, tm in meshes:
            ev = load_ev(a, int(oid), cfg); tiles = []
            for k, c in enumerate(novel_cams(ev, c0, s, R, t)):
                img, mk, _ = render(sc, Fall, Call, c, floor=(nd, dd, (0.62, 0.62, 0.6), 0.25 / s)); tiles.append(label(cv2.resize(img, (480, 360)), f"object {oid}: {('behind', 'behind-left', 'behind-right', 'above')[k]}"))
            cv2.imwrite(os.path.join(rd, f"obj{oid}.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    cost = os.path.join(a.dir, "vlm", "cost.json")
    json.dump(dict(objects=out, vlm_cost=json.load(open(cost)) if os.path.exists(cost) else None, cfg=cfg), open(os.path.join(a.dir, "objects_layers.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------ floor + compose
def final_meshes(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "obj*_mesh.npz"))):
        z = np.load(f); out.append(dict(id=int(os.path.basename(f)[3:].split("_")[0]), V=z["V"].astype(float), F=z["F"], C=z["C"].astype(float), N=z["N"].astype(float),
                                        area=z["area"].astype(float), L=z["L"].astype(float), seen=z["seen"].astype(bool), vis=(z["vis"] if "vis" in z.files else z["seen"].astype(np.int32) * 99), fr=Frame(z["fr_c0"], z["fr_a1"], z["fr_a2"], z["fr_n"], float(z["fr_d"]))))
    return out


def accepted_meshes(a, cfg):
    gf = os.path.join(a.dir, "gate.json"); gj = json.load(open(gf))["objects"] if os.path.exists(gf) else {}
    return [m for m in final_meshes(a.dir) if gate_ok(gj.get(str(m["id"]), {}), cfg)]


def gate_pass(g, cfg):
    if "face_psnr_mesh" in g:
        return g["face_psnr_mesh"] is None or g["face_psnr_mesh"] >= cfg["min_face_psnr"]
    return g.get("loss_db", 0.0) <= cfg["max_object_loss_db"]


def gate_ok(g, cfg):
    """strict_gates true: a mesh failing the per-object PSNR gate keeps its splat. false (default): the gate is an alarm only
    (logged in gate.json / the compose report); the unseen-angle look decides."""
    if cfg["fallback_object_loss_db"] > 0 and g.get("loss_db", 0.0) > cfg["fallback_object_loss_db"]:
        return False                                                        # breaks its own training frames: keeps its splat
    return gate_pass(g, cfg) or not cfg["strict_gates"]


def floormap(a):
    """cells to complete in the top-down floor texture, next to every ACCEPTED object mesh: its footprint, the floor no training
    camera ever saw unoccluded (its view ray passed through an object), and nearby cells whose re-projected colour is closer to
    the object's material than to the surrounding floor (object pixels smeared onto the floor). Their colour is synthesised
    from the seen floor around them (LaMa, floor_fill_ibr.py inpaint) with those cells excluded from the context."""
    from scipy import ndimage
    cfg = load_cfg(a); z = dict(np.load(a.map)); mc = float(z["cell"]); e1, e2 = z["e1"], z["e2"]; x0, y0 = float(z["x0"]), float(z["y0"]); sh = z["occ"].shape
    foot = np.zeros(sh, bool); ms = accepted_meshes(a, cfg); info = {}
    for m in ms:
        i = ((m["V"] @ e1 - x0) / mc).astype(int); j = ((m["V"] @ e2 - y0) / mc).astype(int); ok = (i >= 0) & (i < sh[0]) & (j >= 0) & (j < sh[1])
        g = np.zeros(sh, bool); g[i[ok], j[ok]] = True; foot |= ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=2))
    it = lambda m_: max(1, int(round(m_ / mc)))
    foot_d = ndimage.binary_dilation(foot, iterations=it(0.02)); other = z["obj"] & ~ndimage.binary_dilation(foot, iterations=it(0.06))
    k = it(cfg["floor_close_m"]); dom = ndimage.binary_closing(np.pad(z["occ"] | foot_d, k), iterations=k)[k:-k, k:-k] & ~other
    dist = ndimage.distance_transform_edt(~foot_d) * mc; near = (dist <= cfg["shadow_reach_m"]) & ~foot_d & dom
    never = np.zeros(sh, bool); colc = np.zeros(sh, bool)
    if ms and a.dataset and a.sim3_json:                                    # floor cells no training camera saw without an object in the way
        s, R, t = sim3(a.sim3_json); cams = load_cams(a.dataset); cams = cams[::max(1, len(cams) // cfg["shadow_cams"])]
        sc = ray_scene(np.concatenate([m["V"] for m in ms]), np.concatenate([m["F"] + o for m, o in zip(ms, np.cumsum([0] + [len(m["V"]) for m in ms[:-1]]))]))
        ci, cj = np.nonzero(near); n_ = z["n"]
        Pm = ((ci + 0.5) * mc + x0)[:, None] * e1 + ((cj + 0.5) * mc + y0)[:, None] * e2 - float(z["d"]) * n_ + 0.01 * n_
        free_ = np.zeros(len(Pm), int); Pd = ((Pm - t) @ R) / s
        for c in cams:
            u, v, r, ok = project(c, Pd); ok &= (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"]) & (r * s < cfg["cam_max_m"]); i = np.nonzero(ok)[0]
            if not len(i):
                continue
            Cm = s * (c["C"] @ R.T) + t; dv = Cm - Pm[i]; dd = np.linalg.norm(dv, axis=1); th, _, _ = cast(sc, Pm[i], dv / dd[:, None]); free_[i] += th >= dd
        never[ci[free_ == 0], cj[free_ == 0]] = True
        ring = (dist > cfg["colour_reach_m"]) & (dist <= 2 * cfg["colour_reach_m"] + 0.1) & z["any"]
        fl = np.median(z["rgb"][ring], 0) if ring.any() else np.median(z["rgb"][z["any"]], 0)
        cand = (dist <= cfg["colour_reach_m"]) & ~foot_d & z["any"]; rgb = z["rgb"][cand]; dfl = np.linalg.norm(rgb - fl, axis=1); dob = np.full(len(rgb), np.inf)
        for m in ms:
            dob = np.minimum(dob, np.linalg.norm(rgb - np.median(m["C"], 0), axis=1))
        cc = np.zeros(sh, bool); cc[cand] = dob < dfl; colc = cc; info = dict(floor_rgb=np.round(fl, 3).tolist(), views=len(cams))
    reg = foot_d | ((never | colc) & (near | ((dist <= cfg["colour_reach_m"]) & dom)))
    lab, nl = ndimage.label(reg); keep = np.unique(lab[foot_d]); hole = np.isin(lab, keep[keep > 0])
    anyc = z["any"] & ~hole
    z["any"] = anyc; z["occ"] = z["occ"] | hole; z["obj"] = other; z["hole_override"] = hole; z["foot_cells"] = foot_d; z["rgb"] = z["rgb"] * anyc[..., None]
    np.savez_compressed(a.out, **z)
    from PIL import Image
    v = np.clip(z["rgb"] * 255, 0, 255).astype(np.uint8); v[hole] = (255, 0, 255); v[foot] = (255, 255, 0); Image.fromarray(v).save(os.path.splitext(a.out)[0] + "_cells.png")
    print("LAYERS_FLOORMAP " + json.dumps(dict(objects=[m["id"] for m in ms], footprint_m2=round(float(foot.sum()) * mc * mc, 3), never_unoccluded_m2=round(float((never & hole).sum()) * mc * mc, 3),
                                              object_colour_m2=round(float((colc & hole).sum()) * mc * mc, 3), completed_m2=round(float(hole.sum()) * mc * mc, 3), out=a.out, **info)), flush=True)


def mesh_dist(L, F, Q, lo, hi, vox=0.005, signed=False):
    """distance (m) from the query points to the closed mesh's SOLID (0 inside), object frame, voxel accurate: surface
    voxels from the (dense, texel-spaced) vertices + edge midpoints, filled (the floor closes the bottom), EDT."""
    from scipy import ndimage
    lo = np.asarray(lo, float); lo[2] = min(lo[2], -vox); sh = np.ceil((np.asarray(hi) - lo) / vox).astype(int) + 2
    P = np.concatenate([L, (L[F[:, 0]] + L[F[:, 1]]) / 2, (L[F[:, 1]] + L[F[:, 2]]) / 2, (L[F[:, 2]] + L[F[:, 0]]) / 2])
    ij = np.clip(np.floor((P - lo) / vox).astype(int), 0, sh - 1); occ = np.zeros(sh, bool); occ[ij[:, 0], ij[:, 1], ij[:, 2]] = True
    occ = ndimage.binary_closing(np.pad(occ, 2), iterations=1)[2:-2, 2:-2, 2:-2]; occ = solid_fill(occ)
    D = ndimage.distance_transform_edt(~occ) * vox
    if signed:                                                              # negative inside: depth below the surface
        D = D - ndimage.distance_transform_edt(np.pad(occ, 1, constant_values=False))[1:-1, 1:-1, 1:-1] * vox
    q = np.floor((Q - lo) / vox).astype(int); ok = np.all((q >= 0) & (q < sh), 1)
    return np.where(ok, D[np.clip(q[:, 0], 0, sh[0] - 1), np.clip(q[:, 1], 0, sh[1] - 1), np.clip(q[:, 2], 0, sh[2] - 1)], np.inf)


def train_weight(a, n):
    """max blending weight of each splat gaussian over the training views (clean_haze.py stats of that exact PLY), or None."""
    f = getattr(a, "train_stats", None)
    if not f or not os.path.exists(f):
        return None
    w = np.load(f)["train_max_w"]
    return w if len(w) == n else None


def layer_rows(a, cfg, v, s, R, t):
    """per final mesh: (id, mesh, rows of the base to delete, surface gaussians as rows of the base's dtype)."""
    from scipy.spatial.transform import Rotation
    Xm = s * (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T) + t; out = []; tw = train_weight(a, len(v))
    if tw is None and cfg["seen_from"] == "splat":
        print("[layers] no training-weight stats for this PLY: splat gaussians the cameras rely on are not protected", flush=True)
    for m in final_meshes(a.dir):
        Lg = m["fr"].loc(Xm); lo, hi = m["L"].min(0), m["L"].max(0); zn = cfg["zone_m"]
        d_ = (Lg[:, 0] > lo[0] - zn) & (Lg[:, 0] < hi[0] + zn) & (Lg[:, 1] > lo[1] - zn) & (Lg[:, 1] < hi[1] + zn) & (Lg[:, 2] > cfg["floor_m"]) & (Lg[:, 2] < hi[2] + zn)
        if cfg["zone_mode"] == "sdf":                                       # inside the mesh volume or within a thin shell of it (plus the
            sm_ = s * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)).max(1)   # gaussian's own reach)
            reach = cfg["shell_m"] + np.minimum(2 * sm_, cfg["extent_cap_m"]); pad = max(cfg["shell_m"] + cfg["extent_cap_m"], cfg["unseen_clear_m"])
            box = (Lg[:, 0] > lo[0] - pad) & (Lg[:, 0] < hi[0] + pad) & (Lg[:, 1] > lo[1] - pad) & (Lg[:, 1] < hi[1] + pad) & (Lg[:, 2] > cfg["floor_m"]) & (Lg[:, 2] < hi[2] + pad)
            i = np.nonzero(box)[0]; d_ = np.zeros(len(Lg), bool)
            if len(i) and cfg["seen_from"] == "splat":                     # the splat keeps drawing what the cameras saw: removed only deep
                from scipy.spatial import cKDTree                           # inside the solid and around the mesh's UNSEEN faces
                sd = mesh_dist(m["L"], m["F"], Lg[i], lo - pad, hi + pad, signed=True); uns = (m["vis"] < cfg["min_vis_views"])[cKDTree(m["L"]).query(Lg[i])[1]]
                # beside a face no camera saw, the splat's content is unconstrained (the glitter seen from behind): cleared up to
                # unseen_clear_m from that face (above the floor); the mesh's completed face is drawn there instead
                # (above unseen_clear_min_h only: the fuzzy floor layer behind an object is what cameras see over its top)
                wide = (sd <= cfg["unseen_clear_m"]) & (Lg[i, 2] > cfg["unseen_clear_min_h"])
                d_[i] = (sd < -cfg["inner_keep_m"]) | (((sd <= reach[i]) | wide) & uns)
                if tw is not None:                                          # what the training views draw stays (misregistered mesh)
                    d_[i] &= tw[i] < cfg["keep_train_w"]
            elif len(i):
                d_[i] = mesh_dist(m["L"], m["F"], Lg[i], lo - pad, hi + pad) <= reach[i]
        k = ~((m["L"][:, 2] < 0.004) & ((m["N"] @ m["fr"].n) < -0.5))                 # the underside lies on the floor layer: not drawn
        if cfg["seen_from"] == "splat":                                     # only faces no training camera could see become gaussians
            k &= m["vis"] < cfg["min_vis_views"]
        if cfg["local_floor"]:                                              # the splat's own floor around the object can sit a few cm above
            sc_ = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)); sc_.sort(1)   # the LiDAR floor
            dx = np.maximum(np.maximum(lo[0] - Lg[:, 0], Lg[:, 0] - hi[0]), 0); dy = np.maximum(np.maximum(lo[1] - Lg[:, 1], Lg[:, 1] - hi[1]), 0)
            ring = (np.hypot(dx, dy) > 0.04) & (np.hypot(dx, dy) < 0.25) & (np.abs(Lg[:, 2]) < 0.08) & (sc_[:, 0] < 0.3 * sc_[:, 2]) & (v["opacity"] > 0)
            hf = float(np.median(Lg[ring, 2])) if ring.sum() > 50 else 0.0   # (misregistration): mesh surface under it would show through
            if hf > 0.003:
                k &= m["L"][:, 2] > hf + 0.003
            print(f"[layers] o{m['id']}: splat floor {hf * 100:+.1f} cm vs LiDAR floor ({int(ring.sum())} floor gaussians)", flush=True)
        nn_ = np.linalg.norm(m["N"], axis=1); k &= np.isfinite(m["N"]).all(1) & np.isfinite(m["V"]).all(1) & (nn_ > 1e-6)   # degenerate vertices carry no surface
        Vm, Nm, Cv, sgm = m["V"][k], m["N"][k], m["C"][k], cfg["gauss_sigma"] * np.sqrt(np.maximum(m["area"][k], 1e-10))
        # backing layer: one larger disc per --backing_m cell, just behind the surface -- stays opaque when the fine discs are
        # sub-pixel (distant views: the viewer's anti-aliasing fades sub-pixel gaussians, the object would turn see-through)
        cell = np.floor(Vm / cfg["backing_m"]).astype(np.int64); _, first, inv = np.unique(cell, axis=0, return_index=True, return_inverse=True); inv = inv.reshape(-1)
        cb = np.zeros((len(first), 3)); np.add.at(cb, inv, Cv); cb /= np.bincount(inv, minlength=len(first))[:, None]
        nb_ = Nm[first] / np.maximum(np.linalg.norm(Nm[first], axis=1, keepdims=True), 1e-9)
        sr = np.concatenate([m["seen"][k], m["seen"][k][first]])          # rows a training photo coloured (the refit may touch only these)
        Vm = np.concatenate([Vm, Vm[first] - cfg["backing_inset_m"] * nb_]); Nm = np.concatenate([Nm, Nm[first]]); Cv = np.concatenate([Cv, cb])
        sgm = np.concatenate([sgm, np.full(len(first), cfg["backing_sigma"] * cfg["backing_m"])])
        Vd = ((Vm - t) @ R) / s; Nd = Nm @ R; Nd /= np.maximum(np.linalg.norm(Nd, axis=1, keepdims=True), 1e-9)
        tan = np.cross(Nd, [0.0, 0.0, 1.0]); bad = np.linalg.norm(tan, axis=1) < 1e-3; tan[bad] = np.cross(Nd[bad], [1.0, 0.0, 0.0]); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        q = Rotation.from_matrix(np.stack([tan, np.cross(Nd, tan), Nd], 2)).as_quat(); new = np.zeros(len(Vd), dtype=v.dtype)
        new["x"], new["y"], new["z"] = Vd.T; sg = sgm / s
        new["scale_0"] = new["scale_1"] = np.log(sg); new["scale_2"] = np.log(cfg["gauss_thick_m"] / s)
        new["rot_0"], new["rot_1"], new["rot_2"], new["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
        new["opacity"] = np.log(cfg["gauss_opacity"] / (1 - cfg["gauss_opacity"])); dc = (np.clip(Cv, 0, 1) - 0.5) / C0
        for i in range(3):
            new[f"f_dc_{i}"] = dc[:, i]
        if "nx" in v.dtype.names:
            new["nx"], new["ny"], new["nz"] = Nd.T
        out.append((m["id"], m, d_, new, sr))
    return out


def to_params(arr, dev):
    import torch
    names = arr.dtype.names; col = lambda pre: np.stack([np.asarray(arr[n], np.float32) for n in sorted((n for n in names if n.startswith(pre)), key=lambda x: int(x.split("_")[-1]))], 1)
    rest = col("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(arr), 0), np.float32); K = rest.shape[1] // 3
    p = {"means": np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32), "scales": col("scale_"), "quats": col("rot_"), "opacities": np.asarray(arr["opacity"], np.float32),
         "sh0": col("f_dc_")[:, None], "shN": rest.reshape(len(arr), 3, K).transpose(0, 2, 1)}
    return {k: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev) for k, x in p.items()}


def gate(a):
    """viewer-exact accept test per object (gsplat venv, GPU): PSNR on the real training frames that see the object, base vs base
    with ONLY this object replaced by its mesh; above --max-loss-db the object keeps its splat (a broken front is not shipped)."""
    import torch
    from plyfile import PlyData
    from finetune_ply_viewer import ViewerRaster, cam_args, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, psnr, ssim
    from solidify_unseen import all_frames
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); v = PlyData.read(a.ply)["vertex"].data; dev = "cuda"
    p0 = to_params(v, dev); deg = sh_deg(p0); Rv = ViewerRaster("ss352", "exact", display_scale=1.8)
    fr = {n_: (cam, c2w) for n_, cam, c2w in all_frames([a.dataset])}; out = {}; cams_by = {c_["name"]: c_ for c_ in load_cams(a.dataset)}
    for oid, m, d_, new, sr in layer_rows(a, cfg, v, s, R, t):
        z = np.load(os.path.join(a.dir, "ev", f"obj{oid}.npz")); names = [str(x) for x in z["views"] if str(x) in fr]; names = names[::max(1, len(names) // cfg["gate_frames"])]
        pk = to_params(np.concatenate([v[~d_], new]), dev); pb, pa = [], []; nb = int((~d_).sum())
        allv = [str(x) for x in z["views"] if str(x) in fr]
        if cfg["refit_iters"] > 0 and allv and sr.any():                                 # texture refit through the viewer itself: mesh colours only, geometry frozen
            seen_t = torch.tensor(sr, device=dev)                            # background a training view shows past them would fake see-through holes
            sh0 = torch.nn.Parameter(pk["sh0"][nb:].clone()); opt = torch.optim.Adam([sh0], lr=2.5e-3, eps=1e-15); rng = np.random.default_rng(0)
            for it in range(cfg["refit_iters"]):
                n_ = allv[rng.integers(len(allv))]; cam, c2w = fr[n_]; gt = load_rgb(os.path.join(a.dataset, "images", n_), dev).float() / 255
                pp = dict(pk); pp["sh0"] = torch.cat([pk["sh0"][:nb], sh0]); img = Rv.render(pp, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0]
                l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])); opt.zero_grad(set_to_none=True); l.backward()
                sh0.grad[~seen_t] = 0; opt.step()                           # unseen faces keep their completed texture: painting them with the
            pk["sh0"] = torch.cat([pk["sh0"][:nb], sh0.detach()]); np.save(os.path.join(a.dir, f"obj{oid}_refit_dc.npy"), sh0.detach()[:, 0].cpu().numpy())
        if cfg["seen_from"] == "splat" and cfg["vd_refit_iters"] > 0 and allv and pk["shN"].shape[1] > 0 and len(new):
            # view-dependent refit of the completed faces: the training views see them only THROUGH the splat's semi-transparent
            # seen surface (they change those pixels). Their SH is fitted so that, seen from the training cameras' side, the
            # pixels match the photos again, while seen from outside the completed face (directions within vd_cone_deg of its
            # inward view direction) the colour stays the completed texture.
            from gsplat import spherical_harmonics
            q_ = torch.nn.functional.normalize(pk["quats"][nb:], dim=1); w_, x_, y_, z_ = q_.unbind(1)
            nrm = torch.stack([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], 1)   # local z = disc normal
            nrm = nrm * torch.sign((nrm * pk["means"][nb:] - nrm * pk["means"][nb:].mean(0)).sum(1, keepdim=True) + 1e-9)  # outward
            K_ = pk["shN"].shape[1]; dgr = int(round(np.sqrt(K_ + 1))) - 1; tgt = (C0 * pk["sh0"][nb:, 0] + 0.5).detach()
            sh0 = torch.nn.Parameter(pk["sh0"][nb:].clone()); shN = torch.nn.Parameter(pk["shN"][nb:].clone())
            opt = torch.optim.Adam([sh0, shN], lr=2.5e-3, eps=1e-15); rng = np.random.default_rng(0); cone = np.radians(cfg["vd_cone_deg"])
            for it in range(cfg["vd_refit_iters"]):
                n_ = allv[rng.integers(len(allv))]; cam, c2w = fr[n_]; gt = load_rgb(os.path.join(a.dataset, "images", n_), dev).float() / 255
                pp = dict(pk); pp["sh0"] = torch.cat([pk["sh0"][:nb], sh0]); pp["shN"] = torch.cat([pk["shN"][:nb], shN])
                img = Rv.render(pp, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0]
                l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
                jit = torch.randn_like(nrm) * float(np.tan(cone)) / 1.7; dirs = torch.nn.functional.normalize(-nrm + jit - (jit * nrm).sum(1, keepdim=True) * nrm, dim=1)
                col = spherical_harmonics(dgr, dirs, torch.cat([sh0, shN], 1)) + 0.5
                l = l + cfg["vd_lambda"] * (col - tgt).abs().mean(); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
            pk["sh0"] = torch.cat([pk["sh0"][:nb], sh0.detach()]); pk["shN"] = torch.cat([pk["shN"][:nb], shN.detach()])
            np.savez(os.path.join(a.dir, f"obj{oid}_refit_sh.npz"), sh0=sh0.detach()[:, 0].cpu().numpy(), shN=shN.detach().cpu().numpy())
        # per-face gate: only pixels where the mesh shows a PHOTO-SEEN face and the original splat shows this object; the completed
        # back and the silhouette change are not judged here (the orbit / behind renders judge them)
        ev = load_ev(a, oid, cfg); pm = {k_: x[nb:].clone() for k_, x in pk.items()}; seen_c = torch.tensor(sr, device=dev, dtype=torch.float32)
        pm["sh0"] = ((seen_c - 0.5) / C0)[:, None, None].repeat(1, 1, 3); pm["shN"] = torch.zeros_like(pm["shN"]); fb, fa, npx = [], [], []
        with torch.no_grad():
            for n_ in names:
                cam, c2w = fr[n_]; gt = load_rgb(os.path.join(a.dataset, "images", n_), dev).float() / 255; w2c = to_t(np.linalg.inv(c2w), dev)
                ib = Rv.render(p0, w2c, *cam_args(cam), sh_degree=deg)[0]; ia = Rv.render(pk, w2c, *cam_args(cam), sh_degree=deg)[0]
                pb.append(float(psnr(ib, gt))); pa.append(float(psnr(ia, gt)))
                SD = depth_of(a.splat_depth_dir, n_) if a.splat_depth_dir else None; c_ = cams_by.get(n_)
                if SD is None or c_ is None:
                    continue
                mk = (Rv.render(pm, w2c, *cam_args(cam), sh_degree=0)[0].mean(-1) > 0.5) & torch.tensor(obj_mask(c_, SD, ev, s, R, t), device=dev)
                if int(mk.sum()) < 100:
                    continue
                fb.append(float(((ib - gt) ** 2)[mk].mean())); fa.append(float(((ia - gt) ** 2)[mk].mean())); npx.append(int(mk.sum()))
        loss = float(np.mean(pb) - np.mean(pa)) if names else 0.0
        floss = float(10 * np.log10(np.average(fa, weights=npx) / max(np.average(fb, weights=npx), 1e-12))) if npx else 0.0
        fpsnr = float(-10 * np.log10(max(np.average(fa, weights=npx), 1e-12))) if npx else None   # absolute: the splat is fitted to these very frames
        out[str(oid)] = dict(frames=len(names), psnr_splat=round(float(np.mean(pb)), 3) if names else None, psnr_mesh=round(float(np.mean(pa)), 3) if names else None,
                             loss_db=round(loss, 3), face_frames=len(npx), face_px=int(np.sum(npx)) if npx else 0, face_loss_db=round(floss, 3),
                             face_psnr_mesh=None if fpsnr is None else round(fpsnr, 2), accept=bool(fpsnr is None or fpsnr >= cfg["min_face_psnr"] or not cfg["strict_gates"]),
                             alarm=bool(fpsnr is not None and fpsnr < cfg["min_face_psnr"]))
        print(f"[layers] gate o{oid}: " + json.dumps(out[str(oid)]), flush=True)
    json.dump(dict(objects=out, min_face_psnr=cfg["min_face_psnr"]), open(os.path.join(a.dir, "gate.json"), "w"), indent=1)


def look_at_cam(c0, eye_d, tgt_d, up_d):
    z = tgt_d - eye_d; z /= np.linalg.norm(z); x = np.cross(z, up_d); x /= np.linalg.norm(x); y = np.cross(z, x); Rc = np.stack([x, y, z])
    return dict(c0, R=Rc, t=-Rc @ eye_d, C=eye_d)


def meshviews(a):
    """3D-consistent pseudo-views of the completed objects + completed floor for scripts/train_distill_gsplat.py (CPU):
    cameras orbit every meshed object at the viewer's distances / heights (--cfg mv_*; line of sight to the object clear of
    other LiDAR returns); per view the meshes (vertex colours) and the completed floor (floor_map_inpainted rgb_filled) are
    ray-cast. Supervision mask (soft, 0..255) = pixels whose surface point NO training camera saw: mesh vertices' seen flags
    (barycentric), floor cells of the completion (hole_override); feathered by --mv_feather_px. Depth target = ray distance
    (dataset units) where supervised. Pixels outside the mask are never used by the trainer (it swaps in the target there).
    -> <out>/{sparse/0, images/<stem>.png, <stem>_mask.png, <stem>_depth.npy}, pseudo.json"""
    import cv2
    from scipy.spatial import cKDTree
    from make_pseudo_views import rot2quat, write_images
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); to_ds = lambda X: ((X - t) @ R) / s; rng = np.random.default_rng(0)
    cams = load_cams(a.dataset); c0 = cams[0]; ms = final_meshes(a.dir); out = a.out; os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    z = np.load(a.map); mc = float(z["cell"]); e1, e2, nf, df = z["e1"], z["e2"], z["n"], float(z["d"]); x0, y0 = float(z["x0"]), float(z["y0"])
    frgb = z["rgb_filled"]; fhole = z["hole_override"] if "hole_override" in z.files else np.zeros(frgb.shape[:2], bool)
    Vd = np.concatenate([to_ds(m["V"]) for m in ms]); Fall = np.concatenate([m["F"] + o for m, o in zip(ms, np.cumsum([0] + [len(m["V"]) for m in ms[:-1]]))])
    Call = np.concatenate([m["C"] for m in ms]); Sall = np.concatenate([m["seen"].astype(np.float32) for m in ms]); owner = np.concatenate([np.full(len(m["F"]), m["id"]) for m in ms])
    sc = ray_scene(Vd, Fall); ltree = None
    flc = np.median(z["rgb"][z["any"]], 0) if "any" in z.files and z["any"].any() else np.median(frgb.reshape(-1, 3), 0)
    obc = np.stack([np.median(m["C"], 0) for m in ms]); fmed, fmad = {}, {}
    gi, gj = np.nonzero(z["any"]) if "any" in z.files else np.nonzero(np.ones(frgb.shape[:2], bool))
    Gm = ((gi + 0.5) * mc + x0)[:, None] * e1 + ((gj + 0.5) * mc + y0)[:, None] * e2 - df * nf
    for m in ms:                                                            # seen floor 0.3 - 1.5 m around each object: its colour distribution
        dd_ = np.linalg.norm(m["fr"].loc(Gm)[:, :2], axis=1); sel = (dd_ > 0.3) & (dd_ < 1.5); sel = sel if sel.sum() > 200 else dd_ < 3.0
        cc = z["rgb"][gi[sel], gj[sel]]; fmed[m["id"]] = np.median(cc, 0); fmad[m["id"]] = 1.4826 * np.median(np.abs(cc - fmed[m["id"]]), 0) + 0.01
    if a.cloud and os.path.exists(a.cloud):
        from plyfile import PlyData
        cv_ = PlyData.read(a.cloud)["vertex"].data; Lp = np.stack([cv_["x"], cv_["y"], cv_["z"]], 1).astype(np.float64); Lp = Lp[np.unique(np.floor(Lp / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
        ltree = cKDTree(Lp)
    W, H = c0["W"], c0["H"]; xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5); rows, info = [], []
    for m in ms:
        fr = m["fr"]; L = m["L"]; rad = float(np.linalg.norm(np.ptp(L[:, :2], 0)) / 2); hmax = float(L[:, 2].max()); ctr = fr.met(np.array([[0.0, 0.0, hmax / 2]]))[0]
        tries, k = 0, 0
        while k < cfg["mv_n"] and tries < 40 * cfg["mv_n"]:
            tries += 1; th = rng.uniform(0, 2 * np.pi); dist = rad + rng.uniform(*cfg["mv_dist"]); hh = rng.uniform(*cfg["mv_h"])
            eye = fr.met(np.array([[dist * np.cos(th), dist * np.sin(th), hh]]))[0]
            tgt = fr.met(np.array([[rng.uniform(-0.3, 0.3) * rad, rng.uniform(-0.3, 0.3) * rad, rng.uniform(0.2, 0.7) * hmax]]))[0]
            if ltree is not None:                                           # clear line of sight: no foreign LiDAR return on the way
                Lg = np.linalg.norm(tgt - eye); u_ = (tgt - eye) / Lg; pts = eye + np.outer(np.arange(0.0, max(Lg - rad - 0.05, 0.0), 0.03), u_)
                if len(pts) and ltree.query(pts)[0].min() < cfg["mv_clear_m"]:
                    continue
                if ltree.query(eye[None])[0][0] < 0.15:
                    continue
            c = look_at_cam(c0, to_ds(eye[None])[0], to_ds(tgt[None])[0], R.T @ fr.n)
            dc = np.stack([(xs - c["cx"]) / c["fx"], (ys - c["cy"]) / c["fy"], np.ones_like(xs)], -1).reshape(-1, 3); dw = dc @ c["R"]; dn = np.linalg.norm(dw, axis=1); dw /= dn[:, None]
            O = np.broadcast_to(c["C"], dw.shape); th_, pid, uv = cast(sc, O, dw); hit = np.isfinite(th_)
            img = np.zeros((H * W, 3), np.float32); wgt = np.zeros(H * W, np.float32); dep = np.zeros(H * W, np.float32)
            f = Fall[pid[hit]]; u, v = uv[hit, 0], uv[hit, 1]; b = np.stack([1 - u - v, u, v], 1)
            img[hit] = (Call[f] * b[..., None]).sum(1); wgt[hit] = 1 - (Sall[f] * b).sum(1); dep[hit] = th_[hit]
            # completed floor where no mesh is hit (plane, map cells within --mv_floor_m of this object)
            Om = s * (c["C"] @ R.T) + t; dm = dw @ R.T; den = dm @ nf; tt = -(Om @ nf + df) / np.where(np.abs(den) > 1e-9, den, 1e-9)
            fl = ~hit & (tt > 0) & (den < -1e-6); Pm = Om + dm[fl] * tt[fl, None]; ci = ((Pm @ e1 - x0) / mc).astype(int); cj = ((Pm @ e2 - y0) / mc).astype(int)
            ok = (ci >= 0) & (ci < frgb.shape[0]) & (cj >= 0) & (cj < frgb.shape[1]) & (np.linalg.norm(fr.loc(Pm)[:, :2], axis=1) < rad + cfg["mv_floor_m"])
            idx = np.nonzero(fl)[0][ok]; img[idx] = frgb[ci[ok], cj[ok]]; wgt[idx] = fhole[ci[ok], cj[ok]]; dep[idx] = tt[fl][ok] / s
            cf = img[idx]; dfl = np.linalg.norm(cf - flc, axis=1); dob = np.min(np.linalg.norm(cf[:, None] - obc[None], axis=2), 1)
            wgt[idx[dob < dfl]] = 0                                         # completed floor that still looks like an object: not a target
            zl = np.abs(cf - fmed[m["id"]]) / fmad[m["id"]]; wgt[idx[zl.max(1) > cfg["mv_floor_z"]]] = 0   # outside the local seen-floor colours
            wgt[img.max(1) < cfg["mv_black"]] = 0                            # black / unfilled pixels are never targets
            wgt = wgt.reshape(H, W)
            if cfg["mv_feather_px"] > 0:                                    # no supervision right at silhouettes / seen-unseen seams
                k_ = 2 * cfg["mv_feather_px"] + 1; wgt = np.minimum(wgt, cv2.erode(wgt, np.ones((k_, k_), np.uint8)))
            share = float((wgt > 0.05).mean())
            if share < cfg["mv_min_share"]:
                continue
            stem = f"msh_o{m['id']}_{k:03d}"; k += 1
            cv2.imwrite(os.path.join(out, "images", stem + ".png"), np.clip(img.reshape(H, W, 3)[..., ::-1] * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out, "images", stem + "_mask.png"), np.clip(wgt * 255, 0, 255).astype(np.uint8))
            np.save(os.path.join(out, "images", stem + "_depth.npy"), np.where(wgt.reshape(-1) > 0.05, dep, 0).reshape(H, W).astype(np.float32))
            rows.append((len(rows) + 1, rot2quat(c["R"]), c["t"], 1, stem + ".png")); info.append(dict(name=stem + ".png", object=m["id"], dist_m=round(dist, 2), eye_h=round(hh, 2), supervised_share=round(share, 3)))
        print(f"[meshviews] o{m['id']}: {k} views ({tries} tries)", flush=True)
    open(os.path.join(out, "sparse", "0", "cameras.txt"), "w").write(f"1 PINHOLE {W} {H} {c0['fx']} {c0['fy']} {c0['cx']} {c0['cy']}\n")
    write_images(os.path.join(out, "sparse", "0", "images.txt"), rows); open(os.path.join(out, "sparse", "0", "points3D.txt"), "w").write("")
    json.dump(dict(views=info, n=len(rows), cfg={k: cfg[k] for k in cfg if k.startswith("mv_")}), open(os.path.join(out, "pseudo.json"), "w"), indent=1, default=str)
    # contact sheet: 16 views, image | mask
    import random
    pick = [info[i]["name"] for i in np.linspace(0, len(info) - 1, min(16, len(info))).astype(int)]; tiles = []
    for nm in pick:
        im = cv2.imread(os.path.join(out, "images", nm)); mk = cv2.imread(os.path.join(out, "images", nm[:-4] + "_mask.png"))
        tiles.append(np.hstack([cv2.resize(im, (240, 180)), cv2.resize(mk, (240, 180))]))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(out, "contact.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("MESHVIEWS " + json.dumps(dict(views=len(rows), out=out, mean_supervised_share=round(float(np.mean([v["supervised_share"] for v in info])), 3) if info else 0)), flush=True)


def mergeviews(a):
    """one pseudo-view dataset from several (train_distill_gsplat.py takes one): --sources DS:weight ...; masks rescaled by
    weight / --pseudo-weight (so each source keeps its own effective weight), images / depths linked; camera ids kept apart."""
    import cv2
    from make_pseudo_views import write_images
    out = a.out; os.makedirs(os.path.join(out, "images"), exist_ok=True); os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    cam_lines, rows, n = [], [], {}
    if a.real:                                                              # real frames, no masks (fine-tunes: real + pseudo in one dataset)
        cams, imgs = read_text_model(os.path.join(a.real, "sparse", "0")); (W, H, K), = cams.values()
        cam_lines.append(f"1 PINHOLE {W} {H} {' '.join(str(x) for x in K)}\n"); n[a.real] = len(imgs)
        for name, _, q, tc in imgs:
            for suf in ("", ):
                src, dst = os.path.join(a.real, "images", name), os.path.join(out, "images", name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(src), dst)
            rows.append((len(rows) + 1, q, tc, 1, name))
    for ci, sw in enumerate(a.sources, 1 + bool(a.real)):
        ds, w = sw.rsplit(":", 1); f = float(w) / a.pseudo_weight; cams, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
        assert len(cams) == 1, "one camera per source"
        (W, H, K), = cams.values(); line = f"PINHOLE {W} {H} {' '.join(str(x) for x in K)}\n"
        same = next((i + 1 for i, l in enumerate(cam_lines) if l.split(" ", 1)[1] == line), None)
        if same:
            ci = same
        else:
            ci = len(cam_lines) + 1; cam_lines.append(f"{ci} {line}")
        n[ds] = len(imgs)
        for name, _, q, tc in imgs:
            st = os.path.splitext(name)[0]
            for suf in (".png", "_depth.npy"):
                src, dst = os.path.join(ds, "images", st + suf), os.path.join(out, "images", st + suf)
                if os.path.exists(src) and not os.path.lexists(dst):
                    os.symlink(src, dst)
            mk = cv2.imread(os.path.join(ds, "images", st + "_mask.png"), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(out, "images", st + "_mask.png"), np.clip(mk.astype(np.float32) * f, 0, 255).astype(np.uint8))
            rows.append((len(rows) + 1, q, tc, ci, name))
    open(os.path.join(out, "sparse", "0", "cameras.txt"), "w").writelines(cam_lines); write_images(os.path.join(out, "sparse", "0", "images.txt"), rows)
    open(os.path.join(out, "sparse", "0", "points3D.txt"), "w").write("")
    json.dump(dict(sources=a.sources, pseudo_weight=a.pseudo_weight, views=n), open(os.path.join(out, "pseudo.json"), "w"), indent=1)
    print("MERGEVIEWS " + json.dumps(dict(views=len(rows), per_source=n, out=out)), flush=True)


def seedpoints(a):
    """copy of the EDGS training dataset (--edgs) whose init points3D.txt gets points on every object mesh's UNSEEN surface
    (mesh colours, --seed_m spacing), so the trainer starts with gaussians there (no densification needed); --subset keeps
    that share of the frames (evenly spaced; cheap tests). Images / depth targets are linked."""
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); out = a.out; s0 = os.path.join(out, "sparse", "0"); os.makedirs(s0, exist_ok=True)
    src = os.path.join(a.edgs, "sparse", "0")
    if not os.path.lexists(os.path.join(out, "images")):
        os.symlink(os.path.realpath(os.path.join(a.edgs, "images")), os.path.join(out, "images"))
    for f in ("edgs_init.json", "depth_targets.json", "point_cloud.ply"):
        if os.path.exists(os.path.join(a.edgs, f)) and not os.path.lexists(os.path.join(out, f)):
            os.symlink(os.path.realpath(os.path.join(a.edgs, f)), os.path.join(out, f))
    open(os.path.join(s0, "cameras.txt"), "w").writelines(open(os.path.join(src, "cameras.txt")))
    lines = [ln for ln in open(os.path.join(src, "images.txt")) if not ln.startswith("#")]; recs = [lines[i:i + 2] for i in range(0, len(lines), 2) if len(lines[i].split()) >= 10]
    keep = recs if a.subset >= 1 else [recs[i] for i in np.unique(np.linspace(0, len(recs) - 1, max(1, int(round(a.subset * len(recs))))).astype(int))]
    open(os.path.join(s0, "images.txt"), "w").writelines(["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"] + [x for r in keep for x in r])
    pts = [ln for ln in open(os.path.join(src, "points3D.txt"))]; nid = 1 + max(int(l.split()[0]) for l in pts if l.strip() and not l.startswith("#")); add = []
    if not a.no_seed:
        for m in final_meshes(a.dir):
            V = m["V"][~m["seen"]]; C = m["C"][~m["seen"]]
            if not len(V):
                continue
            _, first = np.unique(np.floor(V / cfg["seed_m"]).astype(np.int64), axis=0, return_index=True); Vd = ((V[first] - t) @ R) / s
            for x, c in zip(Vd, np.clip(C[first] * 255, 0, 255).astype(int)):
                add.append(f"{nid} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n"); nid += 1
    open(os.path.join(s0, "points3D.txt"), "w").writelines(pts + add)
    print("SEEDPOINTS " + json.dumps(dict(frames=len(keep), of=len(recs), seeds=len(add), out=out)), flush=True)


def floorviews(a):
    """floor-only renders (gsplat venv, GPU): from behind / beside / above every meshed object, the input splat and the layered
    background (the floor completion) with every object's zone removed -> <dir>/floorviews/obj<id>.jpg (top: before, bottom: after)."""
    import cv2, torch
    from plyfile import PlyData
    from finetune_ply_viewer import ViewerRaster, sh_deg, to_t
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); dev = "cuda"; Rv = ViewerRaster("ss352", "exact", display_scale=1.8); od = os.path.join(a.dir, "floorviews"); os.makedirs(od, exist_ok=True)
    c0 = load_cams(a.dataset)[0]; ps = {}
    for tag, f in (("before", a.base_ply), ("after", a.ply)):
        v = PlyData.read(f)["vertex"].data; dele = np.zeros(len(v), bool)
        for _, m, d_, _, _ in layer_rows(a, cfg, v, s, R, t):
            dele |= d_
        ps[tag] = to_params(v[~dele], dev)
    for m in accepted_meshes(a, cfg):
        ev = load_ev(a, m["id"], cfg); rows = []
        for tag in ("before", "after"):
            tiles = []
            for k, c in enumerate(novel_cams(ev, c0, s, R, t)):
                w2c = np.eye(4); w2c[:3, :3] = c["R"]; w2c[:3, 3] = c["t"]
                with torch.no_grad():
                    img = Rv.render(ps[tag], to_t(w2c, dev), c["fx"], c["fy"], c["cx"], c["cy"], c["W"], c["H"], sh_degree=sh_deg(ps[tag]))[0].clamp(0, 1).cpu().numpy()
                tiles.append(label(cv2.resize(img, (400, 300)), f"o{m['id']} {tag}: {('behind', 'behind-left', 'behind-right', 'above')[k]}"))
            rows.append(np.hstack(tiles))
        cv2.imwrite(os.path.join(od, f"obj{m['id']}.jpg"), np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[layers] floor views -> {od}", flush=True)


def compose(a):
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    cfg = load_cfg(a); s, R, t = sim3(a.sim3_json); v = PlyData.read(a.ply)["vertex"].data; N0 = len(v); dele = np.zeros(N0, bool); rows = []; rep = {}
    gf = os.path.join(a.dir, "gate.json"); gj = json.load(open(gf))["objects"] if os.path.exists(gf) else {}; kept = []
    for oid, m, d_, new, _ in layer_rows(a, cfg, v, s, R, t):
        if gj and not gate_ok(gj.get(str(oid), {}), cfg):
            rep[oid] = dict(decision=f"kept its splat: own training frames -{gj[str(oid)].get('loss_db')} dB (fallback_object_loss_db "
                                     f"{cfg['fallback_object_loss_db']}) / seen-face PSNR {gj[str(oid)].get('face_psnr_mesh')} dB"); continue
        rs = os.path.join(a.dir, f"obj{oid}_refit_sh.npz")                 # view-dependent refit (gate, seen_from splat)
        if os.path.exists(rs):
            z_ = np.load(rs); K_ = z_["shN"].shape[1]
            if len(z_["sh0"]) == len(new) and all(f"f_rest_{c * K_ + k}" in new.dtype.names for c in range(3) for k in range(K_)):
                for c in range(3):
                    new[f"f_dc_{c}"] = z_["sh0"][:, c]
                    for k in range(K_):
                        new[f"f_rest_{c * K_ + k}"] = z_["shN"][:, k, c]
        rf = os.path.join(a.dir, f"obj{oid}_refit_dc.npy")
        if os.path.exists(rf) and not os.path.exists(rs):
            dc_ = np.load(rf)
            if len(dc_) == len(new):
                for i in range(3):
                    new[f"f_dc_{i}"] = dc_[:, i]
        dele |= d_; rows.append(new); kept.append(m); rep[oid] = dict(deleted=int(d_.sum()), inserted=int(len(new)))
        if gj and not gate_pass(gj.get(str(oid), {}), cfg):
            rep[oid]["alarm"] = f"seen-face PSNR {gj[str(oid)].get('face_psnr_mesh')} dB < {cfg['min_face_psnr']} (strict_gates off: composited anyway)"
    if a.map and os.path.exists(a.map) and kept:                            # completed floor under every meshed object (hidden by the mesh except at its rim)
        z = np.load(a.map); fc = z["foot_cells"] if "foot_cells" in z.files else np.zeros_like(z["occ"]); hi, hj = np.nonzero(fc); mc = float(z["cell"])
        e1, e2, n_, d_ = z["e1"], z["e2"], z["n"], float(z["d"])
        Cm = ((hi + 0.5) * mc + float(z["x0"]))[:, None] * e1 + ((hj + 0.5) * mc + float(z["y0"]))[:, None] * e2 - d_ * n_; keep = np.zeros(len(Cm), bool)
        for m in kept:
            L_ = m["fr"].loc(Cm); lo, hi_ = m["L"].min(0) - 0.03, m["L"].max(0) + 0.03; keep |= np.all((L_[:, :2] >= lo[:2]) & (L_[:, :2] <= hi_[:2]), 1)
        Cm, hi, hj = Cm[keep], hi[keep], hj[keep]
        q = Rotation.from_matrix(R.T @ np.stack([e1, e2, n_], 1)).as_quat(); new = np.zeros(len(hi), dtype=v.dtype); new["x"], new["y"], new["z"] = (((Cm - t) @ R) / s).T
        new["scale_0"] = new["scale_1"] = np.log(0.7 * mc / s); new["scale_2"] = np.log(0.0008 / s); new["rot_0"], new["rot_1"], new["rot_2"], new["rot_3"] = q[3], q[0], q[1], q[2]
        new["opacity"] = np.log(0.98 / 0.02); dc = (np.clip(z["rgb_filled"][hi, hj], 0, 1) - 0.5) / C0
        for i in range(3):
            new[f"f_dc_{i}"] = dc[:, i]
        if "nx" in v.dtype.names:
            nd_ = R.T @ n_; new["nx"], new["ny"], new["nz"] = nd_[0], nd_[1], nd_[2]
        rows.append(new); rep["floor_under_objects"] = int(len(new))
    outv = np.concatenate([v[~dele]] + rows) if rows else v
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); PlyData([PlyElement.describe(outv, "vertex")], text=False).write(a.out)
    info = dict(objects=rep, rows_in=int(N0), deleted=int(dele.sum()), inserted=int(sum(len(r_) for r_ in rows)), rows_out=int(len(outv)), out=a.out)
    json.dump(info, open(os.path.splitext(a.out)[0] + "_layers.json", "w"), indent=1); print("LAYERS_COMPOSE " + json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["evidence", "candidates", "finalize", "floormap", "gate", "compose", "floorviews", "meshviews", "mergeviews", "seedpoints"])
    ap.add_argument("--dir", required=True, help="work dir of this run"); ap.add_argument("--cfg-json", default=None, help="the config's object_layers block as JSON")
    ap.add_argument("--objects"); ap.add_argument("--sim3-json"); ap.add_argument("--dataset"); ap.add_argument("--splat-depth-dir"); ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="debug override: only these objects")
    ap.add_argument("--extra-gen-dir", nargs="*", default=None, help="more generated-asset dirs (e.g. Hunyuan3D-2.1); every seed of every generator is registered");
    ap.add_argument("--train-stats", default=None, help="gate / compose / floorviews: clean_haze.py stats of the --ply (train_max_w): protects the gaussians the training views draw")
    ap.add_argument("--seg-dir", default=None, help="evidence: detect_objects.py segment output (no VLM crop from a frame with a person in it)")
    ap.add_argument("--apply-detection", action="store_true", help="evidence: split / merge clusters as the vision-language detection check says");
    ap.add_argument("--base-ply", default=None, help="floorviews: the input splat (before)"); ap.add_argument("--cloud", default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="mergeviews: DS:weight ..."); ap.add_argument("--real", default=None, help="mergeviews: real frames first"); ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--subset", type=float, default=1.0, help="seedpoints: share of the training frames kept (cheap tests)"); ap.add_argument("--edgs", default=None); ap.add_argument("--pseudo-weight", type=float, default=0.5); ap.add_argument("--round", type=int, default=1); ap.add_argument("--map"); ap.add_argument("--ply"); ap.add_argument("--out")
    a = ap.parse_args(); os.makedirs(a.dir, exist_ok=True)
    {"evidence": evidence, "candidates": candidates, "finalize": finalize, "floormap": floormap, "gate": gate, "compose": compose, "floorviews": floorviews, "meshviews": meshviews, "mergeviews": mergeviews, "seedpoints": seedpoints}[a.cmd](a)


if __name__ == "__main__":
    main()
