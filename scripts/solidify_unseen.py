"""Solidify every surface the training cameras never saw: needles / specks / random-opacity gaussians that no training
view constrains are replaced by flat opaque discs on the measured (LiDAR) surface, and free-standing objects are
closed (backs, sides) so a block looks like a block from any angle. Post-process on any standard 3DGS .ply; training
views are unchanged (measured in the viewer-exact rasteriser, scripts/finetune_ply_viewer.py ViewerRaster ss352).

  support  (gsplat venv, GPU) per-gaussian training-view support: every training view rendered with the viewer-exact
           compositor (SuperSplat 3.5.2 kernel, --display-scale, size cull OFF so a sub-pixel splat counts with its
           dilated footprint = generous), each gaussian's blending weight w = alpha*T accumulated: sum, per-pixel max,
           number of views where max > 1/255. Removing one gaussian changes any pixel by at most its w (its own term
           plus the rescaled terms behind it, both <= alpha*T), so max w < --max-weight (default 1/255, one 8-bit
           level) = UNCONSTRAINED: no training view can tell it is there.
  build    (CPU venv) within --domain-m of the LiDAR cloud (mapped space): REMOVE (candidates; harm restores what a
           training view shows) unconstrained gaussians and every gaussian farther than --free-m from any LiDAR point
           (free-space floaters), except ones already shaped like a surface disc (opaque, flat, cell-sized, on the
           LiDAR: e.g. an earlier fill); SNAP needles (s_max > --needle-ratio s_mid) whose long axis passes within
           --snap-m of the LiDAR surface into a flat disc there (s_mid in-plane, LiDAR normal, colour/opacity kept).
           Close free-standing objects (find_objects.py) with complete_objects.py's 2.5D surface (flat tops, vertical
           walls down to the floor); its samples with no LiDAR return join the measured surface. A surface point gets
           one flat opaque disc (normal from the cloud or the object's face) if the KEPT gaussians are not solid there
           (opacity a viewer looking along the normal accumulates within --window-m < --solid-alpha) and no training
           camera sees it (fill_unseen_surfaces.py's test on the input's rendered depth/opacity; views more grazing
           than --max-incidence do not count). Colour: blend of nearby kept gaussians (fill_inpaint.py recolours it);
           object discs also get a MIRROR colour: same height on the opposite face of the same object (or same
           height on the nearest seen face; tops mirrored through the centre) from the gaussians the cameras did see.
  objtex   (CPU venv) after fill_inpaint.py: object discs with a mirror source take it (brick courses continue
           round the back), the rest keep LaMa; colours smoothed over 1.5 cells on the same face.
  harm     (gsplat venv, GPU) every training view rendered with the viewer-exact rasteriser, changed pixel = any channel
           off the input's render by > 2/255, exact attribution from the compositor's (pixel, gaussian, weight) pairs:
           (A) removals only -> removed gaussians blending > --min-w into a changed pixel of the INPUT are restored,
           (M) + snapped needles -> those blending into a changed pixel (either render) are reverted, (B) + discs ->
           discs blending into a changed pixel are dropped; each repeated until nothing changes. Untouched input
           gaussians are byte-identical.
  shfit    (gsplat venv, GPU) view-dependent colour off the recorded directions: every gaussian first goes matte (its
           blend-weighted mean training colour), then the minimum-norm SH fit (colours at the directions the training
           cameras saw it from kept, no extrapolation elsewhere), then its original SH - each step only for gaussians
           that blend > --min-w into a training pixel changed by > 2/255. Removes rainbow sparkles from unseen angles.
  refit    (gsplat venv, GPU) before harm (REFIT=1 in the sbatch): fine-tune ONLY the added discs and snapped needles
           (every input gaussian frozen; harm then restores removals / reverts what still shows) - viewer-exact L1+SSIM on every training frame (display scale
           1.5:4), surface prior on discs / snapped needles (thin, normal kept, centre within --pos-tol-m of the
           surface), anti-needle term (s_max/s_mid > --needle-ratio) in object zones, and from random orbit views an
           alpha-hole + colour-smoothness term on the objects' pixels (no RGB target there).
  budget   (gsplat venv, GPU) after `refit --train-all` (JOINT=1): per-view visibility budget instead of harm's 2/255 rule -
           reverts / drops / restores only what makes a visible cluster (> 8/255) or a view lose > 0.3 dB.
  orbit    (CPU) COLMAP text cameras orbiting each object (--n-az azimuths x --heights, --dist m) in the PLY's frame
           for viewer_shots.py; per view, the angle to the nearest training camera (how unseen the view is).
  eval     (gsplat venv, GPU) at the orbit cameras, per PLY: viewer-exact alpha + depth vs the LiDAR/closure depth in
           each object's pixels (alpha-hole fraction, depth within 5 cm), needle fraction (s_max/s_min > 10) of the
           gaussians in the object zones; with --live name=<viewer_shots dir>, also the live editor's alpha.
  montage  (CPU) per object: rows of azimuths, before | after columns, from viewer_shots.py renders.

    python scripts/solidify_unseen.py support --ply in.ply --dataset <train ds> [<ds> ...] --out S/support.npz
    python scripts/solidify_unseen.py build --ply in.ply --support S/support.npz --cloud lidar_static.ply \
        --sim3-json sim3.json --objects objects --dataset <train ds> --splat-depth-dir D --out-dir S
    python scripts/fill_inpaint.py --splat-ply S/base_kept.ply --fill-ply S/raw.ply ... --out-ply S/inpainted.ply
    python scripts/solidify_unseen.py objtex --ply S/inpainted.ply --meta S/meta.npz --out S/textured.ply
    python scripts/solidify_unseen.py harm --input in.ply --cand S/textured.ply --meta S/meta.npz --dataset <ds> ... --out S/harm.ply
    python scripts/solidify_unseen.py shfit --ply S/harm.ply --ref in.ply --dataset <ds> ... --out out.ply
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model  # noqa: E402

C0 = 0.28209479177387814


def sim3(path):
    j = json.load(open(path)); return float(j["scale"]), np.array(j["R"]), np.array(j["t"])


def read_vertex(path):
    from plyfile import PlyData
    return PlyData.read(path)["vertex"].data


def write_vertex(path, a):
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(a, "vertex")], text=False).write(path)


def gauss_geom(v, s, R, t):
    """-> metric centres, opacity, metric scales (N,3), unit quats xyzw."""
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)) * s
    return s * (X @ R.T) + t, op, sc


def load_objects(prefix):
    if not prefix or not os.path.exists(prefix + ".json"):
        return None, []
    meta = json.load(open(prefix + ".json")); pts = np.load(prefix + ".npz")
    return meta, [pts[f"obj{o['id']}"] for o in meta["objects"]]


def seen_by_cameras(Pm, datasets, depth_dir, s, R, t, tol_rel, solid_opacity, min_hidden_m, normals=None, max_incidence=90.0):
    """fill_unseen_surfaces.py's test: a metric point is seen if some training camera has it in view and it is in front
    of / within min_hidden_m behind the splat's rendered surface there, or that surface is not solid. With normals, views
    more grazing than max_incidence (deg) do not count (a surface only ever seen edge-on is not constrained)."""
    seen = np.zeros(len(Pm), bool)
    if not depth_dir or not len(Pm):
        return seen
    Xd = ((Pm - t) @ R) / s
    Nd = normals @ R if normals is not None else None
    cmin = np.cos(np.radians(max_incidence))
    for ds in datasets:
        cams, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
        for name, cid, q, tc in imgs:
            f = os.path.join(depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy")
            if not os.path.exists(f):
                continue
            W, H, (fx, fy, cx, cy) = cams[cid]
            Pc = Xd @ quat_to_R(q).T + tc; z = Pc[:, 2]
            ok = (z > 0.05) & ~seen
            u = np.full(len(z), -1); vv = np.full(len(z), -1)
            u[ok] = np.floor(fx * Pc[ok, 0] / z[ok] + cx); vv[ok] = np.floor(fy * Pc[ok, 1] / z[ok] + cy)
            ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            ia = np.nonzero(ok)[0]
            if not len(ia):
                continue
            D = np.load(f).astype(np.float32)[vv[ia], u[ia]]
            fo = f.replace("_splatdepth.npy", "_splatopacity.npy")
            A = np.load(fo).astype(np.float32)[vv[ia], u[ia]] if os.path.exists(fo) else np.ones(len(ia), np.float32)
            r = np.linalg.norm(Pc[ia], axis=1)
            vis = (D <= 0) | (A < solid_opacity) | (r < D * (1 + tol_rel)) | (r < D + min_hidden_m / s)
            if Nd is not None:
                vis &= np.abs(np.einsum("nd,nd->n", Pc[ia] / r[:, None], Nd[ia] @ quat_to_R(q).T)) >= cmin
            seen[ia[vis]] = True
    return seen


def solidity(Pq, Nq, G, op, Sinv, radius, window, chunk=40000):
    """Opacity a viewer looking along each point's normal accumulates from the gaussians within `window` of the point
    along that line (metric): 1 - prod(1 - op_i exp(-m_i^2 / 2)), m_i = Mahalanobis distance of the line to gaussian i."""
    from scipy.spatial import cKDTree
    tree = cKDTree(G); out = np.zeros(len(Pq))
    for a in range(0, len(Pq), chunk):
        nb = tree.query_ball_point(Pq[a:a + chunk], radius, workers=-1)
        cnt = np.fromiter((len(x) for x in nb), int, len(nb))
        if not cnt.sum():
            continue
        qi = np.repeat(np.arange(len(nb)), cnt); gi = np.fromiter((j for x in nb for j in x), int, int(cnt.sum()))
        r = Pq[a + qi] - G[gi]; n = Nq[a + qi]; Si = Sinv[gi]
        Sr = np.einsum("pij,pj->pi", Si, r); Sn = np.einsum("pij,pj->pi", Si, n)
        aa = np.einsum("pi,pi->p", r, Sr); bb = np.einsum("pi,pi->p", n, Sr); cc = np.maximum(np.einsum("pi,pi->p", n, Sn), 1e-12)
        m2 = np.maximum(aa - bb * bb / cc, 0); sstar = -bb / cc
        al = np.where(np.abs(sstar) <= window, np.minimum(op[gi] * np.exp(-0.5 * m2), 0.999), 0.0)
        out[a:a + len(nb)] = 1 - np.exp(np.bincount(qi, np.log1p(-al), len(nb)))
    return out


# ---------------------------------------------------------------- support (GPU)
def pair_weights(Rv, p, w2c, fx, fy, cx, cy, W, H):
    """Viewer-exact compositing (ViewerRaster._exact, ss352 kernel) -> (gaussian id, pixel id, weight alpha*T) per pair."""
    import torch
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_indices_in_range
    from finetune_ply_viewer import E4
    md, conics, op, z, radii, keep = Rv.project(p, w2c, fx, fy, cx, cy, W, H)
    z_sort = torch.where(keep, z, torch.full_like(z, 1e10))
    tw, th = (W + Rv.tile - 1) // Rv.tile, (H + Rv.tile - 1) // Rv.tile
    _, isect_ids, flatten_ids = isect_tiles(md[None].contiguous(), radii[None].contiguous(), z_sort[None].contiguous(), Rv.tile, tw, th)
    offsets = isect_offset_encode(isect_ids, 1, tw, th)
    op_idx = (op / (1 - E4)).clamp_max(1.0)
    gid, pid, _ = rasterize_to_indices_in_range(0, 1 << 30, torch.ones(1, H, W, device=md.device), md[None].contiguous(),
                                                conics[None].contiguous(), op_idx[None].contiguous(), W, H, Rv.tile, offsets, flatten_ids)
    pix = torch.stack([pid % W, pid // W], -1).float() + 0.5
    d = pix - md[gid]; cq = conics[gid]
    s2 = cq[:, 0] * d[:, 0] ** 2 + 2 * cq[:, 1] * d[:, 0] * d[:, 1] + cq[:, 2] * d[:, 1] ** 2
    al = op[gid] * (torch.exp(-0.5 * s2) - E4) / (1 - E4)
    al = torch.where((s2 <= 8) & (al > 0), al, torch.zeros_like(al)).clamp(0, 1 - 1e-6)
    lt = torch.log1p(-al.double()); cs = torch.cumsum(lt, 0) - lt
    first = torch.ones_like(pid, dtype=torch.bool); first[1:] = pid[1:] != pid[:-1]
    seg = torch.cumsum(first.long(), 0) - 1
    cs = cs - cs[first][seg]
    return gid.long(), pid.long(), (al.double() * torch.exp(cs)).float()


def gt_image(ds_list, name, dev):
    """The real image of a training frame (first dataset that has it), float [0,1] (H,W,3) on dev, or None."""
    import torch
    from PIL import Image
    for ds in ds_list:
        f = os.path.join(ds, "images", name)
        if os.path.exists(f):
            return torch.tensor(np.asarray(Image.open(f).convert("RGB")), device=dev).half() / 255
    return None


def changed_mask(img, ref, thr, gt=None, eref=None):
    """Pixels the edit changed by > thr; with the real image, only those that got > thr further from it (not worse)."""
    d = (img - ref.float()).abs().amax(-1) > thr
    if gt is not None:
        d &= (img - gt.float()).abs().amax(-1) > eref.float() + thr
    return d


def all_frames(datasets):
    from finetune_ply_viewer import frames
    out, names = [], set()
    for ds in datasets:
        for f in frames(ds, 0, "all"):
            if f[0] not in names:
                names.add(f[0]); out.append(f)
    return out


def support(args):
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, to_t
    dev = "cuda"
    _, p = load_ply(args.ply, dev); N = len(p["means"])
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale, cull=False)
    fr = all_frames(args.dataset)
    wsum = torch.zeros(N, dtype=torch.float64, device=dev); wmax = torch.zeros(N, device=dev)
    nview = torch.zeros(N, dtype=torch.int32, device=dev); t0 = time.time()
    with torch.no_grad():
        for k, (name, cam, c2w) in enumerate(fr):
            gid, _, w = pair_weights(Rv, p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam))
            wsum.index_add_(0, gid, w.double()); wmax.scatter_reduce_(0, gid, w, reduce="amax")
            vm = torch.zeros(N, device=dev).scatter_reduce_(0, gid, w, reduce="amax")
            nview += (vm > 1 / 255).int()
            if k % 50 == 0:
                print(f"[support] {k + 1}/{len(fr)} {name} pairs {len(gid):,} ({time.time() - t0:.0f}s)", flush=True)
    wsum, wmax, nview = wsum.cpu().numpy(), wmax.cpu().numpy(), nview.cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, wsum=wsum.astype(np.float32), wmax=wmax.astype(np.float32), nview=nview.astype(np.int32))
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    rep = {"n": int(N), "views": len(fr), "display_scale": args.display_scale,
           "wmax_quantiles": {str(q): float(np.quantile(wmax, q)) for q in qs},
           "wsum_quantiles": {str(q): float(np.quantile(wsum, q)) for q in qs},
           "frac_wmax_below": {t: float((wmax < float(eval(t))).mean()) for t in ("1e-4", "1/255", "2/255", "0.01", "0.05", "0.1")},
           "frac_never_drawn": float((wmax == 0).mean()),
           "hist_log10_wmax": np.histogram(np.log10(np.maximum(wmax, 1e-8)), bins=np.arange(-8, 0.5, 0.5))[0].tolist(),
           "hist_log10_wsum": np.histogram(np.log10(np.maximum(wsum, 1e-8)), bins=np.arange(-8, 6.5, 0.5))[0].tolist()}
    json.dump(rep, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print("SUPPORT", json.dumps({k: v for k, v in rep.items() if not k.startswith("hist")}), flush=True)


# ---------------------------------------------------------------- build (CPU)
def disc_rows(dtype, Xd, Nd, dc, cell, s, opacity):
    """Flat opaque matte discs (fill_unseen_surfaces.py's shape) in the PLY's frame."""
    from scipy.spatial.transform import Rotation
    Nd = Nd / np.linalg.norm(Nd, axis=1, keepdims=True)
    tan = np.cross(Nd, np.array([0.0, 0.0, 1.0])); bad = np.linalg.norm(tan, axis=1) < 1e-3
    tan[bad] = np.cross(Nd[bad], np.array([1.0, 0.0, 0.0])); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    bit = np.cross(Nd, tan)
    qq = Rotation.from_matrix(np.stack([tan, bit, Nd], 2)).as_quat()
    new = np.zeros(len(Xd), dtype=dtype)
    new["x"], new["y"], new["z"] = Xd[:, 0], Xd[:, 1], Xd[:, 2]
    for i in range(3):
        new[f"f_dc_{i}"] = dc[:, i]
    if "nx" in dtype.names:
        new["nx"], new["ny"], new["nz"] = Nd[:, 0], Nd[:, 1], Nd[:, 2]
    new["opacity"] = np.log(opacity / (1 - opacity))
    ls = np.log(np.array([0.6 * cell, 0.6 * cell, 0.001]) / s)
    new["scale_0"], new["scale_1"], new["scale_2"] = ls
    new["rot_0"], new["rot_1"], new["rot_2"], new["rot_3"] = qq[:, 3], qq[:, 0], qq[:, 1], qq[:, 2]
    return new


def build(args):
    import open3d as o3d
    from scipy.spatial import cKDTree
    from complete_objects import object_surface
    t0 = time.time()
    s, R, t = sim3(args.sim3_json)
    v = read_vertex(args.ply); N = len(v)
    Gm, op, sc = gauss_geom(v, s, R, t)
    ss = np.sort(sc, 1); smin, smid, smax = ss[:, 0], ss[:, 1], ss[:, 2]
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    sup = np.load(args.support); wmax = sup["wmax"]; assert len(wmax) == N, "support was computed for another PLY"
    unc = wmax < args.max_weight
    pc = o3d.io.read_point_cloud(args.cloud).voxel_down_sample(args.cell)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * args.cell, max_nn=30))
    P = np.asarray(pc.points); Nn = np.asarray(pc.normals); ltree = cKDTree(P)
    dl = ltree.query(Gm, distance_upper_bound=args.domain_m)[0]
    disc_like = (op >= 0.9) & (smin <= 0.2 * smid) & (smax <= 2.5 * smid) & (smax <= 2 * args.cell) & (dl < 2 * args.cell)
    free = dl > args.free_m                                                # not on any measured surface
    remove = ~disc_like & (dl < args.domain_m) & (unc | (free & (wmax < args.free_max_weight)))
    keep = ~remove
    # needles / pins on a surface: a kept elongated gaussian whose long axis passes within --snap-m of the LiDAR surface
    # becomes a flat disc there (in-plane s_mid, LiDAR normal; colour, opacity kept). Seen along its axis (as the
    # training cameras mostly do) it covers about the same pixels; harm reverts every one a training view notices.
    from scipy.spatial.transform import Rotation
    q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    spike = keep & (op > 0.1) & (smax > args.needle_ratio * smid) & (dl < args.domain_m)
    si = np.nonzero(spike)[0]
    mod = np.zeros(N, bool); vmod = v.copy()
    if len(si):
        Rl = Rotation.from_quat(q[si]).as_matrix()                         # local -> dataset frame
        ax = Rl[np.arange(len(si)), :, np.argmax(sc[si], 1)] @ R.T           # long axis, metric
        best_d = np.full(len(si), np.inf); best_i = np.zeros(len(si), int)
        for tau in np.linspace(-2.5, 2.5, 11):
            dd_, ii_ = ltree.query(Gm[si] + tau * smax[si, None] * ax)
            b = dd_ < best_d; best_d[b] = dd_[b]; best_i[b] = ii_[b]
        ok = best_d < args.snap_m
        si, bi = si[ok], best_i[ok]
        Nl = Nn[bi] / np.maximum(np.linalg.norm(Nn[bi], axis=1, keepdims=True), 1e-12)
        Xn = ((P[bi] - t) @ R) / s
        disc = disc_rows(v.dtype, Xn, Nl @ R, dc[si], args.cell, s, 0.5)
        for k in ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3") + (("nx", "ny", "nz") if "nx" in v.dtype.names else ()):
            vmod[k][si] = disc[k]
        ls = np.log(np.stack([smid[si], smid[si], np.maximum(smin[si], 0.001)], 1) / s)
        for i in range(3):
            vmod[f"scale_{i}"][si] = ls[:, i]
        mod[si] = True
        Gm[si] = P[bi]                                                     # coverage below sees the snapped discs
    print(f"[solid] {spike.sum():,} kept spikes (s_max > {args.needle_ratio} s_mid); {mod.sum():,} snapped onto the LiDAR surface", flush=True)
    needle = smax / np.maximum(smin, 1e-9) > 10
    st = {"gaussians_in": int(N), "unconstrained": int(unc.sum()), "unconstrained_disc_like_kept": int((unc & disc_like).sum()),
          "unconstrained_outside_domain_kept": int((unc & ~disc_like & ~(dl < args.domain_m)).sum()),
          "removed": int(remove.sum()), "removed_free_space_candidates": int((remove & free & ~unc).sum()),
          "removed_near_surface": int((remove & (dl < args.cover_m)).sum()),
          "removed_free_space": int((remove & ~(dl < args.cover_m)).sum()),
          "needle_frac_unconstrained": round(float(needle[unc].mean()) if unc.any() else 0.0, 4),
          "needle_frac_constrained": round(float(needle[~unc].mean()), 4),
          "median_opacity_unconstrained": round(float(np.median(op[unc])) if unc.any() else 0.0, 4),
          "median_opacity_constrained": round(float(np.median(op[~unc])), 4), "max_weight": args.max_weight,
          "spikes_kept": int(spike.sum()), "needles_snapped": int(mod.sum())}
    print(f"[solid] {N:,} gaussians; {unc.sum():,} unconstrained (max w < {args.max_weight:.4g}), {(unc & disc_like).sum():,} of "
          f"them disc-like (kept); removing {remove.sum():,} ({st['removed_near_surface']:,} on a LiDAR surface, "
          f"{st['removed_free_space']:,} in free space)", flush=True)

    # surface points: LiDAR voxels + object closure samples the LiDAR never measured
    obj = np.full(len(P), -1); face = np.full(len(P), -2); loc = np.zeros((len(P), 3))
    ometa, opts = load_objects(args.objects)
    extra_P, extra_N, extra_o, extra_f, extra_l = [], [], [], [], []
    osamp = []                                                             # every closure sample (texture sources)
    if ometa is not None and len(opts):
        n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
        for k, (o, X) in enumerate(zip(ometa["objects"], opts)):
            pos, nor, kind, fc, local, centre = object_surface(X, n_f, d_f, args.cell, args.cell)
            local = local - np.array([centre[0], centre[1], 0.0])           # object-centred (u, v, height)
            fc = np.where(kind == 0, -1, fc)
            ctree = cKDTree(pos)
            dd, ci = ctree.query(P, distance_upper_bound=1.5 * args.cell)
            m = np.isfinite(dd) & (obj < 0)
            obj[m] = k; face[m] = fc[ci[m]]; loc[m] = local[ci[m]]; Nn[m] = nor[ci[m]]  # crisp faces for its LiDAR points
            un = ltree.query(pos)[0] > args.support_m
            extra_P.append(pos[un]); extra_N.append(nor[un]); extra_o.append(np.full(un.sum(), k)); extra_f.append(fc[un]); extra_l.append(local[un])
            osamp.append((np.full(len(pos), k), pos, fc, local))
            o_st = {"id": o["id"], "samples": int(len(pos)), "unsupported_added": int(un.sum())}
            st.setdefault("objects", []).append(o_st)
        P = np.concatenate([P] + extra_P); Nn = np.concatenate([Nn] + extra_N); obj = np.concatenate([obj] + extra_o)
        face = np.concatenate([face] + extra_f); loc = np.concatenate([loc] + extra_l)
    st["surface_points"] = int(len(P)); st["closure_points"] = int(sum(map(len, extra_P)))

    # represented = the KEPT gaussians form an opaque surface there (seen head-on along its normal, within --window-m);
    # candidates for a disc: not represented and no training camera sees the point (grazing views excluded)
    from scipy.spatial.transform import Rotation as Rot
    ki = np.nonzero(keep)[0]
    Gk, _, sck = gauss_geom(vmod[ki], s, R, t)
    qk = np.stack([vmod["rot_1"][ki], vmod["rot_2"][ki], vmod["rot_3"][ki], vmod["rot_0"][ki]], 1).astype(np.float64)
    Rk = R @ Rot.from_quat(qk / np.maximum(np.linalg.norm(qk, axis=1, keepdims=True), 1e-12)).as_matrix()
    Sinv = np.einsum("nij,nj,nkj->nik", Rk, 1.0 / np.maximum(sck, 1e-5) ** 2, Rk)
    Nu = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
    dk = cKDTree(Gk).query(P, distance_upper_bound=args.cover_m)[0]
    sol = np.zeros(len(P)); near_k = np.nonzero(np.isfinite(dk))[0]
    sol[near_k] = solidity(P[near_k], Nu[near_k], Gk, op[ki], Sinv, args.solid_radius_m, args.window_m)
    covered = sol >= args.solid_alpha
    cand = np.nonzero(~covered)[0]
    seen_c = seen_by_cameras(P[cand], args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity,
                             args.min_hidden_m, normals=Nu[cand], max_incidence=args.max_incidence)
    add = cand[~seen_c]
    kk = keep & (op > args.cover_opacity)
    gtree = cKDTree(Gm[kk]); kidx = np.nonzero(kk)[0]
    _, gi = gtree.query(P)
    st.update({"solid_fraction": round(float(covered.mean()), 4), "not_solid": int(len(cand)), "not_solid_seen": int(seen_c.sum()),
               "discs": int(len(add)), "discs_on_objects": int((obj[add] >= 0).sum()),
               "solidity_hist": np.histogram(sol, bins=np.linspace(0, 1, 11))[0].tolist()})
    print(f"[solid] {len(P):,} surface points ({st['closure_points']:,} object closure); {covered.mean():.1%} solid in the kept "
          f"gaussians; {len(cand):,} not solid, {seen_c.sum():,} of them seen by a training camera; {len(add):,} discs "
          f"({st['discs_on_objects']:,} on objects)", flush=True)

    # blend colour from covered points (their nearest kept gaussian), normal-weighted (fill_unseen_surfaces.py)
    cov = np.nonzero(covered)[0]
    col_cov = dc[kidx[gi[cov]]]
    dd, ii = cKDTree(P[cov]).query(P[add], k=args.k)
    nag = np.abs(np.einsum("nkd,nd->nk", Nn[cov][ii], Nn[add]))
    w = np.exp(-dd / max(np.median(dd[:, 0]), args.cell)) * (nag ** 4 + 1e-3)
    col = (w[..., None] * col_cov[ii]).sum(1) / w.sum(1, keepdims=True)

    # object mirror colours from the seen (camera-constrained) part of the same object
    mir = np.full((len(add), 3), np.nan); src = np.zeros(len(add), np.int8)
    if osamp:
        so, sp, sf, sl = (np.concatenate(a) for a in zip(*osamp))
        seen_s = seen_by_cameras(sp, args.dataset, args.splat_depth_dir, s, R, t, args.tol_rel, args.solid_opacity, 0.0)
        cons = (~unc) & (op > args.cover_opacity)
        ctree = cKDTree(Gm[cons]); cidx = np.nonzero(cons)[0]
        nb = ctree.query_ball_point(sp, args.tex_radius_m)
        scol = np.full((len(sp), 3), np.nan)
        for i in np.nonzero(seen_s)[0]:
            j = cidx[np.asarray(nb[i], int)]
            if len(j):
                ww = op[j] * np.exp(-0.5 * (np.linalg.norm(Gm[j] - sp[i], axis=1) / (args.tex_radius_m / 2)) ** 2)
                scol[i] = (ww[:, None] * dc[j]).sum(0) / max(ww.sum(), 1e-9)
        obs = np.isfinite(scol[:, 0])
        st["object_samples_observed"] = int(obs.sum()); st["object_samples"] = int(len(sp))
        ao = obj[add]; af = face[add]; al = loc[add]
        for k in np.unique(ao[ao >= 0]):
            mk = obs & (so == k)
            if not mk.any():
                continue
            ok_idx = np.nonzero(ao == k)[0]
            for fk in (-1, 0, 1, 2, 3):
                ii_ = ok_idx[af[ok_idx] == fk]
                if not len(ii_):
                    continue
                q = al[ii_].copy()
                if fk == -1:
                    q[:, :2] *= -1; tgt = [-1]
                else:
                    q[:, fk // 2] *= -1; tgt = [fk ^ 1]
                for stage, faces in ((1, tgt), (2, [f for f in (0, 1, 2, 3) if f != fk] if fk >= 0 else [-1])):
                    todo = ii_[~np.isfinite(mir[ii_, 0])]
                    if not len(todo):
                        break
                    ms = mk & np.isin(sf, faces)
                    if not ms.any():
                        continue
                    if stage == 1:
                        qq = q[np.isin(ii_, todo)]
                        dd_, jj = cKDTree(sl[ms]).query(qq, k=4, distance_upper_bound=args.mirror_tol_m)
                    else:                                                       # same height (course) on another seen face
                        qq = al[todo].copy()
                        sc_ = sl[ms].copy()
                        sc_[:, 2] *= 50.0; qq[:, 2] *= 50.0                     # height must match (50x weight)
                        dd_, jj = cKDTree(sc_).query(qq, k=4, distance_upper_bound=50 * args.cell + 1.0)
                    okm = np.isfinite(dd_)
                    jj = np.where(okm, jj, 0); hit = okm.any(1)
                    cc = (okm[..., None] * scol[ms][jj]).sum(1) / np.maximum(okm.sum(1, keepdims=True), 1)
                    mir[todo[hit]] = cc[hit]; src[todo[hit]] = stage
        st["mirror_colours"] = {"opposite_face": int((src == 1).sum()), "same_height_other_face": int((src == 2).sum()),
                                "none": int(((obj[add] >= 0) & (src == 0)).sum())}
        print(f"[solid] object texture sources: {st['mirror_colours']}", flush=True)

    Xd = ((P[add] - t) @ R) / s; Nd = Nn[add] @ R
    new = disc_rows(v.dtype, Xd, Nd, col, args.cell, s, args.opacity)
    os.makedirs(args.out_dir, exist_ok=True)
    write_vertex(os.path.join(args.out_dir, "base_kept.ply"), vmod[keep])
    write_vertex(os.path.join(args.out_dir, "raw.ply"), np.concatenate([vmod[keep], new]))
    np.savez_compressed(os.path.join(args.out_dir, "meta.npz"), keep=keep, mod=mod, removed_near_surface=remove & (dl < args.cover_m),
                        disc_obj=obj[add], disc_face=face[add], disc_local=loc[add], disc_metric=P[add], disc_normal=Nn[add],
                        disc_mirror=mir, disc_src=src, cell=args.cell, sim3=np.array([s]), n_in=N)
    st["seconds"] = round(time.time() - t0, 1)
    json.dump(st, open(os.path.join(args.out_dir, "build.json"), "w"), indent=1)
    print("SOLID_BUILD", json.dumps({k: v_ for k, v_ in st.items() if k != "objects"}), flush=True)


# ---------------------------------------------------------------- objtex (CPU)
def objtex(args):
    from scipy.spatial import cKDTree
    v = read_vertex(args.ply).copy(); M = np.load(args.meta)
    nk = int(M["keep"].sum()); m = len(v) - nk
    assert m == len(M["disc_obj"]), "--ply must be base_kept + the build's discs (fill_inpaint.py keeps the order)"
    dcf = np.stack([v[f"f_dc_{i}"][nk:] for i in range(3)], 1).astype(np.float64)
    mir = M["disc_mirror"]; has = np.isfinite(mir[:, 0])
    new = dcf.copy(); new[has] = mir[has]
    obj, fc = M["disc_obj"], M["disc_face"]; X = M["disc_metric"]; cell = float(M["cell"])
    if args.smooth > 0:                                                    # de-speckle object discs on the same face
        for k in np.unique(obj[obj >= 0]):
            for f in np.unique(fc[obj == k]):
                ii = np.nonzero((obj == k) & (fc == f))[0]
                if len(ii) < 3:
                    continue
                nb = cKDTree(X[ii]).query_ball_point(X[ii], args.smooth * cell)
                src = new[ii].copy()
                new[ii] = np.array([src[np.asarray(j)].mean(0) for j in nb])
    for i in range(3):
        v[f"f_dc_{i}"][nk:] = new[:, i].astype(np.float32)
    write_vertex(args.out, v)
    print("SOLID_OBJTEX", json.dumps({"discs": int(m), "mirror_coloured": int(has.sum()), "object_discs": int((obj >= 0).sum()),
                                      "mean_abs_rgb_change": round(float(C0 * np.abs(new - dcf).mean()), 4)}), flush=True)


# ---------------------------------------------------------------- harm (GPU)
def harm(args):
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, sh_deg, to_t
    dev = "cuda"
    vi, pi = load_ply(args.input, dev); vc, pc = load_ply(args.cand, dev)
    M = np.load(args.meta); keep0 = torch.tensor(M["keep"], device=dev); N = len(pi["means"]); nk = int(keep0.sum())
    mod0 = torch.tensor(M["mod"] if "mod" in M.files else np.zeros(N, bool), device=dev)
    m = len(pc["means"]) - nk
    kidx = torch.nonzero(keep0)[:, 0]
    pmod = {k: v.clone() for k, v in pi.items()}                          # input with the snapped needles
    for k in pmod:
        pmod[k][kidx] = pc[k][:nk]
    um = keep0 & ~mod0
    assert torch.equal(pmod["means"][um], pi["means"][um]), "--cand must start with the kept input gaussians"
    disc = {k: v[nk:] for k, v in pc.items()}
    deg = sh_deg(pi)
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    fr = all_frames(args.dataset)[::args.every]
    cams = [(name, to_t(np.linalg.inv(c2w), dev), cam_args(cam)) for name, cam, c2w in fr]
    with torch.no_grad():
        ref = {n: Rv.render(pi, w2c, *ca, sh_degree=deg)[0].half() for n, w2c, ca in cams}
        gts = {n: gt_image(args.dataset, n, dev) for n, _, _ in cams} if args.gt_aware else {}
        eref = {n: (ref[n].float() - g.float()).abs().amax(-1).half() for n, g in gts.items() if g is not None}

    def check(p, attrib):
        """changed-pixel share vs the input; attrib(w2c, cam, changed mask) -> flagged gaussians (exact attribution)."""
        flagged = None; changed = total = 0
        for n, w2c, ca in cams:
            img = Rv.render(p, w2c, *ca, sh_degree=deg)[0]
            d = changed_mask(img, ref[n], args.thr, gts.get(n), eref.get(n))
            changed += int(d.sum()); total += d.numel()
            if d.any():
                f = attrib(w2c, ca, d.reshape(-1)); flagged = f if flagged is None else flagged | f
        return changed / max(total, 1), flagged

    def contributors(p, w2c, ca, d, n):
        """gaussians of p blending > --min-w into a changed pixel (pairs of the viewer-exact compositor)."""
        gid, pid, w = pair_weights(Rv, p, w2c, *ca)
        hit = d[pid] & (w > args.min_w)
        out = torch.zeros(n, dtype=torch.bool, device=dev); out[gid[hit]] = True
        return out

    def loop(phase, build_p, attrib, active):
        for it in range(args.iters):
            p = build_p()
            share, fl = check(p, lambda w2c, ca, d: attrib(p, w2c, ca, d) & active)
            nfl = int(fl.sum()) if fl is not None else 0
            hist.append({"phase": phase, "iter": it, "changed_pixel_share": share, "active": int(active.sum()), "flagged": nfl})
            print(f"[solid-harm] {phase} iter {it}: changed px {share:.2e}, active {int(active.sum()):,}, flagged {nfl:,}", flush=True)
            if not nfl:
                break
            active &= ~fl

    hist = []
    removed = ~keep0                                                       # active = still removed / snapped / added
    mod = mod0.clone(); dkeep = torch.ones(m, dtype=torch.bool, device=dev)
    mix = lambda: {k: torch.where(mod.view(-1, *[1] * (v.dim() - 1)), pmod[k], v) for k, v in pi.items()}
    with torch.no_grad():
        # (A) removals: restore every removed gaussian that blends into a changed pixel of the INPUT's render
        loop("removal", lambda: {k: v[~removed] for k, v in pi.items()},
             lambda p, w2c, ca, d: contributors(pi, w2c, ca, d, N), removed)
        # (M) snapped needles: revert those blending into a changed pixel in either render
        def att_m(p, w2c, ca, d):
            sel = torch.nonzero(~removed)[:, 0]
            c = torch.zeros(N, dtype=torch.bool, device=dev); c[sel[contributors(p, w2c, ca, d, len(sel))]] = True
            return c | contributors(pi, w2c, ca, d, N)
        loop("needles", lambda: {k: v[~removed] for k, v in mix().items()}, att_m, mod)
        # (B) discs: drop those blending into a changed pixel
        def build_b():
            q = mix(); return {k: torch.cat([q[k][~removed], disc[k][dkeep]]) for k in q}

        def att_b(p, w2c, ca, d):
            nb = int((~removed).sum()); didx = torch.nonzero(dkeep)[:, 0]
            c = contributors(p, w2c, ca, d, len(p["means"]))[nb:]
            out = torch.zeros(m, dtype=torch.bool, device=dev); out[didx[c]] = True
            return out
        loop("discs", build_b, att_b, dkeep)
    k_np, mod_np, d_np = (~removed).cpu().numpy(), mod.cpu().numpy(), dkeep.cpu().numpy()
    vin = vi.data.copy(); vcd = vc.data
    pos = np.cumsum(M["keep"]) - 1                                         # row of input i in --cand
    mi = np.nonzero(mod_np)[0]; vin[mi] = vcd[pos[mi]]
    write_vertex(args.out, np.concatenate([vin[k_np], vcd[nk:][d_np]]))
    res = {"input": int(N), "removed_final": int((~k_np).sum()), "restored": int(k_np.sum() - nk), "needles_in": int(mod0.sum()),
           "needles_snapped": int(mod_np.sum()), "discs_in": int(m), "discs_kept": int(d_np.sum()),
           "out_gaussians": int(k_np.sum() + d_np.sum()), "views": len(cams), "display_scale": args.display_scale,
           "thr": args.thr, "min_w": args.min_w, "iterations": hist}
    np.savez_compressed(os.path.splitext(args.out)[0] + "_harm_masks.npz", keep=k_np, mod=mod_np, disc_keep=d_np)
    json.dump(res, open(os.path.splitext(args.out)[0] + "_harm.json", "w"), indent=1)
    print("SOLID_HARM", json.dumps({k: v for k, v in res.items() if k != "iterations"}), flush=True)


# ---------------------------------------------------------------- shfit (GPU)
def sh_basis(dirs, K):
    """(n,3) unit view directions -> (n, 1 + K) basis of the viewers' SH evaluation (rgb = 0.5 + basis . coeffs)."""
    import torch
    from finetune_ply_viewer import C0, C1, C2, C3
    x, y, z = dirs.unbind(-1); xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    b = [torch.full_like(x, C0), -C1 * y, C1 * z, -C1 * x,
         C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
         C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
         C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return torch.stack(b[:1 + K], -1)


def shfit(args):
    """Minimum-norm view-dependent colour: per gaussian, the SH coefficients that reproduce its colours at the directions
    the training cameras saw it from (weighted by its blend weight there), with a ridge on bands >= 1 - directions no
    camera constrained no longer extrapolate into rainbow sparkles. x = (A + lam tr D)^-1 A x0, A = sum_v w_v B_v B_v^T.
    Then measured: SH of gaussians blending into a changed training pixel are reverted until nothing changes."""
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, sh_deg, to_t
    dev = "cuda"
    vtx, p = load_ply(args.ply, dev); N = len(p["means"]); K = p["shN"].shape[1]
    _, pref = load_ply(args.ref, dev); deg = sh_deg(p)
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    cams = [(n, to_t(np.linalg.inv(c2w), dev), cam_args(cam), c2w) for n, cam, c2w in all_frames(args.dataset)]
    A = torch.zeros(N, 1 + K, 1 + K, device=dev)
    with torch.no_grad():
        for n, w2c, ca, c2w in cams:
            gid, _, w = pair_weights(Rv, p, w2c, *ca)
            wv = torch.zeros(N, device=dev).index_add_(0, gid, w)
            ii = torch.nonzero(wv > 1e-4)[:, 0]
            d = p["means"][ii] - torch.tensor(c2w[:3, 3], dtype=torch.float32, device=dev)
            B = sh_basis(d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12), K)
            A[ii] += wv[ii, None, None] * B[:, :, None] * B[:, None, :]
        X0 = torch.cat([p["sh0"], p["shN"]], 1)                            # (N, 1+K, 3)
        Dg = torch.ones(1 + K, device=dev); Dg[0] = 0
        lam = args.lam * A[:, 0, 0] / (C0 ** 2)                            # x sum of weights
        Xn = torch.empty_like(X0)                                          # tier 1: minimum-norm fit
        for a in range(0, N, 200000):
            Aa = A[a:a + 200000]
            L = Aa + lam[a:a + 200000, None, None] * torch.diag(Dg)[None] + 1e-8 * torch.eye(1 + K, device=dev)[None]
            Xn[a:a + 200000] = torch.linalg.solve(L.double(), (Aa.double() @ X0[a:a + 200000].double())).float()
        unseen = A[:, 0, 0] <= 0
        Xm = torch.zeros_like(X0)                                          # tier 0: matte, the weighted mean colour
        Xm[:, 0] = torch.einsum("nk,nkc->nc", A[:, 0], X0) / A[:, 0, 0].clamp_min(1e-12)[:, None]
        Xm[unseen, 0] = X0[unseen, 0]; Xn[unseen] = Xm[unseen]             # never drawn: matte
        tiers = torch.stack([Xm, Xn, X0])                                  # matte -> min-norm -> original
        del A
        ref = {n: Rv.render(pref, w2c, *ca, sh_degree=deg)[0].half() for n, w2c, ca, _ in cams}
        gts = {n: gt_image(args.dataset, n, dev) for n, _, _, _ in cams} if args.gt_aware else {}
        eref = {n: (ref[n].float() - g.float()).abs().amax(-1).half() for n, g in gts.items() if g is not None}
        tier = torch.zeros(N, dtype=torch.long, device=dev); hist = []
        ar = torch.arange(N, device=dev)
        for it in range(args.iters):
            X = tiers[tier, ar]; q = dict(p); q["sh0"], q["shN"] = X[:, :1].contiguous(), X[:, 1:].contiguous()
            fl = torch.zeros(N, dtype=torch.bool, device=dev); changed = total = 0
            for n, w2c, ca, _ in cams:
                img = Rv.render(q, w2c, *ca, sh_degree=deg)[0]
                dd = changed_mask(img, ref[n], args.thr, gts.get(n), eref.get(n)).reshape(-1)
                changed += int(dd.sum()); total += dd.numel()
                if dd.any():
                    gid, pid, w = pair_weights(Rv, q, w2c, *ca)
                    fl[gid[dd[pid] & (w > args.min_w)]] = True
            fl &= tier < 2
            hist.append({"iter": it, "changed_pixel_share": changed / max(total, 1), "matte": int((tier == 0).sum()),
                         "min_norm": int((tier == 1).sum()), "original": int((tier == 2).sum()), "flagged": int(fl.sum())})
            print(f"[shfit] iter {it}: changed px {changed / max(total, 1):.2e}, matte {int((tier == 0).sum()):,}, "
                  f"min-norm {int((tier == 1).sum()):,}, original {int((tier == 2).sum()):,}, flagged {int(fl.sum()):,}", flush=True)
            if not fl.any():
                break
            tier[fl] += 1
        X = tiers[tier, ar]
        out = dict(p); out["sh0"], out["shN"] = X[:, :1].contiguous(), X[:, 1:].contiguous()
        e0 = p["shN"].norm(dim=(1, 2)); e1 = out["shN"].norm(dim=(1, 2))
    from finetune_ply_viewer import save_ply
    save_ply(args.out, vtx, out)
    res = {"gaussians": N, "matte": int((tier == 0).sum()), "min_norm": int((tier == 1).sum()), "original": int((tier == 2).sum()),
           "never_drawn": int(unseen.sum()), "lam": args.lam,
           "shN_norm_median_before": float(e0.median()), "shN_norm_median_after": float(e1.median()), "iterations": hist}
    json.dump(res, open(os.path.splitext(args.out)[0] + "_shfit.json", "w"), indent=1)
    print("SOLID_SHFIT", json.dumps({k: v for k, v in res.items() if k != "iterations"}), flush=True)


# ---------------------------------------------------------------- refit (GPU)
def refit(args):
    """Constrained fine-tune of the build (all discs, all snapped needles, all removals) instead of harm's keep/revert:
    viewer-exact photometric loss on every training frame (seen views are re-fitted), a surface prior on the discs and
    snapped needles (thin, normal along the LiDAR / height-map normal, centre within --pos-tol-m of the surface), an
    anti-needle term on gaussians in object zones, and from random virtual orbit views around the objects (no RGB
    target) an alpha-hole and a colour-smoothness term on the objects' pixels."""
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, save_ply, sh_deg, to_t, quat_to_rot
    from refine_poses_gsplat import load_rgb, ssim, psnr
    from build_depth_targets import zbuffer_target
    from complete_objects import object_surface
    from scipy.spatial import cKDTree
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"
    s, R, t = sim3(args.sim3_json)
    vtx, p0 = load_ply(args.ply, dev); deg = sh_deg(p0)
    M = np.load(args.meta); keep = M["keep"]; nk = int(keep.sum()); m = len(p0["means"]) - nk
    mod = M["mod"][keep] if "mod" in M.files else np.zeros(nk, bool)
    src = np.concatenate([np.nonzero(keep)[0], np.full(m, -1)])           # input row of each candidate row (-1 = disc)
    nres = 0
    if args.input and args.restore_masks:                                  # put back the removals harm found visible
        from plyfile import PlyData
        rk = np.load(args.restore_masks)["keep"] & ~keep; ri = np.nonzero(rk)[0]; nres = len(ri)
        vin, pin_ = load_ply(args.input, dev)
        vd = vtx.data
        vtx = np.concatenate([vd[:nk], vin.data[ri], vd[nk:]])
        p0 = {k: torch.cat([p0[k][:nk], pin_[k][torch.tensor(ri, device=dev)], p0[k][nk:]]) for k in p0}
        mod = np.concatenate([mod, np.zeros(nres, bool)]); src = np.concatenate([src[:nk], ri, src[nk:]])
        del pin_
    N = len(p0["means"])
    surf = torch.tensor(np.concatenate([mod, np.ones(m, bool)]), device=dev)
    si = torch.nonzero(surf)[:, 0]
    cell_d = float(M["cell"]) / s; tol_d = args.pos_tol_m / s
    with torch.no_grad():
        Rq = quat_to_rot(p0["quats"][si])
        n0 = Rq[torch.arange(len(si)), :, p0["scales"][si].argmin(1)]       # initial thin axis = surface normal
        x0 = p0["means"][si].clone()
    # object zones and orbit masks
    ometa, opts = load_objects(args.objects)
    zone = torch.zeros(N, dtype=torch.bool, device=dev); tzone = torch.zeros_like(zone); orb = []
    if ometa is not None and len(opts):
        n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
        osurf = np.concatenate([object_surface(X, n_f, d_f, 0.02, 0.02)[0] for X in opts])
        Gm = s * (p0["means"].double().cpu().numpy() @ R.T) + t
        dz = cKDTree(osurf).query(Gm, distance_upper_bound=max(args.zone_m, args.train_zone_m))[0]
        zone = torch.tensor(dz < args.zone_m, device=dev)
        tzone = torch.tensor(dz < args.train_zone_m, device=dev)
        if args.orbit and args.lam_hole + args.lam_tv > 0:
            from plyfile import PlyData
            cv = PlyData.read(args.cloud)["vertex"].data; L = np.stack([cv["x"], cv["y"], cv["z"]], 1).astype(np.float64)
            L = L[np.unique(np.floor(L / 0.02).astype(np.int64), axis=0, return_index=True)[1]]
            to_d = lambda A: torch.tensor(((A - t) @ R) / s, device=dev)
            allX, objX = to_d(np.concatenate([L, osurf])), to_d(osurf)
            for name, cam, c2w in frames(args.orbit, 0, "all"):
                fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w)
                Rc = torch.tensor(w2c[:3, :3], device=dev); tc = torch.tensor(w2c[:3, 3], device=dev)
                Da, _ = zbuffer_target(allX, Rc, tc, W, H, (fx, fy, cx, cy), 0.02 / s, 1e6)
                Do, _ = zbuffer_target(objX, Rc, tc, W, H, (fx, fy, cx, cy), 0.02 / s, 1e6, min_support=3)
                mk = (Do > 0) & (Da > 0) & ((Do - Da).abs() < 0.03 / s)
                if mk.sum() > 200:
                    orb.append((to_t(w2c, dev), (fx, fy, cx, cy, W, H), mk))
    tm = (surf | (tzone if args.train_zone_m > 0 else torch.ones_like(surf))) if args.train_all else surf   # trainable rows
    slow = tm & ~surf                                                      # input gaussians that may move (slowly, anchored)
    print(f"[refit] trainable {int(tm.sum()):,}, of them input gaussians {int(slow.sum()):,}", flush=True)
    print(f"[refit] {N:,} gaussians ({nk:,} kept input incl. {int(mod.sum()):,} snapped needles, {m:,} discs); surface prior on "
          f"{len(si):,}; object zone {int(zone.sum()):,}; {len(orb)} orbit views", flush=True)
    tr = all_frames(args.dataset)
    def gt_of(n):
        for ds in args.dataset:
            f = os.path.join(ds, "images", n)
            if os.path.exists(f):
                return f
    rgb = [load_rgb(gt_of(n), dev) for n, _, _ in tr]
    centres = np.stack([c[:3, 3] for _, _, c in tr]); scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    lr = {"means": 1.6e-5 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    rs = [float(x) for x in args.display_scale.split(":")]
    Rv = ViewerRaster("ss352", "exact", display_scale=rs[0])
    zi = torch.nonzero(zone)[:, 0]; log3 = float(np.log(args.needle_ratio))
    order, t0, log = [], time.time(), []
    for step in range(1, args.iters + 1):
        if not order:
            order = list(np.random.permutation(len(tr)))
        k = int(order.pop()); name, cam, c2w = tr[k]
        Rv.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[-1]))))
        wu = min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu * (0.1 ** (step / args.iters) if g["name"] == "means" else 1.0)
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = rgb[k].float() / 255
        lph = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        # surface prior
        sc = params["scales"][si]; ax = quat_to_rot(params["quats"][si])
        thin = ax[torch.arange(len(si)), :, sc.detach().argmin(1)]
        lflat = (torch.exp(sc.min(1).values) / cell_d).mean()
        lnorm = (1 - (thin * n0).sum(-1).abs()).mean()
        dx = params["means"][si] - x0; dn = (dx * n0).sum(-1); dt = (dx - dn[:, None] * n0).norm(dim=-1)
        lpos = ((torch.relu(dn.abs() - tol_d) ** 2).mean() + (torch.relu(dt - 2 * tol_d) ** 2).mean()) / tol_d ** 2
        ss_ = torch.sort(params["scales"][zi], 1).values
        lneed = torch.relu(ss_[:, 2] - ss_[:, 1] - log3).mean() if len(zi) else torch.zeros((), device=dev)
        loss = lph + args.lam_flat * lflat + args.lam_normal * lnorm + args.lam_pos * lpos + args.lam_needle * lneed
        la = torch.zeros((), device=dev)
        if args.train_all and args.lam_anchor > 0:                         # input gaussians stay near their trained values
            fz = slow
            la = (((params["means"][fz] - p0["means"][fz]) / cell_d) ** 2).sum() + ((params["scales"][fz] - p0["scales"][fz]) ** 2).sum() \
                + ((params["opacities"][fz] - p0["opacities"][fz]) ** 2).sum() + ((params["quats"][fz] - p0["quats"][fz]) ** 2).sum() \
                + ((params["sh0"][fz] - p0["sh0"][fz]) ** 2).sum() + ((params["shN"][fz] - p0["shN"][fz]) ** 2).sum()
            loss = loss + args.lam_anchor * la
        lh = lt = torch.zeros((), device=dev)
        if orb and step % args.orbit_every == 0:
            w2c_o, ca, mk = orb[np.random.randint(len(orb))]
            io, ao, _ = Rv.render(params, w2c_o, *ca, sh_degree=deg)
            lh = (1 - ao.clamp(0, 1))[mk].mean()
            gx = (io[:, 1:] - io[:, :-1]).abs().sum(-1)[mk[:, 1:] & mk[:, :-1]]
            gy = (io[1:] - io[:-1]).abs().sum(-1)[mk[1:] & mk[:-1]]
            lt = torch.cat([gx, gy]).mean()
            loss = loss + args.lam_hole * lh + args.lam_tv * lt
        opt.zero_grad(set_to_none=True); loss.backward()
        for prm in params.values():
            if prm.grad is not None:
                prm.grad[~torch.isfinite(prm.grad)] = 0
                prm.grad[~tm] = 0                                          # frozen rows (Adam: exactly still)
        if args.train_all and args.lr_input < 1:
            old_ = {k: v.detach()[slow].clone() for k, v in params.items()}
        opt.step()
        if args.train_all and args.lr_input < 1:                           # input gaussians move at lr_input x the rate
            with torch.no_grad():
                for k_, v_ in params.items():
                    v_[slow] = old_[k_] + args.lr_input * (v_[slow] - old_[k_])
            del old_
        if step % args.log_every == 0 or step == 1:
            e = {"step": step, "psnr": round(psnr(img.detach(), gt), 2), "flat": round(float(lflat), 4), "normal": round(float(lnorm), 4),
                 "pos": round(float(lpos), 4), "needle": round(float(lneed), 4), "hole": round(float(lh), 4), "tv": round(float(lt), 4), "anchor": round(float(la), 2)}
            log.append(e); print(f"[refit] {e} ({time.time() - t0:.0f}s, {torch.cuda.max_memory_allocated() / 2**30:.1f}G)", flush=True)
    with torch.no_grad():                                                  # held-out-free check: mean PSNR over the training views
        Rv.r = rs[0]; pv = [psnr(Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255)
                            for k, (n, cam, c2w) in list(enumerate(tr))[::10]]
        p0v = [psnr(Rv.render(p0, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0], rgb[k].float() / 255)
               for k, (n, cam, c2w) in list(enumerate(tr))[::10]]
    with torch.no_grad():                                                  # exact copy-back of the frozen rows
        for k_ in params:
            params[k_][~tm] = p0[k_][~tm]
    save_ply(args.out, vtx, params)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_src.npz", src=src)
    res = {"gaussians": N, "restored_rows": nres, "surface_prior": int(len(si)), "object_zone": int(zone.sum()), "orbit_views": len(orb), "iters": args.iters,
           "train_psnr_every10_before": round(float(np.mean(p0v)), 3), "train_psnr_every10_after": round(float(np.mean(pv)), 3),
           "seconds": round(time.time() - t0), "curve": log, "args": {k: v for k, v in vars(args).items()}}
    json.dump(res, open(os.path.splitext(args.out)[0] + "_refit.json", "w"), indent=1)
    print("SOLID_REFIT", json.dumps({k: v for k, v in res.items() if k not in ("curve", "args")}), flush=True)


# ---------------------------------------------------------------- budget (GPU)
def budget(args):
    """Per-view visibility budget for a joint refit (refit --train-all): a training pixel counts as changed if it moved
    > --vis-thr from the input's render inside a cluster (3x3 opening), or > --thr anywhere in a view whose PSNR vs the
    real image dropped by > --view-db. Gaussians blending > --min-w into changed pixels are reverted to the input's
    parameters (input rows), dropped (discs), or restored (removed input rows the input's render shows there); repeated."""
    import torch
    import torch.nn.functional as F
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, sh_deg, to_t
    from refine_poses_gsplat import psnr
    dev = "cuda"
    vi, pi = load_ply(args.input, dev); vc, pc = load_ply(args.cand, dev); N = len(pi["means"])
    src = torch.tensor(np.load(args.src)["src"], device=dev); isin = src >= 0
    ci = torch.full((N,), -1, dtype=torch.long, device=dev); ci[src[isin]] = torch.nonzero(isin)[:, 0]
    present = ci >= 0; use_c = present.clone(); di = torch.nonzero(~isin)[:, 0]; dkeep = torch.ones(len(di), dtype=torch.bool, device=dev)
    deg = sh_deg(pi); Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    cams = [(n, to_t(np.linalg.inv(c2w), dev), cam_args(cam)) for n, cam, c2w in all_frames(args.dataset)]

    def build():
        q = {}
        for k in pi:
            a = pi[k].clone(); u = present & use_c
            a[u] = pc[k][ci[u]]
            q[k] = torch.cat([a[present], pc[k][di[dkeep]]])
        return q
    with torch.no_grad():
        ref = {n: Rv.render(pi, w2c, *ca, sh_degree=deg)[0].half() for n, w2c, ca in cams}
        gts = {n: gt_image(args.dataset, n, dev) for n, _, _ in cams}
        hist = []
        for it in range(args.iters):
            q = build(); pidx = torch.nonzero(present)[:, 0]; npres = len(pidx); dsel = di[dkeep]
            rev = torch.zeros(N, dtype=torch.bool, device=dev); drop = torch.zeros(len(di), dtype=torch.bool, device=dev)
            res_ = torch.zeros(N, dtype=torch.bool, device=dev); drops, nchg = [], 0
            for n, w2c, ca in cams:
                img = Rv.render(q, w2c, *ca, sh_degree=deg)[0]; d = (img - ref[n].float()).abs().amax(-1)
                g = gts[n]; db = psnr(ref[n].float(), g.float()) - psnr(img, g.float()) if g is not None else 0.0
                drops.append(db)
                big = (d > args.vis_thr).float()[None, None]
                op_ = F.max_pool2d(-F.max_pool2d(-big, 3, 1, 1), 3, 1, 1)[0, 0] > 0      # clusters only (3x3 opening)
                chg = op_ | ((d > args.thr) if db > args.view_db else torch.zeros_like(op_))
                if not chg.any():
                    continue
                nchg += int(chg.sum()); cf = chg.reshape(-1)
                gid, pid, w = pair_weights(Rv, q, w2c, *ca); h = gid[cf[pid] & (w > args.min_w)]
                hi = h[h < npres]; rev[pidx[hi]] = True
                hd = h[h >= npres] - npres; full = torch.nonzero(dkeep)[:, 0]; drop[full[hd]] = True
                gid, pid, w = pair_weights(Rv, pi, w2c, *ca); res_[gid[cf[pid] & (w > args.min_w)]] = True
            rev |= res_ & present                                          # moved away from where the input drew them
            res_ &= ~present; rev &= use_c
            hist.append({"iter": it, "changed_px": nchg, "views_over_db": int(sum(x > args.view_db for x in drops)),
                         "mean_view_drop_db": float(np.mean(drops)), "max_view_drop_db": float(np.max(drops)),
                         "revert": int(rev.sum()), "drop_discs": int(drop.sum()), "restore": int(res_.sum())})
            print(f"[budget] {hist[-1]}", flush=True)
            if not (rev.any() or drop.any() or res_.any()):
                break
            use_c &= ~rev; dkeep &= ~drop; present |= res_
    pr, uc = present.cpu().numpy(), use_c.cpu().numpy(); cin = ci.cpu().numpy()
    rows = vi.data.copy(); u = pr & uc; rows[u] = vc.data[cin[u]]
    out = np.concatenate([rows[pr], vc.data[di.cpu().numpy()][dkeep.cpu().numpy()]])
    write_vertex(args.out, out)
    r = {"input": N, "present": int(pr.sum()), "refit_rows_kept": int(u.sum()), "reverted_to_input": int((pr & ~uc).sum()),
         "discs_in": int(len(di)), "discs_kept": int(dkeep.sum()), "out": int(len(out)), "iterations": hist}
    json.dump(r, open(os.path.splitext(args.out)[0] + "_budget.json", "w"), indent=1)
    print("SOLID_BUDGET", json.dumps({k: v for k, v in r.items() if k != "iterations"}), flush=True)


# ---------------------------------------------------------------- orbit cameras (CPU)
def look_at(C, target, up):
    from scipy.spatial.transform import Rotation
    z = target - C; z /= np.linalg.norm(z)
    y = -up - (-up @ z) * z
    y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-6 else np.cross(z, [1.0, 0, 0])
    x = np.cross(y, z); Rcw = np.stack([x, y, z]); q = Rotation.from_matrix(Rcw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]]), -Rcw @ C


def orbit(args):
    s, R, t = sim3(args.sim3_json)
    ometa, opts = load_objects(args.objects)
    assert ometa is not None and len(opts), "no objects"
    n_up = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    W, H, (fx, fy, cx, cy) = cams[imgs[0][1]]
    Ctr = np.array([s * ((-quat_to_R(q).T @ tc) @ R.T) + t for _, _, q, tc in imgs])   # training centres, metric
    Rtr = [quat_to_R(q) for _, _, q, _ in imgs]; Ttr = [tc for _, _, _, tc in imgs]
    e1 = np.cross(n_up, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(n_up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n_up, e1)
    ltree = None
    if args.cloud and os.path.exists(args.cloud):
        from plyfile import PlyData
        from scipy.spatial import cKDTree
        cv = PlyData.read(args.cloud)["vertex"].data; L_ = np.stack([cv["x"], cv["y"], cv["z"]], 1).astype(np.float64)
        ltree = cKDTree(L_[np.unique(np.floor(L_ / 0.02).astype(np.int64), axis=0, return_index=True)[1]])
    os.makedirs(os.path.join(args.out, "sparse", "0"), exist_ok=True)
    with open(os.path.join(args.out, "sparse", "0", "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    rows, info = [], []
    for o, X in zip(ometa["objects"], opts):
        h = X @ n_up + d_f; foot = X - np.outer(h, n_up)
        c = foot.mean(0) + n_up * 0.5 * h.max()                             # aim at mid-height of the footprint centre
        rad = 0.5 * float(np.hypot(*o["footprint_m"]))
        dist0 = max(args.dist, 1.2 * rad + 0.6)
        for hh in args.heights:
            for a in range(args.n_az):
                th = 2 * np.pi * a / args.n_az
                # closest-first free line of sight: no LiDAR point within --clear-m of the eye -> object segment
                for dist in np.arange(dist0, rad + 0.45, -0.15):
                    Cm = foot.mean(0) + dist * (np.cos(th) * e1 + np.sin(th) * e2) + hh * n_up   # foot points lie on the floor
                    if ltree is None:
                        break
                    L = np.linalg.norm(c - Cm); u_ = (c - Cm) / L
                    pts = Cm + np.outer(np.arange(0.0, max(L - rad - 0.1, 0.0), 0.03), u_)
                    if not len(pts) or ltree.query(pts)[0].min() > args.clear_m:
                        break
                else:
                    continue                                                # no clear view from this azimuth
                q, tt = look_at(((Cm - t) @ R) / s, ((c - t) @ R) / s, R.T @ n_up)
                name = f"o{o['id']}_h{int(round(hh * 100)):03d}_a{a:02d}.png"
                rows.append(f"{len(rows) + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in tt)} 1 {name}\n\n")
                dv = Cm - c; dv /= np.linalg.norm(dv)
                cd = ((c - t) @ R) / s                                         # training cams with the object in view
                inv = []
                for Rc, tc in zip(Rtr, Ttr):
                    pc_ = Rc @ cd + tc
                    inv.append(pc_[2] > 0.05 and 0 <= fx * pc_[0] / pc_[2] + cx < W and 0 <= fy * pc_[1] / pc_[2] + cy < H)
                near = Ctr[np.array(inv) & (np.linalg.norm(Ctr - c, axis=1) < 4.0)]
                ang = float(np.degrees(np.arccos(np.clip(((near - c) / np.linalg.norm(near - c, axis=1, keepdims=True)) @ dv, -1, 1)).min())) if len(near) else 180.0
                info.append({"name": name, "object": o["id"], "azimuth_deg": round(np.degrees(th), 1), "eye_h": hh,
                             "dist_m": round(dist, 2), "min_angle_to_training_cam_deg": round(ang, 1)})
    open(os.path.join(args.out, "sparse", "0", "images.txt"), "w").writelines(rows)
    open(os.path.join(args.out, "sparse", "0", "points3D.txt"), "w").write("")
    json.dump(info, open(os.path.join(args.out, "orbit.json"), "w"), indent=1)
    print(f"[orbit] {len(rows)} cameras around {len(opts)} objects -> {args.out}", flush=True)


# ---------------------------------------------------------------- eval (GPU)
def eval_(args):
    import torch
    from PIL import Image
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t
    from build_depth_targets import zbuffer_target
    from complete_objects import object_surface
    dev = "cuda"
    s, R, t = sim3(args.sim3_json)
    ometa, opts = load_objects(args.objects)
    n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    info = {d["name"]: d for d in json.load(open(os.path.join(args.orbit, "orbit.json")))}
    # expected surface: LiDAR (2 cm voxels) + every object's closed 2.5D surface, dataset units
    from plyfile import PlyData
    cv = PlyData.read(args.cloud)["vertex"].data; L = np.stack([cv["x"], cv["y"], cv["z"]], 1).astype(np.float64)
    _, ui = np.unique(np.floor(L / args.cell).astype(np.int64), axis=0, return_index=True); L = L[ui]
    osurf = [object_surface(X, n_f, d_f, args.cell, args.cell)[0] for X in opts]
    to_d = lambda A: torch.tensor(((A - t) @ R) / s, device=dev)
    allX = to_d(np.concatenate([L] + osurf))
    zones = []
    for X, S in zip(opts, osurf):                                          # object zone (metric): footprint box + 5 cm
        h = S @ n_f + d_f; zones.append((S, float(h.max())))
    fr = frames(args.orbit, 0, "all")
    Rv = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    res = {}
    for spec in args.ply:
        tag, path = spec.split("=", 1)
        _, p = load_ply(path, dev); deg = sh_deg(p)
        Gm = s * (p["means"].double().cpu().numpy() @ R.T) + t
        sc = torch.exp(p["scales"]).cpu().numpy(); op = torch.sigmoid(p["opacities"]).cpu().numpy()
        from scipy.spatial import cKDTree
        ss = np.sort(sc, 1)
        needle = ss[:, 2] / np.maximum(ss[:, 0], 1e-12) > 10               # brief's definition (also counts flat discs)
        spike = ss[:, 2] / np.maximum(ss[:, 1], 1e-12) > 4                 # 1-D elongated: needles / pins, not surfels
        nz, nn, nsp = 0, 0, 0
        for S, hmax in zones:
            dd = cKDTree(S).query(Gm, distance_upper_bound=0.08)[0]
            z = np.isfinite(dd) & (op > 0.1); nz += int(z.sum()); nn += int((z & needle).sum()); nsp += int((z & spike).sum())
        per = {"needle_frac_object_zone": round(nn / max(nz, 1), 4), "spike_frac_object_zone": round(nsp / max(nz, 1), 4),
               "gaussians_object_zone": nz, "views": []}
        live = dict(x.split("=", 1) for x in args.live) if args.live else {}
        for name, cam, c2w in fr:
            fx, fy, cx, cy, W, H = cam_args(cam)
            w2c = np.linalg.inv(c2w); ok_ = to_t(w2c, dev)
            with torch.no_grad():
                _, alpha, dep = Rv.render(p, ok_, fx, fy, cx, cy, W, H, sh_degree=deg, depth=True)
            oid = info[name]["object"]; k = [o["id"] for o in ometa["objects"]].index(oid)
            Rcw = torch.tensor(w2c[:3, :3], device=dev); tcw = torch.tensor(w2c[:3, 3], device=dev)
            Dall, _ = zbuffer_target(allX, Rcw, tcw, W, H, (fx, fy, cx, cy), args.cell / s, 1e6)
            Dobj, _ = zbuffer_target(to_d(osurf[k]), Rcw, tcw, W, H, (fx, fy, cx, cy), args.cell / s, 1e6, min_support=3)
            mask = (Dobj > 0) & (Dall > 0) & ((Dobj - Dall).abs() < 0.03 / s)  # the object is the front surface
            if mask.sum() < 50:
                continue
            uu, vv = torch.meshgrid(torch.arange(W, device=dev) + 0.5, torch.arange(H, device=dev) + 0.5, indexing="xy")
            ray = torch.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1)
            dr = dep * ray
            hole = (alpha < 0.98)[mask].float().mean().item()
            err = ((dr - Dall).abs() * s)[mask & (alpha > 0.5)]
            row = {"name": name, "px": int(mask.sum()), "alpha_hole": round(hole, 4),
                   "depth_within_5cm": round(float((err < 0.05).float().mean()) if len(err) else 0.0, 4),
                   "depth_median_err_m": round(float(err.median()) if len(err) else 9.9, 4),
                   "min_angle_to_training_cam_deg": info[name]["min_angle_to_training_cam_deg"]}
            if tag in live:
                af = os.path.join(live[tag], "_alpha", os.path.splitext(name)[0] + ".png")
                if os.path.exists(af):
                    A = torch.tensor(np.asarray(Image.open(af)), device=dev).float() / 255
                    row["live_alpha_hole"] = round(float((A < 0.98)[mask].float().mean()), 4)
            per["views"].append(row)
        V = per["views"]
        unseen = [r for r in V if r["min_angle_to_training_cam_deg"] >= args.unseen_deg]
        agg = lambda rows, k: round(float(np.mean([r[k] for r in rows])), 4) if rows and all(k in r for r in rows) else None
        per["summary"] = {sub: {k: agg(rows, k) for k in ("alpha_hole", "live_alpha_hole", "depth_within_5cm", "depth_median_err_m")}
                          | {"n": len(rows)} for sub, rows in (("all", V), (f"unseen_ge{int(args.unseen_deg)}deg", unseen))}
        res[tag] = per
        print(f"[solid-eval] {tag}: needle frac {per['needle_frac_object_zone']}, spike frac {per['spike_frac_object_zone']} "
              f"({nz:,} in zones); {json.dumps(per['summary'])}", flush=True)
        del p; torch.cuda.empty_cache()
    json.dump(res, open(args.json_out, "w"), indent=1)
    print("SOLID_EVAL", json.dumps({k: {"needle": v["needle_frac_object_zone"], "spike": v["spike_frac_object_zone"], **v["summary"]}
                                    for k, v in res.items()}), flush=True)


# ---------------------------------------------------------------- montage (CPU)
def montage(args):
    import cv2
    cols = [x.split("=", 1) for x in args.dirs]
    names = sorted(f for f in os.listdir(cols[0][1]) if f.endswith(".png") and not f.startswith("_"))
    groups = {}
    for n in names:
        g = re.match(r"(o\d+_h\d+)_a\d+", n)
        groups.setdefault(g.group(1) if g else "views", []).append(n)
    os.makedirs(args.out_dir, exist_ok=True)
    for g, ns in groups.items():
        rows = []
        for n in ns:
            ims = []
            for lab, d in cols:
                im = cv2.imread(os.path.join(d, n))
                im = np.zeros((480, 640, 3), np.uint8) if im is None else im
                im = cv2.resize(im, (args.w, int(args.w * im.shape[0] / im.shape[1])))
                cv2.putText(im, f"{lab} {n[:-4]}", (6, 20), 0, 0.55, (0, 0, 255), 2); ims.append(im)
            rows.append(np.hstack(ims))
        # two azimuth columns per row block to keep the montage compact
        half = (len(rows) + 1) // 2
        if len(rows) > 1:
            L, Rr = rows[:half], rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
            rows = [np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b]) for a, b in zip(L, Rr)]
        cv2.imwrite(os.path.join(args.out_dir, f"{g}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[montage] {len(groups)} montages -> {args.out_dir}", flush=True)


def crops(args):
    """On-path before | after | real | |diff| x4 crops around the most-changed region of the --n most-changed views."""
    import cv2
    names = sorted(f for f in os.listdir(args.after) if f.endswith(".png") and not f.startswith("_"))
    sc = []
    for n in names:
        a = cv2.imread(os.path.join(args.before, n)).astype(np.float32); b = cv2.imread(os.path.join(args.after, n)).astype(np.float32)
        d = np.abs(a - b).max(2); db = cv2.blur(d, (41, 41)); y, x = np.unravel_index(np.argmax(db), db.shape)
        sc.append((float(db.max()), n, y, x))
    rows = []
    for _, n, y, x in sorted(sc, reverse=True)[:args.n]:
        ims = [cv2.imread(os.path.join(d_, n)) for d_ in (args.before, args.after)]
        g = cv2.imread(os.path.join(args.gt, n)) if args.gt else None
        g = cv2.resize(g, ims[0].shape[1::-1]) if g is not None else np.zeros_like(ims[0])
        H, W = ims[0].shape[:2]; h = args.size // 2; y0 = int(np.clip(y - h, 0, H - args.size)); x0 = int(np.clip(x - h, 0, W - args.size))
        dif = np.clip(np.abs(ims[0].astype(int) - ims[1].astype(int)) * 4, 0, 255).astype(np.uint8)
        tiles = [cv2.resize(im[y0:y0 + args.size, x0:x0 + args.size], (2 * args.size, 2 * args.size), interpolation=cv2.INTER_NEAREST)
                 for im in (ims[0], ims[1], g, dif)]
        for t_, l in zip(tiles, ("before", "after", "real", "|diff|x4")):
            cv2.putText(t_, f"{l} {n[:-4][-14:]}", (4, 16), 0, 0.45, (0, 0, 255), 1)
        rows.append(np.hstack(tiles))
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[crops] {len(rows)} most-changed on-path views -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("support")
    a.add_argument("--ply", required=True); a.add_argument("--dataset", nargs="+", required=True); a.add_argument("--out", required=True)
    a.add_argument("--display-scale", type=float, default=1.8)
    b = sub.add_parser("build")
    b.add_argument("--ply", required=True); b.add_argument("--support", required=True)
    b.add_argument("--cloud", required=True); b.add_argument("--sim3-json", required=True)
    b.add_argument("--objects", default=None, help="find_objects.py prefix (objects are closed); missing = skip")
    b.add_argument("--dataset", nargs="+", default=[], help="training frames (visibility test)")
    b.add_argument("--splat-depth-dir", default=None, help="render_ply_depth.py output for --ply at --dataset")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--max-weight", type=float, default=1 / 255, help="unconstrained: max per-pixel blend weight below this")
    b.add_argument("--domain-m", type=float, default=1.0, help="only remove within this distance of the LiDAR cloud (mapped space)")
    b.add_argument("--free-m", type=float, default=0.10, help="farther than this from every LiDAR point = free space")
    b.add_argument("--free-max-weight", type=float, default=1.0,
                   help="free-space gaussians below this max weight are removal CANDIDATES (harm restores what shows)")
    b.add_argument("--needle-ratio", type=float, default=4.0, help="spike: s_max > ratio x s_mid")
    b.add_argument("--snap-m", type=float, default=0.03, help="spike axis this close to the LiDAR surface -> snapped to a disc")
    b.add_argument("--cell", type=float, default=0.02)
    b.add_argument("--cover-m", type=float, default=0.05, help="no kept gaussian this close = not represented")
    b.add_argument("--solid-alpha", type=float, default=0.7, help="represented: kept gaussians' opacity along the normal >= this")
    b.add_argument("--solid-radius-m", type=float, default=0.10); b.add_argument("--window-m", type=float, default=0.06)
    b.add_argument("--max-incidence", type=float, default=75.0, help="views more grazing than this (deg) do not count as seeing")
    b.add_argument("--cover-opacity", type=float, default=0.3)
    b.add_argument("--support-m", type=float, default=0.03, help="closure sample with a LiDAR return this close = measured")
    b.add_argument("--tol-rel", type=float, default=0.05); b.add_argument("--solid-opacity", type=float, default=0.98)
    b.add_argument("--min-hidden-m", type=float, default=0.05, help="hidden by less than this behind a seen surface = seen")
    b.add_argument("--k", type=int, default=24); b.add_argument("--opacity", type=float, default=0.95)
    b.add_argument("--tex-radius-m", type=float, default=0.03, help="object texture: seen gaussians within this of a sample")
    b.add_argument("--mirror-tol-m", type=float, default=0.04)
    o = sub.add_parser("objtex")
    o.add_argument("--ply", required=True); o.add_argument("--meta", required=True); o.add_argument("--out", required=True)
    o.add_argument("--smooth", type=float, default=1.5, help="average object-disc colours within this many cells (0: off)")
    h = sub.add_parser("harm")
    h.add_argument("--input", required=True); h.add_argument("--cand", required=True); h.add_argument("--meta", required=True)
    h.add_argument("--dataset", nargs="+", required=True); h.add_argument("--out", required=True)
    h.add_argument("--display-scale", type=float, default=1.8); h.add_argument("--thr", type=float, default=2.0 / 255)
    h.add_argument("--iters", type=int, default=6); h.add_argument("--every", type=int, default=1)
    h.add_argument("--min-w", type=float, default=0.5 / 255, help="a gaussian blending more than this into a changed pixel is flagged")
    h.add_argument("--gt-aware", action="store_true",
                   help="only pixels that move > thr FURTHER from the real image count as changed (not-worse, instead of unchanged)")
    f = sub.add_parser("shfit")
    f.add_argument("--ply", required=True); f.add_argument("--ref", required=True, help="the input splat (training views must match it)")
    f.add_argument("--dataset", nargs="+", required=True); f.add_argument("--out", required=True)
    f.add_argument("--lam", type=float, default=1e-4, help="min-norm tier: ridge on bands >= 1, x the gaussian's total blend weight")
    f.add_argument("--display-scale", type=float, default=1.8); f.add_argument("--thr", type=float, default=2.0 / 255)
    f.add_argument("--min-w", type=float, default=0.5 / 255); f.add_argument("--iters", type=int, default=8)
    f.add_argument("--gt-aware", action="store_true",
                   help="only pixels that move > thr FURTHER from the real image count as changed (not-worse, instead of unchanged)")
    g = sub.add_parser("refit")
    g.add_argument("--ply", required=True, help="build output after colouring (textured.ply)"); g.add_argument("--meta", required=True)
    g.add_argument("--dataset", nargs="+", required=True); g.add_argument("--out", required=True)
    g.add_argument("--sim3-json", required=True); g.add_argument("--objects", default=None); g.add_argument("--cloud", default=None)
    g.add_argument("--orbit", default=None, help="orbit cameras (solidify_unseen.py orbit) for the alpha-hole / smoothness terms")
    g.add_argument("--iters", type=int, default=3000); g.add_argument("--display-scale", default="1.5:4")
    g.add_argument("--pos-tol-m", type=float, default=0.015); g.add_argument("--zone-m", type=float, default=0.08)
    g.add_argument("--needle-ratio", type=float, default=3.0)
    g.add_argument("--lam-flat", type=float, default=0.1); g.add_argument("--lam-normal", type=float, default=0.1)
    g.add_argument("--lam-pos", type=float, default=0.1); g.add_argument("--lam-needle", type=float, default=0.05)
    g.add_argument("--lam-hole", type=float, default=0.1); g.add_argument("--lam-tv", type=float, default=0.05)
    g.add_argument("--orbit-every", type=int, default=2); g.add_argument("--log-every", type=int, default=250)
    g.add_argument("--train-all", action="store_true", help="also fine-tune the input's gaussians (changes seen views)")
    g.add_argument("--train-zone-m", type=float, default=0.0,
                   help="--train-all: only input gaussians within this of an object's closed surface train (0 = all)")
    g.add_argument("--lr-input", type=float, default=0.1, help="--train-all: input gaussians' learning-rate multiplier")
    g.add_argument("--lam-anchor", type=float, default=1e-2, help="--train-all: squared deviation from the input's parameters (sum)")
    g.add_argument("--input", default=None, help="the input splat (with --restore-masks)")
    g.add_argument("--restore-masks", default=None, help="harm's _harm_masks.npz: removals it restored start restored")
    bu = sub.add_parser("budget")
    bu.add_argument("--input", required=True); bu.add_argument("--cand", required=True); bu.add_argument("--src", required=True)
    bu.add_argument("--dataset", nargs="+", required=True); bu.add_argument("--out", required=True)
    bu.add_argument("--vis-thr", type=float, default=8 / 255); bu.add_argument("--thr", type=float, default=2 / 255)
    bu.add_argument("--view-db", type=float, default=0.3); bu.add_argument("--min-w", type=float, default=0.02)
    bu.add_argument("--display-scale", type=float, default=1.8); bu.add_argument("--iters", type=int, default=6)
    r = sub.add_parser("orbit")
    r.add_argument("--objects", required=True); r.add_argument("--sim3-json", required=True)
    r.add_argument("--dataset", required=True, help="a dataset in the PLY's frame (intrinsics, training centres)")
    r.add_argument("--out", required=True); r.add_argument("--n-az", type=int, default=8)
    r.add_argument("--heights", type=float, nargs="+", default=[0.5, 0.9]); r.add_argument("--dist", type=float, default=1.2)
    r.add_argument("--cloud", default=None, help="metric LiDAR cloud: cameras need a free line of sight to the object")
    r.add_argument("--clear-m", type=float, default=0.08)
    e = sub.add_parser("eval")
    e.add_argument("--ply", nargs="+", required=True, help="tag=path"); e.add_argument("--orbit", required=True)
    e.add_argument("--objects", required=True); e.add_argument("--sim3-json", required=True); e.add_argument("--cloud", required=True)
    e.add_argument("--live", nargs="*", default=[], help="tag=<viewer_shots out dir> (its _alpha/)")
    e.add_argument("--cell", type=float, default=0.02); e.add_argument("--display-scale", type=float, default=1.8)
    e.add_argument("--unseen-deg", type=float, default=45.0); e.add_argument("--json-out", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--before", required=True); c.add_argument("--after", required=True); c.add_argument("--gt", default=None)
    c.add_argument("--out", required=True); c.add_argument("--n", type=int, default=8); c.add_argument("--size", type=int, default=160)
    mo = sub.add_parser("montage")
    mo.add_argument("--dirs", nargs="+", required=True, help="label=dir"); mo.add_argument("--out-dir", required=True)
    mo.add_argument("--w", type=int, default=400)
    args = ap.parse_args()
    {"support": support, "build": build, "objtex": objtex, "harm": harm, "shfit": shfit, "refit": refit, "budget": budget, "orbit": orbit, "eval": eval_, "montage": montage, "crops": crops}[args.cmd](args)


if __name__ == "__main__":
    main()
