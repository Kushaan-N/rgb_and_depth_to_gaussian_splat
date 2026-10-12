"""Re-texture the unseen-surface fill discs (fill_unseen_surfaces.py) by image inpainting (GPU optional; venv-inpaint).

fill_unseen_surfaces.py colours each disc with a neighbour blend, which smears/speckles on large unseen areas
(centre floor, object backs, shelf tops). Here the discs keep their geometry and only their colour changes:

  PATCHES  fill discs grouped into surface patches: neighbours within 2.5 x spacing whose normals agree
           (--normal-deg), as connected components (leftovers: by proximity alone); each gets its PCA plane.
  VIEW     per patch, an orthographic texture of the splat on that plane (--px m/pixel, --context-m around the
           patch): every opaque base gaussian in a slab around the plane, facing along its normal, is splatted
           with its SH-DC colour (rgb = 0.5 + C0 dc); small gaps are closed by normalised convolution; the discs'
           footprint + pixels without splat weight are the hole. Too little observed around the patch: the slab
           and context widen (and the normal test is dropped) up to twice.
  INPAINT  LaMa (big-lama TorchScript, texture continuation, no objects hallucinated into a plane texture) or
           OpenCV's shift-map / Telea (--method) fills the hole from the observed texture around it.
  COLOUR   each disc samples the inpainted texture at its projection -> SH DC = (rgb - 0.5) / C0.
Only the fill discs' f_dc change; every other gaussian (all the camera-supported splat) is copied untouched,
so real views are unchanged. Discs with too little observed context keep the blend colour, de-speckled by
averaging over neighbouring fill discs (--smooth-m).

    python scripts/fill_inpaint.py --splat-ply X_filled.ply --fill-ply X_complete.ply --sim3-json sim3.json \
        --out-ply X_inpainted.ply [--method lama --lama-pt big-lama.pt --debug-dir dbg/]
Round trip (does SuperSplat show rgb = 0.5 + C0 dc?): --roundtrip-make DIR --dataset <ds>, render DIR/roundtrip.ply
with viewer_shots.py --every 100000, then --roundtrip-check <shots dir> --roundtrip-make DIR.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C0 = 0.28209479177387814


def read_vertices(path):
    from plyfile import PlyData
    return PlyData.read(path)["vertex"].data


def segment(X, N, spacing, normal_deg, min_pts):
    """Connected components of fill points: neighbours within 2.5 spacing with agreeing (unoriented) normals."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pr = cKDTree(X).query_pairs(2.5 * spacing, output_type="ndarray")
    pr = pr[np.abs((N[pr[:, 0]] * N[pr[:, 1]]).sum(1)) > np.cos(np.radians(normal_deg))]
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    _, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    order = np.argsort(-cnt)
    return [np.nonzero(lab == l)[0] for l in order if cnt[l] >= min_pts], int((cnt < min_pts).sum())


def splat_texture(u, v, rgb, w, sig, H, W):
    """Normalised gaussian splatting of points (pixel coords u, v; sigma px) -> colour image, weight image."""
    acc = np.zeros((H * W, 3)); wsum = np.zeros(H * W)
    r = np.clip(np.ceil(2 * sig), 1, 6).astype(int)
    for rr in np.unique(r):
        s = r == rr; us, vs, ws, ss, cs = u[s], v[s], w[s], sig[s], rgb[s]
        u0, v0 = np.round(us).astype(int), np.round(vs).astype(int)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                x, y = u0 + dx, v0 + dy
                ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                k = ws[ok] * np.exp(-0.5 * ((x[ok] - us[ok]) ** 2 + (y[ok] - vs[ok]) ** 2) / ss[ok] ** 2)
                idx = y[ok] * W + x[ok]
                wsum += np.bincount(idx, k, H * W)
                for c in range(3):
                    acc[:, c] += np.bincount(idx, k * cs[ok, c], H * W)
    return (acc / np.maximum(wsum, 1e-9)[:, None]).reshape(H, W, 3), wsum.reshape(H, W)


class Inpainter:
    def __init__(self, method, lama_pt):
        self.method = method
        if method == "lama":
            import torch
            self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(lama_pt, map_location=self.dev).eval()
            print(f"[inpaint] LaMa {lama_pt} on {self.dev}", flush=True)

    def __call__(self, img, hole):
        """img HxWx3 float 0..1, hole HxW bool -> inpainted HxWx3 float 0..1."""
        import cv2
        if self.method == "lama":
            torch = self.torch; H, W = hole.shape; ph, pw = (-H) % 8, (-W) % 8
            im = np.pad(img * ~hole[..., None], ((0, ph), (0, pw), (0, 0)), mode="symmetric")
            mk = np.pad(hole.astype(np.float32), ((0, ph), (0, pw)), mode="symmetric")
            with torch.no_grad():
                out = self.model(torch.from_numpy(im).float().permute(2, 0, 1)[None].to(self.dev),
                                 torch.from_numpy(mk)[None, None].to(self.dev))
            out = out[0].permute(1, 2, 0).float().cpu().numpy()[:H, :W]
            out = np.clip(out if out.max() > 1.5 else out * 255, 0, 255) / 255.0   # big-lama.pt returns 0..255
        else:
            u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            if self.method == "shiftmap":
                dst = np.zeros_like(u8)
                cv2.xphoto.inpaint(cv2.cvtColor(u8, cv2.COLOR_RGB2LAB), (~hole).astype(np.uint8), dst,
                                   cv2.xphoto.INPAINT_SHIFTMAP)
                out = cv2.cvtColor(dst, cv2.COLOR_LAB2RGB) / 255.0
            else:
                out = cv2.inpaint(u8, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA) / 255.0
        return np.where(hole[..., None], out, img)


def bilinear(img, u, v):
    H, W = img.shape[:2]; u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int); fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def roundtrip_make(out_dir, dataset):
    """Nine big opaque flat gaussians of known SH DC, 2 units in front of the dataset's first camera."""
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    from build_depth_targets import quat_to_R, read_text_model
    cams, imgs = read_text_model(os.path.join(dataset, "sparse", "0"))
    name, cid, q, t = sorted(imgs, key=lambda i: i[0])[0]
    W, H, (fx, fy, cx, cy) = cams[cid]; Rcw = quat_to_R(q); D = 2.0
    rgb = np.array([[.1, .1, .1], [.25, .25, .25], [.5, .5, .5], [.75, .75, .75], [.9, .9, .9],
                    [.8, .2, .2], [.2, .7, .3], [.2, .3, .8], [.6, .5, .3]])
    gx, gy = np.meshgrid([-.6, 0, .6], [-.45, 0, .45]); Pc = np.stack([gx.ravel(), gy.ravel(), np.full(9, D)], 1)
    Xw = (Pc - t) @ Rcw                                                      # camera -> world
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] \
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(9, dtype=[(n, "f4") for n in names])
    v["x"], v["y"], v["z"] = Xw.T
    nz = Rcw.T @ np.array([0, 0, 1.0]); v["nx"], v["ny"], v["nz"] = nz
    dc = (rgb - 0.5) / C0
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc.T
    v["opacity"] = np.log(0.999 / 0.001)
    v["scale_0"] = v["scale_1"] = np.log(0.1); v["scale_2"] = np.log(0.001)
    qq = Rotation.from_matrix(Rcw.T).as_quat()                               # disc axes = camera axes
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = qq[3], qq[0], qq[1], qq[2]
    os.makedirs(out_dir, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(out_dir, "roundtrip.ply"))
    uv = np.stack([fx * Pc[:, 0] / D + cx, fy * Pc[:, 1] / D + cy], 1)
    json.dump({"image": name, "uv": uv.tolist(), "rgb": rgb.tolist()}, open(os.path.join(out_dir, "roundtrip.json"), "w"))
    print("[roundtrip] ->", os.path.join(out_dir, "roundtrip.ply"))


def roundtrip_check(shots, make_dir):
    import cv2
    j = json.load(open(os.path.join(make_dir, "roundtrip.json")))
    im = cv2.imread(os.path.join(shots, os.path.splitext(j["image"])[0] + ".png"))[..., ::-1] / 255.0
    got = np.array([im[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3].reshape(-1, 3).mean(0) for u, v in j["uv"]])
    exp = np.array(j["rgb"]); err = np.abs(got - exp)
    srgb = np.where(exp <= 0.0031308, 12.92 * exp, 1.055 * exp ** (1 / 2.4) - 0.055)
    res = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
           "mean_abs_err_if_srgb_encoded": float(np.abs(got - srgb).mean()),
           "rows": [{"expected": e.round(3).tolist(), "got": g.round(3).tolist()} for e, g in zip(exp, got)]}
    print("ROUNDTRIP", json.dumps(res))
    json.dump(res, open(os.path.join(make_dir, "roundtrip_result.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat-ply", help="the splat before the unseen fill (fill_unseen_surfaces.py --splat-ply)")
    ap.add_argument("--fill-ply", help="fill_unseen_surfaces.py output: the same splat + appended fill discs")
    ap.add_argument("--sim3-json", help="COLMAP -> metric Sim3 (patch sizes and the slab are in metres)")
    ap.add_argument("--out-ply")
    ap.add_argument("--method", default="lama", choices=["lama", "shiftmap", "telea"])
    ap.add_argument("--lama-pt", default=None, help="big-lama TorchScript (env/make_inpaint_venv.sh)")
    ap.add_argument("--px", type=float, default=0.01, help="texture resolution (m/pixel); coarsened for big patches")
    ap.add_argument("--max-side", type=int, default=1536, help="largest texture side (pixels)")
    ap.add_argument("--context-m", type=float, default=0.6, help="observed surroundings around a patch (m), min")
    ap.add_argument("--slab-m", type=float, default=0.06, help="splat gaussians this close to the patch plane")
    ap.add_argument("--normal-deg", type=float, default=30.0, help="patch: neighbour normals within this angle")
    ap.add_argument("--min-pts", type=int, default=30, help="smaller patches keep their blend colour")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--known-w", type=float, default=0.5, help="splat weight per pixel to count as observed")
    ap.add_argument("--min-known", type=float, default=0.15, help="observed fraction of the 20 px ring around a patch needed to inpaint")
    ap.add_argument("--gap-px", type=float, default=2.0, help="gaps between splats closed by normalised convolution (sigma, px)")
    ap.add_argument("--smooth-m", type=float, default=0.06, help="blend colours kept elsewhere are averaged over this radius (0: off)")
    ap.add_argument("--debug-dir", default=None, help="save [blend | input+mask | inpainted] for the largest patches")
    ap.add_argument("--debug-n", type=int, default=12)
    ap.add_argument("--roundtrip-make", default=None, metavar="DIR")
    ap.add_argument("--roundtrip-check", default=None, metavar="SHOTS_DIR")
    ap.add_argument("--dataset", default=None, help="with --roundtrip-make: COLMAP text dataset of the camera")
    args = ap.parse_args()
    if args.roundtrip_check:
        return roundtrip_check(args.roundtrip_check, args.roundtrip_make)
    if args.roundtrip_make:
        return roundtrip_make(args.roundtrip_make, args.dataset)
    import cv2
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    t0 = time.time()

    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    base = read_vertices(args.splat_ply); full = read_vertices(args.fill_ply)
    nb = len(base); assert len(full) > nb or len(full) == nb, "fill ply must be the splat + appended discs"
    for k in ("x", "y", "z", "f_dc_0"):                                  # the prefix must be the base splat itself
        assert np.array_equal(full[k][:nb], base[k]), f"{args.fill_ply} does not start with {args.splat_ply}"
    out = full.copy(); fill = full[nb:]; m = len(fill)
    print(f"[inpaint] base {nb:,} gaussians, {m:,} fill discs", flush=True)
    stats = {"fill_discs": int(m), "method": args.method, "patches": []}
    if m:
        def metric(v):
            return s * (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T) + t
        F = metric(fill)
        qf = np.stack([fill["rot_1"], fill["rot_2"], fill["rot_3"], fill["rot_0"]], 1)
        Nf = Rotation.from_quat(qf).as_matrix()[:, :, 2] @ R.T               # disc normal = local z, metric
        Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
        spacing = float(np.median(cKDTree(F).query(F, k=2)[0][:, 1]))
        patches, _ = segment(F, Nf, spacing, args.normal_deg, args.min_pts)
        left = np.setdiff1d(np.arange(m), np.concatenate(patches) if patches else [])
        if len(left):                                                      # clutter: proximity only, any normals
            more, _ = segment(F[left], Nf[left], spacing, 180.0, args.min_pts)
            patches += [left[p] for p in more]
        n_small = int(m - sum(map(len, patches)))
        print(f"[inpaint] spacing {spacing:.3f} m; {len(patches)} patches >= {args.min_pts} discs "
              f"({sum(map(len, patches)):,} discs); {n_small} discs in smaller clusters keep the blend colour", flush=True)

        G = metric(base)
        op = 1 / (1 + np.exp(-np.asarray(base["opacity"], np.float64)))
        sel = op > args.min_opacity
        gi = np.nonzero(sel)[0]; G = G[gi]; opg = op[gi]
        rgbg = np.clip(0.5 + C0 * np.stack([base["f_dc_0"], base["f_dc_1"], base["f_dc_2"]], 1)[gi].astype(np.float64), 0, 1)
        sc = np.exp(np.stack([base["scale_0"], base["scale_1"], base["scale_2"]], 1)[gi].astype(np.float64)) * s
        qg = np.stack([base["rot_1"], base["rot_2"], base["rot_3"], base["rot_0"]], 1)[gi].astype(np.float64)
        qg /= np.maximum(np.linalg.norm(qg, axis=1, keepdims=True), 1e-12)
        Rg = Rotation.from_quat(qg).as_matrix()
        Ng = Rg[np.arange(len(gi)), :, np.argmin(sc, 1)] @ R.T               # gaussian normal (thinnest axis), metric
        sc_in = np.sqrt(np.sort(sc, 1)[:, 1] * np.sort(sc, 1)[:, 2])         # in-plane extent
        gtree = cKDTree(G)
        dcf = np.stack([fill["f_dc_0"], fill["f_dc_1"], fill["f_dc_2"]], 1).astype(np.float64)
        new_dc = dcf.copy()
        inp = Inpainter(args.method, args.lama_pt)
        if args.debug_dir:
            os.makedirs(args.debug_dir, exist_ok=True)
        def texture(c, n, e1, e2, lo, size, px, slab, npow):
            """Orthographic splat texture on the plane (c, n): colour, observed mask, #gaussians used."""
            W, H = int(np.ceil(size[0] / px)), int(np.ceil(size[1] / px))
            cen = c + e1 * (lo[0] + size[0] / 2) + e2 * (lo[1] + size[1] / 2)
            near = np.array(gtree.query_ball_point(cen, float(np.linalg.norm(size)) / 2 + 0.05), dtype=int)
            d = (G[near] - c) @ n; near, d = near[np.abs(d) < slab], d[np.abs(d) < slab]
            gu = ((G[near] - c) @ e1 - lo[0]) / px - 0.5; gv = ((G[near] - c) @ e2 - lo[1]) / px - 0.5
            w = opg[near] * (np.abs(Ng[near] @ n) ** npow + 0.05) * np.exp(-0.5 * (d / (slab / 2)) ** 2)
            ins = (gu > -6) & (gu < W + 6) & (gv > -6) & (gv < H + 6) & (w > 1e-3)
            near, gu, gv, w = near[ins], gu[ins], gv[ins], w[ins]
            img, wsum = splat_texture(gu, gv, rgbg[near], w, np.clip(sc_in[near] / px, 1.0, 3.0), H, W)
            # close small gaps between splats by normalised convolution (a few cm), not by inpainting
            acc = cv2.GaussianBlur(img * wsum[..., None], (0, 0), args.gap_px); wb = cv2.GaussianBlur(wsum, (0, 0), args.gap_px)
            gap = (wsum < args.known_w) & (wb > args.known_w * 0.5)
            img[gap] = (acc / np.maximum(wb, 1e-9)[..., None])[gap]
            known = ((wsum >= args.known_w) | gap).astype(np.uint8)
            known = cv2.morphologyEx(known, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop isolated speckle
            return img, known.astype(bool), int(len(near))

        for pi, idx in enumerate(patches):
            X = F[idx]; c = X.mean(0)
            ev, evec = np.linalg.eigh(np.cov((X - c).T) if len(idx) > 2 else np.eye(3))
            n = evec[:, 0]; e1 = evec[:, 2]; e2 = np.cross(n, e1)
            dfill = (X - c) @ n
            uv = np.stack([(X - c) @ e1, (X - c) @ e2], 1)
            ext = uv.max(0) - uv.min(0)
            info = {"patch": pi, "discs": int(len(idx)), "centre_metric": c.round(2).tolist(), "size_m": [round(float(x), 2) for x in ext],
                    "normal_metric": n.round(2).tolist(), "planar_rms_m": round(float(dfill.std()), 3)}
            # observed surroundings: same-plane first; widen slab/context and drop the normal test if too little
            for lvl, (sm, cm, npow) in enumerate(((1, 1, 4), (2.5, 1.5, 1), (5, 2, 0))):
                ctx = max(args.context_m * cm, 0.5 * cm * float(ext.max()))
                lo = uv.min(0) - ctx; size = ext + 2 * ctx
                px = max(args.px, float(size.max()) / args.max_side)
                img, known, ng = texture(c, n, e1, e2, lo, size, px, float(np.abs(dfill).max()) + sm * args.slab_m, npow)
                H, W = known.shape
                fu = (uv[:, 0] - lo[0]) / px - 0.5; fv = (uv[:, 1] - lo[1]) / px - 0.5
                fp = np.zeros((H, W), np.uint8)
                fp[np.clip(fv.round().astype(int), 0, H - 1), np.clip(fu.round().astype(int), 0, W - 1)] = 1
                fp = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(spacing / px)) + 3,) * 2)).astype(bool)
                hole = fp | ~known
                hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # half-covered rims
                ring = cv2.dilate(fp.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~fp
                kf = float((~hole & ring).sum() / max(ring.sum(), 1))           # observed fraction around the patch
                if kf >= args.min_known:
                    break
            info.update({"tex": [W, H], "px_m": round(px, 4), "level": lvl, "gaussians": ng, "known_ring": round(kf, 3)})
            if kf < args.min_known:
                info["kept_blend"] = True; stats["patches"].append(info); continue
            res = inp(img, hole)
            new_dc[idx] = (np.clip(bilinear(res, fu, fv), 0, 1) - 0.5) / C0
            stats["patches"].append(info)
            if args.debug_dir and pi < args.debug_n:
                blend, bw = splat_texture(fu, fv, np.clip(0.5 + C0 * dcf[idx], 0, 1), np.ones(len(idx)),
                                          np.full(len(idx), 0.6 * spacing / px + 0.3), H, W)
                a = np.where(fp[..., None], blend, img * ~hole[..., None])
                b = img.copy(); b[hole] = [1, 0, 1]
                cimg = res.copy(); edge = fp & ~cv2.erode(fp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                cimg[edge] = [0, 1, 1]
                tile = np.hstack([a, b, res, cimg])
                cv2.imwrite(os.path.join(args.debug_dir, f"patch{pi:03d}_{len(idx)}.jpg"),
                            (np.clip(tile, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            if pi < 20 or pi % 50 == 0:
                print(f"[inpaint] patch {pi}: {json.dumps(info)}", flush=True)
        # discs without an inpainted texture (small clusters, too little context): de-speckle their blend colour
        # by averaging it over neighbouring same-orientation fill discs (--smooth-m)
        keep = np.abs(new_dc - dcf).max(1) <= 1e-6
        if keep.any() and args.smooth_m > 0:
            ftree = cKDTree(F); ki = np.nonzero(keep)[0]
            dd, ii = ftree.query(F[ki], k=16, distance_upper_bound=args.smooth_m)
            ok = np.isfinite(dd); ii = np.where(ok, ii, 0)
            wk = ok * np.abs(np.einsum("nkd,nd->nk", Nf[ii], Nf[ki])) ** 4
            new_dc[ki] = (wk[..., None] * dcf[ii]).sum(1) / np.maximum(wk.sum(1, keepdims=True), 1e-9)
            stats["blend_smoothed"] = int(len(ki))
        for i, k in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            out[k][nb:] = new_dc[:, i].astype(np.float32)
        changed = np.abs(new_dc - dcf).max(1) > 1e-6
        stats.update({"recoloured": int(changed.sum()), "patches_n": len(patches), "blend_kept_small": n_small,
                      "mean_abs_rgb_change": round(float((C0 * np.abs(new_dc - dcf)).mean()), 4)})
    assert all(np.array_equal(out[k][:nb], base[k]) for k in out.dtype.names), "base splat must stay untouched"
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(args.out_ply)
    stats["out"] = args.out_ply; stats["seconds"] = round(time.time() - t0, 1)
    if args.debug_dir:
        json.dump(stats, open(os.path.join(args.debug_dir, "patches.json"), "w"), indent=1)
    stats.pop("patches")
    print("FILL_INPAINT", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
