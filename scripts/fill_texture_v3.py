"""Fill v3: texture the unseen-surface fill discs from the REAL training images (GPU optional; venv-inpaint).

fill_inpaint.py inpaints each fill patch from a top-down render of the splat's SH-DC colours, which is soft (the
splat's own blur + gaussian splatting), and LaMa then continues that softness. Here the context comes from the
camera frames themselves; the discs keep their geometry (fill_unseen_surfaces.py) and only their colour changes:

  PATCHES  as fill_inpaint.py (same segmentation, same patch ids), each with its PCA plane; texel grid on the plane
           over the patch + --context-m of surroundings. Texel 3D points follow the actual surface: height above
           the plane from the fill discs and the splat gaussians in a slab (normalised convolution), so the
           projection is exact for non-planar patches too (a homography when the patch is flat).
  RECTIFY  every training frame that sees the area: texels projected, bilinear-sampled from the image. A sample
           counts where the splat's rendered depth (eval_views --save-depth-dir) agrees with the texel's ray
           distance (not occluded, opaque), or where the LiDAR depth target (<image>_depth.npy, nothing closer in
           a 5x5 window) does and the splat is not in front; texels inside the fill footprint (glimpses: the
           LiDAR says the camera really saw the patch) need the LiDAR test and >= --min-glimpse views. Depth edges
           and grazing / coarse views (cos < --min-cos, footprint > --max-foot-m) are dropped. Each frame gets a
           per-patch RGB gain vs the splat texture (exposure), the --top-k best views per texel (resolution x
           obliqueness) are median-composited -> a sharp texture of the observed surround.
  COLOUR   per-patch 3x4 affine colour transform camera -> splat (robust least squares on the observed ring,
           ridge to identity), so the fill matches what the viewer shows next to it; SuperSplat shows
           rgb = 0.5 + C0 dc (linear, fill_inpaint.py --roundtrip-*).
  FILL     what no camera saw: (1) straight coloured features (tape lines: long Hough segments, chromatic vs the
           local background, in a parallel family) continued through the hole where they re-emerge on the far side
           or the gap is < 0.3 m; (2) periodic texture (tiles): masked normalised autocorrelation of the camera
           texture, a prominent peak (--lattice-min-corr, >= 6 observed periods) -> each hole texel copies the
           nearest observed background lattice translate; (3) the rest by LaMa from the sharp context (splat
           texture only where no camera saw the context).
  FALLBACK a patch whose camera-observed ring is < --min-known (or tiny clusters) keeps its colours from
           --prev-ply (the fill_inpaint.py result), so v3 is never worse than v2 for want of data.
Only fill discs' f_dc change; the base splat is copied untouched (asserted), so real views are unchanged.

    python scripts/fill_texture_v3.py --splat-ply X.ply --fill-ply X_complete.ply --prev-ply X_inpainted.ply \
        --sim3-json sim3.json --dataset <training frames> --splat-depth-dir <dir> --out-ply X_fillv3.ply \
        [--lama-pt big-lama.pt --debug-dir dbg/ --crops-dataset crops/]
--crops-dataset writes a COLMAP text model of one oblique camera per large patch (pipeline frame) for
viewer_shots.py, to compare the fill per patch.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model
from fill_inpaint import C0, Inpainter, bilinear, read_vertices, segment, splat_texture


class Frames:
    """Training frames (COLMAP text) + per-frame splat depth/opacity and LiDAR depth, loaded lazily and cached."""
    def __init__(self, dataset, depth_dir, s, R, t):
        cams, imgs = read_text_model(os.path.join(dataset, "sparse", "0"))
        self.dataset, self.depth_dir, self.cache, self.f = dataset, depth_dir, {}, []
        for name, cid, q, tc in sorted(imgs, key=lambda i: i[0]):
            W, H, K = cams[cid]; Rcw = quat_to_R(q); Cd = -Rcw.T @ tc
            self.f.append(dict(name=name, W=W, H=H, K=K, Rcw=Rcw, t=np.asarray(tc), Cm=s * (R @ Cd) + t,
                               fwd=R @ Rcw[2]))                              # optical axis, metric
        self.Cm = np.array([f["Cm"] for f in self.f])

    def get(self, i):
        if i in self.cache:
            return self.cache[i]
        import cv2
        f = self.f[i]; stem = os.path.splitext(f["name"])[0]
        img = cv2.imread(os.path.join(self.dataset, "images", f["name"]), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255
        def ld(p):
            return np.load(p).astype(np.float32) if os.path.exists(p) else None
        D = ld(os.path.join(self.depth_dir, stem + "_splatdepth.npy")) if self.depth_dir else None
        A = ld(os.path.join(self.depth_dir, stem + "_splatopacity.npy")) if self.depth_dir else None
        L = ld(os.path.join(self.dataset, "images", stem + "_depth.npy"))
        if L is None:
            L = ld(os.path.join(self.dataset, "depth", stem + "_depth.npy"))
        H, W = img.shape[:2]
        if A is None and D is not None:
            A = np.ones((H, W), np.float32)
        Lmin = None
        if L is not None:                                                    # nearest LiDAR surface in a 5x5 window
            Lmin = cv2.erode(np.where(L > 0, L, 1e6).astype(np.float32), np.ones((5, 5), np.uint8))
        edge = np.zeros((H, W), bool)
        if D is not None:                                                    # depth discontinuities: colours bleed there
            Dv = np.where(D > 0, D, 0).astype(np.float32)
            dmax = cv2.dilate(Dv, np.ones((5, 5), np.uint8)); dmin = cv2.erode(np.where(D > 0, D, 1e6).astype(np.float32), np.ones((5, 5), np.uint8))
            edge = (dmax - dmin) > 0.08 * np.maximum(Dv, 1e-3)
        self.cache[i] = (img, D, A, L, Lmin, edge)
        return self.cache[i]


def robust_affine(X, Y, ridge=0.05, iters=3):
    """Y ~ [X 1] M (3x4, ridge toward identity), IRLS dropping > 2.5 MAD residuals -> M (4x3), rms."""
    A = np.hstack([X, np.ones((len(X), 1))]); M0 = np.vstack([np.eye(3), np.zeros((1, 3))])
    w = np.ones(len(X))
    for _ in range(iters):
        Aw = A * w[:, None]; reg = ridge * len(X) * np.diag([1, 1, 1, 0.1])
        M = np.linalg.solve(Aw.T @ A + reg, Aw.T @ Y + reg @ M0)
        r = np.linalg.norm(A @ M - Y, axis=1); mad = np.median(r) + 1e-6
        w = (r < 2.5 * 1.4826 * mad).astype(float)
    return M, float(np.sqrt(np.mean(r[w > 0] ** 2))) if w.sum() else float("nan")


def extend_lines(img, known, hole, px, min_len_m, dev_thr, chroma_min=0.06, max_lines=12, max_gap_m=0.3):
    """Straight, colour-distinct features (tape lines) of the observed texture continued through the hole.
    -> painted image, painted mask, list of lines."""
    import cv2
    H, W = known.shape
    k = int(2 * round(0.15 / px) + 1) | 1
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    bg = np.stack([cv2.medianBlur(u8[..., c], min(k, 255)) for c in range(3)], -1) / 255.0
    dev = np.abs(img - bg).sum(-1)
    feat = (dev > dev_thr) & known & ~cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    feat = cv2.morphologyEx(feat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    segs = cv2.HoughLinesP(feat * 255, 1, np.pi / 360, threshold=max(20, int(0.2 / px)),
                           minLineLength=max(10, int(min_len_m / px)), maxLineGap=max(2, int(0.04 / px)))
    out, painted, lines = img.copy(), np.zeros((H, W), bool), []
    if segs is None:
        return out, painted, lines
    segs = segs[:, 0].astype(float)
    segs = segs[np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))]
    used = []
    for x0, y0, x1, y1 in segs:
        d = np.array([x1 - x0, y1 - y0]); L = np.linalg.norm(d); d /= L; nrm = np.array([-d[1], d[0]])
        rho = nrm @ [x0, y0]
        if any(abs(abs(d @ du) - 1) < 1e-3 and abs(abs(nrm @ [x0, y0]) - abs(r)) < 4 for du, r in used):
            continue                                                         # duplicate of a stronger segment
        used.append((d, rho))
        # width + colour from the perpendicular profile of the feature mask along the observed segment
        ts = np.linspace(0, L, max(int(L), 2)); off = np.arange(-15, 16)
        P = np.array([x0, y0]) + ts[:, None, None] * d + off[None, :, None] * nrm   # (T, O, 2)
        xi, yi = np.clip(P[..., 0].round().astype(int), 0, W - 1), np.clip(P[..., 1].round().astype(int), 0, H - 1)
        prof = feat[yi, xi].mean(0)
        width = max(2, int(np.sum(prof > 0.5)))
        on = feat[yi[:, 15], xi[:, 15]].astype(bool)
        if on.mean() < 0.6 or width > int(0.12 / px) + 2:
            continue
        col = np.median(img[yi[:, 15][on], xi[:, 15][on]], 0)
        bgc = np.median(bg[yi[:, 15][on], xi[:, 15][on]], 0)
        if (col.max() - col.min()) - (bgc.max() - bgc.min()) < chroma_min:
            continue                                                         # tape is coloured; shading edges are not
        fam = sum(abs(abs(d @ du) - 1) < 1.5e-3 and abs(rho - r) > 4 for du, r in used[:-1]) + \
            sum(abs(abs(d @ (np.array([x3 - x2, y3 - y2]) / max(np.hypot(x3 - x2, y3 - y2), 1e-6))) - 1) < 1.5e-3
                and abs(rho - nrm @ [x2, y2]) > 4 for x2, y2, x3, y3 in segs)
        if fam < 1:
            continue                                                         # tape lines come in parallel families
        # walk outward from both ends through contiguous hole pixels; paint where the walk crossed a hole
        n_paint = 0
        for sgn, start in ((-1, np.array([x0, y0])), (1, np.array([x1, y1]))):
            p, entered, seg_pts = start.copy(), False, []
            for _ in range(int(max(H, W) * 1.5)):
                p = p + sgn * d; xi_, yi_ = int(round(p[0])), int(round(p[1]))
                if not (0 <= xi_ < W and 0 <= yi_ < H):
                    break
                if hole[yi_, xi_]:
                    entered = True; seg_pts.append(p.copy())
                elif entered:
                    break
                elif len(seg_pts) == 0 and np.linalg.norm(p - start) > 0.1 / px:
                    break                                                    # the segment does not reach a hole
            reemerge = False
            if entered and 0 <= xi_ < W and 0 <= yi_ < H:                    # the walk left the hole: line continues there?
                tt = np.arange(1, int(0.1 / px) + 2)
                q_ = p[None] + sgn * tt[:, None] * d[None]
                qx, qy = np.clip(q_[:, 0].round().astype(int), 0, W - 1), np.clip(q_[:, 1].round().astype(int), 0, H - 1)
                reemerge = feat[qy, qx].mean() > 0.5
            if len(seg_pts) > 1 and (reemerge or len(seg_pts) * px < max_gap_m):
                a, b = seg_pts[0], seg_pts[-1]
                m = np.zeros((H, W), np.uint8)
                cv2.line(m, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), 1, width)
                m = m.astype(bool) & hole
                out[m] = col; painted |= m; n_paint += int(m.sum())
        if n_paint:
            lines.append({"p0": [round(x0, 1), round(y0, 1)], "p1": [round(x1, 1), round(y1, 1)], "width_px": width,
                          "rgb": np.round(col, 3).tolist(), "painted_px": n_paint})
        if len(lines) >= max_lines:
            break
    return out, painted, lines


def lattice_fill(img, known, hole, px, min_corr, min_shift_m=0.15, max_shift_m=1.2, min_prom=0.1):
    """Periodic texture (floor tiles): the two strongest translation peaks T1, T2 of the masked normalised
    autocorrelation of the observed high-pass texture; each hole pixel copies the nearest lattice translate
    x + k T1 + l T2 that is observed and background (not an object or tape: those are not tiled).
    -> image, copied mask, info."""
    import cv2
    H, W = known.shape
    med = np.median(img[known], 0) if known.sum() > 100 else None
    if med is None:
        return img, np.zeros((H, W), bool), {}
    bgok = known & (np.abs(img - med).sum(-1) < 0.15)                      # plain background texels
    grey = img.mean(-1); hp = grey - cv2.GaussianBlur(grey, (0, 0), max(1.5, 0.03 / px))   # seams, not shading
    w = bgok.astype(np.float64); g = np.where(bgok, hp, 0.0)
    Hp, Wp = cv2.getOptimalDFTSize(2 * H), cv2.getOptimalDFTSize(2 * W)
    Fw = np.fft.rfft2(w, (Hp, Wp)); Fg = np.fft.rfft2(g, (Hp, Wp)); Fg2 = np.fft.rfft2(g * g, (Hp, Wp))
    A = np.fft.irfft2(Fg * np.conj(Fg), (Hp, Wp)); B = np.fft.irfft2(Fg2 * np.conj(Fw), (Hp, Wp))
    C = np.fft.irfft2(Fw * np.conj(Fg2), (Hp, Wp)); N = np.fft.irfft2(Fw * np.conj(Fw), (Hp, Wp))
    corr = A / np.sqrt(np.maximum(B * C, 1e-12))
    dy = np.fft.fftfreq(Hp, 1 / Hp).round().astype(int)[:, None]; dx = np.fft.fftfreq(Wp, 1 / Wp).round().astype(int)[None, :]
    rr = np.hypot(dx, dy) * px
    ok = (rr >= min_shift_m) & (rr <= max_shift_m) & (N > max(0.15 * bgok.sum(), 400))
    cm = np.where(ok, corr, -1.0)
    loc = (cm >= cv2.dilate(cm.astype(np.float32), np.ones((9, 9), np.uint8)) - 1e-9) & ok & (rr >= min_shift_m + 3 * px)
    # a lattice peak stands out of the correlation at the same radius (smooth texture decays radially)
    rb = np.round(rr / px).astype(int); prof = np.bincount(rb[ok], cm[ok], rb.max() + 1) / np.maximum(np.bincount(rb[ok], None, rb.max() + 1), 1)
    loc &= (cm - prof[rb]) > min_prom
    pk = np.argwhere(loc); vals = cm[loc]
    if not len(vals) or vals.max() < min_corr:
        return img, np.zeros((H, W), bool), {"corr": round(float(vals.max()) if len(vals) else -1, 3)}
    order = np.argsort(-vals); pk, vals = pk[order], vals[order]
    T1 = np.array([dx[0, pk[0][1]], dy[pk[0][0], 0]], float); c1 = float(vals[0]); T2, c2 = None, 0.0
    if bgok.sum() < 6 * (T1 @ T1):                                         # too few observed periods to trust it
        return img, np.zeros((H, W), bool), {"corr": round(c1, 3), "T1_m": (T1 * px).round(3).tolist(), "rejected": "small source"}
    for (i, j), v_ in zip(pk[1:], vals[1:]):
        T = np.array([dx[0, j], dy[i, 0]], float)
        if v_ < min_corr:
            break
        if abs(T @ T1) / (np.linalg.norm(T) * np.linalg.norm(T1)) < 0.85:   # not collinear: second lattice axis
            T2, c2 = T, float(v_); break
    K = int(np.ceil(max(H, W) / np.linalg.norm(T1))) + 1
    cands = [(k, 0) for k in range(-K, K + 1)] if T2 is None else \
        [(k, l) for k in range(-K, K + 1) for l in range(-K, K + 1)]
    shifts = sorted(((k * T1 + (l * T2 if T2 is not None else 0)) for k, l in cands if (k, l) != (0, 0)), key=np.linalg.norm)
    out = img.copy(); todo = hole.copy(); copied = np.zeros((H, W), bool)
    ys, xs = np.nonzero(todo)
    for S_ in shifts:
        if not len(ys):
            break
        sx, sy = int(round(xs[0] * 0 + S_[0])), int(round(S_[1]))
        x2, y2 = xs + sx, ys + sy
        ins = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
        hit = np.zeros(len(xs), bool); hit[ins] = bgok[y2[ins], x2[ins]]
        out[ys[hit], xs[hit]] = img[y2[hit], x2[hit]]; copied[ys[hit], xs[hit]] = True
        ys, xs = ys[~hit], xs[~hit]
    return out, copied, {"T1_m": (T1 * px).round(3).tolist(), "corr1": round(c1, 3),
                         "T2_m": None if T2 is None else (T2 * px).round(3).tolist(), "corr2": round(c2, 3)}


def look_at_quat(C, target, up):
    """OpenCV camera at C looking at target (y down ~ -up) -> COLMAP (qw qx qy qz), t."""
    from scipy.spatial.transform import Rotation
    z = target - C; z /= np.linalg.norm(z)
    y = -up - (-up @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, [1.0, 0, 0])
    y /= np.linalg.norm(y); x = np.cross(y, z)
    Rcw = np.stack([x, y, z]); q = Rotation.from_matrix(Rcw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]]), -Rcw @ C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat-ply", required=True, help="the splat before the unseen fill")
    ap.add_argument("--fill-ply", required=True, help="fill_unseen_surfaces.py output (splat + appended discs)")
    ap.add_argument("--prev-ply", default=None, help="fill_inpaint.py output: per-patch fallback colours")
    ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--dataset", required=True, help="training frames, COLMAP text, images/<name> (+ <stem>_depth.npy LiDAR)")
    ap.add_argument("--splat-depth-dir", default=None, help="<stem>_splatdepth.npy / _splatopacity.npy per training frame")
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--lama-pt", default=None)
    ap.add_argument("--method", default="lama", choices=["lama", "shiftmap", "telea"])
    ap.add_argument("--px", type=float, default=0.008, help="texture resolution (m/texel)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--context-m", type=float, default=0.6)
    ap.add_argument("--slab-m", type=float, default=0.06)
    ap.add_argument("--normal-deg", type=float, default=30.0)
    ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--known-w", type=float, default=0.5)
    ap.add_argument("--min-known", type=float, default=0.15, help="camera-observed fraction of the ring around a patch needed")
    ap.add_argument("--max-dist-m", type=float, default=6.0, help="frames farther than this from the patch are skipped")
    ap.add_argument("--min-cos", type=float, default=0.12, help="grazing views below this cos(view, normal) dropped")
    ap.add_argument("--max-foot-m", type=float, default=0.05, help="image pixel footprint on the surface (m), max")
    ap.add_argument("--tol-rel", type=float, default=0.03)
    ap.add_argument("--tol-m", type=float, default=0.04, help="depth agreement tolerance along the normal (m)")
    ap.add_argument("--top-k", type=int, default=7)
    ap.add_argument("--min-glimpse", type=int, default=2, help="views needed for a texel inside the fill footprint")
    ap.add_argument("--height-sigma-m", type=float, default=0.1, help="surface height smoothing (m)")
    ap.add_argument("--lattice", type=int, default=1, help="copy the hole from lattice translates of the observed texture")
    ap.add_argument("--lattice-min-corr", type=float, default=0.25)
    ap.add_argument("--lines", type=int, default=1, help="continue straight colour features (tape) through holes")
    ap.add_argument("--line-min-m", type=float, default=0.3)
    ap.add_argument("--line-dev", type=float, default=0.12, help="colour deviation from the local background (sum RGB)")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--debug-n", type=int, default=14)
    ap.add_argument("--crops-dataset", default=None, help="write oblique cameras on the largest patches here")
    ap.add_argument("--crops-n", type=int, default=12)
    ap.add_argument("--max-patches", type=int, default=0, help="smoke: only the first N patches (0 = all)")
    args = ap.parse_args()
    import cv2
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    t0 = time.time()

    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    base = read_vertices(args.splat_ply); full = read_vertices(args.fill_ply); nb = len(base)
    for k in ("x", "y", "z", "f_dc_0"):
        assert np.array_equal(full[k][:nb], base[k]), f"{args.fill_ply} does not start with {args.splat_ply}"
    out = full.copy(); fill = full[nb:]; m = len(fill)
    dcf = np.stack([fill["f_dc_0"], fill["f_dc_1"], fill["f_dc_2"]], 1).astype(np.float64)
    prev_dc = dcf.copy()
    if args.prev_ply:
        prev = read_vertices(args.prev_ply)
        assert len(prev) == len(full) and np.array_equal(prev["x"], full["x"]), "--prev-ply must be the same discs"
        prev_dc = np.stack([prev["f_dc_0"], prev["f_dc_1"], prev["f_dc_2"]], 1)[nb:].astype(np.float64)
    new_dc = prev_dc.copy(); src = np.zeros(m, np.int8)                     # 0 prev, 1 camera, 2 line, 3 lama
    print(f"[v3] base {nb:,} gaussians, {m:,} fill discs", flush=True)
    stats = {"fill_discs": int(m), "patches": []}

    def metric(v):
        return s * (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T) + t
    F = metric(fill)
    qf = np.stack([fill["rot_1"], fill["rot_2"], fill["rot_3"], fill["rot_0"]], 1)
    Nf = Rotation.from_quat(qf).as_matrix()[:, :, 2] @ R.T; Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
    spacing = float(np.median(cKDTree(F).query(F, k=2)[0][:, 1]))
    patches, _ = segment(F, Nf, spacing, args.normal_deg, args.min_pts)    # identical to fill_inpaint.py
    left = np.setdiff1d(np.arange(m), np.concatenate(patches) if patches else [])
    if len(left):
        more, _ = segment(F[left], Nf[left], spacing, 180.0, args.min_pts)
        patches += [left[p] for p in more]
    print(f"[v3] spacing {spacing:.3f} m; {len(patches)} patches", flush=True)

    G = metric(base); op = 1 / (1 + np.exp(-np.asarray(base["opacity"], np.float64)))
    gi = np.nonzero(op > args.min_opacity)[0]; G = G[gi]; opg = op[gi]
    rgbg = np.clip(0.5 + C0 * np.stack([base["f_dc_0"], base["f_dc_1"], base["f_dc_2"]], 1)[gi].astype(np.float64), 0, 1)
    sc = np.exp(np.stack([base["scale_0"], base["scale_1"], base["scale_2"]], 1)[gi].astype(np.float64)) * s
    qg = np.stack([base["rot_1"], base["rot_2"], base["rot_3"], base["rot_0"]], 1)[gi].astype(np.float64)
    qg /= np.maximum(np.linalg.norm(qg, axis=1, keepdims=True), 1e-12)
    Ng = Rotation.from_quat(qg).as_matrix()[np.arange(len(gi)), :, np.argmin(sc, 1)] @ R.T
    sc_in = np.sqrt(np.sort(sc, 1)[:, 1] * np.sort(sc, 1)[:, 2]); gtree = cKDTree(G)
    fr = Frames(args.dataset, args.splat_depth_dir, s, R, t)
    print(f"[v3] {len(fr.f)} training frames; splat depth {'yes' if args.splat_depth_dir else 'NO'}", flush=True)
    inp = Inpainter(args.method, args.lama_pt)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    def splat_tex(c, n, e1, e2, lo, W, H, px, slab, npow=4):
        """Orthographic splat SH-DC texture on the plane + height above it (as fill_inpaint.py's VIEW)."""
        size = np.array([W, H]) * px; cen = c + e1 * (lo[0] + size[0] / 2) + e2 * (lo[1] + size[1] / 2)
        near = np.array(gtree.query_ball_point(cen, float(np.linalg.norm(size)) / 2 + 0.05), dtype=int)
        d = (G[near] - c) @ n; near, d = near[np.abs(d) < slab], d[np.abs(d) < slab]
        gu = ((G[near] - c) @ e1 - lo[0]) / px - 0.5; gv = ((G[near] - c) @ e2 - lo[1]) / px - 0.5
        w = opg[near] * (np.abs(Ng[near] @ n) ** npow + 0.05) * np.exp(-0.5 * (d / (slab / 2)) ** 2)
        ins = (gu > -6) & (gu < W + 6) & (gv > -6) & (gv < H + 6) & (w > 1e-3)
        near, gu, gv, w, d = near[ins], gu[ins], gv[ins], w[ins], d[ins]
        sig = np.clip(sc_in[near] / px, 1.0, 3.0)
        img, wsum = splat_texture(gu, gv, rgbg[near], w, sig, H, W)
        hgt, _ = splat_texture(gu, gv, np.stack([d, d, d], 1), w, sig, H, W)
        acc = cv2.GaussianBlur(img * wsum[..., None], (0, 0), 2.0); wb = cv2.GaussianBlur(wsum, (0, 0), 2.0)
        gap = (wsum < args.known_w) & (wb > args.known_w * 0.5)
        img[gap] = (acc / np.maximum(wb, 1e-9)[..., None])[gap]
        known = cv2.morphologyEx(((wsum >= args.known_w) | gap).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return img, known.astype(bool), hgt[..., 0], wsum

    def camera_tex(X, nrm, inside, Ts, Ms, Cpatch, radius):
        """Rectify + median-composite the training frames onto texel points X (T,3 metric)."""
        T = len(X); K = args.top_k
        Q = np.zeros((T, K), np.float32); Cs = np.zeros((T, K, 3), np.float32); Fi = np.full((T, K), -1, np.int32)
        Xd = ((X - t) @ R) / s; nd = R.T @ nrm
        dist = np.linalg.norm(fr.Cm - Cpatch, axis=1)
        cand = np.nonzero(dist < args.max_dist_m + radius)[0]
        nview = 0; glimpse_hits = np.zeros(T, np.int32)
        ts_flat = Ts.reshape(-1, 3); ms_flat = Ms.reshape(-1)
        gains = {}
        for fi in cand:
            f = fr.f[fi]; fx, fy, cx, cy = f["K"]; W, H = f["W"], f["H"]
            # cheap frustum cull on the patch's bounding sphere
            pc = f["Rcw"] @ (((Cpatch - t) @ R) / s) + f["t"]
            if pc[2] < -radius / s:
                continue
            Pc = Xd @ f["Rcw"].T + f["t"]; z = Pc[:, 2]; ok = z > 0.05
            u = np.where(ok, fx * Pc[:, 0] / np.maximum(z, 1e-6) + cx, -1); v = np.where(ok, fy * Pc[:, 1] / np.maximum(z, 1e-6) + cy, -1)
            ok &= (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
            if ok.sum() < 20:
                continue
            ii = np.nonzero(ok)[0]; Pc, u, v = Pc[ii], u[ii], v[ii]
            r = np.linalg.norm(Pc, axis=1); cosv = np.abs((Pc / r[:, None]) @ (f["Rcw"] @ nd))
            foot = r * s / (fx * np.maximum(cosv, 1e-3))
            good = (cosv > args.min_cos) & (foot < args.max_foot_m)
            if good.sum() < 20:
                continue
            ii, u, v, r, cosv, foot = ii[good], u[good], v[good], r[good], cosv[good], foot[good]
            img, D, A, L, Lmin, edge = fr.get(fi)
            ui, vi = u.astype(int), v.astype(int)
            tol = args.tol_rel * r + args.tol_m / (s * np.maximum(cosv, 0.25))
            if D is not None:
                d, a = D[vi, ui], A[vi, ui]
                vis_s = (d > 0) & (a > 0.5) & (np.abs(r - d) < tol)
                not_front = (d <= 0) | (a < 0.5) | (d > r - tol)
            else:
                vis_s = np.zeros(len(r), bool); not_front = np.ones(len(r), bool)
            if L is not None:
                l, lm = L[vi, ui], Lmin[vi, ui]
                vis_l = (l > 0) & (np.abs(r - l) < tol) & (lm > r - tol)
            else:
                vis_l = np.zeros(len(r), bool)
            ins = inside[ii]
            # LiDAR agreement wins over the splat: a floater in front of the floor was fit to the floor's colour
            use = np.where(ins, vis_l, vis_s | vis_l) & ~edge[vi, ui]
            if use.sum() < 20:
                continue
            ii, u, v, foot, cosv, ins = ii[use], u[use], v[use], foot[use], cosv[use], ins[use]
            col = bilinear(img, u - 0.5, v - 0.5).astype(np.float32)      # pixel centres at +0.5 (COLMAP)
            # exposure: per-frame RGB gain vs the splat texture on observed context texels
            ref = ~ins & ms_flat[ii]
            g = np.ones(3, np.float32)
            if ref.sum() >= 50:
                g = np.clip(np.median(ts_flat[ii[ref]], 0) / np.maximum(np.median(col[ref], 0), 1e-3), 0.75, 1.33)
            gains[int(fi)] = g.round(3).tolist(); col = col * g
            q = (cosv ** 0.5 / foot).astype(np.float32)                       # resolution x obliqueness
            am = Q[ii].argmin(1); better = q > Q[ii, am]
            Q[ii[better], am[better]] = q[better]; Cs[ii[better], am[better]] = col[better]; Fi[ii[better], am[better]] = fi
            glimpse_hits[ii[ins]] += 1; nview += 1
        good = (Q > 0) & (Q >= 0.4 * Q.max(1, keepdims=True))
        nsamp = good.sum(1)
        Cn = np.where(good[..., None], Cs, np.nan)
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore"); tex = np.nanmedian(Cn, 1)
        valid = (nsamp >= 1) & ~(inside & (nsamp < args.min_glimpse))
        tex[~valid] = 0
        return tex, valid, nsamp, nview, gains

    crops = []
    todo = patches if not args.max_patches else patches[:args.max_patches]
    for pi, idx in enumerate(todo):
        X = F[idx]; c = X.mean(0)
        _, evec = np.linalg.eigh(np.cov((X - c).T) if len(idx) > 2 else np.eye(3))
        n = evec[:, 0]; e1 = evec[:, 2]; e2 = np.cross(n, e1)
        dfill = (X - c) @ n; uv = np.stack([(X - c) @ e1, (X - c) @ e2], 1); ext = uv.max(0) - uv.min(0)
        info = {"patch": pi, "discs": int(len(idx)), "centre_metric": c.round(2).tolist(),
                "size_m": [round(float(x), 2) for x in ext], "normal_metric": n.round(2).tolist()}
        res = None
        for lvl, cm in enumerate((1.0, 1.6)):
            ctx = max(args.context_m * cm, 0.5 * cm * float(ext.max()))
            lo = uv.min(0) - ctx; size = ext + 2 * ctx
            px = max(args.px, float(size.max()) / args.max_side)
            W, H = int(np.ceil(size[0] / px)), int(np.ceil(size[1] / px))
            slab = float(np.abs(dfill).max()) + args.slab_m
            Ts, Ms, hgt, wsum = splat_tex(c, n, e1, e2, lo, W, H, px, slab)
            fu = (uv[:, 0] - lo[0]) / px - 0.5; fv = (uv[:, 1] - lo[1]) / px - 0.5
            fp = np.zeros((H, W), np.uint8)
            fp[np.clip(fv.round().astype(int), 0, H - 1), np.clip(fu.round().astype(int), 0, W - 1)] = 1
            fp = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(spacing / px)) + 3,) * 2)).astype(bool)
            # surface height: fill discs inside the footprint, splat gaussians around; normalised convolution
            hacc = np.zeros((H, W)); hw = np.zeros((H, W))
            np.add.at(hacc, (np.clip(fv.round().astype(int), 0, H - 1), np.clip(fu.round().astype(int), 0, W - 1)), dfill * 5)
            np.add.at(hw, (np.clip(fv.round().astype(int), 0, H - 1), np.clip(fu.round().astype(int), 0, W - 1)), 5.0)
            hacc += hgt * wsum * ~fp; hw += wsum * ~fp
            sg = max(1.5, args.height_sigma_m / px)                  # smooth: height noise shifts grazing views
            hb = cv2.GaussianBlur(hacc, (0, 0), sg) / np.maximum(cv2.GaussianBlur(hw, (0, 0), sg), 1e-9)
            hb2 = cv2.GaussianBlur(hacc, (0, 0), 8 * sg) / np.maximum(cv2.GaussianBlur(hw, (0, 0), 8 * sg), 1e-9)
            hfin = np.where(cv2.GaussianBlur(hw, (0, 0), sg) > 1e-3, hb, hb2)
            gu, gv = np.meshgrid(lo[0] + (np.arange(W) + 0.5) * px, lo[1] + (np.arange(H) + 0.5) * px)
            Xt = c + gu.reshape(-1, 1) * e1 + gv.reshape(-1, 1) * e2 + hfin.reshape(-1, 1) * n
            radius = float(np.linalg.norm(size)) / 2
            tex, valid, nsamp, nview, gains = camera_tex(Xt, n, fp.reshape(-1), Ts, Ms, c, radius)
            Tc = tex.reshape(H, W, 3); Mc = valid.reshape(H, W)
            ring = cv2.dilate(fp.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~fp
            kf = float((Mc & ring).sum() / max(ring.sum(), 1))
            kfs = float((Ms & ring).sum() / max(ring.sum(), 1))
            if kf >= args.min_known:
                break
        info.update({"tex": [W, H], "px_m": round(px, 4), "level": lvl, "views": nview, "cam_ring": round(kf, 3),
                     "splat_ring": round(kfs, 3), "glimpse_frac": round(float((Mc & fp).sum() / max(fp.sum(), 1)), 3)})
        if kf < args.min_known:
            info["fallback"] = "prev"; stats["patches"].append(info)
            if pi < 25 or pi % 50 == 0:
                print(f"[v3] patch {pi}: {json.dumps(info)}", flush=True)
            continue
        # colour: camera -> splat affine on texels both observed (outside the footprint)
        both = Mc & Ms & ~fp
        if both.sum() >= 200:
            Mx, rms = robust_affine(Tc[both].astype(np.float64), Ts[both].astype(np.float64))
        else:
            Mx, rms = np.vstack([np.eye(3), np.zeros((1, 3))]), float("nan")
        Tca = np.clip(np.concatenate([Tc, np.ones((H, W, 1))], -1) @ Mx, 0, 1)
        info.update({"affine_rms": round(rms, 4), "affine_diag": np.diag(Mx[:3]).round(3).tolist(),
                     "affine_off": Mx[3].round(3).tolist()})
        I = np.where(Mc[..., None], Tca, Ts)
        known = Mc | (Ms & ~fp)
        hole = ~known
        hole = hole | (cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~Mc)   # half-covered rims
        painted = np.zeros((H, W), bool); lines = []; lat = {}
        if args.lines and hole.any():
            I, painted, lines = extend_lines(I, known & ~hole, hole, px, args.line_min_m, args.line_dev)
        copied = np.zeros((H, W), bool)
        if args.lattice and (hole & ~painted).any():
            I, copied, lat = lattice_fill(I, Mc & ~hole, hole & ~painted, px, args.lattice_min_corr)
        rest = hole & ~painted & ~copied
        res = inp(I, rest) if rest.any() else I
        dc_new = (np.clip(bilinear(res, fu, fv), 0, 1) - 0.5) / C0
        new_dc[idx] = dc_new
        fi_, fj_ = np.clip(fv.round().astype(int), 0, H - 1), np.clip(fu.round().astype(int), 0, W - 1)
        src[idx] = np.where(Mc[fi_, fj_], 1, np.where(painted[fi_, fj_], 2, np.where(copied[fi_, fj_], 4, 3)))
        info.update({"lines": len(lines), "lattice": lat, "discs_lattice": int((src[idx] == 4).sum()), "discs_camera": int(Mc[fi_, fj_].sum()), "discs_line": int((~Mc[fi_, fj_] & painted[fi_, fj_]).sum())})
        stats["patches"].append(info)
        if args.debug_dir and pi < args.debug_n:
            def lab(x, s_):
                x = (np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
                cv2.putText(x, s_, (4, 16), 0, 0.5, (0, 0, 255), 1); return x
            a = Ts.copy(); a[~Ms] = [1, 0, 1]
            b = Tca.copy(); b[~Mc] = [1, 0, 1]
            cimg = I.copy(); cimg[hole & ~painted] = [1, 0, 1]
            pv, _ = splat_texture(fu, fv, np.clip(0.5 + C0 * prev_dc[idx], 0, 1), np.ones(len(idx)),
                                  np.full(len(idx), 0.6 * spacing / px + 0.3), H, W)
            nv_, _ = splat_texture(fu, fv, np.clip(0.5 + C0 * dc_new, 0, 1), np.ones(len(idx)),
                                   np.full(len(idx), 0.6 * spacing / px + 0.3), H, W)
            edge = fp & ~cv2.erode(fp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            pv = np.where(fp[..., None], pv, Ts); nv_ = np.where(fp[..., None], nv_, I)
            pv[edge] = [0, 1, 1]; nv_[edge] = [0, 1, 1]
            tile = np.hstack([lab(a, "splat DC"), lab(b, "camera"), lab(cimg, "ctx+lines"), lab(res, "filled"),
                              lab(pv, "v2 discs"), lab(nv_, "v3 discs")])
            sc_ = min(1.0, 2400 / tile.shape[1])
            cv2.imwrite(os.path.join(args.debug_dir, f"patch{pi:03d}_{len(idx)}.jpg"),
                        cv2.resize(tile, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA))
        if pi < 25 or pi % 50 == 0:
            print(f"[v3] patch {pi}: {json.dumps(info)}", flush=True)
        if len(crops) < args.crops_n and len(idx) >= 150:
            crops.append((pi, c, n, float(ext.max())))

    for i, k in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        out[k][nb:] = new_dc[:, i].astype(np.float32)
    assert all(np.array_equal(out[k][:nb], base[k]) for k in out.dtype.names), "base splat must stay untouched"
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)), exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(args.out_ply)
    stats.update({"patches_n": len(patches), "patches_done": int(sum("fallback" not in p for p in stats["patches"])),
                  "discs_prev": int((src == 0).sum()), "discs_camera": int((src == 1).sum()),
                  "discs_line": int((src == 2).sum()), "discs_lama": int((src == 3).sum()),
                  "discs_lattice": int((src == 4).sum()),
                  "mean_abs_rgb_change_vs_prev": round(float((C0 * np.abs(new_dc - prev_dc)).mean()), 4),
                  "out": args.out_ply, "seconds": round(time.time() - t0, 1)})

    if args.crops_dataset and crops:                                         # oblique views onto the largest patches
        up = np.array([0, 0, 1.0]); fr0 = fr.f[0]; os.makedirs(os.path.join(args.crops_dataset, "sparse", "0"), exist_ok=True)
        fx, fy, cx, cy = fr0["K"]
        with open(os.path.join(args.crops_dataset, "sparse", "0", "cameras.txt"), "w") as fc:
            fc.write(f"1 PINHOLE {fr0['W']} {fr0['H']} {fx} {fy} {cx} {cy}\n")
        with open(os.path.join(args.crops_dataset, "sparse", "0", "images.txt"), "w") as fi:
            for k, (pi, c, n, e) in enumerate(crops):
                cm = fr.Cm[np.argsort(np.linalg.norm(fr.Cm - c, axis=1))[:20]].mean(0)   # side the robot was on
                ns = n if n @ (cm - c) >= 0 else -n
                tdir = (cm - c) - ((cm - c) @ ns) * ns
                tdir = tdir / np.linalg.norm(tdir) if np.linalg.norm(tdir) > 1e-6 else np.cross(ns, [1, 0, 0])
                D = max(1.0, 1.3 * e); Cmet = c + D * (0.75 * ns + 0.66 * tdir) / np.linalg.norm(0.75 * ns + 0.66 * tdir)
                Cd = ((Cmet - t) @ R) / s; cd = ((c - t) @ R) / s; upd = R.T @ up
                q, tt = look_at_quat(Cd, cd, upd)
                fi.write(f"{k + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in tt)} 1 patch{pi:03d}.png\n\n")
        stats["crops"] = [p[0] for p in crops]
    if args.debug_dir:
        json.dump(stats, open(os.path.join(args.debug_dir, "patches.json"), "w"), indent=1)
    stats.pop("patches")
    print("FILL_TEXTURE_V3", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
