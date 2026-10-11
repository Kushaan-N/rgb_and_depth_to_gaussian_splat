"""Appearance for completed object surfaces from the REAL training images (3DGRUT venv; GPU if present).

complete_objects.py samples each object's 2.5D surface (tops + walls facing the 4 object axes) every --cell
and writes <out>_samples.npz. This script textures those faces:
  1. observed texture: every sample is projected into every training frame where it is in view, facing the
     camera (< --max-incidence) and not occluded — its distance must match the base splat's rendered depth
     (<image>_splatdepth.npy) and, where present, the LiDAR depth target (<image>_depth.npy), with no depth
     edge in a 3x3 window — and the pixel colours are median-ed over frames (>= --min-views). This is the
     face rectified onto its plane from real images, at --cell resolution.
  2. per face image (tops: (a1, a2) plane; walls of direction f: (along-wall, height)), the pixels of
     samples to colour (the unseen, added ones) are filled, in order:
       mirror   the same (along-wall, height) pixel of the opposite face, if observed
       lattice  periodic tiling: the face's 2D lattice (masked normalised autocorrelation, bricks are periodic)
                maps the pixel onto an observed one (of this face, or the object's best-observed face)
       synth    patch-based exemplar synthesis (onion peel, --patch window SSD against the exemplar face)
       nearest  the nearest observed sample of the object
  3. colours -> 3DGRUT SH DC exactly as the renderer decodes it (rgb = max(C0 * dc + 0.5, 0), specular 0).

Library use: complete_objects.py --color-source textures calls texture_colours(). Standalone it recolours
the added gaussians of an existing completed checkpoint (e.g. one made with --color-source nearest):

    python scripts/face_textures.py --samples completed_samples.npz --dataset <train frames> --splat-depth-dir <dir> \
        --sim3-json sim3.json --checkpoint completed.pt --out completed_tex.pt [--kept harm_kept.npy] [--tex-dir textures/]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covis_masks import cams_of

C0 = 0.28209479177387814
HOW = {0: "direct", 1: "mirror", 2: "lattice", 3: "synth", 4: "nearest", -1: "none"}


def rgb2sh(rgb):
    return (np.asarray(rgb, np.float64) - 0.5) / C0


def gather_colours(posd, nord, dataset, splat_depth_dir, s, max_incidence=70.0, tol_rel=0.03, min_views=2, stride=1):
    """median real-image colour per sample (nan where seen by < min_views frames) and the view count."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(posd, device=dev); Nn = torch.tensor(nord, device=dev)
    cams = cams_of(dataset)[::stride]; cosmax = np.cos(np.deg2rad(max_incidence))
    acc = torch.full((len(cams), len(X), 3), float("nan"), dtype=torch.float16, device=dev)
    for f, (name, W, H, fx, fy, cx, cy, Rcw, tcw, dep) in enumerate(cams):
        sd = os.path.join(splat_depth_dir, os.path.splitext(name)[0] + "_splatdepth.npy") if splat_depth_dir else None
        have_sd = sd is not None and os.path.exists(sd); have_ld = os.path.exists(dep)
        if not (have_sd or have_ld):
            continue
        Rc = torch.tensor(Rcw, device=dev); tc = torch.tensor(tcw, device=dev)
        Pc = X @ Rc.T + tc; z = Pc[:, 2]
        uf = fx * Pc[:, 0] / z + cx - 0.5; vf = fy * Pc[:, 1] / z + cy - 0.5     # pixel centres at integers
        Cw = -Rc.T @ tc; vd = torch.nn.functional.normalize(Cw[None] - X, dim=1)
        ok = (z > 0.05) & (uf >= 1) & (uf < W - 2) & (vf >= 1) & (vf < H - 2) & ((vd * Nn).sum(1) > cosmax)
        ia = torch.nonzero(ok, as_tuple=True)[0]
        if not len(ia):
            continue
        u, v = uf[ia].round().long(), vf[ia].round().long(); r = torch.linalg.norm(Pc[ia], dim=1)
        good = torch.ones(len(ia), dtype=torch.bool, device=dev)
        for path, kind in ((sd, "splat"), (dep if have_ld else None, "lidar")):
            if path is None or not os.path.exists(path):
                continue
            D = torch.tensor(np.load(path).astype(np.float32), device=dev)[None, None]
            Dmax = F.max_pool2d(D, 3, 1, 1)[0, 0].double(); Dmin = -F.max_pool2d(-torch.where(D > 0, D, torch.full_like(D, 1e9)), 3, 1, 1)[0, 0].double()
            d0 = D[0, 0].double()[v, u]; tol = torch.clamp(tol_rel * d0, min=0.02 / s)
            if kind == "splat":                                          # dense: required everywhere
                good &= (d0 > 0) & ((r - d0).abs() < tol) & (Dmax[v, u] - Dmin[v, u] < 2 * tol)
            else:                                                        # sparse LiDAR: checked where present
                good &= (d0 <= 0) | (((r - d0).abs() < tol) & (Dmax[v, u] - Dmin[v, u] < 2 * tol))
        if not have_sd:
            good &= torch.tensor(np.load(dep).astype(np.float32), device=dev).double()[v, u] > 0
        ia = ia[good]
        if not len(ia):
            continue
        img = torch.tensor(np.asarray(Image.open(os.path.join(dataset, "images", name)).convert("RGB")), device=dev).float() / 255
        g = torch.stack([uf[ia] / (W - 1) * 2 - 1, vf[ia] / (H - 1) * 2 - 1], 1).float()[None, None]
        col = F.grid_sample(img.permute(2, 0, 1)[None], g, mode="bilinear", align_corners=True)[0, :, 0].T
        acc[f, ia] = col.half()
    nv = (~torch.isnan(acc[..., 0])).sum(0)
    med = torch.nanmedian(acc.float(), dim=0).values
    med[nv < min_views] = float("nan")
    return med.cpu().numpy().astype(np.float64), nv.cpu().numpy()


# ------------------------------------------------------------------ 2D face images
def lattice(V, K, min_corr=0.5, min_px=30):
    """lattice vectors [(dy, dx), ...] (0-2 of them) of a partially known image V (H,W,3), known mask K."""
    G = V.mean(-1); n = K.sum()
    if n < min_px:
        return []
    Gc = np.where(K, G - G[K].mean(), 0.0)
    if Gc[K].std() < 0.02:                                              # flat face: no pattern to tile
        return []
    H, W = G.shape; sh = (2 * H, 2 * W)
    fa, fb, fc = np.fft.rfft2(Gc, sh), np.fft.rfft2(K.astype(float), sh), np.fft.rfft2(Gc ** 2, sh)
    cor = lambda f, g: np.fft.irfft2(np.conj(f) * g, sh)
    num, A, B, cnt = cor(fa, fa), cor(fc, fb), cor(fb, fc), cor(fb, fb)
    ncc = num / np.sqrt(np.maximum(A * B, 1e-12))
    dy = np.fft.fftfreq(sh[0], 1 / sh[0]).astype(int)[:, None]; dx = np.fft.fftfreq(sh[1], 1 / sh[1]).astype(int)[None, :]
    valid = (cnt > max(min_px, 0.25 * n)) & (dy ** 2 + dx ** 2 >= 4) & ((dy > 0) | ((dy == 0) & (dx > 0)))
    from scipy import ndimage
    peak = valid & (ncc >= ndimage.maximum_filter(np.where(valid, ncc, -9), size=3, mode="wrap")) & (ncc >= min_corr)
    ys, xs = np.nonzero(peak)
    if not len(ys):
        return []
    cand = sorted(zip(ncc[ys, xs], dy[ys, 0], dx[0, xs]), reverse=True)
    best = cand[0][0]
    good = [(c, a, b) for c, a, b in cand if c >= 0.85 * best]
    v1 = min(good, key=lambda t: t[1] ** 2 + t[2] ** 2)[1:]
    out = [v1]
    c2 = [(c, a, b) for c, a, b in cand if abs(a * v1[1] - b * v1[0]) > 0.5 * np.hypot(a, b) * np.hypot(*v1)]   # > 30 deg off v1
    if c2:
        b2 = c2[0][0]
        out.append(min([t for t in c2 if t[0] >= 0.85 * b2], key=lambda t: t[1] ** 2 + t[2] ** 2)[1:])
    return out


def lattice_fill(T, need, E, EK, vecs, reach=12):
    """fill need pixels of T from known pixels of E at p + i v1 + j v2 (nearest lattice offset first)."""
    if not vecs:
        return np.zeros_like(need)
    v1 = np.array(vecs[0]); v2 = np.array(vecs[1]) if len(vecs) > 1 else np.zeros(2, int)
    rng = range(-reach, reach + 1)
    offs = sorted({tuple(i * v1 + j * v2) for i in rng for j in (rng if len(vecs) > 1 else [0])} - {(0, 0)},
                  key=lambda o: o[0] ** 2 + o[1] ** 2)
    ys, xs = np.nonzero(need); done = np.zeros(len(ys), bool); H, W = EK.shape
    for oy, ox in offs:
        if done.all():
            break
        qy, qx = ys + oy, xs + ox
        ok = ~done & (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
        ok[ok] = EK[qy[ok], qx[ok]]
        T[ys[ok], xs[ok]] = E[qy[ok], qx[ok]]; done |= ok
    filled = np.zeros_like(need); filled[ys[done], xs[done]] = True
    return filled


def synth_fill(T, TK, need, E, EK, patch=7, rng=None):
    """patch-based exemplar synthesis (Efros-Leung onion peel): each need pixel takes the centre colour of
    the exemplar window that best matches its already-known neighbourhood (masked, Gaussian-weighted SSD)."""
    rng = rng or np.random.default_rng(0)
    from scipy import ndimage
    for p in (patch, 5, 3):
        h = p // 2; H, W = EK.shape
        if H < p or W < p:
            continue
        cy, cx = np.mgrid[h:H - h, h:W - h]; cy, cx = cy.ravel(), cx.ravel()
        oy, ox = np.mgrid[-h:h + 1, -h:h + 1]; oy, ox = oy.ravel(), ox.ravel()
        SM = EK[cy[:, None] + oy, cx[:, None] + ox]
        sel = EK[cy, cx] & (SM.mean(1) >= 0.7)
        if sel.sum() >= 10:
            break
    else:
        return np.zeros_like(need)
    S = E[cy[sel][:, None] + oy, cx[sel][:, None] + ox]; SM = SM[sel]; centre = E[cy[sel], cx[sel]]
    g = np.exp(-(oy ** 2 + ox ** 2) / (2 * (p / 6.4) ** 2))
    Hh, Ww = TK.shape; K = TK.copy(); todo = need & ~K; filled = np.zeros_like(need)
    Tp = np.pad(T, ((h, h), (h, h), (0, 0))); Kp = np.pad(K, h)
    stuck = False
    while todo.any():
        nb = ndimage.convolve(K.astype(int), np.ones((3, 3), int), mode="constant") * todo
        if nb.max() == 0 or stuck:                                               # disconnected from anything known: seed
            y, x = np.argwhere(todo)[0]; c = centre[rng.integers(len(centre))]
            T[y, x] = c; Tp[y + h, x + h] = c; K[y, x] = Kp[y + h, x + h] = True; todo[y, x] = False; filled[y, x] = True
            stuck = False
            continue
        front = np.argwhere(nb == nb.max()); before = todo.sum()
        for y, x in front[rng.permutation(len(front))]:
            win = Tp[y:y + p, x:x + p].reshape(-1, 3); wm = Kp[y:y + p, x:x + p].ravel()
            w = g * wm; m = SM * w[None]
            den = m.sum(1)
            d = (((S - win[None]) ** 2).sum(2) * m).sum(1) / np.maximum(den, 1e-9)
            d[den < 1e-6] = np.inf
            if not np.isfinite(d).any():
                continue
            ok = np.nonzero(d <= d.min() * 1.1 + 1e-6)[0]; c = centre[ok[rng.integers(len(ok))]]
            T[y, x] = c; Tp[y + h, x + h] = c; K[y, x] = Kp[y + h, x + h] = True; todo[y, x] = False; filled[y, x] = True
        stuck = todo.sum() == before
    return filled


def texture_colours(S, rgb_obs, need, cell, tex_dir=None, seed=0, patch=7, min_exemplar_px=40):
    """colours for the `need` samples from face images of each object. S: samples dict (obj, kind, face,
    local). Returns rgb (n,3) (nan where impossible) and how (n,) codes (see HOW)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    n = len(S["obj"]); rgb = np.full((n, 3), np.nan); how = np.full(n, -1)
    direct = need & ~np.isnan(rgb_obs[:, 0]); rgb[direct] = rgb_obs[direct]; how[direct] = 0
    for o in np.unique(S["obj"]):
        io = np.nonzero(S["obj"] == o)[0]; L = S["local"][io]; kd = S["kind"][io]; fc = S["face"][io]
        org = L.min(0)
        keys = {("top",): kd == 0}
        for f in range(4):
            keys[("wall", f)] = (kd == 1) & (fc == f)
        imgs = {}
        for key, m in keys.items():
            if not m.any():
                continue
            if key[0] == "top":
                xy = np.round((L[m][:, [1, 0]] - org[[1, 0]]) / cell).astype(int)       # rows a2, cols a1
            else:
                ax = 1 if key[1] in (0, 1) else 0
                xy = np.round(np.c_[L[m][:, 2], L[m][:, ax] - org[ax]] / cell).astype(int)   # rows height, cols along
                xy[:, 0] = int(round(L[:, 2].max() / cell)) - xy[:, 0]                       # top row = highest
            imgs[key] = {"idx": io[m], "yx": xy}
        Hm = {k: tuple(v["yx"].max(0) + 1) for k, v in imgs.items()}
        # walls of opposite faces share one grid so mirroring is a same-pixel copy
        for a, b in ((("wall", 0), ("wall", 1)), (("wall", 2), ("wall", 3))):
            if a in Hm and b in Hm:
                Hm[a] = Hm[b] = tuple(np.maximum(Hm[a], Hm[b]))
        for k, im in imgs.items():
            H, W = Hm[k]; V = np.zeros((H, W, 3)); C = np.zeros((H, W)); NEED = np.zeros((H, W), bool)
            ob = ~np.isnan(rgb_obs[im["idx"], 0]); y, x = im["yx"].T
            np.add.at(V, (y[ob], x[ob]), rgb_obs[im["idx"][ob]]); np.add.at(C, (y[ob], x[ob]), 1)
            nd = need[im["idx"]] & (how[im["idx"]] < 0); NEED[y[nd], x[nd]] = True
            im.update(V=V / np.maximum(C, 1)[..., None], K=C > 0, NEED=NEED & ~(C > 0), HOW=np.full((H, W), -1))
            im["HOW"][C > 0] = 0; im["V0"] = im["V"].copy()
        for k, im in imgs.items():                                      # 1. mirror
            if k[0] != "wall":
                continue
            o_ = ("wall", {0: 1, 1: 0, 2: 3, 3: 2}[k[1]])
            if o_ in imgs:
                m = im["NEED"] & imgs[o_]["K"]
                im["V"][m] = imgs[o_]["V0"][m]; im["HOW"][m] = 1; im["NEED"] &= ~m
        known_px = lambda im: int((im["HOW"] == 0).sum())
        for k, im in imgs.items():
            if not im["NEED"].any():
                continue
            same = [kk for kk in imgs if kk[0] == k[0]]; other = [kk for kk in imgs if kk[0] != k[0]]
            pool = sorted(same, key=lambda kk: (kk != k, -known_px(imgs[kk]))) + sorted(other, key=lambda kk: -known_px(imgs[kk]))
            ex = next((kk for kk in pool if known_px(imgs[kk]) >= min_exemplar_px),
                      max(pool, key=lambda kk: known_px(imgs[kk])))
            E = imgs[ex]; EK = E["HOW"] == 0
            if not EK.any():
                continue
            vecs = lattice(E["V0"], EK)                                   # 2. lattice tiling
            m = lattice_fill(im["V"], im["NEED"], E["V0"], EK, vecs)
            im["HOW"][m] = 2; im["NEED"] &= ~m
            TK = im["HOW"] >= 0                                          # 3. exemplar synthesis
            m = synth_fill(im["V"], TK, im["NEED"], E["V0"], EK, patch, rng)
            im["HOW"][m] = 3; im["NEED"] &= ~m
            im["exemplar"] = "/".join(map(str, ex)); im["lattice"] = [list(map(int, v)) for v in vecs]
        for k, im in imgs.items():                                      # back to samples
            y, x = im["yx"].T; sel = (how[im["idx"]] < 0) & need[im["idx"]] & (im["HOW"][y, x] >= 0)
            rgb[im["idx"][sel]] = im["V"][y[sel], x[sel]]; how[im["idx"][sel]] = im["HOW"][y[sel], x[sel]]
        left = io[need[io] & (how[io] < 0)]; src = io[~np.isnan(rgb_obs[io, 0])]   # 4. nearest observed sample
        if len(left) and len(src):
            _, j = cKDTree(S["local"][src]).query(S["local"][left]); rgb[left] = rgb_obs[src[j]]; how[left] = 4
        if tex_dir:
            save_montage(imgs, os.path.join(tex_dir, f"object{int(o)}.png"))
            json.dump({"/".join(map(str, k)): {"shape": list(im["V"].shape[:2]), "observed_px": known_px(im),
                       "filled": {HOW[c]: int((im["HOW"] == c).sum()) for c in (1, 2, 3)},
                       "exemplar": im.get("exemplar"), "lattice": im.get("lattice")} for k, im in imgs.items()},
                      open(os.path.join(tex_dir, f"object{int(o)}.json"), "w"), indent=1)
    return rgb, how


def save_montage(imgs, path, z=6):
    """per face: observed texture (magenta = to fill, grey = no sample) | filled texture | fill source map."""
    import cv2
    rows = []
    pal = {-1: (40, 40, 40), 0: (0, 160, 0), 1: (255, 160, 0), 2: (0, 120, 255), 3: (200, 0, 200), 4: (255, 255, 0)}
    for k, im in imgs.items():
        a = im["V0"].copy(); a[~im["K"]] = 0.15; a[(im["HOW"] > 0) | im["NEED"]] = (1, 0, 1)
        b = im["V"].copy(); b[im["HOW"] < 0] = 0.15
        c = np.zeros_like(b)
        for code, col in pal.items():
            c[im["HOW"] == code] = np.array(col) / 255
        t = np.concatenate([a, np.ones((a.shape[0], 1, 3)), b, np.ones((a.shape[0], 1, 3)), c], 1)
        t = cv2.resize((t[..., ::-1] * 255).clip(0, 255).astype(np.uint8), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        lab = np.full((18, t.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, "/".join(map(str, k)) + "  observed | filled | source (green obs, orange mirror, blue lattice, purple synth)",
                    (2, 13), 0, 0.4, (0, 0, 0), 1)
        rows.append(np.vstack([lab, t]))
    if rows:
        W = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cv2.imwrite(path, np.vstack(rows))


def load_samples(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="complete_objects.py <out>_samples.npz")
    ap.add_argument("--dataset", required=True, help="training frames with images (+ *_depth.npy)")
    ap.add_argument("--splat-depth-dir", default=None)
    ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--checkpoint", default=None, help="completed checkpoint whose added gaussians to recolour")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kept", default=None, help="bool per added sample kept by a later filter (harm/finetune _kept.npy)")
    ap.add_argument("--tex-dir", default=None)
    ap.add_argument("--max-incidence", type=float, default=70.0)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--patch", type=int, default=7)
    args = ap.parse_args()
    import torch
    S = load_samples(args.samples); s = float(json.load(open(args.sim3_json))["scale"])
    need = S["added"].astype(bool)
    rgb_obs, nv = gather_colours(S["posd"], S["nord"], args.dataset, args.splat_depth_dir, s, args.max_incidence, min_views=args.min_views)
    rgb, how = texture_colours(S, rgb_obs, need, float(S["cell"]), args.tex_dir, patch=args.patch)
    stats = {HOW[c]: int((how[need] == c).sum()) for c in HOW}
    print(f"[textures] {need.sum():,} samples to colour: " + ", ".join(f"{k} {v:,}" for k, v in stats.items()), flush=True)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ai = np.nonzero(need)[0]
        if args.kept:
            ai = ai[np.load(args.kept)]
        N = ck["positions"].shape[0]; n0 = N - len(ai)
        ok = ~np.isnan(rgb[ai, 0]); rows = torch.from_numpy(n0 + np.nonzero(ok)[0])
        for key, val in (("features_albedo", torch.tensor(rgb2sh(rgb[ai[ok]]))), ("features_specular", None)):
            v = ck[key]; t = v.detach().clone()
            t[rows] = val.to(t.dtype) if val is not None else 0
            ck[key] = torch.nn.Parameter(t, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else t
        torch.save(ck, args.out)
        print(f"[textures] recoloured {ok.sum():,} of {len(ai):,} added gaussians -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
