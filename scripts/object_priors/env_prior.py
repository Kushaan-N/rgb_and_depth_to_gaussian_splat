"""Environment prior around free-standing objects (cosmetic: invented content allowed): the floor and walls the scene
knows are continued behind / under every object, textured from the surrounding visible floor / wall (inpev venv:
venv-inpaint, LaMa on GPU if present). Composable: input = any candidate PLY + its src map (object_priors.py
textured.ply + src.npz, or another agent's output with a src map); output = the same rows + appended discs.

  layout   floor plane + Manhattan axes from find_objects.py (objects.json); walls = vertical planes along the
           Manhattan axes, peaks of the LiDAR static cloud's projection (>= --wall-min-pts, near an object).
  floor    per object, a top-down orthophoto (--px m/pixel, window = footprint + --win-m) of the candidate's floor band
           (|h| < --band-m, opacity-weighted splat footprints): colour + accumulated opacity. Holes = opacity < --solid.
           Texture: LaMa on the orthophoto (holes only), then structured patterns: tape-like lines (strongly saturated
           colour, Hough segments) that are collinear (angle < 4 deg, offset < 2 px) are joined straight through hole
           spans, painted with their median colour / width. Soft contact shadow around the object's footprint
           (1 - --shadow exp(-d / --shadow-m)), darker under it (--under). Discs (opaque, flat, on the plane) on hole
           pixels within --zone-m of the footprint.
  walls    within --zone-m of an object, wall-plane samples (up to --wall-h) not covered by an opaque candidate
           gaussian within --cover-m get discs, colour = median of the wall's covered samples within 0.5 m.
  Protection: no disc within --protect-m of a well-seen gaussian (support max blend weight >= --face-w).

    python scripts/object_priors/env_prior.py --ply T/textured.ply --src T/src.npz --support D/support.npz \
        --objects D/objects --cloud lidar.ply --sim3-json sim3.json --out-dir T/env [--lama-pt big-lama.pt]
  -> T/env/textured_env.ply, src_env.npz, env.json, ortho_o<id>.jpg (before | holes | inpainted+lines+shadow)
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import sim3, read_vertex, write_vertex, gauss_geom, load_objects, disc_rows, C0  # noqa: E402


def ortho(U, V, op, rgb, rad, W, H):
    """splat footprints (pixel coords U,V, radius rad px) -> colour HxWx3, opacity HxW."""
    acc = np.zeros((H, W, 3)); ws = np.zeros((H, W)); lt = np.zeros((H, W))
    R = int(np.ceil(rad.max())) if len(rad) else 0
    ui, vi = np.round(U).astype(int), np.round(V).astype(int)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d2 = (ui + dx - U) ** 2 + (vi + dy - V) ** 2
            m = (d2 <= rad ** 2) & (ui + dx >= 0) & (ui + dx < W) & (vi + dy >= 0) & (vi + dy < H)
            if not m.any():
                continue
            a = np.minimum(op[m] * np.exp(-0.5 * d2[m] / np.maximum(rad[m] / 2, 0.5) ** 2), 0.99)
            x, y = ui[m] + dx, vi[m] + dy
            np.add.at(acc, (y, x), a[:, None] * rgb[m]); np.add.at(ws, (y, x), a); np.add.at(lt, (y, x), np.log1p(-a))
    return acc / np.maximum(ws, 1e-9)[..., None], 1 - np.exp(lt)


def tape_lines(img, known, hole, sat_min=0.35, ang_tol=4.0, off_tol=2.0):
    """join collinear saturated line segments across holes; returns painted image + the painted mask."""
    import cv2
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8); hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    tape = known & (hsv[..., 1] / 255.0 > sat_min) & (hsv[..., 2] > 40)
    out = img.copy(); painted = np.zeros(hole.shape, bool)
    if tape.sum() < 30:
        return out, painted, 0
    segs = cv2.HoughLinesP(tape.astype(np.uint8) * 255, 1, np.pi / 360, 15, minLineLength=12, maxLineGap=4)
    if segs is None:
        return out, painted, 0
    segs = segs[:, 0].astype(float)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    width = float(np.clip(2 * cv2.distanceTransform(tape.astype(np.uint8), cv2.DIST_L2, 3).max(), 2, 12))
    col = np.median(img[tape], 0); used = np.zeros(len(segs), bool); njoin = 0
    for i in np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])):
        if used[i]:
            continue
        th = np.radians(ang[i]); dvec = np.array([np.cos(th), np.sin(th)]); nvec = np.array([-dvec[1], dvec[0]])
        p0 = segs[i, :2]
        da = np.abs((ang - ang[i] + 90) % 180 - 90)
        off = np.abs(np.stack([segs[:, 0] - p0[0], segs[:, 1] - p0[1]], 1) @ nvec)
        off2 = np.abs(np.stack([segs[:, 2] - p0[0], segs[:, 3] - p0[1]], 1) @ nvec)
        grp = (da < ang_tol) & (off < off_tol + width / 2) & (off2 < off_tol + width / 2) & ~used
        used |= grp
        if grp.sum() < 2:
            continue
        ts = np.concatenate([(segs[grp, :2] - p0) @ dvec, (segs[grp, 2:] - p0) @ dvec])
        a, b = ts.min(), ts.max()
        line = np.zeros(hole.shape, np.uint8)
        q0, q1 = p0 + a * dvec, p0 + b * dvec
        cv2.line(line, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), 1, int(round(width)))
        m = (line > 0) & hole
        out[m] = col; painted |= m; njoin += 1
    return out, painted, njoin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--src", required=True); ap.add_argument("--support", required=True)
    ap.add_argument("--objects", required=True); ap.add_argument("--cloud", required=True); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--lama-pt", default=None)
    ap.add_argument("--px", type=float, default=0.01); ap.add_argument("--win-m", type=float, default=0.8)
    ap.add_argument("--zone-m", type=float, default=0.35); ap.add_argument("--band-m", type=float, default=0.03)
    ap.add_argument("--solid", type=float, default=0.6); ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--opacity", type=float, default=0.98); ap.add_argument("--lift-m", type=float, default=0.002)
    ap.add_argument("--shadow", type=float, default=0.3); ap.add_argument("--shadow-m", type=float, default=0.06)
    ap.add_argument("--under", type=float, default=0.55, help="floor under an object (A-frame inside): colour x this")
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--protect-m", type=float, default=0.012)
    ap.add_argument("--wall-min-pts", type=int, default=800); ap.add_argument("--wall-h", type=float, default=0.6)
    ap.add_argument("--cover-m", type=float, default=0.02)
    ap.add_argument("--walls", action="store_true", help="also continue room walls (only at the LiDAR room boundary)")
    ap.add_argument("--wall-boundary-m", type=float, default=0.3, help="a wall plane must be this close to the cloud's extreme")
    ap.add_argument("--wall-obj-m", type=float, default=0.5, help="only behind objects this close to the wall")
    ap.add_argument("--dataset", nargs="*", default=[], help="training frames: discs a camera would see in front of the input are dropped")
    ap.add_argument("--splat-depth-dir", default=None, help="the input's rendered depth at --dataset (render_ply_depth.py)")
    ap.add_argument("--floor-tol-m", type=float, default=0.01)
    a = ap.parse_args()
    import cv2
    from scipy.spatial import cKDTree
    from scipy import ndimage
    from plyfile import PlyData
    os.makedirs(a.out_dir, exist_ok=True)
    s, R, t = sim3(a.sim3_json)
    v = read_vertex(a.ply); N = len(v); src = np.load(a.src)["src"]; assert len(src) == N
    wmax_in = np.load(a.support)["wmax"]; wmax = np.where(src >= 0, wmax_in[np.maximum(src, 0)], 0.0)
    Gm, op, sc = gauss_geom(v, s, R, t); smid = np.sort(sc, 1)[:, 1]
    rgb = np.clip(0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64), 0, 1)
    meta, opts = load_objects(a.objects); n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"])
    e1, e2 = np.array(meta["axes"]["e1"]), np.array(meta["axes"]["e2"])
    h = Gm @ n + d; U0, V0 = Gm @ e1, Gm @ e2
    seen_tree = cKDTree(Gm[(wmax >= a.face_w) & (op > 0.7)])
    cv_ = PlyData.read(a.cloud)["vertex"].data; L = np.stack([cv_["x"], cv_["y"], cv_["z"]], 1).astype(np.float64)
    L = L[np.unique(np.floor(L / 0.02).astype(np.int64), axis=0, return_index=True)[1]]; hL = L @ n + d
    band = (np.abs(h) < a.band_m) & (op > 0.2)
    new_X, new_N, new_c, rep = [], [], [], {"objects": []}
    if a.lama_pt and os.path.exists(a.lama_pt):
        from fill_inpaint import Inpainter
        inp = Inpainter("lama", a.lama_pt)
    else:
        from fill_inpaint import Inpainter
        inp = Inpainter("telea", None)
    # ---- walls: Manhattan peaks of the LiDAR projection (0.1 .. 1.5 m above the floor)
    walls = []
    if a.walls:
        Lw = L[(hL > 0.1) & (hL < 1.5)]
        for ax, e in (("e1", e1), ("e2", e2)):
            p = Lw @ e; hist, edges = np.histogram(p, bins=np.arange(p.min(), p.max() + 0.02, 0.02))
            lo_p, hi_p = np.percentile(p, [0.5, 99.5])
            for i in np.nonzero(hist >= a.wall_min_pts)[0]:
                c_ = (edges[i] + edges[i + 1]) / 2
                if hist[i] == hist[max(i - 2, 0):i + 3].max() and min(abs(c_ - lo_p), abs(c_ - hi_p)) < a.wall_boundary_m:
                    walls.append((ax, e, float((edges[i] + edges[i + 1]) / 2), int(hist[i])))
    rep["walls"] = [(w[0], round(w[2], 3), w[3]) for w in walls]
    for o, X in zip(meta["objects"], opts):
        hx = X @ n + d; F = X[hx > 0.03]
        uc, vc = float(np.median(F @ e1)), float(np.median(F @ e2))
        fu, fv = F @ e1, F @ e2
        lo = np.array([fu.min(), fv.min()]) - a.win_m; hi = np.array([fu.max(), fv.max()]) + a.win_m
        W_, H_ = int(np.ceil((hi[0] - lo[0]) / a.px)), int(np.ceil((hi[1] - lo[1]) / a.px))
        sel = band & (U0 > lo[0]) & (U0 < hi[0]) & (V0 > lo[1]) & (V0 < hi[1])
        Uq, Vq = (U0[sel] - lo[0]) / a.px, (V0[sel] - lo[1]) / a.px
        col, alpha = ortho(Uq, Vq, op[sel], rgb[sel], np.clip(1.5 * smid[sel] / a.px, 0.7, 4.0), W_, H_)
        known = alpha >= a.solid
        foot = np.zeros((H_, W_), bool)
        fx_, fy_ = np.clip(((fu - lo[0]) / a.px).astype(int), 0, W_ - 1), np.clip(((fv - lo[1]) / a.px).astype(int), 0, H_ - 1)
        foot[fy_, fx_] = True; foot = ndimage.binary_fill_holes(ndimage.binary_closing(foot, iterations=3))
        dist = ndimage.distance_transform_edt(~foot) * a.px
        zone = dist <= a.zone_m
        hole = ~known
        inpainted = inp(np.where(known[..., None], col, 0.5), hole)
        lined, painted, njoin = tape_lines(inpainted, known, hole)
        shade = np.where(foot, a.under, 1 - a.shadow * np.exp(-dist / a.shadow_m))
        final = np.clip(lined * shade[..., None], 0, 1)
        put = hole & zone
        step = max(1, int(round(a.cell / a.px)))
        yy, xx = np.nonzero(put); keepi = (yy % step == 0) & (xx % step == 0); yy, xx = yy[keepi], xx[keepi]
        P = np.outer((xx + 0.5) * a.px + lo[0], e1) + np.outer((yy + 0.5) * a.px + lo[1], e2)
        P = P - np.outer(P @ n + d - a.lift_m, n)                          # onto the floor plane (+ lift)
        ok = ~np.isfinite(seen_tree.query(P, distance_upper_bound=a.protect_m)[0])
        new_X.append(P[ok]); new_N.append(np.tile(n, (ok.sum(), 1))); new_c.append(final[yy[ok], xx[ok]])
        orep = {"id": o["id"], "ortho_px": [W_, H_], "floor_known_in_zone": round(float(known[zone].mean()), 3),
                "floor_discs": int(ok.sum()), "tape_lines_joined": njoin, "tape_px_painted": int(painted.sum())}
        tiles = [col * known[..., None], np.dstack([hole & zone] * 3).astype(float), final]
        cv2.imwrite(os.path.join(a.out_dir, f"ortho_o{o['id']}.jpg"),
                    cv2.cvtColor((np.hstack(tiles) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)[::-1])
        # ---- walls near this object
        nw = 0
        for ax, e, off, cnt in walls:
            other = e2 if ax == "e1" else e1
            dpl = F @ e - off
            if np.abs(dpl).min() > a.wall_obj_m:
                continue
            side = np.sign(np.median(dpl)) or 1.0
            along = np.arange((F @ other).min() - a.zone_m, (F @ other).max() + a.zone_m, a.cell)
            hh = np.arange(a.cell / 2, a.wall_h, a.cell)
            A_, Hh = np.meshgrid(along, hh, indexing="ij")
            base = off * e + np.outer(A_.ravel(), other)
            P = base - np.outer(base @ n + d, n) + np.outer(Hh.ravel(), n)
            ctree = cKDTree(Gm[op > 0.5]); cov = np.isfinite(ctree.query(P, distance_upper_bound=a.cover_m)[0])
            ok = ~cov & ~np.isfinite(seen_tree.query(P, distance_upper_bound=a.protect_m)[0])
            if not ok.any() or cov.sum() < 20:
                continue
            gi = np.nonzero(op > 0.5)[0][ctree.query(P[cov])[1]]; ccol = rgb[gi]
            ct = cKDTree(P[cov]); dd, jj = ct.query(P[ok], k=min(16, cov.sum()))
            wcol = np.median(ccol[np.atleast_2d(jj)], 1)
            new_X.append(P[ok]); new_N.append(np.tile(-side * e, (ok.sum(), 1))); new_c.append(wcol); nw += int(ok.sum())
        orep["wall_discs"] = nw
        rep["objects"].append(orep); print(f"[env] {orep}", flush=True)
    X = np.concatenate(new_X) if new_X else np.zeros((0, 3)); Nn = np.concatenate(new_N) if new_N else np.zeros((0, 3))
    C = np.concatenate(new_c) if new_c else np.zeros((0, 3))
    keep = np.ones(len(X), bool)
    if a.dataset and a.splat_depth_dir and len(X):                       # never in front of what a training camera saw
        from object_priors import seen_by_cameras
        keep = ~seen_by_cameras(X, a.dataset, a.splat_depth_dir, s, R, t, 0.05, 0.98, 0.05, thru=True)
    fl = np.abs(X @ n + d - a.lift_m) < a.floor_tol_m
    isfloor = np.all(np.abs(Nn - n) < 1e-6, 1); keep &= ~isfloor | fl
    rep["dropped_in_front_of_cameras"] = int((~keep).sum())
    X, Nn, C = X[keep], Nn[keep], C[keep]
    rows = disc_rows(v.dtype, ((X - t) @ R) / s, Nn @ R, (np.clip(C, 0, 1) - 0.5) / C0, a.cell, s, a.opacity)
    write_vertex(os.path.join(a.out_dir, "textured_env.ply"), np.concatenate([v, rows]))
    np.savez_compressed(os.path.join(a.out_dir, "src_env.npz"), src=np.concatenate([src, np.full(len(rows), -1)]))
    rep["added"] = int(len(rows)); json.dump(rep, open(os.path.join(a.out_dir, "env.json"), "w"), indent=1)
    print("ENV_PRIOR", json.dumps({"added": rep["added"], "walls": rep["walls"]}), flush=True)


if __name__ == "__main__":
    main()
