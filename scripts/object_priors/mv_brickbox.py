"""Brick-stack rebuild as a clean textured stepped box (gsplat venv for the visibility test; cosmetic: unseen faces get
texture tiled from the stack's best-seen face).

Per object (--ids): geometry = object_priors.py's stepped box fitted to the splat's well-seen surfels (the --splat-faces
evidence: support max blend weight >= --face-w, opaque, flat) + a 1/10 LiDAR subsample; its surface sampled every --cell.
Colour of every surface sample a training camera sees (in view, facing it, in front of the base's rendered depth within
--tol-rel) = the real image pixel from the most frontal / closest such camera. Unseen samples: walls copy the best-seen
wall face at the same height (brick courses / mortar lines stay level across faces) and the same along-face coordinate
modulo that face's seen extent; tops copy the best-seen top region modulo its extent. Rows: well-seen base rows in the box
are kept (front faces keep their original gaussians), every other base row in the box above the floor is deleted, and an
opaque flat disc is added per surface sample farther than --keep-m from a kept row.

    python scripts/object_priors/mv_brickbox.py --ply base.ply --support D/support.npz --objects D/objects \
        --sim3-json sim3.json --dataset <train ds> --splat-depth-dir D/depth --ids 3 4 5 --out T/refit.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from solidify_unseen import sim3, gauss_geom, load_objects, read_vertex  # noqa: E402
from object_priors import Obj, fit_stepbox, stepbox_surface  # noqa: E402
from mv_refit import quat_from_z  # noqa: E402

C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--support", required=True); ap.add_argument("--objects", required=True)
    ap.add_argument("--sim3-json", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--splat-depth-dir", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.004); ap.add_argument("--fit-cell", type=float, default=0.02)
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--ev-min-h", type=float, default=0.03)
    ap.add_argument("--zone-m", type=float, default=0.02); ap.add_argument("--floor-m", type=float, default=0.01)
    ap.add_argument("--keep-m", type=float, default=0.008); ap.add_argument("--tol-rel", type=float, default=0.02)
    ap.add_argument("--min-cos", type=float, default=0.35); ap.add_argument("--opacity", type=float, default=0.98)
    ap.add_argument("--keep-seen", type=int, default=1, help="0: delete every base row in the box (all faces rebuilt)")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from scipy.spatial import cKDTree
    from covis_masks import cams_of
    from finetune_ply_viewer import load_ply, save_ply
    dev = "cuda"
    s, R, t = sim3(a.sim3_json); to_ds = lambda X: ((X - t) @ R) / s
    v = read_vertex(a.ply); Gm, op, sc = gauss_geom(v, s, R, t); N0 = len(v)
    ss = np.sort(sc, 1); wmax = np.load(a.support)["wmax"]; assert len(wmax) == N0
    surfel = (wmax >= a.face_w) & (op > 0.7) & (ss[:, 0] < 0.3 * ss[:, 1]) & (ss[:, 2] < 3 * ss[:, 1])
    ometa, opts = load_objects(a.objects); n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    cams = cams_of(a.dataset)
    imgs = {}
    sdep = [np.load(os.path.join(a.splat_depth_dir, os.path.splitext(c[0])[0] + "_splatdepth.npy")).astype(np.float32) for c in cams]
    T = lambda x: torch.tensor(np.asarray(x), device=dev, dtype=torch.float64)
    dele = np.zeros(N0, bool); add_X, add_N, add_C, rep = [], [], [], {}
    for o, X in zip(ometa["objects"], opts):
        if o["id"] not in a.ids:
            continue
        ob = Obj(X, n_f, d_f); Lg = ob.loc(Gm); zh = ob.half + 0.12
        z0 = (np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > a.ev_min_h)
        ob = Obj(np.concatenate([X[::10], Gm[z0 & surfel]]), n_f, d_f, min_h=a.ev_min_h)
        Lg = ob.loc(Gm); Lx = ob.loc(X)
        hmax = float(np.percentile(Lx[:, 2], 99))
        inb = (np.abs(Lg[:, 0]) <= ob.half[0] + a.zone_m) & (np.abs(Lg[:, 1]) <= ob.half[1] + a.zone_m) & (Lg[:, 2] > a.floor_m) & (Lg[:, 2] < hmax + 0.05)
        Lev = np.concatenate([Lx[Lx[:, 2] > a.ev_min_h][::10], Lg[inb & surfel & (Lg[:, 2] > a.ev_min_h)]])
        pr = fit_stepbox(Lev, ob.half, a.fit_cell)
        if pr is None:
            print(f"[brickbox] object {o['id']}: no box", flush=True); continue
        Ps, Ns, Fs = stepbox_surface(pr, a.cell)
        Pm = ob.met(Ps); Nm = ob.vec(Ns); Pd = to_ds(Pm); Nd = Nm @ R
        # colour from the most frontal / closest training camera that sees the sample
        X_ = T(Pd); N_ = T(Nd); best = torch.full((len(Pd),), -1.0, device=dev, dtype=torch.float64)
        col = torch.zeros(len(Pd), 3, device=dev, dtype=torch.float64)
        for (name, W, H, fx, fy, cx, cy, Rcw, tcw, _), SD in zip(cams, sdep):
            Pc = X_ @ T(Rcw).T + T(tcw); z = Pc[:, 2]
            u = torch.floor(fx * Pc[:, 0] / z + cx).long(); vv = torch.floor(fy * Pc[:, 1] / z + cy).long()
            ok = (z > 0.05) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            if not ok.any():
                continue
            Cw = -T(Rcw).T @ T(tcw); dv = Cw[None] - X_; r = torch.linalg.norm(dv, dim=1); cs = (dv / r[:, None] * N_).sum(1)
            ia = torch.nonzero(ok & (cs > a.min_cos), as_tuple=True)[0]
            if not len(ia):
                continue
            ds_ = T(SD)[vv[ia], u[ia]]
            vis = (ds_ > 0) & (r[ia] < ds_ * (1 + a.tol_rel)) & (r[ia] > ds_ * (1 - 3 * a.tol_rel))
            ia = ia[vis]; sc_ = cs[ia] / r[ia]
            better = sc_ > best[ia]; ia, sc_ = ia[better], sc_[better]
            if not len(ia):
                continue
            if name not in imgs:
                imgs[name] = T(np.asarray(Image.open(os.path.join(a.dataset, "images", name)).convert("RGB"), np.float64) / 255)
            col[ia] = imgs[name][vv[ia], u[ia]]; best[ia] = sc_
        col = col.cpu().numpy(); seen = (best > 0).cpu().numpy()
        # tile unseen samples from the best-seen wall face (same height, along-face coordinate modulo its seen extent)
        tcoord = np.where(np.isin(Fs, (0, 1)), Ps[:, 1], Ps[:, 0]); hcoord = Ps[:, 2]
        walls = Fs < 10; fs = [(f, int((seen & (Fs == f)).sum())) for f in range(4)]; fsrc = max(fs, key=lambda x: x[1])[0]
        src = seen & (Fs == fsrc)
        if src.sum() > 50:
            ts, hs = tcoord[src], hcoord[src]; t0, t1 = np.percentile(ts, 2), np.percentile(ts, 98); Wd = max(t1 - t0, 4 * a.cell)
            tree = cKDTree(np.c_[ts, hs]); need = walls & ~seen
            q = np.c_[t0 + np.mod(tcoord[need] - t0, Wd), hcoord[need]]
            col[need] = col[src][tree.query(q)[1]]
        tops = Fs >= 10; srt = seen & tops
        if srt.sum() > 50:
            P2 = Ps[srt][:, :2]; lo2, hi2 = np.percentile(P2, 2, 0), np.percentile(P2, 98, 0); ext = np.maximum(hi2 - lo2, 4 * a.cell)
            tree = cKDTree(P2); need = tops & ~seen
            col[need] = col[srt][tree.query(lo2 + np.mod(Ps[need][:, :2] - lo2, ext))[1]]
        elif (tops & ~seen).any():
            col[tops & ~seen] = np.median(col[seen], 0) if seen.any() else 0.5
        # rows: keep the well-seen base rows in the box, delete the rest above the floor, discs where nothing is kept
        keep_rows = inb & surfel & (op > 0.5) if a.keep_seen else np.zeros(N0, bool)
        d_ = inb & ~keep_rows; dele |= d_
        far = cKDTree(Gm[keep_rows]).query(Pm)[0] > a.keep_m if keep_rows.any() else np.ones(len(Pm), bool)
        add_X.append(Pd[far]); add_N.append(Nd[far]); add_C.append(col[far])
        rep[o["id"]] = {"levels": pr["levels"], "half_m": ob.half.round(3).tolist(), "samples": len(Ps), "seen": int(seen.sum()),
                        "source_wall_face": int(fsrc), "deleted": int(d_.sum()), "kept": int(keep_rows.sum()), "added": int(far.sum())}
        print(f"[brickbox] object {o['id']}: {rep[o['id']]}", flush=True)
    vtx, p0 = load_ply(a.ply, dev); K = p0["shN"].shape[1]
    Xa, Na, Ca = np.concatenate(add_X), np.concatenate(add_N), np.concatenate(add_C); M = len(Xa)
    sc_in = np.log(0.7 * a.cell / s); sc_n = np.log(0.15 * a.cell / s)
    newp = {"means": Xa, "scales": np.tile([sc_in, sc_in, sc_n], (M, 1)), "quats": quat_from_z(Na / np.linalg.norm(Na, axis=1, keepdims=True)),
            "opacities": np.full(M, np.log(a.opacity / (1 - a.opacity))), "sh0": ((np.clip(Ca, 0, 1) - 0.5) / C0)[:, None], "shN": np.zeros((M, K, 3))}
    kb = torch.tensor(~dele, device=dev)
    P = {k: torch.cat([p0[k][kb], torch.tensor(newp[k], dtype=torch.float32, device=dev)]) for k in p0}
    save_ply(a.out, np.concatenate([vtx.data[~dele], np.zeros(M, vtx.data.dtype)]), P)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=np.r_[np.nonzero(~dele)[0], -np.ones(M, int)])
    json.dump({"objects": rep, "rows": int((~dele).sum()) + M, "args": vars(a)}, open(os.path.splitext(a.out)[0] + "_mvrefit.json", "w"), indent=1)
    print("MV_BRICKBOX", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
