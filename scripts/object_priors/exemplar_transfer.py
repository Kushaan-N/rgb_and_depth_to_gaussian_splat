"""Exemplar transfer for the unseen sides of free-standing objects (gsplat venv, GPU; cosmetic): a poorly covered object
borrows the REAL gaussians of a well-covered similar object in the same scene (or its own best-seen side, mirrored).

  coverage  per object: azimuth bins (--bins, object frame) holding a training camera that has the object's centre in view
            (within --cam-max-m, not hidden behind the base's rendered depth); objects with >= --min-cover of the bins are
            exemplar candidates (real backs).
  match     per target (--ids) x exemplar (every covered object, + the target itself mirrored = self-exemplar): the
            exemplar's rows (object frame) mapped by yaw {0,90,180,270} x mirror {no, u} x per-axis scale (box halves /
            heights, after the yaw) onto the target's box, refined by a planar ICP (yaw + translation) against the target's
            well-seen surfels; score = fit (share of the target's seen surfels within --fit-m of the mapped exemplar)
            x (0.3 + 0.7 coverage gain: share of the target's unseen bins the exemplar's seen rows fill) x colour similarity
            (mean SH-DC colour of the seen rows); the self-exemplar (identity excluded) competes on the same score.
  transplant the target keeps its well-seen rows (front faces), the rest of its box above the floor is deleted; the mapped
            exemplar rows that the exemplar's own cameras saw (weight >= --ex-w) come in where no kept row lies within
            --front-m; covariances mapped by the full linear map (rotation / mirror / scale), SH bands > 0 dropped; DC colour
            recoloured per channel (mean / std of the mapped rows near the target's seen rows -> those rows).
  refit     hybrid v2 schedule on the real frames that see the targets: kept rows full params, transplanted rows colour +
            opacity for --warm iters, then everything with geometry steps x --new-lr-k.

    python scripts/object_priors/exemplar_transfer.py --ply base.ply --support D/support.npz --objects D/objects \
        --sim3-json sim3.json --dataset <train> <test> --ids 3 4 5 --out T/refit.ply
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from solidify_unseen import sim3, gauss_geom, load_objects, read_vertex, all_frames  # noqa: E402
from object_priors import Obj  # noqa: E402

C0 = 0.28209479177387814


def rotz(th):
    c, s_ = np.cos(th), np.sin(th); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--support", required=True); ap.add_argument("--objects", required=True)
    ap.add_argument("--sim3-json", required=True); ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=[]); ap.add_argument("--force", nargs="*", default=[],
                    help="target:exemplar pairs (e.g. 5:1) used instead of the search")
    ap.add_argument("--ring-ids", type=int, nargs="*", default=[], help="clean a ring behind these objects: unconstrained haze / needles pruned, "
                    "floor discs where no opaque floor row is left")
    ap.add_argument("--del-w", type=float, default=0.02, help="ring: rows below this support weight count as unconstrained")
    ap.add_argument("--ring-m", type=float, default=0.5); ap.add_argument("--ring-cell", type=float, default=0.015)
    ap.add_argument("--graze-ids", type=int, nargs="*", default=[], help="drop unconstrained rows of these objects that the training "
                    "cameras see only edge-on (max |normal . view| < --graze-cos)")
    ap.add_argument("--graze-cos", type=float, default=0.3); ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="objects never used as exemplars (e.g. a different kind of object)")
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--ex-w", type=float, default=0.05)
    ap.add_argument("--splat-depth-dir", default=None); ap.add_argument("--cam-max-m", type=float, default=4.0)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--min-cover", type=float, default=0.6)
    ap.add_argument("--fit-m", type=float, default=0.015); ap.add_argument("--front-m", type=float, default=0.01)
    ap.add_argument("--zone-m", type=float, default=0.02); ap.add_argument("--floor-m", type=float, default=0.01)
    ap.add_argument("--ev-min-h", type=float, default=0.03); ap.add_argument("--no-self", action="store_true")
    ap.add_argument("--iters", type=int, default=1200); ap.add_argument("--warm", type=int, default=400)
    ap.add_argument("--new-lr-k", type=float, default=0.3); ap.add_argument("--display-scale", type=float, default=1.8)
    a = ap.parse_args()
    import torch
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, save_ply, sh_deg, to_t, quat_to_rot
    from refine_poses_gsplat import load_rgb, ssim, psnr
    torch.manual_seed(0); np.random.seed(0); dev = "cuda"; t0 = time.time()
    s, R, t = sim3(a.sim3_json)
    v = read_vertex(a.ply); Gm, op, sc = gauss_geom(v, s, R, t); N0 = len(v); ss = np.sort(sc, 1)
    wmax = np.load(a.support)["wmax"]; assert len(wmax) == N0, "support was computed for another PLY"
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64); col = np.clip(0.5 + C0 * dc, 0, 1)
    seen = (wmax >= a.face_w) & (op > 0.5)
    surfel = seen & (ss[:, 0] < 0.3 * ss[:, 1]) & (ss[:, 2] < 3 * ss[:, 1])
    from covis_masks import cams_of
    cams = cams_of(a.dataset[0])
    sdep = [np.load(os.path.join(a.splat_depth_dir, os.path.splitext(c[0])[0] + "_splatdepth.npy")).astype(np.float32)
            if a.splat_depth_dir else None for c in cams]
    ometa, opts = load_objects(a.objects); n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    # ---------------------------------------------------------------- per-object frames, boxes, coverage
    O = {}
    for o, X in zip(ometa["objects"], opts):
        ob = Obj(X, n_f, d_f); Lg = ob.loc(Gm); zh = ob.half + 0.12
        z0 = (np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > a.ev_min_h)
        if (z0 & surfel).sum() > 100:
            ob = Obj(np.concatenate([X[::10], Gm[z0 & surfel]]), n_f, d_f, min_h=a.ev_min_h)
        Lg = ob.loc(Gm); hmax = float(np.percentile(ob.loc(X)[:, 2], 99))
        inb = (np.abs(Lg[:, 0]) <= ob.half[0] + a.zone_m) & (np.abs(Lg[:, 1]) <= ob.half[1] + a.zone_m) & (Lg[:, 2] > a.floor_m) & (Lg[:, 2] < hmax + 0.05)
        # camera coverage: azimuth bins (object frame) holding a training camera that has the object's centre in view
        # within --cam-max-m and in front of the base's rendered depth there (no other surface in between)
        cm_ = ob.c0 + ob.n * 0.5 * hmax; cdd = ((cm_ - t) @ R) / s; cov = np.zeros(a.bins, bool)
        rob = float(np.linalg.norm(np.r_[ob.half, 0.5 * hmax]))
        for (nm_, W, H, fx, fy, cx, cy, Rcw, tcw, _), SD in zip(cams, sdep):
            pc = Rcw @ cdd + tcw
            if pc[2] < 0.05:
                continue
            u = int(fx * pc[0] / pc[2] + cx); vv = int(fy * pc[1] / pc[2] + cy)
            if not (0 <= u < W and 0 <= vv < H) or np.linalg.norm(pc) * s > a.cam_max_m:
                continue
            if SD is not None and SD[vv, u] > 0 and (np.linalg.norm(pc) - SD[vv, u]) * s > rob + 0.1:
                continue                                                    # something else in front of the object
            Cm = s * ((-Rcw.T @ tcw) @ R.T) + t; lc = Cm - cm_
            ang = np.arctan2(lc @ ob.a2, lc @ ob.a1); cov[int(np.floor((ang + np.pi) / (2 * np.pi) * a.bins)) % a.bins] = True
        O[o["id"]] = {"ob": ob, "L": Lg, "inb": inb, "hmax": hmax, "cover": cov,
                      "col": col[inb & seen].mean(0) if (inb & seen).any() else np.full(3, 0.5)}
    cover = {k: round(float(v_["cover"].mean()), 3) for k, v_ in O.items()}
    print(f"[exemplar] seen-surfel azimuth coverage per object: {cover}", flush=True)
    exemplars = [k for k in O if cover[k] >= a.min_cover and k not in a.exclude]
    # ---------------------------------------------------------------- match + transplant
    dele = np.zeros(N0, bool); keep_front = np.zeros(N0, bool); new = {k: [] for k in ("X", "cov", "dc", "op")}; rep = {}
    Fm = lambda ob: np.stack([ob.a1, ob.a2, ob.n], 1)                       # local -> metric directions
    for tid in a.ids:
        T = O[tid]; obT = T["ob"]; LT = T["L"]; sT = T["inb"] & surfel
        if sT.sum() < 50:
            print(f"[exemplar] target {tid}: too few seen surfels", flush=True); continue
        treeT = cKDTree(LT[sT]); best = None
        force = dict(tuple(int(x) for x in f.split(":")) for f in a.force)
        cands = [(force[tid], False)] if tid in force else \
            [(e, False) for e in exemplars if e != tid] + ([] if a.no_self else [(tid, True)])
        for eid, is_self in cands:
            E = O[eid]; rowsE = E["inb"] & (wmax >= a.ex_w) & (op > 0.3); LE = E["L"][rowsE]
            if len(LE) < 50:
                continue
            csim = float(np.exp(-4 * np.linalg.norm(E["col"] - T["col"])))
            for k90 in range(4):
                for mir in (False, True):
                    if is_self and k90 == 0 and not mir:
                        continue                                            # identity
                    Mm = np.diag([-1.0 if mir else 1.0, 1, 1]); Rz = rotz(k90 * np.pi / 2)
                    he = np.abs(Rz[:2, :2]) @ E["ob"].half                    # exemplar halves after the yaw
                    S = np.diag([obT.half[0] / max(he[0], 1e-3), obT.half[1] / max(he[1], 1e-3), T["hmax"] / max(E["hmax"], 1e-3)])
                    Ml = S @ Rz @ Mm; P = LE @ Ml.T; th, tr2 = 0.0, np.zeros(2)
                    for _ in range(10):                                     # planar ICP on the target's seen surfels
                        Q = P.copy(); Q[:, :2] = Q[:, :2] @ rotz(th)[:2, :2].T + tr2
                        d, i = treeT.query(Q, distance_upper_bound=0.04)
                        ok = np.isfinite(d)
                        if ok.sum() < 30:
                            break
                        A_ = Q[ok, :2]; B_ = LT[sT][i[ok], :2]; ma, mb = A_.mean(0), B_.mean(0)
                        U, _, Vt = np.linalg.svd((A_ - ma).T @ (B_ - mb)); Rr = (U @ Vt).T
                        if np.linalg.det(Rr) < 0:
                            Vt[-1] *= -1; Rr = (U @ Vt).T
                        dth = np.arctan2(Rr[1, 0], Rr[0, 0])
                        if abs(dth) > np.radians(10) or np.linalg.norm(mb - Rr @ ma) > 0.05:
                            break
                        th += dth; tr2 = Rr @ tr2 + (mb - Rr @ ma)
                    Q = P.copy(); Q[:, :2] = Q[:, :2] @ rotz(th)[:2, :2].T + tr2
                    fit = float((cKDTree(Q).query(LT[sT])[0] < a.fit_m).mean())
                    # the exemplar's camera-covered directions, mapped into the target's frame (rotation / mirror part)
                    bc = (np.nonzero(E["cover"])[0] + 0.5) / a.bins * 2 * np.pi - np.pi
                    dvec = np.c_[np.cos(bc), np.sin(bc), np.zeros(len(bc))] @ (rotz(th) @ Rz @ Mm).T
                    bq = np.floor((np.arctan2(dvec[:, 1], dvec[:, 0]) + np.pi) / (2 * np.pi) * a.bins).astype(int) % a.bins
                    gain_bins = np.zeros(a.bins, bool); gain_bins[bq] = True
                    unseen = ~T["cover"]; gain = float((gain_bins & unseen).sum() / max(unseen.sum(), 1))
                    score = fit * (0.3 + 0.7 * gain) * csim
                    if best is None or score > best["score"]:
                        best = {"score": score, "fit": fit, "gain": gain, "csim": csim, "eid": eid, "self": is_self, "k90": k90, "mir": mir,
                                "Ml": rotz(th) @ Ml, "tr": np.r_[tr2, 0.0], "rowsE": rowsE}
        if best is None:
            print(f"[exemplar] target {tid}: no exemplar", flush=True); continue
        E = O[best["eid"]]; rowsE = np.nonzero(best["rowsE"])[0]
        # keep the target's well-seen rows, delete the rest of its box
        kf = T["inb"] & seen; keep_front |= kf; dele |= T["inb"] & ~kf
        # map exemplar rows: metric X_T = c0T + F_T (Ml L_E + tr), L_E = F_E^T (X - c0E) + (0, 0, d-offset)
        LE = E["L"][rowsE]; LQ = LE @ best["Ml"].T + best["tr"]
        XmQ = obT.met(LQ)
        A_m = Fm(obT) @ best["Ml"] @ Fm(E["ob"]).T                          # metric linear map
        A_d = R.T @ A_m @ R                                                 # same map in the PLY frame (similarity-invariant)
        dfr = cKDTree(Gm[kf]).query(XmQ)[0] if kf.any() else np.full(len(XmQ), 1e9)
        inT = (np.abs(LQ[:, 0]) <= obT.half[0] + a.zone_m) & (np.abs(LQ[:, 1]) <= obT.half[1] + a.zone_m) & (LQ[:, 2] > a.floor_m)
        put = (dfr > a.front_m) & inT
        rows = rowsE[put]
        # covariance in the PLY frame
        q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1)[rows].astype(np.float64)
        Rg = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
        Sg = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)[rows].astype(np.float64))
        Cv = Rg @ (Sg[:, :, None] ** 2 * np.transpose(Rg, (0, 2, 1)))
        Cv = A_d[None] @ Cv @ A_d.T[None]
        # recolour: mapped rows near the target's seen rows -> those rows (per-channel mean / std)
        XQd = ((XmQ[put] - t) @ R) / s
        dn, inn = cKDTree(Gm[kf]).query(XmQ[put], distance_upper_bound=0.03) if kf.any() else (np.full(len(rows), np.inf), None)
        cE = col[rows]; ref = np.isfinite(dn)
        if ref.sum() > 30:
            src_c, dst_c = cE[ref], col[kf][inn[ref]]
        else:
            src_c, dst_c = col[E["inb"] & seen], col[T["inb"] & seen]
        g_ = dst_c.std(0) / np.maximum(src_c.std(0), 1e-3); g_ = np.clip(g_, 0.5, 2.0)
        cN = np.clip((cE - src_c.mean(0)) * g_ + dst_c.mean(0), 0, 1)
        new["X"].append(XQd); new["cov"].append(Cv); new["dc"].append((cN - 0.5) / C0); new["op"].append(np.asarray(v["opacity"])[rows])
        rep[tid] = {"exemplar": int(best["eid"]), "self": bool(best["self"]), "yaw_deg": 90 * best["k90"], "mirror": bool(best["mir"]),
                    "score": round(best["score"], 3), "fit": round(best["fit"], 3), "coverage_gain": round(best["gain"], 3),
                    "colour_sim": round(best["csim"], 3), "kept_front": int(kf.sum()), "deleted": int((T["inb"] & ~kf).sum()),
                    "transplanted": int(len(rows)), "recolour_gain": g_.round(2).tolist()}
        print(f"[exemplar] target {tid}: {rep[tid]}", flush=True)
    # ---------------------------------------------------------------- grazing rows (white streaks on oblique faces)
    qv = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], 1).astype(np.float64)
    lsc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)
    Xd_all = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    for gid in a.graze_ids:
        G_ = O[gid]; rows = np.nonzero(G_["inb"] & ~dele & (wmax < a.face_w))[0]
        if not len(rows):
            continue
        Rg = Rotation.from_quat(qv[rows] / np.linalg.norm(qv[rows], axis=1, keepdims=True)).as_matrix()
        nrm = Rg[np.arange(len(rows)), :, np.argmin(lsc[rows], 1)]; Xr = Xd_all[rows]
        best_c = np.zeros(len(rows)); shown = np.zeros(len(rows), bool)
        for (nm_, W, H, fx, fy, cx, cy, Rcw, tcw, _), SD in zip(cams, sdep):
            Pc = Xr @ Rcw.T + tcw; z = Pc[:, 2]; ok = z > 0.05
            u = np.floor(fx * Pc[:, 0] / np.where(ok, z, 1) + cx).astype(int); vv = np.floor(fy * Pc[:, 1] / np.where(ok, z, 1) + cy).astype(int)
            ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
            if SD is not None:
                r = np.linalg.norm(Pc, axis=1); ok[ok] &= (SD[vv[ok], u[ok]] <= 0) | (r[ok] < SD[vv[ok], u[ok]] * 1.02)
            if not ok.any():
                continue
            C = -Rcw.T @ tcw; dv = C[None] - Xr[ok]; dv /= np.linalg.norm(dv, axis=1, keepdims=True)
            best_c[ok] = np.maximum(best_c[ok], np.abs((dv * nrm[ok]).sum(1))); shown[ok] = True
        drop = shown & (best_c < a.graze_cos); dele[rows[drop]] = True
        rep.setdefault(f"graze_{gid}", {})["dropped"] = int(drop.sum())
        print(f"[exemplar] object {gid}: {int(drop.sum())} unconstrained rows seen only edge-on dropped (of {len(rows)})", flush=True)
    # ---------------------------------------------------------------- ring behind each object: haze prune + floor fill
    boxes = [(k, O[k]) for k in O]
    for rid in a.ring_ids:
        Rr = O[rid]; ob = Rr["ob"]; L = Rr["L"]
        dbox = np.maximum(np.abs(L[:, 0]) - ob.half[0], np.abs(L[:, 1]) - ob.half[1])
        other = np.zeros(N0, bool)
        for k, Ok in boxes:
            if k != rid:
                Lk = Ok["L"]; other |= (np.abs(Lk[:, 0]) <= Ok["ob"].half[0] + 0.03) & (np.abs(Lk[:, 1]) <= Ok["ob"].half[1] + 0.03)
        ring = (dbox > 0.0) & (dbox <= a.ring_m) & (L[:, 2] < 0.6) & (L[:, 2] > -0.05) & ~other & ~dele
        unc = ring & (wmax < a.del_w)
        haze = unc & ((op < 0.3) | (ss[:, 2] > 4 * ss[:, 1]) | (L[:, 2] > 0.03))
        dele |= haze
        floor_ok = ring & ~dele & (op > 0.5) & (np.abs(L[:, 2]) < 0.02)
        g_ = np.arange(-ob.half[0] - a.ring_m, ob.half[0] + a.ring_m, a.ring_cell); h_ = np.arange(-ob.half[1] - a.ring_m, ob.half[1] + a.ring_m, a.ring_cell)
        U, Vv = np.meshgrid(g_, h_, indexing="ij"); Q = np.c_[U.ravel(), Vv.ravel()]
        dq = np.maximum(np.abs(Q[:, 0]) - ob.half[0], np.abs(Q[:, 1]) - ob.half[1]); Q = Q[(dq > 0.01) & (dq <= a.ring_m)]
        for k, Ok in boxes:                                                 # not under another object
            if k != rid:
                Lq = Ok["ob"].loc(ob.met(np.c_[Q, np.zeros(len(Q))]))
                Q = Q[~((np.abs(Lq[:, 0]) <= Ok["ob"].half[0] + 0.03) & (np.abs(Lq[:, 1]) <= Ok["ob"].half[1] + 0.03))]
        has = cKDTree(L[floor_ok][:, :2]).query(Q)[0] < a.ring_cell if floor_ok.any() else np.zeros(len(Q), bool)
        Qf = Q[~has]
        seenfloor = (np.abs(L[:, 2]) < 0.02) & (wmax >= a.face_w) & (op > 0.5) & (dbox <= a.ring_m + 0.3) & (dbox > 0)
        if len(Qf) and seenfloor.sum() > 20:
            sf = cKDTree(L[seenfloor][:, :2]); _, nn = sf.query(Qf, k=min(8, int(seenfloor.sum())))
            cf = np.median(col[seenfloor][nn], 1)
            Xm_f = ob.met(np.c_[Qf, np.full(len(Qf), 0.003)]); Xd_f = ((Xm_f - t) @ R) / s
            Rm = R.T @ np.stack([ob.a1, ob.a2, ob.n], 1); sd_ = 0.6 * a.ring_cell / s
            Cf = Rm @ np.diag([sd_ ** 2, sd_ ** 2, (0.1 * sd_) ** 2]) @ Rm.T
            new["X"].append(Xd_f); new["cov"].append(np.repeat(Cf[None], len(Qf), 0)); new["dc"].append((cf - 0.5) / C0)
            new["op"].append(np.full(len(Qf), np.log(0.95 / 0.05)))
        rep[f"ring_{rid}"] = {"haze_pruned": int(haze.sum()), "floor_discs": int(len(Qf))}
        print(f"[exemplar] ring behind object {rid}: {rep[f'ring_{rid}']}", flush=True)
    # ---------------------------------------------------------------- assemble
    vtx, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); K = p0["shN"].shape[1]
    Xn = np.concatenate(new["X"]) if new["X"] else np.zeros((0, 3)); Cn = np.concatenate(new["cov"]) if new["cov"] else np.zeros((0, 3, 3)); M = len(Xn)
    w_, V_ = np.linalg.eigh(Cn); w_ = np.maximum(w_, 1e-12); det = np.linalg.det(V_); V_[det < 0, :, 0] *= -1
    qn = Rotation.from_matrix(V_).as_quat(); qn = np.c_[qn[:, 3], qn[:, :3]]
    newp = {"means": Xn, "scales": 0.5 * np.log(w_), "quats": qn, "opacities": np.concatenate(new["op"]),
            "sh0": (np.concatenate(new["dc"]) if new["dc"] else np.zeros((0, 3)))[:, None], "shN": np.zeros((M, K, 3))}
    if not new["op"]:
        newp["opacities"] = np.zeros(0)
    kb = torch.tensor(~dele, device=dev)
    P = {k: torch.cat([p0[k][kb], torch.tensor(newp[k], dtype=torch.float32, device=dev)]) for k in p0}
    nb = int(kb.sum()); src = np.r_[np.nonzero(~dele)[0], -np.ones(M, int)]
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(x.clone()) for k, x in P.items()})
    # ---------------------------------------------------------------- refit on the real frames that see the targets
    newm = torch.zeros(len(src), dtype=torch.bool, device=dev); newm[nb:] = True
    frontm = torch.zeros(len(src), dtype=torch.bool, device=dev); frontm[:nb] = torch.tensor(keep_front[~dele], device=dev); trm = frontm | newm
    tr = all_frames(a.dataset); ks_ = sorted(set(a.ids) | set(a.ring_ids) | set(a.graze_ids)); cen = np.stack([O[k]["ob"].c0 for k in ks_]) if ks_ else np.zeros((0, 3)); cd = ((cen - t) @ R) / s
    fr = []
    for i, (n_, cam, c2w) in enumerate(tr):
        fx, fy, cx, cy, W, H = cam_args(cam); w2c = np.linalg.inv(c2w); Pc = cd @ w2c[:3, :3].T + w2c[:3, 3]
        u = fx * Pc[:, 0] / Pc[:, 2] + cx; vv = fy * Pc[:, 1] / Pc[:, 2] + cy
        if np.any((Pc[:, 2] > 0.05) & (u > 0) & (u < W) & (vv > 0) & (vv < H)):
            fr.append(i)
    gtf = lambda n_: next(os.path.join(ds, "images", n_) for ds in a.dataset if os.path.exists(os.path.join(ds, "images", n_)))
    sc_ = float(np.linalg.norm(np.stack([c[:3, 3] for _, _, c in tr]) - np.stack([c[:3, 3] for _, _, c in tr]).mean(0), axis=1).max()) * 1.1
    lrs = {"means": 1.6e-5 * sc_, "scales": 5e-3, "quats": 1e-3, "opacities": 0.025, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k} for k in params], eps=1e-15)
    Rv = ViewerRaster("ss352", "exact", display_scale=a.display_scale); geo = ("means", "scales", "quats", "shN")
    for step in range(1, a.iters + 1 if fr else 1):
        i = fr[np.random.randint(len(fr))]; n_, cam, c2w = tr[i]
        img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
        gt = load_rgb(gtf(n_), dev).float() / 255
        l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(set_to_none=True); l.backward()
        for k in params:
            g = params[k].grad
            if g is None:
                continue
            g[~trm] = 0; g[~torch.isfinite(g)] = 0
            if step <= a.warm and k in geo:
                g[newm] = 0
        old = {k: params[k].detach()[newm].clone() for k in geo}
        opt.step()
        if step > a.warm:
            with torch.no_grad():
                for k in geo:
                    params[k][newm] = old[k] + a.new_lr_k * (params[k][newm] - old[k])
    with torch.no_grad():
        pv, p0v = [], []
        for i in fr[::max(1, len(fr) // 30)]:
            n_, cam, c2w = tr[i]; gt = load_rgb(gtf(n_), dev).float() / 255; w2c = to_t(np.linalg.inv(c2w), dev)
            pv.append(psnr(Rv.render(params, w2c, *cam_args(cam), sh_degree=deg)[0], gt)); p0v.append(psnr(Rv.render(p0, w2c, *cam_args(cam), sh_degree=deg)[0], gt))
    save_ply(a.out, np.concatenate([vtx.data[~dele], np.zeros(M, vtx.data.dtype)]), params)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=src)
    out = {"coverage": cover, "exemplars": exemplars, "targets": rep, "rows": len(src), "new": M, "deleted": int(dele.sum()),
           "refit_frames": len(fr), "frames_psnr_before": round(float(np.mean(p0v)), 3) if p0v else None,
           "frames_psnr_after": round(float(np.mean(pv)), 3) if pv else None, "seconds": round(time.time() - t0), "args": vars(a)}
    json.dump(out, open(os.path.splitext(a.out)[0] + "_mvrefit.json", "w"), indent=1)
    print("EXEMPLAR", json.dumps({k: v_ for k, v_ in out.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
