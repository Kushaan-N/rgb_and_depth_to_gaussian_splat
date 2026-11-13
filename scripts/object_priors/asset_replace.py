"""Full generated-asset replacement of one-sided free-standing objects (gsplat venv, GPU; cosmetic: every side of the
object is the image-to-3D model's asset, not only the unseen hemisphere).

Targets are chosen automatically: every object of find_objects.py whose BACK COVERAGE (share of --bins azimuth bins around
it holding a training camera that has it in view within --cam-max-m, unoccluded) is below --min-back-coverage and that has
a generated asset (--ids only as a debug override). Per object, an automatic ACCEPT TEST decides: registration (trimmed
evidence residual <= --max-resid-m, share of the asset outside the object's box <= --max-outside) and on-path cost (mean
PSNR drop, on the real frames that see the object, of the scene with ONLY this object replaced <= --max-onpath-loss-db);
a rejected object keeps its original gaussians. No per-object or per-scene constants.

Per target: the generated assets <gen-dir>/obj<id>_s<seed>.npz (learned_prior.py: TRELLIS multi-image gaussians in
a unit cube, +z up) are registered to the object's evidence (well-seen splat surfels + a LiDAR subsample) by
object_priors.align_generated (yaw scan x scale, trimmed ICP with yaw + scale + translation; lowest residual of the
seeds). Default (--hybrid 1): the base's well-seen front gaussians stay, the rest of the object's tight box above the floor
is deleted, and the asset's gaussians (made opaque, DC colour matched to the kept front's statistics) come in only where no
training camera sees them at / in front of the base surface: the asset is the unseen side and the interior shell.
--hybrid 0: every base row in the box is deleted and the whole asset inserted.
Harmonisation: colour only (SH-DC of the asset rows; geometry frozen) on the real training frames that see the object,
viewer-exact L1+SSIM, --harm-iters.

    python scripts/object_priors/asset_replace.py --ply base.ply --support D/support.npz --objects D/objects \
        --gen-dir <learned_gen> --sim3-json sim3.json --dataset <train> <test> --ids 3 4 5 --out T/refit.ply
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from solidify_unseen import sim3, gauss_geom, load_objects, read_vertex, all_frames  # noqa: E402
from object_priors import Obj, align_generated  # noqa: E402

C0 = 0.28209479177387814


def box_align(files, Lev, cev, hmax, ob, a):
    """asset (unit cube, +z up) -> object-local metric pose constrained by the object's box: uniform scale from the height
    (x 0.9..1.1), yaw scan (5 deg), footprint centred on the box then a translation-only ICP; score = trimmed mean distance
    evidence -> asset + 0.05 x share of the asset outside the box (+2 cm). Lowest score over the seeds."""
    from scipy.spatial import cKDTree
    from object_priors import quat_wxyz_to_R
    rng = np.random.default_rng(0); E = Lev[rng.choice(len(Lev), min(3000, len(Lev)), replace=False)]; best = None; cands = []
    for f in files:
        g = np.load(f); X = g["xyz"].astype(np.float64); G = X[g["opacity"] > a.min_op]
        if len(G) < 200:
            continue                                                        # empty / degenerate asset
        G = G[rng.choice(len(G), min(6000, len(G)), replace=False)]
        z0, z1 = np.percentile(G[:, 2], [0.5, 99.5]); off = np.r_[(np.percentile(G[:, :2], 1, 0) + np.percentile(G[:, :2], 99, 0)) / 2, z0]
        fbest = None
        for yaw in np.radians(np.arange(0, 360, 5)):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
            for sm in (0.9, 1.0, 1.1):
                sc = sm * hmax / max(z1 - z0, 1e-3); Gt = sc * (G - off) @ Rz.T; tt = np.zeros(3)
                for _ in range(5):                                          # translation only (xy)
                    d, i = cKDTree(Gt + tt).query(E); k = d <= np.quantile(d, 0.8)
                    dt = (E[k] - (Gt + tt)[i[k]]).mean(0); dt[2] = 0; tt += np.clip(dt, -0.03, 0.03)
                Gt = Gt + tt; d = cKDTree(Gt).query(E)[0]; res = float(np.sort(d)[:int(0.8 * len(d))].mean())
                out = float(((np.abs(Gt[:, 0]) > ob.half[0] + 0.02) | (np.abs(Gt[:, 1]) > ob.half[1] + 0.02) | (Gt[:, 2] > hmax + 0.02)).mean())
                score = res + 0.05 * out
                if fbest is None or score < fbest[0]:
                    fbest = (score, f, Rz, sc, tt.copy(), off, res, out)
        cands.append({"file": os.path.basename(f), "score": round(fbest[0], 4), "resid_m": round(fbest[6], 4), "outside_box": round(fbest[7], 3),
                      "scale_m_per_unit": round(fbest[3], 3)})
        if best is None or fbest[0] < best[0]:
            best = fbest
    if best is None:
        return None
    _, f, Rz, sc, tt, off, _, _ = best; g = np.load(f); X = g["xyz"].astype(np.float64)
    L = sc * (X - off) @ Rz.T + tt; Rm = np.stack([ob.a1, ob.a2, ob.n], 1) @ Rz
    dc = g["dc"].astype(np.float64); d, i = cKDTree(Lev).query(L); m = (d < 0.015) & (g["opacity"] > a.min_op); cal = None
    if m.sum() > 50:                                                         # colour: mean / std match on the seen part
        rg, re_ = 0.5 + C0 * dc[m], 0.5 + C0 * cev[i[m]]
        gn = np.clip(re_.std(0) / np.maximum(rg.std(0), 1e-3), 0.7, 1.4); cal = [gn.round(3).tolist(), re_.mean(0).round(3).tolist()]
        dc = (np.clip((0.5 + C0 * dc - rg.mean(0)) * gn + re_.mean(0), 0, 1) - 0.5) / C0
    return {"xyz": ob.met(L), "rot": Rm @ quat_wxyz_to_R(g["rot"]), "scale": g["scale"] * sc, "opacity": g["opacity"], "dc": dc,
            "report": {"chosen": os.path.basename(f), "candidates": cands, "colour_gain_mean": cal, "calib_pairs": int(m.sum())}}


def back_coverage(ob, hmax, cams, sdep, s, R, t, bins, cam_max_m):
    cm_ = ob.c0 + ob.n * 0.5 * hmax; cdd = ((cm_ - t) @ R) / s; cov = np.zeros(bins, bool); rob = float(np.linalg.norm(np.r_[ob.half, 0.5 * hmax])); fr = []
    for i, ((nm_, W, H, fx, fy, cx, cy, Rcw, tcw, _), SD) in enumerate(zip(cams, sdep)):
        pc = Rcw @ cdd + tcw
        if pc[2] < 0.05:
            continue
        u = int(fx * pc[0] / pc[2] + cx); vv = int(fy * pc[1] / pc[2] + cy); r = np.linalg.norm(pc)
        if not (0 <= u < W and 0 <= vv < H) or r * s > cam_max_m:
            continue
        if SD is not None and SD[vv, u] > 0 and (r - SD[vv, u]) * s > rob + 0.1:
            continue
        lc = s * ((-Rcw.T @ tcw) @ R.T) + t - cm_
        cov[int(np.floor((np.arctan2(lc @ ob.a2, lc @ ob.a1) + np.pi) / (2 * np.pi) * bins)) % bins] = True; fr.append(nm_)
    return float(cov.mean()), fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--support", required=True); ap.add_argument("--objects", required=True)
    ap.add_argument("--gen-dir", required=True); ap.add_argument("--sim3-json", required=True); ap.add_argument("--splat-depth-dir", default=None)
    ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="debug override of the automatic target selection")
    ap.add_argument("--min-back-coverage", type=float, default=0.5); ap.add_argument("--max-onpath-loss-db", type=float, default=0.3)
    ap.add_argument("--hybrid", type=int, default=1, help="1: keep the base gaussians the training views constrain (support weight >= --keep-w) and insert "
                    "asset gaussians only where no training camera sees them at / in front of the base surface (within --front-m); colour = the "
                    "kept front's statistics (no optimisation). 0: wholesale replacement + colour harmonisation on real frames")
    ap.add_argument("--keep-w", type=float, default=0.02, help="hybrid: base rows with training-view support weight >= this stay (below it no "
                    "training view can tell the row is there, so deleting it costs nothing on-path)")
    ap.add_argument("--front-m", type=float, default=0.015); ap.add_argument("--max-scale-m", type=float, default=0.02)
    ap.add_argument("--max-resid-m", type=float, default=0.02); ap.add_argument("--max-outside", type=float, default=0.15)
    ap.add_argument("--bins", type=int, default=16); ap.add_argument("--cam-max-m", type=float, default=4.0)
    ap.add_argument("--face-w", type=float, default=0.2); ap.add_argument("--ev-min-h", type=float, default=0.03)
    ap.add_argument("--zone-m", type=float, default=0.02); ap.add_argument("--floor-m", type=float, default=0.01)
    ap.add_argument("--min-op", type=float, default=0.3); ap.add_argument("--opaque", type=float, default=0.98)
    ap.add_argument("--align", default="box", choices=["box", "icp"], help="box: scale / position constrained by the object's box "
                    "(box_align); icp: object_priors.align_generated (free scale: ambiguous on box-like objects)")
    ap.add_argument("--harm-iters", type=int, default=800); ap.add_argument("--display-scale", type=float, default=1.8)
    a = ap.parse_args()
    import torch
    from scipy.spatial.transform import Rotation
    from covis_masks import cams_of
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, save_ply, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, ssim, psnr
    dev = "cuda"; t0 = time.time(); torch.manual_seed(0); np.random.seed(0)
    s, R, t = sim3(a.sim3_json)
    v = read_vertex(a.ply); Gm, op, sc = gauss_geom(v, s, R, t); N0 = len(v); ss = np.sort(sc, 1)
    wmax = np.load(a.support)["wmax"]; assert len(wmax) == N0, "support was computed for another PLY"
    dcb = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
    surfel = (wmax >= a.face_w) & (op > 0.7) & (ss[:, 0] < 0.3 * ss[:, 1]) & (ss[:, 2] < 3 * ss[:, 1])
    ometa, opts = load_objects(a.objects); n_f = np.array(ometa["floor"]["n"]); d_f = float(ometa["floor"]["d"])
    cams = cams_of(a.dataset[0])
    sdep = [np.load(os.path.join(a.splat_depth_dir, os.path.splitext(c[0])[0] + "_splatdepth.npy")).astype(np.float32) if a.splat_depth_dir else None for c in cams]
    rep, cand = {}, {}
    for o, X in zip(ometa["objects"], opts):
        oid = o["id"]; r = rep[oid] = {"decision": None}
        ob = Obj(X, n_f, d_f); Lg = ob.loc(Gm); zh = ob.half + 0.12
        z0 = (np.abs(Lg[:, 0]) <= zh[0]) & (np.abs(Lg[:, 1]) <= zh[1]) & (Lg[:, 2] > a.ev_min_h)
        if (z0 & surfel).sum() > 100:
            ob = Obj(np.concatenate([X[::10], Gm[z0 & surfel]]), n_f, d_f, min_h=a.ev_min_h)
        Lg = ob.loc(Gm); hmax = float(np.percentile(ob.loc(X)[:, 2], 99))
        r["back_coverage"], frames_k = back_coverage(ob, hmax, cams, sdep, s, R, t, a.bins, a.cam_max_m); r["frames"] = len(frames_k)
        if a.ids is not None and oid not in a.ids:
            r["decision"] = "not selected (--ids override)"; continue
        if a.ids is None and r["back_coverage"] >= a.min_back_coverage:
            r["decision"] = f"not selected: back coverage {r['back_coverage']:.2f} >= {a.min_back_coverage} (seen from most sides)"; continue
        files = sorted(glob.glob(os.path.join(a.gen_dir, f"obj{oid}_s*.npz")))
        if not files:
            r["decision"] = "skipped: no generated asset (no usable training view of the object)"; continue
        inb = (np.abs(Lg[:, 0]) <= ob.half[0] + a.zone_m) & (np.abs(Lg[:, 1]) <= ob.half[1] + a.zone_m) & (Lg[:, 2] > a.floor_m) & (Lg[:, 2] < hmax + 0.05)
        ev = inb & surfel & (Lg[:, 2] > a.ev_min_h)
        if ev.sum() < 100:
            r["decision"] = f"skipped: too little well-seen evidence ({int(ev.sum())} surfels)"; continue
        al = (box_align(files, Lg[ev], dcb[ev], hmax, ob, a) if a.align == "box" else
              align_generated(files, Lg[ev], dcb[ev], hmax, ob, lambda P: np.zeros(len(P), bool)))
        if al is None:
            r["decision"] = "skipped: generated assets are empty"; continue
        r.update(al["report"]); ch = next((c for c in al["report"]["candidates"] if c["file"] == al["report"]["chosen"]), {})
        r["resid_m"], r["outside_box"] = ch.get("resid_m"), ch.get("outside_box", 0.0)
        if r["resid_m"] is not None and (r["resid_m"] > a.max_resid_m or r["outside_box"] > a.max_outside):
            r["decision"] = f"rejected: registration (residual {r['resid_m']} m > {a.max_resid_m} or outside box {r['outside_box']} > {a.max_outside})"; continue
        k = al["opacity"] > a.min_op; k2 = ob.loc(al["xyz"][k])[:, 2] > a.floor_m * 0.5
        xyz = al["xyz"][k][k2]; rot = al["rot"][k][k2]; scl = al["scale"][k][k2]; dc = al["dc"][k][k2]
        opa = np.clip(np.maximum(al["opacity"][k][k2], a.opaque), 1e-4, 1 - 1e-4)
        scl = np.minimum(scl, a.max_scale_m)                               # no large blobs: surface-sized gaussians only
        xd = ((xyz - t) @ R) / s; dele_k = inb; r["kept_front"] = 0
        if a.hybrid:   # the real front stays: asset rows only where no training camera sees them at / in front of the base surface
            kf = inb & (wmax >= a.keep_w); dele_k = inb & ~kf; r["kept_front"] = int(kf.sum())
            shown = np.zeros(len(xd), bool)
            for (nm_, W, H, fx, fy, cx, cy, Rcw, tcw, _), SD in zip(cams, sdep):
                if SD is None or nm_ not in set(frames_k):
                    continue
                Pc = xd @ Rcw.T + tcw; z = Pc[:, 2]; ok = z > 0.05; zz = np.where(ok, z, 1.0)
                u = np.floor(fx * Pc[:, 0] / zz + cx).astype(int); vv = np.floor(fy * Pc[:, 1] / zz + cy).astype(int)
                ok &= (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                rr = np.linalg.norm(Pc, axis=1); dd = SD[vv[ok], u[ok]]
                sh = np.zeros(len(xd), bool); sh[ok] = (dd <= 0) | (rr[ok] < dd + a.front_m / s); shown |= sh
            if kf.any():
                from scipy.spatial import cKDTree
                shown |= cKDTree(Gm[kf]).query(xyz)[0] < a.front_m          # coincident with a kept surfel
                rgb_a = 0.5 + C0 * dc; rgb_f = np.clip(0.5 + C0 * dcb[kf], 0, 1)   # seam: the kept front's colour statistics
                gn = np.clip(rgb_f.std(0) / np.maximum(rgb_a.std(0), 1e-3), 0.7, 1.4)
                dc = (np.clip((rgb_a - rgb_a.mean(0)) * gn + rgb_f.mean(0), 0, 1) - 0.5) / C0
            xyz, xd, rot, scl, dc, opa = xyz[~shown], xd[~shown], rot[~shown], scl[~shown], dc[~shown], opa[~shown]
            r["asset_rows_seen_dropped"] = int(shown.sum())
        dc = (np.clip(0.5 + C0 * dc, 0, 1) - 0.5) / C0                       # DC colour only, in gamut
        Rd = R.T[None] @ rot; det = np.linalg.det(Rd); Rd[det < 0, :, 0] *= -1
        q = Rotation.from_matrix(Rd).as_quat(); q = np.c_[q[:, 3], q[:, :3]]
        cand[oid] = {"del": dele_k, "frames": set(frames_k), "new": {"means": xd, "scales": np.log(np.maximum(scl / s, 1e-9)), "quats": q,
                                                                     "opacities": np.log(opa / (1 - opa)), "sh0": dc[:, None]}}
        r["deleted"], r["inserted"] = int(dele_k.sum()), int(len(xyz))
    vtx, p0 = load_ply(a.ply, dev); deg = sh_deg(p0); K = p0["shN"].shape[1]
    Rv = ViewerRaster("ss352", "exact", display_scale=a.display_scale)
    tr = all_frames(a.dataset); name2i = {n_: i for i, (n_, _, _) in enumerate(tr)}
    gtf = lambda n_: next(os.path.join(ds, "images", n_) for ds in a.dataset if os.path.exists(os.path.join(ds, "images", n_)))

    def assemble(ids_):
        dele = np.zeros(N0, bool)
        for i_ in ids_:
            dele |= cand[i_]["del"]
        kb = torch.tensor(~dele, device=dev); P = {}
        for k_ in p0:
            add = [np.zeros((len(cand[i_]["new"]["means"]), K, 3)) if k_ == "shN" else cand[i_]["new"][k_] for i_ in ids_]
            P[k_] = torch.cat([p0[k_][kb]] + [torch.tensor(x, dtype=torch.float32, device=dev) for x in add])
        return P, dele, int(kb.sum())
    ids_ = sorted(cand)
    if ids_ and a.harm_iters > 0 and not a.hybrid:                         # wholesale mode: colour-only harmonisation on real frames
        P, dele, nb = assemble(ids_); params = torch.nn.ParameterDict({k_: torch.nn.Parameter(x.clone()) for k_, x in P.items()})
        newm = torch.zeros(len(P["means"]), dtype=torch.bool, device=dev); newm[nb:] = True
        fr = sorted({name2i[n_] for i_ in ids_ for n_ in cand[i_]["frames"] if n_ in name2i})
        opt = torch.optim.Adam([params["sh0"]], lr=2.5e-3, eps=1e-15)
        for step in range(1, a.harm_iters + 1 if fr else 1):
            i = fr[np.random.randint(len(fr))]; n_, cam, c2w = tr[i]
            img, _, _ = Rv.render(params, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)
            gt = load_rgb(gtf(n_), dev).float() / 255
            l = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            opt.zero_grad(set_to_none=True); l.backward(); params["sh0"].grad[~newm] = 0; opt.step()
        off = nb
        for i_ in ids_:
            m_ = len(cand[i_]["new"]["means"]); cand[i_]["new"]["sh0"] = params["sh0"].detach()[off:off + m_].cpu().numpy(); off += m_
    acc = []
    with torch.no_grad():                                                  # accept test: only this object replaced
        for i_ in ids_:
            fk = sorted(name2i[n_] for n_ in cand[i_]["frames"] if n_ in name2i); fk = fk[::max(1, len(fk) // 20)]
            Pk, _, _ = assemble([i_]); pb, pa = [], []
            for i in fk:
                n_, cam, c2w = tr[i]; gt = load_rgb(gtf(n_), dev).float() / 255; w2c = to_t(np.linalg.inv(c2w), dev)
                pb.append(psnr(Rv.render(p0, w2c, *cam_args(cam), sh_degree=deg)[0], gt)); pa.append(psnr(Rv.render(Pk, w2c, *cam_args(cam), sh_degree=deg)[0], gt))
            loss = float(np.mean(pb) - np.mean(pa)) if fk else 0.0; rep[i_]["onpath_loss_db"] = round(loss, 3)
            if loss <= a.max_onpath_loss_db:
                acc.append(i_); rep[i_]["decision"] = f"accepted (object-frame loss {loss:.2f} dB <= {a.max_onpath_loss_db})"
            else:
                rep[i_]["decision"] = f"rejected: on-path loss {loss:.2f} dB > {a.max_onpath_loss_db} (original kept)"
    P, dele, nb = assemble(acc); M = len(P["means"]) - nb
    save_ply(a.out, np.concatenate([vtx.data[~dele], np.zeros(M, vtx.data.dtype)]), P)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_src.npz", src=np.r_[np.nonzero(~dele)[0], -np.ones(M, int)])
    for i_, r in rep.items():
        print(f"[asset] object {i_}: {r['decision']} | back coverage {r['back_coverage']:.2f}, resid {r.get('resid_m')}, outside {r.get('outside_box')}, "
              f"asset {r.get('chosen')}", flush=True)
    out = {"objects": rep, "accepted": acc, "rows": nb + M, "new": M, "deleted": int(dele.sum()), "seconds": round(time.time() - t0), "args": vars(a)}
    json.dump(out, open(os.path.splitext(a.out)[0] + "_mvrefit.json", "w"), indent=1)
    print("ASSET_REPLACE", json.dumps({k_: v_ for k_, v_ in out.items() if k_ not in ("args", "objects")}), flush=True)


if __name__ == "__main__":
    main()
