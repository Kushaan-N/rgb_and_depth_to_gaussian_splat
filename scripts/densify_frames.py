"""Use more of the recording: add low-blur frames to the existing SfM model without moving it (CPU; COLMAP).

The pipeline's frame gate (gate_frames.py) keeps ~8% of the RGB frames (slower half by |omega|, then 3 cm
thinning). Here extra frames are chosen by PREDICTED MOTION BLUR in pixels instead (blur_diagnosis.py:
per-frame image-flow rate x the fitted exposure time) and a finer pose thinning, then registered into the
EXISTING COLMAP model ($P/dataset/sparse/0, its own database) with the old frames and the camera fixed, so the
coordinate frame, the intrinsics and the held-out test poses are untouched:
  1. select   candidates = frames not in the model, after the sync preamble, blur <= --max-blur-px,
              Laplacian variance >= gating.laplacian_min, NOT within --guard-frames frames (recording order)
              or --guard-m metres of any held-out test camera (default gating.spatial_min_sep_m: no new frame
              is closer to a test view than the gate already allowed); then, lowest blur first, a candidate
              is dropped if a frame already selected (the model's frames included) is within --sep-m AND
              --sep-deg (mocap poses); --max-new caps the count.
  2. register SIFT (same options as colmap_sfm.py) on the new frames into a copy of the model's database, on
              the model's camera; pairs = each new frame x its --pairs-knn nearest frames (mocap pose, view
              angle <= 60 deg) + +-3 neighbours in time; incremental mapper continuing from the model with
              fix_existing_frames and no intrinsics refinement (= COLMAP image_registrator + BA, old frames
              fixed).
  3. verify   old frames must not move (max centre/rotation change reported); per new frame: reprojection
              error, #3D points, and pose consistency with mocap through $P/sim3.json (residuals compared with
              the old frames'). A new frame is kept only if registered, mean reproj <= --max-reproj-px,
              >= --min-points observations, centre residual <= max(--max-res-m, 3 x old-frame p95) and rotation
              residual <= --max-rot-deg (relative to the old frames' median rotation offset).
  4. write    <out>/data: COLMAP text dataset = the training frames of --base-dataset (its poses verbatim,
              images/ incl. _depth.npy linked) + the kept new frames (images -> precond/rgb), test frames
              EXCLUDED (train with test_split_interval 0; score on --test as before), points3D.txt copied
              (LiDAR seeds). <out>/newframes: the new frames alone (real images/ dir) for build_depth_targets.
              <out>/densify.json: counts, thresholds, registration + residual stats, timings.

    python scripts/densify_frames.py --config configs/mocap2_well-lit_trot.yaml --out $P/densify_frames \
        [--blur-dir $P/blur_dt24] [--max-blur-px 0.8 --sep-m 0.01 --sep-deg 1.5] [--select-only]
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
import gating_utils as gu
from build_poses import compute_world_cam
from sim3_utils import load_sim3


def rot_deg(Ra, Rb):
    """Geodesic angle(s) between rotation stacks (..., 3, 3), degrees."""
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def read_text_images(d):
    """images.txt -> {name: (line, R_wc, C)} (pose line kept verbatim)."""
    from scipy.spatial.transform import Rotation
    out = {}
    lines = [ln for ln in open(os.path.join(d, "images.txt")) if not ln.startswith("#")]
    for ln in lines[::2]:
        p = ln.split()
        if len(p) < 10: continue
        q, t = np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8])))
        Rcw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        out[p[9]] = (ln, Rcw.T, -Rcw.T @ t)
    return out


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a))} if not len(a) else {"n": int(len(a)), "median": round(float(np.median(a)), 5),
            "p90": round(float(np.quantile(a, .9)), 5), "p95": round(float(np.quantile(a, .95)), 5), "max": round(float(a.max()), 5)}


def select(args, cfg, P):
    kept, Twc, *_ = compute_world_cam(cfg)                        # mocap camera-to-world, metric
    names = [f.rgb_name for f in kept]; ts = np.array([f.t for f in kept]); idx = {n: i for i, n in enumerate(names)}
    model = set(read_text_images(os.path.join(args.base_dataset, "sparse", "0")) or [])
    test = set(n for n in os.listdir(os.path.join(args.test, "images")) if n.endswith(".png"))
    model |= test
    bj = json.load(open(os.path.join(args.blur_dir, "blur_diagnosis.json")))
    texp = args.texp_ms or float(bj["image_evidence"]["texp_fit_ms"])
    blur = {r["name"]: float(r[args.blur_col]) * texp for r in csv.DictReader(open(os.path.join(args.blur_dir, "per_frame.csv")))}
    ok, n_pre = gu.apply_preamble_exclusion(cfg, ts, np.ones(len(names), bool))
    g = cfg.get("gating", {}); guard_m = args.guard_m if args.guard_m is not None else float(g.get("spatial_min_sep_m", 0.03))
    lap_min = args.laplacian_min if args.laplacian_min is not None else float(g.get("laplacian_min", 0))
    ti = np.array(sorted(idx[n] for n in test if n in idx))
    C, R = Twc[:, :3, 3], Twc[:, :3, :3]
    near_t = np.zeros(len(names), bool)
    for k in range(-args.guard_frames, args.guard_frames + 1):
        near_t[np.clip(ti + k, 0, len(names) - 1)] = True
    dmin_test = np.linalg.norm(C[:, None] - C[ti][None], axis=2).min(1)
    b = np.array([blur.get(n, np.inf) for n in names])
    cand = ok & ~np.isin(names, list(model)) & (b <= args.max_blur_px) & ~near_t & (dmin_test > guard_m)
    funnel = {"frames": len(names), "in_model": int(np.isin(names, list(model)).sum()), "after_preamble": int(ok.sum()),
              "blur_ok": int((ok & (b <= args.max_blur_px)).sum()), "test_guard_ok": int(cand.sum())}
    root_img = os.path.join(cfg["paths"]["out_root"], "precond", cfg["sequence"].get("rgb_dir", "rgb"))
    if lap_min > 0:                                               # same sharpness floor as the gate
        for i in np.where(cand)[0]:
            cand[i] = iu.laplacian_var(iu.read_rgb(os.path.join(root_img, names[i]))) >= lap_min
    funnel["laplacian_ok"] = int(cand.sum())
    sel = np.zeros(len(names), bool); sel[[idx[n] for n in model if n in idx]] = True; out = []
    for i in sorted(np.where(cand)[0], key=lambda i: b[i]):        # sharpest first
        j = np.where(sel)[0]; j = j[np.linalg.norm(C[j] - C[i], axis=1) < args.sep_m]
        if len(j) and (rot_deg(R[j], R[i][None]) < args.sep_deg).any(): continue    # near-duplicate view
        sel[i] = True; out.append(i)
        if args.max_new and len(out) >= args.max_new: break
    funnel["thinned"] = len(out)
    out = sorted(out)
    info = {"texp_ms": texp, "blur_dir": args.blur_dir, "blur_col": args.blur_col, "guard_m": guard_m, "laplacian_min": lap_min,
            "n_preamble": int(n_pre), "funnel": funnel, "new_blur_px": stats(b[out]),
            "model_blur_px": stats([b[idx[n]] for n in model if n in idx]),
            "new_min_dist_to_test_m": stats(dmin_test[out])}
    return [names[i] for i in out], names, Twc, info


def register(args, cfg, new, names, Twc, W):
    import pycolmap
    root_img = os.path.join(cfg["paths"]["out_root"], "precond", cfg["sequence"].get("rgb_dir", "rgb"))
    db_path = os.path.join(W, "database.db"); shutil.copyfile(args.base_db, db_path)
    rec0 = pycolmap.Reconstruction(args.base_model)
    cam = next(iter(rec0.cameras.values()))
    db = pycolmap.Database.open(db_path); db.update_camera(cam); have = {im.name for im in db.read_all_images()}; db.close()
    todo = [n for n in new if n not in have]
    ro = pycolmap.ImageReaderOptions(); ro.existing_camera_id = cam.camera_id
    eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = args.threads
    t0 = time.time()
    pycolmap.extract_features(db_path, root_img, image_names=todo, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=eo, device=pycolmap.Device.cpu)
    t_feat = time.time() - t0
    # pairs: each new frame x its nearest frames (mocap pose), plus temporal neighbours
    allr = sorted(set(new) | {im.name for im in rec0.images.values()})
    idx = {n: i for i, n in enumerate(names)}; ii = np.array([idx[n] for n in allr])
    C, Z = Twc[ii, :3, 3], Twc[ii, :3, 2]; pos = {n: k for k, n in enumerate(allr)}
    pairs = set()
    for n in new:
        k = pos[n]; d = np.linalg.norm(C - C[k], axis=1); d[k] = np.inf
        d[np.degrees(np.arccos(np.clip(Z @ Z[k], -1, 1))) > 60] = np.inf
        nn = [j for j in np.argsort(d)[:args.pairs_knn] if np.isfinite(d[j])]
        for j in nn + [k + o for o in (-3, -2, -1, 1, 2, 3) if 0 <= k + o < len(allr)]:
            pairs.add(tuple(sorted((allr[k], allr[j]))))
    pl = os.path.join(W, "pairs.txt"); open(pl, "w").write("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = args.threads
    po = pycolmap.ImportedPairingOptions(); po.match_list_path = pl
    t0 = time.time(); pycolmap.match_image_pairs(db_path, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
    t_match = time.time() - t0
    opts = pycolmap.IncrementalPipelineOptions()
    opts.fix_existing_frames = True; opts.num_threads = args.threads
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    opts.multiple_models = False
    sp = os.path.join(W, "sparse"); shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db_path, root_img, sp, options=opts, input_path=args.base_model)
    t_map = time.time() - t0
    if not recs: raise SystemExit("[densify] mapper returned no model")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())             # (mapper wrote it to <W>/sparse)
    return rec0, rec, {"pairs": len(pairs), "extract_s": round(t_feat, 1), "match_s": round(t_match, 1), "map_s": round(t_map, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=None, help="SfM model to extend (default $P/dataset/sparse/0)")
    ap.add_argument("--base-db", default=None, help="its database (default $P/dataset_work/database.db)")
    ap.add_argument("--base-dataset", default=None, help="text dataset whose frames/poses/points are reused ($P/onpath/data_orig)")
    ap.add_argument("--test", default=None, help="held-out set; its frames are excluded and guarded ($P/onpath/test)")
    ap.add_argument("--blur-dir", default=None, help="blur_diagnosis.py output (default $P/blur)")
    ap.add_argument("--blur-col", default="rate_px_per_ms_med", help="per_frame.csv flow-rate column (px/ms)")
    ap.add_argument("--texp-ms", type=float, default=0.0, help="exposure time; 0 = the blur dir's fitted texp_fit_ms")
    ap.add_argument("--max-blur-px", type=float, default=0.8)
    ap.add_argument("--laplacian-min", type=float, default=None, help="default gating.laplacian_min")
    ap.add_argument("--sep-m", type=float, default=0.01); ap.add_argument("--sep-deg", type=float, default=1.5)
    ap.add_argument("--guard-frames", type=int, default=2); ap.add_argument("--guard-m", type=float, default=None)
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--pairs-knn", type=int, default=15)
    ap.add_argument("--max-reproj-px", type=float, default=1.5); ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--max-res-m", type=float, default=0.03); ap.add_argument("--max-rot-deg", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--include-test", action="store_true",
                    help="deliverable dataset: keep the held-out test frames too (selection still guards them)")
    args = ap.parse_args()
    cfg = pu.load_config(args.config); P = os.path.join(cfg["paths"]["out_root"], "pipeline")
    args.base_model = args.base_model or os.path.join(P, "dataset", "sparse", "0")
    args.base_db = args.base_db or os.path.join(P, "dataset_work", "database.db")
    args.base_dataset = args.base_dataset or os.path.join(P, "onpath", "data_orig")
    args.test = args.test or os.path.join(P, "onpath", "test")
    args.blur_dir = args.blur_dir or os.path.join(P, "blur")
    T0 = time.time(); os.makedirs(args.out, exist_ok=True)
    new, names, Twc, info = select(args, cfg, P)
    info["select_s"] = round(time.time() - T0, 1); info["args"] = vars(args)
    print(f"[densify] selection funnel {info['funnel']}; new-frame blur {info['new_blur_px']}", flush=True)
    if args.select_only:
        open(os.path.join(args.out, "selected.txt"), "w").write("\n".join(new) + "\n")
        pu.save_json(os.path.join(args.out, "densify.json"), info); print("DENSIFY_SELECT_DONE", len(new)); return
    W = os.path.join(args.out, "colmap_work"); os.makedirs(W, exist_ok=True)
    rec0, rec, tinfo = register(args, cfg, new, names, Twc, W); info["timing"] = tinfo

    # ---- verify: old frames fixed, new frames' reprojection + consistency with mocap through the sim3
    from fuse_sequences import cam_to_world
    s, Rs, tS = load_sim3(os.path.join(P, "sim3.json"))
    idx = {n: i for i, n in enumerate(names)}
    by = {im.name: im for im in rec.images.values() if im.has_pose}
    moved_c, moved_r = [], []
    for im0 in rec0.images.values():
        R0, C0 = cam_to_world(im0); R1, C1 = cam_to_world(by[im0.name]) if im0.name in by else (None, None)
        if R1 is None: moved_c.append(np.inf); continue
        moved_c.append(np.linalg.norm(C1 - C0)); moved_r.append(float(rot_deg(R0, R1)))
    cam0, cam1 = next(iter(rec0.cameras.values())), rec.cameras[next(iter(rec0.cameras))]
    info["fixed_check"] = {"old_frames_missing": int(np.isinf(moved_c).sum()), "max_centre_change": float(np.max(moved_c)),
                           "max_rot_change_deg": float(np.max(moved_r)), "camera_params_change": float(np.abs(np.array(cam1.params) - np.array(cam0.params)).max())}
    def resid(n):
        R, C = cam_to_world(by[n]); Tm = Twc[idx[n]]
        return np.linalg.norm(s * Rs @ C + tS - Tm[:3, 3]), (Rs @ R).T @ Tm[:3, :3]
    old = [n for n in (im.name for im in rec0.images.values()) if n in idx]
    ro = [resid(n) for n in old]; old_res = np.array([r[0] for r in ro])
    from scipy.spatial.transform import Rotation
    off = Rotation.from_matrix(np.stack([r[1] for r in ro])).mean().as_matrix()     # constant camera/body rotation offset
    old_rot = rot_deg(np.stack([r[1] for r in ro]), off)
    res_thr = max(args.max_res_m, 3 * float(np.quantile(old_res, .95)))
    rows, keep = [], []
    for n in new:
        r = {"name": n, "registered": n in by}
        if n in by:
            im = by[n]; e, rr = resid(n)
            errs = [rec.points3D[p.point3D_id].error for p in im.points2D if p.has_point3D()]
            r.update(points=im.num_points3D, reproj=round(float(np.mean(errs)), 4) if errs else None,
                     res_m=round(float(e), 5), rot_deg=round(float(rot_deg(rr, off)), 4))
            r["kept"] = bool(errs and r["reproj"] <= args.max_reproj_px and im.num_points3D >= args.min_points
                             and e <= res_thr and r["rot_deg"] <= args.max_rot_deg)
            if r["kept"]: keep.append(n)
        rows.append(r)
    reg = [r for r in rows if r["registered"]]
    info["registration"] = {"selected": len(new), "registered": len(reg), "kept": len(keep), "res_threshold_m": res_thr,
                            "model_mean_reproj_px": round(float(rec.compute_mean_reprojection_error()), 4),
                            "base_model_mean_reproj_px": round(float(rec0.compute_mean_reprojection_error()), 4),
                            "new_reproj_px": stats([r["reproj"] for r in reg if r["reproj"] is not None]),
                            "new_points": stats([r["points"] for r in reg]),
                            "new_sim3_residual_m": stats([r["res_m"] for r in reg]), "old_sim3_residual_m": stats(old_res),
                            "new_rot_residual_deg": stats([r["rot_deg"] for r in reg]), "old_rot_residual_deg": stats(old_rot),
                            "kept_sim3_residual_m": stats([r["res_m"] for r in reg if r["kept"]])}
    with open(os.path.join(args.out, "new_frames.csv"), "w") as f:
        w = csv.DictWriter(f, ["name", "registered", "kept", "points", "reproj", "res_m", "rot_deg"]); w.writeheader(); w.writerows(rows)
    print(f"[densify] fixed-frame check {info['fixed_check']}", flush=True)
    print(f"[densify] registration {json.dumps(info['registration'])}", flush=True)
    if info["fixed_check"]["max_centre_change"] > 1e-6 or info["fixed_check"]["camera_params_change"] > 1e-6:
        raise SystemExit("[densify] the existing frames or the camera moved -- refusing to write a dataset")

    # ---- write: base training frames (verbatim) + kept new frames, test frames excluded
    test = set(n for n in os.listdir(os.path.join(args.test, "images")) if n.endswith(".png"))
    bsp = os.path.join(args.base_dataset, "sparse", "0"); base = read_text_images(bsp)
    D = os.path.join(args.out, "data"); ND = os.path.join(args.out, "newframes")
    for d in (D, ND):
        if os.path.isdir(d): shutil.rmtree(d)
        os.makedirs(os.path.join(d, "sparse", "0")); os.makedirs(os.path.join(d, "images"))
        shutil.copyfile(os.path.join(bsp, "cameras.txt"), os.path.join(d, "sparse", "0", "cameras.txt"))
    shutil.copyfile(os.path.join(bsp, "points3D.txt"), os.path.join(D, "sparse", "0", "points3D.txt"))
    open(os.path.join(ND, "sparse", "0", "points3D.txt"), "w").close()
    cid = open(os.path.join(bsp, "cameras.txt")).read().split()[0]
    root_img = os.path.join(cfg["paths"]["out_root"], "precond", cfg["sequence"].get("rgb_dir", "rgb"))
    ent = [(n, base[n][0].split()[1:8]) for n in base if args.include_test or n not in test]
    for n in keep:
        M = np.asarray(by[n].cam_from_world().matrix() if callable(by[n].cam_from_world) else by[n].cam_from_world.matrix())
        q = Rotation.from_matrix(M[:, :3]).as_quat()
        ent.append((n, [f"{q[3]:.12g}", f"{q[0]:.12g}", f"{q[1]:.12g}", f"{q[2]:.12g}"] + [f"{x:.12g}" for x in M[:, 3]]))
    for d, sub in ((D, sorted(ent)), (ND, sorted(e for e in ent if e[0] in set(keep)))):
        with open(os.path.join(d, "sparse", "0", "images.txt"), "w") as f:
            for i, (n, p) in enumerate(sub, 1): f.write(f"{i} {' '.join(p)} {cid} {n}\n\n")
    bimg = os.path.join(args.base_dataset, "images")
    for n, _ in ent:
        if n in base:
            for fn in (n, os.path.splitext(n)[0] + "_depth.npy"):
                if os.path.exists(os.path.join(bimg, fn)): os.symlink(os.path.realpath(os.path.join(bimg, fn)), os.path.join(D, "images", fn))
        else:
            os.symlink(os.path.join(root_img, n), os.path.join(D, "images", n)); os.symlink(os.path.join(root_img, n), os.path.join(ND, "images", n))
    info["dataset"] = {"dir": D, "frames": len(ent), "base_train": sum(n in base for n, _ in ent), "new": len(keep),
                       "test_excluded": 0 if args.include_test else len(test)}
    info["total_s"] = round(time.time() - T0, 1)
    pu.save_json(os.path.join(args.out, "densify.json"), info)
    print(f"[densify] wrote {D}: {len(ent)} training frames ({info['dataset']['base_train']} base + {len(keep)} new), "
          f"{info['dataset']['test_excluded']} test frames excluded; {info['total_s']} s")
    print("DENSIFY_DONE", len(keep))


if __name__ == "__main__":
    main()
