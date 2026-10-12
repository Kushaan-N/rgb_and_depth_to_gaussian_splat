"""EDGS-style dense-correspondence initialisation for 3DGRUT (venv-edgs: env/make_edgs_venv.sh; GPU).

EDGS (Kotovenko et al., CVPR 2026, arXiv 2504.13204) replaces the sparse SfM init by triangulated dense
RoMa matches: K reference frames (k-means over poses), each matched against its nearest neighbour frames,
pixels sampled by match certainty, each triangulated with every neighbour (known poses) and the best
(lowest reprojection error) kept; colour from the reference pixel, scale = distance-to-camera x factor.
This is a self-contained reimplementation of source/corr_init.py (which is tied to the 3DGS Scene/
GaussianModel classes) on top of RoMa's own matcher, with fixes for short-baseline video:
  - neighbours must have a baseline >= --min-baseline-rel x the median scene depth (EDGS picks the closest
    poses, which for a 10 Hz walking sequence are nearly degenerate), and a viewing angle <= --max-view-deg,
  - a triangulation must have parallax >= --min-parallax-deg and reprojection error <= --proj-err-px in
    both views, the neighbour's own certainty >= --cert-min, and positive depth in both cameras.
Then, optionally:
  - LiDAR check: where the reference frame has a LiDAR target (<image>_depth.npy, ray distance in dataset
    units), keep a point only if its ray distance agrees within max(--depth-rel x D, --depth-abs-m); pixels
    without a target are kept or dropped (--no-lidar-keep). Auto-skipped if no targets exist. Default 10%: on
    CEAR the photo-consistent geometry itself differs from LiDAR by ~3.5% median (independent of baseline;
    the stock RGB splat is only ~54% within 5%), so a 3% gate would drop ~60% of good points; 10% removes
    gross mismatches only. edgs_init.json reports the error quantiles.
  - statistical outlier removal (k-NN mean distance > mean + --outlier-std x std),
  - merge with the dataset's existing points3D.txt (the LiDAR seeds; --no-merge to drop them).
Only TRAINING frames are used: frames are sorted by name and, if --test-split-interval N > 0, every N-th
(index % N == 0) is skipped — exactly 3DGRUT's dataset_colmap.py rule (images sorted by name).

Writes a new dataset dir: sparse/0/{cameras,images}.txt copied (all frames, so the trainer's split is
unchanged), sparse/0/points3D.txt = merged init with colours (3DGRUT's default `initialization=colmap`),
point_cloud.ply = the same points as full Gaussians with EDGS scales/opacity (`initialization=point_cloud`),
images/ symlinked per file (incl. _depth.npy / _covis.png), depth_targets.json copied, edgs_init.json stats.

    python scripts/edgs_init.py --dataset <colmap text ds> --out <new ds> [--test-split-interval 8] \
        [--sim3-json <pipeline>/sim3.json] [--num-refs 180 --nns 3 --matches-per-ref 15000]
"""
from __future__ import annotations
import argparse, json, os, shutil, time
import numpy as np

SH_C0 = 0.28209479177387814


def read_model(d):
    cams = {}
    for ln in open(os.path.join(d, "cameras.txt")):
        if ln.strip() and not ln.startswith("#"):
            p = ln.split(); m, W, H, v = p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]]
            if m == "SIMPLE_PINHOLE": K = (v[0], v[0], v[1], v[2])
            elif m == "PINHOLE": K = tuple(v[:4])
            elif m in ("SIMPLE_RADIAL", "RADIAL", "OPENCV") and max(abs(x) for x in v[3 if m != "OPENCV" else 4:]) < 1e-9:
                K = (v[0], v[0], v[1], v[2]) if m != "OPENCV" else tuple(v[:4])
            else: raise SystemExit(f"camera model {m} with distortion not supported (undistort first)")
            cams[int(p[0])] = (W, H, K)
    imgs = []
    lines = [ln for ln in open(os.path.join(d, "images.txt")) if not ln.startswith("#")]
    for ln in lines[::2]:
        p = ln.split()
        if len(p) >= 10:
            imgs.append((p[9], int(p[8]), np.array(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8])))))
    return cams, sorted(imgs, key=lambda x: x[0])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_points(path):
    xyz, rgb = [], []
    if os.path.isfile(path):
        for ln in open(path):
            if ln.strip() and not ln.startswith("#"):
                p = ln.split(); xyz.append([float(x) for x in p[1:4]]); rgb.append([int(x) for x in p[4:7]])
    return np.array(xyz, np.float64).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)


def select_refs(poses, k, seed):
    """EDGS: k-means over flattened world-to-camera matrices, the frame nearest each centre."""
    from scipy.cluster.vq import kmeans2
    if k >= len(poses):
        return list(range(len(poses)))
    X = poses.reshape(len(poses), -1)
    cent, lab = kmeans2(X, k, seed=seed, minit="++")
    out = set()
    for c in range(k):
        m = np.where(lab == c)[0]
        if len(m): out.add(int(m[np.argmin(np.linalg.norm(X[m] - cent[c], axis=1))]))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--test-split-interval", type=int, default=8, help="3DGRUT's; <=0 = all frames are training")
    ap.add_argument("--num-refs", type=int, default=180); ap.add_argument("--nns", type=int, default=3)
    ap.add_argument("--matches-per-ref", type=int, default=15000)
    ap.add_argument("--roma-model", choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--roma-upsample", action="store_true", help="RoMa's 864px refinement (EDGS turns it off)")
    ap.add_argument("--cert-min", type=float, default=0.05, help="neighbour's own match certainty (RoMa sample_thresh)")
    ap.add_argument("--min-baseline-rel", type=float, default=0.15, help="x median scene depth")
    ap.add_argument("--max-view-deg", type=float, default=60.0)
    ap.add_argument("--min-parallax-deg", type=float, default=1.0)
    ap.add_argument("--proj-err-px", type=float, default=2.0)
    ap.add_argument("--depth-rel", type=float, default=0.10); ap.add_argument("--depth-abs-m", type=float, default=0.03)
    ap.add_argument("--sim3-json", default=None, help="for --depth-abs-m in metres (else read as dataset units)")
    ap.add_argument("--no-lidar-keep", action="store_true", help="drop points where the ref frame has no LiDAR target")
    ap.add_argument("--no-depth-filter", action="store_true")
    ap.add_argument("--outlier-k", type=int, default=16); ap.add_argument("--outlier-std", type=float, default=2.0)
    ap.add_argument("--no-merge", action="store_true", help="drop the dataset's existing points3D (LiDAR seeds)")
    ap.add_argument("--max-points", type=int, default=0, help="random cap on the EDGS points (0 = none)")
    ap.add_argument("--scaling-factor", type=float, default=0.001, help="EDGS: scale = dist to ref camera x this")
    ap.add_argument("--seed-scale-factor", type=float, default=0.01, help="3DGRUT colmap default for the seeds")
    ap.add_argument("--opacity", type=float, default=0.5, help="EDGS points (EDGS: logit 0)")
    ap.add_argument("--seed-opacity", type=float, default=0.1, help="LiDAR seeds (3DGRUT default_density)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from scipy.spatial import cKDTree
    from romatch import roma_outdoor, roma_indoor
    t0 = time.time(); dev = "cuda"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    sp = os.path.join(args.dataset, "sparse", "0")
    cams, imgs = read_model(sp)
    tr = [im for k, im in enumerate(imgs) if args.test_split_interval <= 0 or k % args.test_split_interval != 0]
    print(f"[edgs] {len(imgs)} frames, {len(tr)} training frames used")
    Rs = np.stack([quat_to_R(q) for _, _, q, _ in tr]); ts = np.stack([t for *_, t in tr])
    C = -np.einsum("nji,nj->ni", Rs, ts)                                          # camera centres
    fwd = Rs[:, 2, :]                                                             # optical axes (world)
    w2c = np.concatenate([np.concatenate([Rs, ts[:, :, None]], 2), np.tile([[[0, 0, 0, 1.]]], (len(tr), 1, 1))], 1)
    seeds, seed_rgb = read_points(os.path.join(sp, "points3D.txt"))
    dpath = lambda n: os.path.splitext(os.path.join(args.dataset, "images", n))[0] + "_depth.npy"
    has_depth = all(os.path.isfile(dpath(n)) for n, *_ in tr[:: max(1, len(tr) // 10)])
    use_depth = has_depth and not args.no_depth_filter
    scale_m = float(json.load(open(args.sim3_json))["scale"]) if args.sim3_json else 1.0
    # median scene depth: LiDAR targets if present, else the seed cloud seen from a few cameras
    zs = []
    for i in np.linspace(0, len(tr) - 1, min(20, len(tr))).astype(int):
        if has_depth:
            d = np.load(dpath(tr[i][0])).astype(np.float32); zs.append(np.median(d[d > 0]) if (d > 0).any() else np.nan)
        elif len(seeds):
            z = (seeds @ Rs[i].T + ts[i])[:, 2]; zs.append(np.median(z[z > 0]) if (z > 0).any() else np.nan)
    scene_depth = float(np.nanmedian(zs)) if zs and np.isfinite(np.nanmedian(zs)) else float(np.median(np.linalg.norm(C - C.mean(0), axis=1))) + 1e-6
    bmin = args.min_baseline_rel * scene_depth
    print(f"[edgs] median scene depth {scene_depth:.3f}, min baseline {bmin:.3f} (dataset units); LiDAR filter {use_depth}")

    refs = select_refs(w2c, min(args.num_refs, len(tr)), args.seed)
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    nn_of = {}
    for r in refs:
        cand = np.where((D[r] >= bmin) & (ang[r] <= args.max_view_deg))[0]
        cand = cand[np.argsort(D[r, cand])][: args.nns]
        if len(cand) == 0:                                                        # fall back to EDGS's nearest poses
            o = np.argsort(np.linalg.norm((w2c - w2c[r]).reshape(len(tr), -1), axis=1)); cand = o[1: args.nns + 1]
        nn_of[r] = cand
    nb = np.array([D[r, nn_of[r]].mean() for r in refs])
    print(f"[edgs] {len(refs)} reference frames, neighbour baseline median {np.median(nb):.3f}")

    roma = (roma_outdoor if args.roma_model == "outdoor" else roma_indoor)(device=dev)
    roma.upsample_preds = args.roma_upsample; roma.symmetric = False
    load = lambda n: Image.open(os.path.join(args.dataset, "images", n)).convert("RGB")
    kmat = lambda c: torch.tensor([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], dtype=torch.float64, device=dev)
    T = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
    out_xyz, out_rgb, out_scale, relerr = [], [], [], []
    st = {"sampled": 0, "triangulated": 0, "lidar_checked": 0, "lidar_agree": 0, "lidar_missing": 0, "kept": 0}
    for ri, r in enumerate(refs):
        nA, cA = tr[r][0], tr[r][1]; W, H, KA = cams[cA]; imA = load(nA)
        warps, certs = [], []
        for j in nn_of[r]:
            w, c = roma.match(imA, load(tr[j][0]), device=dev)
            warps.append(w.reshape(-1, 4).double()); certs.append(c.reshape(-1).double())
        certs = torch.stack(certs); cmax = certs.max(0).values.clone()
        cmax[cmax > roma.sample_thresh] = 1
        idx = torch.multinomial(cmax.float(), min(args.matches_per_ref, len(cmax)), replacement=False)
        st["sampled"] += len(idx)
        xa = warps[0][idx, :2]                                                    # same grid for all neighbours
        uA = (xa[:, 0] + 1) * W / 2; vA = (xa[:, 1] + 1) * H / 2                  # COLMAP px (centre at +0.5)
        RA, tA, CA = T(Rs[r]), T(ts[r]), T(C[r])
        rayA = torch.stack([(uA - KA[2]) / KA[0], (vA - KA[3]) / KA[1], torch.ones_like(uA)], 1)
        best = torch.full((len(idx),), float("inf"), dtype=torch.float64, device=dev)
        bestX = torch.zeros(len(idx), 3, dtype=torch.float64, device=dev)
        PA = torch.cat([RA, tA[:, None]], 1)
        for k, j in enumerate(nn_of[r]):
            WB, HB, KB = cams[tr[j][1]]
            xb = warps[k][idx, 2:]; uB = (xb[:, 0] + 1) * WB / 2; vB = (xb[:, 1] + 1) * HB / 2
            RB, tB, CB = T(Rs[j]), T(ts[j]), T(C[j]); PB = torch.cat([RB, tB[:, None]], 1)
            rayB = torch.stack([(uB - KB[2]) / KB[0], (vB - KB[3]) / KB[1], torch.ones_like(uB)], 1)
            A = torch.stack([rayA[:, :1] * PA[2] - PA[0], rayA[:, 1:2] * PA[2] - PA[1],
                             rayB[:, :1] * PB[2] - PB[0], rayB[:, 1:2] * PB[2] - PB[1]], 1)   # DLT, normalised coords
            Xh = torch.linalg.svd(A).Vh[:, -1]; X = Xh[:, :3] / Xh[:, 3:4]
            pa, pb = X @ RA.T + tA, X @ RB.T + tB
            ea = torch.hypot(KA[0] * pa[:, 0] / pa[:, 2] + KA[2] - uA, KA[1] * pa[:, 1] / pa[:, 2] + KA[3] - vA)
            eb = torch.hypot(KB[0] * pb[:, 0] / pb[:, 2] + KB[2] - uB, KB[1] * pb[:, 1] / pb[:, 2] + KB[3] - vB)
            e = torch.maximum(ea, eb)
            da, db = X - CA, X - CB
            par = torch.rad2deg(torch.arccos(torch.clamp((da * db).sum(1) / (da.norm(dim=1) * db.norm(dim=1)), -1, 1)))
            ok = (Xh[:, 3].abs() > 1e-12) & (pa[:, 2] > 0) & (pb[:, 2] > 0) & (e <= args.proj_err_px) \
                & (par >= args.min_parallax_deg) & (certs[k][idx] >= args.cert_min) & (e < best)
            best = torch.where(ok, e, best); bestX[ok] = X[ok]
        good = torch.isfinite(best)
        st["triangulated"] += int(good.sum())
        dist = (bestX - CA).norm(dim=1)
        ui = torch.clamp(uA.floor().long(), 0, W - 1); vi = torch.clamp(vA.floor().long(), 0, H - 1)
        if use_depth:
            Dt = torch.tensor(np.load(dpath(nA)).astype(np.float32), dtype=torch.float64, device=dev)[vi, ui]
            have = good & (Dt > 0)
            agree = (dist - Dt).abs() <= torch.maximum(args.depth_rel * Dt, torch.full_like(Dt, args.depth_abs_m / scale_m))
            st["lidar_checked"] += int(have.sum()); st["lidar_agree"] += int((have & agree).sum())
            st["lidar_missing"] += int((good & ~(Dt > 0)).sum()); relerr.append(((dist - Dt) / Dt)[have].cpu().numpy())
            good = good & torch.where(Dt > 0, agree, torch.full_like(agree, not args.no_lidar_keep))
        im = torch.tensor(np.asarray(imA), device=dev)
        out_xyz.append(bestX[good].cpu().numpy()); out_rgb.append(im[vi[good], ui[good]].cpu().numpy())
        out_scale.append((dist[good] * args.scaling_factor).cpu().numpy())
        if ri % 20 == 0 or ri == len(refs) - 1:
            print(f"[edgs] ref {ri + 1}/{len(refs)} {nA}: kept {int(good.sum())}/{len(idx)}  ({time.time() - t0:.0f}s)", flush=True)
    xyz, rgb, scl = np.concatenate(out_xyz), np.concatenate(out_rgb), np.concatenate(out_scale)
    st["after_lidar"] = len(xyz)
    if relerr:                                                                    # triangulation vs LiDAR, before the filter
        re = np.concatenate(relerr); st["lidar_relerr_signed_median"] = round(float(np.median(re)), 4)
        st["lidar_relerr_quantiles"] = {q: round(float(np.quantile(np.abs(re), float(q))), 4) for q in ("0.25", "0.5", "0.75", "0.9")}
    if args.outlier_k > 0 and len(xyz) > args.outlier_k:
        dk = cKDTree(xyz).query(xyz, k=args.outlier_k + 1, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))[0][:, 1:].mean(1)
        m = dk <= dk.mean() + args.outlier_std * dk.std(); xyz, rgb, scl = xyz[m], rgb[m], scl[m]
    st["after_outlier"] = len(xyz)
    if args.max_points and len(xyz) > args.max_points:
        m = rng.choice(len(xyz), args.max_points, replace=False); xyz, rgb, scl = xyz[m], rgb[m], scl[m]
    st["kept"] = len(xyz)
    if args.no_merge: seeds = seeds[:0]; seed_rgb = seed_rgb[:0]
    st["seeds"] = len(seeds)
    # seeds get 3DGRUT's colmap-init scale: distance to the nearest training camera x seed factor
    seed_scl = cKDTree(C).query(seeds)[0] * args.seed_scale_factor if len(seeds) else np.zeros(0)
    P = np.concatenate([seeds, xyz]); RGB = np.concatenate([seed_rgb, rgb]).astype(np.uint8)
    S = np.clip(np.concatenate([seed_scl, scl]), 1e-7, None)
    OP = np.concatenate([np.full(len(seeds), args.seed_opacity), np.full(len(xyz), args.opacity)])

    o = args.out; os.makedirs(os.path.join(o, "sparse", "0"), exist_ok=True); os.makedirs(os.path.join(o, "images"), exist_ok=True)
    for f in ("cameras.txt", "images.txt"):
        shutil.copyfile(os.path.join(sp, f), os.path.join(o, "sparse", "0", f))
    with open(os.path.join(o, "sparse", "0", "points3D.txt"), "w") as fh:
        fh.write(f"# EDGS dense-correspondence init ({len(xyz)}) + seeds ({len(seeds)}); scripts/edgs_init.py\n")
        fh.writelines(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n" for i, (p, c) in enumerate(zip(P, RGB)))
    write_gaussian_ply(os.path.join(o, "point_cloud.ply"), P, RGB, S, OP)
    src = os.path.join(args.dataset, "images")
    for n in os.listdir(src):
        d = os.path.join(o, "images", n)
        if not os.path.lexists(d): os.symlink(os.path.realpath(os.path.join(src, n)), d)
    for f in ("depth_targets.json",):
        if os.path.isfile(os.path.join(args.dataset, f)): shutil.copyfile(os.path.join(args.dataset, f), os.path.join(o, f))
    st.update({"frames": len(imgs), "train_frames": len(tr), "refs": len(refs), "scene_depth": scene_depth,
               "min_baseline": bmin, "lidar_filter": use_depth, "total_points": len(P),
               "lidar_agree_fraction": round(st["lidar_agree"] / max(1, st["lidar_checked"]), 4),
               "seconds": round(time.time() - t0, 1), "args": vars(args)})
    json.dump(st, open(os.path.join(o, "edgs_init.json"), "w"), indent=2)
    print("EDGS_INIT " + json.dumps({k: v for k, v in st.items() if k != "args"}))


def write_gaussian_ply(path, P, RGB, S, OP):
    """3DGRUT `initialization=point_cloud` layout: raw (pre-activation) params, SH degree 3."""
    n = len(P); names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    a = np.zeros((n, len(names)), np.float32)
    a[:, :3] = P; a[:, 3:6] = (RGB / 255.0 - 0.5) / SH_C0
    a[:, 51] = np.log(OP / (1 - OP)); a[:, 52:55] = np.log(S)[:, None]; a[:, 55] = 1.0
    with open(path, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
                  + "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
        fh.write(a.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
