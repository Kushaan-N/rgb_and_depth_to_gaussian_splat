"""FIORD adapter (Gunes et al. 2025, arXiv 2504.01732; CC-BY-4.0, Zenodo 15181976): Insta360 One RS 1-Inch dual 200 deg
fisheye (two back-to-back lenses = a 2-camera rig, 3264x3264 each, OPENCV_FISHEYE per lens), still captures, a COLMAP
model already aligned to a Faro Focus terrestrial laser scan (metric; .e57), indoor + outdoor scenes. CPU venv.

  fetch    download <scene>.zip from Zenodo (md5-verified, resumable) into $CEAR_DATA/fiord and unpack only what the
           pipeline reads: fisheyeimages/, model/, the TLS .e57 and the SfM .ply
  prepare  scripts/datasets/omni_common.prepare(): every `split.every`-th CAPTURE (both lenses of one shot, grouped
           by the shared capture id) is held out;
           rect/{train,test}: each fisheye -> virtual pinhole views (`rect.views`, default a 90 deg centre view +
           four 70 deg views tilted 40 deg, all inside 85 deg of the axis) for the CEAR pinhole recipe and for
           SuperSplat scoring; native/{all,train}: the fisheye frames downsampled by `native.downsample` with
           <stem>_mask.png limiting the loss to `native.max_deg` from the axis (3DGUT reads OPENCV_FISHEYE natively;
           sbatch/omni_native.sbatch)
  align    Sim3 scan <- model (the .e57 is in the scanner's own frame): FPFH/RANSAC + ICP with scale (venv-e57,
           open3d) -> <out>/pipeline/sim3_tls.json; check `fitness` (share of SfM points within 5 cm of the scan)
  geom     splat vs the TLS ground truth (splat mapped by sim3_tls.json when present): accuracy (share of opaque splat centres within --tol of the scan) and
           completeness (share of scan points within --tol of a splat centre), inside the cameras' bounding box

    python scripts/datasets/fiord.py fetch   configs/fiord_meetingroom.yaml
    python scripts/datasets/fiord.py prepare configs/fiord_meetingroom.yaml [--frames 2]
    $CEAR_WS/venv-e57/bin/python scripts/datasets/fiord.py align configs/fiord_meetingroom.yaml
    python scripts/datasets/fiord.py geom    configs/fiord_meetingroom.yaml --ply splat.ply [--tol 0.05] [--json-out J]
"""
from __future__ import annotations
import argparse, json, os, re, sys, zipfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu  # noqa: E402
import omni_common as oc  # noqa: E402


def cfg_paths(cfg):
    data = os.path.expandvars(cfg["paths"].get("data_root", "${CEAR_DATA}/fiord"))
    s = cfg["sequence"]
    return data, os.path.join(data, s["folder"]), os.path.join(cfg["paths"]["out_root"], "pipeline")


def capture_id(name, pattern):
    """Shared id of the two lenses' images of one shot (regex group 1 of `split.capture_regex`)."""
    m = re.search(pattern, os.path.basename(name))
    return m.group(1) if m else os.path.splitext(os.path.basename(name))[0]


def fetch(cfg, args):
    data, sdir, _ = cfg_paths(cfg)
    d = cfg["download"]
    z = oc.fetch(d["url"], os.path.join(data, d["file"]), d.get("md5"), d.get("bytes"))
    keep = [f"{cfg['sequence']['folder']}/{k}" for k in d.get("extract", ["fisheyeimages/", "model/"])]
    with zipfile.ZipFile(z) as zf:
        want = [n for n in zf.namelist() if any(n.startswith(k) for k in keep) and not n.endswith("/")]
        todo = [n for n in want if not os.path.exists(os.path.join(data, n))]
        for n in todo:
            zf.extract(n, data)
    print(f"[fiord] {len(want)} files ({len(todo)} new) -> {sdir}")


def prepare(cfg, args):
    data, sdir, P = cfg_paths(cfg)
    model = os.path.join(sdir, cfg["sequence"].get("model", "model"))
    sp, r, nat = cfg.get("split", {}), cfg.get("rect", {}), cfg.get("native", {})
    if args.rig:                                       # rig variant: rect_variants.<name> overrides rect.* -> rect_<name>/
        r = {**r, **cfg["rect_variants"][args.rig]}
    _, imgs, _ = oc.read_model(model)
    caps = sorted({capture_id(n, sp.get("capture_regex", r"_(\d+)_fisheye")) for n, *_ in imgs})
    every = int(sp.get("every", 8)); held = set(caps[::every])
    test = {n for n, *_ in imgs if capture_id(n, sp.get("capture_regex", r"_(\d+)_fisheye")) in held}
    frames = None
    if args.frames:
        tr = [n for n, *_ in imgs if n not in test]; te = [n for n, *_ in imgs if n in test]
        frames = tr[: args.frames] + te[: args.frames]
    views = r.get("views") or ([{"yaw": 0, "pitch": 0, "fov": 90, "name": "c"}] +
                               [{"yaw": y, "pitch": p, "fov": 70, "name": n} for y, p, n in
                                ((40, 0, "r"), (-40, 0, "l"), (0, 40, "u"), (0, -40, "d"))])
    img_dir = os.path.join(sdir, cfg["sequence"].get("images", "fisheyeimages"))
    meta = oc.prepare(P, model, lambda n: os.path.join(img_dir, n), test, {"default": views},
                      rect_scale=r.get("scale", 0.5), rect_size=r.get("size"), native_ds=int(nat.get("downsample", 2)),
                      native_max_deg=nat.get("max_deg", 90), ext=r.get("ext", ".png"), workers=args.workers, frames=frames,
                      native=not args.rig, rect_name=f"rect_{args.rig}" if args.rig else "rect", gravity=bool(r.get("gravity")))
    print(f"[fiord] {meta['n_frames']} frames ({len(caps)} captures, {len(held)} held out), median depth "
          f"{meta.get('median_depth', 0):.2f} m, valid fractions {meta['valid_fraction']}")


def read_e57(path, max_points=0, seed=0):
    """All scans of an .e57 in the file's world frame (pye57) -> xyz (N,3) float64. Without pye57 in this venv the
    small reader venv ($E57_VENV, default $CEAR_WS/venv-e57, env/make_e57_venv.sh) runs this function instead."""
    try:
        import pye57
    except ImportError:
        import subprocess, tempfile
        py = os.path.join(os.environ.get("E57_VENV", os.path.join(os.environ.get("CEAR_WS", ""), "venv-e57")), "bin", "python")
        tmp = tempfile.mktemp(suffix=".npy")
        subprocess.run([py, "-c", "import sys, numpy as np; sys.path.insert(0, sys.argv[1]); import fiord; "
                        "np.save(sys.argv[3], fiord.read_e57(sys.argv[2], int(sys.argv[4])))",
                        os.path.dirname(os.path.abspath(__file__)), path, tmp, str(max_points)], check=True)
        xyz = np.load(tmp); os.remove(tmp); return xyz
    e = pye57.E57(path); out = []
    for k in range(e.scan_count):
        d = e.read_scan(k, ignore_missing_fields=True, transform=True)
        out.append(np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1))
    xyz = np.concatenate(out)
    if max_points and len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    return xyz


def umeyama(src, dst):
    """Sim3 (s, R, t) minimising |s R src + t - dst|^2."""
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / a.var(0).sum()
    return s, R, md - s * R @ ms


def align(cfg, args):
    """Sim3 TLS <- model frame: the released .e57 is in the scanner's frame, not the COLMAP model's (and the model is
    not metric). Global registration (FPFH + RANSAC, open3d; venv-e57) of the model-frame points (--src-points, e.g.
    the rect arm's dense EDGS init, else the SfM points) to the scan over a log-spaced scale grid (--scales lo:hi:n)
    at a coarse voxel (--coarse-voxel), then point-to-point ICP alternated with Umeyama scale refits at --voxel.
    Best = highest share of source points within --voxel of the scan.
    -> <out>/pipeline/sim3_tls.json {s/scale, R, t, fitness, median_m}: x_tls = s R x_model + t;
       <out>/pipeline/tls_cloud.ply (scan, --voxel downsampled; the metric cloud for clean_haze.py --cloud)."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    data, sdir, P = cfg_paths(cfg)
    os.makedirs(P, exist_ok=True)
    if args.src_points:
        xyz = np.loadtxt(args.src_points, comments="#", usecols=(1, 2, 3)) if args.src_points.endswith(".txt") else \
            np.asarray(o3d.io.read_point_cloud(args.src_points).points)
    else:
        _, _, (xyz, _rgb) = oc.read_model(os.path.join(sdir, cfg["sequence"].get("model", "model")))
    gt_npy = os.path.join(P, "gt_tls.npy")
    if not os.path.exists(gt_npy):
        np.save(gt_npy, read_e57(os.path.join(sdir, cfg["sequence"]["tls"]), max_points=args.max_points).astype(np.float32))
    gt = np.load(gt_npy).astype(np.float64)
    c = np.median(xyz, 0); keep = np.linalg.norm(xyz - c, axis=1) < np.percentile(np.linalg.norm(xyz - c, axis=1), 95)
    src0 = xyz[keep]; vox, cv = args.voxel, args.coarse_voxel
    def pcd(X, v):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)); p = p.voxel_down_sample(v)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 6, max_nn=100))
        return p, f
    tp, tf = pcd(gt, cv); fine = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt)).voxel_down_sample(vox)
    F = np.asarray(fine.points); tree = cKDTree(F)
    o3d.io.write_point_cloud(os.path.join(P, "tls_cloud.ply"), fine)
    sub = src0[np.random.default_rng(0).choice(len(src0), min(len(src0), 300000), replace=False)]
    lo, hi, n = (float(x) for x in args.scales.split(":")); best = None
    for sc in np.geomspace(lo, hi, int(n)):
        sp, sf = pcd(sub * sc, cv)
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            sp, tp, sf, tf, True, cv * 1.5, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(cv * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 0.9999))
        T = r.transformation; s, R, t = sc, T[:3, :3], T[:3, 3]
        for it in range(40):                                     # ICP with scale (Umeyama on nearest pairs)
            X = s * sub @ R.T + t; d, j = tree.query(X)
            thr = max(cv * 2 * (0.85 ** it), vox); m = d < thr
            if m.sum() < 100:
                break
            s, R, t = umeyama(sub[m], F[j[m]])
        d = tree.query(s * sub @ R.T + t)[0]; fit = float((d < vox).mean())
        print(f"[fiord] align scale {sc:.4f}: ransac fitness {r.fitness:.3f} -> s {s:.4f}, within {vox} m {fit:.3f}, "
              f"median {np.median(d):.3f} m", flush=True)
        if best is None or fit > best["fitness"]:
            best = {"s": float(s), "scale": float(s), "R": R.tolist(), "t": t.tolist(), "fitness": fit,
                    "median_m": float(np.median(d)), "src": args.src_points or "sfm"}
    json.dump(best, open(os.path.join(P, "sim3_tls.json"), "w"), indent=1); print(json.dumps(best))


def geom(cfg, args):
    """Splat centres (opacity > --min-opacity) vs the TLS cloud, within the camera bounding box + --margin."""
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    data, sdir, P = cfg_paths(cfg)
    gt_npy = os.path.join(P, "gt_tls.npy")
    if not os.path.exists(gt_npy):
        gt = read_e57(os.path.join(sdir, cfg["sequence"]["tls"]), max_points=args.max_points)
        np.save(gt_npy, gt.astype(np.float32))
    gt = np.load(gt_npy).astype(np.float64)
    v = PlyData.read(args.ply)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    _, imgs, _ = oc.read_model(os.path.join(sdir, cfg["sequence"].get("model", "model")))
    C = np.array([x[3][:3, 3] for x in imgs])
    sj = os.path.join(P, "sim3_tls.json")                     # scan frame <- model frame (align); else identity
    if os.path.exists(sj):
        T = json.load(open(sj)); s, R, t = T["s"], np.array(T["R"]), np.array(T["t"])
        if T.get("fitness", 0) < 0.3:
            print(f"[fiord] WARNING: scan alignment fitness {T.get('fitness'):.3f} < 0.3: geometry numbers are not valid")
        xyz, C = s * xyz @ R.T + t, s * C @ R.T + t
    lo, hi = C.min(0) - args.margin, C.max(0) + args.margin
    inb = lambda X: np.all((X >= lo) & (X <= hi), 1)  # noqa: E731
    S = xyz[(op > args.min_opacity) & inb(xyz)]; G = gt[inb(gt)]
    dS = cKDTree(G).query(S, workers=-1)[0]; dG = cKDTree(S).query(G, workers=-1)[0]
    res = {"ply": args.ply, "tol_m": args.tol, "align_fitness": json.load(open(sj)).get("fitness") if os.path.exists(sj) else None, "n_splat": int(len(S)), "n_gt": int(len(G)),
           "accuracy": float((dS < args.tol).mean()), "completeness": float((dG < args.tol).mean()),
           "acc_median_m": float(np.median(dS)), "comp_median_m": float(np.median(dG))}
    res["fscore"] = 2 * res["accuracy"] * res["completeness"] / max(res["accuracy"] + res["completeness"], 1e-9)
    print(json.dumps(res))
    if args.json_out:
        json.dump(res, open(args.json_out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fetch", "prepare", "align", "geom"))
    ap.add_argument("config")
    ap.add_argument("--frames", type=int, default=0, help="prepare smoke test: only the first N train + N test frames")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rig", default=None, help="prepare: a rect_variants.<name> rig from the config -> <out>/pipeline/rect_<name>/")
    ap.add_argument("--ply", default=None); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-opacity", type=float, default=0.5); ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--max-points", type=int, default=5_000_000); ap.add_argument("--json-out", default=None)
    ap.add_argument("--voxel", type=float, default=0.05, help="align: fine (ICP / fitness) voxel, m")
    ap.add_argument("--coarse-voxel", type=float, default=0.15, help="align: FPFH/RANSAC voxel, m")
    ap.add_argument("--scales", default="0.25:1.6:15", help="align: model->scan scale grid lo:hi:n (log-spaced)")
    ap.add_argument("--src-points", default=None, help="align: model-frame points (points3D.txt or .ply); default SfM")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    {"fetch": fetch, "prepare": prepare, "align": align, "geom": geom}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
