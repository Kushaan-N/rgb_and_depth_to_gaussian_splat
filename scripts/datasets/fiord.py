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


def _rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (np.linalg.norm(v) ** 2))


def align(cfg, args):
    """Sim3 TLS <- model frame. The released .e57 is in the scanner's frame (levelled: +z up) and the COLMAP model is
    neither metric nor levelled, so: (1) level the model with its world up (mean camera up, -y in each camera) and
    match the floors (2nd height percentile); (2) brute-force scale (--scales lo:hi:n, log grid) x yaw (2 deg) and find
    the horizontal shift by FFT cross-correlation of 2D wall-occupancy maps (points between floor + 0.3 and floor + 2 m
    in scan units, --cell m grid); (3) point-to-point ICP alternated with Umeyama scale refits on the best few
    candidates. Source points: --src-points (e.g. the rect arm's dense EDGS init) else the SfM points.
    Best = highest share of source points within 2 x --voxel of the scan (`fitness`; SfM/EDGS points are noisy: on
    FIORD MeetingRoom a correct alignment of the SfM points has 16% within 5 cm, 24% within 10 cm, walls coincide).
    -> <out>/pipeline/sim3_tls.json {s/scale, R, t, fitness, median_m}: x_tls = s R x_model + t;
       <out>/pipeline/tls_cloud.ply (scan at --voxel; the metric cloud for clean_haze.py --cloud)."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    data, sdir, P = cfg_paths(cfg)
    os.makedirs(P, exist_ok=True)
    _, imgs, (sfm, _rgb) = oc.read_model(os.path.join(sdir, cfg["sequence"].get("model", "model")))
    if args.src_points and args.src_points.endswith(".ply"):          # a trained splat: its opaque centres
        from plyfile import PlyData
        v = PlyData.read(args.src_points)["vertex"]; op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)[op > 0.5]
    else:
        xyz = np.loadtxt(args.src_points, comments="#", usecols=(1, 2, 3)) if args.src_points else sfm
    gt_npy = os.path.join(P, "gt_tls.npy")
    if not os.path.exists(gt_npy):
        np.save(gt_npy, read_e57(os.path.join(sdir, cfg["sequence"]["tls"]), max_points=args.max_points).astype(np.float32))
    gt = np.load(gt_npy).astype(np.float64)
    vox = args.voxel; fine = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt)).voxel_down_sample(vox)
    o3d.io.write_point_cloud(os.path.join(P, "tls_cloud.ply"), fine); F = np.asarray(fine.points); tree = cKDTree(F)
    rng = np.random.default_rng(0)
    src = xyz[rng.choice(len(xyz), min(len(xyz), 300000), replace=False)]
    up = np.mean([c[:3, :3] @ np.array([0, -1.0, 0]) for *_, c in imgs], 0)
    R0 = _rot_a_to_b(up, np.array([0, 0, 1.0])); L = src @ R0.T            # levelled model, z up
    fl_m, fl_s = np.percentile(L[:, 2], 2), np.percentile(F[:, 2], 2)
    cell = args.cell
    def occ(XY, lo, shape):
        ij = np.floor((XY - lo) / cell).astype(int); ok = np.all((ij >= 0) & (ij < shape), 1)
        g = np.zeros(shape); np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0); return np.minimum(g, 3.0)
    band = lambda Z, fl, k: (Z > fl + 0.3 * k) & (Z < fl + 2.0 * k)  # noqa: E731
    Fw = F[band(F[:, 2], fl_s, 1.0)]; lo = Fw[:, :2].min(0) - 2.0; hi = Fw[:, :2].max(0) + 2.0
    shape = tuple(np.ceil((hi - lo) / cell).astype(int) * 2)
    G = occ(Fw[:, :2], lo, shape); FG = np.conj(np.fft.rfft2(G))
    lo_s, hi_s, n = (float(x) for x in args.scales.split(":")); cands = []
    for sc in np.geomspace(lo_s, hi_s, int(n)):
        Lw = L[band(L[:, 2], fl_m, 1.0 / sc)] * sc
        for yaw in np.radians(np.arange(0, 360, 2.0)):
            Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]); XY = Lw[:, :2] @ Rz.T
            c0 = XY.mean(0); M = occ(XY - c0 + (hi - lo) / 2 + lo, lo, shape)
            corr = np.fft.irfft2(np.fft.rfft2(M) * FG, s=shape); k = np.unravel_index(np.argmax(corr), shape)
            sh = np.array([(k[0] + shape[0] // 2) % shape[0] - shape[0] // 2, (k[1] + shape[1] // 2) % shape[1] - shape[1] // 2])
            score = corr[k] / max(M.sum(), 1)
            t2 = -c0 + (hi - lo) / 2 + lo - sh * cell
            cands.append((score, sc, yaw, t2))
    cands.sort(key=lambda x: -x[0]); best = None; pick = []
    for c in cands:                                   # distinct hypotheses (rooms are often 180-deg symmetric)
        if all(abs(np.log(c[1] / q[1])) > 0.05 or abs((np.degrees(c[2] - q[2]) + 180) % 360 - 180) > 20 for q in pick):
            pick.append(c)
        if len(pick) >= args.top:
            break
    for score, sc, yaw, t2 in pick:
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        s, R = sc, Rz @ R0; t = np.array([t2[0], t2[1], fl_s - sc * fl_m])
        for it in range(40):                                     # ICP with scale (Umeyama on nearest pairs)
            X = s * src @ R.T + t; d, j = tree.query(X)
            m = d < max(0.5 * (0.85 ** it), vox)
            if m.sum() < 100:
                break
            s, R, t = umeyama(src[m], F[j[m]])
        d = tree.query(s * src @ R.T + t)[0]; fit = float((d < 2 * vox).mean())
        if args.trim:                                   # score on the best-fitting share (glass, clutter)
            fit = float(np.mean(np.sort(d)[: int(len(d) * args.trim)] < 2 * vox))
        print(f"[fiord] align cand corr {score:.3f} scale {sc:.3f} yaw {np.degrees(yaw):.0f}: -> s {s:.4f}, within "
              f"{vox}/{2 * vox} m {(d < vox).mean():.3f}/{fit:.3f}, median {np.median(d):.3f} m", flush=True)
        if best is None or fit > best["fitness"]:
            best = {"s": float(s), "scale": float(s), "R": R.tolist(), "t": t.tolist(), "fitness": fit,
                    "fitness_def": f"share of source points within {2 * vox} m of the scan",
                    "within_voxel": float((d < vox).mean()), "median_m": float(np.median(d)), "src": args.src_points or "sfm"}
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
        if T.get("fitness", 0) < 0.2:
            print(f"[fiord] WARNING: scan alignment fitness {T.get('fitness'):.3f} < 0.2: geometry numbers are not valid")
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
    ap.add_argument("--cell", type=float, default=0.05, help="align: 2D occupancy cell, m")
    ap.add_argument("--top", type=int, default=8, help="align: distinct (scale, yaw) candidates refined by ICP")
    ap.add_argument("--trim", type=float, default=0.0, help="align: score candidates on this best-fitting share of points (0 = all)")
    ap.add_argument("--scales", default="0.2:1.2:40", help="align: model->scan scale grid lo:hi:n (log-spaced)")
    ap.add_argument("--src-points", default=None, help="align: model-frame points: points3D.txt, or a splat .ply (its "
                    "opaque centres; cleaner surfaces than SfM/EDGS points); default the SfM points")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    {"fetch": fetch, "prepare": prepare, "align": align, "geom": geom}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
