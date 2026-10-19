#!/usr/bin/env python3
"""Oxford Spires sequence -> scan2sim datasets (CPU venv; ~10 min, ~20 GB RAM for the TLS crop: run in a CPU job,
sbatch/spires_prepare.sbatch). Inputs are fetched by scripts/datasets/spires_fetch.py (files listed in the config).

Frames: the NVS-benchmark images (novel_view_synthesis_benchmark/*/transforms_train_eval.json: `train_` / `eval_`
prefixes = their split), taken as the ORIGINAL fisheye frames (reconstruction_benchmark input/images, 1440x1080,
same timestamps) so the real Kalibr equidistant calibration applies. Poses are not SfM: every image gets
  T_world_cam = T_world_tls . T_tls_base(t) . T_base_lidar . T_cam_lidar^-1
from the TLS-registered ground-truth trajectory (gt-tum.txt, base frame, ~20 Hz, interpolated at the image stamp)
and the rig calibration (cam-lidar-imu.yaml, T_base_lidar from the config). World = the TLS frame shifted by a
rounded offset (world.json; float32-friendly, z up kept), so the TLS cloud, the LiDAR cloud and every camera share
one metric frame and any exported PLY can be scored against the TLS directly.
Splits (their protocol): train = the train-json frames minus every 10th (name order, nerfstudio's interval rule)
-> test_in (in-sequence), eval frames -> test_out (out-of-sequence: a later pass on a different route).

Writes under adapter.root (default $CEAR_OUT/spires/<seq>):
  fisheye/{train,test_in,test_out}/   COLMAP text, 3 OPENCV_FISHEYE cameras (real intrinsics + distortion),
                                      images/cam<k>_<stamp>.jpg (symlinks), points3D.txt = coloured LiDAR init
  pinhole_<variant>/{train,test_in,test_out}/   rectified virtual pinhole views (scripts/rectify_views.py; variant
                                      from adapter.rect_variant: rect_center = one view per image, rect_cube = several)
  benchmark/{test_in,test_out}/       their own undistorted 1000x5xx eval images + their COLMAP poses, Sim3-aligned
                                      to our world on all frames (the published protocol's images, exactly)
  lidar.ply (VILENS motion-undistorted keyframe clouds placed with the GT poses, cropped, voxelised),
  tls.ply (TLS ground truth, cropped to the cameras' bbox + margin, voxelised), world.json, adapter_report.json
  (counts, pose agreement with their independent COLMAP model, TLS/LiDAR sizes), checks/overlay_*.jpg (TLS points
  projected into fisheye frames: the visual pose/calibration check)
and the export_viewer.sbatch layout at paths.out_root: pipeline/onpath/{data_orig,test} -> pinhole train/test_in,
pipeline/sim3.json (identity: the datasets are metric), pipeline/depth_cloud.ply -> lidar.ply.

    python scripts/datasets/spires.py configs/spires_roq01.yaml [--stages fisheye,lidar,tls,rect,bench,layout]
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import pose_utils as pu  # noqa: E402
from rectify_views import SourceCamera, make_views, rectify_dataset, rot2qvec, qvec2rot, write_model  # noqa: E402


# ----------------------------------------------------------------------------- small utils
def T_from_txyz_qxyzw(v):
    from scipy.spatial.transform import Rotation
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix(); T[:3, 3] = v[:3]
    return T


def read_pcd(path):
    """PCD (binary or ascii) -> structured array (np.memmap for binary, no copy) with the file's fields."""
    with open(path, "rb") as f:
        hdr = {}
        while True:
            ln = f.readline().decode("ascii", "ignore").strip()
            if not ln or ln.startswith("#"):
                continue
            k, *v = ln.split(); hdr[k.upper()] = v
            if k.upper() == "DATA":
                break
        off = f.tell()
    fields, sizes, types = hdr["FIELDS"], list(map(int, hdr["SIZE"])), hdr["TYPE"]
    counts = list(map(int, hdr.get("COUNT", ["1"] * len(fields)))); n = int(hdr["POINTS"][0])
    dt = np.dtype([(fl, {"F": "f", "U": "u", "I": "i"}[t] + str(s), (c,) if c > 1 else ())
                   for fl, s, t, c in zip(fields, sizes, types, counts)])
    if hdr["DATA"][0] == "binary":
        return np.memmap(path, dtype=dt, mode="r", offset=off, shape=(n,))
    if hdr["DATA"][0] == "ascii":
        with open(path, "rb") as f:
            f.seek(off); a = np.loadtxt(f, max_rows=n, ndmin=2)
        return np.rec.fromarrays([a[:, i] for i in range(len(fields))], names=fields)
    raise SystemExit(f"{path}: compressed PCD not supported")


def pcd_points(path):
    m = read_pcd(path)
    return np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)


def voxel_down(P, vox):
    """Voxel-grid mean: one point per occupied voxel."""
    if len(P) == 0:
        return P
    key = np.floor(P / vox).astype(np.int64); key -= key.min(0); dims = key.max(0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    for k in range(3):
        out[:, k] = np.bincount(inv.ravel(), weights=P[:, k], minlength=len(cnt))
    return out / cnt[:, None]


def write_ply(path, P, rgb=None):
    P = np.asarray(P, np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\n" % len(P)).encode())
        if rgb is not None:
            f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        if rgb is None:
            f.write(P.tobytes())
        else:
            a = np.zeros(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); a["p"] = P; a["c"] = rgb
            f.write(a.tobytes())


def interp_pose(g, t, max_dt):
    """gt-tum rows (t x y z qx qy qz qw) -> 4x4 at time t (lerp + slerp), None if no sample within max_dt."""
    from scipy.spatial.transform import Rotation, Slerp
    j = np.searchsorted(g[:, 0], t)
    if j <= 0 or j >= len(g) or min(abs(g[j, 0] - t), abs(g[j - 1, 0] - t)) > max_dt:
        return None
    a, b = g[j - 1], g[j]; w = (t - a[0]) / max(b[0] - a[0], 1e-9)
    T = np.eye(4); T[:3, 3] = (1 - w) * a[1:4] + w * b[1:4]
    T[:3, :3] = Slerp([0, 1], Rotation.from_quat([a[4:8], b[4:8]]))([w]).as_matrix()[0]
    return T


def umeyama(src, dst, scale=True):
    """dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0); xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (xs ** 2).sum(1).mean() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ----------------------------------------------------------------------------- the adapter
class Spires:
    def __init__(self, cfg_path):
        self.cfg = pu.load_config(cfg_path); c = self.cfg
        self.seq = c["sequence"]["name"]; self.ad = c["adapter"]
        self.raw = os.path.join(c["sequence"]["data_root"], "raw")
        self.root = self.ad.get("root") or os.path.join(os.environ.get("CEAR_OUT", "."), "spires", self.seq)
        self.out_root = c["paths"]["out_root"]
        os.makedirs(self.root, exist_ok=True)
        self.report_path = os.path.join(self.root, "adapter_report.json")
        self.report = json.load(open(self.report_path)) if os.path.exists(self.report_path) else {}
        self._load_calib(); self._load_frames()

    def save_report(self):
        json.dump(self.report, open(self.report_path, "w"), indent=1)

    # --- calibration --------------------------------------------------------
    def _load_calib(self):
        import yaml
        rig = self.cfg["rig"]; cd = rig["calib_dir"]
        ext = yaml.safe_load(open(os.path.join(cd, "cam-lidar-imu.yaml")))
        T_base_lidar = T_from_txyz_qxyzw(rig["T_base_lidar_t_xyz_q_xyzw"])
        self.cams = {}
        for k, name in enumerate(rig["cameras"]):
            y = yaml.safe_load(open(os.path.join(cd, f"{name}.yaml")))
            assert y["distortion_model"] == "equidistant", y["distortion_model"]
            K = np.array(y["camera_matrix"]["data"]).reshape(3, 3); D = y["distortion_coefficients"]["data"]
            # Kalibr/OpenCV pixel convention (centre of the first pixel = 0) -> COLMAP (= 0.5)
            params = [K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, *D]
            T_cam_lidar = np.array(ext[name]["T_cam_lidar"], float)
            self.cams[name] = dict(id=k + 1, cam=SourceCamera("OPENCV_FISHEYE", y["image_width"], y["image_height"], params),
                                   T_base_cam=T_base_lidar @ np.linalg.inv(T_cam_lidar),
                                   dir=rig["image_dir_pattern"].format(cam=name))

    # --- frames / split -----------------------------------------------------
    def _nvs_dir(self):
        js = glob.glob(os.path.join(self.raw, "novel_view_synthesis_benchmark", "**", "transforms_train_eval.json"),
                       recursive=True)
        if not js:
            raise SystemExit("no NVS transforms_train_eval.json under raw/ (fetch the nvs zip)")
        return os.path.dirname(js[0])

    def _load_frames(self):
        self.nvs = self._nvs_dir(); d = json.load(open(os.path.join(self.nvs, "transforms_train_eval.json")))
        dir2cam = {v["dir"]: k for k, v in self.cams.items()}
        frames = []
        for fr in d["frames"]:
            camdir, base = fr["file_path"].split("/")[-2:]
            split, stamp = re.match(r"(train|eval)_(\d+\.\d+)\.jpg", base).groups()
            frames.append(dict(cam=dir2cam[camdir], stamp=stamp, t=float(stamp), split=split, nvs=fr,
                               name=f"{dir2cam[camdir]}_{stamp}.jpg"))
        # nerfstudio sorts by file path; in-sequence test = every 10th training frame (its interval rule)
        tr = sorted([f for f in frames if f["split"] == "train"], key=lambda f: f["nvs"]["file_path"])
        every = int(self.ad.get("inseq_every", 10))
        for i, f in enumerate(tr):
            f["set"] = "test_in" if i % every == 0 else "train"
        for f in frames:
            if f["split"] == "eval":
                f["set"] = "test_out"
        self.frames = frames
        g = np.loadtxt(os.path.join(self.raw, "sequences", self.seq, "processed", "trajectory", "gt-tum.txt"))
        self.gt = g[np.argsort(g[:, 0])]

    def T_tls_cam(self, f):
        Tb = interp_pose(self.gt, f["t"], float(self.ad.get("max_pose_dt", 0.03)))
        return None if Tb is None else Tb @ self.cams[f["cam"]]["T_base_cam"]

    def world(self):
        p = os.path.join(self.root, "world.json")
        if os.path.exists(p):
            return np.array(json.load(open(p))["offset_tls"])
        C = np.array([self.T_tls_cam(f)[:3, 3] for f in self.frames if self.T_tls_cam(f) is not None])
        off = np.round(C.mean(0)); off[2] = 0.0                                    # keep the TLS heights
        json.dump({"offset_tls": off.tolist(), "note": "x_world = x_tls - offset_tls (no rotation; TLS frame z-up)"},
                  open(p, "w"), indent=1)
        return off

    def image_src(self, f):
        p = os.path.join(self.raw, "reconstruction_benchmark", self.seq, "input", "images", self.cams[f["cam"]]["dir"],
                         f["stamp"] + ".jpg")
        if not os.path.exists(p):
            p = os.path.join(self.raw, "sequences", self.seq, "raw", "images", self.cams[f["cam"]]["dir"], f["stamp"] + ".jpg")
        return p

    # --- stage: fisheye datasets ---------------------------------------------
    def stage_fisheye(self):
        off = self.world(); n_skip = 0
        sets = {"train": [], "test_in": [], "test_out": []}
        for f in self.frames:
            T = self.T_tls_cam(f); src = self.image_src(f)
            if T is None or not os.path.exists(src):
                n_skip += 1; continue
            T = T.copy(); T[:3, 3] -= off; f["T_wc"] = T
            sets[f["set"]].append(f)
        cams = {v["id"]: v["cam"] for v in self.cams.values()}
        for s, fs in sets.items():
            d = os.path.join(self.root, "fisheye", s); os.makedirs(os.path.join(d, "images"), exist_ok=True)
            sp = os.path.join(d, "sparse", "0"); os.makedirs(sp, exist_ok=True)
            with open(os.path.join(sp, "cameras.txt"), "w") as fh:
                fh.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                for cid, c in sorted(cams.items()):
                    fh.write(f"{cid} {c.model} {c.width} {c.height} {' '.join(f'{p:.12g}' for p in c.params)}\n")
            with open(os.path.join(sp, "images.txt"), "w") as fh:
                fh.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                for i, f in enumerate(sorted(fs, key=lambda f: f["name"])):
                    Rcw = f["T_wc"][:3, :3].T; t = -Rcw @ f["T_wc"][:3, 3]
                    fh.write(f"{i + 1} {' '.join(f'{v:.12g}' for v in rot2qvec(Rcw))} {' '.join(f'{v:.12g}' for v in t)} "
                             f"{self.cams[f['cam']]['id']} {f['name']}\n\n")
                    dst = os.path.join(d, "images", f["name"])
                    if not os.path.lexists(dst):
                        os.symlink(self.image_src(f), dst)
        self.report["frames"] = {s: len(v) for s, v in sets.items()}; self.report["frames_skipped"] = n_skip
        self.report["cameras"] = {k: dict(id=v["id"], params=v["cam"].params, max_theta_deg=np.degrees(v["cam"].max_theta()))
                                  for k, v in self.cams.items()}
        self._pose_check(sets)
        print(f"[spires] fisheye datasets: {self.report['frames']} (skipped {n_skip})", flush=True)

    def _pose_check(self, sets):
        """Agreement with THEIR independent COLMAP model (transforms_train_eval.json, nerfstudio axes): Sim3 on the
        camera centres, then per-image rotation residual (same world up to that Sim3) -> report."""
        ours, theirs, Ro, Rt, cam = [], [], [], [], []
        for f in [f for s in sets.values() for f in s]:
            M = np.array(f["nvs"]["transform_matrix"], float)
            Rcv = M[:3, :3] @ np.diag([1, -1, -1])                         # OpenGL -> OpenCV camera axes
            ours.append(f["T_wc"][:3, 3]); theirs.append(M[:3, 3]); Ro.append(f["T_wc"][:3, :3]); Rt.append(Rcv)
            cam.append(f["cam"])
        ours, theirs = np.array(ours), np.array(theirs)
        s, R, t = umeyama(theirs, ours)
        res = np.linalg.norm((s * theirs @ R.T + t) - ours, axis=1)
        rr = np.array([rot_deg(Ro[i].T @ R @ Rt[i]) for i in range(len(Ro))])
        cam = np.array(cam)
        self.report["pose_check_vs_their_colmap"] = {
            "n": len(res), "sim3_scale": float(s), "centre_res_m": {"median": float(np.median(res)), "p95": float(np.percentile(res, 95)),
                                                                  "max": float(res.max())},
            "rot_res_deg": {"median": float(np.median(rr)), "p95": float(np.percentile(rr, 95))},
            "rot_res_deg_per_cam": {c: float(np.median(rr[cam == c])) for c in self.cams}}
        self.sim3_theirs = (s, R, t)
        json.dump({"scale": float(s), "R": R.tolist(), "t": t.tolist()}, open(os.path.join(self.root, "sim3_theirs_to_world.json"), "w"), indent=1)
        print(f"[spires] pose check vs their COLMAP: centre res median {np.median(res) * 100:.1f} cm, rot res median "
              f"{np.median(rr):.2f} deg (per cam {self.report['pose_check_vs_their_colmap']['rot_res_deg_per_cam']})", flush=True)

    # --- stage: LiDAR --------------------------------------------------------
    def _bbox(self, margin):
        C = []
        for s in ("train", "test_in", "test_out"):
            for ln in open(os.path.join(self.root, "fisheye", s, "sparse", "0", "images.txt")):
                p = ln.split()
                if len(p) >= 10 and not ln.startswith("#"):
                    R = qvec2rot(np.array(p[1:5], float)); C.append(-R.T @ np.array(p[5:8], float))
        C = np.array(C)
        return C.min(0) - margin, C.max(0) + margin

    def stage_lidar(self):
        off = self.world(); lo, hi = self._bbox(float(self.ad.get("crop_margin", 15.0)))
        cd = os.path.join(self.raw, "reconstruction_benchmark", self.seq, "input", "vilens_slam", "individual_clouds")
        if not os.path.isdir(cd):
            cd = glob.glob(os.path.join(self.raw, "sequences", self.seq, "processed", "vilens-slam", "undist-clouds", "**", ""),
                           recursive=True)[-1]
        files = sorted(glob.glob(os.path.join(cd, "cloud_*.pcd"))); pts, used = [], 0
        rmin, rmax = float(self.ad.get("lidar_min_range", 1.0)), float(self.ad.get("lidar_max_range", 60.0))
        for fp in files:
            sec, nsec = re.match(r"cloud_(\d+)_(\d+)\.pcd", os.path.basename(fp)).groups()
            T = interp_pose(self.gt, float(f"{sec}.{nsec}"), 0.06)
            if T is None:
                continue
            P = pcd_points(fp); r = np.linalg.norm(P, axis=1); P = P[(r > rmin) & (r < rmax)]   # drop the operator
            P = P @ T[:3, :3].T + T[:3, 3] - off
            m = np.all((P >= lo) & (P <= hi), axis=1)
            if m.sum() > 1000:
                pts.append(P[m]); used += 1
        P = np.concatenate(pts); vox = float(self.ad.get("lidar_voxel", 0.05))
        P = voxel_down(P, vox)
        write_ply(os.path.join(self.root, "lidar.ply"), P)
        self.report["lidar"] = {"keyframes_used": used, "keyframes_total": len(files), "points": int(len(P)), "voxel_m": vox,
                                "bbox_world": [lo.tolist(), hi.tolist()]}
        print(f"[spires] LiDAR: {used}/{len(files)} keyframes -> {len(P)} pts ({vox} m voxels)", flush=True)
        self._init_points(P)

    def _init_points(self, P):
        """points3D.txt for every fisheye split: a LiDAR subset coloured from the TRAINING fisheye frames (z-buffered
        at 1/4 resolution, mean of the frames where the point is the front surface; unseen -> grey)."""
        import cv2
        rng = np.random.default_rng(0); n = int(self.ad.get("lidar_init_points", 600000))
        X = P[rng.choice(len(P), min(n, len(P)), replace=False)] if len(P) > n else P
        acc, cnt = np.zeros((len(X), 3)), np.zeros(len(X))
        d = os.path.join(self.root, "fisheye", "train"); sp = os.path.join(d, "sparse", "0")
        from rectify_views import read_cameras, read_images
        cams = read_cameras(os.path.join(sp, "cameras.txt"))
        for iid, q, t, cid, name in read_images(os.path.join(sp, "images.txt")):
            R = qvec2rot(q); Pc = X @ R.T + t; cam = cams[cid]
            uv, ok = cam.project(Pc); rng_ = np.linalg.norm(Pc, axis=1)
            img = cv2.imread(os.path.join(d, "images", name))[:, :, ::-1]
            s = 4; u = (uv[:, 0] / s).astype(int); v = (uv[:, 1] / s).astype(int)
            ok &= (u >= 0) & (v >= 0) & (u < cam.width // s) & (v < cam.height // s) & (rng_ < 40)
            idx = np.where(ok)[0]
            zb = np.full((cam.height // s, cam.width // s), np.inf)
            np.minimum.at(zb, (v[idx], u[idx]), rng_[idx])
            vis = idx[rng_[idx] <= zb[v[idx], u[idx]] * 1.03 + 0.05]
            ui = np.clip(np.round(uv[vis, 0]).astype(int), 0, cam.width - 1); vi = np.clip(np.round(uv[vis, 1]).astype(int), 0, cam.height - 1)
            acc[vis] += img[vi, ui]; cnt[vis] += 1
        rgb = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], 128).astype(np.uint8)
        for s_ in ("train", "test_in", "test_out"):
            with open(os.path.join(self.root, "fisheye", s_, "sparse", "0", "points3D.txt"), "w") as fh:
                fh.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
                for i, (p, c) in enumerate(zip(X, rgb)):
                    fh.write(f"{i + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n")
        self.report["init_points"] = {"n": int(len(X)), "coloured_fraction": float((cnt > 0).mean())}
        print(f"[spires] init points: {len(X)} ({100 * (cnt > 0).mean():.0f}% coloured)", flush=True)

    # --- stage: TLS ----------------------------------------------------------
    def stage_tls(self):
        off = self.world(); lo, hi = self._bbox(float(self.ad.get("crop_margin", 15.0)))
        fp = os.path.join(self.raw, "reconstruction_benchmark", self.seq, "gt_cloud_merged.pcd")
        m = read_pcd(fp); n = len(m); keep = []; ch = 10_000_000
        for i in range(0, n, ch):
            b = m[i:i + ch]; P = np.stack([b["x"], b["y"], b["z"]], 1).astype(np.float64) - off
            keep.append(P[np.all((P >= lo) & (P <= hi), axis=1)])
        P = np.concatenate(keep); vox = float(self.ad.get("gt_voxel", 0.02)); n_crop = len(P)
        P = voxel_down(P, vox)
        write_ply(os.path.join(self.root, "tls.ply"), P)
        self.report["tls"] = {"source": fp, "points_total": int(n), "points_cropped": int(n_crop), "points": int(len(P)), "voxel_m": vox}
        print(f"[spires] TLS: {n} -> crop {n_crop} -> {len(P)} pts ({vox} m)", flush=True)
        self._overlays(P)

    def _overlays(self, P, n=4):
        """TLS points (z-buffered, coloured by range) drawn over a few training fisheye frames -> checks/."""
        import cv2
        from rectify_views import read_cameras, read_images
        d = os.path.join(self.root, "fisheye", "train"); sp = os.path.join(d, "sparse", "0")
        cams = read_cameras(os.path.join(sp, "cameras.txt")); ims = read_images(os.path.join(sp, "images.txt"))
        os.makedirs(os.path.join(self.root, "checks"), exist_ok=True)
        for iid, q, t, cid, name in ims[:: max(1, len(ims) // n)][:n]:
            R = qvec2rot(q); Pc = P @ R.T + t; cam = cams[cid]; uv, ok = cam.project(Pc); r = np.linalg.norm(Pc, axis=1)
            ok &= r < 30; img = cv2.imread(os.path.join(d, "images", name))
            u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int); rr = r[ok]
            zb = np.full((cam.height, cam.width), np.inf); np.minimum.at(zb, (v, u), rr)
            front = rr <= zb[v, u] * 1.02
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160)
            col = cv2.applyColorMap((np.clip(rr[front] / 20, 0, 1) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
            ov = (img * 0.5).astype(np.uint8); ov[edges > 0] = (255, 255, 255)
            ov[v[front], u[front]] = (0.6 * col + 0.4 * ov[v[front], u[front]]).astype(np.uint8)
            cv2.imwrite(os.path.join(self.root, "checks", f"overlay_{os.path.splitext(name)[0]}.jpg"), ov)

    # --- stage: rectified pinhole variants -----------------------------------
    def stage_rect(self):
        variant = self.ad.get("rect_variant", "center"); r = self.ad[f"rect_{variant}"]
        views = make_views(r["fov"], r["width"], r["height"], r.get("yaws", [0]), r.get("pitches", [0]))
        for s in ("train", "test_in", "test_out"):
            out = os.path.join(self.root, f"pinhole_{variant}", s)
            # held-out sets: always the single centre view (the frames scored in the viewer), train: the variant
            vv = views if s == "train" else make_views(**{k: self.ad["rect_center"][k] for k in ("fov", "width", "height")})
            summ = rectify_dataset(os.path.join(self.root, "fisheye", s), out, vv, ss=int(self.ad.get("rect_ss", 2)),
                                   max_invalid=float(self.ad.get("rect_max_invalid", 0.0)))
            self.report.setdefault("rect", {})[f"{variant}/{s}"] = {"n": summ["n_images"], "invalid": summ["invalid_fraction"]}

    # --- stage: their benchmark images ---------------------------------------
    def stage_bench(self):
        """Their undistorted eval images at their COLMAP poses mapped into our world (Sim3 from _pose_check)."""
        import shutil
        S = json.load(open(os.path.join(self.root, "sim3_theirs_to_world.json")))
        s, R, t = S["scale"], np.array(S["R"]), np.array(S["t"])
        for st in ("test_in", "test_out"):
            fs = sorted([f for f in self.frames if f["set"] == st], key=lambda f: f["name"])
            out = os.path.join(self.root, "benchmark", st); os.makedirs(os.path.join(out, "images"), exist_ok=True)
            cams, imgs = {}, []
            for f in fs:
                fr = f["nvs"]; k = (int(fr["w"]), int(fr["h"]), fr["fl_x"], fr["fl_y"], fr["cx"], fr["cy"])
                cid = next((c for c, kk in cams.items() if kk == k), len(cams) + 1); cams[cid] = k
                M = np.array(fr["transform_matrix"], float); Rwc = R @ M[:3, :3] @ np.diag([1, -1, -1]); C = s * R @ M[:3, 3] + t
                Rcw = Rwc.T; imgs.append((len(imgs) + 1, rot2qvec(Rcw), -Rcw @ C, cid, f["name"]))
                dst = os.path.join(out, "images", f["name"])
                if not os.path.exists(dst):
                    shutil.copyfile(os.path.join(self.nvs, fr["file_path"]), dst)
            write_model(os.path.join(out, "sparse", "0"), cams, imgs, None)
            self.report.setdefault("benchmark", {})[st] = len(imgs)
        print(f"[spires] benchmark sets: {self.report['benchmark']}", flush=True)

    # --- stage: export_viewer layout -----------------------------------------
    def stage_layout(self):
        variant = self.ad.get("rect_variant", "center"); P = os.path.join(self.out_root, "pipeline")
        os.makedirs(os.path.join(P, "onpath"), exist_ok=True)
        train = os.path.join(self.root, f"pinhole_{variant}", "train"); D = os.path.join(P, "onpath", "data_orig")
        if self.ad.get("layout_own_train") and not os.path.lexists(D):
            # a private training dir (model copied, images linked per file) that per-run sidecars (LiDAR
            # <stem>_depth.npy targets) can be written into without touching the shared rectified set
            import shutil
            shutil.copytree(os.path.join(train, "sparse"), os.path.join(D, "sparse")); os.makedirs(os.path.join(D, "images"))
            for f in os.listdir(os.path.join(train, "images")):
                os.symlink(os.path.join(train, "images", f), os.path.join(D, "images", f))
        for link, tgt in ((D, train),
                          (os.path.join(P, "onpath", "test"), os.path.join(self.root, f"pinhole_{variant}", "test_in")),
                          (os.path.join(P, "depth_cloud.ply"), os.path.join(self.root, "lidar.ply"))):
            if not os.path.lexists(link):
                os.symlink(tgt, link)
        json.dump({"scale": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "note": "spires datasets are metric (world.json)"},
                  open(os.path.join(P, "sim3.json"), "w"), indent=1)
        print(f"[spires] export_viewer layout at {P}", flush=True)


STAGES = ("fisheye", "lidar", "tls", "rect", "bench", "layout")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args()
    S = Spires(a.config)
    for st in a.stages.split(","):
        t0 = time.time(); getattr(S, f"stage_{st}")(); S.save_report()
        print(f"[spires] stage {st} done in {time.time() - t0:.0f} s", flush=True)
    print("SPIRES_ADAPTER_DONE")


if __name__ == "__main__":
    main()
