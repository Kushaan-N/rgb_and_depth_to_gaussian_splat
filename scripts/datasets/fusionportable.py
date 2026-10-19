"""FusionPortable v2 (multi-camera rig + Ouster LiDAR + GT) -> the pinhole COLMAP text datasets the CEAR splat recipe trains on.

No ROS: the ROS1 bag is read with the `rosbags` package (CPU venv). Every step is config-driven (configs/fp2_*.yaml,
base configs/fp2_base.yaml) and generic over the rig: `rig.cameras` lists any number of frame cameras, each with its
calibration yaml (K + plumb_bob distortion + extrinsic to the reference camera), all treated as ONE rigid rig.

  extract   bag -> <out>/extract/: every camera image (raw JPEG bytes, header stamp ns) + per-image sharpness
            (variance of Laplacian, half-res grey) + exposure/gain metadata if the bag carries it; every
            `lidar.every`-th LiDAR scan as float32 (x, y, z, t_s) in the sensor frame (t_s = absolute point time,
            from the per-point ns offset and the end-of-scan stamp), range-gated.
  prepare   keyframes: one rig frame per 1/select.fps s slot, the one whose WORST camera is sharpest (legged gait /
            handheld motion blur); cameras paired by nearest stamp (|dt| < 5 ms). Undistorted to pinhole per camera
            (cv2.getOptimalNewCameraMatrix, undistort.alpha) -> <P>/colmap/images/<cam>/<ts>.png + cameras.json.
  sfm       rig-constrained COLMAP (pycolmap 4.x): calibrated pinhole cameras, the calibrated camera-to-camera extrinsics
            as a fixed rig (rig.refine to let BA refine them), pairs = sequential (poses.overlap keyframes) UNION
            spatial (keyframes whose GT positions are within poses.spatial_m — loop closures); then a similarity to the
            GT positions (Umeyama + the constant reference-point lever arm in the camera frame + a GT clock offset,
            all solved), so the reconstruction is METRIC in the GT frame (sim3.json = identity). GT that is position
            only (room sequences: Leica MS60 prism) fixes scale/gauge; orientation comes from SfM. Writes
            <P>/dataset (sparse/0 text, images/ symlinks named <cam>_<ts>.png) + <P>/sfm_align.json.
  lidar     static LiDAR cloud in the same metric frame: each scan placed by the rig pose interpolated at every
            point's own time (deskew) composed with the LiDAR->camera extrinsic (direction measured: the one whose
            cloud the SfM points lie on), refined per scan by point-to-plane ICP against the first-pass map, then
            free-space carving on the organised range images removes moving people / the operator (a voxel some
            later scan sees THROUGH is dynamic). -> <P>/lidar_static.ply (+ depth_cloud.ply link, lidar.json).
  split     <P>/onpath/{train,test}: every split.test_every-th keyframe (all its cameras) held out, like CEAR's on-path
            benchmark; points3D.txt = split.init_points LiDAR points coloured from the nearest keyframe that sees
            them; <stem>_depth.npy LiDAR ray-distance targets via scripts/build_depth_targets.py if --depth.
  all       extract + prepare + sfm + lidar + split.

    python scripts/datasets/fusionportable.py all configs/fp2_legged_room00.yaml
    python scripts/datasets/fusionportable.py sfm configs/fp2_legged_room00.yaml --force

Cameras are global-shutter pinhole + mild radial distortion (FLIR BFS, 83 deg HFOV), so no fisheye path is needed;
scripts/rectify_views.py is therefore not used. Out layout: <paths.out_root>/{extract,pipeline}.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, shutil, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pose_utils as pu  # noqa: E402

NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))


# ------------------------------------------------------------------ small geometry helpers
def qwxyz_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_qwxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def T_of(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def interp_poses(ts, Ts, tq):
    """SE(3) interpolation (slerp + linear) of world-from-X poses Ts (N,4,4) at times ts, queried at tq (clamped)."""
    from scipy.spatial.transform import Rotation, Slerp
    tq = np.clip(np.asarray(tq, float), ts[0], ts[-1])
    R = Slerp(ts, Rotation.from_matrix(Ts[:, :3, :3]))(tq).as_matrix()
    t = np.stack([np.interp(tq, ts, Ts[:, i, 3]) for i in range(3)], 1)
    out = np.tile(np.eye(4), (len(tq), 1, 1)); out[:, :3, :3] = R; out[:, :3, 3] = t
    return out


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd(D.T @ S / len(src)); W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt; s = np.trace(np.diag(d) @ W) / ((S ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


# ------------------------------------------------------------------ config / calibration
class Ctx:
    def __init__(self, cfg_path):
        self.cfg = c = pu.load_config(cfg_path)
        self.seq = c["sequence"]["name"]; self.root = c["sequence"]["data_root"]
        self.out = c["paths"]["out_root"]; self.ex = os.path.join(self.out, "extract"); self.P = os.path.join(self.out, "pipeline")
        self.rig = c["rig"]; self.cams = list(self.rig["cameras"]); self.ref = self.rig.get("ref_camera", self.cams[0])
        self.cdir = os.path.join(c["download"]["calib_dir"], self.rig["calib_release"], "calib")
        os.makedirs(self.P, exist_ok=True)

    def get(self, dotted, default=None):
        v = self.cfg
        for k in dotted.split("."):
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                return default
        return v

    def calib(self, name):
        import yaml
        return yaml.safe_load(open(os.path.join(self.cdir, name + ".yaml")))

    def intr(self, cam):
        y = self.calib(cam)
        K = np.array(y["camera_matrix"]["data"], float).reshape(3, 3)
        D = np.array(y["distortion_coefficients"]["data"], float)
        return K, D, int(y["image_width"]), int(y["image_height"]), y.get("distortion_model", "plumb_bob")

    def T_cam_ref(self, cam):
        """T mapping reference-camera coordinates into `cam` (identity for the reference)."""
        if cam == self.ref:
            return np.eye(4)
        y = self.calib(cam)
        T = T_of(qwxyz_to_R(y["quaternion_stereo"]["data"]), y["translation_stereo"]["data"])
        return T if self.rig.get("stereo_direction", "ref_to_cam") == "ref_to_cam" else np.linalg.inv(T)

    def T_lidar_yaml(self):
        y = self.calib("ouster00")
        return T_of(qwxyz_to_R(y[f"quaternion_sensor_{self.ref}"]["data"]), y[f"translation_sensor_{self.ref}"]["data"])


def load_tum(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1:4], a[:, 4:8]          # t, xyz, q(xyzw)


# ------------------------------------------------------------------ extract
def pc2_array(msg):
    """sensor_msgs/PointCloud2 -> structured numpy array (fields by name)."""
    tmap = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    names, fmts, offs = [], [], []
    for f in msg.fields:
        names.append(f.name); fmts.append(("<" if not msg.is_bigendian else ">") + tmap[f.datatype]); offs.append(f.offset)
    dt = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def cmd_extract(cx, args):
    from pathlib import Path
    import cv2
    from rosbags.highlevel import AnyReader
    bag = os.path.join(cx.root, cx.seq + ".bag")
    done = os.path.join(cx.ex, "meta.json")
    if os.path.exists(done) and not args.force:
        print(f"[fp2] extract: have {done}"); return
    topics = cx.rig["topics"]; tcam = {v: k for k, v in topics.items()}
    L = cx.cfg["lidar"]; ltopic = L["topic"]; every = int(L.get("every", 2))
    rmin, rmax = float(L.get("min_range", 0.5)), float(L.get("max_range", 50))
    for c in cx.cams:
        os.makedirs(os.path.join(cx.ex, c), exist_ok=True)
    os.makedirs(os.path.join(cx.ex, "lidar"), exist_ok=True)
    rows, nl, li, t0 = [], 0, 0, time.time()
    with AnyReader([Path(bag)]) as r:
        conns_all = {c.topic: c for c in r.connections}
        # per-image camera metadata (FP2: <cam ns>/image_chunk_data = PointStamped, x = exposure ms, y = gain dB)
        meta_topics = [t for t in conns_all if any(t.startswith(v.rsplit("/", 2)[0]) for v in topics.values())
                       and any(k in t.lower() for k in ("exposure", "chunk", "metadata", "gain"))]
        want = [conns_all[t] for t in list(topics.values()) + [ltopic] + meta_topics if t in conns_all]
        missing = [t for t in list(topics.values()) + [ltopic] if t not in conns_all]
        if missing:
            raise SystemExit(f"[fp2] topics missing from bag: {missing}; bag has {sorted(conns_all)}")
        meta = {}
        for conn, bt, raw in r.messages(connections=want):
            msg = r.deserialize(raw, conn.msgtype)
            st = msg.header.stamp if hasattr(msg, "header") else None
            ns = (st.sec * 1_000_000_000 + st.nanosec) if st is not None else bt
            if conn.topic in tcam:
                cam = tcam[conn.topic]; data = bytes(msg.data)
                ext = ".png" if "png" in (getattr(msg, "format", "") or "").lower() else ".jpg"
                fn = os.path.join(cx.ex, cam, f"{ns}{ext}")
                with open(fn, "wb") as f:
                    f.write(data)
                g = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
                rows.append((cam, ns, os.path.basename(fn), float(cv2.Laplacian(g, cv2.CV_32F).var()), float(g.mean())))
            elif conn.topic == ltopic:
                li += 1
                if (li - 1) % every:
                    continue
                a = pc2_array(msg)
                xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
                rng = np.linalg.norm(xyz, axis=1)
                tt = a["t"].astype(np.float64) * 1e-9 if "t" in a.dtype.names else np.zeros(len(a))
                ok = np.isfinite(rng) & (rng > rmin) & (rng < rmax)
                tabs = ns * 1e-9 - (tt.max() - tt) if L.get("stamp", "end") == "end" else ns * 1e-9 + tt
                ring = a["ring"].astype(np.float32) if "ring" in a.dtype.names else np.zeros(len(a), np.float32)
                np.save(os.path.join(cx.ex, "lidar", f"{ns}.npy"),
                        np.concatenate([xyz[ok], (tabs[ok] - ns * 1e-9)[:, None].astype(np.float32), ring[ok, None]], 1))
                nl += 1
            else:
                src = msg.point if hasattr(msg, "point") else msg
                meta.setdefault(conn.topic, []).append((ns, {k: getattr(src, k) for k in dir(src)
                                                               if not k.startswith("_") and isinstance(getattr(src, k), (int, float))}))
            if (len(rows) + nl) % 2000 == 0:
                print(f"[fp2] extract: {len(rows)} images, {nl} scans, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(cx.ex, "images.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cam", "ns", "file", "sharp", "mean"]); w.writerows(rows)
    json.dump({"bag": bag, "images": len(rows), "scans": nl, "lidar_every": every,
               "meta_topics": {k: v[:3] for k, v in meta.items()}}, open(done, "w"), indent=1, default=str)
    if meta:
        json.dump(meta, open(os.path.join(cx.ex, "image_meta.json"), "w"), default=str)
        for k, v in meta.items():
            for f in sorted(v[0][1]):
                a = np.array([d[f] for _, d in v], float)
                print(f"[fp2] meta {k}.{f}: median {np.median(a):.4g}, range {a.min():.4g}..{a.max():.4g}")
    print(f"[fp2] extract: {len(rows)} images, {nl} LiDAR scans -> {cx.ex}")


# ------------------------------------------------------------------ prepare (keyframes + undistort)
def read_images_csv(cx):
    by = {c: [] for c in cx.cams}
    for r in csv.DictReader(open(os.path.join(cx.ex, "images.csv"))):
        if r["cam"] in by:
            by[r["cam"]].append((int(r["ns"]), r["file"], float(r["sharp"]), float(r["mean"])))
    return {c: sorted(v) for c, v in by.items()}


def rig_frames(cx, tol_ns=5_000_000):
    """[(ref_ns, {cam: (ns, file, sharp, mean)})] with every camera present within tol of the reference stamp."""
    by = read_images_csv(cx); out = []
    arr = {c: np.array([x[0] for x in v], np.int64) for c, v in by.items()}
    for rec in by[cx.ref]:
        fr = {cx.ref: rec}
        for c in cx.cams:
            if c == cx.ref or not len(arr[c]):
                continue
            j = int(np.clip(np.searchsorted(arr[c], rec[0]), 1, len(arr[c]) - 1))
            j = j if abs(arr[c][j] - rec[0]) < abs(arr[c][j - 1] - rec[0]) else j - 1
            if abs(arr[c][j] - rec[0]) <= tol_ns:
                fr[c] = by[c][j]
        if len(fr) == len(cx.cams):
            out.append((rec[0], fr))
    return out


def cmd_prepare(cx, args):
    import cv2
    S = cx.cfg["select"]; frames = rig_frames(cx)
    if not frames:
        raise SystemExit("[fp2] no complete rig frames")
    ts = np.array([f[0] for f in frames]) * 1e-9
    # GT coverage: keep only frames inside the GT time span (poses are aligned to it)
    gtf = os.path.join(cx.root, cx.get("gt.file", "gt_traj.txt"))
    if os.path.exists(gtf):
        tg = load_tum(gtf)[0]; keep = (ts >= tg[0]) & (ts <= tg[-1])
        frames = [f for f, k in zip(frames, keep) if k]; ts = ts[keep]
    slot = 1.0 / float(S.get("fps", 4.0)); sel = []
    score = np.array([min(f[1][c][2] for c in cx.cams) for f in frames])
    for s0 in np.arange(ts[0], ts[-1], slot):
        idx = np.where((ts >= s0) & (ts < s0 + slot))[0]
        if len(idx):
            sel.append(int(idx[np.argmax(score[idx])]) if S.get("sharpest_in_window", True) else int(idx[0]))
    if len(sel) > int(S.get("max_frames", 10 ** 9)):
        sel = list(np.array(sel)[np.linspace(0, len(sel) - 1, int(S["max_frames"])).round().astype(int)])
    # blur report: chosen keyframes vs all frames (relative to the local median sharpness)
    from scipy.ndimage import median_filter
    rel = score / np.maximum(median_filter(score, size=41, mode="nearest"), 1e-6)
    print(f"[fp2] prepare: {len(frames)} rig frames, {len(sel)} keyframes @ {1 / slot:.1f} fps; sharpness rel. to local "
          f"median: all {np.median(rel):.2f}, keyframes {np.median(rel[sel]):.2f}; worst-decile frames avoided "
          f"{100 * np.mean(rel[sel] > np.quantile(rel, 0.1)):.0f}%", flush=True)
    alpha = float(cx.get("undistort.alpha", 0.0)); camj = {}
    img_root = os.path.join(cx.P, "colmap", "images")
    for c in cx.cams:
        K, D, W, H, model = cx.intr(c)
        if model not in ("plumb_bob", "radtan", "radial-tangential"):
            raise SystemExit(f"[fp2] {c}: distortion model {model} needs the fisheye path (scripts/rectify_views.py)")
        Kn, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha, (W, H))
        Kn[0, 0] = Kn[1, 1] = 0.5 * (Kn[0, 0] + Kn[1, 1])          # square pixels
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Kn, (W, H), cv2.CV_32FC1)
        camj[c] = {"K": Kn.tolist(), "W": W, "H": H, "K_raw": K.tolist(), "D": D.tolist(),
                   "T_cam_ref": cx.T_cam_ref(c).tolist()}
        os.makedirs(os.path.join(img_root, c), exist_ok=True)
        for i in sel:
            ns, fn = frames[i][1][c][0], frames[i][1][c][1]
            dst = os.path.join(img_root, c, f"{frames[i][0]}.png")
            if os.path.exists(dst) and not args.force:
                continue
            im = cv2.imread(os.path.join(cx.ex, c, fn), cv2.IMREAD_COLOR)
            if im.shape[1] != W or im.shape[0] != H:
                raise SystemExit(f"[fp2] {c}: image {im.shape[1]}x{im.shape[0]} != calibration {W}x{H} (wrong calib_release?)")
            cv2.imwrite(dst, cv2.remap(im, m1, m2, cv2.INTER_LANCZOS4))
    kf = [{"ref_ns": int(frames[i][0]), "t": float(ts[i]), "sharp_rel": float(rel[i]),
           "cams": {c: {"ns": int(frames[i][1][c][0]), "sharp": frames[i][1][c][2], "mean": frames[i][1][c][3]} for c in cx.cams}}
          for i in sel]
    json.dump({"cameras": camj, "keyframes": kf}, open(os.path.join(cx.P, "cameras.json"), "w"), indent=1)
    print(f"[fp2] prepare: undistorted {len(sel)} x {len(cx.cams)} images -> {img_root}")


# ------------------------------------------------------------------ sfm (rig COLMAP) + GT alignment
def gt_positions_at(cx, t):
    tg, pg, _ = load_tum(os.path.join(cx.root, cx.get("gt.file", "gt_traj.txt")))
    return np.stack([np.interp(t, tg, pg[:, i], left=np.nan, right=np.nan) for i in range(3)], 1)


def align_to_gt(cx, C, Rwc, t, iters=8):
    """Similarity world_gt <- world_sfm with the GT reference point at a constant lever arm l (camera frame, metres)
    and a GT clock offset dt: g(t + dt) = s R C + R Rwc l + t0. Returns dict + per-frame residuals."""
    best = None
    for dt in np.arange(-0.30, 0.3001, 0.01):
        G = gt_positions_at(cx, t + dt); ok = np.isfinite(G).all(1)
        if ok.sum() < 10:
            continue
        s, R, t0 = umeyama(C[ok], G[ok]); l = np.zeros(3)
        for _ in range(iters):
            # linear in (t0, l) for fixed s, R: G - s R C = t0 + (R Rwc) l
            A = np.concatenate([np.tile(np.eye(3), (ok.sum(), 1)), np.concatenate([R @ Rwc[i] for i in np.where(ok)[0]], 0)], 1)
            b = (G[ok] - s * C[ok] @ R.T).reshape(-1)
            x = np.linalg.lstsq(A, b, rcond=None)[0]; l = x[3:]
            Cl = C[ok] + np.einsum("nij,j->ni", Rwc[ok], l) / s       # reference point in SfM units
            s, R, t0 = umeyama(Cl, G[ok])
        Cl = C + np.einsum("nij,j->ni", Rwc, l) / s
        res = np.linalg.norm(s * Cl @ R.T + t0 - G, axis=1)
        med = float(np.nanmedian(res))
        if best is None or med < best["median_m"]:
            best = {"dt": float(dt), "scale": float(s), "R": R, "t": t0, "lever_cam_m": l, "median_m": med, "res": res}
    return best


def cmd_sfm(cx, args):
    import pycolmap
    camj = json.load(open(os.path.join(cx.P, "cameras.json"))); kfs = camj["keyframes"]; cams = camj["cameras"]
    work = os.path.join(cx.P, "colmap"); img_root = os.path.join(work, "images"); db = os.path.join(work, "database.db")
    out_ds = os.path.join(cx.P, "dataset")
    if os.path.exists(os.path.join(out_ds, "sparse", "0", "images.txt")) and not args.force:
        print(f"[fp2] sfm: have {out_ds}"); return
    if os.path.exists(db):
        os.remove(db)
    names = {c: [f"{c}/{k['ref_ns']}.png" for k in kfs] for c in cx.cams}
    for c in cx.cams:
        K = np.array(cams[c]["K"])
        ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"
        ro.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        eo = pycolmap.FeatureExtractionOptions(); eo.num_threads = NCPU; eo.sift.max_num_features = 8192
        pycolmap.extract_features(db, img_root, image_names=names[c], camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo, device=pycolmap.Device.cpu)
    # rig: reference camera + fixed calibrated extrinsics
    rcs = []
    for c in cx.cams:
        T = np.array(cams[c]["T_cam_ref"])
        rc = pycolmap.RigConfigCamera(); rc.image_prefix = c + "/"; rc.ref_sensor = (c == cx.ref)
        if c != cx.ref:
            rc.cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(T[:3, :3]), T[:3, 3])
        rcs.append(rc)
    rcfg = pycolmap.RigConfig(); rcfg.cameras = rcs
    dbh = pycolmap.Database.open(db)
    pycolmap.apply_rig_config([rcfg], dbh); dbh.close()
    # pairs: sequential over keyframes (all camera combinations) + spatial from GT positions (loop closures)
    t = np.array([k["t"] for k in kfs]); n = len(kfs); ov = int(cx.get("poses.overlap", 15)); pairs = set()
    for i in range(n):
        for j in range(i, min(n, i + ov + 1)):
            pairs.add((i, j))
    G = gt_positions_at(cx, t) if os.path.exists(os.path.join(cx.root, cx.get("gt.file", "gt_traj.txt"))) else None
    if G is not None:
        from scipy.spatial import cKDTree
        ok = np.isfinite(G).all(1); idx = np.where(ok)[0]; tree = cKDTree(G[ok])
        rad = float(cx.get("poses.spatial_m", 1.5)); kmax = int(cx.get("poses.spatial_k", 12))
        for a, nb in enumerate(tree.query_ball_point(G[ok], rad)):
            i = idx[a]; nb = sorted((idx[b] for b in nb if abs(idx[b] - i) > ov), key=lambda j: np.linalg.norm(G[j] - G[i]))
            for j in nb[:kmax]:
                pairs.add((min(i, j), max(i, j)))
    pl = []
    for i, j in sorted(pairs):
        for ca in cx.cams:
            for cb in cx.cams:
                if i == j and ca >= cb:
                    continue
                pl.append(f"{names[ca][i]} {names[cb][j]}")
    plf = os.path.join(work, "pairs.txt"); open(plf, "w").write("\n".join(pl) + "\n")
    print(f"[fp2] sfm: {n} rig frames x {len(cx.cams)} cams, {len(pl)} image pairs ({len(pairs)} frame pairs)", flush=True)
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = NCPU
    po = pycolmap.ImportedPairingOptions(); po.match_list_path = plf
    pycolmap.match_image_pairs(db, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False; opts.ba_refine_principal_point = False; opts.ba_refine_extra_params = False
    opts.ba_refine_sensor_from_rig = bool(cx.get("rig.refine", False)); opts.num_threads = NCPU
    sp = os.path.join(work, "sparse"); shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
    recs = pycolmap.incremental_mapping(db, img_root, sp, options=opts)
    if not recs:
        raise SystemExit("[fp2] sfm: no reconstruction")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"[fp2] sfm: {len(recs)} model(s); best registers {rec.num_reg_images()}/{n * len(cx.cams)} images, "
          f"{rec.num_points3D()} points, mean reproj {rec.compute_mean_reprojection_error():.3f}px", flush=True)
    # align the reference camera trajectory to GT positions
    byname = {im.name: im for im in rec.images.values()}
    ref_i = [i for i in range(n) if names[cx.ref][i] in byname]
    Cs = np.array([byname[names[cx.ref][i]].projection_center() for i in ref_i])
    Rwc = np.array([byname[names[cx.ref][i]].cam_from_world().rotation.matrix().T for i in ref_i])
    if G is None:
        raise SystemExit("[fp2] sfm: no GT trajectory to fix metric scale")
    al = align_to_gt(cx, Cs, Rwc, t[ref_i])
    print(f"[fp2] sfm: GT alignment scale {al['scale']:.4f}, clock offset {al['dt'] * 1e3:+.0f} ms, lever arm "
          f"{np.round(al['lever_cam_m'], 3).tolist()} m (cam frame), residual median {al['median_m'] * 100:.1f} cm, "
          f"p90 {np.nanpercentile(al['res'], 90) * 100:.1f} cm", flush=True)
    rec.transform(pycolmap.Sim3d(al["scale"], pycolmap.Rotation3d(al["R"]), al["t"]))
    raw = os.path.join(work, "metric"); os.makedirs(raw, exist_ok=True); rec.write(raw)
    # dataset: flat names <cam>_<ns>.png, poses (world-to-camera), SfM points (replaced by LiDAR in `split`)
    write_dataset(cx, rec, names, kfs, cams, out_ds, img_root)
    json.dump({"registered": rec.num_reg_images(), "images": n * len(cx.cams), "points": rec.num_points3D(),
               "scale": al["scale"], "R": al["R"].tolist(), "t": al["t"].tolist(), "gt_clock_offset_s": al["dt"],
               "lever_cam_m": al["lever_cam_m"].tolist(), "gt_residual_median_m": al["median_m"],
               "gt_residual_p90_m": float(np.nanpercentile(al["res"], 90))},
              open(os.path.join(cx.P, "sfm_align.json"), "w"), indent=1)
    json.dump({"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
               "note": "FP2 datasets are written in the metric GT frame"}, open(os.path.join(cx.P, "sim3.json"), "w"), indent=1)


def write_dataset(cx, rec, names, kfs, cams, out_ds, img_root):
    s0 = os.path.join(out_ds, "sparse", "0"); im_dir = os.path.join(out_ds, "images")
    os.makedirs(s0, exist_ok=True); os.makedirs(im_dir, exist_ok=True)
    cid = {c: i + 1 for i, c in enumerate(cx.cams)}
    with open(os.path.join(s0, "cameras.txt"), "w") as f:
        for c in cx.cams:
            K = np.array(cams[c]["K"]); f.write(f"{cid[c]} PINHOLE {cams[c]['W']} {cams[c]['H']} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    byname = {im.name: im for im in rec.images.values()}; lines = []; k = 0
    for c in cx.cams:
        for i, kf in enumerate(kfs):
            im = byname.get(names[c][i])
            if im is None or not im.has_pose:
                continue
            T = im.cam_from_world(); q = R_to_qwxyz(T.rotation.matrix()); tr = T.translation; k += 1
            nm = f"{c}_{kf['ref_ns']}.png"
            lines.append(f"{k} {q[0]} {q[1]} {q[2]} {q[3]} {tr[0]} {tr[1]} {tr[2]} {cid[c]} {nm}\n\n")
            dst = os.path.join(im_dir, nm)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(img_root, names[c][i]), dst)
    open(os.path.join(s0, "images.txt"), "w").write("".join(lines))
    with open(os.path.join(s0, "points3D.txt"), "w") as f:
        for pid, p in rec.points3D.items():
            x = p.xyz; cc = p.color
            f.write(f"{pid} {x[0]} {x[1]} {x[2]} {int(cc[0])} {int(cc[1])} {int(cc[2])} 0\n")
    print(f"[fp2] dataset: {k} posed images -> {out_ds}")


def read_dataset(ds):
    from build_depth_targets import read_text_model
    cams, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
    out = []
    for name, cid, q, t in imgs:
        Rcw = qwxyz_to_R(q); out.append((name, cid, T_of(Rcw, t)))
    return cams, out


# ------------------------------------------------------------------ lidar static cloud
def ref_trajectory(cx):
    """Reference-camera world poses (T_world_cam) at keyframe times, from the metric dataset."""
    _, imgs = read_dataset(os.path.join(cx.P, "dataset")); kfs = json.load(open(os.path.join(cx.P, "cameras.json")))["keyframes"]
    tmap = {f"{cx.ref}_{k['ref_ns']}.png": k["t"] for k in kfs}
    tr = sorted((tmap[n], np.linalg.inv(T)) for n, _, T in imgs if n in tmap)
    return np.array([a for a, _ in tr]), np.array([b for _, b in tr])


def scan_world(cx, f, tk, Tk, T_cam_lidar, pose_fn=None):
    a = np.load(f); ns = int(os.path.basename(f)[:-4]); tabs = ns * 1e-9 + a[:, 3].astype(np.float64)
    if pose_fn is None:
        # deskew: bin the point times (2 ms) and interpolate the rig pose per bin
        tb = np.round(tabs / 0.002) * 0.002; ub, inv = np.unique(tb, return_inverse=True)
        Tw = interp_poses(tk, Tk, ub) @ T_cam_lidar
        P = np.einsum("nij,nj->ni", Tw[inv, :3, :3], a[:, :3].astype(np.float64)) + Tw[inv, :3, 3]
    else:
        P = a[:, :3] @ pose_fn[:3, :3].T + pose_fn[:3, 3]
    return P, tabs


def cmd_lidar(cx, args):
    import open3d as o3d
    from scipy.spatial import cKDTree
    L = cx.cfg["lidar"]; outp = os.path.join(cx.P, "lidar_static.ply")
    if os.path.exists(outp) and not args.force:
        print(f"[fp2] lidar: have {outp}"); return
    tk, Tk = ref_trajectory(cx)
    files = sorted(glob.glob(os.path.join(cx.ex, "lidar", "*.npy")))
    files = [f for f in files if tk[0] + 0.1 <= int(os.path.basename(f)[:-4]) * 1e-9 <= tk[-1]]
    vox = float(L.get("voxel", 0.02))
    # 1. extrinsic direction: the cloud the SfM points lie on
    M = cx.T_lidar_yaml(); cands = {"lidar_to_cam": M, "cam_to_lidar": np.linalg.inv(M)}
    want = cx.rig.get("lidar_cam_direction", "auto")
    sfm = np.loadtxt(os.path.join(cx.P, "dataset", "sparse", "0", "points3D.txt"), usecols=(1, 2, 3))
    sub = files[:: max(1, len(files) // 60)]; scores = {}
    for k, T in cands.items():
        P = np.concatenate([scan_world(cx, f, tk, Tk, T)[0] for f in sub])
        d, _ = cKDTree(P[:: max(1, len(P) // 2_000_000)]).query(sfm, k=1, workers=NCPU)
        scores[k] = float(np.median(d))
    direction = min(scores, key=scores.get) if want == "auto" else want
    T_cam_lidar = cands[direction]
    print(f"[fp2] lidar: SfM-point -> LiDAR median distance per extrinsic direction {scores} -> {direction}", flush=True)
    # 2. first pass map (deskewed, interpolated poses), then per-scan point-to-plane ICP against it
    def to_pcd(P):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); return pc
    first = to_pcd(np.concatenate([scan_world(cx, f, tk, Tk, T_cam_lidar)[0][::4] for f in files])).voxel_down_sample(0.05)
    first.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    def icp(f):
        P, _ = scan_world(cx, f, tk, Tk, T_cam_lidar)
        src = to_pcd(P).voxel_down_sample(0.05)
        r = o3d.pipelines.registration.registration_icp(src, first, 0.15, np.eye(4),
                                                         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        D = r.transformation if (r.fitness > 0.5 and np.linalg.norm(r.transformation[:3, 3]) < 0.2) else np.eye(4)
        return f, D, float(np.linalg.norm(D[:3, 3])), r.fitness
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NCPU) as ex:
        res = list(ex.map(icp, files))
    scans = [(f, D) for f, D, _, _ in res]; corr = [c for _, _, c, _ in res]; fit = [x for *_, x in res]
    print(f"[fp2] lidar: {len(files)} scans; ICP correction median {np.median(corr) * 100:.1f} cm, p90 "
          f"{np.percentile(corr, 90) * 100:.1f} cm, fitness median {np.median(fit):.2f}", flush=True)
    # 3. accumulate + free-space carving (dynamic objects)
    # per scan: unique voxels (mean point); globally: voxel -> mean point + #distinct scans that hit it
    def vkey(P):
        k = np.floor(P / vox).astype(np.int64) + (1 << 20)
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    keys, sums, sids, poses = [], [], [], []
    for i, (f, D) in enumerate(scans):
        P, tabs = scan_world(cx, f, tk, Tk, T_cam_lidar); P = P @ D[:3, :3].T + D[:3, 3]
        u, inv = np.unique(vkey(P), return_inverse=True)
        sm = np.stack([np.bincount(inv, P[:, j], len(u)) for j in range(3)], 1); cnt = np.bincount(inv, minlength=len(u))
        keys.append(u); sums.append(np.concatenate([sm, cnt[:, None]], 1)); sids.append(np.full(len(u), i, np.int32))
        poses.append(D @ interp_poses(tk, Tk, [tabs.mean()])[0] @ T_cam_lidar)
    keys = np.concatenate(keys); sums = np.concatenate(sums); sids = np.concatenate(sids)
    u, inv = np.unique(keys, return_inverse=True)
    acc = np.stack([np.bincount(inv, sums[:, j], len(u)) for j in range(4)], 1)
    V = acc[:, :3] / acc[:, 3:4]; nhit = np.bincount(inv, minlength=len(u))     # (voxel, scan) pairs are unique
    del keys, sums, sids
    free = carve(cx, V, scans, poses, L)
    keep = (free < int(L.get("carve_min_free", 3))) | (free < float(L.get("carve_ratio", 0.5)) * nhit)
    sparse = nhit < int(L.get("min_scans", 2)); keep &= ~sparse
    print(f"[fp2] lidar: {len(V)} voxels: {sparse.sum()} hit by < {L.get('min_scans', 2)} scans, "
          f"{((~keep) & ~sparse).sum()} carved as dynamic (seen through)", flush=True)
    pcd = to_pcd(V)
    out = pcd.select_by_index(np.where(keep)[0])
    o3d.io.write_point_cloud(outp, out)
    lnk = os.path.join(cx.P, "depth_cloud.ply")
    if os.path.lexists(lnk):
        os.remove(lnk)
    os.symlink("lidar_static.ply", lnk)
    json.dump({"direction": direction, "direction_scores_m": scores, "scans": len(files), "voxel": vox,
               "icp_corr_median_m": float(np.median(corr)), "points_raw_voxels": int(len(V)),
               "points_static": int(keep.sum()), "removed_dynamic": int((~keep).sum()),
               "T_cam_lidar": T_cam_lidar.tolist()}, open(os.path.join(cx.P, "lidar.json"), "w"), indent=1)
    print(f"[fp2] lidar: {len(V)} voxels -> {keep.sum()} static ({(~keep).sum()} dynamic/sparse removed) -> {outp}")


def carve(cx, V, scans, poses, L):
    """#scans that see THROUGH each voxel: its range in that scan's range image (ring x azimuth bin) is shorter than
    the measured range by > margin (0.1 m + 2 %)."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import minimum_filter
    nb_az = int(L.get("carve_az_bins", 1024)); free = np.zeros(len(V), np.int32)
    step = max(1, len(scans) // int(L.get("carve_scans", 200)))

    def one(i):
        f, _ = scans[i]; a = np.load(f); ring = a[:, 4].astype(int); nr = ring.max() + 1
        az = ((np.arctan2(a[:, 1], a[:, 0]) + np.pi) / (2 * np.pi) * nb_az).astype(int) % nb_az
        rng = np.linalg.norm(a[:, :3], axis=1)
        img = np.zeros((nr, nb_az), np.float32); np.maximum.at(img, (ring, az), rng)
        # free only if the WHOLE 3x3 neighbourhood measured farther (grazing ceilings/floors are not carved;
        # no return = 0 = unknown, never free)
        img = minimum_filter(img, size=3, mode="wrap")
        el = np.array([np.median(np.arctan2(a[ring == r, 2], np.linalg.norm(a[ring == r, :2], axis=1))) if (ring == r).any() else np.nan
                       for r in range(nr)])
        Tinv = np.linalg.inv(poses[i]); Q = V @ Tinv[:3, :3].T + Tinv[:3, 3]
        qr = np.linalg.norm(Q, axis=1); qe = np.arctan2(Q[:, 2], np.linalg.norm(Q[:, :2], axis=1))
        okr = np.isfinite(el); order = np.argsort(el[okr]); els = el[okr][order]; rings = np.arange(nr)[okr][order]
        j = np.clip(np.searchsorted(els, qe), 1, len(els) - 1); j = np.where(np.abs(els[j] - qe) < np.abs(els[j - 1] - qe), j, j - 1)
        inside = (np.abs(els[j] - qe) < np.deg2rad(0.5)) & (qr > 0.5)
        qa = ((np.arctan2(Q[:, 1], Q[:, 0]) + np.pi) / (2 * np.pi) * nb_az).astype(int) % nb_az
        meas = img[rings[j], qa]
        return inside & (meas > qr + 0.10 + 0.02 * qr)

    with ThreadPoolExecutor(max_workers=NCPU) as ex:
        for m in ex.map(one, range(0, len(scans), step)):
            free += m
    return free * step


# ------------------------------------------------------------------ split + init points + depth targets
def cmd_split(cx, args):
    import cv2
    import open3d as o3d
    ds = os.path.join(cx.P, "dataset"); cams, imgs = read_dataset(ds); every = int(cx.get("split.test_every", 8))
    kfs = json.load(open(os.path.join(cx.P, "cameras.json")))["keyframes"]
    test_ns = {k["ref_ns"] for i, k in enumerate(kfs) if i % every == 0}
    ns_of = lambda n: int(os.path.splitext(n)[0].rsplit("_", 1)[1])
    cloud = os.path.join(cx.P, "lidar_static.ply")
    pts = np.asarray(o3d.io.read_point_cloud(cloud).points) if os.path.exists(cloud) else None
    lines = [ln for ln in open(os.path.join(ds, "sparse", "0", "images.txt")) if ln.strip() and not ln.startswith("#")]
    for sub, want in (("train", False), ("test", True)):
        d = os.path.join(cx.P, "onpath", sub); s0 = os.path.join(d, "sparse", "0"); idir = os.path.join(d, "images")
        shutil.rmtree(d, ignore_errors=True); os.makedirs(s0); os.makedirs(idir)
        shutil.copy(os.path.join(ds, "sparse", "0", "cameras.txt"), s0)
        sel = [ln for ln in lines if (ns_of(ln.split()[9]) in test_ns) == want]
        open(os.path.join(s0, "images.txt"), "w").write("".join(ln + "\n" for ln in sel))
        for ln in sel:
            n = ln.split()[9]; os.symlink(os.path.realpath(os.path.join(ds, "images", n)), os.path.join(idir, n))
        if pts is not None:      # LiDAR init, coloured from the training frames (test frames never touch the splat)
            write_lidar_points3d(cx, pts, d if not want else os.path.join(cx.P, "onpath", "train"), os.path.join(s0, "points3D.txt"))
        else:
            shutil.copy(os.path.join(ds, "sparse", "0", "points3D.txt"), s0)
        print(f"[fp2] split: {sub} {len(sel)} images -> {d}")
    if args.depth and pts is not None:
        import subprocess
        for sub in ("train", "test"):
            subprocess.run([os.environ.get("FP2_TORCH_PY", sys.executable),   # needs torch (3DGRUT venv; CPU ok)
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build_depth_targets.py"),
                            "--dataset", os.path.join(cx.P, "onpath", sub), "--cloud", cloud,
                            "--sim3-json", os.path.join(cx.P, "sim3.json")], check=True)


def write_lidar_points3d(cx, pts, color_ds, out, max_points=None):
    import cv2
    n = int(max_points or cx.get("lidar.init_points", 400000)); rng = np.random.default_rng(0)
    P = pts[rng.choice(len(pts), n, replace=False)] if len(pts) > n else pts
    cams, imgs = read_dataset(color_ds); col = np.full((len(P), 3), 128, np.uint8); best = np.full(len(P), np.inf)
    for name, cid, Tcw in imgs[:: max(1, len(imgs) // 150)]:
        W, H, (fx, fy, cx_, cy_) = cams[cid]
        Q = P @ Tcw[:3, :3].T + Tcw[:3, 3]; z = Q[:, 2]
        u = fx * Q[:, 0] / np.maximum(z, 1e-6) + cx_; v = fy * Q[:, 1] / np.maximum(z, 1e-6) + cy_
        ok = (z > 0.3) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z < best)
        if not ok.any():
            continue
        im = cv2.imread(os.path.join(color_ds, "images", name))[:, :, ::-1]
        col[ok] = im[v[ok].astype(int), u[ok].astype(int)]; best[ok] = z[ok]
    with open(out, "w") as f:
        for i, (p, c) in enumerate(zip(P, col)):
            f.write(f"{i + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "prepare", "sfm", "lidar", "split", "all"])
    ap.add_argument("config"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--depth", action="store_true", help="split: also write LiDAR <stem>_depth.npy targets")
    args = ap.parse_args(); cx = Ctx(args.config)
    for c in (["extract", "prepare", "sfm", "lidar", "split"] if args.cmd == "all" else [args.cmd]):
        t0 = time.time(); globals()["cmd_" + c](cx, args); print(f"[fp2] {c} done in {time.time() - t0:.0f}s", flush=True)
    print(f"FP2_{args.cmd.upper()}_DONE")


if __name__ == "__main__":
    main()
