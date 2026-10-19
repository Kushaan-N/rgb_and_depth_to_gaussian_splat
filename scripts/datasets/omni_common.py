"""Shared helpers for the fisheye / 360 dataset adapters (scripts/datasets/fiord.py, ricoh360.py; CPU venv).

Camera models (COLMAP names; params as in cameras.txt):
  PINHOLE / SIMPLE_PINHOLE            standard
  OPENCV_FISHEYE fx fy cx cy k1..k4   Kannala-Brandt: theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
  EQUIRECT (aliases OPENCV_SPHERICAL, SPHERICAL, SPHERE, EQUIRECTANGULAR)  W x H, no params; OpenCV camera axes
           (x right, y down, z forward), image centre = +z, lon = (u / W - 0.5) 2 pi, lat = (0.5 - v / H) pi.
           detect_equirect_convention() checks it (and the mirrored/rolled variants) against the model's own
           2D-3D tracks, so a dataset written in another convention is caught instead of silently misprojected.
Virtual pinhole rigs: rig_views() turns a list of {yaw, pitch, fov, [name]} (degrees, about the source camera's
own axes: yaw right, pitch up) into rotations R_src_from_virt; rectify() resamples a source image into each view
(area-prefiltered, cv2.remap) and returns the validity mask (pixels whose ray lies inside the source FOV).
COLMAP I/O: read_model() reads text (any model name, incl. EQUIRECT, which pycolmap rejects) or binary (via
pycolmap) and returns cams {id: (model, W, H, params)}, imgs [(name, image_id, cam_id, c2w 4x4)] in name order,
points (xyz, rgb uint8), and optionally the 2D tracks; write_model() writes a text model.
"""
from __future__ import annotations
import hashlib, json, math, os, sys, urllib.request
import numpy as np

EQUI = {"EQUIRECT", "OPENCV_SPHERICAL", "SPHERICAL", "SPHERE", "EQUIRECTANGULAR"}


# ---------------------------------------------------------------- rotations
def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot2qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def rot_yaw_pitch(yaw_deg, pitch_deg, roll_deg=0.0):
    """R_src_from_virt for a virtual camera turned yaw right (about +y down axis) then pitch up (OpenCV axes)."""
    a, b, c = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])   # +pitch looks up (-y)
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


# ---------------------------------------------------------------- COLMAP I/O
def _read_text(d, tracks=False):
    cams = {}
    for ln in open(os.path.join(d, "cameras.txt")):
        if ln.strip() and not ln.startswith("#"):
            p = ln.split(); cams[int(p[0])] = (p[1].upper(), int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    imgs, obs = [], {}
    lines = [ln.rstrip("\n") for ln in open(os.path.join(d, "images.txt")) if not ln.startswith("#")]
    k = 0
    while k < len(lines):
        p = lines[k].split()
        if len(p) < 10:
            k += 1; continue
        w2c = np.eye(4); w2c[:3, :3] = qvec2rot([float(x) for x in p[1:5]]); w2c[:3, 3] = [float(x) for x in p[5:8]]
        imgs.append((p[9], int(p[0]), int(p[8]), np.linalg.inv(w2c)))
        if tracks and k + 1 < len(lines):
            a = np.array(lines[k + 1].split(), dtype=np.float64).reshape(-1, 3)
            obs[p[9]] = a[a[:, 2] >= 0]
        k += 2
    xyz, rgb, pid = [], [], []
    f = os.path.join(d, "points3D.txt")
    if os.path.exists(f) and os.path.getsize(f) > 0:
        a = np.loadtxt(f, comments="#", usecols=(0, 1, 2, 3, 4, 5, 6), ndmin=2)
        if len(a):
            pid, xyz, rgb = a[:, 0].astype(np.int64), a[:, 1:4], a[:, 4:7].astype(np.uint8)
    pts = (np.asarray(xyz, np.float64).reshape(-1, 3), np.asarray(rgb, np.uint8).reshape(-1, 3))
    return cams, sorted(imgs, key=lambda x: x[0]), pts, (obs, np.asarray(pid, np.int64))


def read_model(d, tracks=False):
    """Text or binary COLMAP model -> cams, imgs (name order), (xyz, rgb) [, (obs {name: (u, v, pid)}, point ids)]."""
    if not os.path.exists(os.path.join(d, "cameras.txt")):
        import pycolmap, tempfile
        tmp = tempfile.mkdtemp(prefix="omni_model_")
        pycolmap.Reconstruction(d).write_text(tmp)
        d = tmp
    out = _read_text(d, tracks)
    return out if tracks else out[:3]


def write_model(d, cams, imgs, xyz=None, rgb=None):
    """cams {id: (model, W, H, params)}, imgs [(name, iid, cid, c2w)], points -> COLMAP text model (no tracks)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cid, (m, w, h, p) in sorted(cams.items()):
            f.write(f"{cid} {m} {int(w)} {int(h)} {' '.join(f'{x:.10g}' for x in p)}\n")
    with open(os.path.join(d, "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for name, iid, cid, c2w in imgs:
            w2c = np.linalg.inv(c2w); q = rot2qvec(w2c[:3, :3]); t = w2c[:3, 3]
            f.write(f"{iid} {' '.join(f'{x:.12g}' for x in q)} {' '.join(f'{x:.12g}' for x in t)} {cid} {name}\n\n")
    with open(os.path.join(d, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        if xyz is not None and len(xyz):
            rgb = np.full((len(xyz), 3), 128, np.uint8) if rgb is None else rgb
            np.savetxt(f, np.column_stack([np.arange(1, len(xyz) + 1), xyz, rgb, np.zeros(len(xyz))]),
                       fmt="%d %.6f %.6f %.6f %d %d %d %.3f")


# ---------------------------------------------------------------- projection
def fisheye_theta_d(theta, k):
    t2 = theta * theta
    return theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def fisheye_max_theta(params, w, h, limit_deg=None):
    """Largest incidence angle (rad) inside the image (corner radius) where theta_d stays monotonic."""
    fx, fy, cx, cy = params[:4]; k = list(params[4:8]) + [0] * (4 - len(params[4:8]))
    rmax = max(math.hypot(x - cx, y - cy) for x in (0, w) for y in (0, h)) / max(fx, fy)
    th = np.linspace(0, math.pi, 20001); td = fisheye_theta_d(th, k)
    mono = np.where(np.diff(td) <= 0)[0]; end = mono[0] if len(mono) else len(th) - 1
    ok = np.where(td[:end + 1] <= rmax)[0]; tmax = th[ok[-1]] if len(ok) else th[end]
    return min(tmax, math.radians(limit_deg)) if limit_deg else tmax


def project(cam, d, conv=None):
    """Directions d (N,3) in the camera frame (OpenCV axes) -> pixel coords (N,2) and validity (N,)."""
    m, w, h, p = cam
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    if m in EQUI:
        lon, lat = np.arctan2(d[:, 0], d[:, 2]), np.arcsin(np.clip(-d[:, 1], -1, 1))
        if conv:                                                     # detected variant (see detect_equirect_convention)
            lon = conv.get("lon_sign", 1) * lon + math.radians(conv.get("lon0_deg", 0.0))
            lon = (lon + math.pi) % (2 * math.pi) - math.pi
        uv = np.stack([(lon / (2 * math.pi) + 0.5) * w, (0.5 - lat / math.pi) * h], 1)
        return uv, np.ones(len(d), bool)
    if m == "OPENCV_FISHEYE":
        fx, fy, cx, cy = p[:4]; k = list(p[4:8]) + [0] * (4 - len(p[4:8]))
        r = np.hypot(d[:, 0], d[:, 1]); th = np.arctan2(r, d[:, 2])
        td = fisheye_theta_d(th, k); s = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 1.0)
        uv = np.stack([fx * d[:, 0] * s + cx, fy * d[:, 1] * s + cy], 1)
        return uv, th <= fisheye_max_theta(p, w, h)
    if m in ("PINHOLE", "SIMPLE_PINHOLE"):
        fx, fy, cx, cy = (p[0], p[1], p[2], p[3]) if m == "PINHOLE" else (p[0], p[0], p[1], p[2])
        z = np.maximum(d[:, 2], 1e-9)
        return np.stack([fx * d[:, 0] / z + cx, fy * d[:, 1] / z + cy], 1), d[:, 2] > 1e-6
    raise SystemExit(f"[omni] unsupported camera model {m}")


def src_pixel_density(cam):
    """Pixels per radian at the source camera's centre (fisheye/pinhole: focal; equirect: W / 2 pi)."""
    m, w, h, p = cam
    return w / (2 * math.pi) if m in EQUI else float(p[0])


def detect_equirect_convention(cam, imgs, obs, pid, xyz, n_img=12):
    """Pick the equirect variant (lon sign, lon offset in 90 deg steps) that best reprojects the SfM tracks."""
    idx = {int(i): k for k, i in enumerate(pid)}
    best = None
    for sign in (1, -1):
        for lon0 in (0, 90, 180, -90):
            conv = {"lon_sign": sign, "lon0_deg": lon0}
            err = []
            for name, _iid, _cid, c2w in imgs[:: max(1, len(imgs) // n_img)]:
                o = obs.get(name)
                if o is None or not len(o):
                    continue
                sel = [idx.get(int(q), -1) for q in o[:, 2]]; ok = np.array(sel) >= 0
                if not ok.any():
                    continue
                X = xyz[np.array(sel)[ok]]; w2c = np.linalg.inv(c2w)
                dc = X @ w2c[:3, :3].T + w2c[:3, 3]
                uv, _ = project(cam, dc, conv)
                du = np.abs(uv[:, 0] - o[ok, 0]); du = np.minimum(du, cam[1] - du)        # wrap-around
                err.append(np.hypot(du, uv[:, 1] - o[ok, 1]))
            if err:
                e = float(np.median(np.concatenate(err)))
                if best is None or e < best[0]:
                    best = (e, conv)
    if best is None:
        return {"lon_sign": 1, "lon0_deg": 0, "median_px": None}
    return dict(best[1], median_px=round(best[0], 3))


# ---------------------------------------------------------------- virtual pinhole rigs
def rig_views(spec, density, scale=1.0, size=None):
    """spec: [{yaw, pitch, fov[, fov_y, roll, name]}] -> [(name, R_src_from_virt, K, W, H)].
    Focal = scale x the source density (px/rad) unless size (px, square side) is given."""
    out = []
    for k, v in enumerate(spec):
        fx_deg, fy_deg = float(v["fov"]), float(v.get("fov_y", v["fov"]))
        if size:
            W = H = int(size); f = W / 2 / math.tan(math.radians(fx_deg) / 2)
        else:
            f = scale * density
            W = int(round(2 * f * math.tan(math.radians(fx_deg) / 2) / 2)) * 2
            H = int(round(2 * f * math.tan(math.radians(fy_deg) / 2) / 2)) * 2
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        out.append((v.get("name", f"v{k}"), rot_yaw_pitch(v.get("yaw", 0), v.get("pitch", 0), v.get("roll", 0)), K, W, H))
    return out


def view_rays(K, W, H):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)


def rectify(img, cam, R, K, W, H, conv=None, src_mask=None):
    """Resample img (source model cam) into the virtual pinhole (R_src_from_virt, K, W, H).
    The source is pre-shrunk by powers of two (area filter) so heavy downsampling does not alias.
    -> image (H, W, C), valid (H, W) bool (ray inside the source FOV and the source mask)."""
    import cv2
    uv, ok = project(cam, view_rays(K, W, H) @ R.T, conv)
    sx = src_pixel_density(cam) / K[0, 0]                     # source px per virtual px at the centre
    im, m = img, src_mask; lv = 0
    while sx >= 2.0 and min(im.shape[:2]) > 64:
        im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if m is not None:
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        sx /= 2; lv += 1
    s = 1.0 / (2 ** lv)
    mx = (uv[:, 0] * s - 0.5).reshape(H, W).astype(np.float32); my = (uv[:, 1] * s - 0.5).reshape(H, W).astype(np.float32)
    border = cv2.BORDER_WRAP if cam[0] in EQUI else cv2.BORDER_CONSTANT
    out = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=border)
    valid = ok.reshape(H, W) & (mx >= 0) & (my >= 0) & (mx <= im.shape[1] - 1) & (my <= im.shape[0] - 1)
    if cam[0] in EQUI:
        valid = ok.reshape(H, W) & (my >= 0) & (my <= im.shape[0] - 1)
    if m is not None:
        valid &= cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    return out, valid


# ---------------------------------------------------------------- downloads
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, md5sum=None, size=None):
    """Single-stream resumable download (HTTP Range) with md5/size verification; skips a verified file."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok_flag = dst + ".ok"
    if os.path.exists(dst) and os.path.exists(ok_flag):
        return dst
    part = dst + ".part"
    if os.path.exists(dst) and not os.path.exists(part):
        os.rename(dst, part)
    for attempt in range(8):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "scan2sim"} if have else
                                     {"User-Agent": "scan2sim"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                total = int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
                size = size or total
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            if not size or os.path.getsize(part) >= size:
                break
        except Exception as e:                                           # noqa: BLE001  (network hiccup: resume)
            print(f"[fetch] {os.path.basename(dst)} attempt {attempt}: {e}", file=sys.stderr)
    if md5sum and md5(part) != md5sum.split(":")[-1]:
        raise SystemExit(f"[fetch] md5 mismatch for {part}")
    os.rename(part, dst); open(ok_flag, "w").write(json.dumps({"url": url, "md5": md5sum, "bytes": os.path.getsize(dst)}))
    return dst


# ---------------------------------------------------------------- dataset preparation (shared by the adapters)
def _flat(name):
    return os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]


def _rect_job(a):
    """One source frame -> its virtual pinhole views (worker; returns per-view valid fractions)."""
    import cv2
    src, cam, views, conv, mask_path, out_dir, ext = a
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[omni] cannot read {src}")
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path and os.path.exists(mask_path) else None
    fr = []
    for vname, R, K, W, H, dst in views:
        out, valid = rectify(img, cam, R, K, W, H, conv, m)
        out[~valid] = 0
        cv2.imwrite(os.path.join(out_dir, dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [])
        fr.append(float(valid.mean()))
    return fr


def _native_job(a):
    """One fisheye frame -> downsampled copy + <stem>_mask.png (pixels whose ray is within max_deg)."""
    import cv2
    src, dst, cam, ds, max_deg = a
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    m, w, h, p = cam
    if ds > 1:
        img = cv2.resize(img, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95] if dst.endswith(".jpg") else [])
    if m == "OPENCV_FISHEYE":
        hh, ww = img.shape[:2]; s = ww / w
        u, v = np.meshgrid((np.arange(ww) + 0.5) / s, (np.arange(hh) + 0.5) / s)
        fx, fy, cx, cy = p[:4]; k = list(p[4:8]) + [0] * (4 - len(p[4:8]))
        r = np.hypot((u - cx) / fx, (v - cy) / fy)                       # = theta_d
        th = np.linspace(0, fisheye_max_theta(p, w, h), 4001)
        theta = np.interp(r, fisheye_theta_d(th, k), th, right=np.inf)
        mask = (theta <= math.radians(max_deg)).astype(np.uint8) * 255
        cv2.imwrite(os.path.splitext(dst)[0] + "_mask.png", mask)


def prepare(out, model_dir, image_path, test_names, rigs, rect_scale=0.5, rect_size=None, native_ds=1,
            native_max_deg=None, extra_points=None, ext=".png", workers=None, frames=None, native=True):
    """Write the rectified + native datasets of one omnidirectional/fisheye COLMAP model.

    image_path(name) -> source image file; test_names: set of COLMAP image names held out; rigs: {model: spec}
    (rig_views spec per source camera model, key 'default' as fallback); extra_points: (xyz, rgb) merged into the
    rectified training set's points3D (e.g. LiDAR seeds); frames: optional subset of names (smoke tests).
      <out>/rect/{train,test}/{images,sparse/0}   virtual pinhole views (<frame>_<view>.<ext>), PINHOLE cameras
      <out>/native/all/{images,sparse/0}, train.txt, test.txt   the source frames (ODGS / any native method)
      <out>/native/{train,test}/{images,sparse/0} training / held-out frames only (3DGRUT/3DGUT, eval_views.py)
      <out>/meta.json                             split, views, equirect convention, valid fractions
    """
    from multiprocessing import Pool
    cams, imgs, (xyz, rgb), (obs, pid) = read_model(model_dir, tracks=True)
    if frames:
        imgs = [x for x in imgs if x[0] in set(frames)]
    conv = {}
    for cid, cam in cams.items():
        if cam[0] in EQUI:
            conv[cid] = detect_equirect_convention(cam, [x for x in imgs if x[2] == cid], obs, pid, xyz)
            print(f"[omni] camera {cid} {cam[0]}: equirect convention {conv[cid]}")
    meta = {"model_dir": model_dir, "n_frames": len(imgs), "test": sorted(n for n, *_ in imgs if n in test_names),
            "conventions": {str(k): v for k, v in conv.items()}, "views": {}, "rect_scale": rect_scale}
    jobs = {"train": [], "test": []}
    rimgs = {"train": [], "test": []}; rcams = {}; vid = {}
    for name, iid, cid, c2w in imgs:
        cam = cams[cid]; spec = rigs.get(cam[0], rigs.get("default"))
        views = rig_views(spec, src_pixel_density(cam), rect_scale, rect_size)
        split = "test" if name in test_names else "train"
        vlist = []
        for vname, R, K, W, H in views:
            key = (cid, vname)
            if key not in vid:
                vid[key] = len(vid) + 1; rcams[vid[key]] = ("PINHOLE", W, H, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                meta["views"][f"{cid}:{vname}"] = {"W": W, "H": H, "f": float(K[0, 0]), "R_src_from_virt": R.tolist()}
            dst = f"{_flat(name)}_{vname}{ext}"
            T = np.eye(4); T[:3, :3] = R
            rimgs[split].append((dst, len(rimgs[split]) + 1, vid[key], c2w @ T))
            vlist.append((vname, R, K, W, H, dst))
        jobs[split].append((image_path(name), cam, vlist, conv.get(cid), None, os.path.join(out, "rect", split, "images"), ext))
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    fracs = {}
    for split in ("train", "test"):
        d = os.path.join(out, "rect", split)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        pts = (xyz, rgb)
        if split == "train" and extra_points is not None and len(extra_points[0]):
            pts = (np.vstack([xyz, extra_points[0]]), np.vstack([rgb, extra_points[1]]))
        used = {c for _, _, c, _ in rimgs[split]}
        write_model(os.path.join(d, "sparse", "0"), {c: rcams[c] for c in used}, rimgs[split],
                    *(pts if split == "train" else (None, None)))
        todo = [j for j in jobs[split] if not all(os.path.exists(os.path.join(j[5], v[5])) for v in j[2])]
        pool = Pool(workers); res = pool.map(_rect_job, todo, chunksize=1); pool.close(); pool.join()
        for j, fr in zip(todo, res):
            for v, f in zip(j[2], fr):
                fracs.setdefault(v[0], []).append(f)
        print(f"[omni] rect/{split}: {len(rimgs[split])} views from {len(jobs[split])} frames -> {d}")
    meta["valid_fraction"] = {k: float(np.mean(v)) for k, v in fracs.items()}
    if native:
        na = os.path.join(out, "native", "all"); os.makedirs(os.path.join(na, "images"), exist_ok=True)
        ncams, nimgs, njobs = {}, [], []
        for cid, (m, w, h, p) in cams.items():
            ds = native_ds if m == "OPENCV_FISHEYE" else 1
            q = list(p)
            if ds > 1:
                q[:4] = [p[0] / ds, p[1] / ds, p[2] / ds, p[3] / ds]
            ncams[cid] = (m if m not in EQUI else "OPENCV_SPHERICAL" if m == "OPENCV_SPHERICAL" else "EQUIRECT",
                          w // ds, h // ds, q)
        for name, iid, cid, c2w in imgs:
            src = image_path(name); dst = _flat(name) + os.path.splitext(src)[1].lower()
            nimgs.append((dst, iid, cid, c2w))
            if cams[cid][0] == "OPENCV_FISHEYE" and (native_ds > 1 or native_max_deg):
                if not os.path.exists(os.path.join(na, "images", dst)):
                    njobs.append((src, os.path.join(na, "images", dst), cams[cid], native_ds, native_max_deg or 180))
            elif not os.path.lexists(os.path.join(na, "images", dst)):
                os.symlink(os.path.abspath(src), os.path.join(na, "images", dst))
        pool = Pool(workers); pool.map(_native_job, njobs, chunksize=1); pool.close(); pool.join()   # (no terminate)
        write_model(os.path.join(na, "sparse", "0"), ncams, nimgs, xyz, rgb)
        test_flat = {_flat(n) for n in test_names}
        open(os.path.join(na, "train.txt"), "w").write("\n".join(os.path.splitext(n)[0] for n, *_ in nimgs
                                                                 if os.path.splitext(n)[0] not in test_flat) + "\n")
        open(os.path.join(na, "test.txt"), "w").write("\n".join(os.path.splitext(n)[0] for n, *_ in nimgs
                                                                if os.path.splitext(n)[0] in test_flat) + "\n")
        for sub, keep in (("train", False), ("test", True)):         # 3DGRUT-style sets (test_split_interval=0)
            dd = os.path.join(out, "native", sub); os.makedirs(os.path.join(dd, "images"), exist_ok=True)
            sel = [x for x in nimgs if (os.path.splitext(x[0])[0] in test_flat) == keep]
            for n, *_ in sel:
                for f in (n, os.path.splitext(n)[0] + "_mask.png"):
                    if os.path.exists(os.path.join(na, "images", f)) and not os.path.lexists(os.path.join(dd, "images", f)):
                        os.symlink(os.path.join(na, "images", f), os.path.join(dd, "images", f))
            write_model(os.path.join(dd, "sparse", "0"), ncams, sel, *((xyz, rgb) if not keep else (None, None)))
        tr = [x for x in nimgs if os.path.splitext(x[0])[0] not in test_flat]
        print(f"[omni] native: {len(nimgs)} frames ({len(tr)} train) -> {os.path.join(out, 'native')}")
    # off-path probe distance: a fraction of the median camera-to-point depth (scale-free; SfM units differ)
    if len(xyz):
        c = np.array([x[3][:3, 3] for x in imgs])[:: max(1, len(imgs) // 100)]; sub = xyz[:: max(1, len(xyz) // 5000)]
        meta["median_depth"] = float(np.median(np.median(np.linalg.norm(sub[None] - c[:, None], axis=2), axis=1)))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta
