#!/usr/bin/env python3
"""Fisheye / distorted / equirectangular images -> virtual PINHOLE views + a COLMAP text model (CPU, cv2 + numpy).

Generic: any COLMAP text dataset (<src>/sparse/0/{cameras,images}.txt + <src>/images/) whose cameras are one of
  PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE (Kannala-Brandt / Kalibr
  "equidistant"), SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, and EQUIRECTANGULAR (not a COLMAP model: `EQUIRECTANGULAR
  W H` with no params; camera frame x right, y down, z forward; u = (atan2(x, z) / 2pi + 0.5) W, v = (asin(y/|d|) / pi
  + 0.5) H, i.e. the image centre looks along +z).
Each source image becomes one or more virtual pinhole views sharing its optical centre, rotated by (yaw, pitch)
about the source camera's y (down) / x (right) axes: yaw > 0 turns right, pitch > 0 turns up. One view at yaw 0 =
classic undistortion; several = cube-map-style crops that keep the full field of view. Pixels are pulled through the
exact source projection (supersampled --ss x then area-downsampled, so no aliasing when the virtual focal is
smaller). Poses are exact (pure rotation about the centre), so the output model lives in the SAME world frame as
the source: points3D.txt is copied unchanged, and the same PLY / LiDAR cloud scores against both.

Outputs <out>/images/<name> (one view: the source name unchanged, so test sets keep their names; several views:
<stem>_<view><ext>), <out>/sparse/0/{cameras,images,points3D}.txt (one PINHOLE camera per (source camera, view)),
<out>/rectify.json (views, per-view invalid fraction). Pixels a view sees outside the source image / fisheye circle
are black and listed in rectify.json; with --masks a <stem>_mask.png (255 = valid) is written next to each image,
and --max-invalid F fails if any view exceeds F (the scan2sim pinhole trainers have no masks: keep views inside).
Per-frame sidecars next to the source image are carried over with nearest sampling: <stem>_depth.npy (ray distance:
invariant under a rotation about the centre, so the values stay exact) and <stem>_covis.png.

    python scripts/rectify_views.py --src DS --out OUT --fov 90 --size 960x720                # one centre view
    python scripts/rectify_views.py --src DS --out OUT --fov 75 --size 800x800 --yaws -50 0 50 [--pitches 0]
    python scripts/rectify_views.py --src DS --out OUT --fov 90 --size 960x720 --only-camera 2 --subset names.txt

Python API (import with sys.path pointing at scripts/):
    from rectify_views import SourceCamera, make_views, rectify_image, rectify_dataset
    cam = SourceCamera("OPENCV_FISHEYE", 1440, 1080, [fx, fy, cx, cy, k1, k2, k3, k4])
    views = make_views(fov=90, width=960, height=720, yaws=[0])        # [View(name, R_src_virt, K, w, h)]
    img_v, valid = rectify_image(img, cam, views[0], ss=2)              # uint8 HxWx3, bool HxW
    rectify_dataset("src_ds", "out_ds", views, ss=2)                    # whole COLMAP dataset -> dict summary
"""
from __future__ import annotations
import argparse, json, os, shutil
from dataclasses import dataclass, field
import numpy as np

MODELS = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8, "FULL_OPENCV": 12,
          "OPENCV_FISHEYE": 8, "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "EQUIRECTANGULAR": 0}


# ----------------------------------------------------------------------------- source camera models
@dataclass
class SourceCamera:
    model: str
    width: int
    height: int
    params: list = field(default_factory=list)

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unsupported camera model {self.model}")
        if len(self.params) < MODELS[self.model]:
            raise ValueError(f"{self.model} needs {MODELS[self.model]} params, got {len(self.params)}")
        self.params = [float(p) for p in self.params]

    def _fcc(self):
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]

    def max_theta(self) -> float:
        """Largest ray angle (rad) the fisheye polynomial maps monotonically (beyond it the model folds back)."""
        if self.model not in ("OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            return np.pi / 2
        t = np.linspace(1e-4, np.pi, 20000); td = self._theta_d(t)
        bad = np.where(np.diff(td) <= 0)[0]
        return float(t[bad[0]]) if len(bad) else float(np.pi)

    def _theta_d(self, t):
        p = self.params
        k = {"OPENCV_FISHEYE": p[4:8], "SIMPLE_RADIAL_FISHEYE": p[3:4], "RADIAL_FISHEYE": p[3:5]}[self.model]
        t2, poly, tp = t * t, 1.0, 1.0
        for ki in k:
            tp = tp * t2; poly = poly + ki * tp
        return t * poly

    def project(self, d: np.ndarray):
        """Unit-free camera-frame rays d (..., 3) -> pixel coords (..., 2) float32, valid (...) bool (in image)."""
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        W, H = self.width, self.height
        if self.model == "EQUIRECTANGULAR":
            n = np.linalg.norm(d, axis=-1)
            u = (np.arctan2(x, z) / (2 * np.pi) + 0.5) * W
            v = (np.arcsin(np.clip(y / np.maximum(n, 1e-12), -1, 1)) / np.pi + 0.5) * H
            uv = np.stack([u - 0.5, v - 0.5], -1)                        # pixel centres at integer coords
            return uv.astype(np.float32), np.ones(x.shape, bool)
        fx, fy, cx, cy = self._fcc(); p = self.params
        if self.model.endswith("FISHEYE"):
            r = np.sqrt(x * x + y * y); th = np.arctan2(r, z)
            s = np.where(r > 1e-12, self._theta_d(th) / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
            xd, yd = x * s, y * s
            valid = th < self.max_theta()
        else:
            valid = z > 1e-6
            zz = np.where(valid, z, 1.0); xn, yn = x / zz, y / zz; r2 = xn * xn + yn * yn
            if self.model in ("PINHOLE", "SIMPLE_PINHOLE"):
                xd, yd = xn, yn
            elif self.model == "SIMPLE_RADIAL":
                f = 1 + p[3] * r2; xd, yd = xn * f, yn * f
            elif self.model == "RADIAL":
                f = 1 + p[3] * r2 + p[4] * r2 * r2; xd, yd = xn * f, yn * f
            else:                                                         # OPENCV / FULL_OPENCV
                k1, k2, p1, p2 = p[4:8]
                if self.model == "FULL_OPENCV":
                    k3, k4, k5, k6 = p[8:12]
                    f = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
                else:
                    f = 1 + k1 * r2 + k2 * r2 ** 2
                xd = xn * f + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                yd = yn * f + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u, v = fx * xd + cx, fy * yd + cy                                  # COLMAP: pixel centre at +0.5
        uv = np.stack([u - 0.5, v - 0.5], -1)                             # cv2.remap: centre at integer coords
        valid = valid & (uv[..., 0] >= -0.5) & (uv[..., 0] <= W - 0.5) & (uv[..., 1] >= -0.5) & (uv[..., 1] <= H - 0.5)
        return uv.astype(np.float32), valid


# ----------------------------------------------------------------------------- virtual pinhole views
@dataclass
class View:
    name: str
    R_src_virt: np.ndarray        # (3,3): virtual-camera ray -> source-camera ray
    K: np.ndarray                 # (3,3) pinhole intrinsics (COLMAP convention, pixel centre at +0.5)
    width: int
    height: int


def rot_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation of a virtual camera turned right by yaw (about +y, down) then up by pitch (about its +x)."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])    # +pitch looks up (y down)
    return Ry @ Rx


def make_views(fov: float, width: int, height: int, yaws=(0.0,), pitches=(0.0,), fov_axis: str = "h",
               center: tuple | None = None) -> list[View]:
    """Virtual pinhole views: square pixels, horizontal (fov_axis 'h') or vertical ('v') field of view `fov` deg,
    principal point at the image centre (or `center` = (cx, cy)); one view per (yaw, pitch). Names: '' for a single
    (0, 0) view, else 'y<yaw>p<pitch>' (e.g. y-50p0)."""
    f = (width if fov_axis == "h" else height) / 2 / np.tan(np.radians(fov) / 2)
    cx, cy = center if center is not None else (width / 2, height / 2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    combos = [(y, p) for p in pitches for y in yaws]
    single = len(combos) == 1 and combos[0] == (0, 0)
    return [View("" if single else f"y{y:g}p{p:g}", rot_yaw_pitch(y, p), K, int(width), int(height))
            for y, p in combos]


def view_maps(cam: SourceCamera, view: View, ss: int = 1):
    """cv2.remap maps (float32 HxW each, at ss x resolution) + valid mask for one (source camera, view)."""
    W, H = view.width * ss, view.height * ss
    K = view.K.copy(); K[:2] *= ss
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ view.R_src_virt.T
    uv, valid = cam.project(d)
    return uv[..., 0], uv[..., 1], valid


def _area_down(a, ss):
    if ss == 1:
        return a
    H, W = a.shape[0] // ss, a.shape[1] // ss
    return a.reshape(H, ss, W, ss, *a.shape[2:]).mean(axis=(1, 3))


def rectify_image(img: np.ndarray, cam: SourceCamera, view: View, ss: int = 2, maps=None):
    """One image -> (virtual view uint8, valid bool HxW: every sub-sample inside the source image)."""
    import cv2
    mx, my, valid = maps if maps is not None else view_maps(cam, view, ss)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = out * valid[..., None] if out.ndim == 3 else out * valid
    out = np.clip(_area_down(out.astype(np.float32), ss) + 0.5, 0, 255).astype(np.uint8)
    return out, _area_down(valid.astype(np.float32), ss) > 0.999


def _nearest(arr, mx, my, valid, ss):
    import cv2
    o = cv2.remap(arr, mx[ss // 2::ss, ss // 2::ss], my[ss // 2::ss, ss // 2::ss], cv2.INTER_NEAREST,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return o * valid[ss // 2::ss, ss // 2::ss] if o.ndim == 2 else o * valid[ss // 2::ss, ss // 2::ss, None]


# ----------------------------------------------------------------------------- COLMAP text I/O
def qvec2rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot2qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z]) * (1 if w >= 0 else -1)


def read_cameras(path):
    cams = {}
    for ln in open(path):
        if ln.strip() and not ln.startswith("#"):
            p = ln.split(); cams[int(p[0])] = SourceCamera(p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    return cams


def read_images(path):
    """[(image_id, qvec wxyz, tvec, camera_id, name)] (world-to-camera, COLMAP)."""
    lines = [ln for ln in open(path) if not ln.startswith("#")]
    out = []
    for ln in lines[::2]:
        p = ln.split()
        if len(p) >= 10:
            out.append((int(p[0]), np.array(p[1:5], float), np.array(p[5:8], float), int(p[8]), " ".join(p[9:])))
    return out


def write_model(out_sparse, cams: dict, images: list, points3d_src: str | None = None):
    """cams {id: (w, h, fx, fy, cx, cy)} PINHOLE; images [(id, qvec, tvec, cam_id, name)]."""
    os.makedirs(out_sparse, exist_ok=True)
    with open(os.path.join(out_sparse, "cameras.txt"), "w") as f:
        f.write(f"# Camera list with one line of data per camera:\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                f"# Number of cameras: {len(cams)}\n")
        for cid, (w, h, fx, fy, cx, cy) in sorted(cams.items()):
            f.write(f"{cid} PINHOLE {w} {h} {fx:.10g} {fy:.10g} {cx:.10g} {cy:.10g}\n")
    with open(os.path.join(out_sparse, "images.txt"), "w") as f:
        f.write(f"# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n"
                f"# Number of images: {len(images)}\n")
        for iid, q, t, cid, name in images:
            f.write(f"{iid} {' '.join(f'{v:.12g}' for v in q)} {' '.join(f'{v:.12g}' for v in t)} {cid} {name}\n\n")
    dst = os.path.join(out_sparse, "points3D.txt")
    if points3d_src and os.path.exists(points3d_src):
        shutil.copyfile(points3d_src, dst)
    else:
        open(dst, "w").write("# 3D point list (empty)\n")


# ----------------------------------------------------------------------------- dataset
def rectify_dataset(src: str, out: str, views: list[View], images_dir: str | None = None, ss: int = 2,
                    jpg_quality: int = 95, masks: bool = False, only_camera: list | None = None,
                    subset: set | None = None, max_invalid: float | None = None, log=print) -> dict:
    """Rectify a whole COLMAP text dataset (see module docstring). Returns the rectify.json summary."""
    import cv2
    sp = os.path.join(src, "sparse", "0")
    cams, imgs = read_cameras(os.path.join(sp, "cameras.txt")), read_images(os.path.join(sp, "images.txt"))
    images_dir = images_dir or os.path.join(src, "images")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    maps, out_cams, out_imgs, inval, cam_key = {}, {}, [], {}, {}
    for iid, q, t, cid, name in imgs:
        if (only_camera and cid not in only_camera) or (subset is not None and name not in subset):
            continue
        img = None
        R_cw = qvec2rot(q)                                                  # source world -> camera
        for vi, v in enumerate(views):
            key = (cid, vi)
            if key not in maps:
                maps[key] = view_maps(cams[cid], v, ss)
                m = _area_down(maps[key][2].astype(np.float32), ss) > 0.999
                inval[f"cam{cid}{'_' + v.name if v.name else ''}"] = float(1 - m.mean())
                if max_invalid is not None and 1 - m.mean() > max_invalid:
                    raise SystemExit(f"[rectify] camera {cid} view '{v.name}': {100 * (1 - m.mean()):.2f}% of the "
                                     f"view falls outside the source image (> --max-invalid {max_invalid})")
                k = (v.width, v.height) + tuple(np.round(v.K[[0, 1, 0, 1], [0, 1, 2, 2]], 9))
                cam_key[key] = cam_key.get(key) or next((c for c, kk in out_cams.items() if kk == k), None) \
                    or len(out_cams) + 1
                out_cams[cam_key[key]] = k
            stem, ext = os.path.splitext(name)
            oname = name if not v.name else f"{stem}_{v.name}{ext}"
            # pose: virtual camera = source camera rotated about its centre; R_vw = R_src_virt^T R_sw
            R_vw = v.R_src_virt.T @ R_cw
            C = -R_cw.T @ t
            out_imgs.append((len(out_imgs) + 1, rot2qvec(R_vw), -R_vw @ C, cam_key[key], oname))
            dst = os.path.join(out, "images", oname)
            if os.path.exists(dst):
                continue
            if img is None:
                img = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
                if img is None:
                    raise SystemExit(f"[rectify] cannot read {os.path.join(images_dir, name)}")
            rv, valid = rectify_image(img, cams[cid], v, ss, maps[key])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, rv, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if ext.lower() in (".jpg", ".jpeg") else [])
            if masks:
                cv2.imwrite(os.path.splitext(dst)[0] + "_mask.png", valid.astype(np.uint8) * 255)
            mx, my, mv = maps[key]
            for suf in ("_depth.npy", "_covis.png"):                       # per-frame sidecars, nearest sampling
                sf = os.path.join(images_dir, stem + suf)
                if os.path.exists(sf):
                    df = os.path.splitext(dst)[0] + suf
                    if suf.endswith(".npy"):
                        np.save(df, _nearest(np.load(sf).astype(np.float32), mx, my, mv, ss).astype(np.float16))
                    else:
                        cv2.imwrite(df, _nearest(cv2.imread(sf, cv2.IMREAD_UNCHANGED), mx, my, mv, ss))
    write_model(os.path.join(out, "sparse", "0"), {c: (k[0], k[1], *k[2:]) for c, k in out_cams.items()},
                out_imgs, os.path.join(sp, "points3D.txt"))
    for extra in ("depth_targets.json",):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copyfile(os.path.join(src, extra), os.path.join(out, extra))
    summary = {"src": os.path.abspath(src), "n_images": len(out_imgs), "supersample": ss,
               "views": [{"name": v.name, "R_src_virt": v.R_src_virt.tolist(), "K": v.K.tolist(),
                          "width": v.width, "height": v.height} for v in views],
               "invalid_fraction": inval}
    json.dump(summary, open(os.path.join(out, "rectify.json"), "w"), indent=1)
    log(f"[rectify] {len(out_imgs)} views from {len(imgs)} images -> {out}; invalid fraction "
        + ", ".join(f"{k} {100 * x:.2f}%" for k, x in inval.items()))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="COLMAP text dataset (sparse/0/*.txt + images/)")
    ap.add_argument("--images", default=None, help="image dir (default <src>/images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=90.0, help="virtual field of view (deg) along --fov-axis")
    ap.add_argument("--fov-axis", choices=("h", "v"), default="h")
    ap.add_argument("--size", default="960x720", help="WxH of each virtual view")
    ap.add_argument("--yaws", type=float, nargs="+", default=[0.0], help="deg, + turns right")
    ap.add_argument("--pitches", type=float, nargs="+", default=[0.0], help="deg, + turns up")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor (area-downsampled)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--masks", action="store_true", help="write <stem>_mask.png (255 = inside the source image)")
    ap.add_argument("--max-invalid", type=float, default=None, help="fail if a view has more invalid pixels (0-1)")
    ap.add_argument("--only-camera", type=int, nargs="*", default=None, help="source camera ids to keep")
    ap.add_argument("--subset", default=None, help="text file of source image names to keep")
    a = ap.parse_args()
    W, H = map(int, a.size.lower().split("x"))
    subset = set(open(a.subset).read().split()) if a.subset else None
    rectify_dataset(a.src, a.out, make_views(a.fov, W, H, a.yaws, a.pitches, a.fov_axis), a.images, a.ss,
                    a.quality, a.masks, a.only_camera, subset, a.max_invalid)


if __name__ == "__main__":
    main()
