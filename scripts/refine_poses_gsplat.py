#!/usr/bin/env python3
"""Photometric camera-pose refinement with gsplat (gsplat venv, GPU): 3DGRUT has no pose optimisation.

`refine`: trains a gsplat 3DGS model on the TRAINING frames of a COLMAP text dataset (LiDAR points3D.txt
init, optional LiDAR depth loss from <image>_depth.npy = ray distance, as patches/3dgrut-depth-loss.patch)
jointly with a per-camera SE(3) correction (c2w <- c2w @ exp(d), small Adam lr, weight decay). The free
gauge (scene + all cameras moving rigidly) is held by a penalty on the mean correction and then removed
exactly: the rigid G that best maps refined onto input poses (chordal-mean rotation, mean centre) is applied
to all refined poses, so they stay in the frame of points3D.txt and the depth targets. Writes:
  <out>/sparse/0/{cameras,images,points3D}.txt  refined TRAINING poses; every other frame corrected without
                                               its pixels (below); images symlinked, other per-image files
                                               (_depth.npy, _covis.png, ...) and *.json COPIED (so rebuilding
                                               depth targets there never writes through a symlink)
  each --test-dataset SRC DST: a copy of SRC with corrected poses (evaluation must use these: refining the
                               training cameras moves the frame the test cameras live in)
  <out>/posefix.json            correction magnitudes (dataset units, mm / deg), gauge, gaps, gsplat PSNR
                                of the non-training frames at input / gauge-only / corrected poses
  <out>/posefix_preview.jpg     GT | gsplat at input pose | at corrected pose for a few non-training frames
Non-training frames: the camera-local correction c2w_in^-1 c2w_ref of the two temporally neighbouring
training frames (frames in name order; leading digits of the name = timestamp) is interpolated (lerp
translation, slerp rotation); with a neighbour on one side only, the nearest one's correction is used.
On-path (3DGRUT's every-8th rule) the neighbours are 1 frame away. For a spatial hold-out the frames of the
held-out cluster can be seconds from any training frame, so their correction is mostly the low-frequency
(drift) part of the neighbours' corrections; with --far-gap-s G and --far-policy identity, frames farther
than G seconds from any training frame keep the input pose instead (the gauge alone).
Training frames = 3DGRUT's rule for the same dataset: frame k (name order) is held out iff
--test-split-interval > 0 and k % interval == 0 (pass the value given to 3DGRUT's dataset.test_split_interval).

`tta` (diagnostic): test-time pose alignment of a FIXED splat (standard 3DGS .ply, e.g. export_gaussian_ply.py
of a 3DGRUT checkpoint, rendered here with gsplat) to each frame of a dataset, using that frame's pixels;
writes a copy of the dataset with the aligned poses for scoring with eval_views.py. Rendered with eps2d=0 (no 2D
dilation), which reproduces 3DGRUT/3DGUT renders of the same .ply to within ~1 dB (default 0.3: -4.5 dB). Apply to baseline and
candidate alike; never used for acceptance.

    python refine_poses_gsplat.py refine --dataset DS --out OUT --test-split-interval 8 [--depth-lambda 0.05] \
        [--iters 15000] [--pose-lr 1e-5] [--sim3-json sim3.json] [--test-dataset SRC DST ...]
    python refine_poses_gsplat.py tta --ply splat.ply --dataset TEST --out ALIGNED [--iters 300]
"""
from __future__ import annotations
import argparse, json, math, os, re, shutil, time
import numpy as np


# ---------------------------------------------------------------- COLMAP text IO + pose math (numpy)
def read_model(d):
    """cameras {id: (model, w, h, params)}, images [(name, image_id, cam_id, c2w 4x4)] in name order."""
    cams = {}
    for ln in open(os.path.join(d, "cameras.txt")):
        if ln.strip() and not ln.startswith("#"):
            p = ln.split(); cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]])
    imgs, lines = [], [ln for ln in open(os.path.join(d, "images.txt")) if not ln.startswith("#")]
    for ln in lines[::2]:
        p = ln.split()
        if len(p) < 10:
            continue
        w2c = np.eye(4); w2c[:3, :3] = qvec2rot(np.array([float(x) for x in p[1:5]])); w2c[:3, 3] = [float(x) for x in p[5:8]]
        imgs.append((p[9], int(p[0]), int(p[8]), np.linalg.inv(w2c)))
    return cams, sorted(imgs, key=lambda x: x[0])


def write_images_txt(path, imgs):
    with open(path, "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (pose-refined: refine_poses_gsplat.py)\n")
        for name, iid, cid, c2w in imgs:
            w2c = np.linalg.inv(c2w); q = rot2qvec(w2c[:3, :3]); t = w2c[:3, 3]
            f.write(f"{iid} {' '.join(f'{x:.12g}' for x in q)} {' '.join(f'{x:.12g}' for x in t)} {cid} {name}\n\n")


def qvec2rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot2qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z]); return q if w >= 0 else -q


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def rot_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def interp_delta(D0, D1, a):
    from scipy.spatial.transform import Rotation, Slerp
    D = np.eye(4); D[:3, 3] = (1 - a) * D0[:3, 3] + a * D1[:3, 3]
    D[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([D0[:3, :3], D1[:3, :3]])))([a]).as_matrix()[0]
    return D


def timestamps(names):
    """Seconds from the leading digits of each name (16 digits = us, 19 = ns, 13 = ms, 10 = s); else the index."""
    m = [re.match(r"(\d+)", os.path.basename(n)) for n in names]
    if not all(m):
        return np.arange(len(names), dtype=float), "index"
    digits = len(m[0].group(1)); div = {10: 1, 13: 1e3, 16: 1e6, 19: 1e9}.get(digits, 1.0)
    return np.array([int(x.group(1)) / div for x in m]), ("s" if digits in (10, 13, 16, 19) else "raw")


def copy_dataset(src, dst, imgs, cams_src):
    """dst = src with new images.txt: cameras/points3D/json copied, images symlinked, other per-image files copied."""
    s0, d0 = os.path.join(src, "sparse", "0"), os.path.join(dst, "sparse", "0")
    os.makedirs(d0, exist_ok=True); os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    for f in ("cameras.txt", "points3D.txt"):
        if os.path.exists(os.path.join(s0, f)):
            shutil.copyfile(os.path.join(s0, f), os.path.join(d0, f))
    write_images_txt(os.path.join(d0, "images.txt"), imgs)
    for f in os.listdir(src):
        if f.endswith(".json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    si, di = os.path.join(src, "images"), os.path.join(dst, "images")
    names = {x[0] for x in imgs}; stems = {os.path.splitext(n)[0] for n in names}
    for f in os.listdir(si):
        s, d = os.path.join(si, f), os.path.join(di, f)
        if os.path.lexists(d):
            os.remove(d)
        if f in names:
            os.symlink(os.path.realpath(s), d)
        elif any(f.startswith(st + "_") for st in stems):
            shutil.copyfile(os.path.realpath(s), d)                       # real copy: safe to overwrite later


# ---------------------------------------------------------------- torch helpers
def so3_exp(w):
    import torch
    th2 = (w * w).sum(-1, keepdim=True); th = th2.clamp_min(1e-12).sqrt()
    A = torch.where(th2 < 1e-8, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(th2 < 1e-8, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K = K - K.transpose(-1, -2)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + A[..., None] * K + B[..., None] * (K @ K)


def delta_mat(d):
    """(..., 6) [t, w] -> (..., 4, 4) camera-local correction."""
    import torch
    T = torch.zeros(d.shape[:-1] + (4, 4), device=d.device, dtype=d.dtype)
    T[..., :3, :3] = so3_exp(d[..., 3:]); T[..., :3, 3] = d[..., :3]; T[..., 3, 3] = 1
    return T


def ssim(a, b):
    """a, b: (1, 3, H, W) in [0, 1]; 11x11 gaussian SSIM."""
    import torch, torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    k = (g[:, None] @ g[None]).expand(3, 1, 11, 11)
    f = lambda x: F.conv2d(x, k, padding=5, groups=3)
    mu_a, mu_b = f(a), f(b)
    s_a, s_b, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))).mean()


def intrinsics(cam):
    model, w, h, p = cam
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1:3]
        if model != "SIMPLE_PINHOLE":
            print(f"[posefix] WARNING: {model} distortion ignored")
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]; print(f"[posefix] WARNING: {model} distortion ignored")
    else:
        raise SystemExit(f"unsupported camera model {model}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), w, h


def load_rgb(path, dev):
    import torch
    from PIL import Image
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).to(dev)          # uint8 (H, W, 3)


class Splat:
    """Gaussian parameters (gsplat conventions: log-scales, logit opacities, wxyz quats, SH (N, K, 3))."""

    def __init__(self, params):
        self.params = params

    @staticmethod
    def from_points(xyz, rgb, dev, init_opacity=0.1, init_scale=1.0):
        import torch
        from scipy.spatial import cKDTree
        d, _ = cKDTree(xyz).query(xyz, k=4)
        s = np.log(np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-7) * init_scale)
        n = len(xyz); C0 = 0.28209479177387814
        p = {"means": torch.tensor(xyz, dtype=torch.float32), "scales": torch.tensor(s, dtype=torch.float32)[:, None].repeat(1, 3),
             "quats": torch.rand(n, 4), "opacities": torch.logit(torch.full((n,), init_opacity)),
             "sh0": ((torch.tensor(rgb, dtype=torch.float32) - 0.5) / C0)[:, None], "shN": torch.zeros(n, 15, 3)}
        return Splat(torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()}).to(dev))

    @staticmethod
    def from_ply(path, dev):
        import torch
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]; names = [p.name for p in v.properties]
        col = lambda pre: np.stack([np.asarray(v[n]) for n in sorted((n for n in names if n.startswith(pre)),
                                                                       key=lambda x: int(x.split("_")[-1]))], 1)
        rest = col("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(v), 0))
        K = rest.shape[1] // 3; rest = rest.reshape(len(v), 3, K).transpose(0, 2, 1)
        p = {"means": np.stack([v["x"], v["y"], v["z"]], 1), "scales": col("scale_"), "quats": col("rot_"),
             "opacities": np.asarray(v["opacity"]), "sh0": col("f_dc_")[:, None], "shN": rest}
        return Splat({k: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev) for k, a in p.items()})

    @property
    def sh_degree_max(self):
        return int(round(math.sqrt(self.params["shN"].shape[1] + 1))) - 1

    def render(self, c2w, K, w, h, sh_degree, depth=False, absgrad=False, eps2d=0.3):
        import torch
        from gsplat import rasterization
        p = self.params
        colors = torch.cat([p["sh0"], p["shN"][:, :(sh_degree + 1) ** 2 - 1]], 1)
        out, alpha, info = rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                         colors, torch.linalg.inv(c2w)[None], K[None], w, h, sh_degree=sh_degree,
                                         packed=False, absgrad=absgrad, eps2d=eps2d, render_mode="RGB+ED" if depth else "RGB")
        return out[0], alpha[0], info


def psnr(a, b):
    import torch
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))


# ---------------------------------------------------------------- refine
def refine(args):
    import torch
    from gsplat.strategy import DefaultStrategy
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    s0 = os.path.join(args.dataset, "sparse", "0")
    cams, imgs = read_model(s0)
    n = len(imgs); names = [x[0] for x in imgs]
    iv = args.test_split_interval
    train = np.array([not (iv > 0 and k % iv == 0) for k in range(n)])
    tr_idx = np.where(train)[0]
    print(f"[posefix] {n} frames, {len(tr_idx)} training (test_split_interval={iv}), {n - len(tr_idx)} corrected by interpolation")
    Ks = {}
    for cid, cam in cams.items():
        K, w, h = intrinsics(cam)
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        Ks[cid] = (torch.tensor(K, dtype=torch.float32, device=dev), w, h,
                   torch.tensor(np.sqrt(((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2 + 1),
                                dtype=torch.float32, device=dev))      # |ray| per pixel: ray distance = z * |ray|
    img_dir = os.path.join(args.dataset, "images")
    rgb = [load_rgb(os.path.join(img_dir, nm), dev) for nm in names]
    dep = [None] * n
    if args.depth_lambda > 0:
        for k, nm in enumerate(names):
            f = os.path.join(img_dir, os.path.splitext(nm)[0] + "_depth.npy")
            if os.path.exists(f):
                dep[k] = torch.from_numpy(np.load(f).astype(np.float32)).to(dev)
        print(f"[posefix] depth targets for {sum(d is not None for d in dep)}/{n} frames, lambda {args.depth_lambda}")
    c2w_in = torch.tensor(np.stack([x[3] for x in imgs]), dtype=torch.float32, device=dev)

    pts = [ln.split() for ln in open(os.path.join(s0, "points3D.txt")) if ln.strip() and not ln.startswith("#")]
    xyz = np.array([[float(x) for x in p[1:4]] for p in pts]); col = np.array([[float(x) for x in p[4:7]] for p in pts]) / 255
    if args.max_points and len(xyz) > args.max_points:
        sel = np.random.choice(len(xyz), args.max_points, replace=False); xyz, col = xyz[sel], col[sel]
    centres = c2w_in[train][:, :3, 3].cpu().numpy()
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    print(f"[posefix] init {len(xyz)} points, scene_scale {scene_scale:.3f}")
    splat = Splat.from_points(xyz, col, dev, init_scale=args.init_scale); params = splat.params
    lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr[k], "name": k}], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / args.iters))
    strategy = DefaultStrategy(absgrad=args.absgrad, grow_grad2d=0.0008 if args.absgrad else 0.0002,
                               refine_stop_iter=int(args.iters * 0.5), reset_every=3000)
    strategy.check_sanity(params, opts); state = strategy.initialize_state(scene_scale=scene_scale)
    # --rig-group REGEX: frames whose names give the same group (first capture group, e.g. the shared timestamp of a
    # stereo pair) share ONE correction, applied to the group's reference frame (first in name order) and carried
    # rigidly to the others: c2w_k <- c2w_ref @ D_g @ c2w_ref^-1 @ c2w_k (the rig keeps its calibrated extrinsics)
    if args.rig_group:
        keys = [re.search(args.rig_group, os.path.basename(nm)) for nm in names]
        if not all(keys):
            raise SystemExit(f"[posefix] --rig-group {args.rig_group!r} does not match every frame name")
        keys = [m.group(1) for m in keys]; uk = sorted(set(keys), key=keys.index); gidx = {k_: i for i, k_ in enumerate(uk)}
        gi = np.array([gidx[k_] for k_ in keys]); ref = np.array([keys.index(k_) for k_ in keys])
        print(f"[posefix] rig groups: {len(uk)} groups for {n} frames (shared correction per group)")
    else:
        gi = np.arange(n); ref = np.arange(n)
    gi_t, ref_t = torch.tensor(gi, device=dev), torch.tensor(ref, device=dev)
    rel = torch.linalg.inv(c2w_in[ref_t]) @ c2w_in                      # frame in its group's reference frame
    delta = torch.nn.Parameter(torch.zeros(int(gi.max()) + 1, 6, device=dev))
    popt = torch.optim.Adam([delta], lr=args.pose_lr, weight_decay=args.pose_reg)
    trw = torch.zeros(len(delta), 1, device=dev); trw[torch.tensor(gi[train], device=dev)] = 1.0   # groups with training frames

    def pose(k):
        return c2w_in[ref[k]] @ delta_mat(delta[gi[k]]) @ rel[k]
    order, t0 = [], time.time()
    for step in range(args.iters):
        if not order:
            order = list(np.random.permutation(tr_idx))
        k = int(order.pop()); cid = imgs[k][2]; K, w, h, rn = Ks[cid]
        pose_on = step >= args.pose_start
        c2w = pose(k) if pose_on else c2w_in[k]
        sh = min(step // 1000, 3)
        out, alpha, info = splat.render(c2w, K, w, h, sh, depth=args.depth_lambda > 0, absgrad=args.absgrad)
        strategy.step_pre_backward(params, opts, state, step, info)
        gt = rgb[k].float() / 255; pred = out[..., :3]
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        ld = torch.zeros((), device=dev)
        if dep[k] is not None:
            m = dep[k] > 0
            if m.any():
                err = ((out[..., 3] * rn)[m] - dep[k][m]).abs()
                if args.depth_trim < 1:
                    err = err.sort().values[: max(1, int(err.numel() * args.depth_trim))]
                ld = err.mean(); loss = loss + args.depth_lambda * ld
        if pose_on:                                       # gauge: no mean correction over the training cameras
            md = (delta * trw).sum(0) / trw.sum()
            loss = loss + args.gauge_weight * (md ** 2).sum()
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        if pose_on:
            popt.step()
        popt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        if step % 1000 == 0 or step == args.iters - 1:
            dn = delta.detach()[trw[:, 0] > 0]
            print(f"[posefix] step {step:6d} loss {loss.item():.4f} psnr {psnr(pred.detach(), gt):.2f} depth {ld.item():.4f} "
                  f"N {len(params['means'])} |dt| med {dn[:, :3].norm(dim=1).median().item():.2e} "
                  f"|dw| med {dn[:, 3:].norm(dim=1).median().item():.2e} ({time.time() - t0:.0f}s)", flush=True)

    # ---- refined poses, gauge removal, corrections
    with torch.no_grad():
        c2w_ref = (c2w_in[ref_t] @ delta_mat(delta[gi_t]) @ rel).double().cpu().numpy()
    Cin = np.stack([x[3] for x in imgs]).astype(np.float64)
    G = np.eye(4)
    if not args.keep_gauge:
        tr = tr_idx
        G[:3, :3] = proj_so3(sum(Cin[i, :3, :3] @ c2w_ref[i, :3, :3].T for i in tr))
        G[:3, 3] = (Cin[tr, :3, 3] - c2w_ref[tr, :3, 3] @ G[:3, :3].T).mean(0)
    c2w_fin = np.einsum("ij,njk->nik", G, c2w_ref)
    D = {int(i): np.linalg.inv(Cin[i]) @ c2w_fin[i] for i in tr_idx}            # camera-local corrections
    ts, tunit = timestamps(names)
    t_tr = ts[tr_idx]

    def correct(name, c2w, t):
        """Corrected pose of a frame not trained on, from the temporally neighbouring training frames."""
        j = np.searchsorted(t_tr, t)
        lo, hi = (tr_idx[j - 1] if j > 0 else None), (tr_idx[j] if j < len(tr_idx) else None)
        gap = min(abs(t - ts[i]) for i in (lo, hi) if i is not None)
        if args.far_gap_s and gap > args.far_gap_s and args.far_policy == "identity":
            Dk = np.linalg.inv(c2w) @ G @ c2w                                    # the gauge alone
        elif lo is not None and hi is not None and ts[hi] > ts[lo]:
            Dk = interp_delta(D[int(lo)], D[int(hi)], (t - ts[lo]) / (ts[hi] - ts[lo]))
        else:
            Dk = D[int(lo if lo is not None else hi)]
        return c2w @ Dk, gap

    out_imgs, gaps = [], {}
    for k, (nm, iid, cid, c2w) in enumerate(imgs):
        if train[k]:
            out_imgs.append((nm, iid, cid, c2w_fin[k]))
        else:
            cc, gaps[nm] = correct(nm, c2w.astype(np.float64), ts[k]); out_imgs.append((nm, iid, cid, cc))
    copy_dataset(args.dataset, args.out, out_imgs, cams)
    mscale = 1.0
    if args.sim3_json and os.path.exists(args.sim3_json):
        mscale = float(json.load(open(args.sim3_json))["scale"]) * 1000           # dataset units -> mm

    def mag(pairs):
        dt = np.array([np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in pairs]) * mscale
        dr = np.array([rot_deg(a[:3, :3].T @ b[:3, :3]) for a, b in pairs])
        q = lambda x: {"median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4),
                       "max": round(float(x.max()), 4)} if len(x) else None
        return {"translation_" + ("mm" if mscale != 1 else "units"): q(dt), "rotation_deg": q(dr)}

    rep = {"dataset": args.dataset, "out": args.out, "frames": n, "train_frames": int(train.sum()),
           "test_split_interval": iv, "iters": args.iters, "pose_lr": args.pose_lr, "depth_lambda": args.depth_lambda,
           "time_unit": tunit, "gauge": {"rotation_deg": round(rot_deg(G[:3, :3]), 5),
                                         "translation": round(float(np.linalg.norm(G[:3, 3]) * mscale), 4)},
           "train_correction": mag([(Cin[i], c2w_fin[i]) for i in tr_idx]),
           "train_correction_before_gauge_removal": mag([(Cin[i], c2w_ref[i]) for i in tr_idx]),
           "interp_correction": mag([(Cin[k], out_imgs[k][3]) for k in range(n) if not train[k]]),
           "interp_gap_s": {"median": float(np.median(list(gaps.values()))), "max": float(max(gaps.values()))} if gaps else None,
           "per_frame": {nm: {"train": bool(train[k]), "dt": round(float(np.linalg.norm(out_imgs[k][3][:3, 3] - Cin[k, :3, 3]) * mscale), 4),
                              "dr_deg": round(rot_deg(Cin[k, :3, :3].T @ out_imgs[k][3][:3, :3]), 5)}
                         for k, nm in enumerate(names)}}

    # ---- gsplat diagnostics: non-training frames at input / gauge-only / corrected poses (model frame = G^-1 final)
    Gi = np.linalg.inv(G); tt = lambda M: torch.tensor(M, dtype=torch.float32, device=dev)
    sh = min(args.iters // 1000, 3); prev = []

    def diag(frames):
        r = {"input": [], "gauge_only": [], "corrected": []}
        with torch.no_grad():
            for nm, cid, c_in, c_corr, gt in frames:
                K, w, h, _ = Ks[cid]
                ims = [splat.render(tt(P), K, w, h, sh)[0][..., :3].clamp(0, 1) for P in (c_in, Gi @ c_in, Gi @ c_corr)]
                for key, im in zip(r, ims):
                    r[key].append(psnr(im, gt))
                if len(prev) < 4 and len(frames) > 0 and nm in sel_prev:
                    prev.append(torch.cat([gt, ims[0], ims[2]], 1).cpu().numpy())
        return {k: round(float(np.mean(v)), 3) for k, v in r.items() if v} | {"n": len(frames)}

    held = [(imgs[k][0], imgs[k][2], Cin[k], out_imgs[k][3], rgb[k].float() / 255) for k in range(n) if not train[k]]
    sel_prev = {f[0] for f in held[:: max(1, len(held) // 4)]}
    if held:
        rep["gsplat_psnr_heldout_in_dataset"] = diag(held)
    rep["gsplat_psnr_train"] = diag([(imgs[k][0], imgs[k][2], Cin[k], c2w_fin[k], rgb[k].float() / 255) for k in tr_idx[::10]])
    rep["test_datasets"] = {}
    for src, dst in args.test_dataset or []:
        tcams, timgs = read_model(os.path.join(src, "sparse", "0"))
        tts, _ = timestamps([x[0] for x in timgs])
        tout, tg = [], []
        for (nm, iid, cid, c2w), t in zip(timgs, tts):
            cc, g = correct(nm, c2w.astype(np.float64), t); tout.append((nm, iid, cid, cc)); tg.append(g)
        copy_dataset(src, dst, tout, tcams)
        frames = [(x[0], x[2], x[3], y[3], load_rgb(os.path.join(src, "images", x[0]), dev).float() / 255)
                  for x, y in zip(timgs, tout) if x[2] in Ks]
        sel_prev |= {f[0] for f in frames[:: max(1, len(frames) // 4)]} if not held else set()
        rep["test_datasets"][dst] = {"src": src, "correction": mag([(x[3], y[3]) for x, y in zip(timgs, tout)]),
                                     "gap_s": {"median": float(np.median(tg)), "max": float(max(tg))},
                                     "gsplat_psnr": diag(frames)}
        print(f"[posefix] corrected test dataset -> {dst}")
    if prev:
        import cv2
        cv2.imwrite(os.path.join(args.out, "posefix_preview.jpg"),
                    cv2.cvtColor((np.vstack(prev) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if args.save_ply:
        save_ply(splat, os.path.join(args.out, "gsplat_refine.ply"), G)
    json.dump(rep, open(os.path.join(args.out, "posefix.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_frame"}, indent=1))
    print("POSEFIX_REFINE_DONE")


def save_ply(splat, path, G=np.eye(4)):
    """Standard 3DGS .ply of the gsplat model, moved into the output frame (means by G, rotations by G)."""
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    p = {k: v.detach().cpu().numpy() for k, v in splat.params.items()}
    m = p["means"] @ G[:3, :3].T + G[:3, 3]
    q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    q = (Rotation.from_matrix(G[:3, :3]) * Rotation.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    rest = p["shN"].transpose(0, 2, 1).reshape(len(m), -1)                        # (rotating SH ignored: diag only)
    cols = [m, np.zeros_like(m), p["sh0"][:, 0], rest, p["opacities"][:, None], p["scales"], q]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.concatenate(cols, 1).astype(np.float32)
    el = np.empty(len(arr), dtype=[(nm, "f4") for nm in names]); el[:] = list(map(tuple, arr))
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


# ---------------------------------------------------------------- tta (diagnostic)
def tta(args):
    import torch
    dev = "cuda"
    splat = Splat.from_ply(args.ply, dev); sh = splat.sh_degree_max
    cams, imgs = read_model(os.path.join(args.dataset, "sparse", "0"))
    out_imgs, before, after, dts, drs = [], [], [], [], []
    for nm, iid, cid, c2w in imgs:
        K, w, h = intrinsics(cams[cid]); K = torch.tensor(K, dtype=torch.float32, device=dev)
        gt = load_rgb(os.path.join(args.dataset, "images", nm), dev).float() / 255
        C = torch.tensor(c2w, dtype=torch.float32, device=dev)
        d = torch.nn.Parameter(torch.zeros(6, device=dev))
        opt = torch.optim.Adam([{"params": [d], "lr": args.lr}])
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / args.iters))
        for it in range(args.iters + 1):
            pred = splat.render(C @ delta_mat(d), K, w, h, sh, eps2d=0.0)[0][..., :3].clamp(0, 1)   # no 2D dilation: matches 3DGUT renders
            if it == 0:
                before.append(psnr(pred.detach(), gt))
            if it == args.iters:
                after.append(psnr(pred.detach(), gt)); break
            loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        Dn = delta_mat(d.detach()).double().cpu().numpy(); dts.append(np.linalg.norm(Dn[:3, 3])); drs.append(rot_deg(Dn[:3, :3]))
        out_imgs.append((nm, iid, cid, c2w @ Dn))
    copy_dataset(args.dataset, args.out, out_imgs, cams)
    rep = {"ply": args.ply, "dataset": args.dataset, "iters": args.iters, "n": len(imgs),
           "gsplat_psnr_before": round(float(np.mean(before)), 3), "gsplat_psnr_after": round(float(np.mean(after)), 3),
           "dt_units_median": float(np.median(dts)), "dr_deg_median": float(np.median(drs)), "dr_deg_max": float(np.max(drs))}
    json.dump(rep, open(os.path.join(args.out, "tta.json"), "w"), indent=2)
    print(json.dumps(rep)); print("POSEFIX_TTA_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refine")
    r.add_argument("--dataset", required=True, help="COLMAP text dataset (sparse/0/*.txt + images/)")
    r.add_argument("--out", required=True)
    r.add_argument("--test-split-interval", type=int, default=0, help="3DGRUT's rule: frame k held out iff k %% N == 0")
    r.add_argument("--test-dataset", nargs=2, action="append", metavar=("SRC", "DST"))
    r.add_argument("--iters", type=int, default=15000)
    r.add_argument("--pose-lr", type=float, default=1e-5)
    r.add_argument("--pose-reg", type=float, default=1e-6, help="Adam weight decay on the corrections")
    r.add_argument("--rig-group", default=None, help="regex; frames with the same first group share one correction (rig)")
    r.add_argument("--pose-start", type=int, default=500)
    r.add_argument("--gauge-weight", type=float, default=1e3, help="penalty on the mean correction")
    r.add_argument("--keep-gauge", action="store_true", help="do not remove the global rigid drift afterwards")
    r.add_argument("--depth-lambda", type=float, default=0.05, help="0 = no depth loss")
    r.add_argument("--depth-trim", type=float, default=0.95)
    r.add_argument("--absgrad", action="store_true")
    r.add_argument("--init-scale", type=float, default=1.0)
    r.add_argument("--max-points", type=int, default=0)
    r.add_argument("--far-gap-s", type=float, default=0.0)
    r.add_argument("--far-policy", choices=("interp", "identity"), default="interp")
    r.add_argument("--sim3-json", default=None, help="COLMAP->metric Sim3 (reports corrections in mm)")
    r.add_argument("--save-ply", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("tta")
    t.add_argument("--ply", required=True); t.add_argument("--dataset", required=True); t.add_argument("--out", required=True)
    t.add_argument("--iters", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    refine(args) if args.cmd == "refine" else tta(args)


if __name__ == "__main__":
    main()
