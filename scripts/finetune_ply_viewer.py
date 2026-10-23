#!/usr/bin/env python3
"""Fine-tune an EXPORTED 3DGS .ply for the viewer that will display it (gsplat venv, GPU).

3DGRUT/3DGUT models are optimised for 3DGUT's renderer; a standard EWA viewer draws the same .ply differently
(no unscented projection, a screen-space dilation, a kernel cut-off, a minimum splat size), so the exported
splat loses several dB in the viewer. This script re-optimises the .ply's parameters (positions, log-scales,
quaternions, opacity logits, SH dc+rest; no densification) on the TRAINING images of a COLMAP text dataset,
rendered by a differentiable copy of the viewer's rasteriser, and writes a .ply with the input's exact layout.

Viewer profiles (--viewer), reimplemented from the shader sources:
  pc274   PlayCanvas engine 2.7.4 gsplat (what scripts/viewer_shots.py renders). The engine computes the 2D
          covariance with focal = viewport * proj00 (= 2 f in pixels) and adds 0.3 in those units, i.e. a
          0.075 px^2 dilation; quad half-extent sqrt(8) sigma (capped at 1024 px), kernel exp(-d^2/2) cut at
          d^2 = 8 and at alpha 1/255; splats with max eigenvalue < 0.125 px^2 (after dilation) are dropped; SH
          band 2 has the engine's operator bug (`sh[3]*(C*xy) * +sh[4]*(C*yz)`: the two terms multiply).
  ss352   SuperSplat 3.5.2 editor (PlayCanvas 2.23, its own WebGPU "projected splat" renderer): true pixel
          focal, 0.3 px^2 dilation, half-extent sqrt(8) sigma (radius capped at min(1024, viewport)),
          normalised kernel (exp(-d^2/2) - e^-4) / (1 - e^-4) cut at d^2 = 8, no 1/255 cut, splats with
          2 sqrt(2 lambda1) < 2 px dropped, opacity quantised to 8 bits, standard SH.
Both: per-gaussian view-depth sort, premultiplied over-blending onto black, colour clamped >= 0, output [0,1].
Rasterisers (--raster): `exact` = gsplat's CUDA tile/indices kernels + per-pixel compositing in torch with
the viewer's kernel (exact cut-offs, autograd); `cuda` = gsplat's fused rasteriser fed the viewer's
projection (fast; keeps the exp tail beyond d^2 = 8 and gsplat's 0.999 alpha clamp, so pc274 only).

Subcommands:
  compare   render a .ply at a dataset's cameras with several renderer variants and score each against the
            viewer's own renders of it (viewer_shots.py output) -> renderer agreement (PSNR, dB)
  finetune  the optimisation; --test-split-interval N excludes frame k (name order) iff k % N == 0, exactly
            3DGRUT's dataset_colmap.py split; optional LiDAR depth term where <image>_depth.npy exists
  render    write <out-dir>/<stem>.png of a .ply through the matched renderer (debug / quick scoring); --ply may be a
            .compressed.ply (decoded as SuperSplat does); --quant-format cply simulates compressing --ply
  cply-check  quantise_cply(--ply) vs splat-transform's real .compressed.ply of it (bit-exactness of the simulation)
Compression-aware fine-tune (finetune --quant-format cply): the user opens splat-transform's .compressed.ply (chunks
of 256 Morton-ordered splats, 11/10/11-bit positions and log-scales within each chunk's box, 8-bit colour/opacity,
10-bit smallest-three quaternions, 8-bit SH over [-4, 4]); rendering the decoded values with straight-through
gradients makes the float .ply optimise for what that file shows.

    python finetune_ply_viewer.py compare --ply s.ply --dataset TEST --viewer-dir shots/ [--every 4] --json-out a.json
    python finetune_ply_viewer.py finetune --ply s.ply --dataset DS --out ft.ply --test-split-interval 8 \
        [--viewer pc274] [--iters 7000] [--depth-lambda 0] [--eval-dataset TEST]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses_gsplat import read_model, intrinsics, ssim, load_rgb, psnr  # noqa: E402

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)
E4 = math.exp(-4.0)


# ---------------------------------------------------------------- PLY IO (layout preserved)
def load_ply(path, dev):
    """-> (plyfile vertex element, {means, scales (log), quats (wxyz raw), opacities (logit), sh0 (N,1,3), shN (N,K,3)})."""
    import torch
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]; names = [p.name for p in v.properties]
    col = lambda pre: np.stack([np.asarray(v[n], dtype=np.float32) for n in
                                sorted((n for n in names if n.startswith(pre)), key=lambda x: int(x.split("_")[-1]))], 1)
    rest = col("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(v), 0), np.float32)
    K = rest.shape[1] // 3
    p = {"means": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32), "scales": col("scale_"), "quats": col("rot_"),
         "opacities": np.asarray(v["opacity"], dtype=np.float32), "sh0": col("f_dc_")[:, None],
         "shN": rest.reshape(len(v), 3, K).transpose(0, 2, 1)}
    return v, {k: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev) for k, a in p.items()}


def save_ply(path, vertex, params):
    """Write params into a copy of the input vertex array (same properties, order, dtypes, header comments)."""
    from plyfile import PlyData, PlyElement
    p = {k: v.detach().float().cpu().numpy() for k, v in params.items()}
    a = (vertex if isinstance(vertex, np.ndarray) else vertex.data).copy(); n = len(a)    # PlyElement or structured array
    for i, c in enumerate("xyz"):
        a[c] = p["means"][:, i]
    for i in range(3):
        a[f"scale_{i}"] = p["scales"][:, i]; a[f"f_dc_{i}"] = p["sh0"][:, 0, i]
    for i in range(4):
        a[f"rot_{i}"] = p["quats"][:, i]
    a["opacity"] = p["opacities"]
    rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)
    for i in range(rest.shape[1]):
        a[f"f_rest_{i}"] = rest[:, i]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(a, "vertex")], text=False).write(path)


# ---------------------------------------------------------------- the viewer's rasteriser, differentiable
def quat_to_rot(q):
    import torch
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def eval_sh(sh0, shN, dirs, degree, pc_bug):
    """RGB = SH_C0*dc + 0.5 + higher bands (PlayCanvas/3DGS basis), clamped >= 0. pc_bug: 2.7.4's band-2 typo."""
    import torch
    c = sh0[:, 0] * C0 + 0.5
    if degree > 0 and shN.shape[1] >= 3:
        x, y, z = [d[:, None] for d in dirs.unbind(-1)]
        s = shN
        c = c + C1 * (-s[:, 0] * y + s[:, 1] * z - s[:, 2] * x)
        if degree > 1 and shN.shape[1] >= 8:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            t01 = (s[:, 3] * (C2[0] * xy)) * (s[:, 4] * (C2[1] * yz)) if pc_bug else s[:, 3] * (C2[0] * xy) + s[:, 4] * (C2[1] * yz)
            c = c + t01 + s[:, 5] * (C2[2] * (2 * zz - xx - yy)) + s[:, 6] * (C2[3] * xz) + s[:, 7] * (C2[4] * (xx - yy))
            if degree > 2 and shN.shape[1] >= 15:
                c = c + (s[:, 8] * (C3[0] * y * (3 * xx - yy)) + s[:, 9] * (C3[1] * xy * z) + s[:, 10] * (C3[2] * y * (4 * zz - xx - yy))
                         + s[:, 11] * (C3[3] * z * (2 * zz - 3 * xx - 3 * yy)) + s[:, 12] * (C3[4] * x * (4 * zz - xx - yy))
                         + s[:, 13] * (C3[5] * z * (xx - yy)) + s[:, 14] * (C3[6] * x * (xx - 3 * yy)))
    return c.clamp_min(0)


def half(t):
    import torch
    return t.to(torch.float16).to(torch.float32)


def quantise_pc274(p):
    """The engine's texture storage: half-float colour/opacity/scale/rotation, SH 11-10-11 bits with per-splat max
    (incl. its `max = c[0]` seed quirk). Forward-only (compare)."""
    import torch
    q = dict(p)
    q["means"] = p["means"]
    q["scales"] = torch.log(half(torch.exp(p["scales"])).clamp_min(1e-30))
    qn = p["quats"] / p["quats"].norm(dim=-1, keepdim=True); qn = torch.where(qn[:, :1] < 0, -qn, qn)
    q["quats"] = torch.cat([torch.sqrt((1 - (half(qn[:, 1:]) ** 2).sum(-1, keepdim=True)).clamp_min(0)), half(qn[:, 1:])], 1)
    a = half(torch.sigmoid(p["opacities"])).clamp(1e-7, 1 - 1e-7); q["opacities"] = torch.log(a / (1 - a))
    q["sh0"] = (half(p["sh0"] * C0 + 0.5) - 0.5) / C0
    s = p["shN"]
    if s.shape[1]:
        c = s.reshape(len(s), -1)                                 # engine order c[j*3+ch]: coeff j, channel ch
        mx = torch.maximum(c[:, :1], c[:, 1:].abs().amax(1, keepdim=True))
        bits = torch.tensor([2047.0, 1023.0, 2047.0], device=s.device).repeat(s.shape[1])
        qq = torch.floor((c / mx.clamp_min(1e-30) * 0.5 + 0.5) * bits + 0.5).clamp(min=torch.zeros_like(bits), max=bits)
        deq = (qq / bits * 2 - 1) * mx
        deq = torch.where(mx > 0, deq, torch.zeros_like(deq))
        q["shN"] = deq.reshape(s.shape)
    return q


# ---------------------------------------------------------------- PlayCanvas .compressed.ply (splat-transform 2.7.x)
CPLY_CHUNK = 256
CPLY_PARTS = ("pos", "rot", "scale", "color", "opacity", "sh")


def _part1by2(x):
    x = x.astype(np.uint32) & np.uint32(0x3FF)
    for s, m in ((16, 0xFF0000FF), (8, 0x0300F00F), (4, 0x030C30C3), (2, 0x09249249)):
        x = (x ^ (x << np.uint32(s))) & np.uint32(m)
    return x


def cply_order(means):
    """splat-transform's sortMortonOrder: 10-bit Morton code over the bounding box (float64 maths on the float32
    centres), stable sort, buckets of > 256 equal codes re-sorted recursively over their own box -> row order (N,)."""
    xyz = np.asarray(means, dtype=np.float32).astype(np.float64)
    idx = np.arange(len(xyz), dtype=np.int64)

    def gen(ix):
        if len(ix) == 0:
            return ix
        p = xyz[ix]; lo, hi = p.min(0), p.max(0); ln = hi - lo
        if not np.isfinite(ln).all() or (ln == 0).all():
            return ix
        mul = np.where(ln == 0, 0.0, 1024.0 / np.where(ln == 0, 1.0, ln))
        q = np.minimum(1023.0, (p - lo) * mul).astype(np.uint32)
        code = (_part1by2(q[:, 2]) << np.uint32(2)) + (_part1by2(q[:, 1]) << np.uint32(1)) + _part1by2(q[:, 0])
        o = np.argsort(code, kind="stable"); ix, code = ix[o], code[o]
        b = np.flatnonzero(np.r_[True, code[1:] != code[:-1], True])
        for s, e in zip(b[:-1], b[1:]):
            if e - s > CPLY_CHUNK:
                ix[s:e] = gen(ix[s:e])
        return ix
    return gen(idx)


def quantise_cply(p, order, parts=CPLY_PARTS, ste=True):
    """What SuperSplat draws after `splat-transform x.ply x.compressed.ply` (writeCompressedPly / CompressedChunk.pack
    and the reader's decompressPly): rows in `order` (cply_order) cut into chunks of 256; per chunk min/max of x/y/z,
    log-scales (clamped to +-20) and SH-dc colour (dc C0 + 0.5) with 11-10-11 / 11-10-11 / 8-8-8 bit unorm codes;
    opacity sigmoid in 8 bits; quaternion smallest-three 2+10+10+10 bits; f_rest 8 bits over [-4, 4] (truncating
    encoder, centred decoder). ste: straight-through (forward = decoded values, gradient = identity; chunk bounds are
    constants). parts: subset of CPLY_PARTS to quantise (attribution)."""
    import torch
    q = dict(p)
    N = len(p["means"]); inv = None
    nch = (N + CPLY_CHUNK - 1) // CPLY_CHUNK
    o = torch.as_tensor(order, device=p["means"].device)
    pad = nch * CPLY_CHUNK - N                                   # the encoder repeats the last row: bounds unchanged

    def chunked(x):                                              # (N,C) -> (nch,256,C) in file order
        y = x[o]
        if pad:
            y = torch.cat([y, y[-1:].expand(pad, -1)], 0)
        return y.reshape(nch, CPLY_CHUNK, -1)

    def unchunk(y):
        nonlocal inv
        if inv is None:
            inv = torch.empty_like(o); inv[o] = torch.arange(N, device=o.device)
        return y.reshape(nch * CPLY_CHUNK, -1)[:N][inv]

    def unorm(x, bits, lo=None, hi=None):                        # chunk min/max + packUnorm + lerp decode (float64)
        c = chunked(x.detach().double())
        mn = c.amin(1, keepdim=True) if lo is None else c.amin(1, keepdim=True).clamp(lo, hi)
        mx = c.amax(1, keepdim=True) if lo is None else c.amax(1, keepdim=True).clamp(lo, hi)
        r = mx - mn
        t = torch.where(r < 1e-5, torch.zeros_like(c), ((c - mn) / torch.where(r < 1e-5, torch.ones_like(r), r)).clamp(0, 1))
        t = torch.where(c <= mn, torch.zeros_like(t), torch.where(c >= mx, torch.ones_like(t), t))
        L = torch.tensor(bits, dtype=torch.float64, device=x.device)
        k = torch.floor(t * L + 0.5).clamp(min=torch.zeros_like(L), max=L) / L
        return unchunk((mn * (1 - k) + mx * k).float())

    def st(x, xq):
        return x + (xq - x).detach() if ste else xq.detach()
    if "pos" in parts:
        q["means"] = st(p["means"], unorm(p["means"], [2047.0, 1023.0, 2047.0]))
    if "scale" in parts:
        q["scales"] = st(p["scales"], unorm(p["scales"], [2047.0, 1023.0, 2047.0], -20.0, 20.0))
    if "color" in parts:
        c = p["sh0"][:, 0] * C0 + 0.5
        q["sh0"] = st(p["sh0"], ((unorm(c, [255.0] * 3) - 0.5) / C0)[:, None])
    if "opacity" in parts:
        a = torch.floor(torch.sigmoid(p["opacities"].detach().double()) * 255 + 0.5).clamp(0, 255) / 255
        q["opacities"] = st(p["opacities"], torch.logit(a.clamp(1e-7, 1 - 1e-7)).float())
    if "rot" in parts:
        v = p["quats"].detach().double(); v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        big = v.abs().argmax(-1, keepdim=True)                   # first index of the largest |component|
        v = torch.where(torch.gather(v, 1, big) < 0, -v, v)
        k = torch.floor((v * (math.sqrt(2) * 0.5) + 0.5) * 1023 + 0.5).clamp(0, 1023)
        d = (k / 1023 - 0.5) * math.sqrt(2)
        oh = torch.zeros_like(d, dtype=torch.bool).scatter_(1, big, True)
        d = torch.where(oh, torch.zeros_like(d), d)
        m = torch.sqrt((1 - (d * d).sum(-1, keepdim=True)).clamp_min(0))
        d = torch.where(oh, m.expand_as(d), d)
        pn = p["quats"] / p["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sgn = torch.where(torch.gather(p["quats"].detach(), 1, big) < 0, -1.0, 1.0)
        q["quats"] = st(pn, (d * sgn).float())                  # same rotation as the input's sign convention
    if "sh" in parts and p["shN"].shape[1]:
        k = torch.floor((p["shN"].detach().double() / 8 + 0.5) * 256).clamp(0, 255)
        n = torch.where(k == 0, torch.zeros_like(k), torch.where(k == 255, torch.ones_like(k), (k + 0.5) / 256))
        q["shN"] = st(p["shN"], ((n - 0.5) * 8).float())
    return q


def read_cply(path):
    """Decode a .compressed.ply exactly like splat-transform's decompressPly -> dict of float arrays in FILE order."""
    from plyfile import PlyData
    d = PlyData.read(path); ch = d["chunk"].data; v = d["vertex"].data
    ci = np.arange(len(v)) // CPLY_CHUNK
    lerp = lambda a, b, t: a[ci] * (1 - t) + b[ci] * t                      # noqa: E731
    un = lambda x, s, bits: ((x.astype(np.uint64) >> np.uint64(s)) & np.uint64((1 << bits) - 1)).astype(np.float64) / ((1 << bits) - 1)  # noqa: E731
    pp, ps, pc, pr = (v[k] for k in ("packed_position", "packed_scale", "packed_color", "packed_rotation"))
    out = {"means": np.stack([lerp(ch["min_" + a], ch["max_" + a], un(pp, s, b)) for a, s, b in (("x", 21, 11), ("y", 11, 10), ("z", 0, 11))], 1),
           "scales": np.stack([lerp(ch[f"min_scale_{a}"], ch[f"max_scale_{a}"], un(ps, s, b)) for a, s, b in (("x", 21, 11), ("y", 11, 10), ("z", 0, 11))], 1)}
    col = np.stack([lerp(ch["min_" + a], ch["max_" + a], un(pc, s, 8)) for a, s in (("r", 24), ("g", 16), ("b", 8))], 1)
    out["sh0"] = ((col - 0.5) / C0)[:, None]
    a = un(pc, 0, 8)
    with np.errstate(divide="ignore"):
        out["opacities"] = -np.log(1 / a - 1)
    abc = np.stack([(un(pr, s, 10) - 0.5) * math.sqrt(2) for s in (20, 10, 0)], 1)
    m = np.sqrt(np.maximum(0, 1 - (abc ** 2).sum(1))); w = (pr.astype(np.uint64) >> np.uint64(30)).astype(int)
    qq = np.zeros((len(v), 4))
    for which in range(4):
        s = w == which; rest = [i for i in range(4) if i != which]
        qq[s, which] = m[s]; qq[np.ix_(s, rest)] = abc[s]
    out["quats"] = qq
    if "sh" in d:
        sh = d["sh"].data; names = sh.dtype.names; K = len(names) // 3
        r = np.stack([sh[n] for n in names], 1).astype(np.float64)
        n = np.where(r == 0, 0, np.where(r == 255, 1, (r + 0.5) / 256))
        out["shN"] = ((n - 0.5) * 8).reshape(len(v), 3, K).transpose(0, 2, 1)
    return out


class ViewerRaster:
    """Differentiable copy of a viewer's splat renderer (see module docstring)."""

    def __init__(self, viewer="pc274", raster="exact", quant=False, tile=16, cull=True, cutoff=True, a255=True,
                 shbug=None, dil=None, display_scale=1.0):
        """Ablation knobs (gap attribution): cull = minimum-size cull, cutoff = d^2 <= 8 kernel cut, a255 = 1/255
        discard (pc274), shbug = 2.7.4's SH band-2 typo (default: on for pc274), dil = dilation in px^2.
        display_scale (ss352): on-screen canvas pixels per rendered pixel. SuperSplat's projector takes focal, the 0.3
        dilation, the 2 px size cull and the radius cap in scene.targetSize (the on-screen canvas) pixels even for
        offscreen renders (render.offscreen / Render > Image), so in rendered pixels: dilation 0.3/r^2, cull 0.5/r^2."""
        assert viewer in ("pc274", "ss352") and raster in ("exact", "cuda")
        if viewer == "ss352" and raster == "cuda":
            raise SystemExit("ss352's normalised kernel needs --raster exact")
        self.viewer, self.raster, self.quant, self.tile = viewer, raster, quant, tile
        self.cull, self.cutoff, self.a255 = cull, cutoff, a255
        self.shbug = (viewer == "pc274") if shbug is None else shbug
        self.r = display_scale if viewer == "ss352" else 1.0
        self._dil = dil
        self.last = {}

    @staticmethod
    def cov2d(p, w2c, fx, fy, zmin=1e-4, zclamp=1e-4):
        """EWA pixel covariance without dilation (no Jacobian clamping, as the viewers) -> u-v centre offsets (x/z, y/z),
        z, (a, b, c) of Sigma2D."""
        import torch
        # float64: near-plane / huge splats overflow float32 (a*c > 3e38) and their inf local Jacobians turn the
        # zero upstream gradient of culled splats into NaN
        R, t = w2c[:3, :3].double(), w2c[:3, 3].double()
        pc = p["means"].double() @ R.T + t
        z = pc[:, 2]; zs = z.clamp_min(zclamp)
        M = R @ quat_to_rot(p["quats"].double()) * torch.exp(p["scales"].double().clamp_max(20))[:, None, :]
        S = M @ M.transpose(1, 2)
        zero = torch.zeros_like(zs)
        J = torch.stack([torch.stack([fx / zs, zero, -fx * pc[:, 0] / zs ** 2], -1),
                         torch.stack([zero, fy / zs, -fy * pc[:, 1] / zs ** 2], -1)], 1)          # (N,2,3)
        cov = J @ S @ J.transpose(1, 2)
        return pc[:, 0] / zs, pc[:, 1] / zs, z, (cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1])

    @property
    def dil(self):
        return (0.075 if self.viewer == "pc274" else 0.3 / self.r ** 2) if self._dil is None else self._dil

    def min_lambda(self):
        """Dilated max eigenvalue (px^2) below which the viewer drops a splat."""
        return 0.125 if self.viewer == "pc274" else 0.5 / self.r ** 2

    def project(self, p, w2c, fx, fy, cx, cy, W, H):
        """-> means2d (N,2), conics (N,3), opacity (N,), depth z (N,), radii int (N,2), keep mask."""
        import torch
        zmin = 1e-4 if self.viewer == "pc274" else 1e-6
        xz, yz, z, (a, b, c) = self.cov2d(p, w2c, fx, fy, zclamp=1e-3 if self.viewer == "ss352" else zmin)
        valid = z > zmin
        u = fx * xz + cx; v = fy * yz + cy
        dil = self.dil
        a, c = a + dil, c + dil
        mid = 0.5 * (a + c); rad = torch.sqrt((0.5 * (a - c)) ** 2 + b * b + 1e-20)
        l1 = mid + rad
        op = torch.sigmoid(p["opacities"])
        if self.viewer == "pc274":
            keep = valid & ((l1 >= 0.125) | (not self.cull))    # l1 < 2 && l2 < 2 in the engine's doubled units
            cap = 1024.0 ** 2 / 8                                # half-extent cap 1024 px -> eigenvalue cap
            l2 = (mid - rad).clamp_min(0.025)
            l1c, l2c = l1.clamp_max(cap), l2.clamp_max(cap)
            big = (l1 > cap) | (l2 > cap)
        else:
            keep = valid & ((l1 >= self.min_lambda()) | (not self.cull)) & (a * c - b * b > 0)    # 2 sqrt(2 l1) < minPixelSize
            s2 = ((min(1024.0, W * self.r, H * self.r) / self.r) ** 2 / (8 * l1)).clamp_max(1.0)  # radius cap, both axes
            l2 = (mid - rad).clamp_min(0.1)
            l1c, l2c = l1 * s2, l2 * s2
            big = s2 < 1
            if self.quant:
                op = torch.round(op * 255) / 255
            keep = keep & (op > 0)
        # effective covariance; rebuilt from the clamped eigen-decomposition only where a clamp binds
        e = torch.stack([b, l1 - a], -1); en = e.norm(dim=-1, keepdim=True)
        e = torch.where(en > 1e-9, e / en.clamp_min(1e-12), torch.tensor([1.0, 0.0], device=e.device))
        ra = l1c * e[:, 0] ** 2 + l2c * e[:, 1] ** 2; rb = (l1c - l2c) * e[:, 0] * e[:, 1]; rc = l1c * e[:, 1] ** 2 + l2c * e[:, 0] ** 2
        a, b, c = torch.where(big, ra, a), torch.where(big, rb, b), torch.where(big, rc, c)
        det = (a * c - b * b).clamp_min(1e-12)
        conics = torch.stack([c / det, -b / det, a / det], -1).float()
        a, c, z = a.float().clamp_max(1e12), c.float().clamp_max(1e12), z.float()
        # bounding box of the d^2 <= 8 ellipse
        ext = 8.0 if self.cutoff else 2 * math.log(255.0)               # no cut-off: the 1/255 tail of an opaque splat
        rx = torch.ceil(torch.sqrt(ext * a.detach())); ry = torch.ceil(torch.sqrt(ext * c.detach()))
        md = torch.stack([u, v], -1).float()
        inside = (md[:, 0].detach() + rx > 0) & (md[:, 0].detach() - rx < W) & (md[:, 1].detach() + ry > 0) & (md[:, 1].detach() - ry < H)
        keep = keep & inside & torch.isfinite(conics).all(-1) & torch.isfinite(md).all(-1)
        radii = torch.stack([rx, ry], -1).clamp(0, 1e6).int() * keep[:, None].int()
        return md, conics, op, z, radii, keep

    def render(self, p, w2c, fx, fy, cx, cy, W, H, sh_degree=3, depth=False):
        """-> rgb (H,W,3) clamped [0,1], alpha (H,W), expected depth (H,W) or None."""
        import torch
        from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode
        if self.quant and self.viewer == "pc274":
            p = quantise_pc274(p)
        md, conics, op, z, radii, keep = self.project(p, w2c, fx, fy, cx, cy, W, H)
        campos = -w2c[:3, :3].T @ w2c[:3, 3]
        dirs = p["means"] - campos; dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        col = eval_sh(p["sh0"], p["shN"], dirs, sh_degree, self.shbug)
        if self.quant:
            col = half(col)
        feats = torch.cat([col, z[:, None]], 1) if depth else col
        z_sort = torch.where(keep, z, torch.full_like(z, 1e10)).detach()
        tw, th = (W + self.tile - 1) // self.tile, (H + self.tile - 1) // self.tile
        _, isect_ids, flatten_ids = isect_tiles(md[None].detach().contiguous(), radii[None].contiguous(), z_sort[None].contiguous(),
                                                self.tile, tw, th)
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        self.last = {"n_visible": int(keep.sum()), "n_isects": int(len(flatten_ids))}
        if self.raster == "cuda":
            from gsplat.cuda._wrapper import rasterize_to_pixels
            out, alpha = rasterize_to_pixels(md[None], conics[None], feats[None].contiguous(), op[None], W, H, self.tile,
                                             offsets, flatten_ids)
            out, alpha = out[0], alpha[0, ..., 0]
        else:
            out, alpha = self._exact(md, conics, op, feats, W, H, offsets, flatten_ids)
        rgb = out[..., :3].clamp(0, 1)
        dep = (out[..., 3] / alpha.clamp_min(1e-6)) if depth else None
        return rgb, alpha, dep

    def _exact(self, md, conics, op, feats, W, H, offsets, flatten_ids):
        """Per-pixel front-to-back compositing in torch over the (pixel, gaussian) pairs from gsplat's indices kernel."""
        import torch
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range
        dev = md.device
        # candidate pairs (depth-ordered, contiguous per pixel); generous opacity so no pair the viewer keeps is lost
        op_idx = (op / (1 - E4) if self.viewer == "ss352" else op).clamp_max(1.0).detach()
        T0 = torch.ones(1, H, W, device=dev)
        gid, pid, _ = rasterize_to_indices_in_range(0, 1 << 30, T0, md[None].detach().contiguous(), conics[None].detach().contiguous(),
                                                    op_idx[None].contiguous(), W, H, self.tile, offsets, flatten_ids)
        self.last["n_pairs"] = int(len(gid))
        pix = torch.stack([pid % W, pid // W], -1).float() + 0.5
        d = pix - md[gid]; cq = conics[gid]
        s2 = cq[:, 0] * d[:, 0] ** 2 + 2 * cq[:, 1] * d[:, 0] * d[:, 1] + cq[:, 2] * d[:, 1] ** 2     # Mahalanobis^2
        if self.viewer == "pc274":
            al = op[gid] * torch.exp(-0.5 * s2)
            ok = ((s2 <= 8) | (not self.cutoff)) & ((al >= 1 / 255) | (not self.a255))
        elif self.cutoff:
            al = op[gid] * (torch.exp(-0.5 * s2) - E4) / (1 - E4)
            ok = (s2 <= 8) & (al > 0)
        else:                                                               # ablation: plain exp kernel, 1/255 cut
            al = op[gid] * torch.exp(-0.5 * s2)
            ok = al >= 1 / 255
        al = torch.where(ok, al, torch.zeros_like(al)).clamp(0, 1 - 1e-6)
        lt = torch.log1p(-al.double())
        cs = torch.cumsum(lt, 0) - lt                                       # exclusive, global
        first = torch.ones_like(pid, dtype=torch.bool); first[1:] = pid[1:] != pid[:-1]
        seg = torch.cumsum(first.long(), 0) - 1
        cs = cs - cs[first][seg]                                            # exclusive within each pixel
        w = (al.double() * torch.exp(cs)).float()
        n = H * W
        out = torch.zeros(n, feats.shape[1], device=dev).index_add_(0, pid, w[:, None] * feats[gid])
        alpha = torch.zeros(n, device=dev).index_add_(0, pid, w)
        return out.view(H, W, -1), alpha.view(H, W)


# ---------------------------------------------------------------- dataset helpers
def frames(dataset, test_split_interval=0, subset="train"):
    """[(name, cam tuple, c2w)] in name order; 3DGRUT's split: frame k is test iff interval > 0 and k % interval == 0."""
    cams, imgs = read_model(os.path.join(dataset, "sparse", "0"))
    out = []
    for k, (name, _iid, cid, c2w) in enumerate(imgs):
        test = test_split_interval > 0 and k % test_split_interval == 0
        if subset == "all" or (subset == "test") == test:
            out.append((name, cams[cid], c2w))
    return out


def viewer_intrinsics(cam, profile="pc274"):
    """The camera viewer_shots.py actually renders: canvas centred on the principal point (integer size), cropped at an
    integer offset, square pixels; pc274 sets the vertical fov (f = fy), SuperSplat 3.x (ss352 / --engine current)
    the horizontal one when the canvas is wider than tall (f = fx) -> (f, f, cx_eff, cy_eff)."""
    _K, W, H = intrinsics(cam)[0], cam[1], cam[2]
    fx, fy, cx, cy = _K[0, 0], _K[1, 1], _K[0, 2], _K[1, 2]
    CW = int(np.ceil(2 * max(cx, W - cx))); CH = int(np.ceil(2 * max(cy, H - cy)))
    x0 = int(round(CW / 2 - cx)); y0 = int(round(CH / 2 - cy))
    f = fx if (profile == "ss352" and CW > CH) else fy
    return f, f, CW / 2 - x0, CH / 2 - y0


def cam_args(cam, viewer_cam=False):
    """viewer_cam: False = the dataset's true intrinsics; 'pc274' / 'ss352' (True = 'pc274') = that viewer's camera."""
    K, W, H = intrinsics(cam)
    if viewer_cam:
        fx, fy, cx, cy = viewer_intrinsics(cam, "pc274" if viewer_cam is True else viewer_cam)
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return float(fx), float(fy), float(cx), float(cy), int(W), int(H)


def to_t(m, dev):
    import torch
    return torch.tensor(np.asarray(m), dtype=torch.float32, device=dev)


def sh_deg(p):
    return int(round(math.sqrt(p["shN"].shape[1] + 1))) - 1


def save_png(path, rgb):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray((rgb.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(path)


def u8(rgb):
    import torch
    return torch.round(rgb.clamp(0, 1) * 255) / 255


# ---------------------------------------------------------------- compare (renderer agreement)
def compare(args):
    import torch
    dev = "cuda"
    _, p = load_ply(args.ply, dev); deg = sh_deg(p)
    fr = frames(args.dataset, 0, "all")[args.offset::args.every]
    variants = args.variants.split(",")
    prof = "ss352" if args.engine == "current" else "pc274"       # whose camera the viewer renders were taken with
    res = {v: {"vs_viewer": [], "vs_gt": [], "ms": []} for v in variants}
    res["viewer"] = {"vs_gt": []}
    t0 = time.time()
    for k, (name, cam, c2w) in enumerate(fr):
        stem = os.path.splitext(name)[0]
        vf = os.path.join(args.viewer_dir, stem + ".png")
        if not os.path.exists(vf):
            continue
        vimg = load_rgb(vf, dev).float() / 255
        gt = load_rgb(os.path.join(args.dataset, "images", name), dev).float() / 255
        res["viewer"]["vs_gt"].append(psnr(vimg, gt))
        w2c = to_t(np.linalg.inv(c2w), dev)
        for var in variants:
            torch.cuda.synchronize(); t1 = time.time()
            with torch.no_grad():
                if var.startswith("gsplat"):                    # stock gsplat.rasterization, classic mode, given eps2d
                    from gsplat import rasterization
                    fx, fy, cx, cy, W, H = cam_args(cam, prof)
                    Kt = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=dev)
                    eps = float(var.split("_")[1]) if "_" in var else 0.3
                    out, _, _ = rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                              torch.cat([p["sh0"], p["shN"]], 1), w2c[None], Kt[None], W, H, sh_degree=deg,
                                              eps2d=eps, rasterize_mode="classic")
                    img = out[0].clamp(0, 1)
                else:                                           # <viewer>_<raster>[_q]
                    parts = var.split("_"); fl = parts[2:]
                    dd = [float(x[1:]) for x in fl if x.startswith("d") and x[1:2].isdigit()]
                    r = ViewerRaster(parts[0], parts[1], quant="q" in fl, cull="nocull" not in fl, cutoff="nocut" not in fl,
                                     a255="no255" not in fl, shbug=False if "nobug" in fl else (True if "bug" in fl else None), dil=dd[0] if dd else None,
                                     display_scale=next((float(x[1:]) for x in fl if x.startswith("r") and x[1:2].isdigit()), args.display_scale))
                    img = r.render(p, w2c, *cam_args(cam, False if "truecam" in parts else prof), sh_degree=deg)[0]
            torch.cuda.synchronize(); res[var]["ms"].append((time.time() - t1) * 1000)
            img = u8(img)
            res[var]["vs_viewer"].append(psnr(img, vimg)); res[var]["vs_gt"].append(psnr(img, gt))
            if args.save_dir and k < args.save_n:
                save_png(os.path.join(args.save_dir, var, stem + ".png"), img)
                save_png(os.path.join(args.save_dir, var, stem + "_absdiff_x8.png"), ((img - vimg).abs() * 8))
        print(f"[compare] {k + 1}/{len(fr)} {name} " + " ".join(f"{v}={res[v]['vs_viewer'][-1]:.2f}" for v in variants if res[v]["vs_viewer"]),
              flush=True)
    summ = {"ply": args.ply, "dataset": args.dataset, "viewer_dir": args.viewer_dir, "n": len(res["viewer"]["vs_gt"]),
            "viewer_psnr_vs_gt": round(float(np.mean(res["viewer"]["vs_gt"])), 3), "variants": {}}
    for v in variants:
        if res[v]["vs_viewer"]:
            summ["variants"][v] = {"psnr_vs_viewer": round(float(np.mean(res[v]["vs_viewer"])), 3),
                                   "psnr_vs_viewer_min": round(float(np.min(res[v]["vs_viewer"])), 3),
                                   "psnr_vs_gt": round(float(np.mean(res[v]["vs_gt"])), 3),
                                   "ms_per_frame": round(float(np.median(res[v]["ms"])), 1)}
    print(json.dumps(summ, indent=1)); print(f"[compare] {time.time() - t0:.0f}s")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(summ, open(args.json_out, "w"), indent=2)
    print("FTV_COMPARE_DONE")


# ---------------------------------------------------------------- render
def load_any(path, dev):
    """.ply params, or a decoded .compressed.ply (read_cply, file order) -> dict of tensors."""
    import torch
    if path.endswith(".compressed.ply"):
        return {k: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=dev) for k, v in read_cply(path).items()}
    return load_ply(path, dev)[1]


def cply_check(args):
    """Simulation (quantise_cply of --ply) vs the real splat-transform output (--cply): per-attribute max |diff|."""
    import torch
    _, p = load_ply(args.ply, "cpu"); order = cply_order(p["means"].numpy())
    sim = quantise_cply(p, order, ste=False); real = read_cply(args.cply)
    res = {"n": len(order), "max_abs_diff": {}, "frac_rows_differing": {}}
    for k in ("means", "scales", "sh0", "opacities", "quats", "shN"):
        if k not in real:
            continue
        a = sim[k][torch.as_tensor(order)].double().numpy().reshape(len(order), -1); b = np.asarray(real[k], np.float64).reshape(len(order), -1)
        if k == "opacities":                           # compare in alpha (logit of 0 / 255 is -inf / inf)
            a, b = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
        if k == "quats":                               # q and -q are the same rotation
            a = a / np.linalg.norm(a, axis=1, keepdims=True); a = a * np.sign((a * b).sum(1, keepdims=True) + 1e-30)
        d = np.abs(a - b); res["max_abs_diff"][k] = float(d.max()); res["frac_rows_differing"][k] = float((d.max(1) > 1e-4).mean())
        o = np.asarray(p[k], np.float64)[order].reshape(len(order), -1)
        if k == "opacities":
            o = 1 / (1 + np.exp(-o))
        if k == "quats":
            o = o / np.linalg.norm(o, axis=1, keepdims=True); o = o * np.sign((o * b).sum(1, keepdims=True) + 1e-30)
        res.setdefault("encode_err_mean", {})[k] = float(np.abs(o - b).mean()); res.setdefault("encode_err_max", {})[k] = float(np.abs(o - b).max())
    print(json.dumps(res, indent=1))
    if args.json_out:
        json.dump(res, open(args.json_out, "w"), indent=2)
    print("FTV_CPLY_CHECK_DONE")


def render(args):
    import torch
    dev = "cuda"
    p = load_any(args.ply, dev); deg = sh_deg(p)
    if args.quant_format == "cply":                    # simulate splat-transform's .compressed.ply of this .ply
        p = quantise_cply(p, cply_order(p["means"].cpu().numpy()), args.quant_parts.split(","), ste=False)
    r = ViewerRaster(args.viewer, args.raster, quant=args.quant, display_scale=args.display_scale)
    bgs = {b: torch.tensor([float(x) for x in b.split(",")] if "," in b else
                           {"black": [0.0] * 3, "grey": [0.5] * 3, "white": [1.0] * 3}[b], device=dev) for b in args.bg}
    for name, cam, c2w in frames(args.dataset, 0, "all")[args.offset::args.every]:
        with torch.no_grad():
            img, alpha, _ = r.render(p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam, args.viewer if args.viewer_cam else False), sh_degree=deg)
        for b, c in bgs.items():                       # premultiplied over the background, as the viewer composites
            od = args.out_dir if len(bgs) == 1 else os.path.join(args.out_dir, b)
            save_png(os.path.join(od, os.path.splitext(name)[0] + ".png"), img + (1 - alpha.clamp(0, 1))[..., None] * c)
    print("FTV_RENDER_DONE")


# ---------------------------------------------------------------- finetune
def evaluate(raster, p, fr, dev, deg, viewer_cam=False):
    import torch
    ps = []
    with torch.no_grad():
        for name, cam, c2w, gt in fr:
            img = raster.render(p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam, viewer_cam), sh_degree=deg)[0]
            ps.append(psnr(u8(img), gt))
    return float(np.mean(ps)) if ps else float("nan")


def finetune(args):
    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    vertex, p0 = load_ply(args.ply, dev); deg = sh_deg(p0); N = len(p0["means"])
    tr = frames(args.dataset, args.test_split_interval, "train")
    held = frames(args.dataset, args.test_split_interval, "test") if args.test_split_interval > 0 else []
    print(f"[ftv] {args.ply}: {N} gaussians, SH degree {deg}; {len(tr)} training frames, {len(held)} held out "
          f"(test_split_interval={args.test_split_interval}); viewer {args.viewer}, raster {args.raster}", flush=True)
    img_dir = os.path.join(args.dataset, "images")
    rgb = [load_rgb(os.path.join(img_dir, n), dev) for n, _, _ in tr]
    dep = [None] * len(tr); rn = {}
    if args.depth_lambda > 0:
        for k, (n, cam, _) in enumerate(tr):
            f = os.path.join(img_dir, os.path.splitext(n)[0] + "_depth.npy")
            if os.path.exists(f):
                dep[k] = torch.from_numpy(np.load(f).astype(np.float32)).to(dev)
        print(f"[ftv] depth targets for {sum(d is not None for d in dep)}/{len(tr)} frames, lambda {args.depth_lambda}", flush=True)
    masks = {}
    if args.mask_dir:                                 # transient pixels: the render copies the real image there
        from PIL import Image
        for k, (n, _, _) in enumerate(tr):
            f = os.path.join(args.mask_dir, os.path.splitext(os.path.basename(n))[0] + "_mask.png")
            if os.path.exists(f):
                mk = torch.from_numpy(np.asarray(Image.open(f).convert("L")) > 127).to(dev)
                if bool(mk.any()):
                    masks[k] = mk[..., None].float()
        print(f"[ftv] masks for {len(masks)}/{len(tr)} training frames", flush=True)
    centres = np.stack([c[:3, 3] for _, _, c in tr])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p0.items()})
    m = args.lr_mult
    lr = {"means": args.means_lr * scene_scale * m, "scales": 5e-3 * m, "quats": 1e-3 * m, "opacities": args.opacity_lr * m,
          "sh0": 2.5e-3 * m, "shN": 2.5e-3 / 20 * m}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr[k], "name": k} for k in params], eps=1e-15)
    decay = args.means_lr_final / args.means_lr
    gm = next(g for g in opt.param_groups if g["name"] == "means")
    rs = [float(x) for x in str(args.display_scale).split(":")]
    r_eval = rs[0] if len(rs) == 1 else float(args.eval_display_scale or rs[0])
    raster = ViewerRaster(args.viewer, args.raster, display_scale=r_eval)
    eval_ds = args.eval_dataset
    ev = []
    if eval_ds:                                       # held-out monitor (matched renderer, viewer camera)
        ev = [(n, cam, c2w, load_rgb(os.path.join(eval_ds, "images", n), dev).float() / 255)
              for n, cam, c2w in frames(eval_ds, 0, "all")[::args.eval_every_frame]]
    trs = [(n, cam, c2w, rgb[k].float() / 255) for k, (n, cam, c2w) in enumerate(tr)][::max(1, len(tr) // 30)]
    log = {"args": vars(args), "n_gaussians": N, "scene_scale": scene_scale, "curve": []}

    def report(step):
        p = {k: v.detach() for k, v in params.items()}
        if args.quant_format != "none":
            p = quantise_cply(p, cply_order(p["means"].cpu().numpy()), args.quant_parts.split(","), ste=False)
        e = {"step": step, "train_psnr": round(evaluate(raster, p, trs, dev, deg), 3)}
        if ev:
            e["heldout_psnr"] = round(evaluate(raster, p, ev, dev, deg, viewer_cam=raster.viewer), 3)
        log["curve"].append(e); print(f"[ftv] eval {e}", flush=True)

    wdnet = None
    if args.lambda_wd > 0:
        from wasserstein_distortion import VGG16WassersteinDistortion
        wdnet = VGG16WassersteinDistortion().to(dev).eval()
        for q in wdnet.parameters():
            q.requires_grad_(False)
    lpnet = None
    if args.lpips_lambda > 0:
        from torchmetrics.functional.image.lpips import _NoTrainLpips
        lpnet = _NoTrainLpips(net="vgg").to(dev).eval()
        for q in lpnet.parameters():
            q.requires_grad_(False)
    qparts = args.quant_parts.split(",")
    qstate = {"order": None}

    def qp(prm, step=0):                                # what the viewer draws: the compressed file's decoded values
        if args.quant_format == "none":
            return prm
        if qstate["order"] is None or (args.quant_reorder_every and step % args.quant_reorder_every == 0):
            qstate["order"] = cply_order(prm["means"].detach().cpu().numpy())
        return quantise_cply(prm, qstate["order"], qparts)
    if args.quant_format != "none":
        print(f"[ftv] quantisation-aware: {args.quant_format} ({','.join(qparts)}), straight-through, chunk order "
              f"recomputed every {args.quant_reorder_every} steps", flush=True)
    report(0)
    order, t0 = [], time.time()
    for step in range(1, args.iters + 1):
        if not order:
            order = list(np.random.permutation(len(tr)))
        k = 0 if args.overfit_one else int(order.pop()); name, cam, c2w = tr[k]
        fx, fy, cx, cy, W, H = cam_args(cam)
        if len(rs) == 2:                                   # robust to the display: log-uniform on-screen scale per step
            raster.r = float(np.exp(np.random.uniform(np.log(rs[0]), np.log(rs[1]))))
        frac = step / args.iters
        wu = min(1.0, step / args.warmup) if args.warmup > 0 else 1.0     # Adam's first steps are full-size: ramp in
        for g in opt.param_groups:
            g["lr"] = lr[g["name"]] * wu
        gm["lr"] = lr["means"] * decay ** frac * wu    # log-linear decay (3DGS position schedule)
        img, alpha, d = raster.render(qp(params, step), to_t(np.linalg.inv(c2w), dev), fx, fy, cx, cy, W, H, sh_degree=deg,
                                      depth=dep[k] is not None)
        gt = rgb[k].float() / 255
        if k in masks:
            img = img * (1 - masks[k]) + gt * masks[k]
        loss = args.rgb_scale * ((1 - args.ssim_lambda) * (img - gt).abs().mean() +
                                 args.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        if wdnet is not None:                           # Wasserstein Distortion: local VGG-16 feature statistics
            pc_, gc_ = img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
            loss = loss + args.lambda_wd * wdnet(pc_, gc_, torch.full_like(gc_[:, :1], args.wd_log2_sigma),
                                                 num_scales=args.wd_num_scales)
        if lpnet is not None:                           # perceptual term (the gate's LPIPS-VGG, torchmetrics weights)
            loss = loss + args.lpips_lambda * lpnet(img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()
        ld = torch.zeros((), device=dev)
        if dep[k] is not None:
            key = (fx, fy, cx, cy, W, H)
            if key not in rn:
                u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
                rn[key] = to_t(np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1), dev)
            msk = (dep[k] > 0) & (alpha.detach() > 0.5)
            if msk.any():
                err = ((d * rn[key])[msk] - dep[k][msk]).abs()
                err = err.sort().values[: max(1, int(err.numel() * args.depth_trim))]
                ld = err.mean(); loss = loss + args.depth_lambda * ld
        if args.anchor > 0:                                             # keep geometry near the input (off-path safety)
            loss = loss + args.anchor * ((params["means"] - p0["means"]) ** 2).sum(-1).mean() / scene_scale ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.freeze_color_last > 0:                  # e.g. inpainted fill discs: keep their colours exactly
            for k in ("sh0", "shN"):
                if params[k].grad is not None:
                    params[k].grad[-args.freeze_color_last:] = 0
        nbad = 0
        for prm in params.values():
            if prm.grad is not None:
                bad = ~torch.isfinite(prm.grad); nb = int(bad.sum())
                if nb:
                    nbad += nb; prm.grad[bad] = 0
        opt.step()
        if nbad:
            print(f"[ftv] WARNING step {step}: {nbad} non-finite gradient entries zeroed", flush=True)
        if step % args.log_every == 0 or step == 1:
            print(f"[ftv] step {step:6d} loss {loss.item():.4f} psnr {psnr(img.detach(), gt):.2f} depth {ld.item():.4f} "
                  f"vis {raster.last.get('n_visible')} pairs {raster.last.get('n_pairs')} "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G ({time.time() - t0:.0f}s)", flush=True)
        if args.eval_interval and step % args.eval_interval == 0 and step != args.iters:
            raster.r = r_eval; report(step)
    raster.r = r_eval; report(args.iters)
    with torch.no_grad():
        if args.freeze_color_last > 0:                  # exact copy-back (Adam with all-zero grads never moves them anyway)
            for k in ("sh0", "shN"):
                params[k][-args.freeze_color_last:] = p0[k][-args.freeze_color_last:]
        dm = (params["means"] - p0["means"]).norm(dim=-1)
        log["delta"] = {"means_median": float(dm.median()), "means_p99": float(torch.quantile(dm[::max(1, N // 1000000)], 0.99)),
                        "opacity_mean_before": float(torch.sigmoid(p0["opacities"]).mean()),
                        "opacity_mean_after": float(torch.sigmoid(params["opacities"]).mean()),
                        "logscale_mean_before": float(p0["scales"].mean()), "logscale_mean_after": float(params["scales"].mean())}
    save_ply(args.out, vertex, params)
    log["seconds"] = time.time() - t0
    json.dump(log, open(os.path.splitext(args.out)[0] + "_ftv.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=1))
    print(f"[ftv] wrote {args.out}"); print("FTV_FINETUNE_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ply", required=True); c.add_argument("--dataset", required=True)
    c.add_argument("--viewer-dir", required=True, help="viewer_shots.py renders of --ply at --dataset's cameras")
    c.add_argument("--variants", default="gsplat_0.3,gsplat_0.075,pc274_cuda,pc274_exact,pc274_exact_q,ss352_exact",
                   help="gsplat_<eps2d> | <pc274|ss352>_<exact|cuda>[_q][_nocull][_nocut][_no255][_nobug|_bug][_d<dilation>][_truecam]")
    c.add_argument("--engine", default="pc274", choices=("pc274", "current"), help="viewer_shots.py --engine of --viewer-dir")
    c.add_argument("--display-scale", type=float, default=1.0, help="ss352: on-screen px per rendered px (variant _r<x>)")
    c.add_argument("--every", type=int, default=1); c.add_argument("--offset", type=int, default=0)
    c.add_argument("--save-dir", default=None); c.add_argument("--save-n", type=int, default=3)
    c.add_argument("--json-out", default=None)
    r = sub.add_parser("render")
    r.add_argument("--ply", required=True); r.add_argument("--dataset", required=True); r.add_argument("--out-dir", required=True)
    r.add_argument("--viewer", default="ss352", choices=("pc274", "ss352")); r.add_argument("--raster", default="exact", choices=("exact", "cuda"))
    r.add_argument("--quant", action="store_true"); r.add_argument("--viewer-cam", action="store_true")
    r.add_argument("--quant-format", default="none", choices=("none", "cply"), help="cply: simulate splat-transform's "
                   ".compressed.ply of --ply (--ply may also be a .compressed.ply: decoded as SuperSplat does)")
    r.add_argument("--quant-parts", default=",".join(CPLY_PARTS), help="subset of " + ",".join(CPLY_PARTS))
    r.add_argument("--display-scale", type=float, default=1.0, help="ss352: on-screen canvas px per image px")
    r.add_argument("--bg", nargs="+", default=["black"], help="black|grey|white|R,G,B; several -> <out-dir>/<bg>/")
    r.add_argument("--every", type=int, default=1); r.add_argument("--offset", type=int, default=0)
    k = sub.add_parser("cply-check")
    k.add_argument("--ply", required=True); k.add_argument("--cply", required=True, help="splat-transform output of --ply")
    k.add_argument("--json-out", default=None)
    f = sub.add_parser("finetune")
    f.add_argument("--ply", required=True); f.add_argument("--dataset", required=True, help="COLMAP text dataset, PLY's frame")
    f.add_argument("--out", required=True)
    f.add_argument("--test-split-interval", type=int, default=0, help="3DGRUT's rule: frame k held out iff k %% N == 0")
    f.add_argument("--viewer", default="ss352", choices=("pc274", "ss352"), help="ss352 = today's SuperSplat (default)")
    f.add_argument("--raster", default="exact", choices=("exact", "cuda"))
    f.add_argument("--display-scale", default="1.0", help="ss352: on-screen canvas px per image px (see ViewerRaster); "
                   "'a:b' samples it log-uniformly per step (robust to the user's window / DPI)")
    f.add_argument("--eval-display-scale", type=float, default=None, help="scale for the monitor evals (default: a)")
    f.add_argument("--iters", type=int, default=7000)
    f.add_argument("--lr-mult", type=float, default=1.0, help="scales every learning rate")
    f.add_argument("--means-lr", type=float, default=1.6e-5, help="x scene scale, decays log-linearly to --means-lr-final")
    f.add_argument("--means-lr-final", type=float, default=1.6e-6)
    f.add_argument("--opacity-lr", type=float, default=0.025)
    f.add_argument("--ssim-lambda", type=float, default=0.2)
    f.add_argument("--lpips-lambda", type=float, default=0.0, help="LPIPS-VGG term (as eval_views / the accept gate)")
    f.add_argument("--lambda-wd", type=float, default=0.0, help="Wasserstein Distortion (VGG-16) term, e.g. 0.025")
    f.add_argument("--wd-log2-sigma", type=float, default=4.0, help="WD pooling scale (log2 px)")
    f.add_argument("--wd-num-scales", type=int, default=3)
    f.add_argument("--rgb-scale", type=float, default=1.0, help="weight of the L1/SSIM terms")
    f.add_argument("--depth-lambda", type=float, default=0.0, help="LiDAR ray-distance L1 where <image>_depth.npy exists; 0 = off")
    f.add_argument("--mask-dir", default=None, help="<stem>_mask.png per training frame (255 = transient: no loss there)")
    f.add_argument("--depth-trim", type=float, default=0.95)
    f.add_argument("--anchor", type=float, default=0.0, help="penalty on centre drift (scene-scale units^2)")
    f.add_argument("--eval-dataset", default=None, help="held-out set monitored during training (never trained on)")
    f.add_argument("--eval-every-frame", type=int, default=2)
    f.add_argument("--eval-interval", type=int, default=2000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--log-every", type=int, default=500)
    f.add_argument("--freeze-color-last", type=int, default=0, help="keep SH (f_dc, f_rest) of the last N gaussians fixed")
    f.add_argument("--warmup", type=int, default=300, help="linear LR warm-up steps")
    f.add_argument("--quant-format", default="none", choices=("none", "cply"),
                   help="quantisation-aware fine-tune: render what the viewer draws from the compressed file (cply = "
                        "splat-transform's .compressed.ply, quantise_cply) with straight-through gradients; the output "
                        ".ply stays float (compress it afterwards)")
    f.add_argument("--quant-parts", default=",".join(CPLY_PARTS), help="quantised attributes (subset of " + ",".join(CPLY_PARTS) + ")")
    f.add_argument("--quant-reorder-every", type=int, default=500, help="recompute the Morton chunking every N steps")
    f.add_argument("--overfit-one", action="store_true", help="debug: train on the first training frame only")
    args = ap.parse_args()
    {"compare": compare, "render": render, "finetune": finetune, "cply-check": cply_check}[args.cmd](args)


if __name__ == "__main__":
    main()
