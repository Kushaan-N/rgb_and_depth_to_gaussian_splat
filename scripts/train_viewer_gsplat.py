#!/usr/bin/env python3
"""Train a 3DGS splat FROM SCRATCH in a rasteriser matched to the viewer that will show it (gsplat venv, GPU).

3DGRUT/3DGUT models score ~3.5 dB lower when the exported PLY is drawn by SuperSplat than in 3DGUT (unscented vs
EWA projection, per-ray vs global sort, kernel cut-off, size cull). Here densification and optimisation happen in
the viewer's own image formation, so the splats allocated are the ones the viewer draws. Rasteriser = gsplat 1.5.3
kernels rebuilt per viewer by env/make_gsplat_viewer_venv.sh (PYTHONPATH=$CEAR_WS/gsplat-viewer-<viewer>; the
trainer refuses other kernels), with the viewer's conventions (shaders documented in scripts/finetune_ply_viewer.py):
  ss352 (default; SuperSplat 3.5.2, scripts/viewer_shots.py --engine current): true-pixel dilation 0.3 px^2,
        normalised kernel (exp(-d^2/2) - e^-4) / (1 - e^-4) zero beyond d^2 = 8, no 1/255 cut, 2 px size cull
        (dilated major eigenvalue < 0.5 px^2), radius capped at min(1024, canvas) by scaling both axes, 8-bit
        opacity (straight-through), correct SH
  pc274 (PlayCanvas 2.7.4, --engine pc274): dilation 0.075 px^2 (0.3 in its 2f units), exp kernel cut at d^2 = 8
        and alpha 1/255, cull below 0.125 px^2, eigenvalues clamped at 1024^2/8, SH band-2 bug (`sh[3]*(C*xy) *
        +sh[4]*(C*yz)`): --sh-band2 zero (pc274 default) keeps that pair at 0 so buggy and correct viewers agree
  both: EWA Jacobian at the true centre (no 1.3x-FOV clamp: the viewers draw off-screen-centred splats near the
        camera stretched), near plane 1e-4, per-gaussian view-depth sort (gsplat per-tile sort by camera z = the
        viewers' global order), black background, SH degree 3.
--sort depth|radial|dual: blend order. SuperSplat 3.7.0 added Settings > Sort Mode, default Radial (distance to the
camera), which reorders big / overlapping splats trained in view-depth order (CEAR baseline: 36.28 dB in Depth mode, 32.26
in Radial); radial trains in that order, dual alternates the two per step (one splat for both modes).
Options: --aa (gsplat antialiased mode: opacity x sqrt(det/det_dilated); the exported PLY is tagged
`comment SplatRenderMode mip` -- PlayCanvas 2.7.4 itself never enables GSPLAT_AA, so its renders lack the
compensation), --ss 2 (SRGS-style sub-pixel constraint: render at 2x with eps2d x 4, 2x2 area-downsample, loss
at 1x), --min-px (3D scale floor: every axis >= min_px pixels at the closest training camera that sees it,
std = sqrt(s^2 + floor^2), recomputed after densification, baked into the export, so the viewer's cull never
removes a trained splat), --depth-lambda (LiDAR ray-distance L1 where <image>_depth.npy exists), --bg random
(default: a random background per step, so the opaque real frames force full coverage; with black, transparent
holes cost nothing in training and show on a grey viewer background).
--appearance gain|affine|bilagrid: per-training-frame colour transform (auto-exposure / white balance) applied to the RENDER
before the loss only -- gain: c' = exp(g) * c + b per channel (6 numbers), affine: c' = (I + A) c + b (12) --
re-centred to the mean identity after every step (the splat keeps the average exposure; no gauge drift) and pulled
toward identity (--app-reg); bilagrid = a 16x16x8 bilateral grid of such affines per frame (BilagridAppearance,
spatially varying: vignetting / local tone, + --bilagrid-tv). The exported PLY is a standard splat; transforms -> <out>/appearance.json. Held-out
scores are also reported with a per-frame least-squares colour affine fitted on the frame itself (heldout_*_fit;
leaks 12 numbers per frame; sbatch/viewer_train_bench.sbatch does the same on the viewer's renders).
--mask-dir DIR: <DIR>/<image stem>_mask.png (255 = transient, e.g. people from scripts/datasets/fusionportable_masks.py)
per training frame: masked pixels of the render are replaced by the real image before every loss term, so they give no
photometric gradient (hence no densification signal) and their edges are dropped from the EAS edge map.
Opacity clean-up (semi-transparent splats draw differently across viewers / scales: haze, floaters, see-through
patches off-path): --opa-reg entropy|binary adds --opa-lambda x mean over the step's drawn gaussians of the binary
entropy (or o(1-o)), ramped in linearly from --opa-start to --opa-start + --opa-ramp (fractions of --iters); after the
ramp, every --prune-every steps and once at the end, --opa-prune T removes opacity < T and --contrib-prune W removes
gaussians whose blending weight sum_px T alpha over ALL training views is < W pixels (unconstrained floaters).
Strategies: improved (Improved-GS: EAS + LAS + RAP + budget + MU, scripts/improved_gs_strategy.py), mcmc
(gsplat MCMCStrategy, cap = --budget, opacity/scale reg 0.01), default (gsplat DefaultStrategy, AbsGS grads).
Split: frame k (name order) is held out iff k % --test-split-interval == 0 (3DGRUT's rule).
Init: points3D.txt (xyz + rgb) of the dataset, scales from the 3-NN distance, opacity --init-opacity.
Output: <out>/splat.ply (standard 3DGS PLY, dataset frame), <out>/train.json (args, curve, timings, counts,
held-out PSNR in this rasteriser at --eval-dataset).

    PYTHONPATH=$CEAR_WS/gsplat-viewer-ss352 python scripts/train_viewer_gsplat.py --dataset DS --out OUT \
        --test-split-interval 8 [--strategy improved --budget 4000000] [--viewer ss352] [--ss 2 --min-px 0.5] \
        [--aa] [--iters 30000] [--eval-dataset TEST] [--appearance affine] [--opa-reg entropy --opa-prune 0.05 --contrib-prune 1]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses_gsplat import read_model, intrinsics, load_rgb, ssim, psnr  # noqa: E402

C0 = 0.28209479177387814
# viewer profiles (scripts/finetune_ply_viewer.py documents the shaders): dilation px^2, size cull (dilated major
# eigenvalue px^2), radius cap (pc274: eigenvalues clamped at 1024^2/8; ss352: both scaled so the radius fits
# min(1024, canvas)), 8-bit opacity, default SH band-2 handling
PROFILES = {"pc274": {"eps2d": 0.075, "cull_l1": 0.125, "cap": "clamp", "opq": False, "sh_band2": "zero"},
            "ss352": {"eps2d": 0.3, "cull_l1": 0.5, "cap": "scale", "opq": True, "sh_band2": "std"}}


# ---------------------------------------------------------------- data
def load_views(dataset, interval, subset, dev, depth=False):
    """[(name, K (3,3) tensor, W, H, viewmat (4,4) tensor, rgb uint8 (H,W,3), depth (H,W) or None)]."""
    import torch
    cams, imgs = read_model(os.path.join(dataset, "sparse", "0"))
    out = []
    for k, (name, _iid, cid, c2w) in enumerate(imgs):
        test = interval > 0 and k % interval == 0
        if subset != "all" and (subset == "test") != test:
            continue
        K, W, H = intrinsics(cams[cid])
        d = None
        f = os.path.join(dataset, "images", os.path.splitext(name)[0] + "_depth.npy")
        if depth and os.path.exists(f):
            d = torch.from_numpy(np.load(f).astype(np.float32)).to(dev)
        out.append((name, torch.tensor(K, dtype=torch.float32, device=dev), int(W), int(H),
                    torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev),
                    load_rgb(os.path.join(dataset, "images", name), dev), d))
    return out


def init_params(dataset, dev, init_opacity, init_scale, sh_degree, max_points=0, seed=0):
    import torch
    from scipy.spatial import cKDTree
    f = os.path.join(dataset, "sparse", "0", "points3D.txt")
    a = np.loadtxt(f, comments="#", usecols=(1, 2, 3, 4, 5, 6), dtype=np.float64)
    xyz, rgb = a[:, :3], a[:, 3:6] / 255.0
    if max_points and len(xyz) > max_points:
        sel = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False); xyz, rgb = xyz[sel], rgb[sel]
    d, _ = cKDTree(xyz).query(xyz, k=4, workers=-1)
    s = np.log(np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-7) * init_scale)
    n, K = len(xyz), (sh_degree + 1) ** 2 - 1
    g = torch.Generator().manual_seed(seed)
    p = {"means": torch.tensor(xyz, dtype=torch.float32), "scales": torch.tensor(s, dtype=torch.float32)[:, None].repeat(1, 3),
         "quats": torch.rand(n, 4, generator=g), "opacities": torch.logit(torch.full((n,), float(init_opacity))),
         "sh0": ((torch.tensor(rgb, dtype=torch.float32) - 0.5) / C0)[:, None], "shN": torch.zeros(n, K, 3)}
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(dev)) for k, v in p.items()})


# ---------------------------------------------------------------- the viewer-matched renderer
class ViewerGS:
    def __init__(self, args):
        prof = PROFILES[args.viewer]
        self.eps2d = prof["eps2d"] if args.eps2d is None else args.eps2d
        self.cull_l1 = prof["cull_l1"] if args.cull_l1 is None else args.cull_l1
        self.aa, self.tile, self.cap_mode, self.opq = args.aa, 16, prof["cap"], prof["opq"]
        self.sh_band2 = args.sh_band2 or prof["sh_band2"]
        self.floor = None                                  # per-gaussian 3D scale floor (--min-px), or None
        self.min_px, self.near, self.cap_px = args.min_px, args.near, args.cap_px
        self.sort, self._n = getattr(args, "sort", "depth"), 0   # blend order (--sort; SuperSplat >= 3.7 default: radial)

    def sort_key(self, p, viewmat, depths):
        """Per-gaussian blend-order key [1,N]: view depth (SuperSplat <= 3.6, its 'Depth' sort mode) or the distance to the
        camera (3.7's default 'Radial'); 'dual' alternates per render so the splat is trained for both orders."""
        import torch
        self._n += 1
        if self.sort == "depth" or (self.sort == "dual" and self._n % 2):
            return depths.detach()
        with torch.no_grad():
            return torch.linalg.norm(p["means"] @ viewmat[:3, :3].T + viewmat[:3, 3], dim=-1)[None]

    def scales(self, p):
        import torch
        s = torch.exp(p["scales"])
        if self.floor is not None and len(self.floor) == len(s):
            s = torch.sqrt(s * s + self.floor[:, None] ** 2)
        return s

    def sh_coeffs(self, p, deg):
        import torch
        c = torch.cat([p["sh0"], p["shN"]], 1)[:, : (deg + 1) ** 2]
        if self.sh_band2 == "zero" and deg >= 2:                # PlayCanvas 2.7.4's multiplied pair: keep it 0
            m = torch.ones(c.shape[1], 1, device=c.device); m[4:6] = 0
            c = c * m
        return c

    def project(self, p, viewmat, K, W, H, ss=1):
        """-> radii [1,N,2] (viewer cull applied), means2d [1,N,2], depths [1,N], conics [1,N,3], opacity [1,N]."""
        import torch
        from gsplat.cuda._wrapper import fully_fused_projection
        Ks = K.clone(); Ks[:2] *= ss
        op = torch.sigmoid(p["opacities"])
        if self.opq:                                        # the viewer stores opacity in 8 bits (straight-through)
            op = op + (torch.round(op * 255) / 255 - op).detach()
        radii, means2d, depths, conics, comp = fully_fused_projection(
            p["means"], None, p["quats"], self.scales(p), viewmat[None], Ks[None], W * ss, H * ss,
            eps2d=self.eps2d * ss * ss, near_plane=self.near, calc_compensations=self.aa, opacities=op)
        with torch.no_grad():                               # viewer cull on the dilated major eigenvalue (1x px^2)
            a, b, c = conics[..., 0].double(), conics[..., 1].double(), conics[..., 2].double()
            det = (a * c - b * b).clamp_min(1e-300)
            ca, cb, cc = c / det, -b / det, a / det
            l1 = 0.5 * (ca + cc) + torch.sqrt((0.5 * (ca - cc)) ** 2 + cb * cb)
            keep = (l1 / (ss * ss) >= self.cull_l1) & torch.isfinite(l1) & (radii > 0).all(-1) & (op > 0)[None]
            if self.cap_mode == "clamp":
                R = self.cap_px
            else:                                           # radius <= min(1024, canvas) (centred canvas of viewer_shots)
                R = min(self.cap_px, math.ceil(2 * max(float(K[0, 2]), W - float(K[0, 2]))), math.ceil(2 * max(float(K[1, 2]), H - float(K[1, 2]))))
            cap = (R * ss) ** 2 / 8.0                       # quad half-extent sqrt(8) sigma <= R px
            big = keep & (l1 > cap)
        if self.cap_px > 0 and bool(big.any()):             # rebuild those covariances with clamped eigenvalues
            i = big[0].nonzero()[:, 0]
            q = conics[0, i].double(); qa, qb, qc = q.unbind(-1); dt = qa * qc - qb * qb
            A, B, Cc = qc / dt, -qb / dt, qa / dt
            mid = 0.5 * (A + Cc); rad = torch.sqrt((0.5 * (A - Cc)) ** 2 + B * B + 1e-30)
            if self.cap_mode == "clamp":                    # PlayCanvas 2.7.4: each eigenvalue clamped
                L1, L2 = (mid + rad).clamp_max(cap), (mid - rad).clamp(0.025 * ss * ss, cap)
            else:                                           # SuperSplat 3.5.2: both scaled by cap / l1
                s2 = (cap / (mid + rad)).clamp_max(1.0)
                L1, L2 = (mid + rad) * s2, (mid - rad).clamp_min(0.1 * ss * ss) * s2
            e = torch.stack([B, mid + rad - A], -1); en = e.norm(dim=-1, keepdim=True)
            e = torch.where(en > 1e-12, e / en.clamp_min(1e-30), torch.tensor([1.0, 0.0], dtype=e.dtype, device=e.device))
            RA = L1 * e[:, 0] ** 2 + L2 * e[:, 1] ** 2; RB = (L1 - L2) * e[:, 0] * e[:, 1]; RC = L1 * e[:, 1] ** 2 + L2 * e[:, 0] ** 2
            d2 = RA * RC - RB * RB
            conics = conics.clone(); conics[0, i] = torch.stack([RC / d2, -RB / d2, RA / d2], -1).float()
            with torch.no_grad():
                rx, ry = torch.ceil(torch.sqrt(8 * RA)), torch.ceil(torch.sqrt(8 * RC))
                m = means2d[0, i].double()
                vis = (m[:, 0] + rx > 0) & (m[:, 0] - rx < W * ss) & (m[:, 1] + ry > 0) & (m[:, 1] - ry < H * ss)
                radii = radii.clone(); radii[0, i] = (torch.stack([rx, ry], -1) * vis[:, None]).to(radii.dtype)
        with torch.no_grad():
            radii = radii * keep[..., None].to(radii.dtype)
        opr = op[None] * comp if self.aa else op[None]
        return radii, means2d, depths, conics, opr

    def raster(self, means2d, radii, depths, conics, feats, opr, W, H, absgrad=False):
        from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_pixels
        tw, th = (W + self.tile - 1) // self.tile, (H + self.tile - 1) // self.tile
        _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, self.tile, tw, th)
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        return rasterize_to_pixels(means2d, conics, feats[None], opr, W, H, self.tile, offsets, flatten_ids, absgrad=absgrad)

    def render(self, p, view, deg, ss=1, depth=False, absgrad=False):
        """-> rgb (H,W,3) at 1x, alpha (H,W), z-depth (H,W) or None, info (strategy keys)."""
        import torch
        import torch.nn.functional as F
        from gsplat.cuda._wrapper import spherical_harmonics
        _, K, W, H, viewmat = view[:5]
        radii, means2d, depths, conics, opr = self.project(p, viewmat, K, W, H, ss)
        vis = (radii[0] > 0).all(-1)
        campos = torch.linalg.inv(viewmat)[:3, 3]
        dirs = p["means"] - campos
        if self.sh_band2 == "pcbug" and deg >= 2:
            from finetune_ply_viewer import eval_sh
            col = eval_sh(p["sh0"], p["shN"], torch.nn.functional.normalize(dirs, dim=-1), deg, True)
        else:
            col = torch.clamp_min(spherical_harmonics(deg, dirs, self.sh_coeffs(p, deg), masks=vis) + 0.5, 0.0)
        feats = torch.cat([col, depths[0][:, None]], 1) if depth else col
        out, alpha = self.raster(means2d, radii, self.sort_key(p, viewmat, depths), conics, feats, opr, W * ss, H * ss, absgrad=absgrad)
        out, alpha = out[0], alpha[0, ..., 0]
        if ss > 1:                                          # area-downsample to the 1x grid
            pool = lambda x: F.avg_pool2d(x.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0)
            out, alpha = pool(out), pool(alpha[..., None])[..., 0]
        d = out[..., 3] / alpha.clamp_min(1e-6) if depth else None
        info = {"means2d": means2d, "radii": radii, "width": W * ss, "height": H * ss, "n_cameras": 1,
                "gaussian_ids": None, "conics": conics, "depths": depths}
        return out[..., :3], alpha, d, info

    def edge_weights(self, p, view, edge):
        """EAS: per-gaussian sum_p edge_p T_i(p) alpha_i(p) (d of an edge-weighted constant render w.r.t. its colour)."""
        import torch
        _, K, W, H, viewmat = view[:5]
        with torch.no_grad():
            radii, means2d, depths, conics, opr = self.project(p, viewmat, K, W, H, 1)
        f = torch.zeros(len(p["means"]), 1, device=means2d.device, requires_grad=True)
        with torch.enable_grad():
            out, _ = self.raster(means2d.detach(), radii, self.sort_key(p, viewmat, depths), conics.detach(), f, opr.detach(), W, H)
            (out[0, ..., 0] * edge).sum().backward()
        return f.grad[:, 0].detach(), (radii[0] > 0).all(-1)

    @staticmethod
    def floors(p, views, min_px, chunk=1 << 21):
        """3D scale floor per gaussian: min_px * z / f at the closest training camera whose frustum holds it."""
        import torch
        best = torch.full((len(p["means"]),), float("inf"), device=p["means"].device)
        with torch.no_grad():
            for v in views:
                _, K, W, H, vm = v[:5]
                f = float(K[0, 0])
                for s in range(0, len(best), chunk):
                    x = p["means"][s:s + chunk] @ vm[:3, :3].T + vm[:3, 3]
                    z = x[:, 2]; u = K[0, 0] * x[:, 0] / z.clamp_min(1e-6) + K[0, 2]; w = K[1, 1] * x[:, 1] / z.clamp_min(1e-6) + K[1, 2]
                    ok = (z > 0.01) & (u > -0.15 * W) & (u < 1.15 * W) & (w > -0.15 * H) & (w < 1.15 * H)
                    val = torch.where(ok, z / f, torch.full_like(z, float("inf")))
                    best[s:s + chunk] = torch.minimum(best[s:s + chunk], val)
        best[~torch.isfinite(best)] = 0
        return best * min_px


# ---------------------------------------------------------------- appearance (per-frame exposure / white balance)
class Appearance:
    """Per-training-frame colour transform on the render: rows of 12 numbers, [A (3x3, row-major) | b (3)].
    gain: diag(A) = log-gain, off-diagonals unused; affine: M = I + A. Sparse rows (only the frame's own row moves)."""

    def __init__(self, mode, n, lr, dev):
        import torch
        self.mode = mode
        self.emb = torch.nn.Embedding(n, 12, sparse=True).to(dev)
        torch.nn.init.zeros_(self.emb.weight)
        self.opt = torch.optim.SparseAdam(self.emb.parameters(), lr=lr)

    def mats(self, w):
        import torch
        A, b = w[..., :9].reshape(*w.shape[:-1], 3, 3), w[..., 9:]
        eye = torch.eye(3, device=w.device)
        M = torch.diag_embed(torch.exp(torch.diagonal(A, dim1=-2, dim2=-1))) if self.mode == "gain" else eye + A
        return M, b

    def __call__(self, img, k):
        import torch
        w = self.emb(torch.tensor([k], device=img.device))[0]
        M, b = self.mats(w)
        return img @ M.T + b, w

    def recentre(self):
        """Mean transform = identity (gain: geometric mean gain 1): the splat keeps the average exposure."""
        import torch
        with torch.no_grad():
            W = self.emb.weight
            if self.mode == "gain":                          # only the diagonal and the bias are live
                m = torch.zeros(12, device=W.device); m[[0, 4, 8, 9, 10, 11]] = 1; W.mul_(m)
            W.sub_(W.mean(0, keepdim=True))

    def summary(self, names):
        import torch
        with torch.no_grad():
            M, b = self.mats(self.emb.weight)
            M, b = M.cpu().numpy(), b.cpu().numpy()
        g = np.stack([np.diagonal(m) for m in M])            # per-channel gain (affine: diagonal of M)
        lum = g @ np.array([0.299, 0.587, 0.114])
        pct = lambda x: [round(float(v), 4) for v in np.percentile(x, [0, 5, 25, 50, 75, 95, 100])]
        return {"mode": self.mode, "gain_pct_0_5_25_50_75_95_100": {"lum": pct(lum), "r": pct(g[:, 0]), "g": pct(g[:, 1]), "b": pct(g[:, 2])},
                "gain_std": [round(float(v), 4) for v in g.std(0)], "bias_std": [round(float(v), 4) for v in b.std(0)],
                "rb_ratio_pct": pct(g[:, 0] / g[:, 2]),
                "frames": {n: {"M": np.round(m, 5).tolist(), "b": np.round(c, 5).tolist()} for n, m, c in zip(names, M, b)}}


class BilagridAppearance:
    """--appearance bilagrid: per-training-frame bilateral grid of colour affines (Wang et al. 2024, gsplat's
    examples/lib_bilagrid.py; $BILAGRID_LIB, default $CEAR_WS/gsplat/examples), sliced by pixel position and luminance:
    spatially varying exposure / vignetting / tone. Same contract as Appearance: the 'w' returned for the identity pull
    is the frame's deviation from the identity grid scaled so app_reg * |w|^2 = app_reg * mean sq deviation, plus
    --bilagrid-tv x total variation; recentred to the mean identity after every step (no gauge drift)."""

    def __init__(self, n, lr, dev, shape=(16, 16, 8), tv=10.0):
        import torch
        sys.path.insert(0, os.environ.get("BILAGRID_LIB") or os.path.join(os.environ.get("CEAR_WS", ""), "gsplat", "examples"))
        from lib_bilagrid import BilateralGrid, slice as bslice
        self.grid = BilateralGrid(n, grid_X=shape[0], grid_Y=shape[1], grid_W=shape[2]).to(dev)
        self.ident = self.grid.grids[:1].detach().clone()
        self.opt = torch.optim.Adam(self.grid.parameters(), lr=lr, eps=1e-15)
        self.slice, self.tv, self._xy, self.mode = bslice, tv, {}, "bilagrid"

    def __call__(self, img, k):
        import torch
        H, W = img.shape[:2]
        if (H, W) not in self._xy:
            yy, xx = torch.meshgrid((torch.arange(H, device=img.device) + 0.5) / H, (torch.arange(W, device=img.device) + 0.5) / W, indexing="ij")
            self._xy[(H, W)] = torch.stack([xx, yy], -1)
        idx = torch.full((H, W, 1), k, device=img.device, dtype=torch.long)
        out = self.slice(self.grid, self._xy[(H, W)], img, idx)["rgb"]
        dev_ = (self.grid.grids[k] - self.ident[0]).flatten() / math.sqrt(self.ident.numel())
        reg = torch.sqrt((self.tv * self.grid.tv_loss()).clamp_min(0) + 1e-20)
        return out, torch.cat([dev_, reg[None]])

    def recentre(self):
        import torch
        with torch.no_grad():
            G = self.grid.grids; G.sub_(G.mean(0, keepdim=True) - self.ident)

    def summary(self, names):
        import torch
        with torch.no_grad():
            d = (self.grid.grids - self.ident).abs().mean((1, 2, 3, 4)).cpu().numpy()
        pct = [round(float(v), 4) for v in np.percentile(d, [0, 5, 25, 50, 75, 95, 100])]
        return {"mode": "bilagrid", "gain_pct_0_5_25_50_75_95_100": {"lum": pct}, "gain_std": [0, 0, 0], "bias_std": [0, 0, 0],
                "tv": float(self.grid.tv_loss()), "frames": {}}


def fit_affine(pred, gt):
    """Least-squares colour affine pred -> gt on one image (H,W,3 float in [0,1]); -> fitted image, M (3,3), b (3)."""
    import torch
    X = torch.cat([pred.reshape(-1, 3), torch.ones_like(pred.reshape(-1, 3)[:, :1])], 1).double()
    sol = torch.linalg.lstsq(X, gt.reshape(-1, 3).double()).solution            # (4,3)
    return (X @ sol).float().reshape(pred.shape).clamp(0, 1), sol[:3].T.float(), sol[3].float()


# ---------------------------------------------------------------- opacity clean-up (--opa-reg / --opa-prune / --contrib-prune)
def opa_term(op, kind):
    """Per-gaussian binarisation penalty, 0 at opacity 0 and 1: binary o(1-o) (max 0.25) or entropy -o ln o - (1-o) ln(1-o)."""
    import torch
    if kind == "binary":
        return op * (1 - op)
    o = op.clamp(1e-6, 1 - 1e-6)
    return -(o * torch.log(o) + (1 - o) * torch.log(1 - o))


def prune_mask(params, opts, state, mask):
    """Remove gaussians (params, Adam moments, per-gaussian strategy state; MCMC's binomial table is left alone)."""
    import torch
    from gsplat.strategy.ops import _update_param_with_optimizer
    n = len(mask); sel = torch.where(~mask)[0]
    with torch.no_grad():
        _update_param_with_optimizer(lambda _k, p: torch.nn.Parameter(p[sel], requires_grad=p.requires_grad),
                                     lambda _k, v: v[sel], params, opts)
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == n:
                state[k] = v[sel]


def contributions(rast, p, views):
    """Per-gaussian blending weight sum_views sum_px T_i(p) alpha_i(p) over the training views (pixel-equivalents)."""
    import torch
    tot = torch.zeros(len(p["means"]), device=p["means"].device)
    for v in views:
        w, _ = rast.edge_weights(p, v, torch.ones(v[3], v[2], device=tot.device))
        tot += w
    return tot


def opa_stats(p):
    import torch
    with torch.no_grad():
        o = torch.sigmoid(p["opacities"].flatten())
        return {"n": len(o), "lt05": round(float((o < 0.05).float().mean()), 4), "semi": round(float(((o >= 0.05) & (o <= 0.95)).float().mean()), 4),
                "gt95": round(float((o > 0.95).float().mean()), 4)}


# ---------------------------------------------------------------- export
def save_ply(path, p, rast, comments=()):
    import torch
    from plyfile import PlyData, PlyElement
    with torch.no_grad():
        m = p["means"].cpu().numpy(); s = torch.log(rast.scales(p)).cpu().numpy()
        q = torch.nn.functional.normalize(p["quats"], dim=-1).cpu().numpy()
        sh0 = p["sh0"][:, 0].cpu().numpy(); shN = p["shN"].clone()
        if rast.sh_band2 == "zero" and shN.shape[1] >= 5:
            shN[:, 3:5] = 0
        rest = shN.permute(0, 2, 1).reshape(len(m), -1).cpu().numpy()       # f_rest channel-major (3DGS layout)
        op = p["opacities"].reshape(-1, 1).cpu().numpy()
    cols = [m, np.zeros_like(m), sh0, rest, op, s, q]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest.shape[1])] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.ascontiguousarray(np.concatenate(cols, 1).astype(np.float32))
    el = arr.view([(nm, "f4") for nm in names]).reshape(-1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")], comments=list(comments)).write(path)


# ---------------------------------------------------------------- training
def viewer_K(K, W, H, viewer):
    """The camera scripts/viewer_shots.py renders: square pixels (pc274: f = fy, vertical fov; ss352: horizontal fov
    of the canvas when wider than tall), canvas centred on the principal point, cropped at an integer offset."""
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    CW, CH = math.ceil(2 * max(cx, W - cx)), math.ceil(2 * max(cy, H - cy))
    x0, y0 = round(CW / 2 - cx), round(CH / 2 - cy)
    f = (fx if CW > CH else fy) if viewer == "ss352" else fy
    Kv = K.clone(); Kv[0, 0] = Kv[1, 1] = f; Kv[0, 2], Kv[1, 2] = CW / 2 - x0, CH / 2 - y0
    return Kv


def evaluate(rast, p, views, deg, save_dir=None, viewer=None, fit=False):
    """-> (PSNR, SSIM) of the 8-bit render; fit: + (PSNR, SSIM) after a per-frame least-squares colour affine."""
    import torch
    ps, ss_, pf, sf = [], [], [], []
    with torch.no_grad():
        for v in views:
            if viewer:
                v = (v[0], viewer_K(v[1], v[2], v[3], viewer)) + tuple(v[2:])
            img = rast.render(p, v, deg)[0]
            img = torch.round(img.clamp(0, 1) * 255) / 255             # what an 8-bit viewer frame holds
            gt = v[5].float() / 255
            ps.append(psnr(img, gt)); ss_.append(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
            if fit:
                fi = torch.round(fit_affine(img, gt)[0] * 255) / 255
                pf.append(psnr(fi, gt)); sf.append(float(ssim(fi.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
            if save_dir:
                from PIL import Image
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
                    os.path.join(save_dir, os.path.splitext(v[0])[0] + ".png"))
    r = (round(float(np.mean(ps)), 3), round(float(np.mean(ss_)), 4)) if ps else (None, None)
    return r + ((round(float(np.mean(pf)), 3), round(float(np.mean(sf)), 4)) if pf else (None, None)) if fit else r


def load_ply_params(path, dev):
    import torch
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]; names = [q.name for q in v.properties]
    col = lambda pre: np.stack([np.asarray(v[n], dtype=np.float32) for n in
                                sorted((n for n in names if n.startswith(pre)), key=lambda x: int(x.split("_")[-1]))], 1)
    rest = col("f_rest_") if any(n.startswith("f_rest_") for n in names) else np.zeros((len(v), 0), np.float32)
    p = {"means": np.stack([v["x"], v["y"], v["z"]], 1), "scales": col("scale_"), "quats": col("rot_"),
         "opacities": np.asarray(v["opacity"]), "sh0": col("f_dc_")[:, None],
         "shN": rest.reshape(len(v), 3, -1).transpose(0, 2, 1)}
    return {k: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev) for k, a in p.items()}


def render_ply(args, rast, dev):
    """--render-ply: render at --eval-dataset's cameras (viewer camera) -> PSNR vs GT and vs --compare-dir renders."""
    import torch
    from PIL import Image
    p = load_ply_params(args.render_ply, dev); deg = int(round(math.sqrt(p["shN"].shape[1] + 1))) - 1
    ev = load_views(args.eval_dataset, 0, "all", dev)[:: args.render_every]
    vs_gt, vs_ref = [], []
    with torch.no_grad():
        for v in ev:
            vv = (v[0], viewer_K(v[1], v[2], v[3], args.viewer)) + tuple(v[2:])
            img = torch.round(rast.render(p, vv, deg)[0].clamp(0, 1) * 255) / 255
            vs_gt.append(psnr(img, v[5].float() / 255))
            stem = os.path.splitext(v[0])[0] + ".png"
            if args.compare_dir and os.path.exists(os.path.join(args.compare_dir, stem)):
                vs_ref.append(psnr(img, load_rgb(os.path.join(args.compare_dir, stem), dev).float() / 255))
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(os.path.join(args.out, stem))
    r = {"ply": args.render_ply, "viewer": args.viewer, "eps2d": rast.eps2d, "cull_l1": rast.cull_l1, "n": len(vs_gt),
         "psnr_vs_gt": round(float(np.mean(vs_gt)), 3), "psnr_vs_compare": round(float(np.mean(vs_ref)), 3) if vs_ref else None}
    print("VGS_RENDER " + json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="COLMAP text dataset (sparse/0/*.txt incl. points3D.txt, images/)")
    ap.add_argument("--out", default=None, help="output dir (required for training; --render-ply: optional render dir)")
    ap.add_argument("--test-split-interval", type=int, default=8)
    ap.add_argument("--eval-dataset", default=None, help="held-out set scored at the end in this rasteriser")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--strategy", choices=("improved", "mcmc", "default"), default="improved")
    ap.add_argument("--budget", type=int, default=4_000_000, help="improved: Improved-GS budget; mcmc: cap_max")
    ap.add_argument("--densify-stop", type=int, default=None, help="default: 15000 (improved/default), 25000 (mcmc), x iters/30000")
    ap.add_argument("--viewer", choices=tuple(PROFILES), default="ss352")
    ap.add_argument("--eps2d", type=float, default=None, help="screen dilation px^2 (default from --viewer)")
    ap.add_argument("--cull-l1", type=float, default=None, help="minimum dilated major eigenvalue px^2 (default from --viewer)")
    ap.add_argument("--near", type=float, default=1e-4, help="near plane (PlayCanvas draws any splat in front of the camera)")
    ap.add_argument("--cap-px", type=float, default=1024.0, help="viewer's quad half-extent cap in px (0 = off)")
    ap.add_argument("--aa", action="store_true", help="antialiased (opacity compensation) mode")
    ap.add_argument("--sort", choices=("depth", "radial", "dual"), default="depth",
                    help="blend order: depth (SuperSplat <= 3.6 / its 'Depth' sort mode), radial (distance to the camera: "
                         "SuperSplat 3.7's default 'Radial'), dual (alternate per step: good in both)")
    ap.add_argument("--ss", type=int, default=1, help="supersampling factor (render ss x, area-downsample, loss at 1x)")
    ap.add_argument("--min-px", type=float, default=0.0, help="3D scale floor in pixels at the closest training camera; 0 = off")
    ap.add_argument("--sh-band2", choices=("zero", "pcbug", "std"), default=None, help="default: zero (pc274), std (ss352)")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--depth-lambda", type=float, default=0.0)
    ap.add_argument("--mask-dir", default=None, help="<stem>_mask.png per training frame (255 = ignore in the loss)")
    ap.add_argument("--depth-trim", type=float, default=0.95)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--bilagrid-tv", type=float, default=10.0, help="--appearance bilagrid: total-variation weight")
    ap.add_argument("--appearance", choices=("none", "gain", "affine", "bilagrid"), default="none",
                    help="per-training-frame colour transform on the render before the loss (exposure / white balance)")
    ap.add_argument("--app-reg", type=float, default=0.01, help="L2 pull of the frame's transform toward identity")
    ap.add_argument("--app-lr", type=float, default=2e-3, help="SparseAdam lr (decays 10x over training)")
    ap.add_argument("--app-start", type=int, default=None, help="first step the transforms are used (default 1000 x iters/30000)")
    ap.add_argument("--bg", choices=("random", "black"), default="random",
                    help="training background (random: full opacity is learnt, any viewer background works)")
    ap.add_argument("--init-opacity", type=float, default=None, help="default 0.1 (0.5 for mcmc)")
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--grad-threshold", type=float, default=None, help="improved: 0.0003; default: 0.0008 (absgrad)")
    ap.add_argument("--no-eas", action="store_true"); ap.add_argument("--no-mu", action="store_true")
    ap.add_argument("--opa-reg", choices=("none", "entropy", "binary"), default="none",
                    help="late opacity binarisation penalty on the gaussians drawn in the step's view")
    ap.add_argument("--opa-lambda", type=float, default=0.01, help="--opa-reg weight (x mean over the view's gaussians)")
    ap.add_argument("--opa-start", type=float, default=0.7, help="fraction of --iters where the --opa-reg ramp starts")
    ap.add_argument("--opa-ramp", type=float, default=0.1, help="ramp length (fraction of --iters) to the full weight")
    ap.add_argument("--opa-prune", type=float, default=0.0, help="after the ramp: prune opacity below this (0 = off)")
    ap.add_argument("--contrib-prune", type=float, default=0.0,
                    help="after the ramp: prune gaussians whose blending weight summed over all training views is "
                         "below this many pixels (0 = off)")
    ap.add_argument("--prune-every", type=int, default=None, help="prune interval after the ramp (default 2000 x iters/30000; "
                    "also once after the last step)")
    ap.add_argument("--ply-comment", action="append", default=[], help="extra PLY header comment (repeatable)")
    ap.add_argument("--render-ply", default=None, help="diagnostics: render this PLY at --eval-dataset (no training)")
    ap.add_argument("--compare-dir", default=None, help="--render-ply: renders to score against (e.g. viewer_shots output)")
    ap.add_argument("--render-every", type=int, default=4)
    ap.add_argument("--allow-kernel-mismatch", action="store_true", help="debug: train with other kernels than --viewer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=0, help="held-out eval interval (0: end only)")
    import methods; methods.add_args(ap)                    # scripts/methods: optional fidelity methods (default off)
    args = ap.parse_args()

    import torch
    import gsplat
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    from improved_gs_strategy import ImprovedStrategy, edge_map
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"; t_start = time.time()
    pf = os.path.join(os.path.dirname(gsplat.__file__), "VIEWER_PROFILE")
    cut = open(pf).read().strip() if os.path.exists(pf) else "stock"
    print(f"[vgs] gsplat {gsplat.__version__} from {gsplat.__file__}: kernels {cut}", flush=True)
    if cut != args.viewer and not args.allow_kernel_mismatch:
        raise SystemExit(f"[vgs] --viewer {args.viewer} needs PYTHONPATH=$CEAR_WS/gsplat-viewer-{args.viewer} "
                         f"(env/make_gsplat_viewer_venv.sh); loaded: {cut}")
    rast = ViewerGS(args)
    if args.render_ply:                                     # diagnostics: render a PLY, score vs GT and a viewer's renders
        render_ply(args, rast, dev); return
    if not (args.dataset and args.out):
        raise SystemExit("--dataset and --out are required for training")
    tr = load_views(args.dataset, args.test_split_interval, "train", dev, depth=args.depth_lambda > 0)
    ev = load_views(args.eval_dataset, 0, "all", dev) if args.eval_dataset else []
    nd = sum(v[6] is not None for v in tr)
    masks = {}
    if args.mask_dir:                                       # transient pixels (people): excluded from every loss term
        from PIL import Image
        for k, v in enumerate(tr):
            f = os.path.join(args.mask_dir, os.path.splitext(os.path.basename(v[0]))[0] + "_mask.png")
            if os.path.exists(f):
                m = torch.from_numpy(np.asarray(Image.open(f).convert("L")) > 127).to(dev)
                if bool(m.any()):
                    masks[k] = m[..., None].float()
        print(f"[vgs] masks: {len(masks)}/{len(tr)} training frames have masked pixels "
              f"(mean {np.mean([float(m.mean()) for m in masks.values()]) if masks else 0:.4f} of those frames)", flush=True)
    print(f"[vgs] {len(tr)} training views ({args.dataset}, test_split_interval {args.test_split_interval}), {len(ev)} eval views; "
          f"viewer {args.viewer}: eps2d {rast.eps2d} px^2, cull l1 < {rast.cull_l1} px^2, aa {args.aa}, ss {args.ss}, "
          f"min_px {args.min_px}, sh band-2 {args.sh_band2}; depth targets {nd} (lambda {args.depth_lambda})", flush=True)
    init_op = args.init_opacity if args.init_opacity is not None else (0.5 if args.strategy == "mcmc" else 0.1)
    params = init_params(args.dataset, dev, init_op, args.init_scale, args.sh_degree, args.max_points, args.seed)
    centres = np.stack([torch.linalg.inv(v[4])[:3, 3].cpu().numpy() for v in tr])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max()) * 1.1
    print(f"[vgs] init {len(params['means'])} points from points3D.txt, opacity {init_op}, scene_scale {scene_scale:.3f}", flush=True)

    lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr[k], "name": k}], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / args.iters))
    f_it = args.iters / 30000.0
    if args.strategy == "improved":
        stop = args.densify_stop or int(15000 * f_it)
        strat = ImprovedStrategy(budget=args.budget, refine_stop_iter=stop, refine_start_iter=int(500 * f_it),
                                 grad_threshold=args.grad_threshold or 0.0003, eas=not args.no_eas, mu=not args.no_mu,
                                 reset_every=int(3000 * f_it), rap_offset=int(300 * f_it), early_prune_iter=int(300 * f_it),
                                 eas_full_every=int(3000 * f_it), eas_full_offset=int(400 * f_it), eas_full_until=int(9000 * f_it),
                                 mu_start=int(20000 * f_it), mu_second_start=int(24000 * f_it))
        state = strat.initialize_state(scene_scale)
    elif args.strategy == "mcmc":
        strat = MCMCStrategy(cap_max=args.budget, refine_stop_iter=args.densify_stop or int(25000 * f_it),
                             refine_start_iter=int(500 * f_it), verbose=False)
        state = strat.initialize_state()
    else:
        strat = DefaultStrategy(absgrad=True, grow_grad2d=args.grad_threshold or 0.0008, refine_stop_iter=args.densify_stop or int(15000 * f_it),
                                refine_start_iter=int(500 * f_it), reset_every=int(3000 * f_it), verbose=False)
        state = strat.initialize_state(scene_scale=scene_scale)
    strat.check_sanity(params, opts)
    fm = methods.Hooks(args, params, opts, strat, scene_scale, rast, tr)
    edges = {}

    def edge_fn(i):
        v = tr[i]
        if i not in edges:
            e = edge_map(v[5])
            if i in masks:
                e = e * (1 - masks[i][..., 0]).reshape(e.shape)
            edges[i] = e.half()
        return rast.edge_weights(params, v, edges[i].float())

    app = (BilagridAppearance(len(tr), args.app_lr, dev, tv=args.bilagrid_tv) if args.appearance == "bilagrid" else
           Appearance(args.appearance, len(tr), args.app_lr, dev)) if args.appearance != "none" else None
    app_start = args.app_start if args.app_start is not None else int(1000 * f_it)
    app_sched = torch.optim.lr_scheduler.ExponentialLR(app.opt, gamma=0.1 ** (1.0 / args.iters)) if app else None
    rn = {}
    s0 = int(args.opa_start * args.iters); s1 = s0 + max(int(args.opa_ramp * args.iters), 1)   # opacity clean-up schedule
    pe = args.prune_every or max(int(2000 * f_it), 1)
    cleaning, clean_log = args.opa_prune > 0 or args.contrib_prune > 0, []
    if args.opa_reg != "none" or cleaning:
        print(f"[vgs] opacity clean-up: reg {args.opa_reg} x {args.opa_lambda} ramped {s0}->{s1}; prune opacity < {args.opa_prune}, "
              f"contribution < {args.contrib_prune} px every {pe} from {s1} and at the end", flush=True)

    def clean(step):
        n = len(params["means"]); rec = {"step": step, "n": n}
        with torch.no_grad():
            m = torch.zeros(n, dtype=torch.bool, device=dev)
            if args.opa_prune > 0:
                m |= torch.sigmoid(params["opacities"].flatten()) < args.opa_prune; rec["opa"] = int(m.sum())
            if args.contrib_prune > 0:
                c = contributions(rast, params, tr); mc = c < args.contrib_prune
                rec["contrib"], rec["contrib_only"] = int(mc.sum()), int((mc & ~m).sum())
                rec["contrib_pct_1_5_25_50"] = [round(float(x), 3) for x in torch.quantile(c[torch.randperm(n, device=dev)[:1 << 20]],
                                                torch.tensor([0.01, 0.05, 0.25, 0.5], device=dev)).tolist()]
                m |= mc
        rec["pruned"] = int(m.sum())
        if rec["pruned"]:
            prune_mask(params, opts, state, m)
        clean_log.append(rec); print(f"[vgs] clean-up {rec}", flush=True)

    curve, order, t0 = [], [], time.time()
    os.makedirs(args.out, exist_ok=True)
    for step in range(args.iters):
        if not order:
            order = list(np.random.permutation(len(tr)))
        k = int(order.pop()); v = fm.view(tr[k], step, k in masks)
        deg = min(step // 1000, args.sh_degree)
        if args.min_px > 0 and (rast.floor is None or len(rast.floor) != len(params["means"]) or step % 1000 == 0):
            rast.floor = ViewerGS.floors(params, tr, args.min_px)
        img, alpha, d, info = rast.render(params, v, deg, ss=args.ss, depth=v[6] is not None,
                                          absgrad=args.strategy in ("improved", "default"))
        strat.step_pre_backward(params, opts, state, step, info)
        gt = v[5].float() / 255
        if args.bg == "random":                             # opaque real frames: transparency must not pay off
            img = img + (1 - alpha[..., None]) * torch.rand(3, device=dev)
        lapp = None
        if app is not None and step >= app_start:          # the camera's exposure / WB for this frame, render side only
            img, w = app(img, k)
            lapp = (w * w).sum()
        img = fm.post(img, v[0], step)                     # scripts/methods render-side transforms (--blur-kernel)
        if k in masks:                                      # transient pixels: the render copies the real image there
            img = img * (1 - masks[k]) + gt * masks[k]
        loss = (1 - args.ssim_lambda) * (img - gt).abs().mean() + \
            args.ssim_lambda * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss = loss + fm.loss(img, gt, step, info)
        ld = torch.zeros((), device=dev)
        if v[6] is not None:
            key = (v[2], v[3], tuple(v[1].flatten().tolist()))
            if key not in rn:
                K = v[1].cpu().numpy(); uu, vv = np.meshgrid(np.arange(v[2]) + 0.5, np.arange(v[3]) + 0.5)
                rn[key] = torch.tensor(np.sqrt(((uu - K[0, 2]) / K[0, 0]) ** 2 + ((vv - K[1, 2]) / K[1, 1]) ** 2 + 1),
                                       dtype=torch.float32, device=dev)
            msk = (v[6] > 0) & (alpha.detach() > 0.5)
            if msk.any():
                err = ((d * rn[key])[msk] - v[6][msk]).abs()
                err = err.sort().values[: max(1, int(err.numel() * args.depth_trim))]
                ld = err.mean(); loss = loss + args.depth_lambda * ld
        if lapp is not None:
            loss = loss + args.app_reg * lapp
        lopa = None
        if args.opa_reg != "none" and step >= s0:          # push the drawn gaussians to opaque or gone
            vis = (info["radii"][0] > 0).all(-1)
            if bool(vis.any()):
                lopa = opa_term(torch.sigmoid(params["opacities"].flatten()[vis]), args.opa_reg).mean()
                loss = loss + args.opa_lambda * min(1.0, (step - s0 + 1) / (s1 - s0)) * lopa
        if args.strategy == "mcmc":
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        stepping = strat.should_step_optimizer(step) if args.strategy == "improved" else True
        if stepping:
            for o in opts.values():
                o.step(); o.zero_grad(set_to_none=True)
        sched.step(); fm.after_step(step, stepping)
        if lapp is not None:
            app.opt.step(); app.opt.zero_grad(set_to_none=True); app.recentre()
        if app_sched is not None:
            app_sched.step()
        if args.strategy == "improved":
            strat.step_post_backward(params, opts, state, step, info, edge_fn=edge_fn if not args.no_eas else None, n_views=len(tr))
        elif args.strategy == "mcmc":
            strat.step_post_backward(params, opts, state, step, info, lr=sched.get_last_lr()[0])
        else:
            strat.step_post_backward(params, opts, state, step, info, packed=False)
        if cleaning and step > s1 and (step - s1) % pe == 0:
            clean(step)
        if step % args.log_every == 0 or step == args.iters - 1:
            e = {"step": step, "loss": round(loss.item(), 5), "psnr": round(psnr(img.detach().clamp(0, 1), gt), 3),
                 "depth": round(ld.item(), 5), "n": len(params["means"]), "visible": int((info["radii"] > 0).all(-1).sum()),
                 "sec": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            if args.opa_reg != "none" or cleaning:
                e["opa"] = opa_stats(params); e["lopa"] = None if lopa is None else round(lopa.item(), 5)
            if args.eval_every and step and step % args.eval_every == 0 and ev:
                e["heldout_psnr"] = evaluate(rast, params, ev, deg, viewer=args.viewer)[0]
            curve.append(e); print(f"[vgs] {e}", flush=True)
    if cleaning:
        clean(args.iters)
    train_s = time.time() - t0
    fm.finalize(args.out)                                   # EMA swap-in before eval / export
    if args.min_px > 0:
        rast.floor = ViewerGS.floors(params, tr, args.min_px)
    deg = min(args.iters // 1000, args.sh_degree)
    res = {"heldout_psnr": None, "heldout_ssim": None, "opacity": opa_stats(params), "clean_up": clean_log}
    print(f"[vgs] opacity: {res['opacity']}", flush=True)
    if ev:      # at the viewer_shots camera (comparable with the viewer metric) and at the true camera
        res["heldout_psnr"], res["heldout_ssim"], res["heldout_psnr_fit"], res["heldout_ssim_fit"] = evaluate(
            rast, params, ev, deg, os.path.join(args.out, "renders_trainer"), args.viewer, fit=True)
        res["heldout_psnr_truecam"], res["heldout_ssim_truecam"] = evaluate(rast, params, ev, deg)
        print(f"[vgs] held-out ({args.eval_dataset}, this rasteriser, viewer camera): PSNR {res['heldout_psnr']} SSIM "
              f"{res['heldout_ssim']} (per-frame colour affine fitted: {res['heldout_psnr_fit']} / {res['heldout_ssim_fit']}); "
              f"true camera {res['heldout_psnr_truecam']} / {res['heldout_ssim_truecam']}", flush=True)
    if app is not None:
        aj = app.summary([v[0] for v in tr]); json.dump(aj, open(os.path.join(args.out, "appearance.json"), "w"), indent=1)
        res["appearance"] = {k: aj[k] for k in aj if k != "frames"}
        print(f"[vgs] appearance ({args.appearance}): luminance gain pct 0/5/25/50/75/95/100 "
              f"{aj['gain_pct_0_5_25_50_75_95_100']['lum']}, gain std rgb {aj['gain_std']}, bias std {aj['bias_std']}", flush=True)
    comments = [f"train_viewer_gsplat viewer={args.viewer} eps2d={rast.eps2d} strategy={args.strategy} ss={args.ss} aa={args.aa} sort={args.sort}"
                + (f" appearance={args.appearance}" if app is not None else "")]
    if args.aa:
        comments.append("SplatRenderMode mip")
    comments += args.ply_comment
    save_ply(os.path.join(args.out, "splat.ply"), params, rast, comments)
    log = {"args": vars(args), "kernel_cut": cut, "gsplat": gsplat.__file__, "eps2d": rast.eps2d, "cull_l1": rast.cull_l1,
           "n_train": len(tr), "n_eval": len(ev), "scene_scale": scene_scale, "n_gaussians": len(params["means"]),
           "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t_start, 1), "curve": curve, **res,
           "strategy_log": getattr(strat, "log", [])[-400:]}
    json.dump(log, open(os.path.join(args.out, "train.json"), "w"), indent=2)
    print(f"[vgs] {len(params['means'])} gaussians -> {os.path.join(args.out, 'splat.ply')} ({train_s / 60:.1f} min training)")
    print("VGS_TRAIN_DONE")


if __name__ == "__main__":
    main()
