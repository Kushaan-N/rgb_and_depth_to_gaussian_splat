"""Remove added gaussians that change what the base splat shows where it is trusted (3DGRUT venv, GPU).

A completion / generation stage may only touch what no training camera observed, so the edited splat must
render the training views exactly like the splat it started from. This measures it and enforces it:
render base and edited splats at many poses, find changed pixels, attribute them to ADDED gaussians (the
last --added entries of the edited checkpoint) and remove those, until every view renders like the base.
Real gaussians are never removed or changed.

Poses (render-only, no ground-truth pixels are read, so held-out / other-recording poses do not leak):
  train   every frame of --dataset (COLMAP text)                                  rule: strict
  jitter  --jitter K perturbed copies of each training pose (--jitter-m, --jitter-deg)  rule: strict
  extra   --extra-poses DIR ... COLMAP models whose images need not exist (held-out arc, off-path poses,
          another recording's poses)                                            rule: --extra-rule
Rules per pixel: changed = mean |rgb_edited - rgb_base| > --thr. strict: every changed pixel is harm.
fill: a changed pixel is allowed where the edited surface lies clearly IN FRONT of the base surface
(depth < base depth * (1 - --fill-margin-rel), or the base is transparent there) — a completed surface
covering what the base sees THROUGH an object; changing a surface the base already has is harm.
Attribution: an added gaussian is flagged if its projected footprint (--footprint-sigma x sigma of its
projected 2D covariance) touches a harmful pixel where it is not hidden behind the edited render's surface
(its distance < rendered depth * (1 + --depth-tol-rel), or the pixel is transparent). If a view is still
below target but nothing is flagged, the depth test is dropped for that view (escalation).
Stops when every view's harm PSNR (render vs base over harmful-rule pixels, see report) >= --target-psnr,
or nothing is flagged, or after --iters rounds.

Outputs: <out> (checkpoint, same layout as --edited minus the dropped rows), <out stem>_harm.json (per-set
min / median harm PSNR, changed-pixel share, per-iteration history), <out stem>_kept.npy (bool per input
added gaussian; used by face_textures.py / preserve_finetune.py / callers that track added gaussians).

    python scripts/harm_filter.py --base base.pt --edited completed.pt --added 15907 --dataset train/ --out filtered.pt \
        [--extra-poses heldout/ xviews/] [--jitter 1 --jitter-m 0.03 --jitter-deg 1.5 --sim3-json sim3.json]

Library use (preserve_finetune.py, face_textures.py, object_back_masks.py): load_model, Gauss, pose_set,
render, footprint_radius, box_max.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ shared 3DGUT rendering helpers
def load_model(path):
    """(MixtureOfGaussians on cuda, raw checkpoint dict) — no dataset / writer side effects."""
    import torch
    from threedgrut.model.model import MixtureOfGaussians
    ck = torch.load(path, map_location="cuda", weights_only=False)
    model = MixtureOfGaussians(ck["config"])
    model.init_from_checkpoint(ck, setup_optimizer=False)
    model.build_acc()
    return model, ck


class Gauss:
    """The tensors 3DGUT's tracer reads, outside the nn.Module so they can be subset / edited / made
    differentiable freely: positions, rotation, scale, density (pre-activation), albedo, specular."""

    def __init__(self, model, sel=None, **over):
        import torch
        self.m = model
        g = lambda t: t.detach() if sel is None else t.detach()[sel]
        self.positions, self.rotation, self.scale = g(model.positions), g(model.rotation), g(model.scale)
        self.density, self.albedo, self.specular = g(model.density), g(model.features_albedo), g(model.features_specular)
        for k, v in over.items():
            setattr(self, k, v)
        self.n_active_features, self.ray_feature_dim = model.n_active_features, model.ray_feature_dim
        self._torch = torch

    @property
    def num_gaussians(self):
        return self.positions.shape[0]

    def get_rotation(self):
        return self.m.rotation_activation(self.rotation)

    def get_scale(self):
        return self.m.scale_activation(self.scale)

    def get_density(self):
        return self.m.density_activation(self.density)

    def get_features(self):
        return self._torch.cat([self.albedo, self.specular], 1)


def _ds_dir(d):
    d = os.path.abspath(d.rstrip("/"))
    if os.path.basename(d) == "0" and os.path.basename(os.path.dirname(d)) == "sparse":
        d = os.path.dirname(os.path.dirname(d))
    assert os.path.exists(os.path.join(d, "sparse", "0")), f"{d}: no sparse/0 COLMAP model"
    return d


def pose_set(ds, tag, rule, every=1, jitter=0, jitter_units=0.0, jitter_deg=0.0, seed=0):
    """Views of a COLMAP model (images need not exist): list of dicts with name, set tag, rule, 4x4 C2W pose,
    pinhole K, image size and the 3DGRUT ray bundle of its camera. With jitter>0: `jitter` perturbed copies
    of every pose instead (translation N(0, jitter_units) per axis, rotation about a random axis N(0, deg))."""
    import torch
    from scipy.spatial.transform import Rotation
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from build_depth_targets import read_text_model
    ds = _ds_dir(ds)
    cams, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
    if any(not os.path.exists(os.path.join(ds, "images", n)) for n, *_ in imgs):
        # 3DGRUT reads an image per camera for its size: stage placeholders for render-only models
        import tempfile
        from PIL import Image
        st = tempfile.mkdtemp(prefix="poses_", dir=os.environ.get("TMPDIR"))
        os.symlink(os.path.join(ds, "sparse"), os.path.join(st, "sparse"))
        for cid, (W, H, _) in cams.items():
            Image.new("RGB", (W, H)).save(os.path.join(st, f"placeholder_{cid}.png"))
        for n, cid, *_ in imgs:
            dst = os.path.join(st, "images", n); os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = os.path.join(ds, "images", n)
            os.symlink(src if os.path.exists(src) else os.path.join(st, f"placeholder_{cid}.png"), dst)
        ds = st
    D = ColmapDataset(ds, device="cuda", split="train", test_split_interval=0)
    cache = D._lazy_worker_intrinsics_cache()
    rng = np.random.default_rng(seed); out = []
    for i in range(0, len(D), every):
        cid = D.cam_extrinsics[i].camera_id; W, H, (fx, fy, cx, cy) = cams[cid]
        prm, ro, rd, cname, _ = cache[D.get_intrinsics_idx(i)]
        name = os.path.splitext(os.path.basename(str(D.image_paths[i])))[0]
        for k in range(max(jitter, 1) if jitter else 1):
            P = D.poses[i].astype(np.float64).copy(); nm = name
            if jitter:
                ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
                P[:3, :3] = Rotation.from_rotvec(ax * np.deg2rad(rng.normal() * jitter_deg)).as_matrix() @ P[:3, :3]
                P[:3, 3] += rng.normal(size=3) * jitter_units; nm = f"{name}~j{k}"
            out.append({"name": nm, "set": tag, "rule": rule, "C2W": P, "K": (fx, fy, cx, cy), "W": W, "H": H,
                        "rays": (ro, rd, cname, prm)})
    return out


def render(gauss, view, grad=False):
    """3DGUT render of a Gauss at a view: dict rgb (H,W,3), depth (H,W, ray distance, 0 where opacity < 0.5),
    opacity (H,W). With grad=False runs under no_grad."""
    import torch
    from threedgrut.datasets.protocols import Batch
    ro, rd, cname, prm = view["rays"]
    b = Batch(rays_ori=ro, rays_dir=rd, T_to_world=torch.tensor(view["C2W"], dtype=torch.float32, device="cuda")[None],
              **{f"intrinsics_{cname}": prm})
    with torch.set_grad_enabled(grad):
        o = gauss.m.renderer.render(gauss, b, train=grad)
    op = o["pred_opacity"][0, ..., 0]
    dep = torch.where(op > 0.5, o["pred_dist"][0, ..., 0] / op.clamp_min(1e-3), torch.zeros_like(op))
    return {"rgb": o["pred_features"][0, ..., :3].clamp(0, 1) if not grad else o["pred_features"][0, ..., :3],
            "depth": dep.detach(), "opacity": op}


def world_cov(rot_q, log_scale, scale_act):
    """(N,3,3) world covariances from wxyz quaternions + pre-activation scales."""
    import torch
    q = torch.nn.functional.normalize(rot_q, dim=1); w, x, y, z = q.unbind(1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).view(-1, 3, 3)
    s = scale_act(log_scale)
    return R @ torch.diag_embed(s * s) @ R.transpose(1, 2)


def footprint_radius(X, Sig, view, k_sigma=3.0):
    """Projection of gaussians (X (N,3), Sig (N,3,3), dataset frame) into a view: u, v (float px), z, ray
    distance r, and the k-sigma radius of the projected 2D covariance (px)."""
    import torch
    P = torch.tensor(view["C2W"], device=X.device, dtype=X.dtype)
    Rcw = P[:3, :3].T; tcw = -Rcw @ P[:3, 3]; fx, fy, cx, cy = view["K"]
    Pc = X @ Rcw.T + tcw; x, y, z = Pc.unbind(1); zc = z.clamp_min(1e-6)
    u, v = fx * x / zc + cx, fy * y / zc + cy
    J = torch.zeros(len(X), 2, 3, device=X.device, dtype=X.dtype)
    J[:, 0, 0] = fx / zc; J[:, 0, 2] = -fx * x / zc ** 2; J[:, 1, 1] = fy / zc; J[:, 1, 2] = -fy * y / zc ** 2
    S2 = J @ (Rcw @ Sig @ Rcw.T) @ J.transpose(1, 2)
    a, b, c = S2[:, 0, 0], S2[:, 0, 1], S2[:, 1, 1]
    lam = (a + c) / 2 + torch.sqrt(((a - c) / 2) ** 2 + b * b)
    return u, v, z, torch.linalg.norm(Pc, dim=1), k_sigma * torch.sqrt(lam.clamp_min(0)) + 0.5


def box_max(M, u, v, rad, levels=8):
    """max of map M (H,W) over the square box of half-size >= rad around (u, v) (pyramid of dilated 3x3
    max-pools, half-sizes 2^k - 1, capped at 2^levels - 1); boxes partly outside the image are handled."""
    import torch
    import torch.nn.functional as F
    H, W = M.shape; pyr = [M]; cur = M[None, None]
    for k in range(levels):
        d = 2 ** k
        cur = F.max_pool2d(F.pad(cur, (d, d, d, d), value=float("-inf")), 3, 1, 0, dilation=d); pyr.append(cur[0, 0])
    pyr = torch.stack(pyr)
    lvl = torch.ceil(torch.log2(rad.clamp_min(0) + 1)).long().clamp(0, levels)
    ui = u.round().long(); vi = v.round().long()
    inside = (ui + rad >= 0) & (ui - rad < W) & (vi + rad >= 0) & (vi - rad < H)
    val = pyr[lvl, vi.clamp(0, H - 1), ui.clamp(0, W - 1)]               # clamped centre, same radius: conservative
    return torch.where(inside, val, torch.full_like(val, float("-inf")))


def harm_map(cur, base, rule, thr, fill_margin):
    """per pixel: |diff| (H,W), harmful-changed mask, squared error counted as harm (H,W)."""
    import torch
    d = (cur["rgb"].float() - base["rgb"].float()).abs().mean(-1)
    se = ((cur["rgb"].float() - base["rgb"].float()) ** 2).mean(-1)
    if rule == "fill":
        Db, Dc = base["depth"].float(), cur["depth"].float()
        legit = (Dc > 0) & ((Db <= 0) | (Dc < Db * (1 - fill_margin)))
        se = torch.where(legit, torch.zeros_like(se), se)
        return d, (d > thr) & ~legit, se
    return d, d > thr, se


def psnr_of(se_mean):
    return float("inf") if se_mean <= 0 else float(-10 * np.log10(se_mean))


def build_views(args):
    """training + jitter + extra pose sets from the common CLI flags."""
    views = pose_set(args.dataset, "train", "strict", args.every)
    if args.jitter:
        s = float(json.load(open(args.sim3_json))["scale"]) if args.sim3_json else 1.0
        views += pose_set(args.dataset, "jitter", "strict", args.every, args.jitter, args.jitter_m / s, args.jitter_deg)
    for i, e in enumerate(args.extra_poses or []):
        views += pose_set(e, f"extra{i}:{os.path.basename(_ds_dir(e))}", args.extra_rule, args.extra_every)
    return views


def add_pose_args(ap):
    """pose / rule flags shared by harm_filter.py and preserve_finetune.py (stable CLI)."""
    ap.add_argument("--dataset", required=True, help="training frames (COLMAP text); images are not read")
    ap.add_argument("--every", type=int, default=1, help="use every n-th training frame")
    ap.add_argument("--extra-poses", nargs="*", default=[], help="COLMAP models of render-only poses (no GT used)")
    ap.add_argument("--extra-every", type=int, default=1)
    ap.add_argument("--extra-rule", choices=("fill", "strict"), default="fill")
    ap.add_argument("--jitter", type=int, default=0, help="perturbed copies per training pose (strict rule)")
    ap.add_argument("--jitter-m", type=float, default=0.03, help="translation sigma (metres with --sim3-json)")
    ap.add_argument("--jitter-deg", type=float, default=1.5)
    ap.add_argument("--sim3-json", default=None, help="COLMAP->metric Sim3 (only for metre-valued flags)")
    ap.add_argument("--thr", type=float, default=0.004, help="changed pixel: mean |rgb diff| above this (0..1)")
    ap.add_argument("--fill-margin-rel", type=float, default=0.05)


def subset_ckpt(ck, keep):
    """checkpoint dict with every per-gaussian tensor row-selected by bool mask `keep` (types preserved)."""
    import torch
    N = len(keep); sel = torch.from_numpy(keep); out = dict(ck)
    for k_, v in ck.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
            t = v.detach()[sel.to(v.device)]
            out[k_] = torch.nn.Parameter(t, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else t
    out.pop("optimizer", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--added", type=int, required=True, help="number of added gaussians (the last entries)")
    ap.add_argument("--out", required=True)
    add_pose_args(ap)
    ap.add_argument("--target-psnr", type=float, default=50.0, help="stop when every view's harm PSNR is above this")
    ap.add_argument("--footprint-sigma", type=float, default=3.0)
    ap.add_argument("--dilate", type=int, default=1, help="dilate harmful pixels by this many px")
    ap.add_argument("--depth-tol-rel", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--stop", type=float, default=0.0, help="(legacy) also stop when the changed-pixel share is below this")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    views = build_views(args)
    sets = sorted({v["set"] for v in views})
    print(f"[harm] {len(views)} poses: " + ", ".join(f"{s} {sum(v['set'] == s for v in views)}" for s in sets), flush=True)
    bm, _ = load_model(args.base)
    gb = Gauss(bm)
    base = []
    for v in views:                                                    # base renders, kept on the GPU (half)
        o = render(gb, v); base.append({"rgb": o["rgb"].half(), "depth": o["depth"].half()})
    del gb, bm; torch.cuda.empty_cache()
    em, ck = load_model(args.edited)
    N = em.positions.shape[0]; n0 = N - args.added; assert 0 < n0 <= N and args.added > 0, "need >= 1 added gaussian"
    keep = np.ones(N, bool)
    Xa = em.positions.detach()[n0:].double()
    Sa = world_cov(em.rotation.detach()[n0:].double(), em.scale.detach()[n0:].double(), em.scale_activation)
    hist = []
    for it in range(args.iters + 1):
        g = Gauss(em, torch.from_numpy(keep).cuda())
        alive = torch.from_numpy(keep[n0:]).cuda()
        flag = torch.zeros(args.added, dtype=torch.bool, device="cuda")
        stats = {s: {"psnr": [], "changed": 0, "px": 0} for s in sets}; low = []
        for vi, v in enumerate(views):
            cur = render(g, v)
            d, hm, se = harm_map(cur, base[vi], v["rule"], args.thr, args.fill_margin_rel)
            p = psnr_of(se.mean().item()); st = stats[v["set"]]
            st["psnr"].append(p); st["changed"] += int(hm.sum()); st["px"] += hm.numel()
            if p < args.target_psnr:
                low.append(vi)
            if not hm.any():
                continue
            m = hm.float()[None, None]
            if args.dilate:
                m = F.max_pool2d(m, 2 * args.dilate + 1, 1, args.dilate)
            m = m[0, 0] > 0
            Dc = cur["depth"].float()
            M = torch.where(m, torch.where(Dc > 0, Dc, torch.full_like(Dc, 1e30)), torch.full_like(Dc, float("-inf")))
            u, vv, z, r, rad = footprint_radius(Xa, Sa, v, args.footprint_sigma)
            mx = box_max(M, u, vv, rad).double()
            hit = alive & (z > 0.02) & (mx * (1 + args.depth_tol_rel) >= r)
            if p < args.target_psnr and not hit.any():                   # escalation: footprint only
                hit = alive & (z > 0.02) & (mx > float("-inf"))
            flag |= hit
        rec = {"iter": it, "added_alive": int(keep[n0:].sum()), "flagged": int(flag.sum()), "views_below_target": len(low),
               "sets": {s: {"min_psnr": round(min(st["psnr"]), 2), "median_psnr": round(float(np.median(st["psnr"])), 2),
                            "changed_share": st["changed"] / max(st["px"], 1)} for s, st in stats.items()}}
        hist.append(rec)
        print(f"[harm] iter {it}: added alive {rec['added_alive']:,}, flagged {rec['flagged']:,}, views < {args.target_psnr:g} dB: "
              f"{len(low)}; " + "; ".join(f"{s} min {r_['min_psnr']} med {r_['median_psnr']} chg {r_['changed_share']:.1e}"
                                          for s, r_ in rec["sets"].items()), flush=True)
        done = not low or not flag.any() or it == args.iters
        if args.stop and all(r_["changed_share"] < args.stop for r_ in rec["sets"].values()):
            done = True
        if done:
            break
        keep[n0:][flag.cpu().numpy()] = False
    out = subset_ckpt(ck, keep)
    for k_ in ("positions", "density", "features_albedo"):             # provably untouched real gaussians
        assert torch.equal(out[k_][:n0].detach().cuda(), ck[k_][:n0].detach().cuda())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    stem = os.path.splitext(args.out)[0]
    np.save(stem + "_kept.npy", keep[n0:])
    last = hist[-1]
    info = {"added_in": args.added, "added_kept": int(keep[n0:].sum()), "converged": last["views_below_target"] == 0,
            "final": last["sets"], "iterations": hist,
            "params": {k: v for k, v in vars(args).items() if k not in ("base", "edited", "out")}}
    json.dump(info, open(stem + "_harm.json", "w"), indent=2)
    print(f"[harm] kept {keep[n0:].sum():,} of {args.added:,} added gaussians "
          f"({'all views >= %g dB' % args.target_psnr if info['converged'] else 'NOT converged: %d views below target' % last['views_below_target']})"
          f" -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
