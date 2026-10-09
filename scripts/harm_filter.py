"""Remove added gaussians that change what the training cameras see (3DGRUT venv, GPU).

A completion / generation stage should only touch what no training camera observed, so its result
must render the training views exactly like the splat it started from. Visibility tests predict that;
this measures it: render every training view with the base and the edited splat, find the pixels that
changed (|diff| > --thr, dilated), and drop each ADDED gaussian (the last --added entries of the edited
checkpoint) whose centre projects onto a changed pixel and is not hidden behind the rendered surface.
Repeats until the changed-pixel share is below --stop or --iters is reached.

    python scripts/harm_filter.py --base base.pt --edited completed.pt --added 15907 --dataset train/ --out filtered.pt
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covis_masks import cams_of


def render_all(ckpt, ds, every):
    """{image name: (rgb (H,W,3), ray distance (H,W), 0 where transparent)} for every `every`-th frame."""
    import torch
    from threedgrut.render import Renderer
    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform
    r = Renderer.from_checkpoint(checkpoint_path=ckpt, path=ds, out_dir=ds, save_gt=False, computes_extra_metrics=False)
    model, conf = r.model, r.conf
    conf.dataset.test_split_interval = 0
    dataset, _ = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(dataset, **configure_dataloader_for_platform(
        {"num_workers": 4, "batch_size": 1, "shuffle": False, "collate_fn": None}))
    names = [os.path.relpath(p, os.path.join(ds, "images")) for p in dataset.image_paths]
    out = {}
    for k, batch in enumerate(loader):
        if k % every:
            continue
        gb = dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            o = model(gb)
        op = o["pred_opacity"][0, ..., 0]
        out[names[k]] = (o["pred_features"][0, ..., :3].clip(0, 1).half(),
                         torch.where(op > 0.5, o["pred_dist"][0, ..., 0] / op.clamp_min(1e-3), torch.zeros_like(op)).half())
    del model, r
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--added", type=int, required=True, help="number of added gaussians (the last entries)")
    ap.add_argument("--dataset", required=True, help="training frames (COLMAP text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.03, help="changed pixel: mean |rgb diff| above this (0..1)")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--depth-tol-rel", type=float, default=0.05)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--stop", type=float, default=1e-4, help="stop when the changed-pixel share is below this")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    cams = {c[0]: c for c in cams_of(args.dataset)}
    base = render_all(args.base, args.dataset, args.every)
    ck = torch.load(args.edited, map_location="cpu", weights_only=False)
    N = ck["positions"].shape[0]; n0 = N - args.added
    keep = np.ones(N, bool); hist = []
    tmp = os.path.splitext(args.out)[0] + "_iter.pt"
    cur_path = args.edited
    for it in range(args.iters + 1):
        cur = render_all(cur_path, args.dataset, args.every)
        X = torch.tensor(ck["positions"].detach().double().numpy()[n0:], device=dev)
        alive = torch.tensor(keep[n0:], device=dev)
        harm = torch.zeros(args.added, dtype=torch.bool, device=dev); changed = total = 0
        for name, (rgb, dep) in cur.items():
            d = (rgb.float() - base[name][0].float()).abs().mean(-1)
            m = (d > args.thr).float()[None, None]
            if args.dilate:
                m = F.max_pool2d(m, 2 * args.dilate + 1, 1, args.dilate)
            m = m[0, 0] > 0; changed += int((d > args.thr).sum()); total += d.numel()
            if not m.any():
                continue
            _, W, H, fx, fy, cx, cy, Rcw, tcw, _ = cams[name]
            Pc = X @ torch.tensor(Rcw, device=dev).T + torch.tensor(tcw, device=dev); z = Pc[:, 2]
            u = torch.floor(fx * Pc[:, 0] / z + cx).long(); v = torch.floor(fy * Pc[:, 1] / z + cy).long()
            ok = alive & (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            ia = torch.nonzero(ok, as_tuple=True)[0]
            if not len(ia):
                continue
            uu, vv = u[ia], v[ia]
            dd = dep.double()[vv, uu]; r = torch.linalg.norm(Pc[ia], dim=1)
            harm[ia[m[vv, uu] & ((dd <= 0) | (r < dd * (1 + args.depth_tol_rel)))]] = True
        share = changed / max(total, 1)
        hist.append({"iter": it, "changed_pixel_share": share, "added_alive": int(keep[n0:].sum()), "harmful": int(harm.sum())})
        print(f"[harm] iter {it}: changed pixels {share:.2e}, added alive {keep[n0:].sum():,}, flagged {int(harm.sum()):,}", flush=True)
        if share < args.stop or not harm.any() or it == args.iters:
            break
        keep[n0:][harm.cpu().numpy()] = False
        sel = torch.from_numpy(keep)
        ed = dict(ck)
        for k_, v in ck.items():                                   # same per-gaussian handling as complete_objects
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
                t = v.detach()[sel]
                ed[k_] = torch.nn.Parameter(t, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else t
        torch.save(ed, tmp); cur_path = tmp
    sel = torch.from_numpy(keep)
    for k_, v in list(ck.items()):
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
            t = v.detach()[sel]
            ck[k_] = torch.nn.Parameter(t, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else t
    torch.save(ck, args.out)
    if os.path.exists(tmp):
        os.remove(tmp)
    info = {"added_in": args.added, "added_kept": int(keep[n0:].sum()), "iterations": hist,
            "params": {k: v for k, v in vars(args).items() if k not in ("base", "edited", "out", "dataset")}}
    json.dump(info, open(os.path.splitext(args.out)[0] + "_harm.json", "w"), indent=2)
    print(f"[harm] kept {keep[n0:].sum():,} of {args.added:,} added gaussians -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
