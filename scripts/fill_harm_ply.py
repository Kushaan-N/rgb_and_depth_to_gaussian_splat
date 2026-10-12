"""Measured harmlessness for an unseen-surface fill on ANY standard PLY: drop fill gaussians that change what the viewer
shows at a training camera (gsplat venv, GPU).

fill_unseen_surfaces.py predicts visibility from a depth/opacity render; where that render differs from the viewer
(windows: LiDAR measures beyond the glass, the viewer draws the glass partly transparent) fill discs show through.
This renders every training view with the viewer-exact rasteriser (scripts/finetune_ply_viewer.py ViewerRaster,
SuperSplat 3.5.2 profile, ~52 dB agreement with the live editor) for the splat alone and for splat + fill, and removes
each fill gaussian whose screen footprint (3 sigma of its largest scale) overlaps a changed pixel; repeats until no
view changes (or --iters). The fill gaussians are the LAST entries of --filled (as fill_unseen_surfaces.py writes).

    python scripts/fill_harm_ply.py --base splat.ply --filled splat_complete.ply --dataset <training frames> --out out.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--filled", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--display-scale", type=float, default=1.8)
    ap.add_argument("--thr", type=float, default=2.0 / 255, help="changed pixel: max |rgb diff| above this")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    import torch
    dev = "cuda"
    _, pb = load_ply(args.base, dev)
    vf, pf = load_ply(args.filled, dev)
    n0 = len(pb["means"]); m = len(pf["means"]) - n0
    assert m >= 0 and torch.equal(pf["means"][:n0], pb["means"]), "--filled must be --base + appended fill gaussians"
    deg = sh_deg(pb)
    R = ViewerRaster("ss352", "exact", display_scale=args.display_scale)
    fr = frames(args.dataset, 0, "all")[::args.every]
    with torch.no_grad():
        base = {}
        for name, cam, c2w in fr:
            base[name] = R.render(pb, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0].half()
    keep = torch.ones(m, dtype=torch.bool, device=dev); hist = []
    smax = torch.exp(pf["scales"][n0:]).max(1).values
    for it in range(args.iters):
        sel = torch.cat([torch.ones(n0, dtype=torch.bool, device=dev), keep])
        p = {k: v[sel] for k, v in pf.items()}
        X = pf["means"][n0:]
        harm = torch.zeros(m, dtype=torch.bool, device=dev); changed = 0; total = 0
        with torch.no_grad():
            for name, cam, c2w in fr:
                w2c = to_t(np.linalg.inv(c2w), dev); fx, fy, cx, cy, W, H = cam_args(cam)
                img = R.render(p, w2c, fx, fy, cx, cy, W, H, sh_degree=deg)[0]
                d = (img - base[name].float()).abs().amax(-1) > args.thr
                changed += int(d.sum()); total += d.numel()
                if not d.any():
                    continue
                ii = torch.nn.functional.pad(d.float().cumsum(0).cumsum(1), (1, 0, 1, 0))   # integral image
                Pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = Pc[:, 2]
                ok = keep & (z > 1e-3)
                u = fx * Pc[:, 0] / z.clamp_min(1e-3) + cx; v = fy * Pc[:, 1] / z.clamp_min(1e-3) + cy
                r = (3 * smax * fx / z.clamp_min(1e-3)).clamp(1, 64)
                u0 = (u - r).floor().clamp(0, W).long(); u1 = (u + r).ceil().clamp(0, W).long()
                v0 = (v - r).floor().clamp(0, H).long(); v1 = (v + r).ceil().clamp(0, H).long()
                ok &= (u1 > u0) & (v1 > v0)
                cnt = ii[v1, u1] - ii[v0, u1] - ii[v1, u0] + ii[v0, u0]
                harm |= ok & (cnt > 0)
        share = changed / max(total, 1)
        hist.append({"iter": it, "changed_pixel_share": share, "fill_kept": int(keep.sum()), "flagged": int(harm.sum())})
        print(f"[fill-harm] iter {it}: changed px {share:.2e}, fill kept {int(keep.sum()):,}, flagged {int(harm.sum()):,}", flush=True)
        if not harm.any():
            break
        keep &= ~harm
    sel = np.concatenate([np.ones(n0, bool), keep.cpu().numpy()])
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(vf.data[sel], "vertex")], text=False).write(args.out)
    json.dump({"fill_in": m, "fill_kept": int(keep.sum()), "iterations": hist, "display_scale": args.display_scale},
              open(os.path.splitext(args.out)[0] + "_harm.json", "w"), indent=2)
    print(f"[fill-harm] kept {int(keep.sum()):,} of {m:,} fill gaussians -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
