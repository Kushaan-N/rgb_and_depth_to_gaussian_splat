"""Held-out PSNR of PLYs in the viewer-exact rasteriser with depth vs radial sort (gsplat venv, GPU): which stage of a chain
opens a radial-sort gap.   python scripts/object_layers/sort_gap.py --dataset TEST --ply name=path ...
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--ply", nargs="+", required=True); ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    import torch
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t
    from refine_poses_gsplat import load_rgb, psnr
    fr = frames(a.dataset, 0, "all"); out = {}
    gts = {n: load_rgb(os.path.join(a.dataset, "images", n), "cuda").float() / 255 for n, _, _ in fr}
    for spec in a.ply:
        nm, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[sort-gap] {nm}: missing {path}"); continue
        _, p = load_ply(path, "cuda"); deg = sh_deg(p); r = out[nm] = {}
        for mode in ("depth", "radial"):
            Rv = ViewerRaster("ss352", "exact", display_scale=1.0, sort=mode); v = []
            with torch.no_grad():
                for n, cam, c2w in fr:
                    v.append(float(psnr(Rv.render(p, to_t(np.linalg.inv(c2w), "cuda"), *cam_args(cam, "ss352"), sh_degree=deg)[0], gts[n])))
            r[mode] = round(float(np.mean(v)), 3)
        r["gap"] = round(r["depth"] - r["radial"], 3); print(f"[sort-gap] {nm}: {json.dumps(r)}", flush=True)
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
