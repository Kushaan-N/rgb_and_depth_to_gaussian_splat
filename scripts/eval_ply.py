"""Score an exported .ply the way a standard 3DGS viewer (SuperSplat etc.) shows it (gsplat venv, GPU).

eval_views.py renders the 3DGRUT checkpoint with 3DGRUT's own renderer; what people look at is the
exported PLY in a standard 3DGS rasteriser (EWA splatting, 0.3 px screen-space dilation, SH degree 3).
This renders the PLY with gsplat in that classic mode at the cameras of a COLMAP dataset (the PLY must
be in that dataset's frame) and scores PSNR / SSIM / LPIPS against the real images.

    python scripts/eval_ply.py --ply splat.ply --dataset <dir with sparse/0 + images> [--every 8 --offset 0]
        [--names-from <dataset whose image names to use>] [--tag t] [--json-out r.json] [--save-dir renders/]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def load_ply(path, dev):
    import torch
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    g = lambda ks: torch.tensor(np.stack([np.asarray(v[k], np.float32) for k in ks], 1), device=dev)
    rest = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    dc = g(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None]                                   # (N,1,3)
    sh = torch.cat([dc, g(rest).reshape(len(dc), 3, -1).transpose(1, 2)], 1) if rest else dc   # (N,K,3)
    return dict(means=g(["x", "y", "z"]), quats=g(["rot_0", "rot_1", "rot_2", "rot_3"]),
                scales=torch.exp(g(["scale_0", "scale_1", "scale_2"])),
                opacities=torch.sigmoid(g(["opacity"])[:, 0]), sh=sh, deg=int(round(np.sqrt(sh.shape[1]))) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--every", type=int, default=1, help="score every n-th image (name order)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--names-from", default=None, help="score only image names present in this dataset's model")
    ap.add_argument("--tag", default="ply")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    import torch, pycolmap
    from PIL import Image
    from gsplat import rasterization
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    dev = "cuda"
    G = load_ply(args.ply, dev)
    rec = pycolmap.Reconstruction(os.path.join(args.dataset, "sparse", "0"))
    imgs = sorted(rec.images.values(), key=lambda i: i.name)
    if args.names_from:
        keep = {i.name for i in pycolmap.Reconstruction(os.path.join(args.names_from, "sparse", "0")).images.values()}
        imgs = [i for i in imgs if i.name in keep]
    imgs = imgs[args.offset::args.every]
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev); ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
    P, S, L = [], [], []
    for im in imgs:
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = (cam.params[0], cam.params[1], cam.params[2], cam.params[3]) if cam.model.name == "PINHOLE" \
            else (cam.params[0], cam.params[0], cam.params[1], cam.params[2])
        cfw = im.cam_from_world; M = np.asarray((cfw() if callable(cfw) else cfw).matrix())
        V = np.eye(4); V[:3] = M
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with torch.no_grad():
            rgb, _, _ = rasterization(G["means"], G["quats"], G["scales"], G["opacities"], G["sh"],
                                      torch.tensor(V, dtype=torch.float32, device=dev)[None],
                                      torch.tensor(K, dtype=torch.float32, device=dev)[None], cam.width, cam.height,
                                      sh_degree=G["deg"], eps2d=0.3, rasterize_mode="classic",
                                      backgrounds=torch.zeros(1, 3, device=dev))
        pred = rgb[0].clamp(0, 1)
        gt = torch.tensor(np.asarray(Image.open(os.path.join(args.dataset, "images", im.name)).convert("RGB")),
                          device=dev).float() / 255
        a, b = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        P.append(psnr(a, b).item()); S.append(ssim(a, b).item()); L.append(lpips(a, b).item())
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            Image.fromarray((pred.cpu().numpy() * 255).round().astype(np.uint8)).save(
                os.path.join(args.save_dir, os.path.splitext(im.name)[0] + ".png"))
    res = {"tag": args.tag, "ply": args.ply, "renderer": "gsplat classic 3DGS (eps2d 0.3)", "n": len(P),
           "psnr": round(float(np.mean(P)), 3), "ssim": round(float(np.mean(S)), 4), "lpips": round(float(np.mean(L)), 4)}
    print("EVAL_PLY", json.dumps(res), flush=True)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        json.dump(res, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
