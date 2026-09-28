"""Ground-truth-pose evaluation: render a trained splat from the SAME GT held-out poses and score
against the SAME GT images (3DGRUT venv, GPU).

The per-arm metrics.json evaluate each model at its OWN poses, which slightly favors COLMAP (its
poses are photometrically self-consistent). This is the pose-fair benchmark: every model is asked
"what does the TRUE camera see?" on a common held-out set (the pipeline's 53 val frames, never
trained on). COLMAP-frame models are rendered from the GT pose mapped into their frame via the
Sim3 (Umeyama) solved from COLMAP<->GT camera centres, with the TRUE intrinsics for everyone.

    python eval_splat_gt.py --checkpoint ckpt.pt --config configs/<seq>.yaml \
        [--colmap-model <ds>/sparse/0 --gt-model <out>/colmap_train/sparse/0]   # omit for a GT-frame model
"""
from __future__ import annotations
import argparse, os, tempfile
import numpy as np


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd((D.T @ S) / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = np.trace(np.diag(d) @ W) / ((S ** 2).sum() / len(src))
    return float(s), R, mu_d - s * R @ mu_s


def sim3_from_models(colmap_model, gt_model):
    import pycolmap
    C = {i.name: np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(colmap_model).images.values()}
    G = {i.name: np.asarray(i.projection_center()) for i in pycolmap.Reconstruction(gt_model).images.values()}
    common = sorted(set(C) & set(G))
    return umeyama(np.array([C[n] for n in common]), np.array([G[n] for n in common]))  # colmap->gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-json", default=None, help="gating/val_frames.json (default from config out_root)")
    ap.add_argument("--colmap-model", default=None, help="if set, map GT poses into this model's frame via Sim3")
    ap.add_argument("--gt-model", default=None, help="GT COLMAP model for the Sim3 reference (colmap_train/sparse/0)")
    ap.add_argument("--sim3-json", default=None, help="Sim3 from sim3_align_splat.py --sim3-json (preferred)")
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    import json, yaml, torch
    from scipy.spatial.transform import Rotation
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pose_utils as pu

    cfg = pu.load_config(args.config)
    out_root = cfg["paths"]["out_root"]
    precond = os.path.join(out_root, "precond", cfg["sequence"].get("rgb_dir", "rgb"))
    cal = yaml.safe_load(open(cfg["intrinsics"]["calib_file"]))
    K = np.array(cal["K"], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W = int(cal.get("width") or cfg["intrinsics"]["image_width"])
    H = int(cal.get("height") or cfg["intrinsics"]["image_height"])
    val_json = args.val_json or os.path.join(out_root, "gating", "val_frames.json")
    val = json.load(open(val_json))
    print(f"[eval-gt] {len(val)} held-out frames; tag={args.tag}", flush=True)

    s, R, t = 1.0, np.eye(3), np.zeros(3)
    if args.sim3_json:                      # preferred: solved once by sim3_align_splat.py (no pycolmap needed here)
        j = json.load(open(args.sim3_json))
        s, R, t = float(j["scale"]), np.array(j["R"]), np.array(j["t"])
    elif args.colmap_model:
        s, R, t = sim3_from_models(args.colmap_model, args.gt_model)
    if args.sim3_json or args.colmap_model:
        print(f"[eval-gt] Sim3 colmap->gt scale={s:.4f}", flush=True)

    def gt_to_model_wc(Twc_gt):
        Rcw = np.array(Twc_gt)[:3, :3]; C = np.array(Twc_gt)[:3, 3]
        Rcw_m = R.T @ Rcw
        C_m = (1.0 / s) * R.T @ (C - t)
        Rwc = Rcw_m.T
        return Rwc, -Rwc @ C_m

    # write a COLMAP eval dataset (true K, GT poses mapped to model frame, val images)
    ds = tempfile.mkdtemp(prefix="evalgt_")
    os.makedirs(os.path.join(ds, "sparse", "0"), exist_ok=True)
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    with open(os.path.join(ds, "sparse", "0", "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    with open(os.path.join(ds, "sparse", "0", "images.txt"), "w") as f:
        for i, fr in enumerate(val, 1):
            Rwc, twc = gt_to_model_wc(fr["T_world_cam"])
            q = Rotation.from_matrix(Rwc).as_quat()  # x,y,z,w
            name = os.path.basename(fr["name"])
            f.write(f"{i} {q[3]} {q[0]} {q[1]} {q[2]} {twc[0]} {twc[1]} {twc[2]} 1 {name}\n\n")
            src = os.path.join(precond, name); dst = os.path.join(ds, "images", name)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    open(os.path.join(ds, "sparse", "0", "points3D.txt"), "w").close()

    # load model + render all frames (test_split_interval<=0 -> all frames used), score vs GT image
    from threedgrut.render import Renderer
    import threedgrut.datasets as datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    r = Renderer.from_checkpoint(checkpoint_path=args.checkpoint, path=ds, out_dir=ds,
                                 save_gt=False, computes_extra_metrics=False)
    model = r.model
    conf = r.conf
    conf.dataset.test_split_interval = 0            # use every frame
    dataset, _ = datasets.make(conf.dataset.type, conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(dataset, **configure_dataloader_for_platform(
        {"num_workers": 4, "batch_size": 1, "shuffle": False, "collate_fn": None}))
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda()
    P, S, L = [], [], []
    for batch in loader:
        gb = dataset.get_gpu_batch_with_intrinsics(batch)
        with torch.no_grad():
            out = model(gb)
        pred = out["pred_features"][..., :3].clip(0, 1)      # (1,H,W,3)
        gt = gb.rgb_gt[..., :3]
        P.append(psnr_m(pred, gt).item())
        pchw = pred.permute(0, 3, 1, 2); gchw = gt.permute(0, 3, 1, 2)
        S.append(ssim_m(pchw, gchw).item())
        L.append(lpips_m(pchw, gchw).item())
    print(f"[eval-gt] {args.tag}: N={len(P)}  PSNR={np.mean(P):.3f}  SSIM={np.mean(S):.4f}  LPIPS={np.mean(L):.4f}", flush=True)
    print(f"EVAL_GT_RESULT tag={args.tag} psnr={np.mean(P):.3f} ssim={np.mean(S):.4f} lpips={np.mean(L):.4f} n={len(P)}")


if __name__ == "__main__":
    main()
