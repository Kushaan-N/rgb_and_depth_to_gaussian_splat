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
import argparse, os, sys, tempfile
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim3_utils import get_sim3
from eval_views import score_dataset


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

    import json, yaml
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
    if args.sim3_json or args.colmap_model:   # sim3.json preferred (no pycolmap needed in the trainer venv)
        s, R, t = get_sim3(args.sim3_json, args.colmap_model, args.gt_model)
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

    m = score_dataset(args.checkpoint, ds)
    print(f"[eval-gt] {args.tag}: N={m['n']}  PSNR={m['psnr']:.3f}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}", flush=True)
    print(f"EVAL_GT_RESULT tag={args.tag} psnr={m['psnr']:.3f} ssim={m['ssim']:.4f} lpips={m['lpips']:.4f} n={m['n']}")


if __name__ == "__main__":
    main()
