"""Gate 4 — reconstruction quality on held-out + off-trajectory views (docs/PLAN.md §7.5).

Three responsibilities, split so the CPU parts are testable without a GPU:

  offtraj-poses  (CPU) : generate off-trajectory camera poses from the held-out val set
                         (lateral +0.5 m, height 0.8 m) and write a COLMAP model the
                         renderer can consume. Quantifying how much worse these look
                         predicts how well the scene serves as a *drivable* sim.
  metrics        (CPU) : PSNR / SSIM / LPIPS between a ground-truth dir and a renders dir.
  render         (GPU) : invoke the trained model's renderer (3dgrut) for given poses.

    python scripts/eval_novel_views.py offtraj-poses --config ...
    python scripts/eval_novel_views.py metrics --gt-dir GT --render-dir R
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu


# --------------------------------------------------------------------------- #
# metrics (CPU)
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global single-scale SSIM on grayscale (no skimage dependency)."""
    import cv2
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mx * my + c1) * (2 * cov + c2)) /
                 ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def lpips_score(a, b):
    """LPIPS if the (torch-based) package is available, else None."""
    try:
        import torch
        import lpips as lpips_lib
    except Exception:
        return None
    net = lpips_lib.LPIPS(net="alex")

    def t(im):
        x = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
        return x.permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(net(t(a), t(b)).item())


def run_metrics(gt_dir: str, render_dir: str) -> dict:
    names = sorted(f for f in os.listdir(render_dir) if f.lower().endswith((".png", ".jpg")))
    ps, ss, lp = [], [], []
    for n in names:
        g = os.path.join(gt_dir, n)
        if not os.path.exists(g):
            continue
        a = iu.read_rgb(g); b = iu.read_rgb(os.path.join(render_dir, n))
        if a.shape != b.shape:
            import cv2
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        ps.append(psnr(a, b)); ss.append(ssim(a, b))
        l = lpips_score(a, b)
        if l is not None:
            lp.append(l)
    return {"n": len(ps),
            "PSNR": float(np.mean(ps)) if ps else float("nan"),
            "SSIM": float(np.mean(ss)) if ss else float("nan"),
            "LPIPS": float(np.mean(lp)) if lp else None}


# --------------------------------------------------------------------------- #
# off-trajectory poses (CPU) — §7.5
# --------------------------------------------------------------------------- #
def offtrajectory_poses(cfg: dict, lateral_m=0.5, height_m=0.8) -> str:
    """Shift each held-out pose sideways (camera +x) and set a higher world z, then write a
    COLMAP model for the renderer. Returns the model dir."""
    out_root = cfg["paths"]["out_root"]
    val = json.load(open(os.path.join(out_root, "gating", "val_frames.json")))
    imgs = []
    for k, fr in enumerate(val):
        T = np.array(fr["T_world_cam"])
        T2 = T.copy()
        T2[:3, 3] = T[:3, 3] + T[:3, 0] * lateral_m      # shift along camera right axis
        T2[2, 3] = height_m                              # raise to a non-capture height (Z-up)
        imgs.append(pu.ColmapImage(id=k + 1, T_world_cam=T2, camera_id=1,
                                   name=f"offtraj_{k:04d}.png"))
    meta = pu.load_config  # noqa (keep import graph obvious)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    fx, fy, cx, cy = iu.K_params(calib.K)
    cam = pu.ColmapCamera(id=1, model="PINHOLE", width=calib.width, height=calib.height,
                          params=[fx, fy, cx, cy])
    model_dir = os.path.join(out_root, "eval_offtraj", "sparse", "0")
    pu.write_colmap_model(model_dir, cam, imgs)
    return model_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("offtraj-poses"); p1.add_argument("--config", required=True)
    p1.add_argument("--lateral", type=float, default=0.5)
    p1.add_argument("--height", type=float, default=0.8)
    p2 = sub.add_parser("metrics")
    p2.add_argument("--gt-dir", required=True); p2.add_argument("--render-dir", required=True)
    p2.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "offtraj-poses":
        cfg = pu.load_config(args.config)
        d = offtrajectory_poses(cfg, args.lateral, args.height)
        print(f"[eval] off-trajectory COLMAP model -> {d}")
        print("[eval] render these with the trained model (GPU), then run `metrics` vs a "
              "blank/no GT (these are novel views; report qualitatively + vs held-out PSNR).")
    elif args.cmd == "metrics":
        m = run_metrics(args.gt_dir, args.render_dir)
        print(f"[Gate 4] n={m['n']}  PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}  "
              f"LPIPS={m['LPIPS'] if m['LPIPS'] is not None else 'n/a (install lpips)'}")
        if args.out:
            pu.save_json(args.out, m)


if __name__ == "__main__":
    main()
