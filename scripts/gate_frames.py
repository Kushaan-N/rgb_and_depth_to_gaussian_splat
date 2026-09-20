"""Phase 4 (part 2) — blur/motion gating + hold-out split (docs/PLAN.md §7.1, §7.4).

Selects the RGB frames worth training on:
  * low angular velocity ‖ω‖ (rolling-shutter + motion-blur mitigation; keep sharpest ~%),
  * high variance-of-Laplacian (sharpness),
  * adequate spatial spread (no 500 near-identical viewpoints).
Then holds out every Nth GATED frame (split AFTER gating, so held-out frames are also
sharp — §7.4) and writes a COLMAP model containing only the TRAIN frames, ready for the
trainer, plus a val_frames.json for Gate 4.

    python scripts/gate_frames.py --config configs/mocap1_well-lit_trot.yaml

Expect to discard a large fraction — that is correct (3DGS wants coverage/parallax, not
frame count).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
import gating_utils as gu
from build_poses import compute_world_cam


def _laplacian_per_frame(cfg, frames):
    """Sharpness per frame; read from preconditioned images if available, else raw."""
    root = cfg["sequence"]["data_root"]
    precond = os.path.join(cfg["paths"]["out_root"], "precond", cfg["sequence"]["rgb_dir"])
    raw = os.path.join(root, cfg["sequence"]["rgb_dir"])
    src = precond if os.path.isdir(precond) else raw
    return np.array([iu.laplacian_var(iu.read_rgb(os.path.join(src, f.rgb_name)))
                     for f in frames])


def _omega_histogram(omega, thr, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(omega, bins=30)
    ax.axvline(thr, color="r", ls="--", label=f"keep ≤ {thr:.2f} rad/s")
    ax.set_xlabel("‖ω‖ (rad/s)"); ax.set_ylabel("frames"); ax.legend()
    ax.set_title("angular velocity gate")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def build(cfg: dict) -> dict:
    out_root = cfg["paths"]["out_root"]
    gate_dir = os.path.join(out_root, "gating")
    os.makedirs(gate_dir, exist_ok=True)

    kept, Twc, calib, *_ = compute_world_cam(cfg)
    ts = [f.t for f in kept]
    lap = _laplacian_per_frame(cfg, kept)
    g = cfg["gating"]
    keep, info = gu.motion_gate(cfg, ts, Twc[:, :3, 3],
                                omega_percentile_keep=float(g["omega_percentile_keep"]),
                                min_sep=float(g["spatial_min_sep_m"]),
                                laplacian=lap, laplacian_min=float(g["laplacian_min"]))
    keep, n_preamble = gu.apply_preamble_exclusion(cfg, ts, keep)   # TRAP 8: drop the ball
    gated = list(np.where(keep)[0])
    if len(gated) < 4:
        raise RuntimeError(f"only {len(gated)} frames survived gating — loosen thresholds "
                           "(gating.omega_percentile_keep / laplacian_min / spatial_min_sep_m)")

    # --- hold-out split: every Nth GATED frame (§7.4) ---
    hold = int(g["holdout_every"])
    val = gated[::hold]
    train = [i for i in gated if i not in set(val)]

    _omega_histogram(info["omega"], info["omega_threshold"],
                     os.path.join(gate_dir, "omega_hist.png"))

    # --- write a COLMAP model with TRAIN frames only (reuse seeded points3D) ---
    full_model = os.path.join(out_root, "colmap", "sparse", "0")
    train_model = os.path.join(out_root, "colmap_train", "sparse", "0")
    os.makedirs(train_model, exist_ok=True)
    for fn in ("cameras.txt", "points3D.txt"):
        srcf = os.path.join(full_model, fn)
        if os.path.exists(srcf):
            shutil.copy(srcf, os.path.join(train_model, fn))
    train_imgs = [pu.ColmapImage(id=k + 1, T_world_cam=Twc[i], camera_id=1,
                                 name=kept[i].rgb_name) for k, i in enumerate(train)]
    pu.write_images_txt(os.path.join(train_model, "images.txt"), train_imgs)

    val_frames = [{"name": kept[i].rgb_name, "t_event_s": kept[i].t,
                   "T_world_cam": Twc[i].tolist()} for i in val]
    pu.save_json(os.path.join(gate_dir, "val_frames.json"), val_frames)

    with open(os.path.join(gate_dir, "train_frames.txt"), "w") as f:
        f.write("\n".join(kept[i].rgb_name for i in train) + "\n")
    with open(os.path.join(gate_dir, "val_frames.txt"), "w") as f:
        f.write("\n".join(kept[i].rgb_name for i in val) + "\n")

    meta = {
        "n_frames_in": len(kept),
        "n_gated": len(gated),
        "n_excluded_preamble": int(n_preamble),
        "n_train": len(train), "n_val": len(val),
        "omega_threshold_rad_s": info["omega_threshold"],
        "laplacian_min": float(g["laplacian_min"]),
        "laplacian_stats": {"min": float(lap.min()), "median": float(np.median(lap)),
                            "max": float(lap.max())},
        "holdout_every": hold,
        "train_model_dir": train_model,
        "discard_fraction": 1.0 - len(gated) / len(kept),
    }
    pu.save_json(os.path.join(gate_dir, "gating_meta.json"), meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    m = build(cfg)
    print(f"[gate_frames] kept {m['n_gated']}/{m['n_frames_in']} frames "
          f"(discarded {m['discard_fraction']*100:.0f}% — expected & healthy)")
    print(f"[gate_frames] ‖ω‖ threshold {m['omega_threshold_rad_s']:.3f} rad/s, "
          f"laplacian_min {m['laplacian_min']} (median {m['laplacian_stats']['median']:.1f})")
    print(f"[gate_frames] train {m['n_train']} / val {m['n_val']} (hold every {m['holdout_every']}th)")
    print(f"[gate_frames] train COLMAP model: {m['train_model_dir']}")


if __name__ == "__main__":
    main()
