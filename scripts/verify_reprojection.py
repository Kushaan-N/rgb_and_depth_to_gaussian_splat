"""Gate 2 + Gate 2b — the reprojection check (docs/PLAN.md §5.1-5.2). MANDATORY.

Gate 2  : warp frame i into frame i+k using frame i's depth + the computed relative pose,
          and compare to the actual frame i+k. Edges landing on edges => the transform
          chain (TRAPs 1-4) is right. A directional offset => wrong transform direction;
          a rotational smear => quaternion-order error.
Gate 2b : reload the WRITTEN COLMAP model and re-run a subset of the warps FROM IT — the
          only thing that catches TRAP 1b (a correct pipeline + a wrong writer). Also
          plots the trajectory extent / viewing-direction spread (§2.6).

    python scripts/verify_reprojection.py --config configs/mocap1_well-lit_trot.yaml

Do not train anything until this passes.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
from build_poses import compute_world_cam, depth_poses


def warp(depth_i_m, valid_i, rgb_i, Twc_i, Twc_j, K):
    """Forward-warp frame i into frame j's view using i's depth. Returns (warped, mask)."""
    H, W = depth_i_m.shape
    pts_c, cols, _ = iu.backproject(depth_i_m, K, valid_i, rgb_i)
    pts_w = iu.transform_points(Twc_i, pts_c)
    pts_cj = iu.transform_points(pu.invert_T(Twc_j), pts_w)
    uv, z = iu.project(pts_cj, K)
    ui = np.round(uv[:, 0]).astype(np.int64)
    vi = np.round(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z, cols = ui[ok], vi[ok], z[ok], cols[ok]
    # z-buffer: write far-to-near so the nearest surface wins per target pixel
    order = np.argsort(-z)
    warped = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    warped[vi[order], ui[order]] = cols[order]
    mask[vi[order], ui[order]] = True
    return warped, mask


def photo_error(warped, mask, rgb_j):
    """Mean absolute grayscale error over warped pixels (0-255)."""
    if mask.sum() == 0:
        return float("nan")
    g_w = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_j = cv2.cvtColor(rgb_j, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(g_w[mask] - g_j[mask]).mean())


def _save_triptych(path, rgb_j, warped, mask):
    diff = np.zeros_like(rgb_j)
    diff[mask] = np.abs(rgb_j[mask].astype(np.int16) - warped[mask].astype(np.int16)).astype(np.uint8)
    strip = np.concatenate([rgb_j, warped, diff], axis=1)
    cv2.imwrite(path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def _pairs(n, k, n_pairs):
    starts = np.unique(np.linspace(0, n - k - 1, n_pairs).astype(int))
    return [(int(i), int(i + k)) for i in starts if 0 <= i and i + k < n]


def run_gate2(cfg, kept, Twc, calib, k, n_pairs, out_dir, tag, save=True, Twc_src=None):
    # Twc_src poses the SOURCE depth (TRAP 7: at the depth timestamp); Twc poses the
    # target RGB view. Defaults to Twc when depth poses aren't supplied.
    if Twc_src is None:
        Twc_src = Twc
    root = cfg["sequence"]["data_root"]
    depth_dir = os.path.join(root, cfg["sequence"]["depth_dir"])
    rgb_dir = os.path.join(root, cfg["sequence"]["rgb_dir"])
    K = calib.K
    errs = []
    os.makedirs(out_dir, exist_ok=True)
    for n, (i, j) in enumerate(_pairs(len(kept), k, n_pairs)):
        depth_i = iu.read_depth_raw(os.path.join(depth_dir, kept[i].depth_name))
        di_m, vi = iu.depth_to_meters(depth_i, cfg)
        rgb_i = iu.read_rgb(os.path.join(rgb_dir, kept[i].rgb_name))
        rgb_j = iu.read_rgb(os.path.join(rgb_dir, kept[j].rgb_name))
        warped, mask = warp(di_m, vi, rgb_i, Twc_src[i], Twc[j], K)
        e = photo_error(warped, mask, rgb_j)
        errs.append(e)
        if save and n < 6:
            _save_triptych(os.path.join(out_dir, f"{tag}_pair_{i:04d}_{j:04d}.png"),
                           rgb_j, warped, mask)
    return np.array(errs, dtype=np.float64)


def plot_trajectory(Twc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = Twc[:, :3, 3]
    fwd = Twc[:, :3, 2]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(C[:, 0], C[:, 1], C[:, 2], "-o", ms=2)
    ax.quiver(C[:, 0], C[:, 1], C[:, 2], fwd[:, 0], fwd[:, 1], fwd[:, 2],
              length=0.2, color="r", normalize=True)
    ax.set_title("camera path + forward dirs"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(fwd[:, 0], fwd[:, 1], s=8)
    ax2.set_title("view-dir (x,y) spread"); ax2.set_aspect("equal"); ax2.grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=10, help="frame gap for the warp")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="max mean grayscale error (0-255) to pass Gate 2")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    out_dir = os.path.join(cfg["paths"]["out_root"], "gate2")

    kept, Twc, calib, interp, mocap, _ = compute_world_cam(cfg)
    Twc_depth = depth_poses(cfg, kept, mocap, interp, calib)   # TRAP 7: source depth poses
    k = min(args.k, max(1, len(kept) // 4))

    # --- Gate 2 (in-memory poses) ---
    errs = run_gate2(cfg, kept, Twc, calib, k, args.pairs, out_dir, "gate2", Twc_src=Twc_depth)
    m = float(np.nanmean(errs))
    print(f"[Gate 2] {len(errs)} pairs (k={k})  mean grayscale err={m:.3f} "
          f"(min {np.nanmin(errs):.2f}, max {np.nanmax(errs):.2f})")
    gate2_pass = m < args.threshold

    # --- Gate 2b (reload written COLMAP model) ---
    model_dir = os.path.join(cfg["paths"]["out_root"], "colmap", "sparse", "0")
    if not os.path.exists(os.path.join(model_dir, "images.txt")):
        print("[Gate 2b] SKIP — run build_poses.py first"); gate2b_pass = False
    else:
        _cams, imgs, _pts = pu.read_colmap_model(model_dir)
        by_name = {im.name: im.T_world_cam for im in imgs}
        Twc_loaded = np.stack([by_name[f.rgb_name] for f in kept])
        errs_b = run_gate2(cfg, kept, Twc_loaded, calib, k, min(args.pairs, 8),
                           out_dir, "gate2b", save=True, Twc_src=Twc_depth)
        mb = float(np.nanmean(errs_b))
        drift = float(np.max(np.abs(Twc - Twc_loaded)))
        print(f"[Gate 2b] reloaded-model mean err={mb:.3f}  |Twc-Twc_loaded|max={drift:.2e}")
        gate2b_pass = (abs(mb - m) < 1.0) and (drift < 1e-5)

    traj_png = os.path.join(out_dir, "trajectory.png")
    plot_trajectory(Twc, traj_png)
    print(f"[Gate 2] triptychs + trajectory in {out_dir}")

    ok = gate2_pass and gate2b_pass
    print(f"GATE 2 {'PASS' if gate2_pass else 'FAIL'} / "
          f"GATE 2b {'PASS' if gate2b_pass else 'FAIL'} "
          f"(threshold {args.threshold})")
    if not gate2_pass:
        print("  -> directional offset = wrong transform direction/frame (TRAP 4); "
              "rotational smear = quaternion order (TRAP 1). STOP, do not train.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
