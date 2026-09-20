"""Phase 4 (part 1) — photometric preconditioning (docs/PLAN.md §7.2).

  * Undistort-first: undistort every frame with the released coefficients and keep a plain
    PINHOLE camera, so the trainer never has to honor a distortion model (§7.2). For a
    zero-distortion calibration this is a straight copy.
  * AE/AWB check: plot mean image intensity over time. Drift/steps => auto-exposure or
    auto-white-balance was on during capture (a mild `Blink` problem) -> use a trainer with
    per-image exposure compensation / appearance embeddings (§3.3 swap path).

Runs on BOTH rgb/ and raw_rgb/ if present (the TRAP-5 A/B), writing to
{out_root}/precond/<dir>/.

    python scripts/precondition_frames.py --config configs/mocap1_well-lit_trot.yaml
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


def ae_awb_report(mean_intensity: np.ndarray, times: np.ndarray, out_png: str) -> dict:
    """Quantify brightness drift over the trajectory and plot it."""
    rng = float(mean_intensity.max() - mean_intensity.min())
    std = float(mean_intensity.std())
    # linear slope of intensity vs time (per second)
    if len(times) > 1 and (times[-1] - times[0]) > 0:
        slope = float(np.polyfit(times - times[0], mean_intensity, 1)[0])
    else:
        slope = float("nan")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(times - times[0], mean_intensity, "-o", ms=3)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean intensity (0-255)")
    ax.set_title(f"AE/AWB check — range {rng:.1f}, std {std:.1f}, slope {slope:.2f}/s")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    likely_ae = rng > 15.0 or abs(slope) > 5.0
    return {"intensity_range": rng, "intensity_std": std, "slope_per_s": slope,
            "likely_auto_exposure": bool(likely_ae)}


def precondition(cfg: dict) -> dict:
    seq = cfg["sequence"]
    root = seq["data_root"]
    out_root = cfg["paths"]["out_root"]
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    frames = pu.parse_realsense_timestamps(os.path.join(root, seq["timestamp_table"]), cfg)

    candidates = [seq.get("rgb_dir")]
    if os.path.isdir(os.path.join(root, "raw_rgb")):
        candidates.append("raw_rgb")

    result = {"undistorted": {}, "distortion_is_zero": bool(np.allclose(calib.dist, 0))}
    for sub in candidates:
        if not sub:
            continue
        src = os.path.join(root, sub)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(out_root, "precond", sub)
        os.makedirs(dst, exist_ok=True)
        means, times = [], []
        n = 0
        for fr in frames:
            p = os.path.join(src, fr.rgb_name)
            if not os.path.exists(p):
                continue
            img = iu.read_rgb(p)
            und = iu.undistort_rgb(img, calib.K, calib.dist)
            cv2.imwrite(os.path.join(dst, fr.rgb_name), cv2.cvtColor(und, cv2.COLOR_RGB2BGR))
            means.append(float(cv2.cvtColor(und, cv2.COLOR_RGB2GRAY).mean()))
            times.append(fr.t)
            n += 1
        ae = ae_awb_report(np.array(means), np.array(times),
                           os.path.join(out_root, "precond", f"ae_check_{sub}.png"))
        result["undistorted"][sub] = {"n": n, "out_dir": dst, "ae": ae}
    pu.save_json(os.path.join(out_root, "precond", "precondition_meta.json"), result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    r = precondition(cfg)
    print(f"[precondition] distortion is zero: {r['distortion_is_zero']} "
          f"(undistortion is a straight copy if so)")
    for sub, info in r["undistorted"].items():
        ae = info["ae"]
        print(f"[precondition] {sub}: undistorted {info['n']} frames -> {info['out_dir']}")
        print(f"               AE/AWB: intensity range {ae['intensity_range']:.1f}, "
              f"slope {ae['slope_per_s']:.2f}/s -> "
              f"{'LIKELY auto-exposure (use appearance embeddings)' if ae['likely_auto_exposure'] else 'looks fixed'}")


if __name__ == "__main__":
    main()
