"""Phase 1 — acquisition inventory + Gate 1 (docs/PLAN.md §4).

Prints one table describing exactly what is on disk and whether the streams are
self-consistent, and writes it to {out_root}/inventory.json. Every VERIFY item from the
plan that can be read off the files is resolved here.

    python scripts/inventory.py --config configs/mocap1_well-lit_trot.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu


def _rate(t: np.ndarray):
    if len(t) < 2:
        return float("nan")
    span = t[-1] - t[0]
    return (len(t) - 1) / span if span > 0 else float("nan")


def _count_images(d: str):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def inventory(cfg: dict) -> dict:
    seq = cfg["sequence"]
    root = seq["data_root"]
    report: dict = {"data_root": root, "dirs": {}, "streams": {}, "intrinsics": {},
                    "verify": {}, "issues": []}

    # --- image folders ---
    dir_keys = {"rgb": seq.get("rgb_dir"), "raw_rgb": "raw_rgb",
                "depth": seq.get("depth_dir"), "raw_depth": seq.get("raw_depth_dir")}
    for label, sub in dir_keys.items():
        if not sub:
            continue
        path = os.path.join(root, sub)
        if not os.path.isdir(path):
            report["dirs"][label] = {"path": path, "exists": False}
            continue
        files = _count_images(path)
        entry = {"path": path, "exists": True, "count": len(files)}
        if files:
            if "depth" in label:
                d = iu.read_depth_raw(files[0])
                nz = d[d != cfg["depth"].get("invalid_value", 0)]
                entry.update(dtype=str(d.dtype), shape=list(d.shape),
                             min_nonzero_m=float(nz.min() / cfg["depth"]["units_per_meter"]) if nz.size else None,
                             max_m=float(d.max() / cfg["depth"]["units_per_meter"]),
                             pct_invalid=float(100.0 * (d == cfg["depth"].get("invalid_value", 0)).mean()))
            else:
                img = iu.read_rgb(files[0])
                entry.update(dtype=str(img.dtype), shape=list(img.shape))
        report["dirs"][label] = entry

    # --- streams: timestamps -> rate ---
    def _safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            report["issues"].append(f"{fn.__name__}: {e}")
            return None

    frames = _safe(pu.parse_realsense_timestamps,
                   os.path.join(root, seq["timestamp_table"]), cfg)
    if frames:
        t = np.array([f.t for f in frames])
        report["streams"]["rgb_frames"] = {"n": len(frames), "rate_hz": _rate(t),
                                           "t0": float(t[0]), "t1": float(t[-1])}
    imu = _safe(pu.parse_vectornav, os.path.join(root, seq["imu_file"]), cfg)
    if imu:
        report["streams"]["imu"] = {"n": len(imu.t), "rate_hz": _rate(imu.t)}
    joints = _safe(pu.parse_joints, os.path.join(root, seq["joint_file"]), cfg)
    if joints:
        report["streams"]["joints"] = {"n": len(joints[0]), "rate_hz": _rate(joints[0])}

    pose_path = os.path.join(root, seq["pose_file"])
    report["verify"]["pose_file"] = seq["pose_file"] if os.path.exists(pose_path) else "MISSING"
    mocap = _safe(pu.parse_mocap, pose_path, cfg) if os.path.exists(pose_path) else None
    if mocap:
        report["streams"]["pose"] = {"n": len(mocap.t), "rate_hz": _rate(mocap.t),
                                     "t0": float(mocap.t[0]), "t1": float(mocap.t[-1])}
        report["verify"]["pose_frame_assumed"] = mocap.frame
        report["verify"]["world_up_assumed"] = mocap.world_up

    if frames:
        report["duration_s"] = float(t[-1] - t[0])

    # --- intrinsics ---
    calib_path = cfg["intrinsics"].get("calib_file", "")
    if os.path.exists(calib_path):
        calib = pu.load_calibration(calib_path, cfg)
        fx, fy, cx, cy = iu.K_params(calib.K)
        report["intrinsics"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                "width": calib.width, "height": calib.height,
                                "dist_model": calib.dist_model,
                                "dist": calib.dist.tolist()}
    else:
        report["issues"].append(f"calibration file not found: {calib_path}")

    # --- VERIFY items the files usually cannot answer (surface them explicitly) ---
    report["verify"]["rgb_shutter"] = "rolling (Intel D455 RGB spec); paper is silent (§2.4)"
    report["verify"]["auto_exposure"] = "ON — paper-confirmed (§7.2); expect exposure drift"
    report["verify"]["auto_white_balance"] = "UNKNOWN — check §7.2 intensity/color drift plot"
    report["verify"]["depth_units_per_meter"] = cfg["depth"]["units_per_meter"]
    report["verify"]["depth_invalid_value"] = cfg["depth"].get("invalid_value", 0)

    # --- sync diagnostics (TRAP 3 drift + TRAP 8 preamble) ---
    try:
        import sync_utils as su
        preamble = su.detect_sync_preamble(cfg)
        report["sync"] = {
            "preamble_end_s": preamble,
            "preamble_detected": preamble is not None,
            "clock_drift": su.clock_drift_check(cfg),
        }
    except Exception as e:  # noqa: BLE001
        report["issues"].append(f"sync diagnostics: {e}")

    # --- Gate 1 consistency ---
    counts = {k: v.get("count") for k, v in report["dirs"].items()
              if v.get("exists") and "count" in v}
    n_ts = report["streams"].get("rgb_frames", {}).get("n")
    consistent = True
    if "rgb" in counts and "depth" in counts and counts["rgb"] != counts["depth"]:
        consistent = False
        report["issues"].append(f"rgb count {counts['rgb']} != depth count {counts['depth']}")
    if n_ts is not None and "rgb" in counts and abs(n_ts - counts["rgb"]) > 1:
        consistent = False
        report["issues"].append(f"timestamp rows {n_ts} != rgb count {counts['rgb']}")
    report["gate1_consistent"] = consistent
    return report


def print_table(r: dict) -> None:
    line = "=" * 68
    print(line); print("GATE 1 — INVENTORY:", r["data_root"]); print(line)
    print("\n[image folders]")
    for k, v in r["dirs"].items():
        if not v.get("exists"):
            print(f"  {k:10s} MISSING ({v['path']})"); continue
        extra = ""
        if "min_nonzero_m" in v:
            extra = f" depth {v['dtype']} range≈[{v.get('min_nonzero_m')},{v.get('max_m')}]m invalid={v.get('pct_invalid'):.1f}%"
        elif "dtype" in v:
            extra = f" {v['dtype']} {v.get('shape')}"
        print(f"  {k:10s} n={v['count']:<6d}{extra}")
    print("\n[streams]  (times on event clock, seconds)")
    for k, v in r["streams"].items():
        print(f"  {k:12s} n={v['n']:<7d} rate≈{v['rate_hz']:.2f} Hz")
    if "duration_s" in r:
        print(f"\n  duration ≈ {r['duration_s']:.2f} s")
    print("\n[intrinsics]")
    if r["intrinsics"]:
        i = r["intrinsics"]
        print(f"  fx={i['fx']:.2f} fy={i['fy']:.2f} cx={i['cx']:.2f} cy={i['cy']:.2f} "
              f"{i['width']}x{i['height']}  dist_model={i['dist_model']} dist={i['dist']}")
    else:
        print("  (none — calibration missing)")
    print("\n[VERIFY]")
    for k, v in r["verify"].items():
        print(f"  {k:24s}: {v}")
    if "sync" in r:
        s = r["sync"]; d = s.get("clock_drift", {})
        pre = f"{s['preamble_end_s']:.2f} s" if s.get("preamble_detected") else "none detected"
        drift = d.get("drift_ms")
        print("\n[sync]  (TRAP 3 / TRAP 8)")
        print(f"  preamble_end            : {pre}")
        print(f"  clock_drift_ms          : "
              f"{drift:.2f}" if drift is not None else "  clock_drift_ms          : n/a")
        if d.get("exceeds_tolerance"):
            print(f"  WARNING: drift exceeds {d.get('tolerance_ms')} ms — apply linear correction")
    print("\n[issues]")
    for m in r["issues"] or ["(none)"]:
        print(f"  - {m}")
    print(f"\nGATE 1 {'PASS' if r['gate1_consistent'] and not r['issues'] else 'REVIEW'}: "
          f"frame counts {'self-consistent' if r['gate1_consistent'] else 'INCONSISTENT'}")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    r = inventory(cfg)
    print_table(r)
    out = os.path.join(cfg["paths"]["out_root"], "inventory.json")
    pu.save_json(out, r)
    print(f"[inventory] wrote {out}")
    return 0 if r["gate1_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
