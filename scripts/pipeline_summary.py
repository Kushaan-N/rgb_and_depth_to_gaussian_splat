"""Collect one pipeline run into <out_root>/pipeline/summary.json and print a readable summary.

    python scripts/pipeline_summary.py --config configs/<seq>.yaml
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu


def _json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _ply_count(path):
    """Gaussian count from the .ply header without loading the body."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    out = cfg["paths"]["out_root"]; P = os.path.join(out, "pipeline")      # shared prep outputs
    pl = cfg.get("pipeline") or {}
    R = pl.get("run_dir") or P                                              # this run (variant) outputs

    sfm = None
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(os.path.join(P, "dataset", "sparse", "0"))
        n_img = len(os.listdir(os.path.join(P, "dataset", "images")))
        sfm = {"registered": rec.num_reg_images(), "frames": n_img, "points": rec.num_points3D(),
               "camera": [round(float(x), 2) for x in list(rec.cameras.values())[0].params]}
    except Exception as e:  # noqa: BLE001
        sfm = {"error": str(e)}
    metrics = sorted(glob.glob(os.path.join(R, "train", "**", "metrics.json"), recursive=True), key=os.path.getmtime)
    m = _json(metrics[-1]) if metrics else None
    depth = os.path.realpath(os.path.join(P, "depth_cloud.ply")) if os.path.exists(os.path.join(P, "depth_cloud.ply")) else None
    sim3 = _json(os.path.join(P, "sim3.json"))

    s = {
        "sequence": cfg["sequence"]["name"],
        "variant": cfg.get("variant", "default"),
        "trainer": {"app": pl.get("trainer_app"), "overrides": pl.get("trainer_overrides") or []},
        "sfm": sfm,
        "sim3": None if not sim3 else {k: sim3[k] for k in ("scale", "n_frames", "residual_mean_m", "residual_p95_m")},
        "depth_source": depth,
        "train_metrics": None if not m else {k: round(m[k], 4) for k in ("mean_psnr", "mean_ssim", "mean_lpips") if k in m},
        "gaussians": {"splat": _ply_count(os.path.join(R, "splat.ply")),
                      "splat_filled": _ply_count(os.path.join(R, "splat_filled.ply"))},
        "floor_fill": _json(os.path.join(R, "report", "infill.json")),
        "floor_coverage": _json(os.path.join(R, "report", "floor_coverage.json")),
        "collider": _json(os.path.join(P, "collider", "collider_meta.json")),
        "outputs": {k: os.path.join(d, v) for k, (d, v) in {
            "splat_supersplat": (R, "splat_filled.ply"), "splat_metric_isaac": (R, "splat_metric.ply"),
            "collider": (P, "collider/collider.obj"), "floor_coverage_fig": (R, "report/floor_coverage.png")}.items()
            if os.path.exists(os.path.join(d, v))},
    }
    fz = _json(os.path.join(P, "fuse.json"))
    if fz:   # fused run: per member, fused splat vs the member's own splat on the member's data
        s["train_metrics"] = None      # trainer's own val = training views here; held-out is per member below
        s["fuse"] = {"frames_total": fz["frames_total"], "frames_train": fz["frames_train"], "members": {}}
        for m, info in fz["members"].items():
            own = _json(os.path.join(info["pipeline"], "summary.json")) or {}
            cov = _json(os.path.join(R, "report", f"floor_coverage_{m}.json")) or {}
            s["fuse"]["members"][m] = {
                "alignment": info["alignment"], "held_out": info["held_out"],
                "heldout_fused": _json(os.path.join(R, "report", "eval", f"{m}.fused.json")),
                "heldout_single": _json(os.path.join(R, "report", "eval", f"{m}.single.json")),
                "floor_fused": cov.get("splat_has_floor"),
                "floor_single": (own.get("floor_coverage") or {}).get("splat_has_floor")}
    json.dump(s, open(os.path.join(R, "summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
