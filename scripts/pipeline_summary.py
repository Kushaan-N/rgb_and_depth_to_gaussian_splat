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
    out = cfg["paths"]["out_root"]; P = os.path.join(out, "pipeline")

    sfm = None
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(os.path.join(P, "dataset", "sparse", "0"))
        n_img = len(os.listdir(os.path.join(P, "dataset", "images")))
        sfm = {"registered": rec.num_reg_images(), "frames": n_img, "points": rec.num_points3D(),
               "camera": [round(float(x), 2) for x in list(rec.cameras.values())[0].params]}
    except Exception as e:  # noqa: BLE001
        sfm = {"error": str(e)}
    metrics = sorted(glob.glob(os.path.join(P, "train", "**", "metrics.json"), recursive=True), key=os.path.getmtime)
    m = _json(metrics[-1]) if metrics else None
    depth = os.path.realpath(os.path.join(P, "depth_cloud.ply")) if os.path.exists(os.path.join(P, "depth_cloud.ply")) else None
    sim3 = _json(os.path.join(P, "sim3.json"))

    s = {
        "sequence": cfg["sequence"]["name"],
        "sfm": sfm,
        "sim3": None if not sim3 else {k: sim3[k] for k in ("scale", "n_frames", "residual_mean_m", "residual_p95_m")},
        "depth_source": depth,
        "train_metrics": None if not m else {k: round(m[k], 4) for k in ("mean_psnr", "mean_ssim", "mean_lpips") if k in m},
        "gaussians": {"splat": _ply_count(os.path.join(P, "splat.ply")),
                      "splat_filled": _ply_count(os.path.join(P, "splat_filled.ply"))},
        "floor_fill": _json(os.path.join(P, "report", "infill.json")),
        "floor_coverage": _json(os.path.join(P, "report", "floor_coverage.json")),
        "collider": _json(os.path.join(P, "collider", "collider_meta.json")),
        "outputs": {k: os.path.join(P, v) for k, v in {
            "splat_supersplat": "splat_filled.ply", "splat_metric_isaac": "splat_metric.ply",
            "collider": "collider/collider.obj", "floor_coverage_fig": "report/floor_coverage.png"}.items()
            if os.path.exists(os.path.join(P, v))},
    }
    json.dump(s, open(os.path.join(P, "summary.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
