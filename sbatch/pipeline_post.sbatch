#!/bin/bash
# Pipeline stage 3/3 (CPU). Submitted by scripts/run_pipeline.sh after training succeeds.
#   floor fill (optional, auto-skips) -> metric splat for Isaac -> floor-coverage report -> summary.json
#SBATCH -J pipe-post
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
get() { python3 scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; RGB="$(get sequence.rgb_dir rgb)"
GT="$OUT/colmap_train/sparse/0"
mkdir -p "$P/report"
echo "== [post] $(get sequence.name) =="

if [ "$(get pipeline.floor_fill 1)" = 1 ] && [ -e "$P/depth_cloud.ply" ]; then
  echo "-- floor fill"
  python3 scripts/infill_floor.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$P/splat.ply" \
    --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" --images "$OUT/precond/$RGB" \
    --out-ply "$P/splat_filled.ply" | tee "$P/report/infill.log"
else
  cp "$P/splat.ply" "$P/splat_filled.ply"
  echo 'INFILL {"skipped": "disabled or no depth source", "added_gaussians": 0}' | tee "$P/report/infill.log"
fi
grep '^INFILL ' "$P/report/infill.log" | tail -1 | sed 's/^INFILL //' > "$P/report/infill.json"

echo "-- metric splat (Sim3 -> the frame the collider and Isaac use)"
python3 scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
  --in-ply "$P/splat_filled.ply" --out-ply "$P/splat_metric.ply"

if [ -e "$P/depth_cloud.ply" ]; then
  echo "-- floor coverage report"
  python3 scripts/floor_coverage.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$P/splat_filled.ply" \
    --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" --out "$P/report/floor_coverage.png" \
    | tee "$P/report/floor_coverage.log"
  grep '^FLOOR_COVERAGE ' "$P/report/floor_coverage.log" | tail -1 | sed 's/^FLOOR_COVERAGE //' > "$P/report/floor_coverage.json"
fi

python3 scripts/pipeline_summary.py --config "$CFG"
echo "PIPELINE_POST_DONE"
