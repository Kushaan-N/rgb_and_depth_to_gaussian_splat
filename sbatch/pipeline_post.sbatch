#!/bin/bash
# Pipeline stage 3/3 (CPU). Submitted by scripts/run_pipeline.sh after training succeeds.
#   floor fill (optional, auto-skips) -> metric splat for Isaac -> floor-coverage report -> summary.json
# Shared prep inputs are read from <out>/pipeline; this run's outputs go to pipeline.run_dir.
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
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; R="$(get pipeline.run_dir "$P")"; RGB="$(get sequence.rgb_dir rgb)"
GT="$OUT/colmap_train/sparse/0"
mkdir -p "$R/report"
echo "== [post] $(get sequence.name) -> $R =="

if [ "$(get pipeline.floor_fill 1)" = 1 ] && [ -e "$P/depth_cloud.ply" ]; then
  echo "-- floor fill"
  python3 scripts/infill_floor.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$R/splat.ply" \
    --sim3-json "$P/sim3.json" --gt-model "$GT" --images "$OUT/precond/$RGB" \
    --out-ply "$R/splat_filled.ply" | tee "$R/report/infill.log"
else
  cp "$R/splat.ply" "$R/splat_filled.ply"
  echo 'INFILL {"skipped": "disabled or no depth source", "added_gaussians": 0}' | tee "$R/report/infill.log"
fi
grep '^INFILL ' "$R/report/infill.log" | tail -1 | sed 's/^INFILL //' > "$R/report/infill.json"

echo "-- metric splat (Sim3 -> the frame the collider and Isaac use)"
python3 scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
  --in-ply "$R/splat_filled.ply" --out-ply "$R/splat_metric.ply"

if [ -e "$P/depth_cloud.ply" ]; then
  echo "-- floor coverage report"
  python3 scripts/floor_coverage.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$R/splat_filled.ply" \
    --sim3-json "$P/sim3.json" --gt-model "$GT" --out "$R/report/floor_coverage.png" \
    | tee "$R/report/floor_coverage.log"
  grep '^FLOOR_COVERAGE ' "$R/report/floor_coverage.log" | tail -1 | sed 's/^FLOOR_COVERAGE //' > "$R/report/floor_coverage.json"
fi

python3 scripts/pipeline_summary.py --config "$CFG"
echo "PIPELINE_POST_DONE"
