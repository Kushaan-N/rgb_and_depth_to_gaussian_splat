#!/bin/bash
# Pipeline stage 1/3 for a FUSED config (fuse.members): replaces pipeline_prep.sbatch. Submitted by
# scripts/run_pipeline.sh after every member's own prep has finished.
#   member datasets -> one metric-frame training set (ICP-checked) -> fused depth cloud -> collider
#   -> depth-seeded init. Train and post then run exactly as for a single sequence.
#SBATCH -J pipe-fuse
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
get() { python3 scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"
mkdir -p "$P/report"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "== [prep-fuse] $(get sequence.name): $(get fuse.members) -> $P =="

if [ ! -f "$P/fuse.json" ]; then
  python3 scripts/fuse_sequences.py --config "$CFG"
fi
DEPTH=""
if [ -f "$P/lidar_cloud.ply" ]; then DEPTH="$P/lidar_cloud.ply"; ln -sfn "$DEPTH" "$P/depth_cloud.ply"; fi

if [ -n "$DEPTH" ] && [ ! -f "$P/collider/collider.obj" ]; then
  echo "-- collider from the fused depth cloud"
  python3 scripts/build_lidar_collider.py --cloud "$P/depth_cloud.ply" --out "$P/collider" \
    > "$P/report/collider.log" 2>&1 || { tail -20 "$P/report/collider.log"; exit 1; }
  echo "-- collider: $(grep -E 'wrote collider' "$P/report/collider.log" | tail -1)"
fi
if [ ! -f "$P/train_data/sparse/0/points3D.txt" ]; then
  if [ "$(get pipeline.init_from_depth 1)" = 1 ] && [ -n "$DEPTH" ]; then
    python3 scripts/lidar_to_colmap_points.py --cloud "$P/depth_cloud.ply" --sim3-json "$P/sim3.json" \
      --out-points3d "$P/train_data/sparse/0/points3D.txt" --max-points "$(get pipeline.init_max_points 400000)"
  else
    echo "no depth cloud for the init -> fusion needs member depth clouds"; exit 1
  fi
fi
echo "PIPELINE_PREP_DONE"
