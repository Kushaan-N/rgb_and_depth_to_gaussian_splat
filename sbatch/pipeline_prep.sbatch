#!/bin/bash
# Pipeline stage 1/3 (CPU). Submitted by scripts/run_pipeline.sh — use that, not this directly.
#   raw data -> calibration -> poses/depth/gating -> LiDAR cloud -> metric depth cloud -> collider
#   -> COLMAP poses -> Sim3 (COLMAP -> metric) -> training dataset (depth-seeded init)
# Every step is idempotent and skips itself when its input is absent (no LiDAR bag, no depth, ...).
#SBATCH -J pipe-prep
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
get() { python3 scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; RGB="$(get sequence.rgb_dir rgb)"
GT="$OUT/colmap_train/sparse/0"
mkdir -p "$P/report"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "== [prep] $(get sequence.name) -> $P =="

echo "-- raw data"
python3 scripts/fetch_sequence.py --config "$CFG"

echo "-- calibration"
python3 scripts/fetch_calibration.py --config "$CFG"

if [ -d "$GT" ] && [ -d "$OUT/precond/$RGB" ] && [ -f "$OUT/gating/train_frames.txt" ]; then
  echo "-- poses/depth/gating: already done"
else
  echo "-- poses/depth/gating (CPU phases 1-5)"
  bash scripts/run_cpu_pipeline.sh --config "$CFG"
fi

if [ ! -f "$P/lidar_cloud.ply" ]; then
  echo "-- LiDAR cloud"
  python3 scripts/build_lidar_cloud.py --config "$CFG" --out "$P/lidar_cloud.ply"
fi

# metric depth cloud = first available of pipeline.depth_sources (used for init, floor fill, collider)
DEPTH=""
for src in $(get pipeline.depth_sources "lidar rgbd"); do
  case "$src" in
    lidar) if [ -f "$P/lidar_cloud.ply" ]; then DEPTH="$P/lidar_cloud.ply"; fi ;;
    rgbd)  if [ -f "$OUT/depth_map/fused_cloud.ply" ]; then DEPTH="$OUT/depth_map/fused_cloud.ply"; fi ;;
  esac
  if [ -n "$DEPTH" ]; then break; fi
done
if [ -n "$DEPTH" ]; then
  ln -sfn "$DEPTH" "$P/depth_cloud.ply"; echo "-- depth source: $DEPTH"
else
  rm -f "$P/depth_cloud.ply"; echo "-- depth source: none (init / floor fill / depth collider will skip)"
fi

# collider and COLMAP are independent: build the collider in the background on ~1/3 of the cores
# while COLMAP runs, and join before the stage ends (a collider failure still fails the stage).
NCPU="${SLURM_CPUS_PER_TASK:-16}"; COLLIDER_PID=""
if [ -n "$DEPTH" ] && [ ! -f "$P/collider/collider.obj" ]; then
  echo "-- collider from the depth cloud (background, overlapping COLMAP; log: report/collider.log)"
  OMP_NUM_THREADS=$(( NCPU / 3 > 0 ? NCPU / 3 : 1 )) \
    python3 scripts/build_lidar_collider.py --cloud "$P/depth_cloud.ply" --out "$P/collider" \
    > "$P/report/collider.log" 2>&1 &
  COLLIDER_PID=$!
fi

if [ ! -d "$P/dataset/sparse/0" ]; then
  echo "-- COLMAP poses (photometric bundle adjustment)"
  python3 scripts/colmap_sfm.py --config "$CFG" --out "$P/dataset" \
    --matcher "$(get pipeline.sfm_matcher auto)" --refine-intrinsics "$(get pipeline.sfm_refine_intrinsics 1)"
fi

echo "-- Sim3 COLMAP -> metric"
python3 scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" --sim3-json "$P/sim3.json"

# training dataset: COLMAP cameras/poses + depth-seeded init points. The pure COLMAP model in
# dataset/ stays untouched because pycolmap cannot read a mixed .bin/.txt model.
if [ ! -e "$P/train_data" ]; then
  if [ "$(get pipeline.init_from_depth 1)" = 1 ] && [ -n "$DEPTH" ]; then
    echo "-- training dataset: depth-seeded init"
    mkdir -p "$P/train_data/sparse/0"
    cp "$P/dataset/sparse/0/cameras.bin" "$P/dataset/sparse/0/images.bin" "$P/train_data/sparse/0/"
    ln -sfn "$P/dataset/images" "$P/train_data/images"
    python3 scripts/lidar_to_colmap_points.py --cloud "$P/depth_cloud.ply" --sim3-json "$P/sim3.json" \
      --out-points3d "$P/train_data/sparse/0/points3D.txt" --max-points "$(get pipeline.init_max_points 400000)"
  else
    echo "-- training dataset: COLMAP points (no depth init)"
    ln -sfn "$P/dataset" "$P/train_data"
  fi
fi
if [ -n "$COLLIDER_PID" ]; then
  if wait "$COLLIDER_PID"; then
    echo "-- collider: $(grep -E 'wrote collider' "$P/report/collider.log" | tail -1)"
  else
    echo "-- collider FAILED:"; tail -20 "$P/report/collider.log"; exit 1
  fi
fi
echo "PIPELINE_PREP_DONE"
