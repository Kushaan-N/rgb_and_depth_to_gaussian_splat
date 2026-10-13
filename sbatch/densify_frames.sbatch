#!/bin/bash
# More frames: scripts/densify_frames.py picks extra low-blur frames, registers them into the existing COLMAP model
# (old frames + camera fixed) and writes $DF/data (training frames only, held-out test frames excluded; train with
# TSI=0, score on $P/onpath/test). Two stages:
#   MODE=cpu (default, CPU partition): select + SIFT + match + register + verify + write          -> $DF/{data,densify.json}
#   MODE=gpu (GPU partition): LiDAR depth targets for the new frames (build_depth_targets.py on $DF/newframes, linked
#            into $DF/data) + EDGS init on all of them (edgs_init.py, test_split_interval 0)       -> $DF/edgs_data
# Then train / score with the viewer bench on the new dataset, gated against the 423-frame arm:
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu,cpu-preempt -c 16 --mem=32G -t 04:00:00 -o logs/%x-%j.out sbatch/densify_frames.sbatch configs/<seq>.yaml
#   MODE=gpu sbatch -A ... -p gpu,gpu-preempt --gres=gpu:a100:1 -c 8 --mem=64G -t 02:00:00 -d afterok:<cpu> sbatch/densify_frames.sbatch configs/<seq>.yaml
#   VT_DATASET=$DF/edgs_data TSI=0 VT_OUT=$P/viewer_train_dense ITERS=60000 BASE_PLY=$P/viewer_train_60k/igs4m_pc274/splat.ply \
#     ARMS=igs4m_pc274 sbatch -A ... -d afterok:<gpu> sbatch/viewer_train_bench.sbatch configs/<seq>.yaml
# Env: DF (default $P/densify_frames), BLUR_DIR (default $P/blur), DENSIFY_ARGS (extra densify_frames.py flags),
#      EDGS_ARGS (extra edgs_init.py flags).
#SBATCH -J densify-frames
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; EPY="${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
P="$(get paths.out_root)/pipeline"; DF="${DF:-$P/densify_frames}"; MODE="${MODE:-cpu}"
hostname; date; export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; mkdir -p "$DF"
if [ "$MODE" = cpu ]; then
  "$CPUPY" scripts/densify_frames.py --config "$CFG" --out "$DF" --blur-dir "${BLUR_DIR:-$P/blur}" ${DENSIFY_ARGS:-}
  date; echo DENSIFY_FRAMES_CPU_DONE; exit 0
fi

export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "$DF/data/sparse/0/images.txt" ] || { echo "no $DF/data: run MODE=cpu first"; exit 1; }
STATIC="$P/offpath/depth/lidar_static.ply"
if [ -f "$STATIC" ]; then                                   # LiDAR targets (EDGS depth check); skipped without LiDAR
  [ -f "$DF/newframes/depth_targets.json" ] || "$GPY" scripts/build_depth_targets.py --dataset "$DF/newframes" --cloud "$STATIC" --sim3-json "$P/sim3.json" | tail -2
  for f in "$DF"/newframes/images/*_depth.npy; do ln -sfn "$(realpath "$f")" "$DF/data/images/$(basename "$f")"; done
  cp -f "$DF/newframes/depth_targets.json" "$DF/data/depth_targets.json"
fi
if [ ! -f "$DF/edgs_data/edgs_init.json" ]; then
  rm -rf "$DF/edgs_data"; T0=$(date +%s)
  "$EPY" scripts/edgs_init.py --dataset "$DF/data" --out "$DF/edgs_data" --test-split-interval 0 --sim3-json "$P/sim3.json" \
    ${EDGS_ARGS:-} > "$DF/edgs_init.log" 2>&1 || { tail -40 "$DF/edgs_init.log"; exit 1; }
  echo "edgs_init $(( $(date +%s) - T0 )) s"
fi
"$CPUPY" -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('args'); print(d)" "$DF/edgs_data/edgs_init.json"
date; echo DENSIFY_FRAMES_GPU_DONE
