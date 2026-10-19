#!/bin/bash
# Oxford Spires (or any adapter-built rig dataset) through the CEAR best recipe, unchanged: EDGS dense init on the
# rectified-pinhole training frames (scripts/edgs_init.py; what export_igs.sbatch ARMS=edgs_igs does, minus the 3DGRUT
# training) and then sbatch/export_viewer.sbatch in this same job (viewer-matched Improved-GS 4M training, viewer-exact
# fine-tune, LiDAR fill + inpaint, haze cleanup). Needs scripts/datasets/spires.py (layout stage) to have run.
#   CEAR_DATA=$MC/data/spires CEAR_OUT=$MC/outputs sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out \
#     sbatch/spires_export.sbatch configs/spires_roq01.yaml
# Env: everything export_viewer.sbatch takes (NAME, ARM_FLAGS, TRAIN_PLY, FT_ARGS, CLEAN, ...); EDGS_ARGS (extra
#      edgs_init.py flags); DEPTH_TARGETS=1 (LiDAR <stem>_depth.npy targets in the training dir first, so EDGS gets
#      its LiDAR check and --depth-lambda in ARM_FLAGS / FT_ARGS has targets; needs adapter.layout_own_train),
#      DEPTH_MAX_RANGE (40 m).
#SBATCH -J spires-export
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
get() { "$CEAR_VENV/bin/python" scripts/cfg_get.py "$CFG" "$@"; }
P="$(get paths.out_root)/pipeline"; DATA="$P/onpath/data_orig"; EDGS="${EDGS_DATA:-$P/igs_full/edgs_data}"
[ -f "$DATA/sparse/0/images.txt" ] || { echo "missing $DATA (run sbatch/spires_prepare.sbatch)"; exit 1; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
if [ "${DEPTH_TARGETS:-0}" = 1 ] && [ ! -f "$DATA/depth_targets.json" ]; then   # LiDAR ray-distance targets
  [ -L "$DATA" ] && { echo "$DATA is a shared symlink: use a config with adapter.layout_own_train: true"; exit 1; }
  module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
  "$THREEDGRUT_DIR/.venv/bin/python" scripts/build_depth_targets.py --dataset "$DATA" --cloud "$P/depth_cloud.ply" \
    --sim3-json "$P/sim3.json" --voxel-m "$(get adapter.lidar_voxel 0.05)" --max-range-m "${DEPTH_MAX_RANGE:-40}" | tail -2
fi
if [ -z "${TRAIN_PLY:-}" ] && [ ! -f "$EDGS/edgs_init.json" ]; then
  echo "== EDGS init on $DATA"
  mkdir -p "$(dirname "$EDGS")"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
  # shellcheck disable=SC2086
  "${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python" scripts/edgs_init.py --dataset "$DATA" --out "$EDGS" \
    --test-split-interval 0 --sim3-json "$P/sim3.json" ${EDGS_ARGS:-} > "$(dirname "$EDGS")/edgs_init.log" 2>&1 \
    || { tail -40 "$(dirname "$EDGS")/edgs_init.log"; exit 1; }
  tail -3 "$(dirname "$EDGS")/edgs_init.log"
fi
if [ -n "${TRAIN_PLY:-}" ]; then                    # stage 1 skipped: export_viewer only checks the init exists
  export EDGS_DATA="$P/igs_full/none_${NAME:-viewer_igs}"; mkdir -p "$EDGS_DATA"
  echo '{"note": "TRAIN_PLY given: stage-1 training skipped"}' > "$EDGS_DATA/edgs_init.json"
fi
bash sbatch/export_viewer.sbatch "$CFG"
