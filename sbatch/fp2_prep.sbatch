#!/bin/bash
# FusionPortable v2 sequence -> metric pinhole COLMAP datasets (scripts/datasets/fusionportable.py, CPU only):
# rosbag extract, sharpest-keyframe selection + undistortion, rig-constrained COLMAP aligned to the GT positions,
# deskewed + ICP-refined + dynamic-carved static LiDAR cloud, on-path train/test split (every 8th keyframe) with
# LiDAR init points and LiDAR depth targets.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_prep.sbatch configs/fp2_legged_room00.yaml
# Env: FP2_WS (default /scratch4/workspace/$USER-multicam), STEPS ("extract prepare sfm lidar split"), FORCE=1
#SBATCH -J fp2-prep
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"; export FP2_TORCH_PY="${FP2_TORCH_PY:-$THREEDGRUT_DIR/.venv/bin/python}"
hostname; date; which python; python --version
for s in ${STEPS:-extract prepare sfm lidar split}; do
  # shellcheck disable=SC2046
  python scripts/datasets/fusionportable.py "$s" "$CFG" $([ "${FORCE:-0}" = 1 ] && echo --force) $([ "$s" = split ] && echo --depth)
done
echo FP2_PREP_DONE
