#!/bin/bash
# Option A — build a collider from SPLAT-rendered depth (novel-view depth fusion).
# Stage 1 (this job, A100 = 3dgrut kernel arch): render dense pred_dist depth from the splat.
# Stage 2 (fuse_splat_collider.py, CPU): TSDF-fuse + floor-align -> collider.
#   sbatch -A pi_donghyunkim_umass_edu sbatch/collider_from_splat.sbatch mocap2_well-lit_trot [stride] [yaw] [limit]
#SBATCH -J cear-splatdepth
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap2_well-lit_trot}"; STRIDE="${2:-4}"; YAW="${3:-25}"; LIMIT="${4:-0}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source "$THREEDGRUT_DIR/.venv/bin/activate"

OUT="$CEAR_OUT/$SEQ"
CKPT="$(ls -t "$OUT"/train_3dgrut/"$SEQ"/*/ckpt_last.pt 2>/dev/null | head -1)"
DATASET="$OUT/dataset_3dgrut"
DEPTHDIR="$OUT/splat_depth"
[ -f "$CKPT" ] || { echo "no checkpoint for $SEQ"; exit 1; }
echo "ckpt=$CKPT stride=$STRIDE yaw=$YAW limit=$LIMIT"
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$THREEDGRUT_DIR"
python "$REPO/scripts/render_splat_depth.py" --checkpoint "$CKPT" --dataset "$DATASET" \
    --out "$DEPTHDIR" --stride "$STRIDE" --yaw "$YAW" --limit "$LIMIT"
echo "SPLAT_DEPTH_DONE $SEQ  ($(ls "$DEPTHDIR"/*.npz 2>/dev/null | wc -l) frames)"
