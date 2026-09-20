#!/bin/bash
# Phase 4 training on an A100 (rasterization path; no RT cores needed) — docs/PLAN.md §3.1.
#   sbatch sbatch/train_a100.sbatch mocap1_well-lit_trot
# Adjust the partition/account to your Unity allocation before submitting.
#SBATCH -J cear-train
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# CPU-side env (paths) — then activate the SEPARATE 3dgrut conda env for the GPU work.
source "$REPO/env/cear_env.sh"
module load conda/latest 2>/dev/null || true
# shellcheck disable=SC1091
conda activate "${THREEDGRUT_ENV:-3dgrut}"
export THREEDGRUT_DIR="${THREEDGRUT_DIR:?set THREEDGRUT_DIR to your 3dgrut clone}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$REPO/scripts/train_splat.sh" "$SEQ"
echo "TRAIN_DONE $SEQ"
