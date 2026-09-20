#!/bin/bash
# Phase 4 training on an A100 (rasterization / 3DGUT; no RT cores needed) — docs/PLAN.md §3.1.
#   sbatch sbatch/train_a100.sbatch mocap1_well-lit_trot
# Requires the 3DGRUT env built first:  sbatch sbatch/build_3dgrut.sbatch
# Adjust partition/account to your Unity allocation before submitting.
#SBATCH -J cear-train
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh"            # sets CEAR_DATA / CEAR_OUT (also activates CPU venv)
deactivate 2>/dev/null || true            # ...but training uses the 3dgrut .venv, not the CPU one

export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
[ -d "$THREEDGRUT_DIR/.venv" ] || { echo "3dgrut .venv missing — run sbatch/build_3dgrut.sbatch first"; exit 1; }
# shellcheck disable=SC1091
source "$THREEDGRUT_DIR/.venv/bin/activate"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
bash "$REPO/scripts/train_splat.sh" "$SEQ"
echo "TRAIN_DONE $SEQ"
