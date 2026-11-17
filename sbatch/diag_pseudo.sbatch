#!/bin/bash
# Short diagnosis of a pseudo-view training run (scripts/object_layers/diag_pseudo.py); args after the config are passed through.
#SBATCH -J diag-pseudo
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010,gpu035
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 01:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$REPO"; source env/cear_env.sh >/dev/null
SPY="$CEAR_WS/venv-gsplat/bin/python"; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
shift; "$SPY" "${DIAG_SCRIPT:-scripts/object_layers/diag_pseudo.py}" "$@"
