#!/bin/bash
# One-time 3DGRUT environment build. The CUDA extensions (tiny-cuda-nn, 3dgut kernels)
# compile with nvcc but need NO live GPU, so this runs on a CPU node — GPUs are only used
# for training. Creates $THREEDGRUT_DIR/.venv (uv). ~30-60 min.
#   sbatch sbatch/build_3dgrut.sbatch
#SBATCH -J 3dgrut-build
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
unset UV_EXCLUDE_NEWER 2>/dev/null || true

module load cuda/12.4.1 gcc/9.4.0 uv/latest 2>/dev/null || module load cuda/12.4.1 gcc/9.4.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
# build for A100 (8.0), L40S (8.9), H100 (9.0) so the env runs on any of our GPUs
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -1
cd "$THREEDGRUT_DIR"
if [ -d .venv ]; then
  echo "[build] .venv already exists — remove it to rebuild. Verifying import instead."
  ./.venv/bin/python -c "import torch, kaolin; print('torch', torch.__version__, 'kaolin', kaolin.__version__)"
else
  bash ./install_env_uv.sh 3dgrut
fi
echo "BUILD_3DGRUT_DONE"
