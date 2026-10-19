#!/usr/bin/env bash
# Venv for ODGS (Lee, Chung et al., NeurIPS 2024, arXiv 2410.20686; MIT): 3D Gaussian splatting trained directly on
# equirectangular 360 images (per-gaussian tangent-plane projection on the unit sphere, its own CUDA rasteriser).
# Used by sbatch/ricoh360_native.sbatch as the native-360 arm (scripts/datasets/ricoh360.py writes its input).
# Clones ODGS at a pinned commit with its submodules (simple-knn, odgs-gaussian-rasterization) and builds them
# against the cluster's CUDA 12.4 / torch 2.6 (the paper's env is torch 2.1 + cu118; the kernels are 3DGS-style
# and build unchanged). Run inside a GPU job (compiles CUDA):
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:1 -c 8 --mem=32G -t 01:00:00 env/make_odgs_venv.sh
#
# Idempotent: re-running skips what exists.
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
ODGS_VENV="${ODGS_VENV:-$CEAR_WS/venv-odgs}"
ODGS_DIR="${ODGS_DIR:-$CEAR_WS/ODGS}"
ODGS_COMMIT="${ODGS_COMMIT:-4cb96a25090ab2f8c53f1298e4fe62dce57dbf1c}"   # esw0116/ODGS main, 2024-12
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
hostname; date

[ -d "$ODGS_DIR/.git" ] || git clone -q https://github.com/esw0116/ODGS.git "$ODGS_DIR"
git -C "$ODGS_DIR" fetch -q origin && git -C "$ODGS_DIR" checkout -q "$ODGS_COMMIT"
git -C "$ODGS_DIR" submodule update --init --recursive -q
echo "[odgs] ODGS @ $(git -C "$ODGS_DIR" rev-parse --short HEAD)"

if [ -x "$ODGS_VENV/bin/python" ] && "$ODGS_VENV/bin/python" -c "import odgs_gaussian_rasterization, simple_knn" 2>/dev/null; then
  echo "[odgs] venv ready: $ODGS_VENV"; exit 0
fi
module load cuda/12.4.1 gcc/11.2.0 2>/dev/null || module load cuda/12.4.1 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$ODGS_VENV/bin/python" ] || "$UV" venv --python 3.10 "$ODGS_VENV"
"$UV" pip install --python "$ODGS_VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "numpy<2" plyfile==1.1 tqdm opencv-python-headless pillow \
  scipy setuptools wheel ninja
# 3DGS-era rasteriser sources miss <cstdint> (gcc >= 13 no longer pulls it in transitively): add it, idempotently
for h in "$ODGS_DIR"/submodules/odgs-gaussian-rasterization/cuda_rasterizer/{rasterizer_impl.h,rasterizer.h,auxiliary.h}; do
  grep -q '#include <cstdint>' "$h" || sed -i '1i #include <cstdint>' "$h"
done
# the CUDA extensions import torch at build time: no build isolation
for s in simple-knn odgs-gaussian-rasterization; do
  "$UV" pip install --python "$ODGS_VENV/bin/python" --no-build-isolation "$ODGS_DIR/submodules/$s"
done
"$ODGS_VENV/bin/python" -c "import torch, odgs_gaussian_rasterization, simple_knn; print('OK torch', torch.__version__, torch.cuda.is_available())"
