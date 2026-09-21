#!/bin/bash
# 2DGS (2D Gaussian Splatting) -> clean surface mesh -> Isaac collider (Option C, premium).
# Idempotent setup (isolated venv + CUDA submodules) + train + TSDF mesh extract + gravity-align.
#   sbatch -A pi_donghyunkim_umass_edu sbatch/setup_train_2dgs.sbatch mocap2_well-lit_trot [iters]
#SBATCH -J cear-2dgs
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 02:30:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap2_well-lit_trot}"; ITERS="${2:-15000}"; MESHMODE="${3:-bounded}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="8.0"                 # A100
unset UV_EXCLUDE_NEWER 2>/dev/null || true
UV=/modules/opt/linux-ubuntu24.04-x86_64/uv/uv
TDGS="$CEAR_WS/2dgs"; VENV="$TDGS/.venv"; PY="$VENV/bin/python"
OUT="$CEAR_OUT/$SEQ"; DATASET="$OUT/dataset_3dgrut"; MODEL="$OUT/2dgs_model"
nvidia-smi --query-gpu=name --format=csv,noheader

# --- idempotent setup: isolated venv + torch 2.6/cu124 + deps + CUDA submodules ---
if [ ! -f "$TDGS/.setup_done" ]; then
  echo "=== setting up 2dgs venv ==="
  [ -d "$VENV" ] || "$UV" venv --python 3.11 "$VENV"   # reuse a partial venv (torch already installed)
  "$UV" pip install --python "$PY" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  "$UV" pip install --python "$PY" open3d plyfile mediapy opencv-python tqdm scipy trimesh
  "$UV" pip install --python "$PY" setuptools wheel ninja        # build deps for the CUDA extensions
  # --no-build-isolation: the submodules' setup.py imports torch at build time (isolated build has none)
  "$UV" pip install --python "$PY" --no-build-isolation \
      "$TDGS/submodules/diff-surfel-rasterization" "$TDGS/submodules/simple-knn"
  touch "$TDGS/.setup_done"
fi
echo "=== 2dgs env ready ==="; "$PY" -c "import torch,open3d,diff_surfel_rasterization; print('torch',torch.__version__,'o3d',open3d.__version__,'surfel ok')"

cd "$TDGS"
# --- train 2DGS on the COLMAP dataset (skip if already trained to ITERS) ---
if [ ! -d "$MODEL/point_cloud/iteration_${ITERS}" ]; then
  "$PY" train.py -s "$DATASET" -m "$MODEL" --iterations "$ITERS" -r 1
  echo "2DGS_TRAIN_DONE"
else
  echo "2DGS already trained to $ITERS; skipping train"
fi
# --- extract surface mesh: bounded (normalized scenes) or unbounded (robust for real scenes) ---
if [ "$MESHMODE" = "unbounded" ]; then
  "$PY" render.py -m "$MODEL" -s "$DATASET" --iteration "$ITERS" --skip_test --unbounded --mesh_res 1024 --num_cluster 50
  MESH="$MODEL/train/ours_${ITERS}/fuse_unbounded_post.ply"
else
  "$PY" render.py -m "$MODEL" -s "$DATASET" --iteration "$ITERS" --skip_test \
      --voxel_size 0.02 --depth_trunc 8.0 --num_cluster 50
  MESH="$MODEL/train/ours_${ITERS}/fuse_post.ply"
fi
echo "2DGS_MESH=$MESH"; ls -la "$MESH" 2>&1 | tail -1
# --- gravity-align into an Isaac collider ---
"$PY" "$REPO/scripts/align_mesh_collider.py" --mesh "$MESH" --out "$OUT/collider_2dgs" --source 2dgs
echo "COLLIDER_2DGS_DONE $SEQ"
