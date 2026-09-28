#!/bin/bash
# Pipeline stage 2/3 (GPU). Submitted by scripts/run_pipeline.sh after prep succeeds.
#   train 3DGRUT on <out>/pipeline/train_data -> export the splat to a standard 3DGS .ply
# 3DGRUT's CUDA kernels are built for A100 (sm_80).
#SBATCH -J pipe-train
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
get() { python3 scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; SEQ="$(get sequence.name)"
ITERS="$(get pipeline.train_iterations 30000)"
echo "== [train] $SEQ ($ITERS iters) =="

deactivate 2>/dev/null || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source "$THREEDGRUT_DIR/.venv/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader

if ! ls "$P"/train/*/*/ckpt_last.pt >/dev/null 2>&1; then
  ( cd "$THREEDGRUT_DIR" && python train.py --config-name apps/colmap_3dgut.yaml \
      path="$P/train_data" out_dir="$P/train" experiment_name="$SEQ" \
      dataset.downsample_factor=1 n_iterations="$ITERS" )
else
  echo "-- checkpoint exists; skipping training"
fi
CKPT="$(ls -t "$P"/train/*/*/ckpt_last.pt | head -1)"
python3 "$REPO/scripts/export_gaussian_ply.py" --checkpoint "$CKPT" --out "$P/splat.ply"
echo "PIPELINE_TRAIN_DONE"
