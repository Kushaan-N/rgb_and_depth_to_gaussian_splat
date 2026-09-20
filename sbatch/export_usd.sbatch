#!/bin/bash
# Export the trained 3DGRUT checkpoint to USD splat assets for Isaac Sim.
# NuRec USDZ is the format Isaac Sim 5.0 consumes; also write ParticleField (standard) for
# forward-compat with Isaac 6.0. Short GPU job.
#   sbatch sbatch/export_usd.sbatch mocap1_well-lit_trot
#SBATCH -J cear-export
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source "$THREEDGRUT_DIR/.venv/bin/activate"

OUT="$CEAR_OUT/$SEQ"
CKPT="$(ls -t "$OUT"/train_3dgrut/"$SEQ"/*/ckpt_last.pt 2>/dev/null | head -1)"
DATASET="$OUT/dataset_3dgrut"
mkdir -p "$OUT/usd"
[ -f "$CKPT" ] || { echo "no checkpoint"; exit 1; }
echo "checkpoint: $CKPT"
cd "$THREEDGRUT_DIR"

echo "=== NuRec USDZ (Isaac Sim 5.0) ==="
python -m threedgrut.export.scripts.export_usd \
    --checkpoint "$CKPT" --output "$OUT/usd/scene_nurec.usdz" \
    --dataset "$DATASET" --format nurec

echo "=== ParticleField USD (Isaac 6.0 forward-compat) ==="
python -m threedgrut.export.scripts.export_usd \
    --checkpoint "$CKPT" --output "$OUT/usd/scene_particlefield.usdz" \
    --dataset "$DATASET" --format standard --up-axis z || echo "(particlefield export optional; NuRec is what Isaac 5.0 uses)"

echo "EXPORT_USD_DONE $SEQ"; ls -la "$OUT/usd/"
