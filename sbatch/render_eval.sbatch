#!/bin/bash
# Gate 4.5 — render held-out on-trajectory AND off-trajectory (lateral 0.5 m / height 0.8 m)
# views from the trained checkpoint, to quantify how fast quality falls off away from the
# recorded path (the drivability predictor). Short job.
#   sbatch sbatch/render_eval.sbatch mocap1_well-lit_trot
#SBATCH -J cear-render
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
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
[ -f "$CKPT" ] || { echo "no checkpoint under $OUT/train_3dgrut/$SEQ"; exit 1; }
echo "checkpoint: $CKPT"
cd "$THREEDGRUT_DIR"

echo "=== render ON-trajectory held-out (real GT -> valid metrics) ==="
python render.py --checkpoint "$CKPT" --path "$OUT/eval_ontraj"     --out-dir "$OUT/render_ontraj"

echo "=== render OFF-trajectory (lateral 0.5m / height 0.8m; GT is placeholder, ignore metrics) ==="
python render.py --checkpoint "$CKPT" --path "$OUT/eval_offtraj_ds" --out-dir "$OUT/render_offtraj"
echo "RENDER_EVAL_DONE $SEQ"
