#!/bin/bash
# Option B — build a collider directly from the gaussian representation (Poisson surface).
# Stage 1 (A100 = 3dgrut arch): export gaussians -> .ply. Stage 2 (CPU): Poisson -> collider.
#   sbatch -A pi_donghyunkim_umass_edu sbatch/collider_from_gaussians.sbatch mocap2_well-lit_trot [poisson_depth]
#SBATCH -J cear-gaussmesh
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap2_well-lit_trot}"; PDEPTH="${2:-9}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

OUT="$CEAR_OUT/$SEQ"
CKPT="$(ls -t "$OUT"/train_3dgrut/"$SEQ"/*/ckpt_last.pt 2>/dev/null | head -1)"
PLY="$OUT/gaussians.ply"
[ -f "$CKPT" ] || { echo "no checkpoint for $SEQ"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

# Stage 1 — export .ply in the 3dgrut venv
source "$THREEDGRUT_DIR/.venv/bin/activate"
cd "$THREEDGRUT_DIR"
python "$REPO/scripts/export_gaussian_ply.py" --checkpoint "$CKPT" --out "$PLY"
echo "GAUSSIAN_PLY_DONE $SEQ"

# Stage 2 — Poisson surface on CPU (Open3D in the CPU venv)
deactivate 2>/dev/null || true
source "$REPO/env/cear_env.sh"
python "$REPO/scripts/mesh_from_gaussians.py" --ply "$PLY" --out "$OUT/collider_splatB" --poisson-depth "$PDEPTH"
echo "COLLIDER_SPLAT_B_DONE $SEQ"
