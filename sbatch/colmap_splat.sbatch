#!/bin/bash
# DEFAULT splat path: COLMAP-optimized poses give a much sharper splat than the raw GT mocap poses
# (measured on mocap2: PSNR 26.2 -> 37.2, +11 dB), then Sim3-align it back into the metric GT/LiDAR
# frame so it stays co-registered with the LiDAR collider for Isaac. End to end:
#   COLMAP SfM -> 3DGRUT train -> export .ply (COLMAP frame, for SuperSplat) -> Sim3 -> metric .ply
#
#   sbatch -A pi_donghyunkim_umass_edu sbatch/colmap_splat.sbatch mocap2_well-lit_trot
# The soft-but-metric GT path is still sbatch/train_a100.sbatch (kept for reference / ablation).
#SBATCH -J cear-colsplat
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap2_well-lit_trot}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"                      # CPU venv: pycolmap + plyfile + scipy + pose_utils
CPU_PY="$(command -v python3)"
NAV="$CEAR_OUT/$SEQ"
DS="$NAV/dataset_colmap"

# --- 1. COLMAP SfM (known-K init, BA-refined poses/intrinsics) -> 3DGRUT dataset ---
if [ ! -d "$DS/sparse/0" ]; then
  echo "=== [1/4] COLMAP SfM ==="
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
  "$CPU_PY" "$REPO/scripts/colmap_sfm.py" --config "$REPO/configs/$SEQ.yaml" --out "$DS" --matcher exhaustive
else
  echo "=== [1/4] dataset_colmap exists; skipping SfM ==="
fi

# --- 2. train 3DGRUT on the COLMAP dataset (needs the 3dgrut venv) ---
deactivate 2>/dev/null || true
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source "$THREEDGRUT_DIR/.venv/bin/activate"
OUTDIR="$NAV/train_3dgrut_colmap"
if ! ls "$OUTDIR"/*/*/ckpt_last.pt >/dev/null 2>&1; then
  echo "=== [2/4] train 3DGRUT (COLMAP poses) ==="
  ( cd "$THREEDGRUT_DIR" && python train.py --config-name apps/colmap_3dgut.yaml \
      path="$DS" out_dir="$OUTDIR" experiment_name="${SEQ}_colmap" dataset.downsample_factor=1 )
else
  echo "=== [2/4] COLMAP checkpoint exists; skipping train ==="
fi
CKPT="$(ls -t "$OUTDIR"/*/*/ckpt_last.pt 2>/dev/null | head -1)"
[ -f "$CKPT" ] || { echo "no checkpoint produced"; exit 1; }

# --- 3. export the splat to a standard 3DGS .ply (COLMAP frame — drop straight into SuperSplat) ---
echo "=== [3/4] export .ply ==="
python3 "$REPO/scripts/export_gaussian_ply.py" --checkpoint "$CKPT" --out "$NAV/gaussians_colmap.ply"

# --- 4. Sim3-align into the metric GT/LiDAR frame (needs pycolmap+plyfile -> back to CPU venv) ---
deactivate 2>/dev/null || true
source "$REPO/env/cear_env.sh"
echo "=== [4/4] Sim3 align -> metric frame ==="
python3 "$REPO/scripts/sim3_align_splat.py" \
  --colmap-model "$DS/sparse/0" --gt-model "$NAV/colmap_train/sparse/0" \
  --in-ply "$NAV/gaussians_colmap.ply" --out-ply "$NAV/gaussians_colmap_metric.ply"

echo "COLMAP_SPLAT_DONE $SEQ"
echo "  SuperSplat (COLMAP frame): $NAV/gaussians_colmap.ply"
echo "  Metric (Isaac/LiDAR frame): $NAV/gaussians_colmap_metric.ply"
