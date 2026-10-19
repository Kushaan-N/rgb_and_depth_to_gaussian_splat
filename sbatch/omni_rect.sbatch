#!/bin/bash
# Fisheye / 360 datasets, arm (a) "rectify, then the CEAR recipe": the dataset adapter (sequence.adapter ->
# scripts/datasets/<adapter>.py fetch + prepare) resamples every fisheye / equirect frame into virtual pinhole views
# (rect/{train,test}, cube-map style, full FOV kept, poses = frame pose x view rotation), then the best CEAR recipe
# runs unchanged on the pinhole set (sbatch/export_viewer.sbatch stages 1-2):
#   1. EDGS dense-correspondence init on the training views (scripts/edgs_init.py; views of one frame share a centre,
#      so EDGS's baseline rule pairs views of different frames only)
#   2. viewer-matched gsplat training (scripts/train_viewer_gsplat.py, Improved-GS 4M, PROF kernels)
#   3. viewer-exact SuperSplat 3.5.2 fine-tune (scripts/finetune_ply_viewer.py ss352, random display scales)
#   4. scoring in the live SuperSplat on the held-out frames + off-path + editor (sbatch/omni_score.sbatch inline)
# LiDAR stages (fill, clean_haze) need a LiDAR cloud + sim3 the omni adapters do not produce yet: skipped.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/omni-rect-%j.out sbatch/omni_rect.sbatch configs/<seq>.yaml
# Env: NAME (rect_igs), ARM_FLAGS (train_viewer_gsplat.py flags, default "--strategy improved --budget 4000000"),
#      PROF (pc274), DSCALE (1.5:4), FT_ITERS (7000), FT_ARGS, NO_EDGS=1 (init from the SfM points only), SCORE (1),
#      EDGS_DATA (reuse an existing EDGS init dir, e.g. <out>/pipeline/rect_igs/edgs_data)
#SBATCH -J omni-rect
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
export CEAR_DATA="${CEAR_DATA:-/scratch4/workspace/${USER}-multicam/data}" CEAR_OUT="${CEAR_OUT:-/scratch4/workspace/${USER}-multicam/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; EPY="${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ADAPTER="$(get sequence.adapter)"
NAME="${NAME:-rect_igs}"; F="$P/$NAME"; PROF="${PROF:-pc274}"; ARM_FLAGS="${ARM_FLAGS:---strategy improved --budget 4000000}"
hostname; date; mkdir -p "$F"; nvidia-smi --query-gpu=name --format=csv,noheader || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"

echo "== 0. adapter $ADAPTER: fetch + prepare (rectified pinhole views + native frames)"
"$CPUPY" "scripts/datasets/$ADAPTER.py" fetch "$CFG"
mkdir -p "$P"; (                                                   # rect + native jobs may start together
  flock 9; [ -f "$P/meta.json" ] && [ -f "$P/rect/test/sparse/0/images.txt" ] || \
    "$CPUPY" "scripts/datasets/$ADAPTER.py" prepare "$CFG" --workers "${SLURM_CPUS_PER_TASK:-12}" ) 9>"$P/.prepare.lock"
TRAIN="$P/rect/train"; TEST="$P/rect/test"

module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
bash env/make_gsplat_viewer_venv.sh

EDGS="${EDGS_DATA:-$F/edgs_data}"                                  # EDGS_DATA: reuse another arm's init
if [ "${NO_EDGS:-0}" = 1 ]; then EDGS="$TRAIN"
else
  echo "== 1. EDGS dense init on the training views"
  [ -f "$EDGS/edgs_init.json" ] || "$EPY" scripts/edgs_init.py --dataset "$TRAIN" --out "$EDGS" --test-split-interval 0 \
    > "$F/edgs_init.log" 2>&1 || { tail -40 "$F/edgs_init.log"; exit 1; }
  tail -2 "$F/edgs_init.log"
fi

echo "== 2. viewer-matched training ($PROF kernels): $ARM_FLAGS"
if [ ! -f "$F/train/splat.ply" ]; then
  mkdir -p "$F/train"
  # shellcheck disable=SC2086
  PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
    "$SPY" scripts/train_viewer_gsplat.py --dataset "$EDGS" --out "$F/train" --test-split-interval 0 \
    --eval-dataset "$TEST" --viewer "$PROF" $ARM_FLAGS > "$F/train.log" 2>&1 || { tail -40 "$F/train.log"; exit 1; }
fi
grep -E '^\[vgs\] (held-out|[0-9]+ gaussians)' "$F/train.log" 2>/dev/null || true

echo "== 3. viewer-exact fine-tune (display scales ${DSCALE:-1.5:4})"
# shellcheck disable=SC2086
[ -f "$F/ft.ply" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$F/train/splat.ply" --out "$F/ft.ply" \
  --dataset "$TRAIN" --test-split-interval 0 --viewer ss352 --display-scale "${DSCALE:-1.5:4}" \
  --iters "${FT_ITERS:-7000}" --eval-dataset "$TEST" ${FT_ARGS:-} > "$F/ft.log" 2>&1 || { tail -40 "$F/ft.log"; exit 1; }
grep -iE 'held-out|eval' "$F/ft.log" | tail -3 || true

if [ "${SCORE:-1}" = 1 ]; then
  echo "== 4. live SuperSplat scoring (held-out rectified views, off-path, editor)"
  bash sbatch/omni_score.sbatch "$CFG" "${NAME}_train=$F/train/splat.ply" "${NAME}_ft=$F/ft.ply"
fi
echo OMNI_RECT_DONE
