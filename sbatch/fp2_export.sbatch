#!/bin/bash
# The CEAR best recipe (sbatch/export_viewer.sbatch, copied: EDGS init + viewer-matched gsplat Improved-GS 4M +
# viewer-exact WD fine-tune + LiDAR unseen-surface fill/harm-check/LaMa inpaint + haze cleanup) on a FusionPortable
# v2 rig dataset (sbatch/fp2_prep.sbatch). Differences from export_viewer.sbatch: trains on the on-path TRAIN split
# only (<P>/onpath/train; every 8th keyframe of every camera held out in <P>/onpath/test, so the held-out score is
# honest), the dataset is already metric (sim3.json = identity) so there is no axes conversion: exports are in the
# GT metric frame. Then scored in the live SuperSplat (sbatch/fp2_check.sbatch does it; CHECK=1 here runs it inline).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_export.sbatch configs/fp2_legged_room00.yaml
# Env: NAME (viewer_igs), ARM_FLAGS (default "--strategy improved --budget 4000000 --iters 60000"), PROF (pc274),
#      DSCALE (1.5:4), FT_ITERS (7000), FT_ARGS ("--lambda-wd 0.025 --wd-log2-sigma 4" = the WD fine-tune),
#      TRAIN_PLY (reuse a trained stage-1 splat, e.g. to test fine-tune variants), TRAIN_DATA (default <P>/onpath/train), TEST_DATA (default <P>/onpath/test), EDGS_ARGS, CLEAN (1), CHECK (1),
#      MASK=1 (person masks on the train frames, scripts/datasets/fusionportable_masks.py, excluded from the training and
#      fine-tune losses via --mask-dir; MASK_PERSIST=1 first unmasks people who stay put, fusionportable_masks.py
#      persist, PERSIST_ARGS), VIEW (1: also <NAME>*_view.ply, scripts/view_frame.py), FP2_WS
#SBATCH -J fp2-export
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; NAME="${NAME:-viewer_igs}"; F="$P/$NAME"
DATA="${TRAIN_DATA:-$P/onpath/train}"; TEST="${TEST_DATA:-$P/onpath/test}"; EDGS="$F/edgs_data"; SIM3="$P/sim3.json"
CLOUD="$P/lidar_static.ply"
PROF="${PROF:-pc274}"; ARM_FLAGS="${ARM_FLAGS:---strategy improved --budget 4000000 --iters 60000}"
FT_ARGS="${FT_ARGS:---lambda-wd 0.025 --wd-log2-sigma 4}"
[ -f "$DATA/sparse/0/images.txt" ] || { echo "missing $DATA (run sbatch/fp2_prep.sbatch)"; exit 1; }
hostname; date; mkdir -p "$F" "$E"; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
bash env/make_gsplat_viewer_venv.sh

MASK_FLAGS=""
if [ "${MASK:-0}" = 1 ]; then
  echo "== 0a. transient (person) masks on the training frames"
  TORCH_HOME="$TORCH_HOME" "${MASK_PY:-$CEAR_WS/venv-edgs/bin/python}" scripts/datasets/fusionportable_masks.py mask --dataset "$DATA"
  MASK_FLAGS="--mask-dir $DATA/images"
  if [ "${MASK_PERSIST:-0}" = 1 ]; then              # people who stay put are scene, not transients: unmask them
    "$CPUPY" scripts/datasets/fusionportable_masks.py persist --dataset "$DATA" --out-dir "$DATA/masks_persist" ${PERSIST_ARGS:-}
    MASK_FLAGS="--mask-dir $DATA/masks_persist"
  fi
fi
echo "== 0. EDGS dense-correspondence init (train frames)"
# shellcheck disable=SC2086
[ -f "$EDGS/edgs_init.json" ] || "${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python" scripts/edgs_init.py \
  --dataset "$DATA" --out "$EDGS" --test-split-interval 0 --sim3-json "$SIM3" ${EDGS_ARGS:-} > "$F/edgs_init.log" 2>&1 \
  || { tail -40 "$F/edgs_init.log"; exit 1; }
tail -2 "$F/edgs_init.log"

echo "== 1. viewer-matched training ($PROF kernels): $ARM_FLAGS"
if [ -n "${TRAIN_PLY:-}" ] && [ ! -f "$F/train/splat.ply" ]; then mkdir -p "$F/train"; cp "$TRAIN_PLY" "$F/train/splat.ply"; fi
if [ ! -f "$F/train/splat.ply" ]; then
  mkdir -p "$F/train"
  # shellcheck disable=SC2086
  PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
    "$SPY" scripts/train_viewer_gsplat.py --dataset "$EDGS" --out "$F/train" --test-split-interval 0 \
    --eval-dataset "$TEST" --viewer "$PROF" $ARM_FLAGS $MASK_FLAGS > "$F/train.log" 2>&1 || { tail -40 "$F/train.log"; exit 1; }
fi
grep -E '^\[vgs\] (held-out|[0-9]+ gaussians)' "$F/train.log" 2>/dev/null || true

echo "== 2. viewer-exact fine-tune (display scales ${DSCALE:-1.5:4}) $FT_ARGS"
# shellcheck disable=SC2086
[ -f "$F/ft.ply" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$F/train/splat.ply" --out "$F/ft.ply" \
  --dataset "$DATA" --test-split-interval 0 --viewer ss352 --display-scale "${DSCALE:-1.5:4}" \
  --iters "${FT_ITERS:-7000}" --eval-dataset "$TEST" $FT_ARGS $MASK_FLAGS > "$F/ft.log" 2>&1 || { tail -40 "$F/ft.log"; exit 1; }
grep -E 'held-out' "$F/ft.log" | tail -2 || true
cp "$F/ft.ply" "$E/$NAME.ply"

VARIANTS=""
if [ -f "$CLOUD" ]; then
  echo "== 3. unseen LiDAR surfaces: fill + inpaint"
  [ -d "$F/depth" ] || "$SPY" scripts/render_ply_depth.py --ply "$E/$NAME.ply" --dataset "$DATA" --out-dir "$F/depth"
  "$CPUPY" scripts/fill_unseen_surfaces.py --splat-ply "$E/$NAME.ply" --cloud "$CLOUD" --sim3-json "$SIM3" \
    --dataset "$DATA" --splat-depth-dir "$F/depth" --out-ply "$E/${NAME}_fill_raw.ply" | tail -3
  "$SPY" scripts/fill_harm_ply.py --base "$E/$NAME.ply" --filled "$E/${NAME}_fill_raw.ply" --dataset "$DATA" \
    --out "$E/${NAME}_complete.ply"
  "$IPY" scripts/fill_inpaint.py --splat-ply "$E/$NAME.ply" --fill-ply "$E/${NAME}_complete.ply" --sim3-json "$SIM3" \
    --method lama --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" --out-ply "$E/${NAME}_inpainted.ply" | tail -3
  VARIANTS="_inpainted"
  if [ "${CLEAN:-1}" = 1 ]; then
    echo "== 4. haze cleanup"
    rm -f "$F/clean_stats.npz"
    # shellcheck disable=SC2086
    "$SPY" scripts/clean_haze.py clean --ply "$E/${NAME}_inpainted.ply" --dataset "$DATA" --cloud "$CLOUD" \
      --sim3-json "$SIM3" --stats "$F/clean_stats.npz" --out "$E/${NAME}_clean.ply" ${CLEAN_ARGS:-} | tail -6
    VARIANTS="_inpainted _clean"
  fi
fi
if [ "${VIEW:-1}" = 1 ]; then
  echo "== 5. viewer-friendly copies (recentred, gravity -> SuperSplat up; scripts/view_frame.py)"
  for v in "" $VARIANTS; do
    "$CPUPY" scripts/view_frame.py --in-ply "$E/$NAME$v.ply" --out-ply "$E/$NAME${v}_view.ply" --dataset "$DATA" | grep -E "^\[view\] (up|[0-9])" || true
  done
fi
ls -la "$E/$NAME"*.ply
for v in "" $VARIANTS; do md5sum "$E/$NAME$v.ply"; done
echo FP2_EXPORT_DONE
if [ "${CHECK:-1}" = 1 ]; then
  PLYS=""; for v in "" $VARIANTS; do PLYS="$PLYS $NAME$v=$E/$NAME$v.ply"; done
  # shellcheck disable=SC2086
  PLYS="$PLYS" TEST_DATA="$TEST" bash sbatch/fp2_check.sbatch "$CFG"
fi
