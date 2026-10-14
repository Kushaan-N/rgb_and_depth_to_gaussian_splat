#!/bin/bash
# Viewer-first deliverable: every stage optimises what a standard viewer (SuperSplat) draws.
#   1. train from scratch in a rasteriser matched to the viewer (scripts/train_viewer_gsplat.py; EDGS init on ALL
#      frames, Improved-GS 4M, random background so nothing is see-through) -> <F>/train/splat.ply
#   2. viewer-exact fine-tune at random display scales (scripts/finetune_ply_viewer.py, SuperSplat 3.5.2 profile)
#   3. every LiDAR surface no camera supports, filled (scripts/fill_unseen_surfaces.py; visibility from this splat's
#      own depth/opacity, scripts/render_ply_depth.py), checked in the viewer-exact rasteriser so no training view
#      changes (scripts/fill_harm_ply.py), and LaMa-inpainted (scripts/fill_inpaint.py)
#   4. exports in the viewer reference's axes: <out>/exports/<NAME>{,_refaxes}.ply
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/export_viewer.sbatch configs/<seq>.yaml
# Env: NAME (viewer_igs), ARM_FLAGS (train_viewer_gsplat.py flags; default the best held-out arm igs4m_pc274),
#      PROF (kernel build pc274|ss352), DSCALE (fine-tune display scales, 1.5:4), FT_ITERS (7000),
#      FT_ARGS (extra fine-tune flags, e.g. "--lambda-wd 0.025 --wd-log2-sigma 4" = perceptual WD-R),
#      TRAIN_PLY (reuse an already trained stage-1 splat instead of training),
#      EDGS_DATA (stage-1 training set; e.g. densify_frames.sbatch DENSIFY_ARGS=--include-test -> <DF>/edgs_data
#      = all frames + extra low-blur frames, with ARM_FLAGS "... --iters 120000").
#SBATCH -J export-viewer
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; NAME="${NAME:-viewer_igs}"; F="$P/$NAME"
DATA="$P/onpath/data_orig"; EDGS="${EDGS_DATA:-$P/igs_full/edgs_data}"; FRAME="$(get pipeline.export_frame_model "")"
PROF="${PROF:-pc274}"; ARM_FLAGS="${ARM_FLAGS:---strategy improved --budget 4000000}"
[ -f "$EDGS/edgs_init.json" ] || { echo "missing $EDGS (run export_igs.sbatch ARMS=edgs_igs once)"; exit 1; }
hostname; date; mkdir -p "$F" "$E"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
bash env/make_gsplat_viewer_venv.sh

echo "== 1. viewer-matched training ($PROF kernels): $ARM_FLAGS"
if [ -n "${TRAIN_PLY:-}" ] && [ ! -f "$F/train/splat.ply" ]; then mkdir -p "$F/train"; cp "$TRAIN_PLY" "$F/train/splat.ply"; fi
if [ ! -f "$F/train/splat.ply" ]; then
  mkdir -p "$F/train"
  # shellcheck disable=SC2086
  PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
    "$SPY" scripts/train_viewer_gsplat.py --dataset "$EDGS" --out "$F/train" --test-split-interval 0 \
    --eval-dataset "$P/onpath/test" --viewer "$PROF" $ARM_FLAGS > "$F/train.log" 2>&1 || { tail -40 "$F/train.log"; exit 1; }
fi
grep -E '^\[vgs\] (held-out|[0-9]+ gaussians)' "$F/train.log" 2>/dev/null || true

echo "== 2. viewer-exact fine-tune (display scales ${DSCALE:-1.5:4})"
[ -f "$F/ft.ply" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$F/train/splat.ply" --out "$F/ft.ply" \
  --dataset "$DATA" --test-split-interval 0 --viewer ss352 --display-scale "${DSCALE:-1.5:4}" \
  --iters "${FT_ITERS:-7000}" --eval-dataset "$P/onpath/test" ${FT_ARGS:-} > "$F/ft.log" 2>&1 || { tail -40 "$F/ft.log"; exit 1; }
cp "$F/ft.ply" "$E/$NAME.ply"

echo "== 3. unseen LiDAR surfaces: fill + inpaint"
[ -d "$F/depth" ] || "$SPY" scripts/render_ply_depth.py --ply "$E/$NAME.ply" --dataset "$DATA" --out-dir "$F/depth"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
"$CPUPY" scripts/fill_unseen_surfaces.py --splat-ply "$E/$NAME.ply" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
  --dataset "$DATA" --splat-depth-dir "$F/depth" --out-ply "$E/${NAME}_fill_raw.ply" | tail -3
# measured, in the viewer itself: drop fill gaussians that change any training view (e.g. seen through windows)
"$SPY" scripts/fill_harm_ply.py --base "$E/$NAME.ply" --filled "$E/${NAME}_fill_raw.ply" --dataset "$DATA" \
  --out "$E/${NAME}_complete.ply"
"$IPY" scripts/fill_inpaint.py --splat-ply "$E/$NAME.ply" --fill-ply "$E/${NAME}_complete.ply" --sim3-json "$P/sim3.json" \
  --method lama --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" --out-ply "$E/${NAME}_inpainted.ply" | tail -3

echo "== 4. exports in the viewer reference's axes"
for v in "" _inpainted; do
  if [ -n "$FRAME" ]; then
    "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
      --in-ply "$E/$NAME$v.ply" --out-ply "$E/$NAME${v}_refaxes.ply"
  fi
done
ls -la "$E/$NAME"*.ply
echo EXPORT_VIEWER_DONE
