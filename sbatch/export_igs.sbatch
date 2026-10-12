#!/bin/bash
# Deliverable splat with Improved-GS densification (patches/3dgrut-improved-densify.patch; best on-path arm of
# sbatch/densify_bench.sbatch): trained on ALL frames, exported in the COLMAP frame and the metric frame.
#   igs   Improved-GS, 1M budget                       -> <out>/exports/igs*.ply
#   igs_d Improved-GS, 2M budget + LiDAR depth loss    -> <out>/exports/igs_depth*.ply
#   edgs_igs  EDGS dense-correspondence init (scripts/edgs_init.py, all frames) + Improved-GS 4M -> edgs_igs*.ply
# then the floor fill (infill_floor.py, as the pipeline's post stage) -> <name>_filled.ply (COLMAP frame),
# <name>_filled_metric.ply (metric) and <name>_filled_refaxes.ply (pipeline.export_frame_model's axes); then
# every other LiDAR-measured surface no camera supports (fill_unseen_surfaces.py) -> <name>_complete*.ply.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/export_igs.sbatch configs/<seq>.yaml
# Env: ARMS ("igs igs_d").
#SBATCH -J export-igs
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"; LAMBDA="${LAMBDA:-0.05}"
export TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}"   # same kernels as densify_bench
export PYTHONPATH="$DDIR${PYTHONPATH:+:$PYTHONPATH}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; GT="$OUT/colmap_train/sparse/0"; E="$OUT/exports"
DATA="$P/onpath/data_orig"; F="$P/igs_full"; ITERS="$(get pipeline.train_iterations 30000)"
[ -f "$DATA/sparse/0/images.txt" ] || { echo "missing $DATA (run onpath_ablation.sbatch first)"; exit 1; }
hostname; date; mkdir -p "$F" "$E" "$TORCH_EXTENSIONS_DIR"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
bash env/make_3dgrut_densify.sh

for arm in ${ARMS:-igs igs_d}; do
  case "$arm" in
    igs)   BUDGET=1000000; DEPTH=false; NAME=igs ;;
    igs_d) BUDGET=2000000; DEPTH=true;  NAME=igs_depth ;;
    edgs_igs) BUDGET=4000000; DEPTH=false; NAME=edgs_igs ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  TRAIN="$DATA"
  if [ "$arm" = edgs_igs ]; then                                    # EDGS dense-correspondence init, all frames
    TRAIN="$F/edgs_data"
    [ -f "$TRAIN/edgs_init.json" ] || "${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python" scripts/edgs_init.py \
      --dataset "$DATA" --out "$TRAIN" --test-split-interval 0 --sim3-json "$P/sim3.json" > "$F/edgs_init.log" 2>&1 \
      || { tail -40 "$F/edgs_init.log"; exit 1; }
  fi
  RUN="$F/$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== $arm: Improved-GS budget $BUDGET, depth loss $DEPTH, all frames"
    ( cd "$DDIR" && "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$TRAIN" out_dir="$RUN" \
        experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
        loss.use_depth="$DEPTH" loss.lambda_depth="$LAMBDA" strategy.budget="$BUDGET" ) > "$RUN.log" 2>&1 \
      || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  if [ ! -f "$E/$NAME.ply" ]; then
    # sanity: fit to the training frames (held-out quality is in densify_bench's report)
    "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$DATA" --tag "$arm" --json-out "$F/$arm.json" | tail -1
    "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$E/$NAME.ply"
  fi
  # floor no camera faced is pruned in training: put back the LiDAR-measured floor, coloured from the real
  # pixels that saw it (or the nearest reconstructed floor) — the pipeline's post stage (infill_floor.py)
  if [ "$(get pipeline.floor_fill 1)" = 1 ]; then
    "$CPUPY" scripts/infill_floor.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$E/$NAME.ply" \
      --sim3-json "$P/sim3.json" --gt-model "$GT" --images "$OUT/precond/$(get sequence.rgb_dir rgb)" \
      --out-ply "$E/${NAME}_filled.ply" | tail -3
  else
    cp "$E/$NAME.ply" "$E/${NAME}_filled.ply"
  fi
  # every other surface the LiDAR measured but no camera supports (object backs, walls behind the path):
  # fill it, coloured from neighbouring observed surfaces, except where a training camera sees it
  ls "$F/${arm}_depth"/*_splatopacity.npy >/dev/null 2>&1 || "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$DATA" --no-metrics \
    --save-depth-dir "$F/${arm}_depth" | tail -1
  CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
  "$CPUPY" scripts/fill_unseen_surfaces.py --splat-ply "$E/${NAME}_filled.ply" --cloud "$CLOUD" \
    --sim3-json "$P/sim3.json" --dataset "$DATA" --splat-depth-dir "$F/${arm}_depth" --out-ply "$E/${NAME}_complete.ply"
  FRAME="$(get pipeline.export_frame_model "")"                     # deliver in the viewer reference's axes
  for v in filled complete; do
    "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
      --in-ply "$E/${NAME}_$v.ply" --out-ply "$E/${NAME}_${v}_metric.ply"
    if [ -n "$FRAME" ]; then
      "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
        --in-ply "$E/${NAME}_$v.ply" --out-ply "$E/${NAME}_${v}_refaxes.ply"
    fi
  done
done
ls -la "$E"/igs*.ply
echo "EXPORT_IGS_DONE"
