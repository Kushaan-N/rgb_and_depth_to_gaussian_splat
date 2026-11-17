#!/bin/bash
# Mesh-teacher splat training with every fix from the diagnosis (scripts/object_layers/diag_pseudo.py, sort_gap.py):
#   - mesh pseudo-views (object_layers.py meshviews: cleaned targets) + LiDAR IBR views, both kept until the END of training
#   - the trainer starts with gaussians on every mesh's unseen surface (object_layers.py seedpoints)
#   - both viewer-exact fine-tunes see the same pseudo-views (finetune_ply_viewer.py --weight-dir) and use --sort dual
#   - no solid / solid3 stage (the mesh teacher replaces it)
# ARM=mesh (all of the above) | ctrl (matched control: IBR views only, no seeds, same everything else).
#   prep   (once, ARM=mesh) mesh pseudo-views -> $MP/mesh2 (+ contact.jpg), pseudo / fine-tune datasets, seeded EDGS copies
#   train  scripts/train_distill_gsplat.py, ITERS iterations on a SUBSET of the frames
#   ft ft2 viewer-exact fine-tunes (FT_ITERS each, dual sort, pseudo-views at weight W)
#   diag   masked PSNR at the mesh pseudo-views per stage, depth vs radial gap per stage, live SuperSplat on-path both sorts
#   WHY=... ARM=mesh STAGES="prep train ft ft2 diag" scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/mesh_teacher.sbatch configs/<seq>.yaml
# Env: ARM, STAGES, SUBSET (0.3), ITERS (15000), FT_ITERS (2000), MESH_W (0.3), IBR_W (0.3), RUN (cheap), NAME (the layers source),
#      MTAG (v9r), ARM_FLAGS (the deliverable recipe minus --iters), FT_ARGS.
#SBATCH -J mesh-teacher
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010,gpu035
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 04:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; TPY="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}/bin/python"
GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ARM="${ARM:?ARM=mesh|ctrl}"; RUN="${RUN:-cheap}"; MTAG="${MTAG:-v9r}"
NAME="${NAME:-viewer_ibr70k_wd_clean_ft2_solid3_floor}"; T="$P/object_priors/$NAME/layers_$MTAG"; MP="$P/mesh_pseudo/$MTAG"; R="$MP/${RUN}_$ARM"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; EDGS="$P/igs_full/edgs_data"; IBR="$P/ibr_views/ibr"; OL=scripts/object_layers/object_layers.py
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
SUBSET="${SUBSET:-0.3}"; ITERS="${ITERS:-15000}"; FT_ITERS="${FT_ITERS:-2000}"; MW="${MESH_W:-0.3}"; IW="${IBR_W:-0.3}"
ARM_FLAGS="${ARM_FLAGS:---strategy improved --budget 4000000 --msssim-lambda 0.2 --sort dual --depth-lambda 0.1}"
FT_ARGS="${FT_ARGS:---lambda-wd 0.025 --wd-log2-sigma 4}"; STAGES="${STAGES:-prep train ft ft2 diag}"; has() { [[ " $STAGES " == *" $1 "* ]]; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$R" "$MP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true; module list 2>&1 | tail -1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
echo "== $RUN ARM=$ARM subset $SUBSET iters $ITERS ft $FT_ITERS mesh $MW ibr $IW stages: $STAGES"; echo "python: $SPY ($("$SPY" --version 2>&1))"
SEED=$([ "$ARM" = mesh ] && echo "" || echo "--no-seed"); SRC=("$IBR:$IW"); [ "$ARM" = mesh ] && SRC=("$MP/mesh2:$MW" "$IBR:$IW")
PS="$MP/pseudo_$ARM"; FD="$MP/ftds_$ARM"; ED="$MP/edgs_${RUN}_$ARM"
if has prep; then
  [ "$ARM" != mesh ] || [ -f "$MP/mesh2/pseudo.json" ] || "$TPY" "$OL" meshviews --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" \
    --dataset "$DATA" --map "$T/floor_map_inpainted.npz" --cloud "$CLOUD" --out "$MP/mesh2"
  [ -f "$PS/pseudo.json" ] || "$TPY" "$OL" mergeviews --dir "$T" --sources "${SRC[@]}" --pseudo-weight "$MW" --out "$PS"
  [ -f "$FD/pseudo.json" ] || "$TPY" "$OL" mergeviews --dir "$T" --real "$DATA" --sources "${SRC[@]}" --pseudo-weight "$MW" --out "$FD"
  # shellcheck disable=SC2086
  [ -f "$ED/sparse/0/points3D.txt" ] || "$TPY" "$OL" seedpoints --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --edgs "$EDGS" \
    --subset "$( [ "$RUN" = cheap ] && echo "$SUBSET" || echo 1.0 )" $SEED --out "$ED"
fi
bash env/make_gsplat_viewer_venv.sh | tail -1
if has train && [ ! -f "$R/train/splat.ply" ]; then
  mkdir -p "$R/train"
  # shellcheck disable=SC2086
  PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-pc274" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-pc274" \
    "$SPY" scripts/train_distill_gsplat.py --pseudo-dataset "$PS" --pseudo-weight "$MW" --pseudo-soft --pseudo-depth --pseudo-until 1.0 -- \
    --dataset "$ED" --out "$R/train" --test-split-interval 0 --eval-dataset "$TEST" --viewer pc274 $ARM_FLAGS --iters "$ITERS" > "$R/train.log" 2>&1 \
    || { tail -30 "$R/train.log"; exit 1; }
fi
grep -E '^\[vgs\] (held-out|[0-9]+ gaussians)|DISTILL_TRAIN_DONE' "$R/train.log" 2>/dev/null | cut -c1-400 || true
ft() {  # <in> <out> <log>
  # shellcheck disable=SC2086
  [ -f "$2" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$1" --out "$2" --dataset "$FD" --test-split-interval 0 --viewer ss352 \
    --display-scale "${DSCALE:-1.5:4}" --iters "$FT_ITERS" --eval-dataset "$TEST" --sort dual --weight-dir "$FD/images" --weight-scale "$MW" $FT_ARGS > "$3" 2>&1 \
    || { tail -30 "$3"; exit 1; }
  grep -E "weight maps|eval \{" "$3" | sed -n '1p;2p;$p'
}
has ft && ft "$R/train/splat.ply" "$R/ft.ply" "$R/ft.log"
has ft2 && ft "$R/ft.ply" "$R/ft2.ply" "$R/ft2.log"
if has diag; then
  PLYS=(train="$R/train/splat.ply" ft="$R/ft.ply" ft2="$R/ft2.ply")
  [ -f "$R/diag/diag.json" ] || "$SPY" scripts/object_layers/diag_pseudo.py --pseudo "$MP/mesh2" --layers "$T" --sim3-json "$P/sim3.json" --out "$R/diag" \
    --ply "${PLYS[@]}" | grep -E "\[diag\]" | cut -c1-600
  [ -f "$R/sort_gap.json" ] || "$SPY" scripts/object_layers/sort_gap.py --dataset "$TEST" --json-out "$R/sort_gap.json" --ply "${PLYS[@]}"
  bash env/make_viewer_venv.sh | tail -1
  for m in depth radial; do
    [ -f "$R/live_$m/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --sort-mode $m --ply "$R/ft2.ply" --dataset "$TEST" \
      --out-dir "$R/live_$m" | tail -1
    [ -f "$R/live_$m.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$R/live_$m" --dataset "$TEST" --tag "${ARM}_$m" --json-out "$R/live_$m.json" | tail -1
  done
fi
date; echo MESH_TEACHER_DONE
