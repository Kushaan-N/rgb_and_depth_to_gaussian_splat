#!/bin/bash
# Fisheye / 360 datasets, arm (b) "train on the native camera, export the standard PLY": the adapter's native/ sets
# (written by `scripts/datasets/<adapter>.py prepare`, also run here if missing) are trained without rectification:
#   fisheye  (sequence.camera: fisheye)  3DGUT (stock $THREEDGRUT_DIR, apps/colmap_3dgut.yaml): OPENCV_FISHEYE
#            projected exactly by the unscented transform; <image>_mask.png limits the loss to native.max_deg
#   equirect (sequence.camera: equirect) ODGS (NeurIPS 2024; venv env/make_odgs_venv.sh): per-gaussian tangent-plane
#            projection on the sphere, trained on the panoramas with the official train/test lists
# then (1) the dataset-native benchmark: the method's own renders of the held-out native frames scored against them,
# (2) the exported PLY as-is and (3) the same PLY after the viewer-exact SuperSplat 3.5.2 fine-tune on the rectified
# training views (scripts/finetune_ply_viewer.py ss352: native geometry, viewer-matched appearance), both scored in
# the live SuperSplat on the rectified held-out views (sbatch/omni_score.sbatch, same views as sbatch/omni_rect.sbatch).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/omni-native-%j.out sbatch/omni_native.sbatch configs/<seq>.yaml
# Env: ITERS (30000), APP (3DGUT app config, apps/colmap_3dgut.yaml), DSCALE (1.5:4), FT_ITERS (7000), SCORE (1)
#SBATCH -J omni-native
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
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"
OPY="${ODGS_VENV:-$CEAR_WS/venv-odgs}/bin/python"; ODGS_DIR="${ODGS_DIR:-$CEAR_WS/ODGS}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ADAPTER="$(get sequence.adapter)"; CAM="$(get sequence.camera)"
ITERS="${ITERS:-30000}"; REP="$P/native_report"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
"$CPUPY" "scripts/datasets/$ADAPTER.py" fetch "$CFG"
mkdir -p "$P"; (                                                   # rect + native jobs may start together
  flock 9; [ -f "$P/meta.json" ] && [ -f "$P/native/train/sparse/0/images.txt" ] && { [ "$CAM" != equirect ] || [ -f "$P/native/all/pcd.ply" ]; } || \
    "$CPUPY" "scripts/datasets/$ADAPTER.py" prepare "$CFG" --workers "${SLURM_CPUS_PER_TASK:-12}" ) 9>"$P/.prepare.lock"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"                      # slangc for the 3DGRUT kernels

case "$CAM" in
fisheye)
  NAME=native_3dgut; F="$P/$NAME"; CK="$(ls -t "$F"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== 3DGUT on the native fisheye frames ($ITERS iters)"
    ( cd "$THREEDGRUT_DIR" && "$GPY" train.py --config-name "${APP:-apps/colmap_3dgut.yaml}" path="$P/native/train" out_dir="$F" \
        experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 ) \
      > "$F.log" 2>&1 || { tail -40 "$F.log"; exit 1; }
    CK="$(ls -t "$F"/run/*/ckpt_last.pt | head -1)"
  fi
  echo "== dataset-native benchmark: 3DGUT renders of the held-out fisheye frames"
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$P/native/test" --tag "$NAME" --json-out "$REP/$NAME.json" \
    --save-dir "$F/test_renders" | tail -1
  [ -f "$F/native.ply" ] || "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$F/native.ply"
  ;;
equirect)
  NAME=native_odgs; F="$P/$NAME"
  bash env/make_odgs_venv.sh
  if [ ! -f "$F/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
    echo "== ODGS on the native equirect frames ($ITERS iters)"
    ( cd "$ODGS_DIR" && "$OPY" train.py -s "$P/native/all" -m "$F" --eval -r 1 --iterations "$ITERS" ) > "$F.log" 2>&1 \
      || { tail -40 "$F.log"; exit 1; }
  fi
  echo "== dataset-native benchmark: ODGS renders of the held-out panoramas (official test list)"
  [ -d "$F/test/iters_$ITERS/renders" ] || ( cd "$ODGS_DIR" && "$OPY" render.py -m "$F" --skip_train --iteration "$ITERS" ) \
    > "$F/render.log" 2>&1 || { tail -20 "$F/render.log"; exit 1; }
  "$GPY" scripts/eval_views.py --pred-dir "$F/test/iters_$ITERS/renders" --gt-dir "$F/test/iters_$ITERS/gt" --tag "$NAME" \
    --json-out "$REP/$NAME.json" | tail -1
  ln -sfn "$F/point_cloud/iteration_$ITERS/point_cloud.ply" "$F/native.ply"
  ;;
*) echo "sequence.camera must be fisheye|equirect (got '$CAM')"; exit 1 ;;
esac

echo "== viewer-exact fine-tune of the native PLY on the rectified training views (display scales ${DSCALE:-1.5:4})"
bash env/make_gsplat_viewer_venv.sh
[ -f "$F/native_ft.ply" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$F/native.ply" --out "$F/native_ft.ply" \
  --dataset "$P/rect/train" --test-split-interval 0 --viewer ss352 --display-scale "${DSCALE:-1.5:4}" \
  --iters "${FT_ITERS:-7000}" --eval-dataset "$P/rect/test" > "$F/ft.log" 2>&1 || { tail -40 "$F/ft.log"; exit 1; }

if [ "${SCORE:-1}" = 1 ]; then
  echo "== live SuperSplat scoring (held-out rectified views, off-path, editor)"
  bash sbatch/omni_score.sbatch "$CFG" "$NAME=$F/native.ply" "${NAME}_ft=$F/native_ft.ply"
fi
echo OMNI_NATIVE_DONE
