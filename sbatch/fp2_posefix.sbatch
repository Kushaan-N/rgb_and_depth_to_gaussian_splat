#!/bin/bash
# Photometric pose refinement for a FusionPortable v2 rig dataset (SfM jitter on texture-poor walls), then the
# fp2 export recipe on the refined poses. scripts/refine_poses_gsplat.py `refine` trains a stock gsplat model with a
# per-camera SE(3) correction (small Adam lr + weight decay, gauge removed) on the TRAINING frames of <P>/onpath/all
# (= train + test, held out by the k % 8 rule, checked by `fusionportable.py union`); held-out frames get the
# interpolated correction of their neighbouring training frames (no test pixels used). The refined set is resplit
# (fusionportable.py resplit) and passed to sbatch/fp2_export.sbatch as TRAIN_DATA / TEST_DATA (inline).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_posefix.sbatch configs/fp2_legged_room00.yaml
# Env: POSE_LR (1e-5), POSE_ITERS (15000), REFINE_ARGS (e.g. "--rig-group _([0-9]+)[.]png$" = one correction per
#      stereo frame), PF_TAG (refinement dir <P>/posefix<PF_TAG>), NAME (viewer_igs_posefix), + fp2_export.sbatch vars.
#SBATCH -J fp2-posefix
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 08:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GSPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
P="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)/pipeline"; D="$P/posefix${PF_TAG:-}"; mkdir -p "$D"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
[ -f "$P/onpath/all/sparse/0/images.txt" ] || "$CPUPY" scripts/datasets/fusionportable.py union "$CFG"
if [ ! -f "$D/full/posefix.json" ]; then
  # shellcheck disable=SC2086
  "$GSPY" scripts/refine_poses_gsplat.py refine --dataset "$P/onpath/all" --out "$D/full" --test-split-interval 8 \
    --iters "${POSE_ITERS:-15000}" --pose-lr "${POSE_LR:-1e-5}" --depth-lambda 0.05 --sim3-json "$P/sim3.json" ${REFINE_ARGS:-} \
    > "$D/refine.log" 2>&1 || { tail -40 "$D/refine.log"; exit 1; }
fi
grep -E "posefix\] step +14999|PSNR|psnr" "$D/refine.log" | tail -4 || true
"$CPUPY" - "$D/full/posefix.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for k in ("train_correction", "interp_correction", "gauge"):
    print(k, r.get(k))
for k in r:
    if "psnr" in k.lower() or "diag" in k.lower():
        print(k, r[k])
PY
"$CPUPY" scripts/datasets/fusionportable.py resplit "$CFG" --src "$D/full"
if [ ! -f "$D/.depth_rebuilt" ]; then                # LiDAR targets at the refined poses (writes into full/ copies)
  for ds in "$D/full_train" "$D/full_test"; do
    "$THREEDGRUT_DIR/.venv/bin/python" scripts/build_depth_targets.py --dataset "$ds" --cloud "$P/lidar_static.ply" \
      --sim3-json "$P/sim3.json" | tail -1
  done; touch "$D/.depth_rebuilt"
fi
NAME="${NAME:-viewer_igs_posefix}" TAGDIR="${TAGDIR:-supersplat_check_posefix}" TRAIN_DATA="$D/full_train" TEST_DATA="$D/full_test" \
  bash sbatch/fp2_export.sbatch "$CFG"
