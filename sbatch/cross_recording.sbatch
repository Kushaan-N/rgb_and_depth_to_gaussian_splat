#!/bin/bash
# Unseen object sides from OTHER recordings of the same room, per object (scripts/object_priors/cross_recording.py):
#   0. dynamic-free LiDAR cloud of every other recording (remove_dynamic_lidar.py + clear_robot_path.py, in parallel)
#      -> <X>/clouds/<seq>/lidar_static.ply (skipped where the recording already has one)
#   1. register: per reference object x other recording, floor-plane ICP (yaw + slide), accept / reject
#   2. views: faint (badly seen) surface of each object, other recordings' frames that see it -> masked extra views
#   3. xfit: object-local fine-tune with the extra views (held-out extra views scored before / after)
#   4. sbatch/object_priors.sbatch stages 3-5 on the result (REFIT=0, BUDGET_MODE=zone: GT-aware budget in the object
#      zones, strict elsewhere; shfit, refaxes, live SuperSplat gates in both sort modes, orbit montages, off-path)
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/cross_recording.sbatch configs/mocap2_well-lit_trot.yaml
# Env: NAME (viewer_msd60k_wd_clean_ft2_solid3), IN (<E>/<NAME>.ply), OTHERS (pipeline dirs; default: every other
#      $CEAR_OUT/<seq>/pipeline with sim3.json + dataset, no fused ones), BASE (object_priors build dir whose
#      textured.ply/meta/zone/src to start from; default: none = the input itself, zone = object boxes + ZONE_M),
#      TAG (xrec), REG_ARGS, VIEW_ARGS, XFIT_ARGS, STOP_AFTER (views | xfit), FORCE=1 (redo stages 1-3; stage 4 only redoes
#      what is missing: use a new TAG for a new candidate).
#SBATCH -J xrec
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
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"
NAME="${NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; IN="${IN:-$E/$NAME.ply}"; TAG="${TAG:-xrec}"
D="$P/object_priors/$NAME"; X="$P/object_priors/xrec"; T="$D/$TAG"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; XR=scripts/object_priors/cross_recording.py
needt() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$X/clouds" "$T"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
[ -f "$D/objects.json" ] || { echo "no objects ($D/objects.json): run sbatch/object_priors.sbatch first"; exit 0; }
if [ -z "${OTHERS:-}" ]; then
  OTHERS=""; for p in "$CEAR_OUT"/*/pipeline; do
    [ "$(realpath "$p")" = "$(realpath "$P")" ] && continue; [ -f "$p/fuse.json" ] && continue
    [ -f "$p/sim3.json" ] && [ -d "$p/dataset/sparse/0" ] && OTHERS="$OTHERS $p"
  done
fi
echo "== $NAME TAG=$TAG others:$OTHERS"

echo "== 0. dynamic-free clouds of the other recordings"
for p in $OTHERS; do
  seq="$(basename "$(dirname "$p")")"; c="$X/clouds/$seq"; cfg="configs/$seq.yaml"
  [ -f "$p/offpath/depth/lidar_static.ply" ] || [ -f "$c/lidar_static.ply" ] || [ ! -f "$cfg" ] || [ ! -f "$p/lidar_cloud.ply" ] && continue
  mkdir -p "$c"
  ( OMP_NUM_THREADS=2 "$CPUPY" scripts/remove_dynamic_lidar.py --config "$cfg" --cloud "$p/lidar_cloud.ply" --out "$c/lidar_nodyn.ply" \
      > "$c/nodyn.log" 2>&1 && "$CPUPY" scripts/clear_robot_path.py --cloud "$c/lidar_nodyn.ply" \
      --gt-model "$(dirname "$p")/colmap_train/sparse/0" --out "$c/lidar_static.ply" >> "$c/nodyn.log" 2>&1; tail -2 "$c/nodyn.log" ) &
done
wait
P_STATIC="$P/offpath/depth/lidar_static.ply"; [ -f "$P_STATIC" ] || echo "WARN: reference has no lidar_static.ply"

echo "== 1. register"
needt "$X/register.json" && "$CPUPY" "$XR" register --objects "$D/objects" --ref-pipeline "$P" --others $OTHERS \
  --cloud-dir "$X/clouds" --out-dir "$X" ${REG_ARGS:-}
echo "== 2. views"
needt "$X/inventory.json" && "$CPUPY" "$XR" views --objects "$D/objects" --ref-pipeline "$P" --ref-dataset "$DATA" \
  --cloud-dir "$X/clouds" --out-dir "$X" ${VIEW_ARGS:-}
[ "${STOP_AFTER:-}" = views ] && { echo XREC_VIEWS_ONLY; exit 0; }

echo "== 3. object-local fine-tune with the extra views"
if [ -n "${BASE:-}" ]; then
  for f in meta.npz zone.npz priors.json; do cp "$BASE/$f" "$T/$f"; done; XIN="$BASE/textured.ply"
else
  needt "$T/meta.npz" && "$CPUPY" "$XR" base --ply "$IN" --objects "$D/objects" --sim3-json "$P/sim3.json" --out-dir "$T"; XIN="$IN"
fi
needt "$T/textured.ply" && "$SPY" "$XR" xfit --ply "$XIN" --meta "$T/meta.npz" --zone "$T/zone.npz" --dataset "$DATA" "$TEST" \
  --xviews "$X/xviews" --out "$T/textured.ply" ${XFIT_ARGS:-}
cp "$T/textured_src.npz" "$T/src.npz"
[ "${STOP_AFTER:-}" = xfit ] && { echo XREC_XFIT_ONLY; exit 0; }

echo "== 4. zone budget, shfit, exports, live checks (object_priors.sbatch, REFIT=0)"
NAME="$NAME" IN="$IN" TAG="$TAG" REFIT=0 BUDGET_MODE="${BUDGET_MODE:-zone}" FORCE=0 bash sbatch/object_priors.sbatch "$CFG"
echo XREC_DONE
