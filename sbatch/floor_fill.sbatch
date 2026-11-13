#!/bin/bash
# A proper floor where no camera looked (scripts/floor_fill_ibr.py), on a finished deliverable PLY:
#   1. top-down floor texture from the training frames (scripts/make_lidar_ibr_views.py views --floor-map: the LiDAR IBR
#      on the fitted floor plane, 1 cm cells) -> $P/ibr_views/floor_map.npz
#   2. LaMa continues the texture into the floor cells no frame saw (inpaint venv)
#   3. unsupported gaussians in those cells' column removed (measured restore), one flat disc per unseen cell appended,
#      measured harm check (scripts/fill_harm_ply.py: no training view changes) -> <E>/<NAME>_floor{,_refaxes}.ply
#   4. live SuperSplat on-path score (both Sort Modes) of input and output -> $P/floor_fill/<NAME>/report/
#   NAME=viewer_ibr60k_wd_clean_ft2_solid3 WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/floor_fill.sbatch configs/<seq>.yaml
# Env: NAME (input <E>/<NAME>.ply), TAG (floor: output suffix), INPAINT_ARGS, DATA ($P/onpath/data_orig), APPLY_ARGS, MAP_ARGS, SCORE (1).
#SBATCH -J floor-fill
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010,gpu035
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
: "${NAME:?}"; IN="$E/$NAME.ply"; DATA="${DATA:-$P/onpath/data_orig}"; TEST="$P/onpath/test"; TAG="${TAG:-floor}"; W="$P/floor_fill/${NAME}_${TAG}"; IV="$P/ibr_views"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
hostname; date; mkdir -p "$W/report" "$IV" logs; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
echo "python: $SPY ($("$SPY" --version 2>&1))"; nvidia-smi --query-gpu=name --format=csv,noheader || true
[ -f "$IV/scene.json" ] || "$CPUPY" scripts/make_lidar_ibr_views.py floor --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" --out "$IV/scene.json"
# shellcheck disable=SC2086
[ -f "$IV/floor_map.npz" ] || "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
  --scene-json "$IV/scene.json" --floor-map "$IV/floor_map.npz" --out "$IV/_floor_map_tmp" ${MAP_ARGS:-} 2>&1 | grep -E "^\[ibr\]|IBR_FLOOR_MAP|Error|Traceback" | cut -c1-600
# shellcheck disable=SC2086
"$CPUPY" -c "import numpy as np,sys; sys.exit(0 if 'maybe' in np.load(sys.argv[1]).files else 1)" "$IV/floor_map_inpainted.npz" 2>/dev/null \
  || "$IPY" scripts/floor_fill_ibr.py inpaint --map "$IV/floor_map.npz" --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" ${INPAINT_ARGS:-} | tail -2
# shellcheck disable=SC2086
[ -f "$W/filled.ply" ] || "$SPY" scripts/floor_fill_ibr.py apply --ply "$IN" --map "$IV/floor_map_inpainted.npz" --dataset "$DATA" \
  --out-base "$W/base.ply" --out-filled "$W/filled.ply" ${APPLY_ARGS:-} | grep -E "floorfill|FLOOR_FILL" | cut -c1-900
[ -f "$E/${NAME}_${TAG}.ply" ] || "$SPY" scripts/fill_harm_ply.py --base "$W/base.ply" --filled "$W/filled.ply" --dataset "$DATA" \
  --out "$E/${NAME}_${TAG}.ply" | tail -4
[ -z "$FRAME" ] || [ -f "$E/${NAME}_${TAG}_refaxes.ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
  --gt-model "$FRAME" --in-ply "$E/${NAME}_${TAG}.ply" --out-ply "$E/${NAME}_${TAG}_refaxes.ply" | tail -1
if [ "${SCORE:-1}" = 1 ]; then
  bash env/make_viewer_venv.sh | tail -1
  for v in "in:$IN" "floor:$E/${NAME}_${TAG}.ply"; do
    l="${v%%:*}"; f="${v#*:}"
    for m in depth radial; do
      [ -f "$W/shots_${l}_$m/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --sort-mode "$m" --ply "$f" \
        --dataset "$TEST" --out-dir "$W/shots_${l}_$m" 2>&1 | tail -1
      [ -f "$W/report/${l}_$m.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$W/shots_${l}_$m" --dataset "$TEST" --tag "${l}_$m" \
        --json-out "$W/report/${l}_$m.json" | tail -1 | cut -c1-300
    done
  done
fi
ls -la "$E/${NAME}_${TAG}"*.ply; date; echo FLOOR_FILL_DONE
