#!/bin/bash
# Phase 6 Isaac Sim on an L40S (RTX cores) — docs/PLAN.md §9. Uses the pip-installed
# isaacsim 5.0 venv (env/make_isaac_venv.sh), not a bundled python.sh.
#   sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot drop   # Gate 5 (physics only)
#   sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot nav    # drive a wheeled robot in the splat
#SBATCH -J cear-isaac
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
MODE="${2:-drop}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true   # use the isaac venv, not CPU venv
module load cuda/12.4.1 2>/dev/null || true

ISAAC_PY="${ISAAC_PY:-$CEAR_WS/venv-isaac/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N
export XDG_CACHE_HOME="$CEAR_WS/xdg-cache" OMNI_CACHE_DIR="$CEAR_WS/omni-cache"
mkdir -p "$XDG_CACHE_HOME" "$OMNI_CACHE_DIR"

OUT="$CEAR_OUT/$SEQ"
COLLIDER="$OUT/collider/collider.obj"
SPLAT="${SPLAT_USD:-$OUT/usd/scene_nurec.usdz}"           # NuRec is what Isaac 5.0 renders
FLOORZ="$($ISAAC_PY - <<PY
import json;print(json.load(open("$OUT/collider/collider_meta.json"))["floor_z_m"])
PY
)"
echo "seq=$SEQ mode=$MODE floor_z=$FLOORZ"
nvidia-smi --query-gpu=name --format=csv,noheader

ARGS=(--collider "$COLLIDER" --mode "$MODE" --floor-z "$FLOORZ" --out-dir "$OUT/isaac")
[ "$MODE" = "nav" ] && ARGS+=(--splat-usd "$SPLAT" --cam-model "$OUT/colmap/sparse/0")
"$ISAAC_PY" "$REPO/scripts/compose_stage.py" "${ARGS[@]}"
echo "ISAAC_DONE $SEQ $MODE"
