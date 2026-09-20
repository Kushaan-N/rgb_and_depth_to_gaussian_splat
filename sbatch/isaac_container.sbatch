#!/bin/bash
# Phase 6 in the full Isaac Sim 5.1 CONTAINER (has omni.rtx.spg + nurec_utils, which the pip
# install lacks) — renders the NuRec splat + drives a rover. L40S.
#   sbatch sbatch/isaac_container.sbatch mocap1_well-lit_trot nav
#SBATCH -J cear-isaacC
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
MODE="${2:-nav}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"
SIF="$CEAR_WS/isaac-sim-5.1.0.sif"
OUT="$CEAR_OUT/$SEQ"
FLOORZ="$(python3 -c "import json;print(json.load(open('$OUT/collider/collider_meta.json'))['floor_z_m'])")"

# writable dirs for the container (its /isaac-sim is read-only): HOME + omni caches on scratch
CHOME="$CEAR_WS/isaac-chome"; mkdir -p "$CHOME"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=N
echo "SIF=$SIF  seq=$SEQ mode=$MODE floor_z=$FLOORZ"
nvidia-smi --query-gpu=name --format=csv,noheader

ARGS="--collider $OUT/collider/collider.obj --mode $MODE --floor-z $FLOORZ --out-dir $OUT/isaac_container"
[ "$MODE" = "nav" ] && ARGS="$ARGS --splat-usd $OUT/usd/scene_nurec.usdz --cam-model $OUT/colmap/sparse/0"

# copy the script to scratch so the container never depends on /work being mounted, and
# force --pwd /isaac-sim (the host CWD on /work isn't cleanly mounted on compute nodes,
# which breaks python.sh's `cd /isaac-sim`).
cp "$REPO/scripts/compose_stage.py" "$CEAR_WS/compose_stage.py"
apptainer exec --nv --pwd /isaac-sim \
  --bind "$CEAR_WS:$CEAR_WS" --home "$CHOME:$HOME" \
  "$SIF" /isaac-sim/python.sh "$CEAR_WS/compose_stage.py" $ARGS
echo "ISAAC_CONTAINER_DONE $SEQ $MODE"
