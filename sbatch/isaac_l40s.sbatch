#!/bin/bash
# Phase 6 Isaac Sim on an L40S (RTX renderer needs RT cores) — docs/PLAN.md §3.1.
#   sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot drop
#   sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot walk
# Adjust partition/account to your Unity allocation before submitting.
#SBATCH -J cear-isaac
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
MODE="${2:-drop}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh"

OUT="$CEAR_OUT/$SEQ"
SPLAT="${SPLAT_USD:-$OUT/usd/scene_particlefield.usdz}"   # compare vs scene_nurec.usdz (§3.4)
COLLIDER="$OUT/collider/collider.ply"
ROBOT="${ROBOT_USD:-}"                                    # lab Mini Cheetah USD, else empty

# Isaac Sim's bundled python. Point ISAAC_PYTHON at your install's python.sh.
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to <isaac-sim>/python.sh}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N

"$ISAAC_PYTHON" "$REPO/scripts/compose_stage.py" \
  --splat-usd "$SPLAT" --collider "$COLLIDER" ${ROBOT:+--robot "$ROBOT"} --mode "$MODE"
echo "ISAAC_DONE $SEQ $MODE"
