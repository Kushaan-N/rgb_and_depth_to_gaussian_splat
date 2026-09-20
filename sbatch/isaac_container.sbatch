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
# NuRec (photoreal Gaussian-volume) rendering needs Isaac Sim >= 6.0.0 (added in 6.0.0). The
# 5.1.0 image renders only the reconstructed GEOMETRY (dome-lit collider + rover). Point at a
# 6.0 SIF via CEAR_SIF to get the photoreal splat once it's pulled.
SIF="${CEAR_SIF:-$CEAR_WS/isaac-sim-5.1.0.sif}"
OUT="$CEAR_OUT/$SEQ"
FLOORZ="$(python3 -c "import json;print(json.load(open('$OUT/collider/collider_meta.json'))['floor_z_m'])")"

# writable dirs: the SIF is read-only, but Isaac writes to $HOME and to /isaac-sim/kit/{cache,
# data} (shader DB, material cache) AND — new — downloads the NuRec extensions (omni.rtx.spg,
# isaacsim.replicator.nurec_utils) from NVIDIA's registry into $HOME/.local/share/ov. Bind HOME
# to scratch via `--home` (APPTAINERENV_HOME is silently rejected by apptainer) so those
# downloads persist across runs; --writable-tmpfs is the catch-all for any other write path.
CHOME="$CEAR_WS/isaac-chome"; mkdir -p "$CHOME" "$CHOME/kit-cache" "$CHOME/kit-data"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=N
echo "SIF=$SIF  seq=$SEQ mode=$MODE floor_z=$FLOORZ"
nvidia-smi --query-gpu=name --format=csv,noheader

ARGS="--collider $OUT/collider/collider.obj --mode $MODE --floor-z $FLOORZ --out-dir $OUT/isaac_container"
[ "$MODE" = "nav" ] && ARGS="$ARGS --splat-usd $OUT/usd/scene_nurec.usdz --cam-model $OUT/colmap/sparse/0"

# Run from scratch (never depend on /work being mounted on compute nodes). --fakeroot is
# REQUIRED: compute nodes run apptainer rootless, and the container's /isaac-sim is 0750
# owned by a non-root uid -> only a root-mapped (fakeroot) namespace can chdir into it.
# --home binds our scratch dir AS $HOME=/root so registry downloads + configs persist.
cp "$REPO/scripts/compose_stage.py" "$CEAR_WS/compose_stage.py"
apptainer exec --nv --fakeroot --writable-tmpfs --pwd /isaac-sim \
  --home "$CHOME:/root" \
  --bind "$CEAR_WS:$CEAR_WS" \
  --bind "$CHOME/kit-cache:/isaac-sim/kit/cache" \
  --bind "$CHOME/kit-data:/isaac-sim/kit/data" \
  "$SIF" /isaac-sim/python.sh "$CEAR_WS/compose_stage.py" $ARGS
echo "ISAAC_CONTAINER_DONE $SEQ $MODE"
