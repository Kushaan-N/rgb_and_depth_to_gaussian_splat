#!/bin/bash
# Render a robot navigating the photoreal NuRec splat world (Isaac Sim 6.x container).
# Reuses NVIDIA's nurec_utils render machinery + injects a moving robot (scripts/nav_in_splat.py).
#   CEAR_SIF=.../isaac-sim-6.1.0.sif sbatch -A pi_donghyunkim_umass_edu \
#     sbatch/nav_in_splat.sbatch mocap1_well-lit_trot walkthrough.tum navsplat
#SBATCH -J cear-navsplat
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
TUMNAME="${2:-walkthrough.tum}"
OUTNAME="${3:-navsplat}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"
SIF="${CEAR_SIF:-$CEAR_WS/isaac-sim-6.1.0.sif}"
OUT="$CEAR_OUT/$SEQ"
USDZ="$OUT/usd/scene_nurec.usdz"
TUM="$CEAR_WS/$TUMNAME"
OUTDIR="$OUT/$OUTNAME"
mkdir -p "$OUTDIR"
shift $(( $# < 3 ? $# : 3 )) || true
EXTRA="$*"                                   # pass-through: --robot-drop, --robot-count, ...

CHOME="$CEAR_WS/isaac-chome-$(basename "$SIF" .sif)"; mkdir -p "$CHOME" "$CHOME/kit-cache" "$CHOME/kit-data"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=N
echo "SIF=$SIF stage=$USDZ tum=$TUM out=$OUTDIR extra='$EXTRA'"
nvidia-smi --query-gpu=name --format=csv,noheader
cp "$REPO/scripts/nav_in_splat.py" "$CEAR_WS/nav_in_splat.py"

apptainer exec --nv --fakeroot --writable-tmpfs --pwd /isaac-sim \
  --home "$CHOME:/root" \
  --bind "$CEAR_WS:$CEAR_WS" \
  --bind "$CHOME/kit-cache:/isaac-sim/kit/cache" \
  --bind "$CHOME/kit-data:/isaac-sim/kit/data" \
  "$SIF" /isaac-sim/python.sh "$CEAR_WS/nav_in_splat.py" \
    --stage "$USDZ" --tum "$TUM" --output "$OUTDIR" --warmup 300 $EXTRA
echo "NAV_IN_SPLAT_DONE"; find "$OUTDIR" -name "*.png" | wc -l
