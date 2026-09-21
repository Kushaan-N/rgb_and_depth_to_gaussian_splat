#!/bin/bash
# Isolation test: render our exported NuRec splat with NVIDIA's OWN standalone example
# (isaac-sim 6.x, /isaac-sim/standalone_examples/nurec/nurec_render.py) at recorded on-path
# camera poses. If this renders -> the asset + renderer are fine and our interactive capture
# loop is the bug; if it's black -> the exported .nurec asset itself is the problem.
#   CEAR_SIF=.../isaac-sim-6.1.0.sif sbatch -A pi_donghyunkim_umass_edu sbatch/nurec_render_test.sbatch mocap1_well-lit_trot
#SBATCH -J cear-nrtest
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
TUMNAME="${2:-onpath_cam.tum}"          # trajectory file in $CEAR_WS
OUTNAME="${3:-nurec_render_test}"       # output subdir under the sequence
CFG="${4:-}"                            # optional render config (warmup override) in $CEAR_WS
RES="${RES:-1280x720}"                   # render resolution (higher = sharper)
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"
SIF="${CEAR_SIF:-$CEAR_WS/isaac-sim-6.1.0.sif}"
OUT="$CEAR_OUT/$SEQ"
USDZ="$OUT/usd/scene_nurec.usdz"
TUM="$CEAR_WS/$TUMNAME"
OUTDIR="$OUT/$OUTNAME"
CFGARG=""; [ -n "$CFG" ] && CFGARG="--config $CEAR_WS/$CFG"
mkdir -p "$OUTDIR"

CHOME="$CEAR_WS/isaac-chome-$(basename "$SIF" .sif)"; mkdir -p "$CHOME" "$CHOME/kit-cache" "$CHOME/kit-data"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=N
echo "SIF=$SIF  stage=$USDZ  tum=$TUM  out=$OUTDIR"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --nv --fakeroot --writable-tmpfs --pwd /isaac-sim \
  --home "$CHOME:/root" \
  --bind "$CEAR_WS:$CEAR_WS" \
  --bind "$CHOME/kit-cache:/isaac-sim/kit/cache" \
  --bind "$CHOME/kit-data:/isaac-sim/kit/data" \
  "$SIF" /isaac-sim/python.sh /isaac-sim/standalone_examples/nurec/nurec_render.py poses \
    --stage "$USDZ" --output "$OUTDIR" --cameras camera_0 --poses "$TUM" --resolution "$RES" $CFGARG
echo "NUREC_RENDER_TEST_DONE"; find "$OUTDIR" -name "*.png" | wc -l; find "$OUTDIR" -name "*.png" -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -3
