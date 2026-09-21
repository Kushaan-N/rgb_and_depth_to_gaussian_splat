#!/bin/bash
# LIVE interactive Isaac Sim session on the NuRec splat world (WebRTC streaming).
# Prints the SSH tunnel + client URL you need to connect. Runs until you cancel or walltime.
#   CEAR_SIF=.../isaac-sim-6.1.0.sif sbatch -A pi_donghyunkim_umass_edu sbatch/isaac_live.sbatch mocap1_well-lit_trot
#SBATCH -J cear-live
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"
SIF="${CEAR_SIF:-$CEAR_WS/isaac-sim-6.1.0.sif}"
OUT="$CEAR_OUT/$SEQ"
USDZ="$OUT/usd/scene_nurec.usdz"
NODE="$(hostname)"

CHOME="$CEAR_WS/isaac-chome-$(basename "$SIF" .sif)"; mkdir -p "$CHOME" "$CHOME/kit-cache" "$CHOME/kit-data"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=N
cp "$REPO/scripts/live_stage.py" "$CEAR_WS/live_stage.py"

cat <<EOF
============================================================
 LIVE Isaac Sim session starting on node: $NODE
 To connect from your laptop:
   1) SSH tunnel (new terminal):
        ssh -N -L 8211:$NODE:8211 -L 8111:$NODE:8111 ${USER}@unity.rc.umass.edu
   2) Open the Isaac Sim WebRTC Streaming Client (or the browser client) and connect to:
        127.0.0.1:8211
   (WebRTC also uses UDP media ports; if video won't start over the tunnel, see docs/RUN.md
    'Live session' for the salloc-on-a-reachable-node / TURN notes.)
 Booting (first frame ~1-2 min)…
============================================================
EOF
nvidia-smi --query-gpu=name --format=csv,noheader

# expose the streaming ports from the container (apptainer shares host network by default)
apptainer exec --nv --fakeroot --writable-tmpfs --pwd /isaac-sim \
  --home "$CHOME:/root" \
  --bind "$CEAR_WS:$CEAR_WS" \
  --bind "$CHOME/kit-cache:/isaac-sim/kit/cache" \
  --bind "$CHOME/kit-data:/isaac-sim/kit/data" \
  "$SIF" /isaac-sim/python.sh "$CEAR_WS/live_stage.py" --stage "$USDZ"
echo "ISAAC_LIVE_DONE"
