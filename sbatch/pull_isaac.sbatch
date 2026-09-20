#!/bin/bash
# Pull an Isaac Sim image to a SIF on a COMPUTE node (dedicated I/O; far faster than the
# shared login node for the slow rootless extract+squashfs of a huge image). Reuses any
# layers already in ~/.apptainer/cache, so a partial login-node pull isn't wasted.
#   sbatch sbatch/pull_isaac.sbatch 6.1.0
#SBATCH -J cear-pull
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
TAG="${1:-6.1.0}"
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
SIF="$CEAR_WS/isaac-sim-${TAG}.sif"

echo "node=$(hostname)  tag=$TAG  ->  $SIF"
# node-local tmp is fastest for the conversion; fall back to scratch if it's too small.
LOCAL_TMP="${SLURM_TMPDIR:-/tmp}/isaacpull.${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL_TMP"
avail_kb=$(df -Pk "$LOCAL_TMP" | awk 'NR==2{print $4}')
if [ "${avail_kb:-0}" -lt 62914560 ]; then    # need ~60 GB free for extract+squashfs
  echo "node-local tmp only $((avail_kb/1024/1024))G free; using scratch tmp instead"
  LOCAL_TMP="$CEAR_WS/.pulltmp.${SLURM_JOB_ID:-$$}"; mkdir -p "$LOCAL_TMP"
fi
export APPTAINER_TMPDIR="$LOCAL_TMP" TMPDIR="$LOCAL_TMP"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$HOME/.apptainer/cache}"   # reuse downloaded layers
echo "APPTAINER_TMPDIR=$APPTAINER_TMPDIR  CACHEDIR=$APPTAINER_CACHEDIR"
df -h "$LOCAL_TMP" | tail -1

# quieter output; login is already cached in ~/.docker/config.json from the user's registry login.
APPTAINER_MESSAGELEVEL=2 apptainer pull --force "$SIF" "docker://nvcr.io/nvidia/isaac-sim:${TAG}"
rc=$?
rm -rf "$LOCAL_TMP" 2>/dev/null || true
echo "PULL_DONE rc=$rc"; ls -lh "$SIF" 2>&1 | tail -1
