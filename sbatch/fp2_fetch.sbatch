#!/bin/bash
# Download + unpack a FusionPortable v2 sequence subset (scripts/datasets/fusionportable_fetch.py; files listed in
# the config) on a CPU node (GB-scale; compute nodes have HTTPS).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_fetch.sbatch configs/fp2_legged_room00.yaml
# Env: FP2_WS (multicam workspace, default /scratch4/workspace/$USER-multicam), ONLY (substrings), KEEP_7Z=1
#SBATCH -J fp2-fetch
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 04:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
hostname; date
# shellcheck disable=SC2086
python scripts/datasets/fusionportable_fetch.py "$CFG" ${ONLY:+--only $ONLY} $([ "${KEEP_7Z:-0}" = 1 ] && echo --keep-7z)
du -sh "$(python scripts/cfg_get.py "$CFG" sequence.data_root)"
