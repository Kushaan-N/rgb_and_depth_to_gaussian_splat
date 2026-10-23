#!/bin/bash
# Download an Oxford Spires sequence subset (scripts/datasets/spires_fetch.py; files listed in the config),
# sha256-verified + unzipped, on a CPU node (GB-scale; compute nodes have HTTPS).
#   CEAR_DATA=$MC/data/spires sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/spires_fetch.sbatch configs/spires_roq01.yaml
# Env: ONLY (space-separated substrings: fetch only matching files), EXTRA=1 (also download.files_extra)
#SBATCH -J spires-fetch
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 04:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
hostname; date
# shellcheck disable=SC2086
python scripts/datasets/spires_fetch.py "$CFG" ${ONLY:+--only $ONLY} ${EXTRA:+--extra}
du -sh "$(python scripts/cfg_get.py "$CFG" sequence.data_root)"
