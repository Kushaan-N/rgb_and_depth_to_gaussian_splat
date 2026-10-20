#!/bin/bash
# sbatch wrapper that records every submission in logs/jobs.tsv (time, job id, purpose, command), so the state of
# long-running experiments survives session restarts.  Usage (same args as sbatch; env vars pass through):
#   WHY="fiord v9 rig vs 5 views" scripts/sub.sh -A pi_donghyunkim_umass_edu -p gpu ... sbatch/omni_rect.sbatch cfg.yaml
# Prints the job id (like sbatch --parsable).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$REPO/logs"
J=$(sbatch --parsable "$@")
J=${J%%;*}
printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$J" "${WHY:--}" "sbatch $*" >> "$REPO/logs/jobs.tsv"
echo "$J"
