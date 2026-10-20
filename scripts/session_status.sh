#!/bin/bash
# One-screen project status for the start of a (resumed) session — login-node safe, read-only:
# scratch workspace expiry (warns < 7 days), my running/pending SLURM jobs, jobs that ended in the last 24 h
# (failures first), and the tail of the job ledger written by scripts/sub.sh.
#   bash scripts/session_status.sh            (also run by the SessionStart hook)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "== scan2sim status $(date '+%F %H:%M')"
if command -v ws_list >/dev/null 2>&1; then
  ws_list -v 2>/dev/null | awk '/^id:/{id=$2} /remaining time/ && !/days/{next}
    /remaining time  /{d=$4; printf "  workspace %-10s %s days left%s\n", id, d, (d+0<7 ? "  <-- EXTEND: ws_extend " id " 30" : "")}'
fi
if command -v squeue >/dev/null 2>&1; then
  echo "== my jobs (squeue)"
  squeue -u "$USER" -h -o "  %.10i %.24j %.9T %.10M %R" 2>/dev/null | head -40
  echo "== ended in last 24 h (sacct; failures first)"
  sacct -u "$USER" -S "$(date -d '-24 hours' '+%FT%T')" -X -n -P -o JobID,JobName,State,Elapsed 2>/dev/null \
    | awk -F'|' '$3!~/RUNNING|PENDING/{f=($3~/COMPLETED/)?1:0; printf "%d  %-12s %-24s %-12s %s\n", f, $1, $2, $3, $4}' \
    | sort -k1,1n | cut -c4- | head -40
fi
L="$REPO/logs/jobs.tsv"
if [ -f "$L" ]; then echo "== job ledger (last 15; scripts/sub.sh)"; tail -n 15 "$L" | sed 's/^/  /'; fi
