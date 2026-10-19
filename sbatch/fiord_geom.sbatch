#!/bin/bash
# FIORD geometry vs the Faro TLS scan for every splat already scored: re-estimate the scan <- model Sim3
# (scripts/datasets/fiord.py align, venv-e57) then accuracy / completeness / F at 5 cm per PLY
# (fiord.py geom) -> <out>/pipeline/supersplat/report/<name>_geom.json (overwrites omni_score's).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/fiord-geom-%j.out sbatch/fiord_geom.sbatch configs/fiord_meetingroom.yaml
#SBATCH -J fiord-geom
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$REPO"
export CEAR_DATA="${CEAR_DATA:-/scratch4/workspace/${USER}-multicam/data}" CEAR_OUT="${CEAR_OUT:-/scratch4/workspace/${USER}-multicam/outputs}"
source env/cear_env.sh >/dev/null
EPY="${E57_VENV:-$CEAR_WS/venv-e57}/bin/python"; P="$("$CEAR_VENV/bin/python" scripts/cfg_get.py "$CFG" paths.out_root)/pipeline"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"; hostname; date
[ -n "${KEEP_ALIGN:-}" ] || rm -f "$P/sim3_tls.json"
[ -f "$P/sim3_tls.json" ] || "$EPY" scripts/datasets/fiord.py align "$CFG" 2>&1 | grep -v WARNING | tail -15
for ply in "$P"/rect_igs*/train/splat.ply "$P"/rect_igs*/ft.ply "$P"/native_3dgut*/native.ply "$P"/native_3dgut*/native_ft.ply; do
  [ -f "$ply" ] || continue
  d="$(basename "$(dirname "$ply")")"; b="$(basename "$ply" .ply)"
  case "$b" in splat) n="${d%/train}"; n="$(basename "$(dirname "$(dirname "$ply")")")_train" ;; ft) n="${d}_ft" ;; native) n="$d" ;; native_ft) n="${d}_ft" ;; esac
  "$CEAR_VENV/bin/python" scripts/datasets/fiord.py geom "$CFG" --ply "$ply" --json-out "$P/supersplat/report/${n}_geom.json" | tail -1
done
echo FIORD_GEOM_DONE
