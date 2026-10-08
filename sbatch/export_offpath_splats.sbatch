#!/bin/bash
# Export the off-path experiment splats of one sequence to .ply (COLMAP frame, like pipeline/splat.ply)
# into <out>/exports/. These are benchmark splats: the held-out frames were NOT trained on.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/export_offpath_splats.sbatch configs/<seq>.yaml
#SBATCH -J export-splats
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"; APY="$AFX_VENV/bin/python"
OUT="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)"; O="$OUT/pipeline/offpath"; E="$OUT/exports"; SEQ="$("$CPUPY" scripts/cfg_get.py "$CFG" sequence.name)"
mkdir -p "$E"; hostname; date
newest() { ls -t $1 2>/dev/null | head -1; }
# name | python | checkpoint glob
while IFS='|' read -r name py glob; do
  ck="$(newest "$glob")"
  if [ -z "$ck" ]; then echo "-- $name: no checkpoint ($glob)"; continue; fi
  if [ "$py" = afx ]; then
    module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true; export PATH="$AFX_VENV/bin:$PATH"; PYX="$APY"
  else
    module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"; PYX="$GPY"
  fi
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  echo "== $name <- $ck"
  "$PYX" scripts/export_gaussian_ply.py --checkpoint "$ck" --out "$E/$name.ply"
done <<LIST
worldmodel_oursplat_visibility|afx|$O/artifixer_ours_covis/prep/$SEQ/artifixer3d/runs/$SEQ/$SEQ/ours_30000/ckpt_30000.pt
worldmodel_ownsplat|afx|$O/artifixer/prep/$SEQ/artifixer3d/runs/$SEQ/$SEQ/ours_30000/ckpt_30000.pt
worldmodel_oursplat_visibility_depthdistill|ours|$O/artifixer_ours_covis/distill_depth/run/*/ckpt_last.pt
depthloss_l0.05|ours|$O/work/depth_l0.05/run/*/ckpt_last.pt
LIST
ls -la "$E"
echo "EXPORT_DONE"
