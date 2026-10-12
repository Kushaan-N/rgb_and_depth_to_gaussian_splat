#!/bin/bash
# Motion-blur diagnosis (scripts/blur_diagnosis.py): predicted blur per frame from the pose track + LiDAR depth,
# exposure time fitted from directional image sharpness. CPU only; output $P/blur/.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/blur-diag-%j.out sbatch/blur_diagnosis.sbatch configs/<seq>.yaml
#SBATCH -J blur-diag
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:40:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
hostname; date; export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
CPUPY="$CEAR_VENV/bin/python"; P="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)/pipeline"
USED="$P/onpath/data_orig"; [ -d "$USED/images" ] || USED="$P/dataset"        # the frames used (with depth targets)
"$CPUPY" scripts/blur_diagnosis.py --config "$CFG" --used-dir "$USED" ${BLUR_ARGS:-}
