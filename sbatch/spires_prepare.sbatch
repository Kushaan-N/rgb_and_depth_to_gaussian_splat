#!/bin/bash
# Oxford Spires sequence -> scan2sim datasets (scripts/datasets/spires.py: fisheye COLMAP splits with GT poses,
# LiDAR + TLS clouds in one metric frame, rectified pinhole variants, their benchmark images, export_viewer layout).
#   CEAR_DATA=$MC/data/spires CEAR_OUT=$MC/outputs sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out \
#     sbatch/spires_prepare.sbatch configs/spires_roq01.yaml
# Env: STAGES (default all: fisheye,lidar,tls,rect,bench,layout)
#SBATCH -J spires-prepare
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
hostname; date; which python
[ -f "$(python scripts/cfg_get.py "$CFG" sequence.data_root)/raw/LICENSE.md" ] || python scripts/datasets/spires_fetch.py "$CFG"
python scripts/datasets/spires.py "$CFG" ${STAGES:+--stages $STAGES}
