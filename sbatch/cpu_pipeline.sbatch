#!/bin/bash
# Run the CPU reconstruction pipeline (Phases 1-5: poses, depth, collider, COLMAP, dataset) for
# one sequence on a compute node — NO GPU. Used to prep a new scene before GPU training.
#   sbatch -A pi_donghyunkim_umass_edu sbatch/cpu_pipeline.sbatch mocap2_well-lit_trot
#SBATCH -J cear-cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:15:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:?usage: sbatch sbatch/cpu_pipeline.sbatch <sequence_name>}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
source env/cear_env.sh
echo "== CPU pipeline for $SEQ on $(hostname) =="
bash scripts/run_cpu_pipeline.sh --config "configs/${SEQ}.yaml"
echo "CPU_PIPELINE_DONE $SEQ"
