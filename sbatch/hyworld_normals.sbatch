#!/bin/bash
# HY-World 2.0 normal priors for the depth-to-normal loss (scripts/methods/hyworld.py, bench tokens hyn / hynm): builds
# $CEAR_WS/venv-hyworld (env/make_hyworld_venv.sh), then scripts/hyworld_normals.py on the bench's training frames
# ($P/edgs/on/data, every 8th held out) with WorldMirror 2.0 (our poses + intrinsics + LiDAR depth as priors) ->
# $P/hyworld/normals_wm and MoGe-2 (the official 3DGS stage's normal model) -> $P/hyworld/normals_moge2; each with
# _normals.json (median angle to the LiDAR-depth normals, coverage) and _preview.png (rgb | prior | LiDAR normals).
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/hyworld_normals.sbatch configs/mocap2_well-lit_trot.yaml
#SBATCH -J hyworld-normals
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010,gpu035
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
OUT="$("$CEAR_VENV/bin/python" scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"; DS="${HY_DATASET:-$P/edgs/on/data}"
export HYWORLD_DIR="${HYWORLD_DIR:-$CEAR_WS/HY-World-2.0}" HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
hostname; date; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
bash env/make_hyworld_venv.sh | tail -3
HPY="${HYWORLD_VENV:-$CEAR_WS/venv-hyworld}/bin/python"; which "$HPY"; "$HPY" --version
for src in ${HY_SOURCES:-worldmirror moge2}; do
  d="$P/hyworld/normals_$([ "$src" = worldmirror ] && echo wm || echo moge2)"
  [ -f "$d/_normals.json" ] || "$HPY" scripts/hyworld_normals.py --dataset "$DS" --out "$d" --source "$src" --test-split-interval "${TSI:-8}" \
    || echo "[$src] FAILED"
  cat "$d/_normals.json" 2>/dev/null || true
done
echo HYWORLD_NORMALS_SBATCH_DONE
