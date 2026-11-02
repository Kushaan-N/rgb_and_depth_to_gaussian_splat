#!/bin/bash
# Learned object prior: TRELLIS image-to-3D (stage 1 + 2) of every free-standing object from its best training views
# (scripts/object_priors/learned_prior.py generate, venv-trellis) -> $P/object_priors/<NAME>/learned_gen/, then
# object_priors.sbatch with LEARNED=<that dir> aligns / gates it. Needs object_priors.sbatch stage 0 (objects, depth).
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/learned_prior.sbatch configs/<seq>.yaml
# Env: NAME (viewer_msd60k_wd_clean_ft2_solid3), GEN_ARGS (e.g. "--views 4 --seeds 0 1 2").
#SBATCH -J learned-prior
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; TPY="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}/bin/python"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-hub}"
OUT="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
NAME="${NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; D="$P/object_priors/$NAME"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
[ -f "$D/objects.json" ] && [ -d "$D/depth" ] || { echo "run object_priors.sbatch stage 0 first"; exit 1; }
"$TPY" scripts/object_priors/learned_prior.py generate --objects "$D/objects" --sim3-json "$P/sim3.json" \
  --dataset "$P/onpath/data_orig" --splat-depth-dir "$D/depth" --out-dir "$D/learned_gen" ${GEN_ARGS:-}
ls -la "$D/learned_gen"; echo LEARNED_PRIOR_DONE
