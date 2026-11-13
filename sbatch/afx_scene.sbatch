#!/bin/bash
# One round of scene-wide GENERATIVE repair (scripts/afx_scene.py): trajectories leaving the path (LEVEL 1..3) ->
# the splat rendered along them (viewer-exact) + seen masks -> ArtiFixer world model (context = covisible real training
# frames; keeps what a camera saw, generates the rest) -> LiDAR IBR at the selected frames -> composed pseudo-dataset
# <W>/pseudo (IBR real pixels where verified, repaired / generated elsewhere at lower weight, LiDAR depth targets) for
# scripts/train_distill_gsplat.py --pseudo-soft (sbatch/offpath_bench.sbatch PSEUDO_EXTRA=<W>/pseudo, or
# sbatch/export_viewer.sbatch PSEUDO_DS).  Needs an 80 GB A100 for the world model (the ArtiFixer venv has no
# flash-attention build for Hopper GPUs: H100 / H200 fail).
#   PLY=<splat.ply> DATA=<training frames> W=<out dir> LEVEL=1 WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT \
#       -o logs/%x-%j.out sbatch/afx_scene.sbatch configs/<seq>.yaml
# Env: PLY, DATA ($P/onpath/data_orig), W, LEVEL (1), SCENE_JSON ($P/ibr_views/scene.json; floor + sim3), PATH_ARGS,
#      PREP_ARGS, AFX_ARGS, EVERY (8), COMPOSE_ARGS, IBR_ARGS ("--depth-where all"), CAPTION.
#SBATCH -J afx-scene
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH -t 03:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"; AFX_DIR="${AFX_DIR:-$CEAR_WS/ArtiFixer}"; APY="$AFX_VENV/bin/python"
export CHECKPOINT_PT="${CHECKPOINT_PT:-$CEAR_WS/artifixer-checkpoints/artifixer-1.3b.pt}"
export MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; SEQ="$(get sequence.name)"
: "${PLY:?}" "${W:?}"; DATA="${DATA:-$P/onpath/data_orig}"; LEVEL="${LEVEL:-1}"; SJ="${SCENE_JSON:-$P/ibr_views/scene.json}"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
CAP="${CAPTION:-$P/offpath/artifixer/prep/$SEQ/captions/$SEQ/caption.h5}"
[ -f "$CAP" ] || { echo "missing caption $CAP (sbatch/cosmetic_afx.sbatch prep writes one)"; exit 1; }
AS=scripts/afx_scene.py; mkdir -p "$W" logs
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; echo "== afx_scene level $LEVEL: $PLY -> $W"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
gs_env() { module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"; }
gs_env
# shellcheck disable=SC2086
[ -f "$W/paths.json" ] || "$CPUPY" "$AS" paths --dataset "$DATA" --cloud "$CLOUD" --scene-json "$SJ" --level "$LEVEL" --out "$W" ${PATH_ARGS:-}
# shellcheck disable=SC2086
[ -f "$W/afx/split.json" ] || env PATH="$(dirname "$SPY"):$PATH" "$SPY" "$AS" prep --ply "$PLY" --dataset "$DATA" --paths "$W" --scene-json "$SJ" \
  --caption "$CAP" --seq "$SEQ" ${PREP_ARGS:-} | grep -vE "Warning|warn" | tail -30
date
if [ ! -d "$W/repaired" ] || [ -z "$(ls "$W/repaired" 2>/dev/null | head -1)" ]; then
  ( module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
    # shellcheck disable=SC2086
    cd "$AFX_DIR" && "$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
      --model_id "$MODEL_ID" --save_dir "$W/afx_run" --split_path "$W/afx/split.json" --render_trajectory trajectory \
      --neighbor_selection_mode covisibility --save_frame_outputs_only ${AFX_ARGS:-} ) > "$W/afx_run.log" 2>&1 || { tail -40 "$W/afx_run.log"; exit 1; }
  "$CPUPY" "$AS" collect --paths "$W" --afx-run "$W/afx_run" --seq "$SEQ"
fi
date; gs_env
"$CPUPY" "$AS" select --paths "$W" --every "${EVERY:-8}"
# shellcheck disable=SC2086
[ -f "$W/ibr/ibr.json" ] || "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
  --scene-json "$SJ" --poses-from "$W/sel" --out "$W/ibr" ${IBR_ARGS:---depth-where all} 2>&1 | grep -E "^IBR_VIEWS|Error|Traceback" | cut -c1-200
# shellcheck disable=SC2086
"$CPUPY" "$AS" compose --paths "$W" --ibr "$W/ibr" --out "$W/pseudo" ${COMPOSE_ARGS:-}
date; echo "AFX_SCENE_DONE $W/pseudo"
