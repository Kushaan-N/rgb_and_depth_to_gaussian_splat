#!/bin/bash
# Check a viewer-frame export (scripts/view_frame.py) in the live SuperSplat: the editor's default view (screenshot)
# and the held-out rectified views moved into the same frame (scored, so the export must match its source PLY).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/omni-view-%j.out sbatch/omni_view_check.sbatch <view.ply> <view_test dataset> <out dir>
#SBATCH -J omni-view
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
set -euo pipefail
PLY="$1"; DS="$2"; O="$3"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$REPO"
source env/cear_env.sh >/dev/null
GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
bash env/make_viewer_venv.sh >/dev/null; mkdir -p "$O"
"$VPY" scripts/viewer_shots.py --engine current --ply "$PLY" --dataset "$DS" --out-dir "$O/onpath" --editor --bg black
"$GPY" scripts/eval_views.py --pred-dir "$O/onpath" --dataset "$DS" --tag view --json-out "$O/view.json" | tail -1
echo OMNI_VIEW_DONE
