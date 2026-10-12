#!/usr/bin/env bash
# Build the headless-viewer venv on scratch ($CEAR_WS/venv-viewer): Playwright + its Chromium, used by
# scripts/viewer_shots.py to render splats with SuperSplat's own engine (PlayCanvas gsplat) and to screenshot
# the SuperSplat editor. Browsers go to $CEAR_WS/playwright-browsers. Idempotent. Run in a job (~200 MB):
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:a100:1 -c 2 --mem=8G -t 00:30:00 env/make_viewer_venv.sh
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
VIEWER_VENV="${VIEWER_VENV:-$CEAR_WS/venv-viewer}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$PLAYWRIGHT_BROWSERS_PATH"

if "$VIEWER_VENV/bin/python" -c "import playwright" 2>/dev/null && ls "$PLAYWRIGHT_BROWSERS_PATH"/chromium* >/dev/null 2>&1; then
  echo "[viewer] venv ready at $VIEWER_VENV"
else
  UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
  "$UV" venv --python 3.11 "$VIEWER_VENV"
  "$UV" pip install --python "$VIEWER_VENV/bin/python" "playwright==1.48.0" "numpy<2" scipy pillow opencv-python-headless
  "$VIEWER_VENV/bin/python" -m playwright install chromium
fi
"$VIEWER_VENV/bin/python" -c "import playwright; print('[viewer] playwright ok')"
