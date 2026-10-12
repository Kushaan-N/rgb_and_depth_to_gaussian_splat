#!/usr/bin/env bash
# Build the texture-inpainting venv on scratch ($CEAR_WS/venv-inpaint) for scripts/fill_inpaint.py: torch (LaMa,
# big-lama TorchScript, Apache-2.0, fetched into $CEAR_WS/torch-hub/lama) + OpenCV-contrib (classical shift-map
# fallback) + plyfile/scipy. Idempotent. Downloads ~3 GB; run in a job:
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 4 --mem=16G -t 01:00:00 env/make_inpaint_venv.sh
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
INPAINT_VENV="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}"
LAMA_PT="${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}"
LAMA_URL="${LAMA_URL:-https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$(dirname "$LAMA_PT")"

if "$INPAINT_VENV/bin/python" -c "import torch, cv2, plyfile, scipy; cv2.xphoto" 2>/dev/null; then
  echo "[inpaint] venv ready at $INPAINT_VENV"
else
  UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
  "$UV" venv --python 3.11 "$INPAINT_VENV"
  "$UV" pip install --python "$INPAINT_VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124" "numpy<2" scipy plyfile pillow "opencv-contrib-python-headless<4.11"
fi
if [ ! -s "$LAMA_PT" ]; then                                    # TorchScript export of big-lama (Apache-2.0)
  curl -L --fail -o "$LAMA_PT.part" "$LAMA_URL" && mv "$LAMA_PT.part" "$LAMA_PT"
fi
sha256sum "$LAMA_PT"
"$INPAINT_VENV/bin/python" -c "import torch, cv2; cv2.xphoto; print('[inpaint] torch', torch.__version__, 'cv2', cv2.__version__)"
