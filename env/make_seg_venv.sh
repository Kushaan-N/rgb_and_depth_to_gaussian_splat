#!/usr/bin/env bash
# Build the image-segmentation venv on scratch ($CEAR_WS/venv-seg) for scripts/object_layers/detect_objects.py:
# torch + transformers (>= 4.56: Grounding-DINO and SAM 2.1 are both native transformers models) + OpenCV / scipy /
# plyfile, and pre-fetch the two open checkpoints into $HF_HOME ($CEAR_WS/hf-cache, never $HOME):
#   IDEA-Research/grounding-dino-base  (Apache-2.0, text-prompted boxes)
#   facebook/sam2.1-hiera-large        (Apache-2.0, box-prompted masks)
# Idempotent. Downloads ~4 GB; run in a job (compute nodes have outbound HTTPS):
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 4 --mem=16G -t 01:00:00 env/make_seg_venv.sh
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
SEG_VENV="${SEG_VENV:-$CEAR_WS/venv-seg}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" HF_HUB_OFFLINE=0
mkdir -p "$CEAR_WS"

if "$SEG_VENV/bin/python" -c "import torch, cv2, plyfile, scipy; from transformers import Sam2Model, AutoModelForZeroShotObjectDetection" 2>/dev/null; then
  echo "[seg] venv ready at $SEG_VENV"
else
  UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
  "$UV" venv --python 3.11 "$SEG_VENV"
  "$UV" pip install --python "$SEG_VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "transformers>=4.56,<5" accelerate "numpy<2" scipy plyfile pillow \
    "opencv-python-headless<4.11" huggingface_hub
fi
"$SEG_VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for r in ("IDEA-Research/grounding-dino-base", "facebook/sam2.1-hiera-large"):
    print("[seg]", r, "->", snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
import torch, transformers; print("[seg] torch", torch.__version__, "transformers", transformers.__version__)
PY
