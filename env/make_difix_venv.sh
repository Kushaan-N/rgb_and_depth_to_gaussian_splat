#!/usr/bin/env bash
# Build the Difix venv on scratch (NVIDIA Difix3D+, CVPR 2025: single-step diffusion that removes
# rendering artifacts from novel views). Kept separate from 3DGRUT's venv because Difix pins old
# diffusers/transformers. Downloads ~6 GB (torch + model weights): run as a CPU job,
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 4 --mem=16G -t 01:00:00 env/make_difix_venv.sh
#
# Code + weights are NVIDIA-licensed for non-commercial (research/evaluation) use only.
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
DIFIX_VENV="${DIFIX_VENV:-$CEAR_WS/venv-difix}"
DIFIX_DIR="${DIFIX_DIR:-$CEAR_WS/Difix3D}"
DIFIX_COMMIT="${DIFIX_COMMIT:-c76edc595586e16732c91ddee82f3a6d83a8a9cc}"   # nv-tlabs/Difix3D main, 2026-10
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$HF_HOME"
hostname; date

if [ ! -d "$DIFIX_DIR/.git" ]; then
  git clone https://github.com/nv-tlabs/Difix3D.git "$DIFIX_DIR"
fi
git -C "$DIFIX_DIR" fetch -q origin && git -C "$DIFIX_DIR" checkout -q "$DIFIX_COMMIT"
echo "[difix] code at $DIFIX_DIR @ $(git -C "$DIFIX_DIR" rev-parse --short HEAD)"

UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
"$UV" venv --python 3.11 "$DIFIX_VENV"
# Difix's requirements.txt minus training-only extras (wandb, xformers, lpips, imageio); torch is
# the same build 3DGRUT already runs on Unity's A100 nodes.
"$UV" pip install --python "$DIFIX_VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" einops pillow "numpy<2" \
  "peft==0.9.0" "diffusers==0.25.1" "huggingface-hub==0.25.1" "transformers==4.38.0"

echo "[difix] fetching weights (nvidia/difix_ref + its SD-Turbo components) into $HF_HOME"
"$DIFIX_VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("nvidia/difix_ref",):
    print(repo, "->", snapshot_download(repo))
PY

echo "[difix] verifying the pipeline loads (CPU)"
PYTHONPATH="$DIFIX_DIR/src" "$DIFIX_VENV/bin/python" - <<'PY'
import torch, diffusers, transformers
from pipeline_difix import DifixPipeline
pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
print("OK torch", torch.__version__, "| diffusers", diffusers.__version__, "| transformers", transformers.__version__,
      "| pipeline", type(pipe).__name__)
PY
echo "MAKE_DIFIX_VENV_DONE"
