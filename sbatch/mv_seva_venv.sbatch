#!/bin/bash
# Stable Virtual Camera (SEVA, Stability AI, research licence; gated HF weights) venv on scratch ($CEAR_WS/venv-seva) for
# scripts/object_priors/mv_seva.py: pose-conditioned novel views (real training crops + their real cameras in, orbit
# cameras out). Code: github Stability-AI/stable-virtual-camera at $CEAR_WS/seva (pinned commit). The SD2.1-base VAE it
# needs is no longer on stabilityai's HF page: mv_seva.py loads it from a mirror (--vae-repo). Weights are fetched
# separately (login node, single stream) into $HF_HOME; this job only installs + import-checks (offline).
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/mv_seva_venv.sbatch
#SBATCH -J mv-seva-venv
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
SV="${SEVA_VENV:-$CEAR_WS/venv-seva}"; SD="${SEVA_DIR:-$CEAR_WS/seva}"; COMMIT="${SEVA_COMMIT:-fe19948}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
hostname; date
[ -d "$SD/.git" ] || git clone https://github.com/Stability-AI/stable-virtual-camera.git "$SD"
git -C "$SD" checkout -q "$COMMIT"; echo "[seva] $(git -C "$SD" rev-parse --short HEAD)"
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$SV/bin/python" ] || "$UV" venv --python 3.11 "$SV"
"$UV" pip install --python "$SV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "numpy==1.24.4" roma tyro fire einops colorama splines kornia \
  "open-clip-torch==2.32.0" "diffusers==0.33.1" "transformers==4.46.3" "huggingface-hub==0.30.2" "imageio[ffmpeg]" \
  opencv-python-headless pillow scipy plyfile safetensors tqdm "gradio==5.17.0" viser
"$UV" pip install --python "$SV/bin/python" --no-deps -e "$SD"
HF_HUB_OFFLINE=1 "$SV/bin/python" -c "import torch, seva, open_clip; from seva.eval import run_one_scene; from seva.model import SGMWrapper; print('OK seva', torch.__version__, open_clip.__version__)"
echo MV_SEVA_VENV_DONE
