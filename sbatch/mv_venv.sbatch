#!/bin/bash
# Multi-view generative-prior venv on scratch ($CEAR_WS/venv-mv) for scripts/object_priors/mv_gen.py:
#   sudo-ai/zero123plus-v1.2 (+ its diffusers custom pipeline sudo-ai/zero123plus-pipeline): 6 novel views of an object
#     from one image (azimuth +30..+330 relative, elevation 20/-10 absolute, fov 30 deg); weights CC-BY-NC (research)
#   ZhengPeng7/BiRefNet (MIT): object masks of the real crops and of the generated views
#   prs-eth/marigold-normals-v1-1 (openrail++): normal maps of the generated views (shape prior for mv_refit.py)
# Ungated weights only (SV3D / Stable Virtual Camera / SAM 3D are gated). Pure pip wheels, ~8 GB with weights.
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/mv_venv.sbatch
#SBATCH -J mv-venv
#SBATCH -p cpu,cpu-preempt
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:30:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
MV_VENV="${MV_VENV:-$CEAR_WS/venv-mv}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
hostname; date
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$MV_VENV/bin/python" ] || "$UV" venv --python 3.11 "$MV_VENV"
"$UV" pip install --python "$MV_VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "numpy<2" pillow scipy opencv-python-headless plyfile \
  "diffusers==0.33.1" "transformers==4.46.3" "huggingface-hub==0.30.2" accelerate safetensors einops timm kornia
"$MV_VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for r in ("sudo-ai/zero123plus-v1.2", "sudo-ai/zero123plus-pipeline", "ZhengPeng7/BiRefNet", "prs-eth/marigold-normals-v1-1"):
    print(r, "->", snapshot_download(r))
PY
"$MV_VENV/bin/python" - <<'PY'
import torch, diffusers, transformers
from diffusers import DiffusionPipeline, MarigoldNormalsPipeline
from transformers import AutoModelForImageSegmentation
p = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="sudo-ai/zero123plus-pipeline", torch_dtype=torch.float16)
m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
n = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16)
print("OK torch", torch.__version__, "diffusers", diffusers.__version__, type(p).__name__, type(m).__name__, type(n).__name__)
PY
echo MV_VENV_DONE
