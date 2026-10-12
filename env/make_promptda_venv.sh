#!/usr/bin/env bash
# Venv for Prompt Depth Anything (scripts/promptda_depth.py): metric depth from RGB prompted by a low-res depth map
# (Lin et al., CVPR 2025, https://github.com/DepthAnything/PromptDA). Uses the Hugging Face transformers port
# (PromptDepthAnythingForDepthEstimation, transformers >= 4.51) — no clone of the reference repo needed — and
# fetches the weights into $HF_HOME on scratch. Run inside a (GPU) job:
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:1 -c 4 --mem=16G -t 00:30:00 env/make_promptda_venv.sh
#
# Idempotent. Licences: PromptDA code Apache-2.0; weights depth-anything/prompt-depth-anything-{vitl,vits,
# vits-transparent}-hf are Apache-2.0 (model cards); transformers Apache-2.0.
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
PROMPTDA_VENV="${PROMPTDA_VENV:-$CEAR_WS/venv-promptda}"
PROMPTDA_MODEL="${PROMPTDA_MODEL:-depth-anything/prompt-depth-anything-vitl-hf}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$HF_HOME"
hostname; date

UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$PROMPTDA_VENV/bin/python" ] || "$UV" venv --python 3.11 "$PROMPTDA_VENV"
"$UV" pip install --python "$PROMPTDA_VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "numpy<2" "transformers>=4.51,<5" safetensors \
  huggingface_hub opencv-python-headless pillow scipy

echo "[promptda] fetching $PROMPTDA_MODEL and smoke-testing one prediction"
"$PROMPTDA_VENV/bin/python" - "$PROMPTDA_MODEL" <<'PY'
import sys, torch, numpy as np
from PIL import Image
from transformers import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingImageProcessor
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = PromptDepthAnythingImageProcessor.from_pretrained(sys.argv[1])
m = PromptDepthAnythingForDepthEstimation.from_pretrained(sys.argv[1]).to(dev).eval()
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
pr = np.full((192, 256), 2000.0, np.float32)                       # 2 m in mm (processor default prompt_scale_to_meter)
inp = proc(images=img, prompt_depth=pr, return_tensors="pt").to(dev)
with torch.no_grad():
    d = proc.post_process_depth_estimation(m(**inp), target_sizes=[(480, 640)])[0]["predicted_depth"]
print("OK torch", torch.__version__, "| device", dev, "| depth", tuple(d.shape), "median", float(d.median()))
print("processor:", proc.to_dict())
PY
echo "MAKE_PROMPTDA_VENV_DONE"
