#!/usr/bin/env bash
# Hunyuan3D-2.1 shape generator (image -> untextured mesh; the texture comes from the real photos in object_layers.py) in
# $CEAR_WS/venv-hunyuan, code in $CEAR_WS/Hunyuan3D-2.1, weights (tencent/Hunyuan3D-2.1, hunyuan3d-dit-v2-1) in $HF_HOME.
# Shape stage only: the paint stage needs custom CUDA rasterisers + bpy and is not used. Idempotent; downloads ~10 GB:
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 4 --mem=16G -t 01:00:00 -o logs/%x-%j.out env/make_hunyuan_venv.sh
#SBATCH -J hunyuan-venv
set -euo pipefail
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"; HY="$CEAR_WS/Hunyuan3D-2.1"; V="${HUNYUAN_VENV:-$CEAR_WS/venv-hunyuan}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}"
hostname; date
[ -d "$HY/.git" ] || git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY"
git -C "$HY" log -1 --format='%H %cd'
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$V/bin/python" ] || "$UV" venv --python 3.10 "$V"
"$UV" pip install --python "$V/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" "numpy<2" "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" safetensors huggingface_hub einops omegaconf \
  trimesh pymeshlab scikit-image opencv-python-headless pillow tqdm pyyaml timm rembg onnxruntime scipy 2>&1 | tail -3
"$V/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*", "*.json", "*.md"])
print("weights", p, sorted(os.listdir(p)))
PY
cd "$HY" && PYTHONPATH="$HY/hy3dshape" "$V/bin/python" -c "from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; print('HUNYUAN_IMPORT_OK')"
date; echo HUNYUAN_VENV_DONE
