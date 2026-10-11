#!/usr/bin/env bash
# Build the TRELLIS venv on scratch (Microsoft TRELLIS, CVPR 2025, MIT: structured-latent image-to-3D).
# Only what the structure-conditioned path of scripts/trellis_complete.py needs: the SLat flow model
# (stage 2), the Gaussian decoder and DINOv2 conditioning. Stage 1 / mesh / radiance-field decoders and the
# renderers (nvdiffrast, diffoctreerast, mip-splatting rasteriser) are not needed and not built;
# transformers + open3d + kaolin (wheel) are only imported (text pipeline, flexicubes). Every package is a
# prebuilt wheel (torch 2.4.0+cu121, xformers 0.0.27.post2, spconv-cu120), so nothing compiles.
# Weights microsoft/TRELLIS-image-large -> $HF_HOME, DINOv2 hub code+weights -> $TORCH_HOME.
# Ends with a GPU smoke test (a box structure + a synthetic image -> gaussians), so run it in a GPU job:
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:1 -c 4 --mem=32G -t 01:00:00 \
#     -o logs/trellis-venv_%j.out env/make_trellis_venv.sh
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
TRELLIS_VENV="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}"
TRELLIS_DIR="${TRELLIS_DIR:-$CEAR_WS/TRELLIS}"
TRELLIS_COMMIT="${TRELLIS_COMMIT:-442aa1e1afb9014e80681d3bf604e8d728a86ee7}"   # microsoft/TRELLIS main, 2025-11-05
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-hub}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$HF_HOME" "$TORCH_HOME"
hostname; date

[ -d "$TRELLIS_DIR/.git" ] || git clone https://github.com/microsoft/TRELLIS.git "$TRELLIS_DIR"
git -C "$TRELLIS_DIR" fetch -q origin && git -C "$TRELLIS_DIR" checkout -q "$TRELLIS_COMMIT"
git -C "$TRELLIS_DIR" submodule update --init --recursive   # flexicubes (imported by trellis.representations)
echo "[trellis] code at $TRELLIS_DIR @ $(git -C "$TRELLIS_DIR" rev-parse --short HEAD)"

UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
PY="$TRELLIS_VENV/bin/python"
[ -x "$PY" ] || "$UV" venv --python 3.10 "$TRELLIS_VENV"
pipi() { "$UV" pip install --python "$PY" "$@"; }
pipi --index-url https://download.pytorch.org/whl/cu121 "torch==2.4.0" "torchvision==0.19.0" "xformers==0.0.27.post2"
pipi "numpy<2" spconv-cu120 pillow imageio tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
  plyfile safetensors "huggingface-hub<1" trimesh setuptools "transformers==4.46.3" open3d \
  "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
# kaolin: only imported (flexicubes, via trellis.representations); NVIDIA's prebuilt wheel for this torch/cu
pipi --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html "kaolin==0.17.0"

echo "[trellis] weights -> $HF_HOME, DINOv2 -> $TORCH_HOME"
"$PY" - <<'PY'
import torch
from huggingface_hub import snapshot_download
print("TRELLIS-image-large ->", snapshot_download("microsoft/TRELLIS-image-large"))
torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)   # caches hub code + weights
PY

echo "[trellis] GPU smoke test: box structure + synthetic image -> sample_slat -> gaussians"
cd "$TRELLIS_DIR" && ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers SPCONV_ALGO=native "$PY" - <<'PY'
import numpy as np, torch
from huggingface_hub import snapshot_download
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
pipe = TrellisImageTo3DPipeline.from_pretrained(snapshot_download("microsoft/TRELLIS-image-large")); pipe.cuda()
g = np.stack(np.meshgrid(*[np.arange(64)] * 3, indexing="ij"), -1).reshape(-1, 3)
box = (np.abs(g - 31.5) <= [20, 12, 8]).all(1) & ~(np.abs(g - 31.5) <= [19, 11, 7]).all(1)   # hollow box shell
coords = torch.tensor(np.c_[np.zeros(box.sum()), g[box]], dtype=torch.int32, device="cuda")
im = np.zeros((518, 518, 3), np.uint8); im[180:340, 100:420] = (200, 60, 40)
cond = pipe.get_cond([Image.fromarray(im)] * 2); cond["neg_cond"] = cond["neg_cond"][:1]
torch.manual_seed(0)
with pipe.inject_sampler_multi_image("slat_sampler", 2, 12, mode="stochastic"):
    slat = pipe.sample_slat(cond, coords, {"steps": 12})
gs = pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]
rgb = 0.5 + 0.28209479177387814 * gs._features_dc[:, 0]
print("OK torch", torch.__version__, "| voxels", len(coords), "| gaussians", gs.get_xyz.shape[0],
      "| mean rgb", rgb.mean(0).tolist(), "| xyz range", gs.get_xyz.min(0).values.tolist(), gs.get_xyz.max(0).values.tolist())
PY
echo "MAKE_TRELLIS_VENV_DONE"
