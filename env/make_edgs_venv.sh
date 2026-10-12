#!/usr/bin/env bash
# Venv for EDGS-style dense-correspondence initialisation (scripts/edgs_init.py): RoMa dense matcher +
# triangulation with known poses (EDGS, Kotovenko et al., CVPR 2026, arXiv 2504.13204). Clones EDGS at a
# pinned commit (reference code + its RoMa submodule commit), installs romatch from that commit, and fetches
# the RoMa-outdoor + DINOv2 weights into a torch-hub cache on scratch. Run inside a (GPU) job:
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:1 -c 4 --mem=16G -t 01:00:00 env/make_edgs_venv.sh
#
# Idempotent: re-running skips what exists. Licence: EDGS code is non-commercial academic/personal use only
# (LICENSE.txt in the clone); RoMa is MIT; DINOv2 weights Apache-2.0.
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
EDGS_VENV="${EDGS_VENV:-$CEAR_WS/venv-edgs}"
EDGS_DIR="${EDGS_DIR:-$CEAR_WS/EDGS}"
EDGS_COMMIT="${EDGS_COMMIT:-f90b022445fc88368f75e66e8fb34aea88372cac}"   # CompVis/EDGS main, 2026-07 (CVPR cite)
ROMA_COMMIT="${ROMA_COMMIT:-370117431ffc5dc000fb46f6e581b74bdb2c3ff8}"   # = EDGS's submodules/RoMa pin
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}" HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$TORCH_HOME" "$HF_HOME"
hostname; date

[ -d "$EDGS_DIR/.git" ] || git clone https://github.com/CompVis/EDGS.git "$EDGS_DIR"
git -C "$EDGS_DIR" fetch -q origin && git -C "$EDGS_DIR" checkout -q "$EDGS_COMMIT"
# the RoMa submodule URL is ssh (git@github.com): clone it over https at the pinned commit instead
R="$EDGS_DIR/submodules/RoMa"
[ -d "$R/.git" ] || { rm -rf "$R"; git clone -q https://github.com/Parskatt/RoMa.git "$R"; }
git -C "$R" fetch -q origin && git -C "$R" checkout -q "$ROMA_COMMIT"
echo "[edgs] EDGS @ $(git -C "$EDGS_DIR" rev-parse --short HEAD), RoMa @ $(git -C "$R" rev-parse --short HEAD)"

UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$EDGS_VENV/bin/python" ] || "$UV" venv --python 3.11 "$EDGS_VENV"
# same torch build 3DGRUT runs on Unity's A100s; romatch without its training-only deps (wandb, poselib,
# albumentations, xformers is optional)
"$UV" pip install --python "$EDGS_VENV/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "numpy<2" einops kornia loguru tqdm timm h5py \
  opencv-python-headless matplotlib pillow scipy
"$UV" pip install --python "$EDGS_VENV/bin/python" --no-deps -e "$R"

echo "[edgs] fetching RoMa-outdoor + DINOv2 weights into $TORCH_HOME and smoke-testing a match"
"$EDGS_VENV/bin/python" - <<'PY'
import torch, numpy as np
from PIL import Image
from romatch import roma_outdoor
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = roma_outdoor(device=dev)
a = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype(np.uint8))
w, c = m.match(a, a, device=dev)
print("OK torch", torch.__version__, "| device", dev, "| warp", tuple(w.shape), "| cert", tuple(c.shape))
PY
echo "MAKE_EDGS_VENV_DONE"
