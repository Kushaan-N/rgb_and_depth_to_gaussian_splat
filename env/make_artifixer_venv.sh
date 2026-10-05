#!/usr/bin/env bash
# Build the ArtiFixer venv on scratch (NVIDIA ArtiFixer, SIGGRAPH 2026: a Wan2.1 video world model
# that fills unobserved regions of a 3DGRUT reconstruction and distills them back into 3D).
# Mirrors the repo's Dockerfile.cuda12 (Docker is unavailable on Unity) minus the Hopper-only
# FlashAttention-3/4 builds: on A100 (sm_80) ArtiFixer uses PyTorch SDPA + Triton flex_attention.
# Downloads ~100 GB (torch, ArtiFixer-1.3B, Wan2.1-T2V-1.3B, Qwen3-VL-30B-A3B captioner): CPU job,
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 8 --mem=32G -t 04:00:00 env/make_artifixer_venv.sh
#
# Code Apache-2.0; ArtiFixer weights NVIDIA One-Way Non-commercial (research use).
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"
AFX_DIR="${AFX_DIR:-$CEAR_WS/ArtiFixer}"
AFX_COMMIT="${AFX_COMMIT:-a392c4dfe17459ef9952407accdb9fcdcdddba98}"   # nv-tlabs/ArtiFixer main, 2026-07-22
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$HF_HOME" "$CEAR_WS/artifixer-checkpoints"
hostname; date

if [ ! -d "$AFX_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/nv-tlabs/ArtiFixer.git "$AFX_DIR"
fi
git -C "$AFX_DIR" fetch -q origin && git -C "$AFX_DIR" checkout -q "$AFX_COMMIT"
git -C "$AFX_DIR" submodule update --init --recursive
echo "[artifixer] code at $AFX_DIR @ $(git -C "$AFX_DIR" rev-parse --short HEAD)"

UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
PY="$AFX_VENV/bin/python"
"$UV" venv --python 3.12 "$AFX_VENV"
pipi() { "$UV" pip install --python "$PY" "$@"; }
pipi --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" torchvision
# 3DGRUT fork (fused-ssim is a CUDA extension: built in the GPU job on first use instead)
grep -v "fused-ssim" "$AFX_DIR/thirdparty/3DGRUT-ArtiFixer/requirements.txt" > "$AFX_VENV/3dgrut-req.txt"
pipi -r "$AFX_VENV/3dgrut-req.txt"
bash "$AFX_DIR/thirdparty/3DGRUT-ArtiFixer/scripts/install_slangc.sh" "$AFX_VENV"
pipi -e "$AFX_DIR/thirdparty/3DGRUT-ArtiFixer"
pipi "accelerate==1.13.0" "diffusers==0.37.1" "transformers==5.5.0" ftfy einops scipy wandb tqdm Pillow \
  matplotlib opencv-python-headless pyyaml torchmetrics imageio-ffmpeg h5py av torch-fidelity \
  "huggingface-hub[cli]" "git+https://github.com/microsoft/MoGe.git"

echo "[artifixer] downloading weights into $HF_HOME (+ checkpoint into $CEAR_WS/artifixer-checkpoints)"
"$PY" - <<PY
from huggingface_hub import hf_hub_download, snapshot_download
print(hf_hub_download("nvidia/ArtiFixer", "artifixer-1.3b.pt", local_dir="$CEAR_WS/artifixer-checkpoints"))
for repo in ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "Qwen/Qwen3-VL-30B-A3B-Instruct"):
    print(repo, "->", snapshot_download(repo))
PY

echo "[artifixer] import check (CPU)"
cd "$AFX_DIR" && "$PY" -c "import torch, diffusers, transformers, threedgrut; print('OK torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"
echo "MAKE_ARTIFIXER_VENV_DONE"
