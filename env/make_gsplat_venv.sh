#!/usr/bin/env bash
# Build the gsplat venv on scratch ($CEAR_WS/venv-gsplat): nerfstudio-project/gsplat (Apache-2.0), used by
# scripts/refine_poses_gsplat.py for photometric camera-pose refinement (3DGRUT has no pose optimisation).
# Pinned release 1.5.3 as the prebuilt CUDA wheel for torch 2.4 / cu124 / CPython 3.10 (no source build),
# plus the deps of gsplat's examples/simple_trainer.py; the gsplat repo (for the examples) is checked out
# at the same tag under $CEAR_WS/gsplat. Idempotent (skips when the venv already imports gsplat). Run in a
# job (downloads ~3 GB; fused-ssim, an optional simple_trainer dep, compiles if nvcc is present):
#
#   sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:a100:1 -c 4 --mem=16G -t 01:00:00 env/make_gsplat_venv.sh
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
GSPLAT_VENV="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}"
GSPLAT_DIR="${GSPLAT_DIR:-$CEAR_WS/gsplat}"
GSPLAT_VER="${GSPLAT_VER:-1.5.3}"
TORCH_TAG="${TORCH_TAG:-pt24cu124}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy TORCH_HOME="$CEAR_WS/torch-cache"
mkdir -p "$CEAR_WS"
hostname; date

if "$GSPLAT_VENV/bin/python" -c "import gsplat, sys; sys.exit(gsplat.__version__.split('+')[0] != '$GSPLAT_VER')" 2>/dev/null; then
  echo "[gsplat] venv ready at $GSPLAT_VENV"
else
  UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
  "$UV" venv --python 3.10 "$GSPLAT_VENV"
  "$UV" pip install --python "$GSPLAT_VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --find-links "https://docs.gsplat.studio/whl/gsplat/" \
    "torch==2.4.1+cu124" "torchvision==0.19.1+cu124" "gsplat==${GSPLAT_VER}+${TORCH_TAG}" \
    "numpy<2" scipy pillow "imageio[ffmpeg]" opencv-python-headless plyfile tqdm pyyaml matplotlib \
    "torchmetrics[image]" "tyro>=0.8.8" scikit-learn tensorboard tensorly splines viser \
    "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
  # optional simple_trainer deps that build from git: never fatal (refine_poses_gsplat.py needs neither)
  if command -v nvcc >/dev/null; then
    "$UV" pip install --python "$GSPLAT_VENV/bin/python" --no-build-isolation \
      "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
      "git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe" \
      || echo "[gsplat] WARNING: fused-ssim/fused-bilagrid build failed (optional)"
  fi
  "$UV" pip install --python "$GSPLAT_VENV/bin/python" \
    "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
    || echo "[gsplat] WARNING: pycolmap (simple_trainer's fork, examples/requirements.txt pins) install failed (optional)"
fi
[ -d "$GSPLAT_DIR/.git" ] || git clone -q https://github.com/nerfstudio-project/gsplat.git "$GSPLAT_DIR"
git -C "$GSPLAT_DIR" fetch -q --tags origin && git -C "$GSPLAT_DIR" checkout -q "v$GSPLAT_VER"
echo "[gsplat] examples at $GSPLAT_DIR @ $(git -C "$GSPLAT_DIR" describe --tags)"

"$GSPLAT_VENV/bin/python" - <<'PY'
import torch, gsplat
print("OK torch", torch.__version__, "| gsplat", gsplat.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():                       # one tiny forward+backward through the CUDA rasteriser
    from gsplat import rasterization
    n, d = 1000, "cuda"
    means = (torch.randn(n, 3, device=d) * 0.3 + torch.tensor([0, 0, 3.0], device=d)).requires_grad_()
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1)
    img, alpha, _ = rasterization(means, quats, torch.full((n, 3), 0.02, device=d), torch.full((n,), 0.8, device=d),
                                  torch.rand(n, 3, device=d), torch.eye(4, device=d)[None],
                                  torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device=d), 64, 64)
    img.sum().backward()
    print("rasterization OK", tuple(img.shape), "grad", float(means.grad.abs().sum()) > 0)
PY
echo "MAKE_GSPLAT_VENV_DONE"
