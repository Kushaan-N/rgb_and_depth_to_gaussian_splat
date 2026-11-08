#!/usr/bin/env bash
# Venv for HY-World 2.0's geometry priors (scripts/hyworld_normals.py): WorldMirror 2.0 (feed-forward multi-view
# depth / normals / cameras, accepts our poses, intrinsics and depth as priors) and MoGe-2 (the monocular normal model the
# official 3DGS stage, hyworld2/worldgen/gen_gs_data.py, uses for its normal loss). Code: github Tencent-Hunyuan/HY-World-2.0
# at $CEAR_WS/HY-World-2.0 (only hyworld2/worldrecon is imported; flash-attn optional -> torch SDPA), MoGe pip from GitHub.
# Weights (ungated HF): tencent/HY-World-2.0 (HY-WorldMirror-2.0/ only, ~1.2B params) and Ruicheng/moge-2-vitl-normal,
# into $HF_HOME on scratch. Run inside a GPU job (sbatch/hyworld_normals.sbatch calls it). Idempotent.
# Licences: HY-World 2.0 Tencent Hunyuan community licence (License.txt in the repo); MoGe MIT.
set -euo pipefail
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
HYV="${HYWORLD_VENV:-$CEAR_WS/venv-hyworld}"; HYD="${HYWORLD_DIR:-$CEAR_WS/HY-World-2.0}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
mkdir -p "$CEAR_WS" "$HF_HOME"; hostname; date
[ -d "$HYD/.git" ] || git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-World-2.0.git "$HYD"
echo "[hyworld] code $(git -C "$HYD" rev-parse --short HEAD)"
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$HYV/bin/python" ] || "$UV" venv --python 3.11 "$HYV"
if ! "$HYV/bin/python" -c "import torch, moge, einops, gsplat" 2>/dev/null; then
  "$UV" pip install --python "$HYV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" numpy einops omegaconf safetensors "huggingface_hub<1" \
    opencv-python-headless pillow scipy trimesh plyfile tqdm matplotlib requests "gsplat==1.5.3" \
    "moge @ git+https://github.com/microsoft/MoGe.git@b942f00bdc"   # MoGe-2 era (main is MoGe-3: flex-gemm build)
fi
if ! "$HYV/bin/python" -c "import flash_attn" 2>/dev/null; then   # WorldMirror imports flash-attn unconditionally: SDPA shim
  SP="$("$HYV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"; mkdir -p "$SP/flash_attn"
  : > "$SP/flash_attn/__init__.py"
  cat > "$SP/flash_attn/flash_attn_interface.py" <<'PY'
"""flash_attn_func shim on torch SDPA (env/make_hyworld_venv.sh): q, k, v (B, L, H, D) -> (B, L, H, D)."""
import torch.nn.functional as F
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p,
                                       is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2)
PY
fi
"$HYV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("tencent/HY-World-2.0", allow_patterns=["HY-WorldMirror-2.0/*"]))
print(snapshot_download("Ruicheng/moge-2-vitl-normal"))
PY
PYTHONPATH="$HYD" "$HYV/bin/python" -c "import torch, moge.model.v2; from hyworld2.worldrecon.pipeline import WorldMirrorPipeline; print('OK hyworld', torch.__version__, torch.cuda.is_available())"
echo MAKE_HYWORLD_VENV_DONE
