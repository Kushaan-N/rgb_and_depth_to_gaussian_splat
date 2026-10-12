#!/usr/bin/env bash
# Viewer-matched gsplat for scripts/train_viewer_gsplat.py: a source copy of the gsplat 1.5.3 package of
# $CEAR_WS/venv-gsplat (env/make_gsplat_venv.sh; that venv is NOT modified) at $GSPLAT_VIEWER_DIR/gsplat,
# put first on PYTHONPATH, with the prebuilt csrc.so removed so its CUDA kernels are JIT-compiled (into
# $TORCH_EXTENSIONS_DIR, default $CEAR_WS/torch-ext-gsplat-viewer) from patched sources:
#   the 3DGS rasteriser (forward, backward, indices) drops a pixel/gaussian pair beyond Mahalanobis d^2 = 8
#   (sigma = d^2/2 > 4, i.e. the exp(-4) cut-off of PlayCanvas / SuperSplat), next to gsplat's 1/255 alpha cut.
# Everything else (EWA projection, per-tile depth sort = global view-depth order, SH) is stock gsplat.
# Idempotent (re-copies when the source version changes; the JIT build is cached). Needs nvcc: run in a job,
#   module load cuda/12.4.1 gcc/9.4.0; bash env/make_gsplat_viewer_venv.sh      (~5-10 min first build)
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
GSPLAT_VENV="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}"
GSPLAT_VIEWER_DIR="${GSPLAT_VIEWER_DIR:-$CEAR_WS/gsplat-viewer}"
export TORCH_EXTENSIONS_DIR="${GSPLAT_VIEWER_EXT:-$CEAR_WS/torch-ext-gsplat-viewer}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
PY="$GSPLAT_VENV/bin/python"; export PATH="$GSPLAT_VENV/bin:$PATH"     # ninja for the JIT build
SRC="$("$PY" -c "import gsplat, os; print(os.path.dirname(gsplat.__file__))")"
VER="$("$PY" -c "import gsplat; print(gsplat.__version__)")"; STAMP="$GSPLAT_VIEWER_DIR/.patched-$VER-cut4"
mkdir -p "$GSPLAT_VIEWER_DIR" "$TORCH_EXTENSIONS_DIR"

if [ ! -f "$STAMP" ]; then
  rm -rf "$GSPLAT_VIEWER_DIR/gsplat" "$TORCH_EXTENSIONS_DIR/gsplat_cuda"
  cp -r "$SRC" "$GSPLAT_VIEWER_DIR/gsplat"
  rm -rf "$GSPLAT_VIEWER_DIR/gsplat/csrc.so" "$GSPLAT_VIEWER_DIR"/gsplat/__pycache__ "$GSPLAT_VIEWER_DIR"/gsplat/*/__pycache__
  C="$GSPLAT_VIEWER_DIR/gsplat/cuda"
  sed -i 's|^#define ALPHA_THRESHOLD (1.f / 255.f)|&\n#define GSPLAT_SIGMA_CUT 4.f  // viewer kernel cut-off: d^2 = 2 sigma <= 8 (gsplat-viewer patch)|' "$C/include/Common.h"
  for f in RasterizeToPixels3DGSFwd.cu RasterizeToPixels3DGSBwd.cu RasterizeToIndices3DGS.cu; do
    sed -i 's/if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {/if (sigma < 0.f || sigma > GSPLAT_SIGMA_CUT || alpha < ALPHA_THRESHOLD) {/' "$C/csrc/$f"
    grep -q GSPLAT_SIGMA_CUT "$C/csrc/$f" || { echo "[gsplat-viewer] patch failed on $f"; exit 1; }
  done
  grep -q GSPLAT_SIGMA_CUT "$C/include/Common.h" || { echo "[gsplat-viewer] patch failed on Common.h"; exit 1; }
  touch "$STAMP"
fi

# JIT build + check: one splat, a pixel at d^2 = 9 must be empty (stock gsplat draws it), d^2 = 6.25 must not
PYTHONPATH="$GSPLAT_VIEWER_DIR" "$PY" - <<'PY'
import math, torch, gsplat
from gsplat.cuda._backend import _C
assert _C is not None, "JIT build failed"
assert "gsplat-viewer" in gsplat.__file__, gsplat.__file__
from gsplat.cuda._wrapper import rasterize_to_pixels, isect_tiles, isect_offset_encode
d = "cuda"; W = H = 32; s = 2.0                                    # isotropic, sigma 2 px, centre on pixel 16
m = torch.tensor([[[16.5, 16.5]]], device=d); con = torch.tensor([[[1 / s**2, 0, 1 / s**2]]], device=d)
r = torch.tensor([[[12, 12]]], device=d, dtype=torch.int32); z = torch.ones(1, 1, device=d)
_, ids, fl = isect_tiles(m, r, z, 16, 2, 2); off = isect_offset_encode(ids, 1, 2, 2)
img, _ = rasterize_to_pixels(m, con, torch.ones(1, 1, 1, device=d), torch.ones(1, 1, device=d), W, H, 16, off, fl)
a6, a9 = float(img[0, 16, 21, 0]), float(img[0, 16, 22, 0])          # 5 / 6 px right: d^2 = 6.25 / 9
print(f"[gsplat-viewer] {gsplat.__file__}: alpha at d^2=6.25 {a6:.4f} (exp {math.exp(-3.125):.4f}), d^2=9 {a9:.4f}")
assert a6 > 0 and a9 == 0, "cut-off not active"
PY
echo "[gsplat-viewer] ready: PYTHONPATH=$GSPLAT_VIEWER_DIR TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo MAKE_GSPLAT_VIEWER_DONE
