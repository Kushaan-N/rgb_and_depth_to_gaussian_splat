#!/usr/bin/env bash
# Viewer-matched gsplat builds for scripts/train_viewer_gsplat.py: per viewer profile, a source copy of the gsplat
# 1.5.3 package of $CEAR_WS/venv-gsplat (env/make_gsplat_venv.sh; that venv is NOT modified) at
# $CEAR_WS/gsplat-viewer-<profile>/gsplat (put it first on PYTHONPATH), prebuilt csrc.so removed so the CUDA kernels
# are JIT-compiled (into $CEAR_WS/torch-ext-gsplat-viewer-<profile>) from patched sources:
#   both     the EWA Jacobian is evaluated at the true centre: gsplat clamps it to 1.3x the field of view, the viewers
#            do not, so splats centred off-screen near the camera are drawn stretched by the viewer (measured: ~176
#            such splats per view cost a stock-gsplat-trained model 5 dB in PlayCanvas)
#   pc274    PlayCanvas 2.7.4 kernel: pixel/gaussian pairs beyond Mahalanobis d^2 = 8 (sigma > 4) dropped, 1/255 cut
#   ss352    SuperSplat 3.5.2 kernel: normalised (exp(-d^2/2) - e^-4) / (1 - e^-4), zero beyond d^2 = 8, no 1/255
#            cut (alpha threshold 1e-5); forward, backward (d alpha/d sigma, d alpha/d opacity) and indices kernels
# Everything else (EWA projection, per-tile depth sort = global view-depth order, SH) is stock gsplat; the trainer
# adds the viewers' size cull, radius cap and opacity quantisation in torch. Idempotent (re-copies when the source
# version or this patch set changes; the JIT build is cached). Needs nvcc: run in a job (~5 min per profile),
#   module load cuda/12.4.1 gcc/9.4.0; bash env/make_gsplat_viewer_venv.sh [pc274] [ss352]
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
GSPLAT_VENV="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}"
PROFILES="${*:-pc274 ss352}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
PY="$GSPLAT_VENV/bin/python"; export PATH="$GSPLAT_VENV/bin:$PATH"     # ninja for the JIT build
SRC="$("$PY" -c "import gsplat, os; print(os.path.dirname(gsplat.__file__))")"
VER="$("$PY" -c "import gsplat; print(gsplat.__version__)")"; PATCHSET=v2

for prof in $PROFILES; do
  D="$CEAR_WS/gsplat-viewer-$prof"; X="$CEAR_WS/torch-ext-gsplat-viewer-$prof"; STAMP="$D/.patched-$VER-$PATCHSET"
  if [ ! -f "$STAMP" ]; then
    rm -rf "${D:?}" "${X:?}"; mkdir -p "$D" "$X"
    cp -r "$SRC" "$D/gsplat"
    rm -rf "${D:?}/gsplat/csrc.so" "${D:?}"/gsplat/__pycache__ "${D:?}"/gsplat/*/__pycache__
    "$PY" - "$D/gsplat/cuda" "$prof" <<'PY'
import sys
C, prof = sys.argv[1], sys.argv[2]
def sub(f, pairs):
    p = f"{C}/{f}"; s = open(p).read()
    for a, b, n in pairs:
        if s.count(a) != n:
            sys.exit(f"[gsplat-viewer] patch failed: {f}: {a!r} found {s.count(a)}x, expected {n}")
        s = s.replace(a, b)
    open(p, "w").write(s)
floor, scale, thr = ("0.f", "1.f", "(1.f / 255.f)") if prof == "pc274" else \
    ("0.018315638888734179f", "1.0186573603637741f", "1e-5f")            # e^-4, 1 / (1 - e^-4)
sub("include/Common.h", [("#define ALPHA_THRESHOLD (1.f / 255.f)",
    f"#define ALPHA_THRESHOLD {thr}  // gsplat-viewer {prof}\n#define GSPLAT_SIGMA_CUT 4.f  // d^2 = 2 sigma <= 8\n"
    f"#define GSPLAT_K_FLOOR {floor}\n#define GSPLAT_K_SCALE {scale}  // alpha = opacity * (exp(-sigma) - FLOOR) * SCALE", 1)])
sub("include/Utils.cuh", [(f" + 0.3f * tan_fov{a};", f" + 1e6f * tan_fov{a};  // gsplat-viewer: no Jacobian clamp", 4) for a in "xy"])
cut = ("if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {", "if (sigma < 0.f || sigma > GSPLAT_SIGMA_CUT || alpha < ALPHA_THRESHOLD) {", 1)
k_fwd = ("float alpha = min(0.999f, opac * __expf(-sigma));",
         "float alpha = min(0.999f, opac * (__expf(-sigma) - GSPLAT_K_FLOOR) * GSPLAT_K_SCALE);", 1)
sub("csrc/RasterizeToPixels3DGSFwd.cu", [k_fwd, cut])
sub("csrc/RasterizeToIndices3DGS.cu", [k_fwd, cut])
sub("csrc/RasterizeToPixels3DGSBwd.cu", [
    ("alpha = min(0.999f, opac * vis);", "alpha = min(0.999f, opac * (vis - GSPLAT_K_FLOOR) * GSPLAT_K_SCALE);", 1), cut,
    ("if (opac * vis <= 0.999f) {", "if (opac * (vis - GSPLAT_K_FLOOR) * GSPLAT_K_SCALE <= 0.999f) {", 1),
    ("const float v_sigma = -opac * vis * v_alpha;", "const float v_sigma = -opac * vis * GSPLAT_K_SCALE * v_alpha;", 1),
    ("v_opacity_local = vis * v_alpha;", "v_opacity_local = (vis - GSPLAT_K_FLOOR) * GSPLAT_K_SCALE * v_alpha;", 1)])
open(f"{C}/../VIEWER_PROFILE", "w").write(prof + "\n")
print(f"[gsplat-viewer] {prof}: patched")
PY
    touch "$STAMP"
  fi
  # JIT build + check: one splat (sigma 2 px, opacity 1): kernel value at d^2 = 6.25 (5 px right), its opacity
  # gradient, and nothing at d^2 = 9 (6 px; stock gsplat draws exp(-4.5) there)
  PYTHONPATH="$D" TORCH_EXTENSIONS_DIR="$X" "$PY" - "$prof" <<'PY'
import math, sys, torch, gsplat
from gsplat.cuda._backend import _C
assert _C is not None, "JIT build failed"
prof = sys.argv[1]
assert f"gsplat-viewer-{prof}" in gsplat.__file__, gsplat.__file__
from gsplat.cuda._wrapper import rasterize_to_pixels, isect_tiles, isect_offset_encode
d = "cuda"; s = 2.0
m = torch.tensor([[[16.5, 16.5]]], device=d); con = torch.tensor([[[1 / s**2, 0, 1 / s**2]]], device=d)
r = torch.tensor([[[12, 12]]], device=d, dtype=torch.int32); z = torch.ones(1, 1, device=d)
_, ids, fl = isect_tiles(m, r, z, 16, 2, 2); off = isect_offset_encode(ids, 1, 2, 2)
op = torch.ones(1, 1, device=d, requires_grad=True)
img, _ = rasterize_to_pixels(m, con, torch.ones(1, 1, 1, device=d), op, 32, 32, 16, off, fl)
a6, a9 = float(img[0, 16, 21, 0]), float(img[0, 16, 22, 0])
E4 = math.exp(-4); want = math.exp(-3.125) if prof == "pc274" else (math.exp(-3.125) - E4) / (1 - E4)
img[0, 16, 21, 0].backward()                                       # d alpha / d opacity = kernel value
print(f"[gsplat-viewer] {prof} {gsplat.__file__}: alpha(d^2=6.25) {a6:.5f} (want {want:.5f}), d alpha/d op "
      f"{float(op.grad):.5f}, alpha(d^2=9) {a9:.5f}")
assert abs(a6 - want) < 1e-4 and a9 == 0 and abs(float(op.grad) - want) < 1e-4, "viewer kernel not active"
PY
  echo "[gsplat-viewer] $prof ready: PYTHONPATH=$D TORCH_EXTENSIONS_DIR=$X"
done
echo MAKE_GSPLAT_VIEWER_DONE
