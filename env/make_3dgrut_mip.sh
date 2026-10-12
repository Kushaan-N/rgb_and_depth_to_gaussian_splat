#!/usr/bin/env bash
# A 3DGRUT checkout for anti-aliasing experiments: git worktree of $THREEDGRUT_DIR (same commit, shares its venv)
# with patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then patches/3dgrut-mip.patch:
#   model.mip_filter_3d.enabled=true: Mip-Splatting's 3D smoothing filter (Yu et al., CVPR 2024; re-implemented
#     from the paper's equations, no code copied — the reference repo is under the non-commercial Inria licence).
#     Per gaussian, filter = min over training cameras that see it of (depth / focal) * sqrt(variance), recomputed
#     after every densification step and every update_interval steps; MixtureOfGaussians.get_scale/get_density
#     return sqrt(s^2 + filter^2) and opacity * sqrt(det(S^2) / det(S^2 + filter^2)), so the renderer, eval,
#     PLY / NuRec export all see the filtered gaussians (exported files have it baked in). Densification
#     strategies see the raw parameters (as in the reference). The filter is stored in the checkpoint.
#   3DGUT's projector already applies the Mip 2D filter (opacity-compensated 0.3 px^2 screen-space dilation,
#     TGUTProjectorParams::MipSplattingScaling), so no 2D change is made.
# Default (enabled=false) behaves exactly as the densify worktree. Jobs using this worktree must set
# TORCH_EXTENSIONS_DIR to a private dir (same kernels as the densify build; no CUDA change here).
#   bash env/make_3dgrut_mip.sh          # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_MIP_DIR:-$CEAR_WS/3dgrut-mip}"
PATCHES="${MIP_PATCHES:-3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-mip}"   # order matters
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-mip] linked submodule $sm"; }
done
LAST="${PATCHES##* }"   # later patches touch the same files, so only the last one can be checked in reverse
if git -C "$DEST" apply --reverse --check "$REPO/patches/$LAST.patch" 2>/dev/null; then
  echo "[3dgrut-mip] $PATCHES already applied"; PATCHES=""
fi
for p in $PATCHES; do
  if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
    git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-mip] $p applied"
  elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
    echo "[3dgrut-mip] $p already applied"
  else
    echo "[3dgrut-mip] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
  fi
done
echo "[3dgrut-mip] ready at $DEST"
