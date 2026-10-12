#!/usr/bin/env bash
# A 3DGRUT checkout for depth-prior experiments: git worktree of $THREEDGRUT_DIR (same commit, shares its venv)
# with patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then patches/3dgrut-depthmask.patch:
#   loss.depth_mask=true (default): the depth target's validity is multiplied by <image>_depthmask.png where that
#     file exists (scripts/depth_reliability.py: 255 = trust the target, 0 = drop: low-intensity / missing LiDAR
#     returns, depth discontinuities, disagreement with the photometric depth on textured pixels).
#   loss.use_normal / lambda_normal: DN-Splatter-style normal consistency — normals from finite differences of the
#     rendered ray distance vs normals from the depth target (dense targets, e.g. scripts/promptda_depth.py), both
#     back-projected along the same unit camera rays; 1 - cos, gated by the same validity (+ no depth edge, opaque).
# Without _depthmask.png files and with use_normal=false it behaves exactly as the densify worktree. Jobs using
# this worktree must set TORCH_EXTENSIONS_DIR to a private dir (same kernels as the densify build; no CUDA change).
#   bash env/make_3dgrut_depthmask.sh          # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_DEPTHMASK_DIR:-$CEAR_WS/3dgrut-depthmask}"
PATCHES="${DEPTHMASK_PATCHES:-3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-depthmask}"   # order matters
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-depthmask] linked submodule $sm"; }
done
LAST="${PATCHES##* }"   # later patches touch the same files, so only the last one can be checked in reverse
if git -C "$DEST" apply --reverse --check "$REPO/patches/$LAST.patch" 2>/dev/null; then
  echo "[3dgrut-depthmask] $PATCHES already applied"; PATCHES=""
fi
for p in $PATCHES; do
  if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
    git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-depthmask] $p applied"
  elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
    echo "[3dgrut-depthmask] $p already applied"
  else
    echo "[3dgrut-depthmask] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
  fi
done
echo "[3dgrut-depthmask] ready at $DEST"
