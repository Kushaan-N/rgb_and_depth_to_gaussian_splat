#!/usr/bin/env bash
# A second 3DGRUT checkout with depth supervision (patches/3dgrut-depth-loss.patch), as a git worktree
# of $THREEDGRUT_DIR at the same commit so it shares the 3DGRUT venv. The main pipeline's checkout
# stays untouched. With loss.use_depth=false (default) the patched trainer behaves exactly as stock.
#   bash env/make_3dgrut_depth.sh          # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules (tiny-cuda-nn headers, OptiX SDK): link the main checkout's
# copies (read-only at build time) instead of cloning them again
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "$DEST/$sm"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-depth] linked submodule $sm"; }
done
if git -C "$DEST" apply --check "$REPO/patches/3dgrut-depth-loss.patch" 2>/dev/null; then
  git -C "$DEST" apply "$REPO/patches/3dgrut-depth-loss.patch"; echo "[3dgrut-depth] patch applied"
elif git -C "$DEST" apply --reverse --check "$REPO/patches/3dgrut-depth-loss.patch" 2>/dev/null; then
  echo "[3dgrut-depth] patch already applied"
else
  echo "[3dgrut-depth] patch does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
fi
echo "[3dgrut-depth] ready at $DEST"
