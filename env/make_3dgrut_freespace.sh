#!/usr/bin/env bash
# 3DGRUT checkout for LiDAR free-space floater suppression: git worktree of $THREEDGRUT_DIR (same commit,
# shares its venv) with patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then
# patches/3dgrut-freespace.patch applied:
#   freespace.grid_path=<npz from scripts/build_freespace_grid.py> (absent / empty = feature off) ->
#     loss on the opacity of gaussians whose centre lies in confidently LiDAR-free voxels
#     (freespace.lambda_opacity) and periodic pruning of them (freespace.prune.*; GS / Improved-GS strategies).
# Without freespace.grid_path it behaves exactly like the densify worktree (env/make_3dgrut_densify.sh). Jobs
# using it must set TORCH_EXTENSIONS_DIR to a private dir (kernels = densify build, which differs from stock).
#   bash env/make_3dgrut_freespace.sh        # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_FREESPACE_DIR:-$CEAR_WS/3dgrut-freespace}"
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-freespace] linked submodule $sm"; }
done
# the last patch's context overlaps the earlier ones: once it reverse-applies, the stack is complete
if git -C "$DEST" apply --reverse --check "$REPO/patches/3dgrut-freespace.patch" 2>/dev/null; then
  echo "[3dgrut-freespace] all patches already applied; ready at $DEST"; exit 0
fi
for p in 3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-freespace; do   # order matters: each builds on the previous
  if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
    git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-freespace] $p applied"
  elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
    echo "[3dgrut-freespace] $p already applied"
  else
    echo "[3dgrut-freespace] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
  fi
done
echo "[3dgrut-freespace] ready at $DEST"
