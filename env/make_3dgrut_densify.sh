#!/usr/bin/env bash
# A third 3DGRUT checkout for densification experiments: git worktree of $THREEDGRUT_DIR (same commit, shares
# its venv) with patches/3dgrut-depth-loss.patch and then patches/3dgrut-improved-densify.patch applied:
#   strategy.method=ImprovedGSStrategy (configs/apps/colmap_3dgut_improved.yaml): port of Improved-GS
#     densification (edge-aware score, long-axis split, recovery-aware pruning, budget growth control,
#     late-stage multi-step updates); 3DGUT backward also accumulates per-pixel |dL/dpos| for it.
#   Slang header is only rewritten when its content changes (no 4-min nvcc rebuild on every run, no
#     races between concurrent jobs sharing this worktree).
# Stock strategies (gs, mcmc) and loss.use_depth=false behave exactly as in the stock checkout. Jobs using
# this worktree must set TORCH_EXTENSIONS_DIR to a private dir (its kernels differ from the stock build).
#   bash env/make_3dgrut_densify.sh          # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-densify] linked submodule $sm"; }
done
for p in 3dgrut-depth-loss 3dgrut-improved-densify; do      # order matters: the second builds on the first
  if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
    git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-densify] $p applied"
  elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
    echo "[3dgrut-densify] $p already applied"
  else
    echo "[3dgrut-densify] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
  fi
done
echo "[3dgrut-densify] ready at $DEST"
