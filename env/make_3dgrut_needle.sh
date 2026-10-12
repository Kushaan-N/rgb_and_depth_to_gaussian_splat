#!/usr/bin/env bash
# A 3DGRUT checkout for needle suppression: git worktree of $THREEDGRUT_DIR (same commit, shares its venv) with
# patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then patches/3dgrut-needle.patch:
#   loss.lambda_erank  effective-rank regulariser (erank-GS, Hyung et al., NeurIPS 2024, arXiv 2406.11672):
#                      erank = exp(-sum q log q), q_i = s_i^2 / sum s^2; loss = mean(relu(-log(erank - 1 + 1e-5)))
#                      penalises gaussians whose covariance has one dominant axis (needles, erank -> 1)
#   loss.lambda_aniso  anisotropy cap (PhysGaussian-style): mean(relu(s_max / s_min - loss.aniso_max))
#   loss.reg_start_iteration  both start at this step (default 0); both are off during color refinement.
# Defaults (both lambdas 0) behave exactly as the densify worktree. Pure Python change (no kernel change): jobs can
# share the densify build's TORCH_EXTENSIONS_DIR.
#   bash env/make_3dgrut_needle.sh          # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_NEEDLE_DIR:-$CEAR_WS/3dgrut-needle}"
PATCHES="${NEEDLE_PATCHES:-3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-needle}"   # order matters
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-needle] linked submodule $sm"; }
done
LAST="${PATCHES##* }"   # later patches touch the same files, so only the last one can be checked in reverse
if [ -f "$REPO/patches/$LAST.patch" ] && git -C "$DEST" apply --reverse --check "$REPO/patches/$LAST.patch" 2>/dev/null; then
  echo "[3dgrut-needle] $PATCHES already applied"; PATCHES=""
fi
for p in $PATCHES; do
  if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
    git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-needle] $p applied"
  elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
    echo "[3dgrut-needle] $p already applied"
  else
    echo "[3dgrut-needle] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
  fi
done
echo "[3dgrut-needle] ready at $DEST"
