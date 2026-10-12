#!/usr/bin/env bash
# 3DGRUT checkout for perceptual-loss experiments: git worktree of $THREEDGRUT_DIR (same commit, shares its venv)
# with patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then
# patches/3dgrut-perceptual.patch applied. The last one adds VGG-16 Wasserstein Distortion (WD; vendored from
# balle-lab/wasserstein-distortion, Apache-2.0, threedgrut/utils/wasserstein_distortion.py) as a training term
# (arXiv 2603.23297 "Drop-In Perceptual Optimization for 3DGS": WD-R = gamma * (WD + beta * original L1/SSIM)):
#   loss.lambda_wd        weight of the WD term (0 = off, default: everything as in the densify worktree)
#   loss.wd_start_iter    WD switches on at this global step (warm-up on the original loss before it)
#   loss.wd_sigma         WD pooling size as the WD code's log2_sigma map (the paper's constant sigma=4)
#   loss.lambda_rgb_scale once WD is on, the original L1/SSIM terms are multiplied by this (WD-R regulariser)
# Python-only on top of the densify patch (same CUDA kernels); jobs must still use a private TORCH_EXTENSIONS_DIR.
#   bash env/make_3dgrut_perceptual.sh       # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_PERCEPTUAL_DIR:-$CEAR_WS/3dgrut-perceptual}"
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-perceptual] linked submodule $sm"; }
done
# the perceptual patch edits lines the depth patch added, so the earlier patches no longer reverse-check once it is
# in: "last patch reverse-applies" = all applied
if git -C "$DEST" apply --reverse --check "$REPO/patches/3dgrut-perceptual.patch" 2>/dev/null; then
  echo "[3dgrut-perceptual] patches already applied"
else
  for p in 3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-perceptual; do   # order matters: each builds on the last
    if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
      git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-perceptual] $p applied"
    elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
      echo "[3dgrut-perceptual] $p already applied"
    else
      echo "[3dgrut-perceptual] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
    fi
  done
fi
echo "[3dgrut-perceptual] ready at $DEST"
