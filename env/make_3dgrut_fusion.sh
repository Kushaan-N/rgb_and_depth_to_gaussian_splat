#!/usr/bin/env bash
# 3DGRUT checkout for multi-recording fusion: git worktree of $THREEDGRUT_DIR (same commit, shares its venv) with
# patches/3dgrut-depth-loss.patch, patches/3dgrut-improved-densify.patch and then patches/3dgrut-fusion.patch:
#   appearance.mode   none | recording | frame: a learnable colour affine (appearance.model: affine = 3x3 + bias,
#                     gain = per-channel gain + bias) applied to the RENDER during training only, one per recording
#                     (= image sub-folder) or per frame; frames of appearance.reference ("" = first sorted
#                     sub-folder) stay identity. Rendering / export / eval are untouched (standard splat), the
#                     learnt transforms are written to <run>/appearance.json with each checkpoint.
#   loss.valid_mask   true: the photometric loss of each training frame is weighted by <stem><valid_suffix>
#                     (default _valid.png, 255 = use, 0 = ignore; e.g. scripts/change_masks.py), missing = all valid.
# Python-only on top of the densify patch (same CUDA kernels); jobs must still use a private TORCH_EXTENSIONS_DIR.
#   bash env/make_3dgrut_fusion.sh           # login node is fine: no build, kernels JIT on first GPU use
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/env/cear_env.sh" >/dev/null
DEST="${THREEDGRUT_FUSION_DIR:-$CEAR_WS/3dgrut-fusion}"
if [ ! -d "$DEST" ]; then
  git -C "$THREEDGRUT_DIR" worktree add --detach "$DEST" "$(git -C "$THREEDGRUT_DIR" rev-parse HEAD)"
fi
# git worktrees do not populate submodules: link the main checkout's copies (read-only at build time)
git -C "$THREEDGRUT_DIR" submodule status | awk '{print $2}' | while read -r sm; do
  if [ -d "$DEST/$sm" ] && [ -z "$(ls -A "$DEST/$sm")" ]; then rmdir "${DEST:?}/${sm:?}"; fi
  [ -e "$DEST/$sm" ] || { ln -s "$THREEDGRUT_DIR/$sm" "$DEST/$sm"; echo "[3dgrut-fusion] linked submodule $sm"; }
done
# the fusion patch edits lines the earlier patches added: "last patch reverse-applies" = all applied
if [ -f "$REPO/patches/3dgrut-fusion.patch" ] && git -C "$DEST" apply --reverse --check "$REPO/patches/3dgrut-fusion.patch" 2>/dev/null; then
  echo "[3dgrut-fusion] patches already applied"
else
  for p in 3dgrut-depth-loss 3dgrut-improved-densify 3dgrut-fusion; do   # order matters: each builds on the last
    [ -f "$REPO/patches/$p.patch" ] || { echo "[3dgrut-fusion] no $p.patch (yet)"; continue; }
    if git -C "$DEST" apply --check "$REPO/patches/$p.patch" 2>/dev/null; then
      git -C "$DEST" apply "$REPO/patches/$p.patch"; echo "[3dgrut-fusion] $p applied"
    elif git -C "$DEST" apply --reverse --check "$REPO/patches/$p.patch" 2>/dev/null; then
      echo "[3dgrut-fusion] $p already applied"
    else
      echo "[3dgrut-fusion] $p does not apply to $DEST ($(git -C "$DEST" rev-parse --short HEAD))"; exit 1
    fi
  done
fi
echo "[3dgrut-fusion] ready at $DEST"
