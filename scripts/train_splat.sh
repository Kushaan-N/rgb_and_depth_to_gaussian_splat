#!/usr/bin/env bash
# Phase 4 (training) — 3D Gaussian Splatting on the gated CEAR frames (docs/PLAN.md §7.3).
#
#   *** THIS NEEDS A GPU (A100/H100). It does NOT run on a login node. ***
#   Launch via sbatch/train_a100.sbatch, or interactively inside a GPU allocation with the
#   3dgrut conda env active. Everything before this step is CPU-only and already done.
#
# Inputs (produced by the CPU pipeline):
#   $CEAR_OUT/<seq>/colmap_train/sparse/0   # gated TRAIN frames, world->cam, PINHOLE,
#                                           # points3D seeded from the fused depth cloud
#   $CEAR_OUT/<seq>/precond/<rgb_dir>       # undistorted images (PINHOLE), §7.2
# Output:
#   $CEAR_OUT/<seq>/train_3dgrut/           # checkpoint + exported PLY (interchange format)
#
# The trainer is SWAPPABLE (plan §3.3): anything that reads a COLMAP model and writes a
# standard 3DGS PLY slots in here. Default = 3DGRUT, 3DGUT (rasterization) config, which is
# A100-friendly (no RT cores needed). Depth supervision is UNVERIFIED (§7.3) — see notes.
set -euo pipefail

SEQ="${1:-mocap1_well-lit_trot}"
RGB_DIR="${RGB_DIR:-rgb}"                           # CEAR indoor download ships only rgb/ (no raw_rgb)
: "${CEAR_OUT:?source env/cear_env.sh first}"
: "${THREEDGRUT_DIR:?set THREEDGRUT_DIR to your cloned github.com/nv-tlabs/3dgrut}"

OUT="$CEAR_OUT/$SEQ"
TRAIN_MODEL="$OUT/colmap_train/sparse/0"
IMAGES="$OUT/precond/$RGB_DIR"
[ -d "$TRAIN_MODEL" ] || { echo "missing $TRAIN_MODEL — run gate_frames.py"; exit 1; }
[ -d "$IMAGES" ]      || { echo "missing $IMAGES — run precondition_frames.py"; exit 1; }

# --- assemble a COLMAP-style dataset dir that 3DGRUT's dataloader expects ---
DATASET="$OUT/dataset_3dgrut"
mkdir -p "$DATASET/sparse"
ln -sfn "$IMAGES"                   "$DATASET/images"
ln -sfn "$OUT/colmap_train/sparse/0" "$DATASET/sparse/0"

echo "[train_splat] dataset:   $DATASET"
echo "[train_splat] 3dgrut:    $THREEDGRUT_DIR"
echo "[train_splat] GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '??')"

cd "$THREEDGRUT_DIR"
# 3DGUT (rasterization) config. Confirm the exact config name against your PINNED commit
# (3dgrut uses Hydra; config names have moved between releases). As of the reference commit
# the rasterization app config is apps/colmap_3dgut.yaml.
python train.py \
  --config-name apps/colmap_3dgut.yaml \
  path="$DATASET" \
  out_dir="$OUT/train_3dgrut" \
  experiment_name="$SEQ" \
  dataset.downsample_factor=1

# --- export the trained model to PLY (the swappable interchange format, §3.3) ---
# 3dgrut writes a checkpoint; convert to a standard 3DGS PLY for the USD export step.
# (Exact export entrypoint is in threedgrut/export/README.md for your commit.)
echo "[train_splat] training done. Next: scripts/export_splat_usd.sh $SEQ"
echo "[train_splat] NOTE (§7.3): depth supervision is UNVERIFIED in 3DGRUT. If Gate 4 shows"
echo "              weak geometry, swap to a depth-regularized gsplat/nerfstudio trainer that"
echo "              reads this same COLMAP dataset and emits a PLY (this is the expected path"
echo "              with LiDAR excluded — the depth stream is the only geometric signal)."
