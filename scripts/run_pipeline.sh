#!/usr/bin/env bash
# One command, any configured sequence: calibration -> poses/depth -> LiDAR/depth cloud + collider
# -> COLMAP poses -> depth-seeded 3DGRUT splat -> floor fill -> metric splat -> report.
#
#   bash scripts/run_pipeline.sh mocap2_well-lit_trot      # resolves configs/<name>.yaml
#   bash scripts/run_pipeline.sh path/to/config.yaml
#   bash scripts/run_pipeline.sh <seq> --force             # delete <out_root>/pipeline and rerun
#
# Submits three chained SLURM jobs (CPU prep -> GPU train -> CPU post) so the GPU is only held while
# training. Stages are idempotent (finished work is skipped) and skip themselves when an input is
# absent (no LiDAR bag, no depth). New dataset = new configs/datasets/<name>.yaml; no code changes.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }
CFG="$1"; FORCE="${2:-}"
[ -f "$CFG" ] || CFG="configs/$1.yaml"
[ -f "$CFG" ] || { echo "no config for '$1' (looked for $1 and $CFG)"; exit 2; }
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
source env/cear_env.sh >/dev/null
OUT="$(python3 scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
if [ "$FORCE" = "--force" ]; then echo "removing $P"; rm -rf "$P"; fi

ACC=(); if [ -n "${CEAR_SLURM_ACCOUNT:-}" ]; then ACC=(-A "$CEAR_SLURM_ACCOUNT"); fi
LOGS="$OUT/logs"; mkdir -p "$LOGS"
J1=$(sbatch --parsable "${ACC[@]}" -o "$LOGS/pipe-prep_%j.out" sbatch/pipeline_prep.sbatch "$CFG")
J2=$(sbatch --parsable "${ACC[@]}" --dependency="afterok:$J1" -o "$LOGS/pipe-train_%j.out" sbatch/pipeline_train.sbatch "$CFG")
J3=$(sbatch --parsable "${ACC[@]}" --dependency="afterok:$J2" -o "$LOGS/pipe-post_%j.out" sbatch/pipeline_post.sbatch "$CFG")
cat <<EOF
submitted  prep=$J1 (CPU) -> train=$J2 (GPU) -> post=$J3 (CPU)
logs       $LOGS/pipe-{prep,train,post}_<jobid>.out
outputs    $P/summary.json
           $P/splat_filled.ply   (drop into SuperSplat)
           $P/splat_metric.ply   (metric frame, for Isaac)
           $P/collider/collider.obj
EOF
