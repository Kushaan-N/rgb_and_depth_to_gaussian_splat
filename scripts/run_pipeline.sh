#!/usr/bin/env bash
# One command, any configured sequence: raw data + calibration -> poses/depth -> LiDAR/depth cloud +
# collider -> COLMAP poses -> depth-seeded 3DGRUT splat -> floor fill -> metric splat -> report.
#
#   bash scripts/run_pipeline.sh mocap2_well-lit_trot                  # resolves configs/<name>.yaml
#   bash scripts/run_pipeline.sh path/to/config.yaml
#   bash scripts/run_pipeline.sh <seq> --force                         # delete <out_root>/pipeline, rerun
#   bash scripts/run_pipeline.sh <seq> --variant mcmc [--force]        # A/B: configs/variants/mcmc.yaml
#
# Submits chained SLURM jobs (CPU prep -> GPU train -> CPU post) so the GPU is only held while
# training. Stages are idempotent and skip themselves when an input is absent (no LiDAR bag, no depth).
# A --variant reuses the sequence's prep and runs only train + post into <out>/pipeline/variants/<name>.
# New dataset = new configs/datasets/<name>.yaml; new experiment = new configs/variants/<name>.yaml.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ $# -ge 1 ] || { sed -n '2,14p' "$0"; exit 2; }
SEQARG="$1"; shift
FORCE=""; VARIANT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --variant) VARIANT="${2:?--variant needs a name}"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
CFG="$SEQARG"
[ -f "$CFG" ] || CFG="configs/$SEQARG.yaml"
[ -f "$CFG" ] || { echo "no config for '$SEQARG' (looked for $SEQARG and $CFG)"; exit 2; }
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
source env/cear_env.sh >/dev/null
OUT="$(python3 scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
ACC=(); if [ -n "${CEAR_SLURM_ACCOUNT:-}" ]; then ACC=(-A "$CEAR_SLURM_ACCOUNT"); fi
LOGS="$OUT/logs"; mkdir -p "$LOGS"

if [ -n "$VARIANT" ]; then
  VCFG="$REPO/configs/variants/$VARIANT.yaml"
  if [ ! -f "$VCFG" ]; then
    echo "no variant '$VARIANT' (available: $(ls configs/variants | sed 's/\.yaml$//' | tr '\n' ' '))"; exit 2
  fi
  R="$P/variants/$VARIANT"
  if [ -n "$FORCE" ]; then echo "removing $R"; rm -rf "$R"; fi
  mkdir -p "$R"
  printf 'base: [%s, %s]\nvariant: %s\npipeline:\n  run_dir: %s\n' "$CFG" "$VCFG" "$VARIANT" "$R" > "$R/config.yaml"
  DEP=()
  if [ ! -e "$P/train_data" ]; then
    J1=$(sbatch --parsable "${ACC[@]}" -o "$LOGS/pipe-prep_%j.out" sbatch/pipeline_prep.sbatch "$CFG")
    DEP=(--dependency="afterok:$J1"); echo "prep not done yet -> queued prep=$J1 first"
  fi
  J2=$(sbatch --parsable "${ACC[@]}" "${DEP[@]}" -o "$LOGS/pipe-train-${VARIANT}_%j.out" sbatch/pipeline_train.sbatch "$R/config.yaml")
  J3=$(sbatch --parsable "${ACC[@]}" --dependency="afterok:$J2" -o "$LOGS/pipe-post-${VARIANT}_%j.out" sbatch/pipeline_post.sbatch "$R/config.yaml")
  echo "submitted  variant '$VARIANT': train=$J2 (GPU) -> post=$J3 (CPU)"
  echo "outputs    $R/summary.json"
  exit 0
fi

if [ -n "$FORCE" ]; then echo "removing $P"; rm -rf "$P"; fi
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
