#!/bin/bash
# Retraining noise of the baseline splat: the same recipe with different seed_initialization, scored on
# the on-path held-out frames (every 8th) and the off-path spatial hold-out. Tells whether a gain/loss is
# larger than what a retrain alone produces (the acceptance gate's tolerances are -0.05 dB / +0.002 LPIPS).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/seed_variance.sbatch configs/<seq>.yaml
# Env: SEEDS ("1 2"), DEPTH (false|true: LiDAR depth loss arms).
#SBATCH -J seed-variance
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"; DEPTH="${DEPTH:-false}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; S="$P/seedvar"; REP="$S/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
bash env/make_3dgrut_depth.sh

# split name -> train dataset, held-out interval, test dataset
declare -A TRAIN=([onpath]="$P/onpath/data_orig" [offpath]="$P/offpath/train_data")
declare -A IVL=([onpath]=8 [offpath]=100000)
declare -A TEST=([onpath]="$P/onpath/test" [offpath]="$P/offpath/test")
for split in onpath offpath; do
  for seed in ${SEEDS:-1 2}; do
    tag="${split}_depth${DEPTH}_seed$seed"; RUN="$S/$tag"
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    if [ -z "$CK" ]; then
      echo "== $tag"
      ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="${TRAIN[$split]}" out_dir="$RUN" experiment_name=run \
          dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="${IVL[$split]}" \
          seed_initialization="$seed" loss.use_depth="$DEPTH" loss.lambda_depth="$LAMBDA" ) > "$RUN.log" 2>&1 \
        || { tail -40 "$RUN.log"; exit 1; }
      CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
    fi
    "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "${TEST[$split]}" --tag "$tag" --json-out "$REP/$tag.json" | tail -1
  done
done

"$CPUPY" - "$P" "$REP" "$DEPTH" <<'PY'
import glob, json, os, sys
import numpy as np
P, rep, depth = sys.argv[1:4]
ref = {"onpath": os.path.join(P, "onpath/report", "arm_B.json" if depth == "true" else "arm_A.json"),
       "offpath": os.path.join(P, "offpath/report", "depth_l0.05.json" if depth == "true" else "baseline.json")}
out = {}
for split, rf in ref.items():
    runs = [json.load(open(rf))] + [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rep, f"{split}_depth{depth}_seed*.json")))]
    out[split] = {k: {"values": [r[k] for r in runs if k in r], "std": float(np.std([r[k] for r in runs if k in r], ddof=1)),
                      "range": float(np.ptp([r[k] for r in runs if k in r]))}
                  for k in ("psnr", "ssim", "lpips", "psnr_observed", "psnr_unobserved") if sum(k in r for r in runs) > 1}
    for k, v in out[split].items():
        print(f"[seedvar] {split:8s} {k:16s} values {v['values']}  std {v['std']:.4f}  range {v['range']:.4f}")
json.dump(out, open(os.path.join(rep, f"seedvar_depth{depth}.json"), "w"), indent=2)
PY
echo "SEED_VARIANCE_DONE"
