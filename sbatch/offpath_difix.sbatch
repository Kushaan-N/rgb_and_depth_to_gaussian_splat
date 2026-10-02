#!/bin/bash
# Off-path benchmark + Difix3D+ (NVIDIA, CVPR 2025) for one prepped sequence. One A100 job.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/offpath_difix.sbatch configs/<seq>.yaml
#
# Holds out a spatial cluster of the trajectory (scripts/offpath_split.py) and scores, on those
# never-seen-from-nearby views:
#   baseline       3DGRUT on the remaining frames (pipeline settings, LiDAR-seeded init)
#   baseline+      baseline renders cleaned by Difix (post-render only, the paper's "+")
#   control        baseline resumed for the same extra iterations as Difix3D, real frames only
#   difix3d        baseline resumed with progressive Difix-fixed novel views (difix3d_views.py)
#   difix3d+       difix3d renders cleaned by Difix
# Needs the Difix venv (env/make_difix_venv.sh). Results: <out>/pipeline/offpath/report/summary.json
#SBATCH -J offpath-difix
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; DPY="$DIFIX_VENV/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; W="$O/work"; REP="$O/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
ROUND_ITERS="${ROUND_ITERS:-1500}"; STEP_M="${STEP_M:-0.3}"; EXTRA_ROUNDS="${EXTRA_ROUNDS:-2}"; REPEAT="${REAL_REPEAT:-3}"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$W" "$REP"

[ -f "$O/split.json" ] || "$CPUPY" scripts/offpath_split.py --config "$CFG"
SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"

# train <data dir> <run dir> <total iterations> [resume ckpt]  -> echoes the new ckpt_last.pt
train() {
  local extra=(); [ -n "${4:-}" ] && extra=(resume="$4")
  ( cd "$THREEDGRUT_DIR" && "$GPY" train.py --config-name "$APP" path="$1" out_dir="$2" experiment_name=run \
      dataset.downsample_factor=1 n_iterations="$3" dataset.test_split_interval=100000 "${extra[@]}" ) \
    > "$2.log" 2>&1 || { tail -30 "$2.log" >&2; exit 1; }
  ls -t "$2"/run/*/ckpt_last.pt | head -1
}
score() {   # score <ckpt> <tag> [save dir]
  local sv=(); [ -n "${3:-}" ] && sv=(--save-dir "$3")
  "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "$O/test" --tag "$2" --json-out "$REP/$2.json" "${sv[@]}" | tail -1
}
plus() {    # plus <render dir> <tag>  : Difix post-render on the test renders, then score
  "$DPY" scripts/difix_images.py --in-dir "$1" --out-dir "$1_difix" --refs "$W/refs_test.json"
  "$GPY" scripts/eval_views.py --pred-dir "$1_difix" --dataset "$O/test" --tag "$2" --json-out "$REP/$2.json" | tail -1
}
"$CPUPY" scripts/difix3d_views.py refs --train "$O/train_data" --views "$O/test" --out "$W/refs_test.json"

echo "== baseline"
CK0="$(ls -t "$W"/baseline/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK0" ] || CK0="$(train "$O/train_data" "$W/baseline" "$ITERS")"
score "$CK0" baseline "$W/renders_baseline"
plus "$W/renders_baseline" baseline_plus

echo "== difix3d (progressive novel views)"
rm -f "$W/state.json"; CK="$CK0"; TOTAL="$ITERS"; r=0; after=0
while :; do
  r=$((r + 1)); R="$W/round_$r"
  st="$("$CPUPY" scripts/difix3d_views.py novel --train "$O/train_data" --targets "$O/test" --state "$W/state.json" \
        --out "$R/novel" --step-m "$STEP_M" --scale "$SCALE" | tee /dev/stderr | tail -1)"
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$R/novel" --no-metrics --save-dir "$R/render" | tail -1
  "$DPY" scripts/difix_images.py --in-dir "$R/render" --out-dir "$R/fixed" --refs "$R/novel/refs.json"
  "$CPUPY" scripts/difix3d_views.py trainset --train "$O/train_data" --fixed "$R/fixed" --novel "$R/novel" \
      --out "$R/train_data" --real-repeat "$REPEAT"
  TOTAL=$((TOTAL + ROUND_ITERS))
  CK="$(train "$R/train_data" "$R/train" "$TOTAL" "$CK")"
  echo "round $r -> $CK"
  if [ "$st" = DIFIX3D_ALL_REACHED ]; then after=$((after + 1)); fi
  [ "$after" -ge "$EXTRA_ROUNDS" ] && break
  [ "$r" -ge 20 ] && { echo "stopping after 20 rounds"; break; }
done
score "$CK" difix3d "$W/renders_difix3d"
plus "$W/renders_difix3d" difix3d_plus

echo "== control: same extra iterations ($((TOTAL - ITERS))), real frames only"
CKC="$(train "$O/train_data" "$W/control" "$TOTAL" "$CK0")"
score "$CKC" control

"$CPUPY" - "$REP" "$O/split.json" "$r" "$TOTAL" <<'PY'
import json, os, sys
rep, split, rounds, total = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4])
arms = ["baseline", "baseline_plus", "control", "difix3d", "difix3d_plus"]
res = {a: json.load(open(os.path.join(rep, a + ".json"))) for a in arms if os.path.exists(os.path.join(rep, a + ".json"))}
out = {"split": {k: split[k] for k in ("train", "test", "test_to_nearest_train_m")}, "rounds": rounds,
       "total_iterations": total, "arms": {a: {k: r[k] for k in ("psnr", "ssim", "lpips", "n")} for a, r in res.items()}}
json.dump(out, open(os.path.join(rep, "summary.json"), "w"), indent=2)
print(f"{'arm':14s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s}")
for a, r in out["arms"].items():
    print(f"{a:14s} {r['psnr']:7.2f} {r['ssim']:7.4f} {r['lpips']:7.4f}")
PY
echo "OFFPATH_DIFIX_DONE"
