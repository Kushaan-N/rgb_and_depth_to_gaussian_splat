#!/bin/bash
# Leave-arc-out test of object completion against REAL images of the faces it fills in.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/object_completion_heldout.sbatch configs/<seq>.yaml
#  1. choose_arc.py        hold out the contiguous 20% of frames whose removal hides the most object surface
#                          the full recording did observe (those frames = ground truth for the "backs")
#  2. base splat           the standard recipe (LiDAR init + depth loss) on the remaining frames only
#  3. completion           complete_objects.py on that base, observedness from the remaining frames only
#                          (LiDAR depth + the base splat's rendered depth at every remaining frame), then
#                          harm_filter.py drops added gaussians that change any remaining frame's render
#  4. scoring              held-out frames: (a) all of them, (b) those showing completed surfaces, with
#                          psnr_unobserved = pixels of completed surfaces; (c) the remaining (training)
#                          frames, which completion must leave unchanged; base and completed splats
#  5. acceptance gate      accept_gate.py on all three sets (never worse than the base on real images)
# Note: the LiDAR cloud (object geometry) uses every scan; only CAMERA coverage is held out, which is the
# real situation (the 360° LiDAR hits faces the forward camera never sees).
# Env: ARC_FRAC (0.2), COMPLETE_ARGS (extra complete_objects.py flags), TAG (output subdir, default v1).
#SBATCH -J object-completion-heldout
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
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; GT="$OUT/colmap_train/sparse/0"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"
H="$P/objects_heldout"; A="$H/arc"; R="$H/${TAG:-v1}"; REP="$R/report"
for need in "$DATA/sparse/0/images.txt" "$STATIC"; do
  [ -e "$need" ] || { echo "missing $need (run onpath_ablation / offpath_depthloss first)"; exit 1; }
done
hostname; date; mkdir -p "$H" "$R" "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
bash env/make_3dgrut_depth.sh

echo "== 1. objects + held-out arc"
[ -f "$H/objects.json" ] || "$CPUPY" scripts/find_objects.py --cloud "$STATIC" --gt-model "$GT" --out "$H/objects"
[ -f "$A/arc.json" ] || "$GPY" scripts/choose_arc.py --objects "$H/objects" --dataset "$DATA" \
  --sim3-json "$P/sim3.json" --out "$A" --frac "${ARC_FRAC:-0.2}"
ls "$A"/heldout/images/*_covis.png >/dev/null 2>&1 || "$GPY" scripts/covis_masks.py --train "$A/train" --test "$A/heldout"

echo "== 2. base splat on the remaining frames"
CK="$(ls -t "$H"/base/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
if [ -z "$CK" ]; then
  ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$A/train" out_dir="$H/base" experiment_name=run \
      dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
      loss.use_depth=true loss.lambda_depth="$LAMBDA" ) > "$H/base.log" 2>&1 || { tail -40 "$H/base.log"; exit 1; }
  CK="$(ls -t "$H"/base/run/*/ckpt_last.pt | head -1)"
fi

echo "== 3. completion (base: $CK)"
# synchronous CUDA: edited checkpoints trigger an asynchronous illegal-memory-access in 3DGUT's tracer
export CUDA_LAUNCH_BLOCKING=1
[ -f "$H/train_base.json" ] || "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$A/train" --tag base \
  --json-out "$H/train_base.json" --save-depth-dir "$H/splat_depth" | tail -1
# shellcheck disable=SC2086
"$GPY" scripts/complete_objects.py --objects "$H/objects" --checkpoint "$CK" --dataset "$A/train" \
  --sim3-json "$P/sim3.json" --out "$R/completed_raw.pt" --splat-depth-dir "$H/splat_depth" ${COMPLETE_ARGS:-}
mv "$R/completed_raw_unseen.npz" "$R/completed_unseen.npz"; mv "$R/completed_raw_report.json" "$R/completed_report.json"
# measured, not predicted: drop added gaussians that change any training view
ADDED="$("$CPUPY" -c "import json;print(json.load(open('$R/completed_report.json'))['added'])")"
"$GPY" scripts/harm_filter.py --base "$CK" --edited "$R/completed_raw.pt" --added "$ADDED" --dataset "$A/train" \
  --out "$R/completed.pt"
"$GPY" scripts/object_back_masks.py --views "$A/heldout" --unseen "$R/completed_unseen.npz" --out "$R/heldout_backs"

echo "== 4. scoring on held-out real frames (and the training frames, which must not change)"
cp "$H/train_base.json" "$REP/train_base.json"
"$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$A/train" --tag completed --json-out "$REP/train_completed.json" | tail -1
BACKS=0; ls "$R"/heldout_backs/images/*_covis.png >/dev/null 2>&1 && BACKS=1
[ "$BACKS" = 1 ] || echo "no held-out view shows completed surfaces: nothing to score them against (seen-region checks only)"
for m in base completed; do
  ck="$CK"; [ "$m" = completed ] && ck="$R/completed.pt"
  "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$A/heldout" --tag "$m" --json-out "$REP/all_$m.json" | tail -1
  if [ "$BACKS" = 1 ]; then
    "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$R/heldout_backs" --tag "$m" \
      --json-out "$REP/backs_$m.json" --save-dir "$R/renders_$m" | tail -1
  fi
done

echo "== 5. acceptance gate"
"$CPUPY" scripts/accept_gate.py --pair heldout_all "$REP/all_base.json" "$REP/all_completed.json" \
  $(if [ "$BACKS" = 1 ]; then echo --pair heldout_backs "$REP/backs_base.json" "$REP/backs_completed.json"; fi) \
  --pair train "$REP/train_base.json" "$REP/train_completed.json" --out "$REP/gate.json" || true
"$CPUPY" - "$R" "$REP" "$A/arc.json" <<'PY'
import json, os, sys, cv2, numpy as np
R, rep, arc = sys.argv[1:4]
r = {k: json.load(open(os.path.join(rep, k + ".json"))) for k in ("all_base", "all_completed", "backs_base", "backs_completed", "train_base", "train_completed")
     if os.path.exists(os.path.join(rep, k + ".json"))}
print(f"{'':16s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'unseen PSNR':>13s}{'seen PSNR':>11s}")
for k, v in r.items():
    print(f"{k:16s}{v['psnr']:8.2f}{v['ssim']:8.4f}{v['lpips']:8.4f}{v.get('psnr_unobserved') or float('nan'):13.2f}{v.get('psnr_observed') or float('nan'):11.2f}")
json.dump({"scores": r, "arc": json.load(open(arc)), "gate": json.load(open(os.path.join(rep, "gate.json"))),
           "completion": json.load(open(os.path.join(R, "completed_report.json")))}, open(os.path.join(rep, "summary.json"), "w"), indent=2)
d = os.path.join(R, "heldout_backs", "images")
cand = [] if not os.path.isdir(d) else sorted(((np.mean(cv2.imread(os.path.join(d, f), 0) == 128), f[:-10]) for f in os.listdir(d) if f.endswith("_covis.png")), reverse=True)
rows = []
for frac, stem in cand[:6]:
    real = cv2.imread(os.path.join(d, stem + ".png")); m = cv2.imread(os.path.join(d, stem + "_covis.png"), 0) == 128
    ims = [real, cv2.imread(os.path.join(R, "renders_base", stem + ".png")), cv2.imread(os.path.join(R, "renders_completed", stem + ".png"))]
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for x in ims: cv2.drawContours(x, cnt, -1, (0, 255, 255), 1)
    ims = [cv2.resize(x, (320, 240)) for x in ims]
    for x, t in zip(ims, ["real (held out)", "base splat", "completed"]): cv2.putText(x, t, (4, 18), 0, 0.5, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
if rows: cv2.imwrite(os.path.join(R, "preview.jpg"), np.vstack(rows)); print("preview ->", os.path.join(R, "preview.jpg"))
PY
echo "OBJECT_COMPLETION_HELDOUT_DONE"
