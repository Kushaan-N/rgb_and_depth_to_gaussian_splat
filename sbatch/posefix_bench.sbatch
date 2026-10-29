#!/bin/bash
# Photometric camera-pose refinement (gsplat pose_opt, scripts/refine_poses_gsplat.py) before 3DGRUT training,
# benchmarked on the on-path (every-8th hold-out) and off-path (1 m spatial hold-out) splits:
#  1. refine the training poses of $P/onpath/data_orig (3DGRUT's test_split_interval=8 rule) and
#     $P/offpath/train_data (interval 100000), correcting the held-out poses from temporal neighbours only;
#     write refined train datasets + corrected copies of the test datasets ($P/posefix/<split>/{data,test})
#  2. rebuild the LiDAR depth targets there with the refined poses (when the static cloud exists)
#  3. train 3DGRUT on the refined datasets without / with LiDAR depth loss (patched worktree)
#  4. score candidates on the CORRECTED test sets (gate) and on the original ones (reference); re-score the
#     baselines (on-path arm_A/arm_B, off-path baseline/depth_l0.05) on the original test sets so every json
#     carries the same metrics; gate each candidate vs its baseline (scripts/accept_gate.py)
#  5. diagnostic only (never gated): test-time pose alignment (refine_poses_gsplat.py tta) of every baseline
#     and candidate splat to its own test set, then re-scored with 3DGRUT
#  6. GT | baseline | candidate previews, pose-correction magnitudes, summary -> $P/posefix/report
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/posefix-%j.out sbatch/posefix_bench.sbatch configs/<seq>.yaml
# Env: SPLITS="onpath offpath", REFINE_ITERS=15000, POSE_LR=1e-5, REFINE_DEPTH=0.05, LAMBDA=0.05, TTA=1
#SBATCH -J posefix-bench
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
GSPLAT_VENV="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}"; GSPY="$GSPLAT_VENV/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
SPLITS="${SPLITS:-onpath offpath}"; REFINE_ITERS="${REFINE_ITERS:-15000}"; POSE_LR="${POSE_LR:-1e-5}"
REFINE_DEPTH="${REFINE_DEPTH:-0.05}"; TTA="${TTA:-1}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; PF="$P/posefix"; REP="$PF/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
STATIC="$P/offpath/depth/lidar_static.ply"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$PF" "$REP"
bash env/make_gsplat_venv.sh | tail -2; bash env/make_3dgrut_depth.sh
"$GSPY" --version

# per split: source train dataset, 3DGRUT split interval, original test set, baseline (no depth) / (depth) checkpoints
declare -A SRC=([onpath]="$P/onpath/data_orig" [offpath]="$P/offpath/train_data")
declare -A IV=([onpath]=8 [offpath]=100000)
declare -A TEST=([onpath]="$P/onpath/test" [offpath]="$P/offpath/test")
declare -A B0=([onpath]="$P/onpath/arm_A" [offpath]="$P/offpath/work/baseline")
declare -A B1=([onpath]="$P/onpath/arm_B" [offpath]="$P/offpath/work/depth_l${LAMBDA}")
ck() { ls -t "$1"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true; }
score() { "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "$2" --tag "$3" --json-out "$REP/$3.json" "${@:4}" | tail -1; }

for S in $SPLITS; do
  D="$PF/$S"; mkdir -p "$D"
  [ -d "${SRC[$S]}/sparse/0" ] || { echo "skip $S: no ${SRC[$S]}"; continue; }
  echo "== [$S] 1. pose refinement (gsplat, $REFINE_ITERS iters, pose lr $POSE_LR, depth $REFINE_DEPTH)"
  if [ ! -f "$D/data/posefix.json" ]; then
    "$GSPY" scripts/refine_poses_gsplat.py refine --dataset "${SRC[$S]}" --out "$D/data" \
      --test-split-interval "${IV[$S]}" --test-dataset "${TEST[$S]}" "$D/test" --iters "$REFINE_ITERS" \
      --pose-lr "$POSE_LR" --depth-lambda "$REFINE_DEPTH" --sim3-json "$P/sim3.json" > "$D/refine.log" 2>&1 \
      || { tail -40 "$D/refine.log"; exit 1; }
  fi
  grep -E "step +(0|[0-9]*000) |gauge|translation|rotation_deg" "$D/refine.log" | tail -14 || true
  echo "== [$S] 2. depth targets with the refined poses"
  if [ -f "$STATIC" ] && [ ! -f "$D/.depth_rebuilt" ]; then
    for ds in "$D/data" "$D/test"; do
      "$GPY" scripts/build_depth_targets.py --dataset "$ds" --cloud "$STATIC" --sim3-json "$P/sim3.json" | tail -1
    done; touch "$D/.depth_rebuilt"
  fi
  echo "== [$S] 3. 3DGRUT on the refined poses"
  for arm in nodepth depth; do
    RUN="$D/arm_$arm"; CK="$(ck "$RUN")"; UD=false; [ "$arm" = depth ] && UD=true
    if [ -z "$CK" ]; then
      ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$D/data" out_dir="$RUN" experiment_name=run \
          dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="${IV[$S]}" \
          loss.use_depth="$UD" loss.lambda_depth="$LAMBDA" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
      CK="$(ck "$RUN")"
    fi
  done
  echo "== [$S] 4. scores"
  for b in 0 1; do
    if [ "$b" = 0 ]; then BR="${B0[$S]}"; arm=nodepth; else BR="${B1[$S]}"; arm=depth; fi
    BCK="$(ck "$BR")"; CCK="$(ck "$D/arm_$arm")"
    [ -n "$BCK" ] || { echo "no baseline checkpoint in $BR"; exit 1; }
    score "$BCK" "${TEST[$S]}" "${S}_base_$arm" --save-dir "$D/renders/base_$arm"
    score "$BCK" "$D/test" "${S}_base_${arm}_at_corrected"
    score "$CCK" "$D/test" "${S}_cand_$arm" --save-dir "$D/renders/cand_$arm"
    score "$CCK" "${TEST[$S]}" "${S}_cand_${arm}_at_original"
  done
  if [ "$TTA" = 1 ]; then
    echo "== [$S] 5. diagnostic: test-time pose alignment of each splat to its own test set"
    for b in 0 1; do
      if [ "$b" = 0 ]; then BR="${B0[$S]}"; arm=nodepth; else BR="${B1[$S]}"; arm=depth; fi
      for who in base cand; do
        if [ "$who" = base ]; then R="$BR"; T="${TEST[$S]}"; else R="$D/arm_$arm"; T="$D/test"; fi
        PLY="$D/ply/${who}_$arm.ply"; mkdir -p "$D/ply"
        [ -f "$PLY" ] || { [ -f "$R/splat.ply" ] && ln -sf "$R/splat.ply" "$PLY"; } || \
          "$GPY" scripts/export_gaussian_ply.py --checkpoint "$(ck "$R")" --out "$PLY" | tail -1
        A="$D/tta/${who}_$arm"
        [ -f "$A/tta.json" ] || "$GSPY" scripts/refine_poses_gsplat.py tta --ply "$PLY" --dataset "$T" --out "$A" | tail -2
        score "$(ck "$R")" "$A" "${S}_${who}_${arm}_tta"
      done
    done
  fi
done

echo "== 6. gate, previews, summary"
GATE_ARGS=()
declare -A BJ=([onpath_nodepth]=onpath_base_nodepth [onpath_depth]=onpath_base_depth
               [offpath_nodepth]=offpath_base_nodepth [offpath_depth]=offpath_base_depth)
for arm in nodepth depth; do
  pairs=()
  for S in $SPLITS; do
    [ -f "$REP/${S}_cand_$arm.json" ] && pairs+=(--pair "$S" "$REP/${BJ[${S}_$arm]}.json" "$REP/${S}_cand_$arm.json")
  done
  [ ${#pairs[@]} -gt 0 ] || continue
  set +e; "$CPUPY" scripts/accept_gate.py "${pairs[@]}" --out "$REP/gate_$arm.json" | tail -8; rc=$?; set -e
  echo "GATE $arm: exit $rc (0 accept, 3 reject)"
done
"$CPUPY" - "$PF" "$REP" "$SPLITS" <<'PY'
import cv2, json, os, sys, numpy as np
pf, rep, splits = sys.argv[1], sys.argv[2], sys.argv[3].split()
J = lambda t: json.load(open(os.path.join(rep, t + ".json"))) if os.path.exists(os.path.join(rep, t + ".json")) else {}
keys = ("psnr", "ssim", "lpips", "psnr_observed", "psnr_unobserved", "depth_absrel")
summ = {"splits": {}, "gates": {a: J("gate_" + a) and {k: J("gate_" + a)[k] for k in ("accept", "failures")} for a in ("nodepth", "depth")}}
for S in splits:
    pj = os.path.join(pf, S, "data", "posefix.json")
    if not os.path.exists(pj):
        continue
    pfx = json.load(open(pj))
    s = {"pose_correction": {k: pfx.get(k) for k in ("gauge", "train_correction", "train_correction_before_gauge_removal",
                                                      "interp_correction", "interp_gap_s", "gsplat_psnr_heldout_in_dataset",
                                                      "gsplat_psnr_train")},
         "test_correction": {os.path.basename(k): v for k, v in pfx.get("test_datasets", {}).items()}, "scores": {}}
    for arm in ("nodepth", "depth"):
        for t in (f"{S}_base_{arm}", f"{S}_base_{arm}_at_corrected", f"{S}_cand_{arm}", f"{S}_cand_{arm}_at_original",
                  f"{S}_base_{arm}_tta", f"{S}_cand_{arm}_tta"):
            r = J(t)
            if r:
                s["scores"][t] = {k: r[k] for k in keys if k in r}
    summ["splits"][S] = s
    # previews: GT | baseline | candidate (+ 4x zoomed |diff| row) for 4 frames spread over the test set
    tdir = os.path.join(pf, S, "test", "images")
    names = sorted(n for n in os.listdir(tdir) if n.endswith(".png") and not n.endswith(("_covis.png", "_mask.png")))
    rows = []
    for n in names[:: max(1, len(names) // 4)][:4]:
        gt = cv2.imread(os.path.join(tdir, n)); ims = [gt]
        for arm in ("nodepth", "depth"):
            for who in ("base", "cand"):
                f = os.path.join(pf, S, "renders", f"{who}_{arm}", n)
                ims.append(cv2.imread(f) if os.path.exists(f) else np.zeros_like(gt))
        for im, lab in zip(ims, ["GT " + n[:16], "base", "posefix", "base+depth", "posefix+depth"]):
            cv2.putText(im, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(np.hstack(ims))
    if rows:
        cv2.imwrite(os.path.join(rep, f"preview_{S}.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
        h, w = rows[0].shape[0], rows[0].shape[1] // 5                       # 2x crop of the image centre
        crop = [r[h // 4: 3 * h // 4, :] for r in rows]
        crop = [np.hstack([cv2.resize(c[:, i * w + w // 4: i * w + 3 * w // 4], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                           for i in range(5)]) for c in crop]
        cv2.imwrite(os.path.join(rep, f"preview_{S}_crop.jpg"), cv2.resize(np.vstack(crop), None, fx=0.5, fy=0.5))
json.dump(summ, open(os.path.join(rep, "posefix_summary.json"), "w"), indent=2)
for S, s in summ["splits"].items():
    tc = s["pose_correction"]["train_correction"]
    print(f"[{S}] train-pose correction: {tc}  gauge: {s['pose_correction']['gauge']}")
    print(f"[{S}] gsplat PSNR non-train frames (input/gauge-only/corrected): {s['pose_correction']['gsplat_psnr_heldout_in_dataset']}")
    for k, v in s["test_correction"].items():
        print(f"[{S}] test '{k}': correction {v['correction']} gap {v['gap_s']} gsplat {v['gsplat_psnr']}")
    print(f"{'tag':38s}" + "".join(f"{k:>16s}" for k in keys))
    for t, r in s["scores"].items():
        print(f"{t:38s}" + "".join(f"{r.get(k, float('nan')):16.4f}" for k in keys))
print("gates:", json.dumps(summ["gates"]))
PY
echo "POSEFIX_BENCH_DONE"
