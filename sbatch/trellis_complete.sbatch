#!/bin/bash
# Structure-conditioned TRELLIS appearance for the unseen backs of free-standing objects (scripts/trellis_complete.py),
# scored against REAL images and gated (never worse than the base splat). Two modes:
#   heldout  leave-arc-out harness (reuses $P/objects_heldout/{arc,base,objects,splat_depth} of
#            object_completion_heldout.sbatch; output $P/objects_heldout/trellis_<TAG>):
#            generate -> merge -> harm_filter -> score held-out frames (all / showing completed backs) and the
#            training frames -> accept_gate
#     sbatch -A $CEAR_SLURM_ACCOUNT -o logs/trellis-heldout_%j.out sbatch/trellis_complete.sbatch heldout configs/<seq>.yaml
#   full     the all-frames splat ($P/worldmodel_full/ours/run/*/ckpt_last.pt), scored on another recording of the
#            same scene (cross_recording_views.py) + the training frames -> gate -> export ONLY on pass to
#            <out>/exports/objects_trellis{,_metric}.ply (output $P/objects_trellis[_<TAG>])
#     sbatch -A $CEAR_SLURM_ACCOUNT -o logs/trellis-full_%j.out sbatch/trellis_complete.sbatch full configs/<seq>.yaml configs/<other>.yaml
# Both write orbit previews (novel views around every object, base | TRELLIS-completed [| copy-completion]):
# no real view shows the backs, so plausibility has to be judged by eye.
# Env: TAG (default v1), GEN_ARGS (extra `generate` flags, e.g. "--views 8 --mode multidiffusion"), MERGE_ARGS,
#      HARM_ARGS (extra harm_filter.py flags), TRELLIS_VENV ($CEAR_WS/venv-trellis, env/make_trellis_venv.sh).
#SBATCH -J trellis-complete
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
MODE="$1"; CFG="$2"; OTHER="${3:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
TPY="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}/bin/python"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-hub}"
[ -x "$TPY" ] || { echo "no TRELLIS venv: run env/make_trellis_venv.sh first"; exit 1; }
get() { "$CPUPY" scripts/cfg_get.py "$1" "$2"; }
OUT="$(get "$CFG" paths.out_root)"; P="$OUT/pipeline"; GT="$OUT/colmap_train/sparse/0"; E="$OUT/exports"
DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"; TAG="${TAG:-v1}"
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$TPY" --version; "$GPY" --version

case "$MODE" in
  heldout)
    H="$P/objects_heldout"; A="$H/arc"; R="$H/trellis_$TAG"; OBJ="$H/objects"; TRAIN="$A/train"; SD="$H/splat_depth"
    CK="$(ls -t "$H"/base/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    for need in "$OBJ.json" "$A/arc.json" "$TRAIN/sparse/0/images.txt" "$CK"; do
      [ -e "$need" ] || { echo "missing $need: run sbatch/object_completion_heldout.sbatch first"; exit 1; }
    done
    COPY="$(ls -t "$H"/v*/completed.pt 2>/dev/null | head -1 || true)" ;;
  full)
    [ -n "$OTHER" ] || { echo "full mode needs a second config (another recording of the same scene)"; exit 1; }
    OOUT="$(get "$OTHER" paths.out_root)"; OP="$OOUT/pipeline"
    R="$P/objects_trellis${TAG:+_$TAG}"; TRAIN="$DATA"; SD="$R/splat_depth"
    CK="$(ls -t "$P"/worldmodel_full/ours/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    [ -n "$CK" ] || { echo "no all-frames splat: run worldmodel_full.sbatch (step 1) first"; exit 1; }
    OBJ="$P/objects/objects"; [ -f "$OBJ.json" ] || OBJ="$R/objects"
    COPY="$( [ -f "$P/objects/completed.pt" ] && echo "$P/objects/completed.pt" || true)" ;;
  *) echo "usage: trellis_complete.sbatch heldout|full CFG [OTHER_CFG]"; exit 1 ;;
esac
REP="$R/report"; mkdir -p "$R" "$REP"
export CUDA_LAUNCH_BLOCKING=1      # edited checkpoints: 3DGUT's tracer faults asynchronously otherwise
echo "== mode $MODE  base $CK  -> $R"

echo "== 1. objects + base splat depth at the training frames"
[ -f "$OBJ.json" ] || "$CPUPY" scripts/find_objects.py --cloud "$STATIC" --gt-model "$GT" --out "$OBJ"
TB="$REP/train_base.json"
if [ "$MODE" = heldout ] && [ -f "$H/train_base.json" ]; then cp "$H/train_base.json" "$TB"; fi
if [ ! -f "$TB" ] || [ "$(ls "$SD" 2>/dev/null | wc -l)" -lt "$(awk 'NF>=10 && $1 !~ /^#/' "$TRAIN/sparse/0/images.txt" | wc -l)" ]; then
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TRAIN" --tag base --json-out "$TB" --save-depth-dir "$SD" | tail -1
fi

echo "== 2. TRELLIS stage 2 on the measured structure, conditioned on real crops (venv-trellis)"
# shellcheck disable=SC2086
"$TPY" scripts/trellis_complete.py generate --objects "$OBJ" --dataset "$TRAIN" --splat-depth-dir "$SD" \
  --sim3-json "$P/sim3.json" --out-dir "$R/gen" ${GEN_ARGS:-}
echo "== 3. merge: exact inverse normalisation, keep only unseen measured faces, colour-calibrate (3DGRUT venv)"
# shellcheck disable=SC2086
"$GPY" scripts/trellis_complete.py merge --gen-dir "$R/gen" --objects "$OBJ" --checkpoint "$CK" --dataset "$TRAIN" \
  --splat-depth-dir "$SD" --sim3-json "$P/sim3.json" --out "$R/completed_raw.pt" ${MERGE_ARGS:-}
mv "$R/completed_raw_unseen.npz" "$R/completed_unseen.npz"; mv "$R/completed_raw_report.json" "$R/completed_report.json"
ADDED="$("$CPUPY" -c "import json;print(json.load(open('$R/completed_report.json'))['added'])")"
[ "$ADDED" -gt 0 ] || { echo "TRELLIS added nothing on unseen measured faces: nothing to test"; exit 0; }
# shellcheck disable=SC2086
"$GPY" scripts/harm_filter.py --base "$CK" --edited "$R/completed_raw.pt" --added "$ADDED" --dataset "$TRAIN" \
  --out "$R/completed.pt" ${HARM_ARGS:-}

echo "== 4. scoring on real images"
"$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$TRAIN" --tag trellis --json-out "$REP/train_trellis.json" | tail -1
PAIRS=(--pair train "$TB" "$REP/train_trellis.json")
if [ "$MODE" = heldout ]; then
  ALL="$A/heldout"; "$GPY" scripts/object_back_masks.py --views "$ALL" --unseen "$R/completed_unseen.npz" --out "$R/backs"
else
  XV="$P/objects/xviews"
  if ! ls "$XV"/images/*_covis.png >/dev/null 2>&1; then
    XV="$R/xviews"
    "$CPUPY" scripts/cross_recording_views.py --src-dataset "$OP/dataset" --src-sim3 "$OP/sim3.json" \
      --dst-sim3 "$P/sim3.json" --out "$XV" --every 2
    "$GPY" scripts/build_depth_targets.py --dataset "$XV" --cloud "$STATIC" --sim3-json "$P/sim3.json"
    "$GPY" scripts/covis_masks.py --train "$DATA" --test "$XV"
  fi
  ALL="$XV"; "$GPY" scripts/object_back_masks.py --views "$XV" --unseen "$R/completed_unseen.npz" --out "$R/backs"
fi
for m in base trellis; do
  ck="$CK"; [ "$m" = trellis ] && ck="$R/completed.pt"
  [ -f "$REP/all_$m.json" ] && [ "$m" = base ] || \
    "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$ALL" --tag "$m" --json-out "$REP/all_$m.json" | tail -1
  if ls "$R"/backs/images/*_covis.png >/dev/null 2>&1; then
    "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$R/backs" --tag "$m" --json-out "$REP/backs_$m.json" \
      --save-dir "$R/renders_$m" | tail -1
  fi
done
SET=heldout; [ "$MODE" = full ] && SET=other_recording
PAIRS+=(--pair "${SET}_all" "$REP/all_base.json" "$REP/all_trellis.json")
[ -f "$REP/backs_trellis.json" ] && PAIRS+=(--pair "${SET}_backs" "$REP/backs_base.json" "$REP/backs_trellis.json") \
  || echo "no $SET view shows the completed backs: nothing to score them against (regression checks only)"

echo "== 5. acceptance gate"
ACCEPT=0; "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate.json" && ACCEPT=1 || true
if [ "$MODE" = full ]; then
  rm -f "$E/objects_trellis.ply" "$E/objects_trellis_metric.ply"
  if [ "$ACCEPT" = 1 ]; then
    mkdir -p "$E"
    "$GPY" scripts/export_gaussian_ply.py --checkpoint "$R/completed.pt" --out "$E/objects_trellis.ply"
    "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
      --in-ply "$E/objects_trellis.ply" --out-ply "$E/objects_trellis_metric.ply"
  else
    echo "TRELLIS completion rejected by the gate: no objects_trellis export (the base splat stays the deliverable)"
  fi
fi

echo "== 6. orbit previews (novel views around each object; no real view shows the backs)"
"$GPY" scripts/trellis_complete.py orbit --objects "$OBJ" --dataset "$TRAIN" --sim3-json "$P/sim3.json" --out "$R/orbit"
RUNS=(base="$R/orbit_base" trellis="$R/orbit_trellis")
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$R/orbit" --no-metrics --save-dir "$R/orbit_base" | tail -1
"$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$R/orbit" --no-metrics --save-dir "$R/orbit_trellis" | tail -1
if [ -n "${COPY:-}" ]; then
  "$GPY" scripts/eval_views.py --checkpoint "$COPY" --dataset "$R/orbit" --no-metrics --save-dir "$R/orbit_copy" | tail -1 \
    && RUNS+=(copy="$R/orbit_copy")
fi
"$GPY" scripts/trellis_complete.py preview --orbit "$R/orbit" --renders "${RUNS[@]}" --out "$R/preview_orbit.jpg"

"$CPUPY" - "$R" "$REP" "$(dirname "$R")" <<'PY'
import glob, json, os, sys, cv2, numpy as np
R, rep, parent = sys.argv[1:4]
r = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(rep, "*.json"))) if not f.endswith(("gate.json", "summary.json"))}
row = lambda k, v: print(f"{k:16s}{v['psnr']:8.2f}{v['ssim']:8.4f}{v['lpips']:8.4f}{v.get('psnr_unobserved') or float('nan'):13.2f}{v.get('psnr_observed') or float('nan'):11.2f}{v['n']:5d}")
print(f"{'':16s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'unseen PSNR':>13s}{'seen PSNR':>11s}{'n':>5s}")
for k, v in r.items(): row(k, v)
gate = json.load(open(os.path.join(rep, "gate.json"))); cr = json.load(open(os.path.join(R, "completed_report.json")))
hj = os.path.join(R, "completed_harm.json"); harm = json.load(open(hj)) if os.path.exists(hj) else None
print("gate:", "ACCEPT" if gate["accept"] else "REJECT", gate.get("failures", []))
print("added (pre harm filter):", cr["added"], "| kept after harm filter:", harm and harm.get("added_kept"))
json.dump({"scores": r, "gate": gate, "completion": cr, "harm": harm,
           "generate": json.load(open(os.path.join(R, "gen", "generate.json")))}, open(os.path.join(rep, "summary.json"), "w"), indent=2)
others = sorted(f for f in glob.glob(os.path.join(parent, "*", "report", "summary.json")) if not f.startswith(R + "/"))
if others:                                                       # other completion runs on the same harness
    print("other runs (gate, all-set psnr/psnr_observed):")
    for f in others:
        s = json.load(open(f)); sc = s.get("scores", {})
        a = sc.get("all_completed") or sc.get("completed") or {}
        print(f"  {f.split('/')[-3]:14s} accept={s.get('gate', {}).get('accept')} psnr={a.get('psnr')} seen={a.get('psnr_observed')}")
tiles = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(R, "gen", "obj*_cond.jpg")))]
if tiles:
    w = max(t.shape[1] for t in tiles)
    cv2.imwrite(os.path.join(R, "preview_cond.jpg"), np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]))
PY
echo "TRELLIS_COMPLETE_DONE"
