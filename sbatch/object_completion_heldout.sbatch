#!/bin/bash
# Leave-arc-out test of object completion against REAL images of the faces it fills in.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/object_completion_heldout.sbatch configs/<seq>.yaml [configs/<other_seq>.yaml]
#  1. choose_arc.py        hold out the contiguous 20% of frames whose removal hides the most object surface
#                          the full recording did observe (those frames = ground truth for the "backs")
#  2. base splat           the standard recipe (LiDAR init + depth loss) on the remaining frames only
#  3. completion           complete_objects.py on that base, observedness from the remaining frames only
#                          (LiDAR depth + the base splat's rendered depth at every remaining frame), appearance
#                          from face textures rectified from the remaining real frames (face_textures.py);
#                          preserve_finetune.py (added gaussians only) then harm_filter.py (footprint
#                          attribution, thr 0.004, until every pose renders >= 50 dB vs the base) over the
#                          remaining frames + jittered copies (strict) and the held-out / other-recording POSES
#                          (render-only, base vs candidate, never their pixels; a completed surface may cover
#                          what the base sees through, nothing else may change)
#  4. scoring              held-out frames: (a) all of them, (b) those showing completed surfaces, with
#                          psnr_unobserved = pixels of completed surfaces; (c) the remaining (training)
#                          frames, which completion must leave unchanged; (d) with a 2nd config, frames of
#                          that recording (same scene) showing completed surfaces; base and completed splats
#  5. acceptance gate      accept_gate.py on all sets (never worse than the base on real images)
# Note: the LiDAR cloud (object geometry) uses every scan; only CAMERA coverage is held out, which is the
# real situation (the 360° LiDAR hits faces the forward camera never sees).
# Env: ARC_FRAC (0.2), TAG (output subdir, default v5), COMPLETE_ARGS (extra complete_objects.py flags, e.g.
# "--color-source nearest"), FINETUNE (1), FINETUNE_ARGS, HARM_ARGS, JITTER (1 copy per training pose).
#SBATCH -J object-completion-heldout
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"; OTHER="${2:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; GT="$OUT/colmap_train/sparse/0"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"
H="$P/objects_heldout"; A="$H/arc"; R="$H/${TAG:-v5}"; REP="$R/report"; X="$H/xviews"
for need in "$DATA/sparse/0/images.txt" "$STATIC"; do
  [ -e "$need" ] || { echo "missing $need (run onpath_ablation / offpath_depthloss first)"; exit 1; }
done
hostname; date; mkdir -p "$H" "$R" "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
module list 2>&1 | tail -n +1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
which "$GPY"; "$GPY" --version
bash env/make_3dgrut_depth.sh

echo "== 1. objects + held-out arc"
[ -f "$H/objects.json" ] || "$CPUPY" scripts/find_objects.py --cloud "$STATIC" --gt-model "$GT" --out "$H/objects"
[ -f "$A/arc.json" ] || "$GPY" scripts/choose_arc.py --objects "$H/objects" --dataset "$DATA" \
  --sim3-json "$P/sim3.json" --out "$A" --frac "${ARC_FRAC:-0.2}"
ls "$A"/heldout/images/*_covis.png >/dev/null 2>&1 || "$GPY" scripts/covis_masks.py --train "$A/train" --test "$A/heldout"
EXTRA=("$A/heldout")
if [ -n "$OTHER" ]; then
  OP="$("$CPUPY" scripts/cfg_get.py "$OTHER" paths.out_root)/pipeline"
  if [ ! -f "$X/.done" ]; then                                   # other recording's frames in this splat's frame
    "$CPUPY" scripts/cross_recording_views.py --src-dataset "$OP/dataset" --src-sim3 "$OP/sim3.json" \
      --dst-sim3 "$P/sim3.json" --out "$X" --every 2
    "$GPY" scripts/build_depth_targets.py --dataset "$X" --cloud "$STATIC" --sim3-json "$P/sim3.json"
    "$GPY" scripts/covis_masks.py --train "$A/train" --test "$X"     # observed = by the REMAINING frames
    touch "$X/.done"
  fi
  EXTRA+=("$X")
fi

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
nadded() { "$CPUPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('added_kept', d.get('added')))" "$1"; }
POSES=(--dataset "$A/train" --extra-poses "${EXTRA[@]}" --jitter "${JITTER:-1}" --sim3-json "$P/sim3.json")
CUR="$R/completed_raw.pt"; ADDED="$(nadded "$R/completed_raw_report.json")"
if [ "${FINETUNE:-1}" = 1 ] && [ "$ADDED" -gt 0 ]; then
  # shellcheck disable=SC2086
  "$GPY" scripts/preserve_finetune.py --base "$CK" --edited "$CUR" --added "$ADDED" "${POSES[@]}" \
    --out "$R/completed_ft.pt" ${FINETUNE_ARGS:-}
  CUR="$R/completed_ft.pt"; ADDED="$(nadded "$R/completed_ft_finetune.json")"
fi
# measured, not predicted: drop added gaussians that change any trusted pixel of any pose
# shellcheck disable=SC2086
"$GPY" scripts/harm_filter.py --base "$CK" --edited "$CUR" --added "$ADDED" "${POSES[@]}" --out "$R/completed.pt" ${HARM_ARGS:-}
ADDED="$(nadded "$R/completed_harm.json")"
for k in unseen report samples; do
  f="$R/completed_raw_$k"; for ext in npz json; do [ -f "$f.$ext" ] && cp "$f.$ext" "$R/completed_$k.$ext"; done
done
[ -d "$R/completed_raw_textures" ] && rm -rf "$R/textures" && mv "$R/completed_raw_textures" "$R/textures"
"$GPY" scripts/object_back_masks.py --views "$A/heldout" --checkpoint "$R/completed.pt" --added "$ADDED" --out "$R/heldout_backs"
[ -n "$OTHER" ] && "$GPY" scripts/object_back_masks.py --views "$X" --checkpoint "$R/completed.pt" --added "$ADDED" --out "$R/xrec_backs"

echo "== 4. scoring on held-out real frames (and the training frames, which must not change)"
cp "$H/train_base.json" "$REP/train_base.json"
"$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$A/train" --tag completed --json-out "$REP/train_completed.json" | tail -1
has() { ls "$1"/images/*_covis.png >/dev/null 2>&1; }
has "$R/heldout_backs" || echo "no held-out view shows completed surfaces: nothing to score them against there"
for m in base completed; do
  ck="$CK"; [ "$m" = completed ] && ck="$R/completed.pt"
  "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$A/heldout" --tag "$m" --json-out "$REP/all_$m.json" | tail -1
  if has "$R/heldout_backs"; then
    "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$R/heldout_backs" --tag "$m" \
      --json-out "$REP/backs_$m.json" --save-dir "$R/renders_$m" | tail -1
  fi
  if [ -n "$OTHER" ] && has "$R/xrec_backs"; then
    "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$R/xrec_backs" --tag "$m" \
      --json-out "$REP/xrec_$m.json" --save-dir "$R/xrenders_$m" | tail -1
  fi
done

echo "== 5. acceptance gate"
PAIRS=(--pair heldout_all "$REP/all_base.json" "$REP/all_completed.json" --pair train "$REP/train_base.json" "$REP/train_completed.json")
[ -f "$REP/backs_completed.json" ] && PAIRS+=(--pair heldout_backs "$REP/backs_base.json" "$REP/backs_completed.json")
[ -f "$REP/xrec_completed.json" ] && PAIRS+=(--pair other_recording "$REP/xrec_base.json" "$REP/xrec_completed.json")
"$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate.json" || true
"$CPUPY" - "$R" "$REP" "$A/arc.json" <<'PY'
import json, os, sys, cv2, numpy as np
R, rep, arc = sys.argv[1:4]
keys = [k + "_" + m for k in ("all", "backs", "xrec", "train") for m in ("base", "completed")]
r = {k: json.load(open(os.path.join(rep, k + ".json"))) for k in keys if os.path.exists(os.path.join(rep, k + ".json"))}
print(f"{'':16s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'unseen PSNR':>13s}{'seen PSNR':>11s}{'n':>5s}")
for k, v in r.items():
    print(f"{k:16s}{v['psnr']:8.2f}{v['ssim']:8.4f}{v['lpips']:8.4f}{v.get('psnr_unobserved') or float('nan'):13.2f}"
          f"{v.get('psnr_observed') or float('nan'):11.2f}{v['n']:5d}")
jl = lambda f: json.load(open(f)) if os.path.exists(f) else None
json.dump({"scores": r, "arc": json.load(open(arc)), "gate": jl(os.path.join(rep, "gate.json")),
           "completion": jl(os.path.join(R, "completed_report.json")), "finetune": jl(os.path.join(R, "completed_ft_finetune.json")),
           "harm": jl(os.path.join(R, "completed_harm.json"))}, open(os.path.join(rep, "summary.json"), "w"), indent=2)
for backs, rb, rc, title, out in (("heldout_backs", "renders_base", "renders_completed", "real (held out)", "preview.jpg"),
                                   ("xrec_backs", "xrenders_base", "xrenders_completed", "real (other recording)", "preview_xrec.jpg")):
    d = os.path.join(R, backs, "images")
    if not os.path.isdir(os.path.join(R, rc)):
        continue
    cand = sorted(((np.mean(cv2.imread(os.path.join(d, f), 0) == 128), f[:-10]) for f in os.listdir(d) if f.endswith("_covis.png")), reverse=True)
    rows = []
    for frac, stem in cand[:6]:
        real = cv2.imread(os.path.join(d, stem + ".png")); m = cv2.imread(os.path.join(d, stem + "_covis.png"), 0) == 128
        ims = [real, cv2.imread(os.path.join(R, rb, stem + ".png")), cv2.imread(os.path.join(R, rc, stem + ".png"))]
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for x in ims: cv2.drawContours(x, cnt, -1, (0, 255, 255), 1)
        ys, xs = np.nonzero(m)                                       # zoom on the completed surfaces
        y0, y1, x0, x1 = max(ys.min() - 40, 0), min(ys.max() + 40, m.shape[0]), max(xs.min() - 40, 0), min(xs.max() + 40, m.shape[1])
        zm = [cv2.resize(x[y0:y1, x0:x1], (320, 240)) for x in ims]
        ims = [cv2.resize(x, (320, 240)) for x in ims]
        for x, t in zip(ims, [title, "base splat", "completed"]): cv2.putText(x, t, (4, 18), 0, 0.5, (0, 0, 255), 2)
        rows.append(np.vstack([np.hstack(ims), np.hstack(zm)]))
    if rows: cv2.imwrite(os.path.join(R, out), np.vstack(rows)); print("preview ->", os.path.join(R, out))
PY
echo "OBJECT_COMPLETION_HELDOUT_DONE"
