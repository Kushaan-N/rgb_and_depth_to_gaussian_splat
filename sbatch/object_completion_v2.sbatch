#!/bin/bash
# Object completion v2 for the full all-frames splat: measured geometry, appearance from real-image face
# textures, provably harmless to what the recording saw; scored on REAL frames of another recording of the
# same, unchanged scene; exported only on a gate pass.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/object_completion_v2.sbatch configs/<seq>.yaml configs/<other_seq>.yaml
#  1. find_objects.py        free-standing objects in the dynamic-free LiDAR cloud
#  2. base renders           the base splat's depth at every training frame (observedness, texture occlusion)
#  3. other recording        its frames mapped into this splat's frame, LiDAR depth targets, observed masks
#  4. complete_objects.py    2.5D surfaces, discs on faces no training camera sees, colours from face textures
#                            rectified from the real training frames (face_textures.py: median, mirror, lattice
#                            tiling, exemplar synthesis)
#     preserve_finetune.py   added gaussians only: fade / blend whatever changes a trusted pixel
#     harm_filter.py         footprint attribution, thr 0.004, until every pose renders >= 50 dB vs the base:
#                            training frames + jittered copies (strict), other-recording POSES (render-only,
#                            completed surfaces may only cover what the base sees through)
#  5. scoring                other recording: frames showing completed surfaces (rendered mask, LiDAR-checked),
#                            psnr_unobserved = completed-surface pixels, psnr_observed = what this recording saw;
#                            training frames (must be unchanged); base vs completed
#  6. acceptance gate        exports <out>/exports/objects_completed{,_metric}.ply ONLY on a pass
# Env: TAG (output subdir, default v2), COMPLETE_ARGS, FINETUNE (1), FINETUNE_ARGS, HARM_ARGS, JITTER (1), XEVERY (2).
#SBATCH -J object-completion-v2
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"; OTHER="$2"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$1" "$2"; }
OUT="$(get "$CFG" paths.out_root)"; P="$OUT/pipeline"; GT="$OUT/colmap_train/sparse/0"; E="$OUT/exports"
OP="$(get "$OTHER" paths.out_root)/pipeline"
B="$P/objects_v2"; R="$B/${TAG:-v2}"; REP="$R/report"; X="$B/xviews"
DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"
CK="$(ls -t "$P"/worldmodel_full/ours/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK" ] || { echo "no all-frames splat: run worldmodel_full.sbatch (step 1) first"; exit 1; }
for need in "$DATA/sparse/0/images.txt" "$STATIC" "$OP/dataset/sparse/0" "$OP/sim3.json"; do
  [ -e "$need" ] || { echo "missing $need"; exit 1; }
done
hostname; date; mkdir -p "$B" "$R" "$REP" "$E"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
module list 2>&1 | tail -n +1
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
which "$GPY"; "$GPY" --version
# synchronous CUDA: edited checkpoints trigger an asynchronous illegal-memory-access in 3DGUT's tracer
export CUDA_LAUNCH_BLOCKING=1

echo "== 1. objects"
[ -f "$B/objects.json" ] || "$CPUPY" scripts/find_objects.py --cloud "$STATIC" --gt-model "$GT" --out "$B/objects"
echo "== 2. base splat depth at the training frames (base: $CK)"
[ -f "$B/train_base.json" ] || "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$DATA" --tag base \
  --json-out "$B/train_base.json" --save-depth-dir "$B/splat_depth" | tail -1
echo "== 3. frames of $(get "$OTHER" sequence.name) in this splat's frame"
if [ ! -f "$X/.done" ]; then
  "$CPUPY" scripts/cross_recording_views.py --src-dataset "$OP/dataset" --src-sim3 "$OP/sim3.json" \
    --dst-sim3 "$P/sim3.json" --out "$X" --every "${XEVERY:-2}"
  "$GPY" scripts/build_depth_targets.py --dataset "$X" --cloud "$STATIC" --sim3-json "$P/sim3.json"
  "$GPY" scripts/covis_masks.py --train "$DATA" --test "$X"
  touch "$X/.done"
fi

echo "== 4. completion"
# shellcheck disable=SC2086
"$GPY" scripts/complete_objects.py --objects "$B/objects" --checkpoint "$CK" --dataset "$DATA" \
  --sim3-json "$P/sim3.json" --out "$R/completed_raw.pt" --splat-depth-dir "$B/splat_depth" ${COMPLETE_ARGS:-}
nadded() { "$CPUPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('added_kept', d.get('added')))" "$1"; }
POSES=(--dataset "$DATA" --extra-poses "$X" --jitter "${JITTER:-1}" --sim3-json "$P/sim3.json")
CUR="$R/completed_raw.pt"; ADDED="$(nadded "$R/completed_raw_report.json")"
if [ "${FINETUNE:-1}" = 1 ] && [ "$ADDED" -gt 0 ]; then
  # shellcheck disable=SC2086
  "$GPY" scripts/preserve_finetune.py --base "$CK" --edited "$CUR" --added "$ADDED" "${POSES[@]}" \
    --out "$R/completed_ft.pt" ${FINETUNE_ARGS:-}
  CUR="$R/completed_ft.pt"; ADDED="$(nadded "$R/completed_ft_finetune.json")"
fi
# shellcheck disable=SC2086
"$GPY" scripts/harm_filter.py --base "$CK" --edited "$CUR" --added "$ADDED" "${POSES[@]}" --out "$R/completed.pt" ${HARM_ARGS:-}
ADDED="$(nadded "$R/completed_harm.json")"
for k in unseen report samples; do
  f="$R/completed_raw_$k"; for ext in npz json; do [ -f "$f.$ext" ] && cp "$f.$ext" "$R/completed_$k.$ext"; done
done
[ -d "$R/completed_raw_textures" ] && rm -rf "$R/textures" && mv "$R/completed_raw_textures" "$R/textures"
"$GPY" scripts/object_back_masks.py --views "$X" --checkpoint "$R/completed.pt" --added "$ADDED" --out "$R/xviews_backs"

echo "== 5. scoring"
cp "$B/train_base.json" "$REP/train_base.json"
"$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$DATA" --tag completed --json-out "$REP/train_completed.json" | tail -1
PAIRS=(--pair train "$REP/train_base.json" "$REP/train_completed.json")
if ls "$R"/xviews_backs/images/*_covis.png >/dev/null 2>&1; then
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$R/xviews_backs" --tag base --json-out "$REP/base.json" \
    --save-dir "$R/renders_base" | tail -1
  "$GPY" scripts/eval_views.py --checkpoint "$R/completed.pt" --dataset "$R/xviews_backs" --tag completed \
    --json-out "$REP/completed.json" --save-dir "$R/renders_completed" | tail -1
  PAIRS+=(--pair other_recording "$REP/base.json" "$REP/completed.json")
else
  echo "no other-recording frame shows completed surfaces: their appearance cannot be scored"
fi

echo "== 6. acceptance gate: ship only if not worse than the base splat on real images"
rm -f "$E/objects_completed.ply" "$E/objects_completed_metric.ply"
if [ "$ADDED" -gt 0 ] && [ -f "$REP/completed.json" ] && "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate.json"; then
  "$GPY" scripts/export_gaussian_ply.py --checkpoint "$R/completed.pt" --out "$E/objects_completed.ply"
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
    --in-ply "$E/objects_completed.ply" --out-ply "$E/objects_completed_metric.ply"
  echo "GATE ACCEPT: exported $E/objects_completed{,_metric}.ply"
else
  [ -f "$REP/gate.json" ] || "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate.json" || true
  echo "completion rejected (or nothing scoreable): no objects_completed export (the base splat stays the deliverable)"
fi
"$CPUPY" - "$R" "$REP" "$E/objects_completed_v2_preview.jpg" <<'PY'
import json, os, sys, cv2, numpy as np
R, rep, out = sys.argv[1:4]
r = {k: json.load(open(os.path.join(rep, k + ".json"))) for k in ("base", "completed", "train_base", "train_completed")
     if os.path.exists(os.path.join(rep, k + ".json"))}
print(f"{'':16s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'backs PSNR':>12s}{'seen PSNR':>11s}{'n':>5s}")
for k, v in r.items():
    print(f"{k:16s}{v['psnr']:8.2f}{v['ssim']:8.4f}{v['lpips']:8.4f}{v.get('psnr_unobserved') or float('nan'):12.2f}"
          f"{v.get('psnr_observed') or float('nan'):11.2f}{v['n']:5d}")
jl = lambda f: json.load(open(f)) if os.path.exists(f) else None
json.dump({"scores": r, "gate": jl(os.path.join(rep, "gate.json")), "completion": jl(os.path.join(R, "completed_report.json")),
           "finetune": jl(os.path.join(R, "completed_ft_finetune.json")), "harm": jl(os.path.join(R, "completed_harm.json"))},
          open(os.path.join(rep, "summary.json"), "w"), indent=2)
d = os.path.join(R, "xviews_backs", "images")
if os.path.isdir(os.path.join(R, "renders_completed")):
    cand = sorted(((np.mean(cv2.imread(os.path.join(d, f), 0) == 128), f[:-10]) for f in os.listdir(d) if f.endswith("_covis.png")), reverse=True)
    rows = []
    for frac, stem in cand[:6]:
        real = cv2.imread(os.path.join(d, stem + ".png")); m = cv2.imread(os.path.join(d, stem + "_covis.png"), 0) == 128
        ims = [real, cv2.imread(os.path.join(R, "renders_base", stem + ".png")), cv2.imread(os.path.join(R, "renders_completed", stem + ".png"))]
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for x in ims: cv2.drawContours(x, cnt, -1, (0, 255, 255), 1)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(ys.min() - 40, 0), min(ys.max() + 40, m.shape[0]), max(xs.min() - 40, 0), min(xs.max() + 40, m.shape[1])
        zm = [cv2.resize(x[y0:y1, x0:x1], (320, 240)) for x in ims]
        ims = [cv2.resize(x, (320, 240)) for x in ims]
        for x, t in zip(ims, [f"real (other rec.) {frac:.1%}", "base splat", "completed"]): cv2.putText(x, t, (4, 18), 0, 0.5, (0, 0, 255), 2)
        rows.append(np.vstack([np.hstack(ims), np.hstack(zm)]))
    if rows: cv2.imwrite(out, np.vstack(rows)); print("preview ->", out)
PY
echo "OBJECT_COMPLETION_V2_DONE"
