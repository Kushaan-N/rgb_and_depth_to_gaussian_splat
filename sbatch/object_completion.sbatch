#!/bin/bash
# Object completion (measured geometry + copied appearance; no generative model) for one sequence, scored
# against REAL images of another recording of the same, unchanged scene.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/object_completion.sbatch configs/<seq>.yaml configs/<other_seq>.yaml
#  1. find_objects.py      free-standing objects in the dynamic-free LiDAR cloud
#  2. complete_objects.py  2.5D surfaces, unseen samples filled by mirror/row copy, junk removed, discs added
#                          (base = the all-frames splat with LiDAR depth loss from worldmodel_full.sbatch)
#  3. exports: <out>/exports/objects_completed{,_metric}.ply
#  4. evaluation: the other recording's frames mapped into this splat's frame; pixels showing completed
#     surfaces (unseen here) vs pixels this recording observed; base and completed splats scored on both
#SBATCH -J object-completion
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
OOUT="$(get "$OTHER" paths.out_root)"; OP="$OOUT/pipeline"
B="$P/objects"; REP="$B/report"; DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"
CK="$(ls -t "$P"/worldmodel_full/ours/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK" ] || { echo "no all-frames splat: run worldmodel_full.sbatch (step 1) first"; exit 1; }
hostname; date; mkdir -p "$B" "$REP" "$E"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

echo "== 1. objects"
"$CPUPY" scripts/find_objects.py --cloud "$STATIC" --gt-model "$GT" --out "$B/objects"
echo "== 2. completion (base: $CK)"
"$GPY" scripts/complete_objects.py --objects "$B/objects" --checkpoint "$CK" --dataset "$DATA" \
  --sim3-json "$P/sim3.json" --out "$B/completed.pt"
echo "== 3. exports"
"$GPY" scripts/export_gaussian_ply.py --checkpoint "$B/completed.pt" --out "$E/objects_completed.ply"
"$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
  --in-ply "$E/objects_completed.ply" --out-ply "$E/objects_completed_metric.ply"

echo "== 4. evaluation against $(get "$OTHER" sequence.name)"
"$CPUPY" scripts/cross_recording_views.py --src-dataset "$OP/dataset" --src-sim3 "$OP/sim3.json" \
  --dst-sim3 "$P/sim3.json" --out "$B/xviews" --every 2
"$GPY" scripts/build_depth_targets.py --dataset "$B/xviews" --cloud "$STATIC" --sim3-json "$P/sim3.json"
"$GPY" scripts/covis_masks.py --train "$DATA" --test "$B/xviews"
"$GPY" scripts/object_back_masks.py --views "$B/xviews" --unseen "$B/completed_unseen.npz" --out "$B/xviews_backs"
# synchronous CUDA: edited checkpoints trigger an asynchronous illegal-memory-access in 3DGUT's tracer
export CUDA_LAUNCH_BLOCKING=1
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$B/xviews_backs" --tag base --json-out "$REP/base.json" \
  --save-dir "$B/renders_base" | tail -1
"$GPY" scripts/eval_views.py --checkpoint "$B/completed.pt" --dataset "$B/xviews_backs" --tag completed \
  --json-out "$REP/completed.json" --save-dir "$B/renders_completed" | tail -1

"$CPUPY" - "$B" "$REP" "$E/objects_completed_preview.jpg" <<'PY'
import json, os, sys, cv2, numpy as np
B, rep, out = sys.argv[1:4]
r = {k: json.load(open(os.path.join(rep, k + ".json"))) for k in ("base", "completed")}
cr = json.load(open(os.path.join(B, "completed_report.json")))
print(f"{'':10s}{'PSNR':>8s}{'LPIPS':>8s}{'backs PSNR':>12s}{'seen PSNR':>11s}")
for k, v in r.items():
    print(f"{k:10s}{v['psnr']:8.2f}{v['lpips']:8.4f}{v.get('psnr_unobserved', float('nan')):12.2f}{v.get('psnr_observed', float('nan')):11.2f}")
json.dump({"scores": r, "completion": cr}, open(os.path.join(rep, "summary.json"), "w"), indent=2)
d = os.path.join(B, "xviews_backs", "images")
cand = sorted(((np.mean(cv2.imread(os.path.join(d, f), 0) == 128), f[:-10]) for f in os.listdir(d) if f.endswith("_covis.png")), reverse=True)
rows = []
for frac, stem in cand[:5]:
    real = cv2.imread(os.path.join(d, stem + ".png")); m = cv2.imread(os.path.join(d, stem + "_covis.png"), 0) == 128
    ims = [real, cv2.imread(os.path.join(B, "renders_base", stem + ".png")), cv2.imread(os.path.join(B, "renders_completed", stem + ".png"))]
    for x in ims:
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(x, cnt, -1, (0, 255, 255), 1)
    ims = [cv2.resize(x, (320, 240)) for x in ims]
    for x, t in zip(ims, ["real (other recording)", "base splat", "completed"]): cv2.putText(x, t, (4, 18), 0, 0.5, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
if rows: cv2.imwrite(out, np.vstack(rows)); print("preview ->", out)
PY
echo "OBJECT_COMPLETION_DONE"
