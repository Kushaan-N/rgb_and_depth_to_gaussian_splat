#!/bin/bash
# Haze in SuperSplat's default / off-path views (scripts/clean_haze.py) on the viewer deliverable <NAME>_inpainted.ply:
#   1. probe: the live SuperSplat's default camera + an orbit around its focal point, for the refaxes file the user
#      opens (cameras mapped back into the pipeline frame) and for the pipeline-frame file
#   2. diagnose: per-gaussian training-view support (viewer-exact, all training frames), LiDAR / camera distances,
#      class shares of the probe views' pixels, renders with / without each candidate class
#   3. clean: unconstrained free-space gaussians removed, measured harm restore (no training pixel moves > 2/255)
#      -> <E>/<NAME>_clean{,_refaxes}.ply
#   4. verify in the live SuperSplat: on-path held-out score + gate vs the input, editor default/orbit views and
#      off-path MOVES (before | after montages), alpha-hole fractions
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/clean-haze-%j.out sbatch/clean_haze.sbatch configs/<seq>.yaml
# Env: NAME (viewer_igs60k_wd), IN (<E>/<NAME>_inpainted.ply), TAG (clean), STAGES ("probe diagnose clean verify"),
#      E_OUT (where <NAME>_<TAG>{,_refaxes}.ply go; default <E>; use a scratch dir for informational variants),
#      CLEAN_ARGS (clean flags, e.g. "--max-train-w 0.004 --min-lidar-dist 0.1"), MOVES ("0 -0.3 1.0;0 -0.6 1.5"),
#      EVERY_OFF (40), BASE_JSON (on-path live score of IN; default supersplat_check_current/report/<NAME>_inpainted.json)
#SBATCH -J clean-haze
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
GPY="$THREEDGRUT_DIR/.venv/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; NAME="${NAME:-viewer_igs60k_wd}"; TAG="${TAG:-clean}"
IN="${IN:-$E/${NAME}_inpainted.ply}"; IN_REF="${IN%.ply}_refaxes.ply"; D="$P/clean_haze/$NAME"; DATA="$P/onpath/data_orig"
FRAME="$(get pipeline.export_frame_model "")"; REF_DS="$(get pipeline.viewer_reference_dataset "")"
STAGES="${STAGES:-probe diagnose clean verify}"; MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"
BASE_JSON="${BASE_JSON:-$P/supersplat_check_current/report/${NAME}_inpainted.json}"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
EO="${E_OUT:-$E}"; mkdir -p "$EO"; OUTPLY="$EO/${NAME}_$TAG.ply"; OUTREF="$EO/${NAME}_${TAG}_refaxes.ply"
hostname; date; mkdir -p "$D"; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true; module list 2>&1 | tail -2 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash env/make_viewer_venv.sh; bash env/make_gsplat_viewer_venv.sh
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
[ -f "$IN_REF" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$IN" --out-ply "$IN_REF"
# refaxes = s R pipeline + t (the same solve sim3_align_splat.py bakes into every *_refaxes.ply)
[ -f "$D/refaxes_sim3.json" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --sim3-json "$D/refaxes_sim3.json"

if has probe; then
  echo "== 1. probe the live editor's default view + orbit"
  "$VPY" scripts/clean_haze.py probe --ply "$IN_REF" --sim3-json "$D/refaxes_sim3.json" --out-dir "$D/probe_ref"
  "$VPY" scripts/clean_haze.py probe --ply "$IN" --out-dir "$D/probe_pipe"
fi
if has diagnose; then
  echo "== 2. diagnose"
  for pr in probe_ref probe_pipe; do
    # shellcheck disable=SC2086
    "$SPY" scripts/clean_haze.py diagnose --ply "$IN" --dataset "$DATA" --cams "$D/$pr/cams.json" --cloud "$CLOUD" \
      --sim3-json "$P/sim3.json" --base-ply "$E/$NAME.ply" --stats "$D/stats.npz" --out-dir "$D/diag_$pr" ${DIAG_ARGS:-} \
      --renders "untrained,weak,nofrustum,fill,untrained_lidar_gt10cm,weak_lidar_gt10cm,lidar_gt10cm,nearcam_lt30cm,big_gt20cm,needle,nofrustum_or_big"
  done
fi
if has clean; then
  echo "== 3. clean + measured harm restore"
  # shellcheck disable=SC2086
  "$SPY" scripts/clean_haze.py clean --ply "$IN" --dataset "$DATA" --stats "$D/stats.npz" --out "$OUTPLY" ${CLEAN_ARGS:-}
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$OUTPLY" --out-ply "$OUTREF"
fi
if has verify; then
  echo "== 4. live SuperSplat checks"
  V="$D/live_$TAG"; RT="$P/viewer_bench/reference_dataset"; mkdir -p "$V/report"
  "$VPY" scripts/viewer_shots.py --engine current --ply "$OUTPLY" --dataset "$P/onpath/test" --out-dir "$V/onpath" --bg black
  "$GPY" scripts/eval_views.py --pred-dir "$V/onpath" --dataset "$P/onpath/test" --tag "${NAME}_$TAG" --json-out "$V/report/onpath.json" | tail -1
  set +e; "$CPUPY" scripts/accept_gate.py --pair onpath_live "$BASE_JSON" "$V/report/onpath.json" --out "$V/report/gate.json"; echo "gate exit $?"; set -e
  "$VPY" scripts/clean_haze.py probe --ply "$OUTREF" --out-dir "$V/probe_ref_after"
  i=0; IFS=';' read -ra MV <<< "$MOVES"
  for m in "${MV[@]}"; do
    i=$((i + 1))
    # base renders shared by every TAG; linked into this TAG's folder for the montages
    [ -f "$D/live_base/off$i/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --ply "$IN_REF" --dataset "$RT" \
      --every "$EVERY_OFF" --move $m --out-dir "$D/live_base/off$i" --bg black $([ $i = 1 ] && echo --editor)
    mkdir -p "$V/off$i"; ln -sfn "$D/live_base/off$i" "$V/off$i/base"
    [ -f "$V/off$i/after/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --ply "$OUTREF" --dataset "$RT" \
      --every "$EVERY_OFF" --move $m --out-dir "$V/off$i/after" --bg black $([ $i = 1 ] && echo --editor)
  done
  "$CPUPY" - "$V" "$D/probe_ref/editor" "$V/probe_ref_after/editor" <<'PY'
import json, os, sys, glob, cv2, numpy as np
V, eb, ea = sys.argv[1:4]
def lab(x, t): x = x.copy(); cv2.putText(x, t, (8, 28), 0, 0.9, (0, 0, 255), 2); return x
rows = []
for f in sorted(glob.glob(os.path.join(eb, "*.png"))):
    if f.endswith("_alpha.png"): continue
    g = os.path.join(ea, os.path.basename(f))
    if os.path.exists(g):
        n = os.path.basename(f)[:-4]
        rows.append(np.hstack([lab(cv2.resize(cv2.imread(f), (640, 384)), n + " before"), lab(cv2.resize(cv2.imread(g), (640, 384)), "after")]))
if rows: cv2.imwrite(os.path.join(V, "editor_before_after.jpg"), np.vstack(rows)); print("->", os.path.join(V, "editor_before_after.jpg"))
summ = {}
for d in sorted(glob.glob(os.path.join(V, "off*"))):
    rows = []
    for w in ("base", "after"):
        j = os.path.join(d, w, "_viewer.json")
        if os.path.exists(j): summ[f"{os.path.basename(d)}_{w}_alpha_hole"] = json.load(open(j)).get("alpha_hole_frac")
    names = sorted(x for x in os.listdir(os.path.join(d, "base")) if x.endswith(".png") and not x.startswith("_"))
    for n in names:
        a, b = os.path.join(d, "base", n), os.path.join(d, "after", n)
        if os.path.exists(b):
            rows.append(np.hstack([lab(cv2.resize(cv2.imread(a), (480, 360)), "before"), lab(cv2.resize(cv2.imread(b), (480, 360)), "after")]))
    if rows: cv2.imwrite(d + "_before_after.jpg", np.vstack(rows)); print("->", d + "_before_after.jpg")
j = os.path.join(V, "onpath", "_viewer.json")
if os.path.exists(j): summ["onpath_alpha_hole"] = json.load(open(j)).get("alpha_hole_frac")
json.dump(summ, open(os.path.join(V, "report", "alpha.json"), "w"), indent=1); print(json.dumps(summ))
PY
  ls -la "$OUTPLY" "$OUTREF"; md5sum "$OUTPLY" "$OUTREF"
fi
echo CLEAN_HAZE_DONE
