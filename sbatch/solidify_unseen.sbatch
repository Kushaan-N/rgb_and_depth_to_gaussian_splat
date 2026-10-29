#!/bin/bash
# Solid surfaces from unseen angles (scripts/solidify_unseen.py) on the viewer deliverable <NAME>_inpainted.ply:
#   1. objects (find_objects.py), the input's depth at the training frames (render_ply_depth.py)
#   2. per-gaussian training-view support in the viewer-exact compositor -> unconstrained gaussians
#   3. build: unconstrained gaussians removed, objects closed (2.5D), discs on every surface no kept gaussian covers
#      and no camera sees; mirror colours for object discs; fill_inpaint.py (LaMa) for the rest; objtex
#   4. harm: viewer-exact, every training + held-out frame: removals restored / snapped needles reverted / discs dropped
#      until no view changes; shfit: minimum-norm view-dependent colour (no rainbow sparkles off the recorded path)
#   5. exports <E>/<NAME>_solid{,_refaxes}.ply
#   6. checks in the live SuperSplat (viewer_shots.py --engine current): on-path held-out score (must equal the
#      input's), orbit cameras around every object (before | after montages), off-path MOVES + editor default view;
#      proxies (alpha holes / depth vs LiDAR in object pixels, needle fraction) at the orbit cameras
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/solidify-%j.out sbatch/solidify_unseen.sbatch configs/<seq>.yaml
# Env: NAME (viewer_igs), IN (<E>/<NAME>_inpainted.ply), TAG (solid), MOVES ("0 -0.3 1.0;0 -0.6 1.5"), EVERY_OFF (40),
#      BUILD_ARGS (extra build flags), HARM_ARGS (extra harm + shfit flags, e.g. "--gt-aware" with TAG=solid_gt: a
#      training pixel may change only toward the real image), FORCE=1 (redo every stage). Stages reuse their outputs
#      in $P/solidify/<NAME>; per-TAG: harm_<TAG>.ply, live_<TAG>/. REFIT=1: before harm, the added discs + snapped needles
#      are fine-tuned (solidify_unseen.py refit, REFIT_ARGS; every input gaussian frozen) so fewer need reverting.
#      JOINT=1 (after a TAG=$BASETAG run): joint refit incl. the input's gaussians (lr x0.1 + anchor, REFIT_ARGS e.g.
#      "--lam-anchor 1e-3") and a per-view visibility budget (solidify_unseen.py budget, BUDGET_ARGS) instead of harm.
#SBATCH -J solidify
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
NAME="${NAME:-viewer_igs}"; TAG="${TAG:-solid}"; IN="${IN:-$E/${NAME}_inpainted.ply}"; S="$P/solidify/$NAME"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; RT="$P/viewer_bench/reference_dataset"; GT="$OUT/colmap_train/sparse/0"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"; VS=(scripts/viewer_shots.py --engine current --bg black)
SO=scripts/solidify_unseen.py; OUTPLY="$E/${NAME}_${TAG}.ply"
need() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$S"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
bash env/make_inpaint_venv.sh; bash env/make_viewer_venv.sh
[ -f "$IN" ] || { echo "missing $IN (run export_viewer.sbatch)"; exit 1; }
md5sum "$IN" | tee "$S/input.md5"

echo "== 1. objects + input depth at the training frames"
OBJ=(); if [ -f "$CLOUD" ] && [ -d "$GT" ]; then
  need "$S/objects.json" && "$CPUPY" scripts/find_objects.py --cloud "$CLOUD" --gt-model "$GT" --out "$S/objects"
  OBJ=(--objects "$S/objects")
fi
need "$S/depth" && "$SPY" scripts/render_ply_depth.py --ply "$IN" --dataset "$DATA" --out-dir "$S/depth"

echo "== 2. training-view support per gaussian"
need "$S/support.npz" && "$SPY" "$SO" support --ply "$IN" --dataset "$DATA" "$TEST" --out "$S/support.npz"
cat "$S/support.json" | head -40

echo "== 3. build (remove unconstrained, close objects, discs) + colours"
need "$S/raw.ply" && "$CPUPY" "$SO" build --ply "$IN" --support "$S/support.npz" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
  "${OBJ[@]}" --dataset "$DATA" --splat-depth-dir "$S/depth" --out-dir "$S" ${BUILD_ARGS:-}
need "$S/inpainted.ply" && "$IPY" scripts/fill_inpaint.py --splat-ply "$S/base_kept.ply" --fill-ply "$S/raw.ply" \
  --sim3-json "$P/sim3.json" --method lama --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" \
  --out-ply "$S/inpainted.ply" --debug-dir "$S/inpaint_debug" | tail -3
need "$S/textured.ply" && "$CPUPY" "$SO" objtex --ply "$S/inpainted.ply" --meta "$S/meta.npz" --out "$S/textured.ply"

CAND="$S/textured.ply"
if [ "${REFIT:-0}" = 1 ]; then
echo "== 4a. refit: discs + snapped needles fine-tuned (originals frozen) on the training views, surface prior, orbit terms"
ORB=(); if [ -f "$S/objects.json" ]; then
  need "$S/orbit/sparse/0/images.txt" && "$CPUPY" "$SO" orbit --objects "$S/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" --cloud "$CLOUD" --out "$S/orbit"
  ORB=(--orbit "$S/orbit")
fi
need "$S/refit_$TAG.ply" && "$SPY" "$SO" refit --ply "$S/textured.ply" --meta "$S/meta.npz" --dataset "$DATA" "$TEST" --sim3-json "$P/sim3.json" \
  "${OBJ[@]}" --cloud "$CLOUD" "${ORB[@]}" --out "$S/refit_$TAG.ply" ${REFIT_ARGS:-}
CAND="$S/refit_$TAG.ply"
fi
if [ "${JOINT:-0}" = 1 ]; then
echo "== 4J. joint refit (input gaussians at lr x0.1 + anchor; removals harm restored put back) + per-view visibility budget"
ORB=(); [ -f "$S/orbit/sparse/0/images.txt" ] && ORB=(--orbit "$S/orbit")
RM="$S/harm_${BASETAG:-solid}_harm_masks.npz"; [ -f "$RM" ] || { echo "missing $RM (run TAG=${BASETAG:-solid} first)"; exit 1; }
need "$S/refit_$TAG.ply" && "$SPY" "$SO" refit --ply "$S/textured.ply" --meta "$S/meta.npz" --dataset "$DATA" "$TEST" --sim3-json "$P/sim3.json" \
  "${OBJ[@]}" --cloud "$CLOUD" "${ORB[@]}" --train-all --input "$IN" --restore-masks "$RM" --out "$S/refit_$TAG.ply" ${REFIT_ARGS:-}
need "$S/budget_$TAG.ply" && "$SPY" "$SO" budget --input "$IN" --cand "$S/refit_$TAG.ply" --src "$S/refit_${TAG}_src.npz" \
  --dataset "$DATA" "$TEST" --out "$S/budget_$TAG.ply" ${BUDGET_ARGS:-}
need "$OUTPLY" && "$SPY" "$SO" shfit --ply "$S/budget_$TAG.ply" --ref "$S/budget_$TAG.ply" --dataset "$DATA" "$TEST" --out "$OUTPLY"
else
echo "== 4. harmlessness at every training / held-out frame (viewer-exact)"
need "$S/harm_$TAG.ply" && "$SPY" "$SO" harm --input "$IN" --cand "$CAND" --meta "$S/meta.npz" --dataset "$DATA" "$TEST" \
  --out "$S/harm_$TAG.ply" ${HARM_ARGS:-}
# minimum-norm SH: colours at the training directions kept, no rainbow extrapolation elsewhere (measured, reverted where seen)
need "$OUTPLY" && "$SPY" "$SO" shfit --ply "$S/harm_$TAG.ply" --ref "$IN" --dataset "$DATA" "$TEST" --out "$OUTPLY" ${HARM_ARGS:-}
fi

echo "== 5. exports"
if [ -n "$FRAME" ]; then
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$OUTPLY" \
    --out-ply "${OUTPLY%.ply}_refaxes.ply"
fi
INR="${IN%.ply}_refaxes.ply"
ls -la "$OUTPLY" "${OUTPLY%.ply}_refaxes.ply"; md5sum "$OUTPLY" "${OUTPLY%.ply}_refaxes.ply" | tee "$S/output_$TAG.md5"

echo "== 6. live SuperSplat checks"
L="$S/live_$TAG"; mkdir -p "$L"
for v in before after; do
  ply="$IN"; [ "$v" = after ] && ply="$OUTPLY"
  "$VPY" "${VS[@]}" --ply "$ply" --dataset "$TEST" --out-dir "$L/onpath/$v"
  "$GPY" scripts/eval_views.py --pred-dir "$L/onpath/$v" --dataset "$TEST" --tag "$v" --json-out "$L/onpath_$v.json" | tail -1
done
"$CPUPY" scripts/accept_gate.py --pair onpath "$L/onpath_before.json" "$L/onpath_after.json" --out "$L/gate.json" || true
if [ -f "$S/objects.json" ]; then
  "$CPUPY" "$SO" orbit --objects "$S/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" --cloud "$CLOUD" --out "$S/orbit"
  for v in before after; do
    ply="$IN"; [ "$v" = after ] && ply="$OUTPLY"
    "$VPY" "${VS[@]}" --ply "$ply" --dataset "$S/orbit" --out-dir "$L/orbit/$v"
  done
  "$CPUPY" "$SO" montage --dirs "before=$L/orbit/before" "after=$L/orbit/after" --out-dir "$L/orbit_montage"
  "$SPY" "$SO" eval --ply "before=$IN" "after=$OUTPLY" --orbit "$S/orbit" --objects "$S/objects" --sim3-json "$P/sim3.json" \
    --cloud "$CLOUD" --live "before=$L/orbit/before" "after=$L/orbit/after" --json-out "$L/proxies.json"
fi
if [ -f "$INR" ] && [ -f "$RT/sparse/0/images.txt" ]; then
  mi=0; IFS=';' read -ra MV <<< "$MOVES"
  for mv in "${MV[@]}"; do
    for v in before after; do
      ply="$INR"; [ "$v" = after ] && ply="${OUTPLY%.ply}_refaxes.ply"
      "$VPY" "${VS[@]}" --ply "$ply" --dataset "$RT" --every "$EVERY_OFF" --move $mv --out-dir "$L/offpath$mi/$v" \
        $([ "$mi" = 0 ] && echo --editor)
    done
    "$CPUPY" "$SO" montage --dirs "before=$L/offpath$mi/before" "after=$L/offpath$mi/after" --out-dir "$L/offpath${mi}_montage"
    mi=$((mi + 1))
  done
fi
"$CPUPY" - "$L" <<'PY'
import json, os, sys, cv2, numpy as np
L = sys.argv[1]; res = {}
for v in ("before", "after"):
    r = json.load(open(os.path.join(L, f"onpath_{v}.json"))); res[v] = {k: r[k] for k in ("psnr", "ssim", "lpips")}
on = sorted(f for f in os.listdir(os.path.join(L, "onpath", "after")) if f.endswith(".png"))
dd = [np.abs(cv2.imread(os.path.join(L, "onpath", "before", n)).astype(int) - cv2.imread(os.path.join(L, "onpath", "after", n)).astype(int)) for n in on]
res["onpath_pixel_diff"] = {"max": int(max(x.max() for x in dd)), "mean": round(float(np.mean([x.mean() for x in dd])), 5),
                            "frac_px_gt8": round(float(np.mean([(x.max(2) > 8).mean() for x in dd])), 6)}
for d in sorted(os.listdir(L)):
    j = os.path.join(L, d, "after", "_viewer.json"); jb = os.path.join(L, d, "before", "_viewer.json")
    if os.path.exists(j) and os.path.exists(jb):
        res[f"{d}_alpha_hole"] = {"before": json.load(open(jb)).get("alpha_hole_frac"), "after": json.load(open(j)).get("alpha_hole_frac")}
eds = [os.path.join(L, "offpath0", v, "_supersplat_editor.png") for v in ("before", "after")]
if all(map(os.path.exists, eds)):
    cv2.imwrite(os.path.join(L, "editor_before_after.jpg"), np.hstack([cv2.imread(e) for e in eds]))
print("SOLID_CHECK", json.dumps(res)); json.dump(res, open(os.path.join(L, "check.json"), "w"), indent=1)
PY
echo SOLIDIFY_DONE
