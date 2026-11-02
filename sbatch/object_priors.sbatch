#!/bin/bash
# Object priors on the delivered splat (scripts/object_priors/object_priors.py), gated by solidify_unseen.py's machinery:
#   0. objects (find_objects.py), the input's depth at the training frames, per-gaussian training-view support, orbit cams
#   1. build METHOD (prim | prim,sym | prim,sym,app | +LEARNED=<learned_prior.py npz>) -> base_kept/textured/meta
#   2. refit (REFIT=1): joint fine-tune of the new rows + the object zones' input gaussians (lr x0.1, anchored)
#   3. budget: per-view visibility budget vs the input (no visible cluster > 8/255, no view losing > 0.3 dB)
#   4. shfit (min-norm view-dependent colour) -> <E>/<NAME>_op_<TAG>.ply (+ _refaxes)
#   5. live SuperSplat: orbit before|after montages + proxies (alpha holes, depth within 5 cm, spikes), on-path held-out
#      scores in BOTH sort modes (black, grey, + DISTS) gated vs the input, off-path moved views + editor view
#   METHOD=prim,sym,app WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/object_priors.sbatch \
#     configs/mocap2_well-lit_trot.yaml
# Env: NAME (viewer_msd60k_wd_clean_ft2_solid3), IN (<E>/<NAME>.ply), METHOD, TAG (METHOD with , -> _), LEARNED,
#      BUILD_ARGS, REFIT (1), REFIT_ARGS ("--train-zone-m 0.25 --iters 2000"), BUDGET_ARGS ("--min-w 0.002 --iters 12"),
#      STOP_AFTER=build (prep + build only), MOVES ("0 -0.3 1.0;0 -0.6 1.5"), EVERY_OFF (40), FORCE=1 (redo every stage of this TAG).
#SBATCH -J obj-priors
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
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
NAME="${NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; IN="${IN:-$E/$NAME.ply}"; METHOD="${METHOD:-prim,sym,app}"
TAG="${TAG:-${METHOD//,/_}}"; D="$P/object_priors/$NAME"; T="$D/$TAG"; OUTPLY="$E/${NAME}_op_${TAG}.ply"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; RT="$P/viewer_bench/reference_dataset"; GT="$OUT/colmap_train/sparse/0"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
SO=scripts/solidify_unseen.py; OP=scripts/object_priors/object_priors.py; VS=(scripts/viewer_shots.py --engine current)
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"
need() { [ ! -e "$1" ]; }; needt() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$T"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
bash env/make_viewer_venv.sh | tail -1
[ -f "$IN" ] && [ -f "$CLOUD" ] && [ -d "$GT" ] || { echo "missing input / LiDAR cloud / metric model: object priors skipped"; exit 0; }
echo "== $NAME METHOD=$METHOD TAG=$TAG"

echo "== 0. objects, input depth, support, orbit cameras"
need "$D/objects.json" && "$CPUPY" scripts/find_objects.py --cloud "$CLOUD" --gt-model "$GT" --out "$D/objects"
need "$D/depth" && "$SPY" scripts/render_ply_depth.py --ply "$IN" --dataset "$DATA" --out-dir "$D/depth"
need "$D/support.npz" && "$SPY" "$SO" support --ply "$IN" --dataset "$DATA" "$TEST" --out "$D/support.npz"
need "$D/orbit/sparse/0/images.txt" && "$CPUPY" "$SO" orbit --objects "$D/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" \
  --cloud "$CLOUD" --out "$D/orbit"

echo "== 1. build"
LA=(); [ -n "${LEARNED:-}" ] && LA=(--learned-dir "$LEARNED")
needt "$T/meta.npz" && "$CPUPY" "$OP" build --ply "$IN" --support "$D/support.npz" --objects "$D/objects" --cloud "$CLOUD" \
  --sim3-json "$P/sim3.json" --dataset "$DATA" --splat-depth-dir "$D/depth" --method "$METHOD" "${LA[@]}" --out-dir "$T" ${BUILD_ARGS:-}
CAND="$T/textured.ply"; SRC="$T/src.npz"
[ "${STOP_AFTER:-}" = build ] && { echo OBJECT_PRIORS_BUILD_ONLY; exit 0; }
if [ "${REFIT:-1}" = 1 ]; then
  echo "== 2. joint refit (new rows + object-zone input gaussians, anchored)"
  needt "$T/refit.ply" && "$SPY" "$SO" refit --ply "$T/textured.ply" --meta "$T/meta.npz" --dataset "$DATA" "$TEST" \
    --sim3-json "$P/sim3.json" --objects "$D/objects" --cloud "$CLOUD" --orbit "$D/orbit" --train-all --out "$T/refit.ply" \
    ${REFIT_ARGS:---train-zone-m 0.25 --iters 2000}
  CAND="$T/refit.ply"; SRC="$T/refit_src.npz"
fi
echo "== 3. per-view visibility budget vs the input"
needt "$T/budget.ply" && "$SPY" "$SO" budget --input "$IN" --cand "$CAND" --src "$SRC" --dataset "$DATA" "$TEST" \
  --out "$T/budget.ply" ${BUDGET_ARGS:---min-w 0.002 --iters 12}
echo "== 4. shfit + exports"
needt "$OUTPLY" && "$SPY" "$SO" shfit --ply "$T/budget.ply" --ref "$T/budget.ply" --dataset "$DATA" "$TEST" --out "$OUTPLY"
[ -z "$FRAME" ] || { needt "${OUTPLY%.ply}_refaxes.ply" && "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
  --gt-model "$FRAME" --in-ply "$OUTPLY" --out-ply "${OUTPLY%.ply}_refaxes.ply"; }
ls -la "$OUTPLY"*; md5sum "$OUTPLY" | tee "$T/output.md5"

echo "== 5. live SuperSplat checks"
LI="$D/live_in"; L="$T/live"; mkdir -p "$LI" "$L" "$L/report"
score() {  # <shots dir with black/ grey/> <tag> <report dir>
  for bg in black grey; do
    [ -f "$3/${2}_$bg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$1/$bg" --dataset "$TEST" --tag "${2}_$bg" --json-out "$3/${2}_$bg.json" | tail -1
    PYTHONPATH="$PIQ_LIB" "$GPY" - "$1/$bg" "$TEST/images" "$3/${2}_$bg.json" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
if "dists" not in r:
    t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
    m = piq.DISTS(reduction="none").cuda(); v = []
    for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
        with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
    r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2)
print("SCORE", json.dumps({k: r[k] for k in ("tag", "psnr", "ssim", "lpips", "dists") if k in r}))
PY
  done
}
for m in depth radial; do
  [ -f "$LI/onpath_$m/grey/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black grey --sort-mode $m --ply "$IN" --dataset "$TEST" --out-dir "$LI/onpath_$m" | tail -1
  score "$LI/onpath_$m" "in_$m" "$LI"
  [ -f "$L/onpath_$m/grey/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black grey --sort-mode $m --ply "$OUTPLY" --dataset "$TEST" --out-dir "$L/onpath_$m" | tail -1
  score "$L/onpath_$m" "op_$m" "$L/report"
  "$CPUPY" scripts/accept_gate.py --pair black "$LI/in_${m}_black.json" "$L/report/op_${m}_black.json" --pair grey "$LI/in_${m}_grey.json" \
    "$L/report/op_${m}_grey.json" --out "$L/report/gate_$m.json" | tail -1 || true
  "$CPUPY" scripts/accept_gate.py --perceptual --pair black "$LI/in_${m}_black.json" "$L/report/op_${m}_black.json" --pair grey \
    "$LI/in_${m}_grey.json" "$L/report/op_${m}_grey.json" --out "$L/report/gate_${m}_perc.json" | tail -1 || true
done
"$CPUPY" "$SO" crops --before "$LI/onpath_depth/black" --after "$L/onpath_depth/black" --gt "$TEST/images" --out "$L/onpath_crops.jpg" || true
[ -f "$LI/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$IN" --dataset "$D/orbit" --out-dir "$LI/orbit" | tail -1
[ -f "$L/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$OUTPLY" --dataset "$D/orbit" --out-dir "$L/orbit" | tail -1
"$CPUPY" "$SO" montage --dirs "before=$LI/orbit" "after=$L/orbit" --out-dir "$L/orbit_montage"
"$SPY" "$SO" eval --ply "before=$IN" "after=$OUTPLY" --orbit "$D/orbit" --objects "$D/objects" --sim3-json "$P/sim3.json" \
  --cloud "$CLOUD" --live "before=$LI/orbit" "after=$L/orbit" --json-out "$L/proxies.json" | tail -1
INR="${IN%.ply}_refaxes.ply"
if [ -f "$INR" ] && [ -f "${OUTPLY%.ply}_refaxes.ply" ] && [ -f "$RT/sparse/0/images.txt" ]; then
  mi=0; IFS=';' read -ra MV <<< "$MOVES"
  for mv in "${MV[@]}"; do
    [ -f "$LI/offpath$mi/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$INR" --dataset "$RT" --every "$EVERY_OFF" --move $mv \
      --out-dir "$LI/offpath$mi" $([ "$mi" = 0 ] && echo --editor) | tail -1 || true
    "$VPY" "${VS[@]}" --bg black --ply "${OUTPLY%.ply}_refaxes.ply" --dataset "$RT" --every "$EVERY_OFF" --move $mv \
      --out-dir "$L/offpath$mi" $([ "$mi" = 0 ] && echo --editor) | tail -1 || true
    "$CPUPY" "$SO" montage --dirs "before=$LI/offpath$mi" "after=$L/offpath$mi" --out-dir "$L/offpath${mi}_montage" || true
    mi=$((mi + 1))
  done
fi
"$CPUPY" - "$L" "$LI" "$T" <<'PY'
import json, os, sys
L, LI, T = sys.argv[1:4]; r = {}
for m in ("depth", "radial"):
    for bg in ("black", "grey"):
        for tag, d in (("in", LI), ("op", os.path.join(L, "report"))):
            f = os.path.join(d, f"{tag}_{m}_{bg}.json")
            if os.path.exists(f):
                j = json.load(open(f)); r[f"{tag}_{m}_{bg}"] = {k: j.get(k) for k in ("psnr", "ssim", "lpips", "dists")}
    for gg in (f"gate_{m}", f"gate_{m}_perc"):
        g = os.path.join(L, "report", f"{gg}.json")
        if os.path.exists(g):
            r[gg] = json.load(open(g)).get("accept")
p = os.path.join(L, "proxies.json")
if os.path.exists(p):
    j = json.load(open(p)); r["orbit"] = {k: {"spike": v["spike_frac_object_zone"], "needle": v["needle_frac_object_zone"], **v["summary"]} for k, v in j.items()}
for f in ("budget_budget.json", "refit_refit.json"):
    if os.path.exists(os.path.join(T, f)):
        j = json.load(open(os.path.join(T, f))); r[f] = {k: v for k, v in j.items() if k not in ("iterations", "curve", "args")}
r["build"] = {k: v for k, v in json.load(open(os.path.join(T, "priors.json"))).items() if k != "objects"}
json.dump(r, open(os.path.join(L, "summary.json"), "w"), indent=1); print("OBJECT_PRIORS_SUMMARY", json.dumps(r))
PY
echo OBJECT_PRIORS_DONE
