#!/bin/bash
# Cosmetic object backs from a multi-view diffusion prior (scripts/object_priors/mv_gen.py + mv_refit.py), evaluated like
# object_priors.sbatch (reuses its stage-0 products in $P/object_priors/<NAME>: objects, input depth, support, orbit):
#   1. generate (venv-mv): Zero123++ v1.2 novel views of each object from its best real crop (BiRefNet masks, Marigold normals)
#   2. register: generated cameras in the splat frame (silhouette IoU vs the real-mask hull), hull seeds on the unseen surface
#   3. refit (venv-gsplat): object-zone gaussians + seeds trained on the generated views (free rows only) + real frames
#   4. shfit + refaxes -> <E>/<NAME>_op_cosmetic_mv[_TAG].ply
#   5. live SuperSplat: on-path held-out both sort modes vs the input, orbit montages + proxies, off-path moved + editor views
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu --exclude=gpu010,gpu035 -o logs/%x-%j.out sbatch/mv_prior.sbatch \
#     configs/mocap2_well-lit_trot.yaml
# Env: PRIOR=zero123|seva (seva: scripts/object_priors/mv_seva.py, venv-seva, GEN_ARGS e.g. "--ids 3 4 5"), COMPARE=<variant dir>,
#      REFIT_MODE=objfit (mv_objfit.py: standalone object splats swapped in; REFIT_ARGS e.g. "--ids 3 4"),
#      NAME (viewer_msd60k_wd_clean_ft2_solid3), IN, REF_NAME (stage-0 objects/orbit source for a new NAME), TAG (variant suffix), GEN (generated-view dir, shared by TAGs: GEN_TAG),
#      GEN_ARGS, REG_ARGS, REFIT_ARGS, BUDGET=1 (+BUDGET_ARGS: solidify_unseen.py budget after the refit), STOP_AFTER=prep|register|refit, FORCE=1 (redo refit + later stages), MOVES, EVERY_OFF.
#SBATCH -J mv-prior
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010,gpu035
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
MPY="${MV_VENV:-$CEAR_WS/venv-mv}/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
NAME="${NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; IN="${IN:-$E/$NAME.ply}"; TAG="${TAG:-}"
D="$P/object_priors/$NAME"; GB=mv_gen; [ "${PRIOR:-zero123}" = seva ] && GB=mv_seva; G="${GEN:-$D/$GB${GEN_TAG:+_$GEN_TAG}}"; T="$D/cosmetic_mv${TAG:+_$TAG}"
OUTPLY="$E/${NAME}_op_cosmetic_mv${TAG:+_$TAG}.ply"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; RT="$P/viewer_bench/reference_dataset"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
SO=scripts/solidify_unseen.py; MG=scripts/object_priors/mv_gen.py; VS=(scripts/viewer_shots.py --engine current)
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"
need() { [ ! -e "$1" ]; }; needt() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$G" "$T"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
# stage 0 for a new input: objects + orbit cameras are geometry (taken from REF_NAME's dir so orbit views stay comparable),
# the input's depth at the training frames + per-gaussian support are computed here
RD="$P/object_priors/${REF_NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; mkdir -p "$D"
for f in objects.json objects.npz objects.png; do [ -e "$D/$f" ] || [ ! -e "$RD/$f" ] || cp "$RD/$f" "$D/$f"; done
[ -e "$D/orbit" ] || [ ! -d "$RD/orbit" ] || cp -r "$RD/orbit" "$D/orbit"
[ -f "$IN" ] && [ -f "$D/objects.json" ] || { echo "missing input / objects (run object_priors.sbatch stage 0)"; exit 1; }
need "$D/depth" && "$SPY" scripts/render_ply_depth.py --ply "$IN" --dataset "$DATA" --out-dir "$D/depth"
need "$D/support.npz" && "$SPY" "$SO" support --ply "$IN" --dataset "$DATA" "$TEST" --out "$D/support.npz"
[ "${STOP_AFTER:-}" = prep ] && { echo MV_PRIOR_PREP_ONLY; exit 0; }
[ -x "$MPY" ] || { echo "venv-mv missing: sbatch/mv_venv.sbatch"; exit 1; }
echo "== $NAME TAG=${TAG:-<none>} GEN=$G"
COMMON=(--objects "$D/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" --splat-depth-dir "$D/depth" --out-dir "$G")
if [ "${PRIOR:-zero123}" = seva ]; then   # Stable Virtual Camera: real crops + real cameras in, orbit views out (mv_seva.py)
  SVPY="${SEVA_VENV:-$CEAR_WS/venv-seva}/bin/python"; MS=scripts/object_priors/mv_seva.py
  echo "== 1. generate (SEVA)"
  need "$G/generate.json" && "$SVPY" "$MS" generate "${COMMON[@]}" ${GEN_ARGS:-}
  echo "== 2. hull + seeds"
  { need "$G/register.json" || [ "${FORCE_REG:-0}" = 1 ]; } && "$MPY" "$MS" hull "${COMMON[@]}" ${REG_ARGS:-}
else
  echo "== 1. generate"
  need "$G/generate.json" && "$MPY" "$MG" generate "${COMMON[@]}" ${GEN_ARGS:-}
  echo "== 2. register"
  { need "$G/register.json" || [ "${FORCE_REG:-0}" = 1 ]; } && "$MPY" "$MG" register "${COMMON[@]}" --ply "$IN" --support "$D/support.npz" ${REG_ARGS:-}
fi
[ "${STOP_AFTER:-}" = register ] && { echo MV_PRIOR_REGISTER_ONLY; exit 0; }
if [ "${REFIT_MODE:-zone}" = objfit ]; then   # standalone object splats from the generated views, swapped into the zones
  echo "== 3. standalone object fit + swap + colour harmonisation (mv_objfit.py)"
  needt "$T/refit.ply" && "$SPY" scripts/object_priors/mv_objfit.py --ply "$IN" --support "$D/support.npz" --gen "$G" \
    --dataset "$DATA" "$TEST" --sim3-json "$P/sim3.json" --objects "$D/objects" --splat-depth-dir "$D/depth" --out "$T/refit.ply" ${REFIT_ARGS:-}
else
  echo "== 3. refit on the generated views"
  needt "$T/refit.ply" && "$SPY" scripts/object_priors/mv_refit.py --ply "$IN" --support "$D/support.npz" --gen "$G" \
    --dataset "$DATA" "$TEST" --sim3-json "$P/sim3.json" --objects "$D/objects" --out "$T/refit.ply" ${REFIT_ARGS:-}
fi
[ "${STOP_AFTER:-}" = refit ] && { echo MV_PRIOR_REFIT_ONLY; exit 0; }
CAND="$T/refit.ply"
if [ "${BUDGET:-0}" = 1 ]; then   # per-view visibility budget vs the input: reverts what shows in a training view (> 8/255 cluster / > 0.3 dB)
  echo "== 3b. visibility budget"
  needt "$T/budget.ply" && "$SPY" "$SO" budget --input "$IN" --cand "$T/refit.ply" --src "$T/refit_src.npz" --dataset "$DATA" "$TEST" \
    --out "$T/budget.ply" ${BUDGET_ARGS:---min-w 0.002 --iters 12}
  CAND="$T/budget.ply"
fi
echo "== 4. shfit + exports"
needt "$OUTPLY" && "$SPY" "$SO" shfit --ply "$CAND" --ref "$CAND" --dataset "$DATA" "$TEST" --out "$OUTPLY"
[ -z "$FRAME" ] || { needt "${OUTPLY%.ply}_refaxes.ply" && "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
  --gt-model "$FRAME" --in-ply "$OUTPLY" --out-ply "${OUTPLY%.ply}_refaxes.ply"; }
ls -la "$OUTPLY"*; md5sum "$OUTPLY" | tee "$T/output.md5"

echo "== 5. live SuperSplat checks"
bash env/make_viewer_venv.sh | tail -1
LI="$D/live_in"; L="$T/live"
[ "${FORCE:-0}" = 1 ] && [ -d "$L" ] && mv "$L" "${L}_prev_$(date +%m%d%H%M%S)"
mkdir -p "$LI" "$L" "$L/report"
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
done
"$CPUPY" "$SO" crops --before "$LI/onpath_depth/black" --after "$L/onpath_depth/black" --gt "$TEST/images" --out "$L/onpath_crops.jpg" || true
[ -f "$LI/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$IN" --dataset "$D/orbit" --out-dir "$LI/orbit" | tail -1
[ -f "$L/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$OUTPLY" --dataset "$D/orbit" --out-dir "$L/orbit" | tail -1
"$CPUPY" "$SO" montage --dirs "before=$LI/orbit" "after=$L/orbit" --out-dir "$L/orbit_montage"
CMP="${COMPARE:-}"; [ -n "$CMP" ] && [ -d "$D/$CMP/live/orbit" ] && "$CPUPY" "$SO" montage --dirs "$CMP=$D/$CMP/live/orbit" "this=$L/orbit" --out-dir "$L/orbit_montage_vs_$CMP" || true
V6="$D/cosmetic_v6/live/orbit"; [ -d "$V6" ] && "$CPUPY" "$SO" montage --dirs "v6=$V6" "mv=$L/orbit" --out-dir "$L/orbit_montage_vs_v6" || true
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
    g = os.path.join(L, "report", f"gate_{m}.json")
    if os.path.exists(g):
        r[f"gate_{m}"] = json.load(open(g)).get("accept")
p = os.path.join(L, "proxies.json")
if os.path.exists(p):
    j = json.load(open(p)); r["orbit"] = {k: {"spike": v["spike_frac_object_zone"], "needle": v["needle_frac_object_zone"], **v["summary"]} for k, v in j.items()}
f = os.path.join(T, "refit_mvrefit.json")
if os.path.exists(f):
    r["refit"] = {k: v for k, v in json.load(open(f)).items() if k not in ("curve", "args")}
json.dump(r, open(os.path.join(L, "summary.json"), "w"), indent=1); print("MV_PRIOR_SUMMARY", json.dumps(r))
PY
echo MV_PRIOR_DONE
