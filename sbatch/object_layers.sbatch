#!/bin/bash
# Layered, object-centric scene (scripts/object_layers/object_layers.py + vlm.py; config block object_layers):
#   prep       objects (find_objects.py), orbit cameras, base-splat depth at the training frames      (as mv_prior.sbatch stage 0)
#   detect     image detection (Grounding-DINO + SAM 2.1 on the training frames, detect_objects.py) lifted with the splat
#              depth, multi-view voted, merged with the LiDAR clusters -> <D>/objects_det.{json,npz,png}; used by every later
#              stage when object_layers.detect.enable                                                  [GPU, venv-seg]
#   assets     BiRefNet crops + image-to-3D assets of every object (mv_asset_gen.py, TRELLIS)          [GPU]
#   evidence   per object: cleaned LiDAR cluster, views, measured free space / seen surface, VLM crops [CPU]
#   recipe     vision-language shape recipe + detection check (vlm.py; needs internet + key file)      [HTTP]
#   candidates primitive fit and generated mesh, both carved + textured from real photos, sheets       [CPU]
#   select     vision-language choice between the candidates                                           [HTTP]
#   finalize   final meshes (+ review rounds: vlm.py review -> edits -> finalize), GLB / USD            [CPU, HTTP]
#   floor      floor texture completed under / behind the objects, flat floor gaussians               [GPU]
#   compose    background splat + floor + object meshes as surface gaussians -> <E>/<NAME>_layers[_TAG]{,_refaxes}.ply
#   eval       live SuperSplat: on-path both sort modes, orbit montages before / after, off-path moves
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/object_layers.sbatch configs/<seq>.yaml
# Env: VLM_ENABLE=0 (no API calls at all, cache unused), NAME (input <E>/<NAME>.ply), IN, TAG, REF_NAME (objects / orbit source), ASSET_GEN (asset dir to reuse), STAGES (subset, in
#      order), IDS (debug: "--ids 3 4"), FORCE=1 (redo from candidates), MOVES, EVERY_OFF. No per-object arguments are needed.
# The HTTP stages are skipped (cache still used) when the node has no route to the API: run them on a node with internet
#   STAGES="recipe" bash sbatch/object_layers.sbatch <cfg>      (tiny client, no compute)   and resubmit; every stage is idempotent.
#SBATCH -J obj-layers
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
MPY="${MV_VENV:-$CEAR_WS/venv-mv}/bin/python"; TPY="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
SEGPY="${SEG_VENV:-$CEAR_WS/venv-seg}/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; LPY="${VLM_VENV:-$CEAR_WS/venv-vlm}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export HF_HOME="${HF_HOME:-$CEAR_WS/hf-cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
NAME="${NAME:-viewer_ibr70k_wd_clean_ft2_solid3_floor}"; IN="${IN:-$E/$NAME.ply}"; TAG="${TAG:-}"
D="$P/object_priors/$NAME"; T="$D/layers${TAG:+_$TAG}"; AG="${ASSET_GEN:-$D/asset_gen}"; IV="$P/ibr_views"
OUTPLY="${OUTPLY_OVERRIDE:-$E/${NAME}_layers${TAG:+_$TAG}.ply}"   # OUTPLY_OVERRIDE: evaluate any finished PLY against NAME (STAGES=eval)
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; RT="$P/viewer_bench/reference_dataset"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
SO=scripts/solidify_unseen.py; OL=scripts/object_layers/object_layers.py; VL=scripts/object_layers/vlm.py; VS=(scripts/viewer_shots.py --engine current)
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"
STAGES="${STAGES:-prep detect assets evidence recipe candidates select finalize floor compose eval}"
has() { [[ " $STAGES " == *" $1 "* ]]; }; need() { [ ! -e "$1" ]; }; needt() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no GPU"); mkdir -p "$T" "$D" logs
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true; module list 2>&1 | tail -2
command -v nvcc >/dev/null && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
[ "$(get object_layers.enable 1)" = 1 ] || { echo "object_layers.enable = false: skipped"; exit 0; }
[ -f "$IN" ] || { echo "missing input $IN"; exit 1; }
"$CPUPY" - "$CFG" "$T/config.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts"); import pose_utils as pu
json.dump(pu.load_config(sys.argv[1]).get("object_layers", {}) or {}, open(sys.argv[2], "w"), indent=1)
PY
OBJ="$D/objects"; DET="${DET:-$D/objects_det}"   # DET: another detection run (prefix), e.g. a re-lift with new settings
if [ "$(get object_layers.detect.enable 0)" = 1 ]; then OBJ="$DET"; fi
SEGA=(); [ "$OBJ" = "$DET" ] && SEGA=(--seg-dir "$D/detect/seg")
COMMON=(--dir "$T" --cfg-json "$T/config.json" --objects "$OBJ" "${SEGA[@]}" --sim3-json "$P/sim3.json" --dataset "$DATA" --splat-depth-dir "$D/depth" --gen-dir "$AG")
VLM=(--dir "$T" --model "$(get object_layers.vlm.model claude-opus-5-5)" --effort "$(get object_layers.vlm.effort high)" --cache-dir "$CEAR_WS/vlm-cache")
vlm_on() { [ "${VLM_ENABLE:-1}" = 1 ] && [ "$(get object_layers.vlm.enable 1)" = 1 ] && [ "$(get "object_layers.vlm.$1" 1)" != 0 ] && [ -x "$LPY" ]; }
echo "== $NAME TAG=${TAG:-<none>} stages: $STAGES"; echo "python: $TPY ($("$TPY" --version 2>&1))"

if has prep; then   # objects + orbit cameras are geometry (taken from REF_NAME's dir when given), depth of the input at the training frames
  RD="$P/object_priors/${REF_NAME:-$NAME}"
  for f in objects.json objects.npz objects.png; do [ -e "$D/$f" ] || [ ! -e "$RD/$f" ] || cp "$RD/$f" "$D/$f"; done
  [ -e "$D/orbit" ] || [ ! -d "$RD/orbit" ] || cp -r "$RD/orbit" "$D/orbit"
  GTM="$OUT/colmap_train/sparse/0"
  need "$D/objects.json" && [ -f "$CLOUD" ] && [ -d "$GTM" ] && "$CPUPY" scripts/find_objects.py --cloud "$CLOUD" --gt-model "$GTM" --out "$D/objects"
  [ -f "$D/objects.json" ] || { echo "no free-standing objects detected / no LiDAR cloud: nothing to do"; exit 0; }
  need "$D/orbit/sparse/0/images.txt" && "$CPUPY" "$SO" orbit --objects "$D/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" --cloud "$CLOUD" --out "$D/orbit"
  need "$D/depth" && "$SPY" scripts/render_ply_depth.py --ply "$IN" --dataset "$DATA" --out-dir "$D/depth"
fi
if has detect && [ "$OBJ" = "$DET" ]; then
  [ -x "$SEGPY" ] || bash env/make_seg_venv.sh
  need "$D/detect/segment.json" && "$SEGPY" scripts/object_layers/detect_objects.py segment --dataset "$DATA" --out-dir "$D/detect" --cfg-json "$T/config.json"
  needt "$DET.json" && "$CPUPY" scripts/object_layers/detect_objects.py lift --dataset "$DATA" --out-dir "$D/detect" --cfg-json "$T/config.json" \
    --splat-depth-dir "$D/depth" --sim3-json "$P/sim3.json" --lidar-objects "$D/objects" --cloud "$CLOUD" --out "$DET" | grep -E "DETECT|Error|Traceback" || true
fi
if [ ! -f "$OBJ.json" ]; then   # e.g. STAGES=eval of another PLY: the LiDAR objects are enough
  if has assets || has evidence; then echo "missing $OBJ.json (run the detect stage)"; exit 1; fi
  OBJ="$D/objects"; SEGA=(); COMMON=(--dir "$T" --cfg-json "$T/config.json" --objects "$OBJ" --sim3-json "$P/sim3.json" --dataset "$DATA" --splat-depth-dir "$D/depth" --gen-dir "$AG")
fi
GEN_ARGS="${GEN_ARGS:-$(get object_layers.assets.crop_args "")}"
if has assets; then
  [ -x "$MPY" ] || { echo "venv-mv missing: sbatch/mv_venv.sbatch"; exit 1; }
  need "$AG/crops.json" && "$MPY" scripts/object_priors/mv_asset_gen.py crops --objects "$OBJ" --sim3-json "$P/sim3.json" --dataset "$DATA" \
    --splat-depth-dir "$D/depth" --out-dir "$AG" ${GEN_ARGS:-}
  need "$AG/generate.json" && TORCH_HOME="$CEAR_WS/torch-hub" "$TPY" scripts/object_priors/mv_asset_gen.py generate --out-dir "$AG" ${TRELLIS_ARGS:-}
fi
export HY3DGEN_MODELS="$CEAR_WS/hy3dgen-cache"   # Hunyuan3D copies its weights here (default would be ~/.cache: never $HOME)
HYD="${HY_GEN:-$D/asset_gen_hunyuan}"; HPY="${HUNYUAN_VENV:-$CEAR_WS/venv-hunyuan}/bin/python"
if has assets && [ "$(get object_layers.hunyuan 1)" = 1 ] && [ -x "$HPY" ] && [ -d "$CEAR_WS/Hunyuan3D-2.1" ]; then   # second generator
  need "$HYD/generate_hy.json" && HF_HUB_OFFLINE=0 "$HPY" scripts/object_priors/mv_asset_gen.py generate_hy --crops-dir "$AG" --out-dir "$HYD" || echo "hunyuan generation failed: TRELLIS only"
fi
[ -d "$HYD" ] && [ "${HUNYUAN:-1}" = 1 ] && COMMON+=(--extra-gen-dir "$HYD")
if has evidence; then
  need "$T/evidence.json" && "$TPY" "$OL" evidence "${COMMON[@]}" ${IDS:-}
fi
if has recipe && vlm_on recipe; then
  "$LPY" "$VL" recipe "${VLM[@]}" || true
  # detection check: clusters the model sees as several objects / a partial object are split / merged, then described again
  "$TPY" "$OL" evidence "${COMMON[@]}" --apply-detection ${IDS:-} && "$LPY" "$VL" recipe "${VLM[@]}" || true
fi
if has candidates; then
  needt "$T/candidates.json" && "$TPY" "$OL" candidates "${COMMON[@]}" ${IDS:-}
fi
if has select && vlm_on select; then "$LPY" "$VL" select "${VLM[@]}" || true; fi
if has finalize; then
  "$TPY" "$OL" finalize "${COMMON[@]}" --round 1
  NR="$(get object_layers.vlm.review_rounds 2)"
  if vlm_on recipe && [ "$NR" -gt 0 ]; then
    for k in $(seq 1 "$NR"); do
      "$LPY" "$VL" review "${VLM[@]}" --round "$k" || true
      [ -f "$T/vlm/review_round$k.json" ] || break
      "$TPY" "$OL" finalize "${COMMON[@]}" --round "$((k + 1))"
      "$CPUPY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('edits') else 1)" "$T/vlm/review_round$k.json" || break
    done
  fi
  if [ -x "$LPY" ] && [ -f "$T/vlm/usage.jsonl" ]; then "$LPY" "$VL" cost "${VLM[@]}" && "$TPY" "$OL" finalize "${COMMON[@]}" --round 9 >/dev/null; fi
fi
BG="$T/bg_floor.ply"; [ "$(get object_layers.floor_harm_check 0)" = 1 ] || BG="$T/floor_filled.ply"
GATE=(--dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --ply "$IN" --dataset "$DATA" --splat-depth-dir "$D/depth" --train-stats "$T/train_w_in.npz")
# per-gaussian training-view weights of the input / the floor-completed background: compose never removes what the cameras draw
twstats() { [ -f "$2" ] || "$SPY" scripts/clean_haze.py stats --ply "$1" --dataset "$DATA" --stats "$2" | tail -1; }
if has floor; then
  # per-object viewer-exact accept test first (object_layers.max_object_loss_db): the floor is completed only around accepted objects
  twstats "$IN" "$T/train_w_in.npz"; needt "$T/gate.json" && "$SPY" "$OL" gate "${GATE[@]}"
  [ -f "$IV/scene.json" ] || "$CPUPY" scripts/make_lidar_ibr_views.py floor --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" --out "$IV/scene.json"
  # shellcheck disable=SC2086
  [ -f "$IV/floor_map.npz" ] || "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
    --scene-json "$IV/scene.json" --floor-map "$IV/floor_map.npz" --out "$IV/_floor_map_tmp" ${MAP_ARGS:-} 2>&1 | grep -E "^\[ibr\]|IBR_FLOOR_MAP|Error|Traceback" | cut -c1-600
  needt "$T/floor_map.npz" && "$TPY" "$OL" floormap --dir "$T" --cfg-json "$T/config.json" --dataset "$DATA" --sim3-json "$P/sim3.json" --map "$IV/floor_map.npz" --out "$T/floor_map.npz"
  needt "$T/floor_map_inpainted.npz" && "$IPY" scripts/floor_fill_ibr.py inpaint --map "$T/floor_map.npz" --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" | tail -2
  needt "$T/floor_filled.ply" && "$SPY" scripts/floor_fill_ibr.py apply --ply "$IN" --map "$T/floor_map_inpainted.npz" --dataset "$DATA" \
    --out-base "$T/floor_base.ply" --out-filled "$T/floor_filled.ply" ${APPLY_ARGS:-} | grep -E "floorfill|FLOOR_FILL" | cut -c1-900
  if [ "$BG" != "$T/floor_filled.ply" ] && needt "$BG"; then
    "$SPY" scripts/fill_harm_ply.py --base "$T/floor_base.ply" --filled "$T/floor_filled.ply" --dataset "$DATA" --out "$BG" | tail -3
  fi
fi
if has compose; then
  MAPA=(); [ -f "$BG" ] || { BG="$IN"; MAPA=(--map "$T/floor_map_inpainted.npz"); }   # no floor stage: discs under the objects only
  [ -f "$T/gate.json" ] || { twstats "$IN" "$T/train_w_in.npz"; "$SPY" "$OL" gate "${GATE[@]}"; }
  twstats "$BG" "$T/train_w_bg.npz"
  if [ "$(get object_layers.floaters.enable 0)" = 1 ]; then
    # compose, then floater removal on the background rows only (the inserted mesh / floor gaussians are protected);
    # object_layers.floaters.args = scripts/clean_haze.py clean flags (measured harm restore unless --no-restore)
    RAW="${OUTPLY%.ply}_nofloat.ply"
    needt "$RAW" && "$TPY" "$OL" compose --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --ply "$BG" "${MAPA[@]}" --train-stats "$T/train_w_bg.npz" --out "$RAW"
    PF=$("$CPUPY" -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['rows_out'] - j['inserted'])" "${RAW%.ply}_layers.json")
    # shellcheck disable=SC2046
    needt "$OUTPLY" && "$SPY" scripts/clean_haze.py clean --ply "$RAW" --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
      --freespace-grid "$P/freespace/grid/onpath.npz" --protect-from "$PF" --stats "$T/floaters_stats.npz" --out "$OUTPLY" \
      $(get object_layers.floaters.args "") | grep -E "\[haze\] (cand|iter|train|removed|per)"
  else
    needt "$OUTPLY" && "$TPY" "$OL" compose --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --ply "$BG" "${MAPA[@]}" --train-stats "$T/train_w_bg.npz" --out "$OUTPLY"
  fi
  [ -z "$FRAME" ] || { needt "${OUTPLY%.ply}_refaxes.ply" && "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
    --gt-model "$FRAME" --in-ply "$OUTPLY" --out-ply "${OUTPLY%.ply}_refaxes.ply" | tail -1; }
  ls -la "$OUTPLY"*; md5sum "$OUTPLY" | tee "$T/output.md5"
  # floor-only views behind / above every meshed object, input vs layered background (objects removed in both)
  "$SPY" "$OL" floorviews --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --dataset "$DATA" --base-ply "$IN" --ply "$BG" || true
fi
has eval || { date; echo OBJ_LAYERS_DONE; exit 0; }

echo "== live SuperSplat checks"
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
done
"$CPUPY" "$SO" crops --before "$LI/onpath_depth/black" --after "$L/onpath_depth/black" --gt "$TEST/images" --out "$L/onpath_crops.jpg" || true
[ -f "$LI/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$IN" --dataset "$D/orbit" --out-dir "$LI/orbit" | tail -1
[ -f "$L/orbit/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --ply "$OUTPLY" --dataset "$D/orbit" --out-dir "$L/orbit" | tail -1
"$CPUPY" "$SO" montage --dirs "before=$LI/orbit" "after=$L/orbit" --out-dir "$L/orbit_montage"
CMP="${COMPARE:-}"; [ -n "$CMP" ] && [ -d "$D/$CMP/live/orbit" ] && "$CPUPY" "$SO" montage --dirs "$CMP=$D/$CMP/live/orbit" "layers=$L/orbit" --out-dir "$L/orbit_montage_vs_$CMP" || true
"$SPY" "$SO" eval --ply "before=$IN" "after=$OUTPLY" --orbit "$D/orbit" --objects "$D/objects" --sim3-json "$P/sim3.json" \
  --cloud "$CLOUD" --live "before=$LI/orbit" "after=$L/orbit" --json-out "$L/proxies.json" | tail -1 || true
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
p = os.path.join(L, "proxies.json")
if os.path.exists(p):
    j = json.load(open(p)); r["orbit"] = {k: {"spike": v["spike_frac_object_zone"], "needle": v["needle_frac_object_zone"], **v["summary"]} for k, v in j.items()}
f = os.path.join(T, "objects_layers.json")
if os.path.exists(f):
    j = json.load(open(f)); r["objects"] = {k: v.get("decision") for k, v in j["objects"].items()}; r["vlm_cost"] = j.get("vlm_cost")
json.dump(r, open(os.path.join(L, "summary.json"), "w"), indent=1); print("OBJ_LAYERS_SUMMARY", json.dumps(r))
PY
date; echo OBJ_LAYERS_DONE
