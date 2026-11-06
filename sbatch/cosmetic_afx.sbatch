#!/bin/bash
# COSMETIC unseen sides by image repair (scripts/cosmetic_afx.py): orbit paths around every free-standing object that
# start at a real training view, the delivered splat rendered along them (viewer-exact), the renders repaired by the
# ArtiFixer world model (context = the covisible real training frames; opacity = measured visibility, so it keeps what
# a camera saw and regenerates the rest), then ONLY the objects' gaussians (+ seeded ones) distilled on the repaired
# frames mixed with the real training frames. Invented content is allowed on unseen sides; seen views are reported.
#   for s in prep afx distill score; do J=$(STAGE=$s WHY="cosmetic afx $s" scripts/sub.sh -A pi_donghyunkim_umass_edu \
#     --exclude=gpu010,gpu035 ${J:+-d afterok:$J} -o logs/%x-%j.out sbatch/cosmetic_afx.sbatch configs/<seq>.yaml); done
#   STAGE=all runs everything in one job.
# Stages (outputs in $P/cosmetic_afx/$RUN):
#   prep     paths + renders + visibility + object masks + ArtiFixer prepared scenes (gsplat venv)
#   afx      ArtiFixer inference along every path (needs an 80 GB A100), collect -> repaired/, strips/
#   distill  zone-only distillation -> <E>/<OUTNAME>.ply (+ _refaxes)
#   score    live SuperSplat on the on-path test set (Sort Mode depth AND radial, black + grey, +DISTS) gated vs the
#            input's reports; orbit cameras of solidify_unseen.py (before | after montages, alpha-hole / spike proxies);
#            the training paths (before | after | repaired); off-path moved views + the editor default view
# Env: STAGE, NAME (viewer_msd60k_wd_clean_ft2_solid3: input <E>/<NAME>.ply), RUN (v1), OUTNAME (<NAME>_cosmetic_afx),
#      SOLID (solidify work dir with objects + orbit: $P/solidify/viewer_msd60k_wd_clean_ft2), PATH_ARGS, PREP_ARGS,
#      AFX_ARGS (extra run_inference flags), DISTILL_ARGS, REF / REF_RAD (input's live report prefixes, default
#      $P/solid_chain/viewer_msd60k_wd/report/solid3 / solid3_rad), MOVES ("0 -0.3 1.0"), EVERY_OFF (40).
#SBATCH -J cosmetic-afx
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"; AFX_DIR="${AFX_DIR:-$CEAR_WS/ArtiFixer}"; APY="$AFX_VENV/bin/python"
export CHECKPOINT_PT="${CHECKPOINT_PT:-$CEAR_WS/artifixer-checkpoints/artifixer-1.3b.pt}"
export MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"; SEQ="$(get sequence.name)"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; RT="$P/viewer_bench/reference_dataset"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
NAME="${NAME:-viewer_msd60k_wd_clean_ft2_solid3}"; IN="$E/$NAME.ply"; RUN="${RUN:-v1}"; OUTNAME="${OUTNAME:-${NAME}_cosmetic_afx}"
OUTPLY="$E/$OUTNAME.ply"; S="${SOLID:-$P/solidify/viewer_msd60k_wd_clean_ft2}"; W="$P/cosmetic_afx/$RUN"; mkdir -p "$W"
CAP="${CAPTION:-$P/offpath/artifixer/prep/$SEQ/captions/$SEQ/caption.h5}"
CA=scripts/cosmetic_afx.py; SO=scripts/solidify_unseen.py
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; echo "== cosmetic_afx $RUN: STAGE=${STAGE:?}"
gs_env() { module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

st_prep() {
  gs_env; [ -f "$IN" ] || { echo "missing $IN"; exit 1; }
  if [ ! -f "$CAP" ]; then                                    # caption of the scene (ArtiFixer's own captioning phase)
    module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
    SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"
    ( cd "$AFX_DIR" && "$APY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$P/dataset" \
        --output_root "$W/capprep/$SEQ" --phases prepare,caption --metric_scale "$SCALE" )
    CAP="$W/capprep/$SEQ/captions/$SEQ/caption.h5"; gs_env
  fi
  "$CPUPY" "$CA" paths --objects "$S/objects" --sim3-json "$P/sim3.json" --dataset "$DATA" --cloud "$CLOUD" --out "$W" ${PATH_ARGS:-}
  "$SPY" "$CA" prep --ply "$IN" --dataset "$DATA" --paths "$W" --objects "$S/objects" --sim3-json "$P/sim3.json" \
    --caption "$CAP" --image-root "$DATA" --seq "$SEQ" ${PREP_ARGS:-}
}
st_afx() {
  module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
  ( cd "$AFX_DIR" && "$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
      --model_id "$MODEL_ID" --save_dir "$W/afx_run" --split_path "$W/afx/split.json" --render_trajectory trajectory \
      --neighbor_selection_mode covisibility --save_frame_outputs_only ${AFX_ARGS:-} )
  "$CPUPY" "$CA" collect --paths "$W" --afx-run "$W/afx_run" --seq "$SEQ"
}
st_distill() {
  gs_env
  "$SPY" "$CA" distill --ply "$IN" --dataset "$DATA" --paths "$W" --objects "$S/objects" --sim3-json "$P/sim3.json" \
    --out "$OUTPLY" ${DISTILL_ARGS:-} 2>&1 | tee "$W/distill_$OUTNAME.log" | grep -E "distill\]|DISTILL|rror" | grep -v " step " || true
  grep -q DISTILL_DONE "$W/distill_$OUTNAME.log" || { tail -30 "$W/distill_$OUTNAME.log"; exit 1; }
  [ -z "$FRAME" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
    --in-ply "$OUTPLY" --out-ply "${OUTPLY%.ply}_refaxes.ply"
  ls -la "$OUTPLY" "${OUTPLY%.ply}_refaxes.ply" 2>/dev/null || true
}
st_score() {
  gs_env; L="$W/score_$OUTNAME"; REP="$L/report"; mkdir -p "$REP"
  REF="${REF:-$P/solid_chain/viewer_msd60k_wd/report/solid3}"; REF_RAD="${REF_RAD:-$P/solid_chain/viewer_msd60k_wd/report/solid3_rad}"
  VS=(scripts/viewer_shots.py --engine current)
  for m in depth radial; do
    [ -f "$L/shots_$m/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black grey --sort-mode $m --ply "$OUTPLY" --dataset "$TEST" \
      --out-dir "$L/shots_$m" | tail -1
    for bg in black grey; do
      "$GPY" scripts/eval_views.py --pred-dir "$L/shots_$m/$bg" --dataset "$TEST" --tag "${m}_$bg" --json-out "$REP/${m}_$bg.json" | tail -1
      PYTHONPATH="$PIQ_LIB" "$GPY" - "$L/shots_$m/$bg" "$TEST/images" "$REP/${m}_$bg.json" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
m = piq.DISTS(reduction="none").cuda(); v = []
for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
    with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2); print("SCORE", json.dumps(r))
PY
    done
  done
  "$CPUPY" scripts/accept_gate.py --pair black "${REF}_black.json" "$REP/depth_black.json" --pair grey "${REF}_grey.json" \
    "$REP/depth_grey.json" --out "$REP/gate_depth.json" | tail -1 || true
  "$CPUPY" scripts/accept_gate.py --pair black "${REF_RAD}_black.json" "$REP/radial_black.json" --pair grey "${REF_RAD}_grey.json" \
    "$REP/radial_grey.json" --out "$REP/gate_radial.json" | tail -1 || true
  echo "== orbit cameras (solidify_unseen.py orbit) before | after, Sort Mode depth"
  for v in before after; do
    ply="$IN"; [ "$v" = after ] && ply="$OUTPLY"
    [ -f "$L/orbit/$v/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --sort-mode depth --ply "$ply" --dataset "$S/orbit" --out-dir "$L/orbit/$v" | tail -1
  done
  "$CPUPY" "$SO" montage --dirs "before=$L/orbit/before" "after=$L/orbit/after" --out-dir "$L/orbit_montage"
  "$SPY" "$SO" eval --ply "before=$IN" "after=$OUTPLY" --orbit "$S/orbit" --objects "$S/objects" --sim3-json "$P/sim3.json" \
    --cloud "$CLOUD" --live "before=$L/orbit/before" "after=$L/orbit/after" --json-out "$REP/orbit_proxies.json" | grep SOLID_EVAL || true
  echo "== training paths (every 4th ring frame) before | after | repaired"
  "$CPUPY" - "$W" "$L/pathsub" <<'PY'
import json, os, sys
W, out = sys.argv[1:3]; meta = json.load(open(os.path.join(W, "paths.json")))
keep = {n for sc in meta["scenes"] for n in sc["frames"][sc["n_approach"]::4]}
os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
src = os.path.join(W, "sparse", "0"); lines = open(os.path.join(src, "images.txt")).read().split("\n")
rows = [ln + "\n\n" for ln in lines if ln.strip() and ln.split()[-1] in keep]
open(os.path.join(out, "sparse", "0", "images.txt"), "w").writelines(rows)
for f in ("cameras.txt", "points3D.txt"):
    open(os.path.join(out, "sparse", "0", f), "w").write(open(os.path.join(src, f)).read())
print(len(rows), "path frames")
PY
  for v in before after; do
    ply="$IN"; [ "$v" = after ] && ply="$OUTPLY"
    [ -f "$L/paths/$v/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --sort-mode depth --ply "$ply" --dataset "$L/pathsub" --out-dir "$L/paths/$v" | tail -1
  done
  "$CPUPY" "$CA" pathmontage --paths "$W" --dirs "before=$L/paths/before" "after=$L/paths/after" "repaired=$W/repaired" \
    --out-dir "$L/path_montage" --n 8 || true
  INR="$E/${NAME}_refaxes.ply"; OUTR="${OUTPLY%.ply}_refaxes.ply"
  if [ -f "$INR" ] && [ -f "$OUTR" ] && [ -f "$RT/sparse/0/images.txt" ]; then
    echo "== off-path moved views + editor default view"
    for v in before after; do
      ply="$INR"; [ "$v" = after ] && ply="$OUTR"
      [ -f "$L/offpath/$v/_viewer.json" ] || "$VPY" "${VS[@]}" --bg black --sort-mode depth --ply "$ply" --dataset "$RT" \
        --every "${EVERY_OFF:-40}" --move ${MOVES:-0 -0.3 1.0} --editor --out-dir "$L/offpath/$v" | tail -1
    done
    "$CPUPY" "$SO" montage --dirs "before=$L/offpath/before" "after=$L/offpath/after" --out-dir "$L/offpath_montage"
    "$CPUPY" -c "import cv2,numpy as np,sys;a,b=[cv2.imread('$L/offpath/'+v+'/_supersplat_editor.png') for v in ('before','after')];cv2.imwrite('$L/editor_before_after.jpg',np.hstack([a,b]))" || true
  fi
  "$CPUPY" - "$REP" "$L" <<'PY'
import json, os, sys
rep, L = sys.argv[1:3]; out = {}
for f in sorted(os.listdir(rep)):
    if f.endswith(".json") and not f.startswith("gate") and f != "orbit_proxies.json":
        r = json.load(open(os.path.join(rep, f))); out[f[:-5]] = {k: r.get(k) for k in ("psnr", "ssim", "lpips", "dists")}
for g in ("gate_depth", "gate_radial"):
    f = os.path.join(rep, g + ".json")
    if os.path.exists(f):
        out[g] = json.load(open(f)).get("accept")
f = os.path.join(rep, "orbit_proxies.json")
if os.path.exists(f):
    out["orbit"] = {k: {"needle": v["needle_frac_object_zone"], "spike": v["spike_frac_object_zone"], **v["summary"]} for k, v in json.load(open(f)).items()}
for d in ("orbit", "paths", "offpath"):
    for v in ("before", "after"):
        j = os.path.join(L, d, v, "_viewer.json")
        if os.path.exists(j):
            out[f"{d}_{v}_alpha_hole"] = json.load(open(j)).get("alpha_hole_frac")
json.dump(out, open(os.path.join(rep, "summary.json"), "w"), indent=1); print("COSMETIC_SCORE", json.dumps(out))
PY
}

case "$STAGE" in
  prep) st_prep ;; afx) st_afx ;; distill) st_distill ;; score) st_score ;;
  all) st_prep; st_afx; st_distill; st_score ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "COSMETIC_AFX_DONE $STAGE"
