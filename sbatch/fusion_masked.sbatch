#!/bin/bash
# Change-masked + appearance-compensated fusion of several recordings (configs/fused_<name>.yaml, after
# scripts/fuse_sequences.py), with the current best recipe: EDGS init (scripts/edgs_init.py) + Improved-GS 4M.
#   1. scripts/change_masks.py prep: training set <W>/data (fused train frames, LiDAR depth in fused units) and
#      cross-render datasets; the reference recording's own splat (members/<m>.json ckpt_glob, or CKPT_<i>) is
#      rendered at the other recordings' frames and at its own held-out frames (threshold calibration);
#      change_masks.py masks: DINOv2 change x observed-by-reference -> <image>_valid.png (other recordings only)
#   2. each member's own EDGS+IGS4M held-out model (= edgs_bench.sbatch arm edgs_igs4m, trained if missing)
#   3. EDGS init on the fused training frames, then the arms (3DGRUT fusion worktree, env/make_3dgrut_fusion.sh):
#        plain     fused, no mask, no appearance            (control: the recipe alone)
#        app       + per-recording colour affine on the render (training only; reference = identity)
#        mask      + change masks
#        mask_app  + both
#        solo      control: the reference recording's frames ONLY, same fused (metric) frame and recipe -> separates
#                  what fusing costs from what the fused frame / units cost
#   4. SuperSplat (scripts/viewer_shots.py; primary: --engine current = live v3.5.2 editor, WebGPU, black + grey
#      background -> report/cur_<bg>_*; also the older PlayCanvas 2.7.4 engine -> report/ss_*) at each member's held-out frames (fuse_sequences.py's
#      eval/<m>/fused vs eval/<m>/single for the member's own splat) -> eval_views.py --pred-dir; never-worse
#      gate per member (accept_gate.py); also 3DGRUT renders, and for non-reference members the score after the
#      learnt appearance transform (what the model knows, not what SuperSplat shows); off-path + editor views in
#      the reference's viewer axes vs its own splat; montages. Everything under <out>/pipeline/fusion_masked.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/fusion-masked-%j.out sbatch/fusion_masked.sbatch configs/fused_<name>.yaml
# Env: ARMS ("plain app mask_app"), ENGINES ("current pc274"), STOP_AFTER (masks|baselines), MASK_ARGS (extra change_masks.py masks flags),
#      REFERENCE (member kept fully supervised; default fuse.members[0]), BUDGET (4000000), APP_MODEL (affine),
#      FORCE_MASKS=1 (recompute masks), MOVE ("0 -0.3 1.0", reference dataset units), EVERY_OFF (40).
#SBATCH -J fusion-masked
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; EPY="${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python"
DPY="${DINO_VENV:-$DIFIX_VENV}/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"; FDIR="${THREEDGRUT_FUSION_DIR:-$CEAR_WS/3dgrut-fusion}"
R_ENV=(env TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}" PYTHONPATH="$DDIR")   # render/export
F_ENV=(env TORCH_EXTENSIONS_DIR="${FUSION_TORCH_EXT:-$CEAR_WS/torch-ext-fusion}" PYTHONPATH="$FDIR")     # fusion training
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; F="$OUT/pipeline"; W="$F/fusion_masked"; REP="$W/report"
ITERS="$(get pipeline.train_iterations 30000)"; BUDGET="${BUDGET:-4000000}"; ARMS="${ARMS:-plain app mask_app}"
MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"; ENGINES="${ENGINES:-current pc274}"
[ -f "$F/fuse.json" ] || { echo "no $F/fuse.json: run scripts/fuse_sequences.py --config $CFG first"; exit 1; }
mapfile -t MEMBERS < <("$CPUPY" -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['members']))" "$F/fuse.json")
REF="${REFERENCE:-${MEMBERS[0]}}"
mpipe() { "$CPUPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['members'][sys.argv[2]][sys.argv[3]])" "$F/fuse.json" "$1" "$2"; }
hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
mkdir -p "$W" "$REP"
bash env/make_3dgrut_densify.sh | tail -1; bash env/make_3dgrut_fusion.sh | tail -1
echo "members: ${MEMBERS[*]}  reference: $REF  arms: $ARMS"

echo "== 1. change masks"
[ -f "$W/change_plan.json" ] || "$CPUPY" scripts/change_masks.py prep --fused "$F" --out "$W" --reference "$REF" \
  ${MASK_CKPTS:+--ckpt $MASK_CKPTS} | tail -4
# renders: owner's splat at the target frames (+ opacity) and at its own held-out frames (calibration)
"$CPUPY" - "$W/change_plan.json" > "$W/renders.txt" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
for r in p["pairs"]: print(r["ckpt"], r["dataset"], r["dataset"])
for c in p["calib"].values(): print(c["ckpt"], c["dataset"], c["out"])
PY
while read -r ck ds o; do
  n="$(ls "$ds/images" | grep -c '\.png$')"
  [ "$(ls "$o/opacity" 2>/dev/null | wc -l)" -ge "$n" ] && continue
  CUDA_LAUNCH_BLOCKING=1 "${R_ENV[@]}" "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$ds" --no-metrics \
    --save-dir "$o/render" --save-opacity-dir "$o/opacity" < /dev/null | tail -1
done < "$W/renders.txt"
if [ ! -f "$W/change_masks.json" ] || [ "${FORCE_MASKS:-0}" = 1 ]; then
  "$DPY" scripts/change_masks.py masks --out "$W" ${MASK_ARGS:-} 2>&1 | { grep -v Warning || true; }
  [ -f "$W/change_masks.json" ] || { echo "change_masks.py masks failed"; exit 1; }
fi
[ "${STOP_AFTER:-}" = masks ] && { echo FUSION_MASKED_STOPPED; exit 0; }

echo "== 2. each member's own EDGS + Improved-GS ${BUDGET} held-out model (edgs_bench arm edgs_igs$((BUDGET / 1000000))m)"
declare -A BCK BPLY
for m in "${MEMBERS[@]}"; do
  MP="$(mpipe "$m" pipeline)"; E="$MP/edgs/on"; arm="edgs_igs$((BUDGET / 1000000))m"
  CK="$(ls -t "$E/$arm"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    [ -f "$MP/onpath/data_orig/sparse/0/images.txt" ] || { echo "no $MP/onpath/data_orig (onpath_ablation.sbatch)"; exit 1; }
    mkdir -p "$E"
    [ -f "$E/data/edgs_init.json" ] || { rm -rf "$E/data"; "$EPY" scripts/edgs_init.py --dataset "$MP/onpath/data_orig" --out "$E/data" \
      --test-split-interval 8 --sim3-json "$MP/sim3.json" > "$E/init_on.log" 2>&1 || { tail -30 "$E/init_on.log"; exit 1; }; }
    echo "== $m: $arm"; date
    ( cd "$DDIR" && "${R_ENV[@]}" "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$E/data" out_dir="$E/$arm" \
        experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=8 \
        strategy.budget="$BUDGET" loss.use_depth=false ) > "$E/$arm.log" 2>&1 || { tail -40 "$E/$arm.log"; exit 1; }
    CK="$(ls -t "$E/$arm"/run/*/ckpt_last.pt | head -1)"; date
  fi
  BCK[$m]="$CK"; BPLY[$m]="$E/splat_$arm.ply"
  [ -f "${BPLY[$m]}" ] || "${R_ENV[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "${BPLY[$m]}" | tail -1
  echo "  $m: $CK"
done
[ "${STOP_AFTER:-}" = baselines ] && { echo FUSION_MASKED_STOPPED; exit 0; }

echo "== 3. EDGS init on the fused training frames + arms"
if [ ! -f "$W/edgs_data/edgs_init.json" ]; then
  rm -rf "$W/edgs_data"
  "$EPY" scripts/edgs_init.py --dataset "$W/data" --out "$W/edgs_data" --test-split-interval 0 --sim3-json "$F/sim3.json" \
    > "$W/edgs_init.log" 2>&1 || { tail -40 "$W/edgs_init.log"; exit 1; }
fi
"$CPUPY" -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('args'); print(' ', d)" "$W/edgs_data/edgs_init.json"
case " $ARMS " in *" solo "*)                              # reference-only training set + its own EDGS init
  SD="$W/solo_data"
  if [ ! -f "$W/solo_edgs/edgs_init.json" ]; then
    mkdir -p "$SD/sparse/0"; ln -sfn "$W/data/images" "$SD/images"
    cp "$W/data/sparse/0/cameras.txt" "$W/data/sparse/0/points3D.txt" "$SD/sparse/0/"
    awk -v r="$REF/" 'NF >= 10 && index($10, r) == 1 {print; print ""}' "$W/data/sparse/0/images.txt" > "$SD/sparse/0/images.txt"
    rm -rf "$W/solo_edgs"; "$EPY" scripts/edgs_init.py --dataset "$SD" --out "$W/solo_edgs" --test-split-interval 0 \
      --sim3-json "$F/sim3.json" > "$W/solo_edgs_init.log" 2>&1 || { tail -40 "$W/solo_edgs_init.log"; exit 1; }
  fi
  for d in "$SD" "$W/solo_edgs"; do                         # keep only the cameras its frames use (3DGRUT needs that)
    awk 'NR == FNR {if (NF >= 10) u[$9] = 1; next} /^#/ || ($1 in u)' "$d/sparse/0/images.txt" "$d/sparse/0/cameras.txt" \
      > "$d/sparse/0/cameras.tmp" && mv "$d/sparse/0/cameras.tmp" "$d/sparse/0/cameras.txt"
  done;;
esac
declare -A ACK
for arm in $ARMS; do
  RUN="$W/$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  X=(strategy.budget="$BUDGET" loss.use_depth=false)
  case $arm in
    plain|solo) ;;
    app) X+=(appearance.mode=recording appearance.model="${APP_MODEL:-affine}" appearance.reference="$REF") ;;
    mask) X+=(loss.valid_mask=true) ;;
    mask_app) X+=(loss.valid_mask=true appearance.mode=recording appearance.model="${APP_MODEL:-affine}" appearance.reference="$REF") ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  if [ -z "$CK" ]; then
    echo "== $arm: ${X[*]}"; date
    TD="$W/edgs_data"; [ "$arm" = solo ] && TD="$W/solo_edgs"
    ( cd "$FDIR" && "${F_ENV[@]}" "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$TD" out_dir="$RUN" \
        experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 "${X[@]}" ) \
      > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"; date
    grep -E "Appearance|Valid masks" "$RUN.log" | head -2 || true
  fi
  ACK[$arm]="$CK"; mkdir -p "$W/ply"
  [ -f "$W/ply/$arm.ply" ] || "${R_ENV[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$W/ply/$arm.ply" | tail -1
  [ -f "$(dirname "$CK")/appearance.json" ] && cat "$(dirname "$CK")/appearance.json" | tr -d '\n ' | head -c 600 && echo
done

echo "== 4. held-out frames of each member: SuperSplat (engines: $ENGINES) + 3DGRUT renders"
bash env/make_viewer_venv.sh | tail -1
appmatch() {  # appearance.json member src dst: the renders through that member's learnt colour transform
  "$CPUPY" - "$@" <<'PY'
import json, os, sys, numpy as np
from PIL import Image
j, m, src, dst = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
T = j["transforms"].get(m); os.makedirs(dst, exist_ok=True)
A, b = (np.array(T["A"]), np.array(T["b"])) if T else (np.eye(3), np.zeros(3))
for n in os.listdir(src):
    if n.endswith(".png") and not n.startswith("_"):
        x = np.asarray(Image.open(os.path.join(src, n)).convert("RGB"), np.float32) / 255
        Image.fromarray((np.clip(x @ A.T + b, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(dst, n))
PY
}
shots() {  # engine ply dataset member tag [appearance.json]: render + score -> report/<kind>_<member>_<tag>.json
  local eng="$1" ply="$2" ds="$3" m="$4" t="$5" aj="${6:-}" o
  if [ "$eng" = current ]; then
    o="$W/cur/$m/$t"
    [ -f "$REP/cur_grey_${m}_$t.json" ] || { "$VPY" scripts/viewer_shots.py --engine current --bg black grey --ply "$ply" --dataset "$ds" --out-dir "$o" | tail -1
      for bg in black grey; do "$GPY" scripts/eval_views.py --pred-dir "$o/$bg" --dataset "$ds" --tag "cur_$bg:$m:$t" \
        --json-out "$REP/cur_${bg}_${m}_$t.json" | tail -1; done; }
    o="$o/black"; k=curam_black
  else
    o="$W/ss/$m/$t"
    [ -f "$REP/ss_${m}_$t.json" ] || { "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$ds" --out-dir "$o" | tail -1
      "$GPY" scripts/eval_views.py --pred-dir "$o" --dataset "$ds" --tag "ss:$m:$t" --json-out "$REP/ss_${m}_$t.json" | tail -1; }
    k=ssam
  fi
  if [ -n "$aj" ] && [ -f "$aj" ] && [ "$m" != "$REF" ] && [ ! -f "$REP/${k}_${m}_$t.json" ]; then
    appmatch "$aj" "$m" "$o" "${o}_am"
    "$GPY" scripts/eval_views.py --pred-dir "${o}_am" --dataset "$ds" --tag "$k:$m:$t" --json-out "$REP/${k}_${m}_$t.json" | tail -1
  fi
}
for m in "${MEMBERS[@]}"; do
  EV="$F/eval/$m"
  for eng in $ENGINES; do shots "$eng" "${BPLY[$m]}" "$EV/single" "$m" base; done
  [ -f "$REP/gr_${m}_base.json" ] || CUDA_LAUNCH_BLOCKING=1 "${R_ENV[@]}" "$GPY" scripts/eval_views.py --checkpoint "${BCK[$m]}" \
    --dataset "$EV/single" --tag "gr:$m:base" --json-out "$REP/gr_${m}_base.json" | tail -1
  for arm in $ARMS; do
    for eng in $ENGINES; do shots "$eng" "$W/ply/$arm.ply" "$EV/fused" "$m" "$arm" "$(dirname "${ACK[$arm]}")/appearance.json"; done
    [ -f "$REP/gr_${m}_$arm.json" ] || CUDA_LAUNCH_BLOCKING=1 "${R_ENV[@]}" "$GPY" scripts/eval_views.py --checkpoint "${ACK[$arm]}" \
      --dataset "$EV/fused" --tag "gr:$m:$arm" --json-out "$REP/gr_${m}_$arm.json" | tail -1
  done
done
for arm in $ARMS; do                       # never-worse gate per member: gate_cur_black_<arm> is THE decision
  for k in cur_black cur_grey ss gr; do
    [ -f "$REP/${k}_${MEMBERS[0]}_base.json" ] || continue
    PAIRS=(); for m in "${MEMBERS[@]}"; do PAIRS+=(--pair "$m" "$REP/${k}_${m}_base.json" "$REP/${k}_${m}_$arm.json"); done
    "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate_${k}_$arm.json" | tail -1 | sed "s/^/  $k $arm: /" || true
  done
done

echo "== 5. off-path + editor views in the reference's viewer axes ($REF)"
RCFG="$(mpipe "$REF" config)"; RP="$(mpipe "$REF" pipeline)"
FRAME="$("$CPUPY" scripts/cfg_get.py "$RCFG" pipeline.export_frame_model "")"; RT="$RP/viewer_bench/reference_dataset"
if [ -n "$FRAME" ] && [ -f "$RT/sparse/0/images.txt" ]; then
  OFFB="$W/offpath_cur/base"; RB="$W/ply/base_refaxes.ply"
  [ -f "$RB" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$RP/dataset/sparse/0" --gt-model "$FRAME" \
    --in-ply "${BPLY[$REF]}" --out-ply "$RB" | tail -1
  [ -f "$OFFB/_supersplat_editor.png" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$RB" --dataset "$RT" \
    --every "$EVERY_OFF" --move $MOVE --out-dir "$OFFB" --editor | tail -1
  for arm in $ARMS; do
    [ -f "$W/ply/${arm}_refaxes.ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$F/eval/$REF/fused/sparse/0" \
      --gt-model "$FRAME" --in-ply "$W/ply/$arm.ply" --out-ply "$W/ply/${arm}_refaxes.ply" | tail -1
    [ -f "$W/offpath_cur/$arm/_supersplat_editor.png" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$W/ply/${arm}_refaxes.ply" --dataset "$RT" \
      --every "$EVERY_OFF" --move $MOVE --out-dir "$W/offpath_cur/$arm" --editor | tail -1
  done
else
  OFFB=""; echo "  no viewer reference for $REF: skipped"
fi

echo "== 6. summary + montages"
"$CPUPY" - "$W" "$REF" "$ARMS" "$OFFB" "$F" "${MEMBERS[@]}" <<'PY'
import cv2, json, os, sys, numpy as np
W, ref, arms, offb, F, members = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4], sys.argv[5], sys.argv[6:]
rep = os.path.join(W, "report"); ld = lambda f: json.load(open(os.path.join(rep, f))) if os.path.exists(os.path.join(rep, f)) else None
keys = ("psnr", "ssim", "lpips")
out = {"reference": ref, "members": {}, "gates": {}, "change_masks": json.load(open(os.path.join(W, "change_masks.json")))["pairs"]}
for m in members:
    out["members"][m] = {}
    for kind in ("cur_black", "cur_grey", "curam_black", "ss", "ssam", "gr"):
        for t in ["base"] + arms:
            r = ld(f"{kind}_{m}_{t}.json")
            if r: out["members"][m][f"{kind}:{t}"] = {k: r[k] for k in keys}
for a in arms:
    for g in ("cur_black", "cur_grey", "ss", "gr"):
        d = ld(f"gate_{g}_{a}.json")
        if d: out["gates"][f"{g}:{a}"] = {"accept": d["accept"], "failures": d["failures"]}
json.dump(out, open(os.path.join(rep, "fusion_summary.json"), "w"), indent=2)
print("SuperSplat v3.5.2 black/grey bg (cur_*), PlayCanvas 2.7.4 (ss), 3DGRUT (gr), + learnt appearance (*am*); held-out per member")
for m, rows in out["members"].items():
    print(f"--- {m}" + ("  (reference)" if m == ref else ""))
    for t, r in rows.items(): print(f"  {t:18s}" + "".join(f"{r[k]:10.4f}" for k in keys))
print("gates:", json.dumps({k: v["accept"] for k, v in out["gates"].items()}))
def lab(im, t):
    im = cv2.resize(im, (400, 300)); cv2.putText(im, t, (6, 22), 0, 0.7, (0, 0, 255), 2); return im
for m in members:                                                    # real | base | arms, 5 held-out frames
    S = os.path.join(W, "cur", m); g = os.path.join(F, "eval", m, "fused", "images")
    names = sorted(n for n in os.listdir(os.path.join(S, "base", "black")) if not n.startswith("_"))
    rows = []
    for n in names[:: max(1, len(names) // 5)][:5]:
        cols = [lab(cv2.imread(os.path.join(g, n)), "real")]
        for t in ["base"] + arms:
            p = os.path.join(S, t, "black", n)
            if os.path.exists(p): cols.append(lab(cv2.imread(p), t))
        rows.append(np.hstack(cols))
    cv2.imwrite(os.path.join(rep, f"onpath_montage_{m}.jpg"), np.vstack(rows)); print("montage ->", os.path.join(rep, f"onpath_montage_{m}.jpg"))
if offb:
    dirs = [("base", offb)] + [(a, os.path.join(W, "offpath_cur", a)) for a in arms]
    names = sorted(n for n in os.listdir(offb) if not n.startswith("_"))[:6]
    rows = [np.hstack([lab(cv2.imread(os.path.join(d, n)), t) for t, d in dirs if os.path.exists(os.path.join(d, n))]) for n in names]
    if rows: cv2.imwrite(os.path.join(rep, "offpath_montage.jpg"), np.vstack(rows)); print("montage ->", os.path.join(rep, "offpath_montage.jpg"))
    eds = [lab(cv2.imread(p), t) for t, d in dirs for p in [os.path.join(d, "_supersplat_editor.png")] if os.path.exists(p)]
    if eds: cv2.imwrite(os.path.join(rep, "editor_montage.jpg"), np.hstack(eds)); print("montage ->", os.path.join(rep, "editor_montage.jpg"))
PY
echo FUSION_MASKED_DONE
