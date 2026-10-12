#!/bin/bash
# Viewer fine-tune of exported splats (scripts/finetune_ply_viewer.py): re-optimise the .ply through a
# differentiable copy of the viewer's rasteriser (VIEWER=ss352: today's SuperSplat 3.x; pc274: PlayCanvas 2.7.4), then
# score with the viewer itself (viewer_shots.py --engine $ENGINE, black bg + grey; eval_views.py --pred-dir).
#   1. renderer agreement: matched renderer vs viewer_shots renders of the same .ply (compare)
#   2. held-out arms (trained WITHOUT every 8th frame): base vs fine-tuned on $P/onpath/test, accept gate,
#      off-path moved views (MOVE) for visuals
#   3. all-frames deliverables: exports/<d>.ply -> exports/<d>_$TAG.ply (+ _refaxes), gated vs the input
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/finetune_viewer-%j.out sbatch/finetune_viewer.sbatch configs/<seq>.yaml
# Env: STAGE (all, or a space list of: compare attrib fix smoke heldout deliver), ARMS ("edgs_igs4m lidar_init"), ITERS (7000),
# ENGINE (current | pc274: the viewer scored), DSCALE (current: 1.8 = on-screen editor canvas px per rendered px in
# viewer_shots' headless editor, fitted: 52.7 dB agreement), DSCALE_TRAIN ($DSCALE, or a:b = random per step), VIEWER (ss352 for current, pc274 for pc274: the fine-tune target), RASTER (exact), DEPTH (0 = no LiDAR term), FTARGS (extra finetune flags), TAG (output suffix),
# MOVE ("0 -0.3 1.0"), EVERY_OFF (40), DELIV ("edgs_igs edgs_igs_complete": exports/<name>.ply).
#SBATCH -J finetune-viewer
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VB="$P/viewer_bench"; S="$P/supersplat_check"
FRAME="$(get pipeline.export_frame_model)"
STAGE="${STAGE:-all}"; ARMS="${ARMS:-edgs_igs4m lidar_init}"; ITERS="${ITERS:-7000}"; ENGINE="${ENGINE:-current}"
VIEWER="${VIEWER:-$([ "$ENGINE" = current ] && echo ss352 || echo pc274)}"
TAG="${TAG:-viewerft$([ "$VIEWER" = ss352 ] || echo "_$VIEWER")}"
DSCALE="${DSCALE:-$([ "$ENGINE" = current ] && echo 1.8 || echo 1)}"; DSCALE_TRAIN="${DSCALE_TRAIN:-$DSCALE}"
RASTER="${RASTER:-exact}"; DEPTH="${DEPTH:-0}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"
has() { [ "$STAGE" = all ] || [[ " $STAGE " == *" $1 "* ]]; }
PLYD="$P/viewer_ft"                                         # fine-tuned / fixed .plys (all engines)
F="$PLYD$([ "$ENGINE" = pc274 ] || echo "/$ENGINE")"; REP="$F/report"; TEST="$P/onpath/test"; TRAIN="$P/onpath/data_orig"
ALLDS="$P/igs_full/edgs_data"; [ -d "$ALLDS/sparse/0" ] || ALLDS="$TRAIN"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
bash env/make_viewer_venv.sh
"$SPY" -c "import gsplat, torch; print('gsplat', gsplat.__version__, 'torch', torch.__version__)"

# held-out-trained source checkpoints (exported to $VB/<arm>.ply unless already there)
declare -A CK=(
  [edgs_igs4m]="$P/edgs/on/edgs_igs4m/run/*/ckpt_last.pt"
  [lidar_init]="$P/onpath/arm_A/run/*/ckpt_last.pt"
)
ply_of() {  # arm -> exported held-out .ply
  local n="$1" ck
  if [ ! -f "$VB/$n.ply" ]; then
    ck="$(ls -t ${CK[$n]} 2>/dev/null | head -1 || true)"; [ -n "$ck" ] || { echo "no checkpoint for $n" >&2; return 1; }
    PYTHONPATH="$CEAR_WS/3dgrut-densify" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-densify" \
      "$GPY" scripts/export_gaussian_ply.py --checkpoint "$ck" --out "$VB/$n.ply" >&2
  fi
  echo "$VB/$n.ply"
}
BGARGS=(); BLK=""; [ "$ENGINE" = current ] && { BGARGS=(--engine current --bg black grey); BLK="/black"; }
shots() {  # ply dataset outdir [extra viewer_shots args]: viewer renders (skipped if complete); current: <od>/{black,grey}
  local ply="$1" ds="$2" od="$3"; shift 3
  [ -f "$od/_viewer.json" ] && return 0
  "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$ds" --out-dir "$od" "${BGARGS[@]}" "$@" || \
    { sleep 60; "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$ds" --out-dir "$od" "${BGARGS[@]}" "$@"; } || \
    echo "VIEWER_SHOTS FAILED: $od"                         # the live editor occasionally never loads: retry once
}
score() {  # tag renderdir -> $REP/<tag>.json (viewer metrics on the held-out frames; black bg) [+ $REP/grey/<tag>.json]
  [ -f "$2/_viewer.json" ] || { echo "no renders for $1"; return 0; }
  "$GPY" scripts/eval_views.py --pred-dir "$2$BLK" --dataset "$TEST" --tag "$1" --json-out "$REP/$1.json" | tail -1
  if [ -d "$2/grey" ]; then mkdir -p "$REP/grey"
    "$GPY" scripts/eval_views.py --pred-dir "$2/grey" --dataset "$TEST" --tag "$1" --json-out "$REP/grey/$1.json" | tail -1; fi
}
FT() {  # in.ply out.ply dataset interval [extra]
  local in="$1" out="$2" ds="$3" iv="$4"; shift 4
  "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$in" --out "$out" --dataset "$ds" --test-split-interval "$iv" \
    --viewer "$VIEWER" --raster "$RASTER" --display-scale "$DSCALE_TRAIN" --eval-display-scale "$DSCALE" --iters "$ITERS" --depth-lambda "$DEPTH" --eval-dataset "$TEST" ${FTARGS:-} "$@"
}

# ---- 1. renderer agreement (needs viewer renders of the base .ply at the test cameras)
if has compare || has smoke; then
  b="$(ply_of edgs_igs4m)"; shots "$b" "$TEST" "$F/onpath/edgs_igs4m"
  "$SPY" scripts/finetune_ply_viewer.py compare --ply "$b" --dataset "$TEST" --viewer-dir "$F/onpath/edgs_igs4m$BLK" \
    --engine "$ENGINE" --display-scale "$DSCALE" --every "$([ "$STAGE" = all ] && echo 1 || echo 4)" --save-dir "$F/compare" \
    --variants "${CMP_VARIANTS:-gsplat_0.3,gsplat_0.075,pc274_exact,ss352_exact,ss352_exact_q}" \
    --json-out "$REP/agreement_edgs_igs4m.json"
fi
# ---- 1b. gap attribution: each viewer effect switched off in the matched renderer, scored vs the real images
if has attrib; then
  b="$(ply_of edgs_igs4m)"; shots "$b" "$TEST" "$F/onpath/edgs_igs4m"
  if [ "$VIEWER" = ss352 ]; then V=gsplat_0.3,gsplat_0,ss352_exact,ss352_exact_q,ss352_exact_nocull,ss352_exact_nocut,ss352_exact_nocull_nocut,ss352_exact_nocull_nocut_d0,ss352_exact_bug,ss352_exact_r1,pc274_exact
  else V=gsplat_0.3,gsplat_0.075,gsplat_0,pc274_exact,pc274_exact_q,pc274_exact_nocull,pc274_exact_nocut,pc274_exact_nobug,pc274_exact_nocull_nocut,pc274_exact_nocull_nocut_nobug,pc274_exact_nocull_nocut_nobug_d0,pc274_exact_d0.3,ss352_exact; fi
  "$SPY" scripts/finetune_ply_viewer.py compare --ply "$b" --dataset "$TEST" --viewer-dir "$F/onpath/edgs_igs4m$BLK" \
    --engine "$ENGINE" --display-scale "$DSCALE" --variants "$V" --json-out "$REP/attribution_edgs_igs4m.json"
fi

# ---- 1c. energy-preserving minimum footprint (scripts/viewer_fix_ply.py), held-out arms, never-worse gate
if has fix; then
  for n in $ARMS; do
    b="$(ply_of "$n")" || continue; shots "$b" "$TEST" "$F/onpath/$n"; score "$n" "$F/onpath/$n"
    for pr in 0 0.02; do
      t="${n}_fix$([ "$VIEWER" = ss352 ] || echo "_$VIEWER")$([ "$pr" = 0 ] || echo "_p$pr")"; o="$PLYD/$n/$t.ply"
      [ -f "$o" ] || "$SPY" scripts/viewer_fix_ply.py --ply "$b" --dataset "$TRAIN" --test-split-interval 8 --viewer "$VIEWER" \
        --display-scale "$DSCALE" --prune-opacity "$pr" --out "$o"
      shots "$o" "$TEST" "$F/onpath/$t"; score "$t" "$F/onpath/$t"
      shots "$o" "$TRAIN" "$F/offpath/$t" --every "$EVERY_OFF" --move $MOVE
      "$CPUPY" scripts/accept_gate.py --pair heldout "$REP/$n.json" "$REP/$t.json" --out "$REP/gate_$t.json" \
        && echo "GATE $t ACCEPT" || echo "GATE $t REJECT"
    done
    shots "$b" "$TRAIN" "$F/offpath/$n" --every "$EVERY_OFF" --move $MOVE
  done
fi

if [[ " $STAGE " == *" smoke "* ]]; then   # 0-LR reproduces step 0; single-frame overfit decreases; short real run
  b="$(ply_of edgs_igs4m)"; mkdir -p "$PLYD/smoke"
  FT "$b" "$PLYD/smoke/lr0.ply" "$TRAIN" 8 --iters 50 --lr-mult 0 --eval-interval 0 --eval-every-frame 8
  FT "$b" "$PLYD/smoke/one.ply" "$TRAIN" 8 --iters 200 --overfit-one --log-every 20 --eval-interval 0 --eval-every-frame 8
  FT "$b" "$PLYD/smoke/short.ply" "$TRAIN" 8 --iters 600 --log-every 100 --eval-interval 300 --eval-every-frame 4
  echo FINETUNE_VIEWER_SMOKE_DONE
fi
[[ " $STAGE " == *" smoke "* ]] && [ "$STAGE" = smoke ] && exit 0

# ---- 2. held-out arms: base vs fine-tuned, SuperSplat engine
if has heldout; then
  for n in $ARMS; do
    b="$(ply_of "$n")" || continue; ft="$PLYD/$n/${n}_$TAG.ply"
    [ -f "$ft" ] || FT "$b" "$ft" "$TRAIN" 8 2>&1 | tee "$PLYD/${n}_$TAG.log" || true
    [ -f "$ft" ] || { echo "fine-tune of $n failed"; continue; }
    shots "$b" "$TEST" "$F/onpath/$n"; shots "$ft" "$TEST" "$F/onpath/${n}_$TAG"
    score "$n" "$F/onpath/$n"; score "${n}_$TAG" "$F/onpath/${n}_$TAG"
    shots "$b" "$TRAIN" "$F/offpath/$n" --every "$EVERY_OFF" --move $MOVE
    shots "$ft" "$TRAIN" "$F/offpath/${n}_$TAG" --every "$EVERY_OFF" --move $MOVE
    "$CPUPY" scripts/accept_gate.py --pair heldout "$REP/$n.json" "$REP/${n}_$TAG.json" --out "$REP/gate_${n}_$TAG.json" \
      && echo "GATE ${n}_$TAG ACCEPT" || echo "GATE ${n}_$TAG REJECT"
    for o in "$PLYD/$n/${n}"_viewerft_*.ply; do             # other targets' fine-tunes, scored in this engine too
      [ -f "$o" ] && [ "$o" != "$ft" ] || continue; t="$(basename "${o%.ply}")"
      shots "$o" "$TEST" "$F/onpath/$t"; score "$t" "$F/onpath/$t"
    done
  done
fi

# ---- 3. all-frames deliverables (trained on every frame: on-path scores are NOT held-out); gate vs the input
if has deliver; then
  EX="$OUT/exports"
  for d in ${DELIV:-edgs_igs edgs_igs_complete}; do
    b="$EX/$d.ply"; ft="$EX/${d}_$TAG.ply"; t="${d}_all"; [ "$d" = edgs_igs ] || t="$d"
    [ -f "$b" ] || { echo "no $b"; continue; }
    [ -f "$ft" ] || FT "$b" "$ft" "$ALLDS" 0 2>&1 | tee "$PLYD/${t}_$TAG.log" || true
    [ -f "$ft" ] || { echo "fine-tune of $d failed"; continue; }
    mv -f "${ft%.ply}_ftv.json" "$PLYD/" 2>/dev/null || true          # keep exports/ to plys
    "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$ft" \
      --out-ply "$EX/${d}_${TAG}_refaxes.ply"
    shots "$b" "$TEST" "$F/onpath/$t"; shots "$ft" "$TEST" "$F/onpath/${t}_$TAG"
    score "$t" "$F/onpath/$t"; score "${t}_$TAG" "$F/onpath/${t}_$TAG"
    "$CPUPY" scripts/accept_gate.py --pair onpath_all_frames "$REP/$t.json" "$REP/${t}_$TAG.json" --out "$REP/gate_${t}_$TAG.json" \
      && echo "GATE ${t}_$TAG ACCEPT" || echo "GATE ${t}_$TAG REJECT"
    shots "$b" "$TRAIN" "$F/offpath/$t" --every "$EVERY_OFF" --move $MOVE
    shots "$ft" "$TRAIN" "$F/offpath/${t}_$TAG" --every "$EVERY_OFF" --move $MOVE
  done
fi

# ---- summary + montages (real | base | fine-tuned per arm)
"$CPUPY" - "$F" "$REP" "$TEST/images" "$TAG" "$BLK" <<'PY'
import json, os, sys, cv2, numpy as np
F, rep, gt, tag, blk = sys.argv[1:6]; blk = blk.strip("/")
print(f"{'splat':32s}{'viewer PSNR':>12s}{'SSIM':>8s}{'LPIPS':>8s}{'grey PSNR':>11s}{'alpha holes':>12s}")
res = {}
for f in sorted(os.listdir(rep)):
    if f.endswith(".json") and not f.startswith(("gate", "agreement", "attribution", "summary")):
        r = json.load(open(os.path.join(rep, f))); res[r["tag"]] = r
        g = os.path.join(rep, "grey", f); r["grey"] = json.load(open(g)) if os.path.exists(g) else None
        v = os.path.join(F, "onpath", r["tag"], "_viewer.json"); r["alpha_hole_frac"] = json.load(open(v)).get("alpha_hole_frac") if os.path.exists(v) else None
        gp = f"{r['grey']['psnr']:11.2f}" if r["grey"] else f"{'-':>11s}"; ah = f"{r['alpha_hole_frac']:12.4f}" if r["alpha_hole_frac"] is not None else f"{'-':>12s}"
        print(f"{r['tag']:32s}{r['psnr']:12.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}{gp}{ah}")
json.dump(res, open(os.path.join(rep, f"summary_{tag}.json"), "w"), indent=2)
bases = [t for t in res if not t.endswith("_" + tag) and t + "_" + tag in res]
def montage(sub, names, with_gt, out):
    rows = []
    for nm in names:
        ims, labs = ([cv2.imread(os.path.join(gt, nm))], ["real"]) if with_gt else ([], [])
        for b in bases:
            for t in (b, b + "_" + tag):
                p = os.path.join(F, sub, t, blk, nm)
                if os.path.exists(p): ims.append(cv2.imread(p)); labs.append(t)
        if not ims: continue
        ims = [cv2.resize(x, (400, 300)) for x in ims]
        for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    if rows:
        w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
if bases:
    d = os.path.join(F, "onpath", bases[0], blk); on = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    montage("onpath", on[::11][:5], True, os.path.join(F, f"onpath_montage_{tag}.jpg"))
    d = os.path.join(F, "offpath", bases[0], blk)
    if os.path.isdir(d):
        off = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        montage("offpath", off[::max(1, len(off) // 5)][:5], False, os.path.join(F, f"offpath_montage_{tag}.jpg"))
PY
echo FINETUNE_VIEWER_DONE
