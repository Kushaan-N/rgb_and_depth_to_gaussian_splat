#!/bin/bash
# Held-out benchmark of the optional fidelity methods in scripts/methods (FreGS frequency loss, hard-pixel mining,
# image-gradient loss, parameter EMA, cosine LR anneal, Pixel-GS densification) on top of the CEAR viewer recipe
# igs4m_pc274 at ITERS (default 60000): train_viewer_gsplat.py on $P/edgs/on/data (every 8th frame held out =
# $P/onpath/test), exported PLY rendered by the LIVE SuperSplat (viewer_shots.py --engine current, black + grey), scored
# with eval_views.py --pred-dir + DISTS (piq), gated (accept_gate.py, strict and --perceptual) against the baseline arm
# $VT/igs4m_pc274 (its viewer scores are copied to $VT/report_fm/ and get DISTS there; $VT/report is left alone).
#   ARMS="fm_ema fm_freq0.1" sbatch -A $CEAR_SLURM_ACCOUNT -p gpu,gpu-preempt --gres=gpu:1 --array=0-1 \
#       sbatch/fidelity_methods_bench.sbatch configs/mocap2_well-lit_trot.yaml      # one arm per array task
#   MODE=report ARMS="..." sbatch ... sbatch/fidelity_methods_bench.sbatch configs/<seq>.yaml   # table only
# Arm = fm_<token>[_<token>...]; tokens: seed<n> | ema[<decay>] (0.999) | ann[<final>] (cosine LR anneal from 0.75,
#   final x0.1) | freq[<lambda>] (0.1) | hard[<lambda>] (0.4) | grad[<lambda>] (0.1) | pixgs | pixgsd<x> |
#   shn<lambda> (--shn-reg) | c2f<a> (--c2f) | emul / eerr (--eas-error mul / err) | aniso<lambda> (--aniso-reg, ratio 10) | ssl<x> (--ssim-lambda) | srad / sdual (--sort radial / dual) | ms<lambda> (--msssim-lambda) | blur<K> (--blur-kernel) | blr<x> (--blur-lr) | gt<threshold> (--grad-threshold) | bilag (--appearance bilagrid) | ss352 (viewer profile; default pc274) |
#   drop<r> (--drop-rate, DropGaussian) | vw<lambda> (--vwarp-lambda) | vwp<p> (--vwarp-prob) | vwj<x> (--vwarp-jitter) | pose<lr> (--pose-lr) | poshp<N> (--pose-hp) | sam<lambda> (--sam-lambda) | samr<rho> (--sam-rho) | hp<t> (flags of sbatch/hp_search.sbatch trial t).
#   hylp / hylpr (HY-World 2.0 LPIPS 0.2->0.1; r: official [0,1] input quirk) | hyn / hynm (HY-World depth-to-normal vs WorldMirror 2.0 /
#   MoGe-2 normals, $P/hyworld/normals_{wm,moge2}) | shw<gamma> (--sharp-gamma) | spj<px> (--spj-px).
# TSI (8): hold-out interval of the training set (0 + VT_DATASET=<all-frames set>: deliverable-style splats); 
# fm_ens[_jt<x>][_jr<deg>][_pf<frac>][_it<n>][_lr<x>][_pmi][_sdual|_srad][_t<label>] (pmi: pseudo views interpolated on the path): multi-seed ensemble distillation (scripts/methods/ensemble_distill.py)
#   of the arms in $TEACHERS (fm_ensv...: the same inside the viewer-exact fine-tune, see ensv_arm()) (default "igs4m_pc274 fm_seed1 fm_seed2", which must exist) into one splat.
# Outputs: $VT/<arm>/{splat.ply,train.json,train.log,methods.json}, $VT/shots/{onpath,offpath}/<arm>/,
#   $VT/report_fm/<arm>_<bg>.json, gate_<arm>{,_perc,_reg}.json, summary_fm.md. FT_ITERS=n: also "<arm>~ft" (see ft_arm()). Also "registered" scores <arm>_<bg>_reg.json:
#   the same renders vs the real frames resampled into the viewer_shots camera (scripts/register_gt.py; removes the
#   ~0.2 px principal-point rounding of the centred canvas that costs every splat ~1.4 dB on CEAR).
#SBATCH -J fm-bench
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 03:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
OUT="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
VT="${VT_OUT:-$P/viewer_train_60k}"; REP="$VT/report_fm"; DS="${VT_DATASET:-$P/edgs/on/data}"; TEST="$P/onpath/test"
ITERS="${ITERS:-60000}"; BASE="${BASE_ARM:-igs4m_pc274}"; ARMS="${ARMS:-fm_ema}"
SORTS="${SORTS:-depth radial}"   # SuperSplat Sort Modes scored (see score())
REG="${REG_GT:-$P/onpath/test_viewreg_current}"   # real frames resampled into the viewer_shots camera (scripts/register_gt.py)
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

arm_args() {  # fm_<tokens> -> A (extra trainer flags), PROF
  local t; IFS=_ read -ra T <<< "${1#fm_}"; A=(); PROF=pc274
  for t in "${T[@]}"; do
    case "$t" in
      base) ;;                                   seed[0-9]*) A+=(--seed "${t#seed}") ;;
      ema|ema0.*) A+=(--ema-decay "$([ "$t" = ema ] && echo 0.999 || echo "${t#ema}")") ;;
      ann|ann[0-9]*) A+=(--anneal-start 0.75 --anneal-final "$([ "$t" = ann ] && echo 0.1 || echo "${t#ann}")") ;;
      freq|freq[0-9]*) A+=(--freq-lambda "$([ "$t" = freq ] && echo 0.1 || echo "${t#freq}")") ;;
      hard|hard[0-9]*) A+=(--hard-lambda "$([ "$t" = hard ] && echo 0.4 || echo "${t#hard}")") ;;
      grad|grad[0-9]*) A+=(--grad-lambda "$([ "$t" = grad ] && echo 0.1 || echo "${t#grad}")") ;;
      pixgs) A+=(--pixelgs) ;;                   pixgsd*) A+=(--pixelgs --pixelgs-depth "${t#pixgsd}") ;;
      shn*) A+=(--shn-reg "${t#shn}") ;;    c2f*) A+=(--c2f "${t#c2f}") ;;
      eerr|emul) A+=(--eas-error "${t#e}") ;;     aniso*) A+=(--aniso-reg "${t#aniso}") ;;
      ssl*) A+=(--ssim-lambda "${t#ssl}") ;;    ms*) A+=(--msssim-lambda "${t#ms}") ;;    blur*) A+=(--blur-kernel "${t#blur}") ;;    blr*) A+=(--blur-lr "${t#blr}") ;;
      gt*) A+=(--grad-threshold "${t#gt}") ;;   srad) A+=(--sort radial) ;;   sdual) A+=(--sort dual) ;;   bilag) A+=(--appearance bilagrid) ;;
      hp[0-9]*) read -ra X < "${HP_OUT:-$P/hp_search}/trial_${t#hp}/flags"; A+=("${X[@]}") ;;   # sbatch/hp_search.sbatch trial
      drop*) A+=(--drop-rate "${t#drop}") ;;  vwp*) A+=(--vwarp-prob "${t#vwp}") ;;  vwj*) A+=(--vwarp-jitter "${t#vwj}") ;;
      vw*) A+=(--vwarp-lambda "${t#vw}") ;;     samr*) A+=(--sam-rho "${t#samr}") ;;  sam*) A+=(--sam-lambda "${t#sam}") ;;     poshp*) A+=(--pose-hp "${t#poshp}") ;;  pose*) A+=(--pose-lr "${t#pose}") ;;
      ss352|pc274) PROF="$t" ;;
      hylp) A+=(--hy-lpips 0.2:0.1) ;;  hylpr) A+=(--hy-lpips 0.2:0.1 --hy-lpips-raw) ;;   # HY-World 2.0 objective (scripts/methods/hyworld.py)
      hyn|hynm) A+=(--hy-normal 0.1:0.5 --hy-normal-dir "$P/hyworld/normals_$([ "$t" = hyn ] && echo wm || echo moge2)") ;;   # sbatch/hyworld_normals.sbatch
      shw*) A+=(--sharp-gamma "${t#shw}") ;;   spj*) A+=(--spj-px "${t#spj}") ;;   # sharpness-weighted frames / sub-pixel jitter
      *) echo "unknown token $t in $1"; return 1 ;;
    esac
  done
}

dists() {  # renderdir json [gt dir]: add DISTS (piq) in place
  PYTHONPATH="$PIQ_LIB" "$GPY" - "$1" "${3:-$TEST/images}" "$2" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
if "dists" not in r:
    t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
    m = piq.DISTS(reduction="none").cuda(); v = []
    for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
        with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
    r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2)
print("DISTS", r["tag"], r["dists"])
PY
}

score() {  # <name> <ply>: live SuperSplat at the held-out cameras (black, grey) -> $REP/<name>_<bg>.json (+DISTS); off-path
  # Sort Mode (SuperSplat >= 3.7.0) is forced per mode in $SORTS: depth -> shots/onpath/<name>, <name>_<bg>.json (the
  # pre-3.7 image formation); radial -> shots/onpath_rad/<name>, <name>_rad_<bg>.json (3.7's default). SORTS=default keeps
  # the old behaviour (whatever the editor defaults to, unrecorded).
  local n="$1" ply="$2" bg m sm sd st
  for m in $SORTS; do
    sm=(); sd=onpath; st=""
    case "$m" in depth) sm=(--sort-mode depth) ;; radial) sm=(--sort-mode radial); sd=onpath_rad; st=_rad ;; esac
    [ -f "$VT/shots/$sd/$n/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black grey "${sm[@]}" --ply "$ply" \
      --dataset "$TEST" --out-dir "$VT/shots/$sd/$n" | tail -1 || { sleep 60; "$VPY" scripts/viewer_shots.py --engine current \
      --bg black grey "${sm[@]}" --ply "$ply" --dataset "$TEST" --out-dir "$VT/shots/$sd/$n" | tail -1; }
    for bg in black grey; do
      [ -f "$REP/${n}${st}_$bg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$VT/shots/$sd/$n/$bg" --dataset "$TEST" \
        --tag "${n}${st}_$bg" --json-out "$REP/${n}${st}_$bg.json" | tail -1
      dists "$VT/shots/$sd/$n/$bg" "$REP/${n}${st}_$bg.json"
      [ -n "$st" ] && continue
      [ -f "$REP/${n}_${bg}_reg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$VT/shots/$sd/$n/$bg" --gt-dir "$REG" \
        --tag "${n}_${bg}_reg" --json-out "$REP/${n}_${bg}_reg.json" | tail -1
      dists "$VT/shots/$sd/$n/$bg" "$REP/${n}_${bg}_reg.json" "$REG"
    done
  done
  [ -f "$VT/shots/offpath/$n/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$ply" \
    --dataset "$TEST" --every 8 --move 0 -0.3 1.0 --out-dir "$VT/shots/offpath/$n" | tail -1 || true
}
gate() {  # <arm> [base]: strict + perceptual gate vs the baseline (default $BASE)
  local B="${2:-$BASE}"
  local pr=(--pair black "$REP/${B}_black.json" "$REP/$1_black.json" --pair grey "$REP/${B}_grey.json" "$REP/$1_grey.json")
  "$CPUPY" scripts/accept_gate.py "${pr[@]}" --out "$REP/gate_$1.json" | tail -1 || true
  "$CPUPY" scripts/accept_gate.py "${pr[@]}" --perceptual --out "$REP/gate_$1_perc.json" | tail -1 || true
  [ -f "$REP/$1_grey_reg.json" ] && [ -f "$REP/${B}_grey_reg.json" ] && { "$CPUPY" scripts/accept_gate.py \
    --pair black_reg "$REP/${B}_black_reg.json" "$REP/$1_black_reg.json" --pair grey_reg "$REP/${B}_grey_reg.json" \
    "$REP/$1_grey_reg.json" --out "$REP/gate_$1_reg.json" | tail -1 || true; }
  [ -f "$REP/$1_rad_grey.json" ] && [ -f "$REP/${B}_rad_grey.json" ] && { "$CPUPY" scripts/accept_gate.py \
    --pair black_rad "$REP/${B}_rad_black.json" "$REP/$1_rad_black.json" --pair grey_rad "$REP/${B}_rad_grey.json" \
    "$REP/$1_rad_grey.json" --out "$REP/gate_$1_rad.json" | tail -1 || true; }
  return 0
}

base_scores() {  # baseline arm: reuse its live renders, copy the scores here, add DISTS (lock: array tasks race)
  until mkdir "$REP/.base_lock" 2>/dev/null; do sleep 15; done
  for bg in black grey; do
    [ -f "$REP/${BASE}_$bg.json" ] || { [ -f "$VT/report/${BASE}_$bg.json" ] && cp "$VT/report/${BASE}_$bg.json" "$REP/"; } || true
  done
  [ -f "$VT/$BASE/splat.ply" ] && { score "$BASE" "$VT/$BASE/splat.ply" || true; }; rmdir "$REP/.base_lock"
}

distill_arm() {  # fm_ens[_jt<x>][_jr<deg>][_pf<frac>][_it<iters>]: scripts/methods/ensemble_distill.py over $TEACHERS (arm names)
  local arm="$1" R="$VT/$1" t D=(); IFS=_ read -ra T <<< "${1#fm_ens}"
  for t in "${T[@]}"; do case "$t" in
    "") ;; jt*) D+=(--jitter-t "${t#jt}") ;; jr*) D+=(--jitter-r "${t#jr}") ;; pf*) D+=(--pseudo-frac "${t#pf}") ;;
    it*) D+=(--iters "${t#it}") ;; lr*) D+=(--lr-mult "${t#lr}") ;; pmi) D+=(--pseudo-mode interp) ;;
    sdual) D+=(--sort dual) ;; srad) D+=(--sort radial) ;; t*) ;;  # t<label>: name only (e.g. which TEACHERS)
    *) echo "unknown token $t in $1"; return 1 ;; esac; done
  local TP=(); for t in ${TEACHERS:-igs4m_pc274 fm_seed1 fm_seed2}; do TP+=("$VT/$t/splat.ply"); done
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] distill: ${TP[*]} ${D[*]}"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-pc274" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-pc274" \
      "$SPY" scripts/methods/ensemble_distill.py --teachers "${TP[@]}" --dataset "$DS" --test-split-interval "${TSI:-8}" --eval-dataset "$TEST" \
      --out "$R" --viewer pc274 "${D[@]}" > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
  fi
  grep -E 'held-out' "$R/train.log" || true
  score "$arm" "$R/splat.ply"; [ "$arm" = "$BASE" ] || gate "$arm"; ft_arm "$arm" "$R/splat.ply"
}

ensv_arm() {  # fm_ensv[_pf<frac>][_it<n>][_jt<x>][_jr<deg>][_t<label>]: ensemble distillation inside the viewer-exact fine-tune
  # (scripts/methods/ftv_distill.py) of $TEACHERS' fine-tuned splats ($VT/<t>/splat_ft.ply, from FT_ITERS runs), student =
  # the first; a post-fine-tune product, so gated vs "$BASE~ft"
  local arm="$1" R="$VT/$1" t D=(); IFS=_ read -ra T <<< "${1#fm_ensv}"
  for t in "${T[@]}"; do case "$t" in
    "") ;; pf*) D+=(--pseudo-frac "${t#pf}") ;; it*) D+=(--iters "${t#it}") ;; jt*) D+=(--jitter-t "${t#jt}") ;;
    jr*) D+=(--jitter-r "${t#jr}") ;; t*) ;; *) echo "unknown token $t in $1"; return 1 ;; esac; done
  local TP=(); for t in ${TEACHERS:-igs4m_pc274 fm_seed1 fm_seed2}; do TP+=("$VT/$t/splat_ft.ply"); done
  mkdir -p "$R"
  [ -f "$R/splat.ply" ] || PATH="$(dirname "$SPY"):$PATH" "$SPY" scripts/methods/ftv_distill.py --teachers "${TP[@]}" --dataset "$DS" \
    --test-split-interval 8 --eval-dataset "$TEST" --out "$R/splat.ply" "${D[@]}" > "$R/train.log" 2>&1 || { tail -30 "$R/train.log"; return 1; }
  grep -E 'held-out' "$R/train.log" || true
  score "$arm" "$R/splat.ply"; gate "$arm" "$BASE~ft"
}

ensm_arm() {  # fm_ensm[_it<n>][_k<keep>][_pf<frac>][_t<label>]: splat-level ensemble (scripts/methods/ens_merge.py) of $TEACHERS'
  # fine-tuned splats ($VT/<t>/splat_ft.ply): merge (opacity 1-(1-a)^(1/n)) -> "<arm>~raw"; joint viewer-exact fine-tune toward the
  # seeds' render mean + real frames (ftv_distill.py --init, it<n>, default 2000) -> "<arm>~full"; prune to k (default: the mean
  # teacher count; <1 = fraction) by training-view blending weight + 1500-iteration viewer-exact fine-tune -> "<arm>" (splat.ply).
  # All gated vs "$BASE~ft"; held-out frames never used.
  local arm="$1" R="$VT/$1" t IT=2000 K="" PF=0.5; IFS=_ read -ra T <<< "${1#fm_ensm}"
  for t in "${T[@]}"; do case "$t" in
    "") ;; it*) IT="${t#it}" ;; k*) K="${t#k}" ;; pf*) PF="${t#pf}" ;; t*) ;; *) echo "unknown token $t in $1"; return 1 ;; esac; done
  local TP=(); for t in ${TEACHERS:-igs4m_pc274 fm_seed1 fm_seed2}; do TP+=("$VT/$t/splat_ft.ply"); done
  mkdir -p "$R"; local GP=(env PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-pc274" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-pc274")
  [ -f "$R/merged.ply" ] || "$SPY" scripts/methods/ens_merge.py merge --plys "${TP[@]}" --out "$R/merged.ply" | tail -1
  score "$arm~raw" "$R/merged.ply"; gate "$arm~raw" "$BASE~ft"
  [ -f "$R/full.ply" ] || "${GP[@]}" "$SPY" scripts/methods/ftv_distill.py --teachers "${TP[@]}" --init "$R/merged.ply" --dataset "$DS" \
    --test-split-interval 8 --eval-dataset "$TEST" --out "$R/full.ply" --iters "$IT" --pseudo-frac "$PF" > "$R/full.log" 2>&1 || { tail -30 "$R/full.log"; return 1; }
  grep -E 'held-out' "$R/full.log" || true
  score "$arm~full" "$R/full.ply"; gate "$arm~full" "$BASE~ft"
  [ -n "$K" ] || K=$("$CPUPY" -c "import sys; print(sum(int(open(f,'rb').read(4096).split(b'element vertex ')[1].split()[0]) for f in sys.argv[1:]) // (len(sys.argv) - 1))" "${TP[@]}")
  [ -f "$R/pruned.ply" ] || "${GP[@]}" "$SPY" scripts/methods/ens_merge.py prune --ply "$R/full.ply" --dataset "$DS" --test-split-interval 8 \
    --keep "$K" --out "$R/pruned.ply" | tail -1
  [ -f "$R/splat.ply" ] || "${GP[@]}" "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$R/pruned.ply" --out "$R/splat.ply" --dataset "$DS" \
    --test-split-interval 8 --viewer ss352 --display-scale 1.5:4 --iters 1500 --sort dual --lambda-wd 0.025 --wd-log2-sigma 4 \
    --eval-dataset "$TEST" > "$R/prune_ft.log" 2>&1 || { tail -30 "$R/prune_ft.log"; return 1; }
  ls -la "$R"/*.ply
  score "$arm" "$R/splat.ply"; gate "$arm" "$BASE~ft"
}

ft_arm() {  # <arm> <ply>: FT_ITERS > 0: the deliverable's viewer-exact fine-tune (finetune_ply_viewer.py, SuperSplat profile,
  # display scales 1.5:4, WD, --sort $FT_SORT) on the TRAINING frames only (every 8th held out) -> $VT/<arm>/splat_ft.ply, scored
  # as "<arm>~ft" and gated vs "$BASE~ft" (run the base arm in ARMS too). Removes most of the trainer-vs-viewer mismatch that
  # makes raw seeds scatter by +-0.4 dB in the live viewer, and is what the deliverable chain does next anyway.
  [ "${FT_ITERS:-0}" -gt 0 ] || return 0
  local R="$VT/$1"
  [ -f "$R/splat_ft.ply" ] || PATH="$(dirname "$SPY"):$PATH" "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$2" --out "$R/splat_ft.ply" \
    --dataset "$DS" --test-split-interval 8 --viewer ss352 --display-scale 1.5:4 --iters "$FT_ITERS" --sort "${FT_SORT:-dual}" \
    --lambda-wd 0.025 --wd-log2-sigma 4 --eval-dataset "$TEST" > "$R/ft.log" 2>&1 || { tail -30 "$R/ft.log"; return 1; }
  grep -E "eval \{" "$R/ft.log" | sed -n '1p;$p'
  score "$1~ft" "$R/splat_ft.ply"; [ "$1" = "$BASE" ] || gate "$1~ft" "$BASE~ft"
}

run_arm() {
  local arm="$1" R="$VT/$1"
  [[ "$arm" == fm_ensv* ]] && { ensv_arm "$arm"; return; }
  [[ "$arm" == fm_ensm* ]] && { ensm_arm "$arm"; return; }
  [[ "$arm" == fm_ens* ]] && { distill_arm "$arm"; return; }
  arm_args "$arm"
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] train: ${A[*]}"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
      "$SPY" scripts/train_viewer_gsplat.py --dataset "$DS" --out "$R" --test-split-interval "${TSI:-8}" --eval-dataset "$TEST" \
      --iters "$ITERS" --strategy improved --budget 4000000 --viewer "$PROF" "${A[@]}" > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    date
  fi
  grep -E '^\[(vgs\] (held-out|[0-9]+ gaussians)|methods\])' "$R/train.log" || true
  score "$arm" "$R/splat.ply"; [ "$arm" = "$BASE" ] || gate "$arm"; ft_arm "$arm" "$R/splat.ply"
}

[ -f "$REG/_register.json" ] || "$CPUPY" scripts/register_gt.py --dataset "$TEST" --out "$REG" --engine current | tail -1
if [ "$MODE" != report ]; then
  bash env/make_viewer_venv.sh | tail -1
  base_scores
  read -ra AL <<< "$ARMS"; [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && AL=("${AL[$SLURM_ARRAY_TASK_ID]}")
  rc=0; for arm in "${AL[@]}"; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done
  [ "$MODE" = arms ] && { echo "FM_BENCH_ARMS_DONE rc=$rc"; exit $rc; }
fi
for arm in $BASE $ARMS; do                       # (re)score what is missing (e.g. registered metrics added later) and re-gate
  [ -f "$VT/$arm/splat.ply" ] && [ -f "$VT/shots/onpath/$arm/_viewer.json" ] && { score "$arm" "$VT/$arm/splat.ply" > /dev/null || true; }
  [ "$arm" != "$BASE" ] && [ -f "$REP/${arm}_grey.json" ] && { gate "$arm" > /dev/null || true; }
done
"$CPUPY" - "$VT" "$REP" "$BASE" $ARMS <<'PY'
import json, os, sys
VT, REP, base, arms = sys.argv[1], sys.argv[2], sys.argv[3], list(dict.fromkeys(sys.argv[4:]))
ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
B = {bg: ld(f"{REP}/{base}_{bg}.json") for bg in ("black", "grey")}
L = ["| arm | black PSNR / SSIM / LPIPS / DISTS | dPSNR | dLPIPS | grey PSNR / SSIM / LPIPS / DISTS | registered black PSNR / SSIM / LPIPS / DISTS | dPSNR reg | gaussians | train min | gate | gate perc | gate reg |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
BR = ld(f"{REP}/{base}_black_reg.json")
for a in [base] + [a for a in arms if a != base]:
    r = {bg: ld(f"{REP}/{a}_{bg}.json") for bg in ("black", "grey")}
    if not r["black"]: continue
    t = ld(f"{VT}/{a}/train.json") or {}; g, gp, gr = ld(f"{REP}/gate_{a}.json"), ld(f"{REP}/gate_{a}_perc.json"), ld(f"{REP}/gate_{a}_reg.json")
    rr = ld(f"{REP}/{a}_black_reg.json")
    f = lambda x: " / ".join("-" if x.get(k) is None else str(x.get(k)) for k in ("psnr", "ssim", "lpips", "dists")) if x else "-"
    d = lambda k: f"{r['black'][k] - B['black'][k]:+.3f}" if B["black"] and r["black"].get(k) is not None else "-"
    v = lambda x: "-" if x is None else ("ACCEPT" if x["accept"] else "reject")
    dr = f"{rr['psnr'] - BR['psnr']:+.3f}" if rr and BR else "-"
    L.append(f"| {a} | {f(r['black'])} | {d('psnr')} | {d('lpips')} | {f(r['grey'])} | {f(rr)} | {dr} | {t.get('n_gaussians', '-')} | "
             f"{round(t['train_seconds'] / 60, 1) if t.get('train_seconds') else '-'} | {v(g)} | {v(gp)} | {v(gr)} |")
open(f"{REP}/summary_fm.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
echo FM_BENCH_DONE
