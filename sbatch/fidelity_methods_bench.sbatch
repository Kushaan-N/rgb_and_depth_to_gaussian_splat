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
#   shn<lambda> (--shn-reg) | c2f<a> (--c2f) | emul / eerr (--eas-error mul / err) | aniso<lambda> (--aniso-reg, ratio 10) | ssl<x> (--ssim-lambda) | srad / sdual (--sort radial / dual) | ms<lambda> (--msssim-lambda) | blur<K> (--blur-kernel) | blr<x> (--blur-lr) | gt<threshold> (--grad-threshold) | bilag (--appearance bilagrid) | ss352 (viewer profile; default pc274).
# fm_ens[_jt<x>][_jr<deg>][_pf<frac>][_it<n>][_lr<x>][_t<label>]: multi-seed ensemble distillation (scripts/methods/ensemble_distill.py)
#   of the arms in $TEACHERS (default "igs4m_pc274 fm_seed1 fm_seed2", which must exist) into one splat.
# Outputs: $VT/<arm>/{splat.ply,train.json,train.log,methods.json}, $VT/shots/{onpath,offpath}/<arm>/,
#   $VT/report_fm/<arm>_<bg>.json, gate_<arm>{,_perc,_reg}.json, summary_fm.md. Also "registered" scores <arm>_<bg>_reg.json:
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
      ss352|pc274) PROF="$t" ;;
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
gate() {  # <arm>: strict + perceptual gate vs the baseline
  local pr=(--pair black "$REP/${BASE}_black.json" "$REP/$1_black.json" --pair grey "$REP/${BASE}_grey.json" "$REP/$1_grey.json")
  "$CPUPY" scripts/accept_gate.py "${pr[@]}" --out "$REP/gate_$1.json" | tail -1 || true
  "$CPUPY" scripts/accept_gate.py "${pr[@]}" --perceptual --out "$REP/gate_$1_perc.json" | tail -1 || true
  [ -f "$REP/$1_grey_reg.json" ] && [ -f "$REP/${BASE}_grey_reg.json" ] && { "$CPUPY" scripts/accept_gate.py \
    --pair black_reg "$REP/${BASE}_black_reg.json" "$REP/$1_black_reg.json" --pair grey_reg "$REP/${BASE}_grey_reg.json" \
    "$REP/$1_grey_reg.json" --out "$REP/gate_$1_reg.json" | tail -1 || true; }
  [ -f "$REP/$1_rad_grey.json" ] && [ -f "$REP/${BASE}_rad_grey.json" ] && { "$CPUPY" scripts/accept_gate.py \
    --pair black_rad "$REP/${BASE}_rad_black.json" "$REP/$1_rad_black.json" --pair grey_rad "$REP/${BASE}_rad_grey.json" \
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
    it*) D+=(--iters "${t#it}") ;; lr*) D+=(--lr-mult "${t#lr}") ;; t*) ;;  # t<label>: name only (e.g. which TEACHERS)
    *) echo "unknown token $t in $1"; return 1 ;; esac; done
  local TP=(); for t in ${TEACHERS:-igs4m_pc274 fm_seed1 fm_seed2}; do TP+=("$VT/$t/splat.ply"); done
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] distill: ${TP[*]} ${D[*]}"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-pc274" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-pc274" \
      "$SPY" scripts/methods/ensemble_distill.py --teachers "${TP[@]}" --dataset "$DS" --test-split-interval 8 --eval-dataset "$TEST" \
      --out "$R" --viewer pc274 "${D[@]}" > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
  fi
  grep -E 'held-out' "$R/train.log" || true
  score "$arm" "$R/splat.ply"; gate "$arm"
}

run_arm() {
  local arm="$1" R="$VT/$1"
  [[ "$arm" == fm_ens* ]] && { distill_arm "$arm"; return; }
  arm_args "$arm"
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] train: ${A[*]}"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
      "$SPY" scripts/train_viewer_gsplat.py --dataset "$DS" --out "$R" --test-split-interval 8 --eval-dataset "$TEST" \
      --iters "$ITERS" --strategy improved --budget 4000000 --viewer "$PROF" "${A[@]}" > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    date
  fi
  grep -E '^\[(vgs\] (held-out|[0-9]+ gaussians)|methods\])' "$R/train.log" || true
  score "$arm" "$R/splat.ply"; gate "$arm"
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
