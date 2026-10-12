#!/bin/bash
# Viewer-matched training benchmark: splats trained FROM SCRATCH in a gsplat rasteriser matched to SuperSplat's
# engine (scripts/train_viewer_gsplat.py; patched kernels from env/make_gsplat_viewer_venv.sh), EDGS init
# ($P/edgs/on/data, every 8th frame held out = $P/onpath/test), scored by THE metric: the exported PLY rendered by
# SuperSplat's engine (scripts/viewer_shots.py, PlayCanvas 2.7.4 in headless Chromium) at $P/onpath/test and
# scored with eval_views.py --pred-dir, gated (accept_gate.py) vs the current best, EDGS + Improved-GS 4M in
# 3DGRUT ($P/supersplat_check/report/edgs_igs4m.json, PLY $P/viewer_bench/edgs_igs4m.ply). Off-path montages:
# viewer_shots --move $MOVE (camera right/down/back) for the baseline and each arm.
#   ARMS="igs4m_pc274 mcmc4m_pc274" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-1 \
#       sbatch/viewer_train_bench.sbatch configs/<seq>.yaml     # one arm per array task (ARMS[task id])
#   MODE=report ARMS="..." sbatch -A ... -d afterany:<array> sbatch/viewer_train_bench.sbatch configs/<seq>.yaml
#   ITERS=3000 VT_OUT=$P/viewer_train_smoke ARMS=... sbatch ...                                       # smoke
# Arm names: <method>[_flag...]; method igs<N>m (Improved-GS, budget N M) | mcmc<N>m (gsplat MCMC, cap N M) |
#   default (gsplat DefaultStrategy, AbsGS) | base (no training: re-render the baseline PLY = metric repeatability);
#   flags: pc274 / ss352 (viewer profile: eps2d 0.075 / 0.3 px^2, cull 0.125 / 0.5 px^2), aa (antialiased mode,
#   PLY tagged `comment SplatRenderMode mip`), ss<k> (render k x, area-downsample, loss at 1x), mp<x> (3D scale
#   floor x px), pcbug / std (SH band-2 handling; default zero), d (LiDAR depth loss, LAMBDA), noeas, nomu.
# Outputs under $VT (default $P/viewer_train): <arm>/{splat.ply,train.json,train.log}, shots/{onpath,offpath}/<arm>/,
#   report/<arm>.json (viewer metric), report/gate_<arm>.json, report/summary.{json,md}, montage_{onpath,offpath}.jpg.
#SBATCH -J viewer-train
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
export GSPLAT_VIEWER_DIR="${GSPLAT_VIEWER_DIR:-$CEAR_WS/gsplat-viewer}" TORCH_EXTENSIONS_DIR="${GSPLAT_VIEWER_EXT:-$CEAR_WS/torch-ext-gsplat-viewer}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VT="${VT_OUT:-$P/viewer_train}"; REP="$VT/report"
DS="${VT_DATASET:-$P/edgs/on/data}"; TEST="$P/onpath/test"; TSI="${TSI:-8}"
BASE_JSON="${BASE_JSON:-$P/supersplat_check/report/edgs_igs4m.json}"; BASE_PLY="${BASE_PLY:-$P/viewer_bench/edgs_igs4m.ply}"
ITERS="${ITERS:-30000}"; LAMBDA="${LAMBDA:-0.05}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-8}"
ARMS="${ARMS:-igs4m_pc274 igs4m_ss352 igs4m_pc274_aa igs4m_pc274_ss2_mp0.3 mcmc4m_pc274}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "$DS/sparse/0/points3D.txt" ] || { echo "no EDGS dataset at $DS (sbatch/edgs_bench.sbatch builds it)"; exit 1; }
until mkdir "$CEAR_WS/.gsplat_viewer_lock" 2>/dev/null; do sleep 20; done     # one JIT build at a time
trap 'rmdir "$CEAR_WS/.gsplat_viewer_lock" 2>/dev/null || true' EXIT
bash env/make_gsplat_viewer_venv.sh | tail -3; bash env/make_viewer_venv.sh | tail -1
rmdir "$CEAR_WS/.gsplat_viewer_lock"; trap - EXIT

arm_args() {  # <arm> -> A (train_viewer_gsplat.py flags)
  local t; IFS=_ read -ra T <<< "$1"; A=()
  case "${T[0]}" in
    igs*m)  A=(--strategy improved --budget "$(( ${T[0]//[!0-9]/} * 1000000 ))") ;;
    mcmc*m) A=(--strategy mcmc --budget "$(( ${T[0]//[!0-9]/} * 1000000 ))") ;;
    default) A=(--strategy default) ;;
    *) echo "unknown arm method in $1"; return 1 ;;
  esac
  for t in "${T[@]:1}"; do
    case "$t" in
      pc274|ss352) A+=(--viewer "$t") ;;  aa) A+=(--aa) ;;  pcbug|std) A+=(--sh-band2 "$t") ;;
      ss[0-9]*) A+=(--ss "${t#ss}") ;;      mp*) A+=(--min-px "${t#mp}") ;;
      d) A+=(--depth-lambda "$LAMBDA") ;;   noeas) A+=(--no-eas) ;;  nomu) A+=(--no-mu) ;;
      *) echo "unknown arm flag $t in $1"; return 1 ;;
    esac
  done
}

score_ply() {  # <name> <ply>: viewer renders at the held-out cameras -> metric json, off-path renders
  local n="$1" ply="$2"
  [ -f "$REP/$n.json" ] || {
    "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$TEST" --out-dir "$VT/shots/onpath/$n" | tail -2
    "$GPY" scripts/eval_views.py --pred-dir "$VT/shots/onpath/$n" --dataset "$TEST" --tag "$n" --json-out "$REP/$n.json" | tail -1; }
  [ -f "$VT/shots/offpath/$n/_viewer.json" ] || \
    "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$TEST" --every "$EVERY_OFF" --move $MOVE --out-dir "$VT/shots/offpath/$n" | tail -1
}

run_arm() {  # <arm>
  local arm="$1" R="$VT/$1"
  if [ "$arm" = base ]; then score_ply base "$BASE_PLY"; return 0; fi
  arm_args "$arm"
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] train: ${A[*]}"; date
    PYTHONPATH="$GSPLAT_VIEWER_DIR" "$SPY" scripts/train_viewer_gsplat.py --dataset "$DS" --out "$R" \
      --test-split-interval "$TSI" --eval-dataset "$TEST" --iters "$ITERS" "${A[@]}" > "$R/train.log" 2>&1 \
      || { tail -40 "$R/train.log"; return 1; }
    grep -E '^\[vgs\] (held-out|[0-9]+ gaussians)' "$R/train.log" || true; date
  fi
  score_ply "$arm" "$R/splat.ply"
  "$CPUPY" scripts/accept_gate.py --pair onpath_viewer "$BASE_JSON" "$REP/$arm.json" --out "$REP/gate_$arm.json" | tail -4 || true
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra AA <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then AA=("${AA[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${AA[@]}"; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done
  [ "$MODE" = arms ] && { echo "VIEWER_TRAIN_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report"
score_ply base "$BASE_PLY"
"$CPUPY" - "$VT" "$BASE_JSON" "$TEST/images" base $ARMS <<'PY'
import cv2, json, os, sys, numpy as np
VT, base_json, gt, tags = sys.argv[1], sys.argv[2], sys.argv[3], list(dict.fromkeys(sys.argv[4:]))
rep = os.path.join(VT, "report"); ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
b = ld(base_json); rows = [{"arm": "baseline (edgs_igs4m, 3DGRUT)", "psnr": b["psnr"], "ssim": b["ssim"], "lpips": b["lpips"]}]
for t in tags:
    r, g, tj = ld(os.path.join(rep, t + ".json")), ld(os.path.join(rep, f"gate_{t}.json")), ld(os.path.join(VT, t, "train.json"))
    if not r: continue
    rows.append({"arm": t, "psnr": r["psnr"], "ssim": r["ssim"], "lpips": r["lpips"],
                 "trainer_psnr": tj.get("heldout_psnr") if tj else None, "gaussians": tj.get("n_gaussians") if tj else None,
                 "train_min": round(tj["train_seconds"] / 60, 1) if tj else None,
                 "gate": None if g is None else ("ACCEPT" if g["accept"] else "reject")})
json.dump(rows, open(os.path.join(rep, "summary.json"), "w"), indent=2)
f = lambda v, p: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
L = ["| arm | viewer PSNR | SSIM | LPIPS | trainer-raster PSNR | gaussians | train min | gate vs baseline |", "|" + "---|" * 8]
L += [f"| {r['arm']} | {f(r['psnr'],3)} | {f(r['ssim'],4)} | {f(r['lpips'],4)} | {f(r.get('trainer_psnr'),3)} | "
      f"{f(r.get('gaussians'),0)} | {f(r.get('train_min'),1)} | {r.get('gate') or '-'} |" for r in rows]
open(os.path.join(rep, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def montage(sub, first_gt, out, k):
    d0 = os.path.join(VT, "shots", sub, "base")
    if not os.path.isdir(d0): return
    names = sorted(n for n in os.listdir(d0) if n.endswith(".png") and not n.startswith("_"))
    rows = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, k).astype(int)]:
        ims = [(cv2.imread(os.path.join(gt, n)), "real")] if first_gt else []
        ims += [(cv2.imread(os.path.join(VT, "shots", sub, t, n)), t) for t in tags]
        ims = [(cv2.resize(i, (400, 300)), l) for i, l in ims if i is not None]
        for i, l in ims: cv2.putText(i, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack([i for i, _ in ims]))
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
montage("onpath", True, os.path.join(VT, "montage_onpath.jpg"), 4)
montage("offpath", False, os.path.join(VT, "montage_offpath.jpg"), 4)
PY
echo VIEWER_TRAIN_BENCH_DONE
