#!/bin/bash
# Viewer-matched training benchmark: splats trained FROM SCRATCH in a gsplat rasteriser matched to SuperSplat
# (scripts/train_viewer_gsplat.py; patched kernels from env/make_gsplat_viewer_venv.sh), EDGS init ($P/edgs/on/data,
# every 8th frame held out = $P/onpath/test), scored by THE metric: the exported PLY rendered by SuperSplat itself
# (scripts/viewer_shots.py --engine $ENGINE, default `current` = the live v3.5.2 editor on headless WebGPU; `pc274` =
# PlayCanvas 2.7.4) at $P/onpath/test on each background of $BGS, scored with eval_views.py --pred-dir, and gated
# (accept_gate.py, every background a separate set) vs the current best, EDGS + Improved-GS 4M trained in 3DGRUT
# (PLY $P/viewer_bench/edgs_igs4m.ply, re-rendered here as arm `base` with the same engine). Off-path renders:
# viewer_shots --move $MOVE (camera right/down/back) for base and each arm (black background).
#   ARMS="igs4m_ss352 mcmc4m_ss352" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-1 \
#       sbatch/viewer_train_bench.sbatch configs/<seq>.yaml     # one arm per array task (ARMS[task id])
#   MODE=report ARMS="..." sbatch -A ... -d afterany:<array> sbatch/viewer_train_bench.sbatch configs/<seq>.yaml
#   ITERS=3000 VT_OUT=$P/viewer_train_smoke ARMS=... sbatch ...                                       # smoke
# Arm names: <method>[_flag...]; method igs<N>m (Improved-GS, budget N M) | mcmc<N>m (gsplat MCMC, cap N M) |
#   default (gsplat DefaultStrategy, AbsGS) | base (no training: the baseline PLY);
#   flags: ss352 (default) / pc274 (viewer profile + kernel build: SuperSplat 3.5.2 / PlayCanvas 2.7.4), aa
#   (antialiased mode, PLY tagged `comment SplatRenderMode mip`), ss<k> (render k x, area-downsample, loss at 1x),
#   mp<x> (3D scale floor x px), e<x> (dilation x px^2 instead of the profile's), zero / pcbug / std (SH band-2 handling), d (LiDAR depth loss, LAMBDA), noeas, nomu,
#   appgain / appaffine (per-frame exposure / white-balance transform on the render in training, --appearance), appr<x> (its
#   identity pull, --app-reg x), ent[x] / bin[x] (late opacity entropy / o(1-o) penalty, --opa-reg, weight x), opp[x] (prune
#   opacity < x, default 0.05, after the ramp), cp[x] (prune gaussians with < x px (default 1) blending weight over all
#   training views). GATE_REF=<arm> gates against that arm's viewer scores instead of `base`.
# Every scored PLY is also scored with a per-frame least-squares colour affine fitted on each held-out frame itself
#   (shots/onpath_fit/, report/<name>_<bg>_fit.json; leaks 12 numbers per frame, so informative only -- the gate uses raw).
# Outputs under $VT (default $P/viewer_train): <arm>/{splat.ply,train.json,train.log,renders_trainer/},
#   shots/{onpath,offpath}/<arm>/, report/<arm>_<bg>.json (viewer metric), report/gate_<arm>.json,
#   report/summary.{json,md}, montage_{onpath,offpath}.jpg.
#SBATCH -J viewer-train
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
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
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VT="${VT_OUT:-$P/viewer_train}"; REP="$VT/report"
DS="${VT_DATASET:-$P/edgs/on/data}"; TEST="$P/onpath/test"; TSI="${TSI:-8}"; BASE_PLY="${BASE_PLY:-$P/viewer_bench/edgs_igs4m.ply}"
ITERS="${ITERS:-30000}"; LAMBDA="${LAMBDA:-0.05}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-8}"
ENGINE="${ENGINE:-current}"; BGS="${BGS:-black grey}"; read -ra BGA <<< "$BGS"
ARMS="${ARMS:-igs4m_ss352 igs4m_ss352_aa igs4m_ss352_ss2_mp0.5 mcmc4m_ss352 igs4m_pc274}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "$DS/sparse/0/points3D.txt" ] || { echo "no EDGS dataset at $DS (sbatch/edgs_bench.sbatch builds it)"; exit 1; }
lock() { until mkdir "$1" 2>/dev/null; do sleep 20; done; }
lock "$CEAR_WS/.gsplat_viewer_lock"; trap 'rmdir "$CEAR_WS/.gsplat_viewer_lock" 2>/dev/null || true' EXIT
bash env/make_gsplat_viewer_venv.sh | grep -E 'gsplat-viewer\]|DONE'; bash env/make_viewer_venv.sh | tail -1
rmdir "$CEAR_WS/.gsplat_viewer_lock"; trap - EXIT

arm_args() {  # <arm> -> A (train_viewer_gsplat.py flags), PROF (kernel build)
  local t; IFS=_ read -ra T <<< "$1"; A=(); PROF=ss352
  case "${T[0]}" in
    igs*m)  A=(--strategy improved --budget "$(( ${T[0]//[!0-9]/} * 1000000 ))") ;;
    mcmc*m) A=(--strategy mcmc --budget "$(( ${T[0]//[!0-9]/} * 1000000 ))") ;;
    default) A=(--strategy default) ;;
    *) echo "unknown arm method in $1"; return 1 ;;
  esac
  for t in "${T[@]:1}"; do
    case "$t" in
      pc274|ss352) PROF="$t" ;;  aa) A+=(--aa) ;;  zero|pcbug|std) A+=(--sh-band2 "$t") ;;
      ss[0-9]*) A+=(--ss "${t#ss}") ;;      mp*) A+=(--min-px "${t#mp}") ;;
      e[0-9]*) A+=(--eps2d "${t#e}") ;;     # screen dilation override (px^2)
      d) A+=(--depth-lambda "$LAMBDA") ;;   noeas) A+=(--no-eas) ;;  nomu) A+=(--no-mu) ;;
      appgain|appaffine) A+=(--appearance "${t#app}") ;;  appr*) A+=(--app-reg "${t#appr}") ;;
      ent|ent[0-9]*) A+=(--opa-reg entropy); [ "$t" = ent ] || A+=(--opa-lambda "${t#ent}") ;;
      bin|bin[0-9]*) A+=(--opa-reg binary); [ "$t" = bin ] || A+=(--opa-lambda "${t#bin}") ;;
      opp|opp[0-9]*) A+=(--opa-prune "$([ "$t" = opp ] && echo 0.05 || echo "${t#opp}")") ;;
      cp|cp[0-9]*) A+=(--contrib-prune "$([ "$t" = cp ] && echo 1 || echo "${t#cp}")") ;;
      *) echo "unknown arm flag $t in $1"; return 1 ;;
    esac
  done
  A+=(--viewer "$PROF")
}

fit_dir() {  # <pred dir> <out dir>: each render with the least-squares colour affine (3x3 + bias) to its own real frame
  "$CPUPY" - "$1" "$TEST/images" "$2" <<'PY'
import cv2, json, os, sys, numpy as np
pd, gd, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True); fits = {}
for n in sorted(x for x in os.listdir(pd) if x.endswith(".png") and not x.startswith("_")):
    p, g = cv2.imread(os.path.join(pd, n))[..., ::-1] / 255.0, cv2.imread(os.path.join(gd, n))[..., ::-1] / 255.0
    X = np.c_[p.reshape(-1, 3), np.ones(p.shape[0] * p.shape[1])]
    S = np.linalg.lstsq(X, g.reshape(-1, 3), rcond=None)[0]                      # (4,3): [M^T; b]
    cv2.imwrite(os.path.join(od, n), np.round((X @ S).reshape(p.shape).clip(0, 1) * 255).astype(np.uint8)[..., ::-1])
    fits[n] = {"M": np.round(S[:3].T, 5).tolist(), "b": np.round(S[3], 5).tolist()}
G = np.array([np.diagonal(f["M"]) for f in fits.values()]); lum = G @ [0.299, 0.587, 0.114]
json.dump({"gain_pct_0_5_50_95_100": np.round(np.percentile(lum, [0, 5, 50, 95, 100]), 4).tolist(),
           "gain_std_rgb": np.round(G.std(0), 4).tolist(), "frames": fits}, open(os.path.join(od, "_fit.json"), "w"), indent=1)
print(f"fit {len(fits)} frames -> {od}: luminance gain pct 0/5/50/95/100 {np.round(np.percentile(lum, [0, 5, 50, 95, 100]), 3).tolist()}")
PY
}

score_ply() {  # <name> <ply>: SuperSplat renders at the held-out cameras -> report/<name>_<bg>.json (+ _fit); off-path renders
  local n="$1" ply="$2" bg d
  if [ ! -f "$REP/${n}_${BGA[-1]}.json" ]; then
    "$VPY" scripts/viewer_shots.py --engine "$ENGINE" --bg $BGS --ply "$ply" --dataset "$TEST" --out-dir "$VT/shots/onpath/$n" | tail -2
    for bg in "${BGA[@]}"; do
      d="$VT/shots/onpath/$n"; [ "${#BGA[@]}" -gt 1 ] && d="$d/$bg"
      "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$TEST" --tag "${n}_$bg" --json-out "$REP/${n}_$bg.json" | tail -1
    done
  fi
  for bg in "${BGA[@]}"; do
    [ -f "$REP/${n}_${bg}_fit.json" ] && continue
    d="$VT/shots/onpath/$n"; [ "${#BGA[@]}" -gt 1 ] && d="$d/$bg"
    fit_dir "$d" "$VT/shots/onpath_fit/$n/$bg" && \
      "$GPY" scripts/eval_views.py --pred-dir "$VT/shots/onpath_fit/$n/$bg" --dataset "$TEST" --tag "${n}_${bg}_fit" \
        --json-out "$REP/${n}_${bg}_fit.json" | tail -1
  done
  [ -f "$VT/shots/offpath/$n/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine "$ENGINE" --bg black --ply "$ply" \
    --dataset "$TEST" --every "$EVERY_OFF" --move $MOVE --out-dir "$VT/shots/offpath/$n" | tail -1
}

run_arm() {  # <arm>
  local arm="$1" R="$VT/$1" bg PAIRS=() ref="${GATE_REF:-base}"
  lock "$VT/.base_lock"; score_ply base "$BASE_PLY" || true; rmdir "$VT/.base_lock"
  [ "$arm" = base ] && return 0
  arm_args "$arm"
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R"; echo "== [$arm] train: ${A[*]}"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
      "$SPY" scripts/train_viewer_gsplat.py --dataset "$DS" --out "$R" --test-split-interval "$TSI" --eval-dataset "$TEST" \
      --iters "$ITERS" "${A[@]}" > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    grep -E '^\[vgs\] (held-out|appearance|[0-9]+ gaussians)' "$R/train.log" || true; date
  fi
  score_ply "$arm" "$R/splat.ply"
  [ "$arm" = "$ref" ] && return 0
  [ -f "$REP/${ref}_${BGA[-1]}.json" ] || { echo "[$arm] gate ref $ref not scored yet (run it first; MODE=report gates)"; return 0; }
  for bg in "${BGA[@]}"; do PAIRS+=(--pair "onpath_$bg" "$REP/${ref}_$bg.json" "$REP/${arm}_$bg.json"); done
  "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate_$arm.json" | tail -$(( 3 * ${#BGA[@]} + 1 )) || true
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra AL <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then AL=("${AL[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${AL[@]}"; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done
  [ "$MODE" = arms ] && { echo "VIEWER_TRAIN_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report ($ENGINE engine, backgrounds: $BGS)"
score_ply base "$BASE_PLY"
REF="${GATE_REF:-base}"
for arm in $ARMS; do                                        # (re)gate every scored arm against GATE_REF
  [ "$arm" = "$REF" ] || [ ! -f "$REP/${arm}_${BGA[-1]}.json" ] || [ ! -f "$REP/${REF}_${BGA[-1]}.json" ] && continue
  PAIRS=(); for bg in "${BGA[@]}"; do PAIRS+=(--pair "onpath_$bg" "$REP/${REF}_$bg.json" "$REP/${arm}_$bg.json"); done
  "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate_$arm.json" > /dev/null || true
done
"$CPUPY" - "$VT" "$TEST/images" "$BGS" base $ARMS <<'PY'
import cv2, json, os, sys, numpy as np
VT, gt, bgs, tags = sys.argv[1], sys.argv[2], sys.argv[3].split(), list(dict.fromkeys(sys.argv[4:]))
rep = os.path.join(VT, "report"); ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
sub = lambda t, bg: os.path.join(VT, "shots", "onpath", t, *([bg] if len(bgs) > 1 else []))
rows = []
for t in tags:
    r = {bg: ld(os.path.join(rep, f"{t}_{bg}.json")) for bg in bgs}
    if not r[bgs[0]]: continue
    g, tj = ld(os.path.join(rep, f"gate_{t}.json")), ld(os.path.join(VT, t, "train.json"))
    agree = []                                              # trainer-raster render vs SuperSplat's, same camera
    td = os.path.join(VT, t, "renders_trainer")
    for n in (sorted(os.listdir(td))[::4] if os.path.isdir(td) else []):
        a_, b_ = cv2.imread(os.path.join(td, n)), cv2.imread(os.path.join(sub(t, bgs[0]), n))
        if a_ is not None and b_ is not None:
            agree.append(-10 * np.log10(max(np.mean((a_.astype(float) - b_) ** 2) / 255 ** 2, 1e-12)))
    row = {"arm": t, "trainer_psnr": tj.get("heldout_psnr") if tj else None,
           "trainer_vs_viewer": round(float(np.mean(agree)), 2) if agree else None,
           "gaussians": tj.get("n_gaussians") if tj else None, "train_min": round(tj["train_seconds"] / 60, 1) if tj else None,
           "gate": None if g is None else ("ACCEPT" if g["accept"] else "reject")}
    for bg in bgs:
        if r[bg]: row.update({f"{k}_{bg}": r[bg][k] for k in ("psnr", "ssim", "lpips")})
    fj = ld(os.path.join(rep, f"{t}_{bgs[0]}_fit.json"))      # per-frame colour affine fitted on the frame (informative)
    if fj: row.update({f"{k}_{bgs[0]}_fit": fj[k] for k in ("psnr", "ssim", "lpips")})
    if tj and tj.get("appearance"): row["app_gain_lum_pct"] = tj["appearance"]["gain_pct_0_5_25_50_75_95_100"]["lum"]
    rows.append(row)
json.dump(rows, open(os.path.join(rep, "summary.json"), "w"), indent=2)
f = lambda v, p: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
hdr = "| arm | " + " | ".join(f"{bg} PSNR / SSIM / LPIPS" for bg in bgs) + f" | {bgs[0]} colour-fitted | trainer PSNR | trainer vs viewer | gaussians | train min | gate vs base |"
L = [hdr, "|" + "---|" * (len(bgs) + 7)]
for r in rows:
    L.append(f"| {r['arm']} | " + " | ".join(f"{f(r.get('psnr_'+bg),3)} / {f(r.get('ssim_'+bg),4)} / {f(r.get('lpips_'+bg),4)}" for bg in bgs)
             + f" | {f(r.get('psnr_'+bgs[0]+'_fit'),3)} / {f(r.get('ssim_'+bgs[0]+'_fit'),4)} / {f(r.get('lpips_'+bgs[0]+'_fit'),4)}"
             + f" | {f(r['trainer_psnr'],3)} | {f(r['trainer_vs_viewer'],2)} | {f(r['gaussians'],0)} | {f(r['train_min'],1)} | {r['gate'] or '-'} |")
open(os.path.join(rep, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def montage(kind, first_gt, out, k):
    d0 = sub("base", bgs[0]) if kind == "onpath" else os.path.join(VT, "shots", "offpath", "base")
    if not os.path.isdir(d0): return
    names = sorted(n for n in os.listdir(d0) if n.endswith(".png") and not n.startswith("_"))
    rows_ = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, k).astype(int)]:
        ims = [(cv2.imread(os.path.join(gt, n)), "real")] if first_gt else []
        ims += [(cv2.imread(os.path.join(sub(t, bgs[0]) if kind == "onpath" else os.path.join(VT, "shots", "offpath", t), n)), t) for t in tags]
        ims = [(cv2.resize(i, (400, 300)), l) for i, l in ims if i is not None]
        for i, l in ims: cv2.putText(i, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows_.append(np.hstack([i for i, _ in ims]))
    w = max(r.shape[1] for r in rows_); rows_ = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows_]
    cv2.imwrite(out, np.vstack(rows_)); print("montage ->", out)
montage("onpath", True, os.path.join(VT, "montage_onpath.jpg"), 4)
montage("offpath", False, os.path.join(VT, "montage_offpath.jpg"), 4)
PY
echo VIEWER_TRAIN_BENCH_DONE
