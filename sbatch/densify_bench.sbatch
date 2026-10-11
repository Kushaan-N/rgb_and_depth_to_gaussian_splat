#!/bin/bash
# Densification benchmark (fidelity of seen regions): MCMC + LiDAR depth loss at several gaussian caps, with
# and without the mcmc_reg opacity/scale regularisation, and the Improved-GS densification port
# (patches/3dgrut-improved-densify.patch, worktree from env/make_3dgrut_densify.sh). Every arm is trained on
# the on-path split (every 8th frame held out) and the off-path split (1 m spatial hold-out), scored on real
# held-out images (eval_views.py), checked for floor coverage (floor_coverage.py) and gated (accept_gate.py)
# vs the stock baseline of that split (on-path arm_A, off-path depth_baseline) and, for depth arms, also vs
# the depth-loss-only baseline (arm_B / depth_l0.05).
#   ARMS="mcmc1m_d igs1m_d" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-1%4 \
#       sbatch/densify_bench.sbatch configs/<seq>.yaml          # one arm per array task (ARMS[task id])
#   MODE=report sbatch -A ... -d afterany:<array job> sbatch/densify_bench.sbatch configs/<seq>.yaml
#   sbatch -A ... sbatch/densify_bench.sbatch configs/<seq>.yaml   # no array: all arms, then the report
# Arm names: <method><cap>m[_flag...]   method mcmc (3DGRUT MCMC app, cap = strategy.add.max_n_gaussians) |
#   igs (Improved-GS, cap = strategy.budget) | gs (stock default, no cap); flags: d = LiDAR depth loss (λ=$LAMBDA;
#   skipped where the split has no *_depth.npy), reg = configs/variants/mcmc_reg.yaml trainer overrides,
#   noeas / nolas / norap / nomu = Improved-GS component ablations.
# Outputs: <out>/pipeline/densify/{onpath,offpath}/<arm>/ (runs, logs, splat.ply), densify/renders/<split>_<arm>/,
#   densify/report/<split>_<arm>.json, floor_<split>_<arm>.json, gate_<split>_<arm>[_vsdepth].json,
#   densify_summary.{json,md}, preview_<split>.jpg (real | baseline | best arm).
#SBATCH -J densify-bench
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"; LAMBDA="${LAMBDA:-0.05}"
# the densify worktree's kernels differ from the stock build: private extension cache, and eval/export import
# the same checkout so only one build exists per job
export TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}"
export PYTHONPATH="$DDIR${PYTHONPATH:+:$PYTHONPATH}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ON="$P/onpath"; O="$P/offpath"; D="${DENSIFY_OUT:-$P/densify}"; REP="$D/report"
ITERS="${ITERS:-$(get pipeline.train_iterations 30000)}"; GT="$OUT/colmap_train/sparse/0"   # ITERS/DENSIFY_OUT: smoke tests
ARMS="${ARMS:-mcmc1m_d mcmc1m_reg_d mcmc2m_d mcmc2m_reg_d mcmc3m_d mcmc3m_reg_d igs1m igs1m_d igs2m_d igs3m_d}"
SPLITS="${SPLITS:-onpath offpath}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
mkdir -p "$REP" "$TORCH_EXTENSIONS_DIR"; bash env/make_3dgrut_densify.sh

split_vars() {  # <split> -> DATA TSI TEST BASE (stock baseline json) BASED (depth-loss baseline json)
  case "$1" in
    onpath)  DATA="$ON/data_orig";    TSI=8;      TEST="$ON/test"; BASE="$ON/report/arm_A.json";        BASED="$ON/report/arm_B.json" ;;
    offpath) DATA="$O/train_data";    TSI=100000; TEST="$O/test";  BASE="$O/report/depth_baseline.json"; BASED="$O/report/depth_l0.05.json" ;;
    *) echo "unknown split $1"; return 1 ;;
  esac
}
arm_cfg() {  # <arm> -> APP OVR DEPTH
  local t n; IFS=_ read -ra T <<< "$1"; OVR=""; DEPTH=false
  case "${T[0]}" in
    mcmc*m) n="${T[0]#mcmc}"; APP=apps/colmap_3dgut_mcmc.yaml;     OVR="strategy.add.max_n_gaussians=$(( ${n%m} * 1000000 ))" ;;
    igs*m)  n="${T[0]#igs}";  APP=apps/colmap_3dgut_improved.yaml; OVR="strategy.budget=$(( ${n%m} * 1000000 ))" ;;
    gs)     APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)" ;;
    *) echo "unknown arm method in $1"; return 1 ;;
  esac
  for t in "${T[@]:1}"; do
    case "$t" in
      d) DEPTH=true ;;
      reg) OVR="$OVR $("$CPUPY" scripts/cfg_get.py configs/variants/mcmc_reg.yaml pipeline.trainer_overrides)" ;;
      noeas) OVR="$OVR strategy.eas.enabled=false" ;;  nolas) OVR="$OVR strategy.las=false" ;;
      norap) OVR="$OVR strategy.rap.enabled=false" ;;  nomu)  OVR="$OVR strategy.mu.enabled=false" ;;
      *) echo "unknown arm flag $t in $1"; return 1 ;;
    esac
  done
}
gate() {  # <out json> <pair...>: accept_gate.py, never fails the job (exit 3 = reject)
  local out="$1"; shift; "$CPUPY" scripts/accept_gate.py "$@" --out "$out" | tail -1 || true
}
floor() {  # <ckpt> <splat ply> <json>
  [ -f "$2" ] || "$GPY" scripts/export_gaussian_ply.py --checkpoint "$1" --out "$2" | tail -1
  "$CPUPY" scripts/floor_coverage.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$2" --sim3-json "$P/sim3.json" \
    --gt-model "$GT" --out "${3%.json}.png" | grep '^FLOOR_COVERAGE ' | sed 's/^FLOOR_COVERAGE //' > "$3"
}

run_arm() {  # <split> <arm>
  local split="$1" arm="$2" tag="$1_$2" RUN CK
  split_vars "$split"; arm_cfg "$arm"
  if [ "$DEPTH" = true ] && ! ls "$DATA"/images/*_depth.npy >/dev/null 2>&1; then
    echo "[$tag] no LiDAR depth targets in $DATA: depth loss off"; DEPTH=false
  fi
  RUN="$D/$split/$arm"; mkdir -p "$D/$split"
  CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== [$tag] train: $APP depth=$DEPTH $OVR"
    ( cd "$DDIR" && "$GPY" train.py --config-name "$APP" path="$DATA" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="$TSI" \
        loss.use_depth="$DEPTH" loss.lambda_depth="$LAMBDA" $OVR ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; return 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  grep -E 'ImprovedGS\] (candidate|EAS check)' "$RUN.log" | head -2 || true
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TEST" --tag "$tag" --json-out "$REP/$tag.json" \
    --save-dir "$D/renders/$tag" | tail -1
  floor "$CK" "$RUN/splat.ply" "$REP/floor_$tag.json"; echo "[$tag] floor $(cat "$REP/floor_$tag.json")"
  gate "$REP/gate_$tag.json" --pair "$split" "$BASE" "$REP/$tag.json"
  [ "$DEPTH" = true ] && gate "$REP/gate_${tag}_vsdepth.json" --pair "$split" "$BASED" "$REP/$tag.json"
  return 0
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra A <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then A=("${A[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${A[@]}"; do for split in $SPLITS; do run_arm "$split" "$arm" || { echo "[${split}_$arm] FAILED"; rc=1; }; done; done
  [ "$MODE" = arms ] && { echo "DENSIFY_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report: baselines (re-rendered with this checkout = render-path check), floor, summary, previews"
for split in $SPLITS; do
  split_vars "$split"; tag="${split}_baseline"
  CK0="$("$CPUPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$BASE")"
  [ -f "$REP/$tag.json" ] || "$GPY" scripts/eval_views.py --checkpoint "$CK0" --dataset "$TEST" --tag "$tag" \
    --json-out "$REP/$tag.json" --save-dir "$D/renders/$tag" | tail -1
  [ -s "$REP/floor_$tag.json" ] || floor "$CK0" "$D/$split/baseline_splat.ply" "$REP/floor_$tag.json"
done
"$CPUPY" - "$D" "$ON/report/arm_A.json" "$O/report/depth_baseline.json" "$SPLITS" $ARMS <<'PY'
import cv2, glob, json, os, re, sys, numpy as np
D, base_on, base_off, splits, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(), sys.argv[5:]
REP = os.path.join(D, "report"); base = {"onpath": base_on, "offpath": base_off}
test = {"onpath": os.path.join(os.path.dirname(os.path.dirname(base_on)), "test"),
        "offpath": os.path.join(os.path.dirname(os.path.dirname(base_off)), "test")}
K = ("psnr", "psnr_observed", "psnr_unobserved", "ssim", "lpips", "depth_absrel")
def ld(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None

def n_gauss(ply):
    if not os.path.exists(ply):
        return None
    with open(ply, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.strip() == b"end_header":
                return None

def train_min(log):
    m = re.findall(r"(\d+\.\d+) s\s*│", open(log, errors="ignore").read()) if os.path.exists(log) else []
    return round(float(m[-1]) / 60, 1) if m else None

rows, best = [], {}
for s in splits:
    b, fb = ld(base[s]), ld(os.path.join(REP, f"floor_{s}_baseline.json"))
    chk = ld(os.path.join(REP, f"{s}_baseline.json"))
    rows.append({"split": s, "arm": "baseline", **{k: b.get(k) for k in K}, "floor": (fb or {}).get("splat_has_floor"),
                 "gaussians": n_gauss(os.path.join(D, s, "baseline_splat.ply")), "accept": None,
                 "rerender_psnr": chk.get("psnr") if chk else None})
    for a in arms:
        r = ld(os.path.join(REP, f"{s}_{a}.json"))
        if r is None:
            continue
        g, gd = ld(os.path.join(REP, f"gate_{s}_{a}.json")), ld(os.path.join(REP, f"gate_{s}_{a}_vsdepth.json"))
        f = ld(os.path.join(REP, f"floor_{s}_{a}.json")) or {}
        rows.append({"split": s, "arm": a, **{k: r.get(k) for k in K}, "floor": f.get("splat_has_floor"),
                     "gaussians": n_gauss(os.path.join(D, s, a, "splat.ply")), "train_min": train_min(os.path.join(D, s, a + ".log")),
                     "accept": g["accept"] if g else None, "failures": g["failures"] if g else None,
                     "accept_vs_depth": gd["accept"] if gd else None})
    cand = [r for r in rows if r["split"] == s and r["arm"] != "baseline"]
    acc = [r for r in cand if r["accept"]]
    if cand:
        best[s] = max(acc or cand, key=lambda r: r["psnr"])["arm"]
json.dump({"rows": rows, "best": best}, open(os.path.join(REP, "densify_summary.json"), "w"), indent=2)

fmt = lambda v, p: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
hdr = "| split | arm | PSNR | PSNR obs | PSNR unobs | SSIM | LPIPS | depth AbsRel | floor | gaussians | train min | gate vs base | gate vs depth |"
lines = [hdr, "|" + "---|" * 13]
for r in rows:
    lines.append(f"| {r['split']} | {r['arm']} | {fmt(r['psnr'],2)} | {fmt(r.get('psnr_observed'),2)} | {fmt(r.get('psnr_unobserved'),2)} | "
                 f"{fmt(r['ssim'],4)} | {fmt(r['lpips'],4)} | {fmt(r.get('depth_absrel'),4)} | {fmt(r['floor'],3)} | {fmt(r['gaussians'],0)} | "
                 f"{fmt(r.get('train_min'),1)} | {'-' if r['accept'] is None else ('ACCEPT' if r['accept'] else 'reject')} | "
                 f"{'-' if r.get('accept_vs_depth') is None else ('ACCEPT' if r['accept_vs_depth'] else 'reject')} |")
lines.append(""); lines += [f"best {s}: {a}" for s, a in best.items()]
open(os.path.join(REP, "densify_summary.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

for s, a in best.items():  # previews: real | baseline | best arm on 4 evenly spaced held-out frames
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(test[s], "images", "*.png")) if "_covis" not in p)
    out = []
    for n in names[:: max(len(names) // 4, 1)][:4]:
        ims = [cv2.imread(os.path.join(test[s], "images", n)), cv2.imread(os.path.join(D, "renders", f"{s}_baseline", n)),
               cv2.imread(os.path.join(D, "renders", f"{s}_{a}", n))]
        if any(i is None for i in ims):
            continue
        for i, t in zip(ims, ("real", "baseline", a)):
            cv2.putText(i, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.append(np.hstack(ims))
    if out:
        cv2.imwrite(os.path.join(REP, f"preview_{s}.jpg"), cv2.resize(np.vstack(out), None, fx=0.5, fy=0.5))
        print("preview ->", os.path.join(REP, f"preview_{s}.jpg"))
PY
echo "DENSIFY_BENCH_DONE"
