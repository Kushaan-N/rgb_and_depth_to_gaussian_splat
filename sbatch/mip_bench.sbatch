#!/bin/bash
# Anti-aliasing benchmark: Mip-Splatting's 3D smoothing filter (patches/3dgrut-mip.patch, worktree from
# env/make_3dgrut_mip.sh = depth loss + Improved-GS + mip) on top of the densify_bench arms. Off-path test views
# are closer to surfaces than any training view, so small sharp gaussians (Improved-GS's edge-aware
# densification makes many) are seen at a finer sampling rate than trained; the 3D filter bounds every
# gaussian's size by the training cameras' sampling rate and is baked into exported PLY / NuRec files.
# Every arm is trained on the on-path split (every 8th frame held out) and the off-path split (1 m spatial
# hold-out), scored on real held-out images (eval_views.py), checked for floor coverage and gated vs the stock
# baseline (on-path arm_A, off-path baseline), for depth arms also vs the depth-loss-only baseline (arm_B /
# depth_l0.05), and (informational) vs the same arm without the filter from densify_bench (gs -> baseline).
#   ARMS="gs_mip igs1m_mip igs1m_d_mip" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-2 \
#       sbatch/mip_bench.sbatch configs/<seq>.yaml          # one arm per array task (ARMS[task id])
#   MODE=report sbatch -A ... -d afterany:<array job> sbatch/mip_bench.sbatch configs/<seq>.yaml
#   ITERS=2000 MIP_OUT=<dir> ARMS=igs1m_d_mip sbatch -A ... sbatch/mip_bench.sbatch configs/<seq>.yaml  # smoke
# Arm names: <method>[<cap>m][_flag...] as in densify_bench.sbatch (gs | igs<cap>m | mcmc<cap>m; d = LiDAR depth
#   loss, skipped where the split has no *_depth.npy) plus mip = model.mip_filter_3d.enabled=true.
# Report mode also checks that the best arm's exported PLY (export_gaussian_ply.py, filter baked in) re-renders
# like its checkpoint: PLY -> checkpoint (no filter) -> eval_views, metrics + max pixel difference.
# Outputs: <out>/pipeline/mip/{onpath,offpath}/<arm>/ (runs, logs, splat.ply), mip/renders/<split>_<arm>/,
#   mip/report/<split>_<arm>.json, floor_*, gate_<split>_<arm>[_vsdepth|_vsnomip].json, plyroundtrip_<split>.json,
#   <split>_<best>_nofilter.json (best checkpoint rendered without its filter),
#   mip_summary.{json,md}, preview_<split>.jpg (real | baseline | igs1m | igs1m+mip).
#SBATCH -J mip-bench
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
MDIR="${THREEDGRUT_MIP_DIR:-$CEAR_WS/3dgrut-mip}"; LAMBDA="${LAMBDA:-0.05}"
# private extension cache (this worktree's sources), and eval/export import the same checkout
export TORCH_EXTENSIONS_DIR="${MIP_TORCH_EXT:-$CEAR_WS/torch-ext-mip}"
export PYTHONPATH="$MDIR${PYTHONPATH:+:$PYTHONPATH}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ON="$P/onpath"; O="$P/offpath"; D="${MIP_OUT:-$P/mip}"; REP="$D/report"
DREP="${DENSIFY_REPORT:-$P/densify/report}"; DREN="$(dirname "$DREP")/renders"     # no-filter reference arms
ITERS="${ITERS:-$(get pipeline.train_iterations 30000)}"; GT="$OUT/colmap_train/sparse/0"   # ITERS/MIP_OUT: smoke tests
ARMS="${ARMS:-gs_mip igs1m_mip igs1m_d_mip}"
SPLITS="${SPLITS:-onpath offpath}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
mkdir -p "$REP" "$TORCH_EXTENSIONS_DIR"; bash env/make_3dgrut_mip.sh

split_vars() {  # <split> -> DATA TSI TEST BASE (stock baseline json) BASED (depth-loss baseline json)
  case "$1" in
    onpath)  DATA="$ON/data_orig";    TSI=8;      TEST="$ON/test"; BASE="$ON/report/arm_A.json";    BASED="$ON/report/arm_B.json" ;;
    offpath) DATA="$O/train_data";    TSI=100000; TEST="$O/test";  BASE="$O/report/baseline.json";  BASED="$O/report/depth_l0.05.json" ;;
    *) echo "unknown split $1"; return 1 ;;
  esac
}
arm_cfg() {  # <arm> -> APP OVR DEPTH REF (same arm without the filter: densify_bench json, or the stock baseline)
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
      mip) OVR="$OVR model.mip_filter_3d.enabled=true" ;;
      reg) OVR="$OVR $("$CPUPY" scripts/cfg_get.py configs/variants/mcmc_reg.yaml pipeline.trainer_overrides)" ;;
      *) echo "unknown arm flag $t in $1"; return 1 ;;
    esac
  done
  local nomip="${1/_mip/}"
  if [ "$nomip" = gs ]; then REF="$BASE"; else REF="$DREP/${2}_$nomip.json"; fi
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
  split_vars "$split"; arm_cfg "$arm" "$split"
  if [ "$DEPTH" = true ] && ! ls "$DATA"/images/*_depth.npy >/dev/null 2>&1; then
    echo "[$tag] no LiDAR depth targets in $DATA: depth loss off"; DEPTH=false
  fi
  RUN="$D/$split/$arm"; mkdir -p "$D/$split"
  CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== [$tag] train: $APP depth=$DEPTH $OVR"
    ( cd "$MDIR" && "$GPY" train.py --config-name "$APP" path="$DATA" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="$TSI" \
        loss.use_depth="$DEPTH" loss.lambda_depth="$LAMBDA" $OVR ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; return 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  grep -E 'Mip 3D filter on|ImprovedGS\] (candidate|EAS check)' "$RUN.log" | head -3 || true
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TEST" --tag "$tag" --json-out "$REP/$tag.json" \
    --save-dir "$D/renders/$tag" | tail -1
  floor "$CK" "$RUN/splat.ply" "$REP/floor_$tag.json"; echo "[$tag] floor $(cat "$REP/floor_$tag.json")"
  gate "$REP/gate_$tag.json" --pair "$split" "$BASE" "$REP/$tag.json"
  [ "$DEPTH" = true ] && gate "$REP/gate_${tag}_vsdepth.json" --pair "$split" "$BASED" "$REP/$tag.json"
  [ -f "$REF" ] && gate "$REP/gate_${tag}_vsnomip.json" --pair "$split" "$REF" "$REP/$tag.json"
  return 0
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra A <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then A=("${A[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${A[@]}"; do for split in $SPLITS; do run_arm "$split" "$arm" || { echo "[${split}_$arm] FAILED"; rc=1; }; done; done
  [ "$MODE" = arms ] && { echo "MIP_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report: summary, previews, PLY round trip of the best arm per split"
"$CPUPY" - "$D" "$DREP" "$DREN" "$ON/report/arm_A.json" "$O/report/baseline.json" "$SPLITS" $ARMS <<'PY'
import cv2, glob, json, os, re, sys, numpy as np
D, DREP, DREN, base_on, base_off, splits, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6].split(), sys.argv[7:]
REP = os.path.join(D, "report"); base = {"onpath": base_on, "offpath": base_off}
test = {s: os.path.join(os.path.dirname(os.path.dirname(base[s])), "test") for s in base}
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

acc = lambda g: None if g is None else bool(g["accept"])
rows, best = [], {}
for s in splits:
    b = ld(base[s]); rows.append({"split": s, "arm": "baseline", **{k: b.get(k) for k in K}})
    for a in arms:
        r = ld(os.path.join(REP, f"{s}_{a}.json"))
        if r is None:
            continue
        nm = a.replace("_mip", "")
        ref = ld(base[s]) if nm == "gs" else ld(os.path.join(DREP, f"{s}_{nm}.json"))
        if ref is not None and nm != "gs":
            rows.append({"split": s, "arm": nm + " (densify_bench)", **{k: ref.get(k) for k in K},
                         "gaussians": n_gauss(os.path.join(os.path.dirname(DREP), s, nm, "splat.ply")),
                         "accept": acc(ld(os.path.join(DREP, f"gate_{s}_{nm}.json"))),
                         "accept_vs_depth": acc(ld(os.path.join(DREP, f"gate_{s}_{nm}_vsdepth.json")))})
        f = ld(os.path.join(REP, f"floor_{s}_{a}.json")) or {}
        g = ld(os.path.join(REP, f"gate_{s}_{a}.json"))
        rows.append({"split": s, "arm": a, **{k: r.get(k) for k in K}, "floor": f.get("splat_has_floor"),
                     "gaussians": n_gauss(os.path.join(D, s, a, "splat.ply")), "train_min": train_min(os.path.join(D, s, a + ".log")),
                     "accept": acc(g), "failures": g["failures"] if g else None,
                     "accept_vs_depth": acc(ld(os.path.join(REP, f"gate_{s}_{a}_vsdepth.json"))),
                     "accept_vs_nomip": acc(ld(os.path.join(REP, f"gate_{s}_{a}_vsnomip.json"))),
                     "d_psnr_vs_nomip": None if ref is None else round(r["psnr"] - ref["psnr"], 3),
                     "d_lpips_vs_nomip": None if ref is None else round(r["lpips"] - ref["lpips"], 4)})
    cand = [r for r in rows if r["split"] == s and r["arm"] in arms]
    if cand:
        best[s] = max([r for r in cand if r["accept"]] or cand, key=lambda r: r["psnr"])["arm"]
json.dump({"rows": rows, "best": best}, open(os.path.join(REP, "mip_summary.json"), "w"), indent=2)

fmt = lambda v, p: "-" if v is None else (f"{v:+.{p}f}" if p < 0 else (f"{v:.{p}f}" if isinstance(v, float) else str(v)))
yn = lambda v: "-" if v is None else ("ACCEPT" if v else "reject")
lines = ["| split | arm | PSNR | PSNR obs | PSNR unobs | SSIM | LPIPS | depth AbsRel | floor | gaussians | train min | "
         "gate vs base | gate vs depth | gate vs no-mip | dPSNR vs no-mip | dLPIPS vs no-mip |", "|" + "---|" * 16]
for r in rows:
    dp, dl = r.get("d_psnr_vs_nomip"), r.get("d_lpips_vs_nomip")
    lines.append(f"| {r['split']} | {r['arm']} | {fmt(r['psnr'],2)} | {fmt(r.get('psnr_observed'),2)} | {fmt(r.get('psnr_unobserved'),2)} | "
                 f"{fmt(r['ssim'],4)} | {fmt(r['lpips'],4)} | {fmt(r.get('depth_absrel'),4)} | {fmt(r.get('floor'),3)} | "
                 f"{fmt(r.get('gaussians'),0)} | {fmt(r.get('train_min'),1)} | {yn(r.get('accept'))} | {yn(r.get('accept_vs_depth'))} | "
                 f"{yn(r.get('accept_vs_nomip'))} | {'-' if dp is None else f'{dp:+.2f}'} | {'-' if dl is None else f'{dl:+.4f}'} |")
lines.append(""); lines += [f"best {s}: {a}" for s, a in best.items()]
open(os.path.join(REP, "mip_summary.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
open(os.path.join(REP, "best.txt"), "w").write("".join(f"{s} {a}\n" for s, a in best.items()))

for s in splits:  # previews: real | baseline | igs1m | igs1m+mip (or the best mip arm) on 4 held-out frames
    mip = "igs1m_mip" if os.path.isdir(os.path.join(D, "renders", f"{s}_igs1m_mip")) else best.get(s)
    if mip is None:
        continue
    cols = [(os.path.join(test[s], "images"), "real"), (os.path.join(DREN, f"{s}_baseline"), "baseline"),
            (os.path.join(DREN, f"{s}_" + mip.replace("_mip", "")), mip.replace("_mip", "")),
            (os.path.join(D, "renders", f"{s}_{mip}"), mip)]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(test[s], "images", "*.png")) if "_covis" not in p)
    out = []
    for n in names[:: max(len(names) // 4, 1)][:4]:
        ims = [cv2.imread(os.path.join(d, n)) for d, _ in cols]
        if any(i is None for i in ims):
            continue
        for i, (_, t) in zip(ims, cols):
            cv2.putText(i, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.append(np.hstack(ims))
    if out:
        cv2.imwrite(os.path.join(REP, f"preview_{s}.jpg"), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("preview ->", os.path.join(REP, f"preview_{s}.jpg"))
PY

# PLY round trip: the exported PLY (filter baked into scale/opacity) loaded back as a filter-free checkpoint must
# render like the checkpoint itself
export CUDA_LAUNCH_BLOCKING=1   # edited checkpoints: synchronous CUDA (see object_completion.sbatch)
while read -r split arm; do
  split_vars "$split"; tag="${split}_${arm}"; RUN="$D/$split/$arm"
  CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  [ -f "$RUN/splat.ply" ] || "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$RUN/splat.ply" | tail -1
  "$GPY" - "$CK" "$RUN/splat.ply" "$RUN/from_ply.pt" <<'PY'
import sys, torch
from threedgrut.model.model import MixtureOfGaussians
ck_path, ply, out = sys.argv[1:4]
ck = torch.load(ck_path, weights_only=False)
m = MixtureOfGaussians(ck["config"]); m.init_from_ply(ply, init_model=False)
new = dict(ck); new.pop("mip_filter_3d", None)
for k in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
    new[k] = torch.nn.Parameter(getattr(m, k).detach().clone().float())   # keep nn.Parameter types
f = ck.get("mip_filter_3d")
print(f"[ply-roundtrip] {m.num_gaussians} gaussians, filter in ckpt: {f is not None}"
      + (f" (median {f.median().item():.3g})" if f is not None else ""))
torch.save(new, out)
PY
  "$GPY" scripts/eval_views.py --checkpoint "$RUN/from_ply.pt" --dataset "$TEST" --tag "${tag}_fromply" \
    --json-out "$REP/${tag}_fromply.json" --save-dir "$D/renders/${tag}_fromply" | tail -1
  "$CPUPY" - "$REP/$tag.json" "$REP/${tag}_fromply.json" "$D/renders/$tag" "$D/renders/${tag}_fromply" \
      "$REP/plyroundtrip_$split.json" <<'PY'
import glob, json, os, sys, cv2, numpy as np
a, b, da, db, out = sys.argv[1:6]
A, B = json.load(open(a)), json.load(open(b))
diffs = []
for p in sorted(glob.glob(os.path.join(da, "*.png"))):
    q = cv2.imread(os.path.join(db, os.path.basename(p)))
    if q is not None:
        diffs.append(np.abs(cv2.imread(p).astype(int) - q.astype(int)))
res = {"ckpt": {k: A.get(k) for k in ("psnr", "ssim", "lpips")}, "from_ply": {k: B.get(k) for k in ("psnr", "ssim", "lpips")},
       "n_images": len(diffs), "max_abs_diff_8bit": int(max(d.max() for d in diffs)) if diffs else None,
       "mean_abs_diff_8bit": float(np.mean([d.mean() for d in diffs])) if diffs else None}
res["identical"] = bool(diffs) and abs(A["psnr"] - B["psnr"]) < 0.01 and res["max_abs_diff_8bit"] <= 2
json.dump(res, open(out, "w"), indent=2); print("PLY_ROUNDTRIP", json.dumps(res))
PY
  # diagnostic: the same checkpoint rendered WITHOUT its 3D filter (raw trained parameters) -> filter's test-time effect
  "$GPY" -c "import sys, torch; c = torch.load(sys.argv[1], weights_only=False); c.pop('mip_filter_3d', None); torch.save(c, sys.argv[2])" \
    "$CK" "$RUN/nofilter.pt"
  "$GPY" scripts/eval_views.py --checkpoint "$RUN/nofilter.pt" --dataset "$TEST" --tag "${tag}_nofilter" \
    --json-out "$REP/${tag}_nofilter.json" | tail -1
done < "$REP/best.txt"
echo "MIP_BENCH_DONE"
