#!/bin/bash
# LiDAR free-space floater suppression benchmark: gaussians centred in space that LiDAR beams crossed are
# floaters (they only show from novel viewpoints). Per split, scripts/build_freespace_grid.py carves the
# free-space grid from the scans near that split's TRAINING cameras (no leak from the off-path hold-out);
# arms train with the 3DGRUT worktree of env/make_3dgrut_freespace.sh (depth + Improved-GS + free-space
# patches) and freespace.grid_path set. Every arm is scored on real held-out images (eval_views.py), checked
# for floor coverage, and gated (accept_gate.py) vs the stock baseline (on-path arm_A, off-path
# depth_baseline), vs arm_B / depth_l0.05 for depth arms, and vs the same arm WITHOUT free-space from the
# densify benchmark ($P/densify/report: igs1m, igs1m_d; gs_* -> the stock baseline).
#   ARMS="igs1m_fs igs1m_d_fs gs_fs" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-2%3 \
#       sbatch/freespace_bench.sbatch configs/<seq>.yaml          # one arm per array task (ARMS[task id])
#   MODE=report sbatch -A ... -d afterany:<array job> sbatch/freespace_bench.sbatch configs/<seq>.yaml
#   sbatch -A ... sbatch/freespace_bench.sbatch configs/<seq>.yaml   # no array: all arms, then the report
# Arm names as in densify_bench.sbatch (<method><cap>m | gs, flag d = LiDAR depth loss) plus free-space flags:
#   fs  = loss + pruning (configs/base_gs.yaml freespace.* defaults)   fsl = loss only   fsp = pruning only
#   fsx = 10x loss weight + pruning;   s<N> = seed_initialization=N (retraining-noise replicates, e.g. gs_fs_s1)
# Smoke tests: ITERS=3000 GRID_STRIDE=10 FS_OUT=$P/freespace_smoke. Without LiDAR the grid stage prints
# SKIP and the arms train without the prior (same as the densify arm).
# Outputs: <out>/pipeline/freespace/{grid/<split>.{npz,json},<split>/<arm>/,renders/<split>_<arm>/,
#   report/<split>_<arm>.json, gate_*.json, freespace_summary.{json,md}, preview_<split>.jpg}
#   (preview: real | baseline | igs1m | best free-space arm).
#SBATCH -J freespace-bench
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
FDIR="${THREEDGRUT_FREESPACE_DIR:-$CEAR_WS/3dgrut-freespace}"; LAMBDA="${LAMBDA:-0.05}"
export TORCH_EXTENSIONS_DIR="${FREESPACE_TORCH_EXT:-$CEAR_WS/torch-ext-freespace}"
export PYTHONPATH="$FDIR${PYTHONPATH:+:$PYTHONPATH}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ON="$P/onpath"; O="$P/offpath"; DEN="$P/densify/report"
FS="${FS_OUT:-$P/freespace}"; REP="$FS/report"
ITERS="${ITERS:-$(get pipeline.train_iterations 30000)}"; GT="$OUT/colmap_train/sparse/0"
ARMS="${ARMS:-igs1m_fs igs1m_d_fs gs_fs}"; SPLITS="${SPLITS:-onpath offpath}"; GRID_STRIDE="${GRID_STRIDE:-1}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
mkdir -p "$REP" "$FS/grid" "$TORCH_EXTENSIONS_DIR"; bash env/make_3dgrut_freespace.sh

split_vars() {  # <split> -> DATA TSI TEST BASE (stock baseline json) BASED (depth-loss baseline json)
  case "$1" in
    onpath)  DATA="$ON/data_orig";    TSI=8;      TEST="$ON/test"; BASE="$ON/report/arm_A.json";        BASED="$ON/report/arm_B.json" ;;
    offpath) DATA="$O/train_data";    TSI=100000; TEST="$O/test";  BASE="$O/report/depth_baseline.json"; BASED="$O/report/depth_l0.05.json" ;;
    *) echo "unknown split $1"; return 1 ;;
  esac
}
arm_cfg() {  # <arm> <grid npz> -> APP OVR DEPTH NOFS (the same arm without free-space, densify report tag)
  local t n; IFS=_ read -ra T <<< "$1"; OVR=""; DEPTH=false; NOFS="${T[0]}"
  case "${T[0]}" in
    mcmc*m) n="${T[0]#mcmc}"; APP=apps/colmap_3dgut_mcmc.yaml;     OVR="strategy.add.max_n_gaussians=$(( ${n%m} * 1000000 ))" ;;
    igs*m)  n="${T[0]#igs}";  APP=apps/colmap_3dgut_improved.yaml; OVR="strategy.budget=$(( ${n%m} * 1000000 ))" ;;
    gs)     APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"; NOFS=baseline ;;
    *) echo "unknown arm method in $1"; return 1 ;;
  esac
  for t in "${T[@]:1}"; do
    case "$t" in
      d)   DEPTH=true; [ "$NOFS" = baseline ] || NOFS="${NOFS}_d" ;;
      fs)  OVR="$OVR freespace.grid_path=$2" ;;
      fsl) OVR="$OVR freespace.grid_path=$2 freespace.prune.enabled=false" ;;
      fsp) OVR="$OVR freespace.grid_path=$2 freespace.lambda_opacity=0" ;;
      fsx) OVR="$OVR freespace.grid_path=$2 freespace.lambda_opacity=0.1" ;;
      s[0-9]*) OVR="$OVR seed_initialization=${t#s}" ;;
      *) echo "unknown arm flag $t in $1"; return 1 ;;
    esac
  done
}
grid() {  # <split>: free-space grid from the scans near this split's training cameras (once, locked)
  local g="$FS/grid/$1.npz" chk=""; split_vars "$1"
  [ -f "$DEN/../$1/baseline_splat.ply" ] && chk="$DEN/../$1/baseline_splat.ply"
  [ -f "$DEN/../$1/igs1m/splat.ply" ] && chk="$chk $DEN/../$1/igs1m/splat.ply"
  ( flock 9
    [ -f "$g" ] || [ -f "$FS/grid/$1.skip" ] || {
      echo "== [grid $1] build_freespace_grid.py (cameras: $DATA)"
      "$CPUPY" scripts/build_freespace_grid.py --config "$CFG" --sim3-json "$P/sim3.json" --cameras "$DATA/sparse/0" \
        --stride "$GRID_STRIDE" --out "$g" --check-ply "$DATA/sparse/0/points3D.txt" $chk | tee "$FS/grid/$1.log" | grep -E 'SKIP|scans within|grid |FREESPACE' || true
      [ -f "$g" ] || ! grep -q 'SKIP' "$FS/grid/$1.log" || touch "$FS/grid/$1.skip"; }
  ) 9> "$FS/grid/.$1.lock"
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
  local split="$1" arm="$2" tag="$1_$2" RUN CK G
  split_vars "$split"; grid "$split"; G="$FS/grid/$split.npz"; arm_cfg "$arm" "$G"
  [ -f "$G" ] || echo "[$tag] no free-space grid (no LiDAR): trained without the prior"
  if [ "$DEPTH" = true ] && ! ls "$DATA"/images/*_depth.npy >/dev/null 2>&1; then
    echo "[$tag] no LiDAR depth targets in $DATA: depth loss off"; DEPTH=false
  fi
  RUN="$FS/$split/$arm"; mkdir -p "$FS/$split"
  CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== [$tag] train: $APP depth=$DEPTH $OVR"
    ( cd "$FDIR" && "$GPY" train.py --config-name "$APP" path="$DATA" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="$TSI" \
        loss.use_depth="$DEPTH" loss.lambda_depth="$LAMBDA" $OVR ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; return 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  grep -E "FreeSpace:" "$RUN.log" | sed -n '1p;$p' || true
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TEST" --tag "$tag" --json-out "$REP/$tag.json" \
    --save-dir "$FS/renders/$tag" | tail -1
  floor "$CK" "$RUN/splat.ply" "$REP/floor_$tag.json"; echo "[$tag] floor $(cat "$REP/floor_$tag.json")"
  gate "$REP/gate_$tag.json" --pair "$split" "$BASE" "$REP/$tag.json"
  [ "$DEPTH" = true ] && gate "$REP/gate_${tag}_vsdepth.json" --pair "$split" "$BASED" "$REP/$tag.json"
  local nofs="$DEN/${split}_$NOFS.json"; [ "$NOFS" = baseline ] && nofs="$BASE"
  [ -f "$nofs" ] && gate "$REP/gate_${tag}_vsnofs.json" --pair "$split" "$nofs" "$REP/$tag.json"
  return 0
}

if [ "$MODE" = grid ]; then for split in $SPLITS; do grid "$split"; done; exit 0; fi
if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra A <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then A=("${A[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${A[@]}"; do for split in $SPLITS; do run_arm "$split" "$arm" || { echo "[${split}_$arm] FAILED"; rc=1; }; done; done
  [ "$MODE" = arms ] && { echo "FREESPACE_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report: summary vs baseline / depth baseline / same arm without free-space, previews"
"$CPUPY" - "$FS" "$DEN" "$ON/report/arm_A.json" "$O/report/depth_baseline.json" "$SPLITS" $ARMS <<'PY'
import cv2, glob, json, os, re, sys, numpy as np
FS, DEN, base_on, base_off, splits, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(), sys.argv[6:]
REP = os.path.join(FS, "report"); base = {"onpath": base_on, "offpath": base_off}
test = {s: os.path.join(os.path.dirname(os.path.dirname(b)), "test") for s, b in base.items()}
K = ("psnr", "psnr_observed", "psnr_unobserved", "ssim", "lpips", "depth_absrel")
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None

def n_gauss(ply):
    if not os.path.exists(ply):
        return None
    with open(ply, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.strip() == b"end_header":
                return None

def pruned(log):  # total gaussians removed by free-space pruning over training
    return sum(int(m) for m in re.findall(r"FreeSpace: it \d+: pruned (\d+)", open(log, errors="ignore").read())) \
        if os.path.exists(log) else None

def nofs_tag(a):
    t = a.split("_"); m = t[0]
    if m == "gs":
        return "baseline"
    return m + ("_d" if "d" in t[1:] else "")

rows, best = [], {}
for s in splits:
    b = ld(base[s])
    rows.append({"split": s, "arm": "baseline", "src": "stock", **{k: b.get(k) for k in K}})
    for a in sorted({nofs_tag(a) for a in arms} - {"baseline"}):
        r = ld(os.path.join(DEN, f"{s}_{a}.json"))
        if r:
            g = ld(os.path.join(DEN, f"gate_{s}_{a}.json"))
            rows.append({"split": s, "arm": a, "src": "densify", **{k: r.get(k) for k in K},
                         "gaussians": n_gauss(os.path.join(os.path.dirname(DEN), s, a, "splat.ply")),
                         "accept": g["accept"] if g else None})
    for a in arms:
        r = ld(os.path.join(REP, f"{s}_{a}.json"))
        if r is None:
            continue
        g, gd, gn = (ld(os.path.join(REP, f"gate_{s}_{a}{x}.json")) for x in ("", "_vsdepth", "_vsnofs"))
        f = ld(os.path.join(REP, f"floor_{s}_{a}.json")) or {}
        rows.append({"split": s, "arm": a, "src": "freespace", **{k: r.get(k) for k in K}, "floor": f.get("splat_has_floor"),
                     "gaussians": n_gauss(os.path.join(FS, s, a, "splat.ply")), "fs_pruned": pruned(os.path.join(FS, s, a + ".log")),
                     "accept": g["accept"] if g else None, "failures": g["failures"] if g else None,
                     "accept_vs_depth": gd["accept"] if gd else None, "accept_vs_nofs": gn["accept"] if gn else None,
                     "nofs": nofs_tag(a)})
    cand = [r for r in rows if r["split"] == s and r["src"] == "freespace"]
    if cand:
        best[s] = max([r for r in cand if r["accept"]] or cand, key=lambda r: (r["psnr"], -r["lpips"]))["arm"]
grids = {s: ld(os.path.join(FS, "grid", f"{s}.json")) for s in splits}
json.dump({"rows": rows, "best": best, "grids": grids}, open(os.path.join(REP, "freespace_summary.json"), "w"), indent=2)

fmt = lambda v, p: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
acc = lambda v: "-" if v is None else ("ACCEPT" if v else "reject")
lines = ["| split | arm | PSNR | PSNR obs | PSNR unobs | SSIM | LPIPS | depth AbsRel | floor | gaussians | fs-pruned | gate vs base | vs depth | vs no-fs |",
         "|" + "---|" * 14]
for r in rows:
    lines.append(f"| {r['split']} | {r['arm']} | {fmt(r['psnr'],2)} | {fmt(r.get('psnr_observed'),2)} | {fmt(r.get('psnr_unobserved'),2)} | "
                 f"{fmt(r['ssim'],4)} | {fmt(r['lpips'],4)} | {fmt(r.get('depth_absrel'),4)} | {fmt(r.get('floor'),3)} | "
                 f"{fmt(r.get('gaussians'),0)} | {fmt(r.get('fs_pruned'),0)} | {acc(r.get('accept'))} | {acc(r.get('accept_vs_depth'))} | "
                 f"{acc(r.get('accept_vs_nofs'))} |")
lines.append("")
for s, g in grids.items():
    if g:
        lines.append(f"grid {s}: {g['scans']} scans, {g['voxel_m']:.3f} m voxels, {g['confident_free_voxels']:,} confidently free; "
                     + "; ".join(f"{k}: {v['share']*100:.2f}% of {v['gaussians']:,} in free space ({v['in_free_opaque']:,} with opacity>0.5)"
                                 for k, v in (g.get("check") or {}).items()))
lines += [f"best {s}: {a}" for s, a in best.items()]
open(os.path.join(REP, "freespace_summary.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

DR = os.path.join(os.path.dirname(DEN), "renders")
for s, a in best.items():  # previews: real | baseline | igs1m (no fs) | best free-space arm, 4 held-out frames
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(test[s], "images", "*.png")) if "_covis" not in p)
    cols = [("real", os.path.join(test[s], "images")), ("baseline", os.path.join(DR, f"{s}_baseline")),
            (nofs_tag(a), os.path.join(DR, f"{s}_{nofs_tag(a)}")), (a, os.path.join(FS, "renders", f"{s}_{a}"))]
    if nofs_tag(a) == "baseline":
        cols[2] = ("igs1m", os.path.join(DR, f"{s}_igs1m"))
    out = []
    for n in names[:: max(len(names) // 4, 1)][:4]:
        ims = [cv2.imread(os.path.join(d, n)) for _, d in cols]
        if any(i is None for i in ims):
            continue
        for i, (t, _) in zip(ims, cols):
            cv2.putText(i, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.append(np.hstack(ims))
    if out:
        cv2.imwrite(os.path.join(REP, f"preview_{s}.jpg"), cv2.resize(np.vstack(out), None, fx=0.5, fy=0.5))
        print("preview ->", os.path.join(REP, f"preview_{s}.jpg"))
PY
echo "FREESPACE_BENCH_DONE"
