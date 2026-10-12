#!/bin/bash
# Needle suppression for off-path / editor views (env/make_3dgrut_needle.sh, patches/3dgrut-needle.patch): the robot's
# single outward-facing loop lets elongated gaussians fit on-path but streak once the camera leaves it. Arms are the
# current best (EDGS init + Improved-GS 4M budget, 'edgs_igs4m' of edgs_bench.sbatch, on-path held-out split) plus
#   needle_erank  loss.lambda_erank=LAM_ERANK   (erank-GS effective-rank regulariser, paper 0.01)
#   needle_aniso  loss.lambda_aniso=LAM_ANISO, loss.aniso_max=ANISO_MAX   (s_max/s_min cap)
#   needle_both   both
# Each arm: full training on $P/edgs/on/data (test_split_interval=8, same as edgs_igs4m), 3DGRUT eval on $P/onpath/test,
# PLY export, then what SuperSplat shows (scripts/viewer_shots.py, PlayCanvas gsplat in headless Chromium):
#   on-path held-out cameras -> eval_views.py --pred-dir (THE metric) -> accept_gate.py vs edgs_igs4m (re-rendered here)
#   off-path cameras (MOVES, reference axes, every EVERY_OFF-th reference camera) + the SuperSplat editor's default view
# plus needle statistics per PLY (fraction of gaussians with s_max/s_min > 10, > 100, erank < 1.5, opacity-weighted),
# side-by-side montages (baseline | arms). Everything under $P/needle/. Re-running skips finished stages.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/needle-bench-%j.out sbatch/needle_bench.sbatch configs/<seq>.yaml
#   ARMS="needle_erank" LAM_ERANK=0.02 sbatch ...      (arm names may carry a suffix: needle_aniso_x -> same args)
# Env: ARMS, LAM_ERANK (0.01), LAM_ANISO (0.01), ANISO_MAX (10), REG_START (0), MOVES ("0,-0.3,1.0 0,-0.6,1.5"),
#   EVERY_OFF (40), ITERS / NEEDLE_OUT (smoke tests: ITERS=500 NEEDLE_OUT=$P/needle_smoke ARMS=needle_both).
# Needs edgs_bench.sbatch (ARMS=edgs_igs4m) to have run (EDGS init + baseline checkpoint) and env/make_viewer_venv.sh.
#SBATCH -J needle-bench
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
NDIR="${THREEDGRUT_NEEDLE_DIR:-$CEAR_WS/3dgrut-needle}"
RUNNER=(env TORCH_EXTENSIONS_DIR="${NEEDLE_TORCH_EXT:-$CEAR_WS/torch-ext-needle}" PYTHONPATH="$NDIR")   # private JIT cache
ARMS="${ARMS:-needle_erank needle_aniso needle_both}"; BASE="${BASE:-edgs_igs4m}"; BUDGET="${BUDGET:-4000000}"
LAM_ERANK="${LAM_ERANK:-0.01}"; LAM_ANISO="${LAM_ANISO:-0.01}"; ANISO_MAX="${ANISO_MAX:-10}"; REG_START="${REG_START:-0}"
MOVES="${MOVES:-0,-0.3,1.0 0,-0.6,1.5}"; EVERY_OFF="${EVERY_OFF:-40}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; N="${NEEDLE_OUT:-$P/needle}"; REP="$N/report"; VB="$P/viewer_bench"
ITERS="${ITERS:-$(get pipeline.train_iterations 30000)}"; FRAME="$(get pipeline.export_frame_model)"
REF_DS="$(get pipeline.viewer_reference_dataset)"; TEST="$P/onpath/test"; DS="$P/edgs/on/data"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"; export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$DS/edgs_init.json" ] || { echo "no EDGS init at $DS: run edgs_bench.sbatch first"; exit 1; }
BASE_CK="$(ls -t "$P"/edgs/on/$BASE/run/*/ckpt_last.pt | head -1)"
mkdir -p "$N/ply" "$REP"; bash env/make_3dgrut_needle.sh; bash env/make_viewer_venv.sh
RT="$VB/reference_dataset"                                 # text-model view of the reference reconstruction (as supersplat_check)
if [ ! -f "$RT/sparse/0/images.txt" ]; then
  mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
  "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
fi

args_for() {  # arm -> trainer overrides
  local x=(strategy.budget="$BUDGET" loss.use_depth=false loss.reg_start_iteration="$REG_START")
  case $1 in
    needle_erank*) x+=(loss.lambda_erank="$LAM_ERANK");;
    needle_aniso*) x+=(loss.lambda_aniso="$LAM_ANISO" loss.aniso_max="$ANISO_MAX");;
    needle_both*)  x+=(loss.lambda_erank="$LAM_ERANK" loss.lambda_aniso="$LAM_ANISO" loss.aniso_max="$ANISO_MAX");;
    *) echo "unknown arm $1" >&2; return 1;;
  esac
  echo "${x[@]}"
}

echo "== 1. train + 3DGRUT eval + export"
for arm in $BASE $ARMS; do
  if [ "$arm" = "$BASE" ]; then CK="$BASE_CK"
  else
    RUN="$N/on/$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    if [ -z "$CK" ]; then
      X=($(args_for "$arm")); echo "== $arm: ${X[*]}"; date; mkdir -p "$N/on"
      ( cd "$NDIR" && "${RUNNER[@]}" "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$DS" out_dir="$RUN" \
          experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=8 "${X[@]}" ) \
        > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
      CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"; date
    fi
    [ -f "$REP/grut_$arm.json" ] || "${RUNNER[@]}" "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TEST" \
      --tag "grut_$arm" --json-out "$REP/grut_$arm.json" | tail -1
  fi
  if [ "$arm" = "$BASE" ] && [ -f "$VB/${arm}_refaxes.ply" ]; then          # supersplat_check already exported it
    ln -sfn "$VB/$arm.ply" "$N/ply/$arm.ply"; ln -sfn "$VB/${arm}_refaxes.ply" "$N/ply/${arm}_refaxes.ply"; fi
  [ -f "$N/ply/$arm.ply" ] || "${RUNNER[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$N/ply/$arm.ply" | tail -1
  [ -f "$N/ply/${arm}_refaxes.ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
    --gt-model "$FRAME" --in-ply "$N/ply/$arm.ply" --out-ply "$N/ply/${arm}_refaxes.ply" | tail -1
done
[ -f "$REP/grut_$BASE.json" ] || cp "$P/edgs/report/on_$BASE.json" "$REP/grut_$BASE.json"

echo "== 2. SuperSplat engine: on-path held-out (scored), off-path + editor (visual)"
for arm in $BASE $ARMS; do
  if [ ! -f "$REP/ss_$arm.json" ]; then
    "$VPY" scripts/viewer_shots.py --ply "$N/ply/$arm.ply" --dataset "$TEST" --out-dir "$N/ss/onpath/$arm" | tail -1
    "$GPY" scripts/eval_views.py --pred-dir "$N/ss/onpath/$arm" --dataset "$TEST" --tag "ss_$arm" --json-out "$REP/ss_$arm.json" | tail -1
  fi
  ed=--editor
  for m in $MOVES; do
    d="$N/ss/off_${m//,/_}/$arm"; [ -f "$d/.done" ] && { ed=; continue; }
    "$VPY" scripts/viewer_shots.py --ply "$N/ply/${arm}_refaxes.ply" --dataset "$RT" --every "$EVERY_OFF" --move ${m//,/ } \
      --out-dir "$d" $ed | tail -1 && touch "$d/.done"; ed=
  done
done

echo "== 3. gates (SuperSplat-engine on-path vs $BASE; 3DGRUT on-path for reference)"
for arm in $ARMS; do
  "$CPUPY" scripts/accept_gate.py --pair supersplat_onpath "$REP/ss_$BASE.json" "$REP/ss_$arm.json" \
    --pair grut_onpath "$REP/grut_$BASE.json" "$REP/grut_$arm.json" --out "$REP/gate_$arm.json" | tail -1 || true
done

echo "== 4. needle statistics + summary + montages"
"$CPUPY" - "$N" "$BASE" "$TEST/images" "$MOVES" $ARMS <<'PY'
import json, os, sys, cv2, numpy as np
from plyfile import PlyData
N, base, gt, moves, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(), sys.argv[5:]
rep = os.path.join(N, "report"); tags = [base] + arms
ld = lambda f: json.load(open(os.path.join(rep, f))) if os.path.isfile(os.path.join(rep, f)) else {}
def needles(ply):
    v = PlyData.read(ply)["vertex"]
    s = np.exp(np.stack([np.asarray(v[f"scale_{i}"], np.float64) for i in range(3)], 1))
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    r = s.max(1) / np.maximum(s.min(1), 1e-12)
    q = s**2 / (s**2).sum(1, keepdims=True); er = np.exp(-(q * np.log(np.maximum(q, 1e-30))).sum(1))
    vis = op > 0.1
    return {"n": int(len(r)), "frac_ratio_gt10": float((r > 10).mean()), "frac_ratio_gt100": float((r > 100).mean()),
            "frac_ratio_gt10_opacity_weighted": float((op * (r > 10)).sum() / op.sum()),
            "frac_ratio_gt10_visible": float((r[vis] > 10).mean()), "frac_erank_lt1.5": float((er < 1.5).mean()),
            "median_ratio": float(np.median(r)), "p99_ratio": float(np.percentile(r, 99))}
out = {}
for t in tags:
    st = ld(f"needles_{t}.json")
    if not st:
        st = needles(os.path.join(N, "ply", t + ".ply")); json.dump(st, open(os.path.join(rep, f"needles_{t}.json"), "w"), indent=2)
    ss, gr, g = ld(f"ss_{t}.json"), ld(f"grut_{t}.json"), ld(f"gate_{t}.json")
    out[t] = {"supersplat": {k: ss.get(k) for k in ("psnr", "ssim", "lpips")}, "grut": {k: gr.get(k) for k in ("psnr", "ssim", "lpips")},
              "needles": st, "gate": {"accept": g.get("accept"), "failures": g.get("failures")} if g else None}
json.dump(out, open(os.path.join(rep, "needle_summary.json"), "w"), indent=2)
h = f"{'arm':16s}{'SS psnr':>9s}{'ssim':>8s}{'lpips':>8s}{'dPSNR':>8s}{'grut psnr':>10s}{'N':>9s}{'r>10':>8s}{'r>10 w':>8s}{'r>100':>8s}{'er<1.5':>8s}  gate"
print(h)
for t, r in out.items():
    s, g, n = r["supersplat"], r["grut"], r["needles"]
    d = (s["psnr"] or 0) - (out[base]["supersplat"]["psnr"] or 0)
    print(f"{t:16s}{s['psnr'] or 0:9.2f}{s['ssim'] or 0:8.4f}{s['lpips'] or 0:8.4f}{d:+8.2f}{g['psnr'] or 0:10.2f}{n['n']:9d}"
          f"{n['frac_ratio_gt10']:8.3f}{n['frac_ratio_gt10_opacity_weighted']:8.3f}{n['frac_ratio_gt100']:8.3f}{n['frac_erank_lt1.5']:8.3f}  "
          f"{'-' if r['gate'] is None else ('ACCEPT' if r['gate']['accept'] else 'REJECT ' + '; '.join(r['gate']['failures']))}")
def lab(im, t): cv2.putText(im, t, (8, 26), 0, 0.8, (0, 0, 255), 2); return im
def montage(sub, names, with_gt, out_p, size=(640, 480)):
    rows = []
    for nm in names:
        ims = [lab(cv2.resize(cv2.imread(os.path.join(gt, nm)), size), "real")] if with_gt else []
        ims += [lab(cv2.resize(cv2.imread(p), size), t) for t in tags for p in [os.path.join(N, "ss", sub, t, nm)] if os.path.exists(p)]
        rows.append(np.hstack(ims))
    cv2.imwrite(out_p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]); print("montage ->", out_p)
on = sorted(f for f in os.listdir(os.path.join(N, "ss", "onpath", base)) if f.endswith(".png") and not f.startswith("_"))
montage("onpath", on[::13][:4], True, os.path.join(rep, "montage_onpath.jpg"))
eds = []
for m in moves:
    sub = "off_" + m.replace(",", "_")
    off = sorted(f for f in os.listdir(os.path.join(N, "ss", sub, base)) if f.endswith(".png") and not f.startswith("_"))
    montage(sub, off[::2][:6], False, os.path.join(rep, f"montage_{sub}.jpg"))
    eds += [lab(cv2.resize(cv2.imread(p), (800, 500)), t) for t in tags for p in [os.path.join(N, "ss", sub, t, "_supersplat_editor.png")] if os.path.exists(p)]
if eds:
    eds += [np.zeros_like(eds[0])] * (-len(eds) % 2)
    cv2.imwrite(os.path.join(rep, "montage_editor.jpg"), np.vstack([np.hstack(eds[i:i + 2]) for i in range(0, len(eds), 2)]))
    print("montage ->", os.path.join(rep, "montage_editor.jpg"))
PY
echo NEEDLE_BENCH_DONE
