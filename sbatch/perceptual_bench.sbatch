#!/bin/bash
# Perceptual objective (R2): Wasserstein Distortion fine-tune (arXiv 2603.23297, WD / WD-R) of the current best held-out
# model, scored as the user sees it: each exported .ply rendered by SuperSplat's engine (scripts/viewer_shots.py) at the
# on-path held-out cameras ($P/onpath/test, every 8th frame, never trained on) -> PSNR/SSIM/LPIPS (eval_views.py
# --pred-dir) + DISTS (piq), plus off-path cameras (MOVE) and the SuperSplat editor view, and montages.
# By default an arm resumes BASE_CKPT (default: EDGS init + Improved-GS 4M, $P/edgs/on/edgs_igs4m) on its training set
# (BASE_DATA, same test_split_interval=8) for FT_ITERS more iterations with densification over (its window ended at
# 15k) and Improved-GS's multi-step updates off (an optimizer step every iteration); _mu / _scr change that (below).
# Worktree from env/make_3dgrut_perceptual.sh. Arms (ARMS):
#   base           the checkpoint itself (no training)
#   ctrl           same fine-tune on the original L1/SSIM loss (isolates the effect of the extra iterations)
#   wd             WD only:  LAMBDA_WD * WD
#   wdr<k>         WD-R:     LAMBDA_WD * WD + k * (L1/SSIM)   (paper: gamma=0.025, beta=1/0.09 -> k ~0.28)
#   <arm>_s<x>     the same with WD log2_sigma x instead of WD_SIGMA, e.g. wdr0.3_s2
#   <arm>_mu       fine-tune keeping Improved-GS's late multi-step updates (as the base run's last 6k its)
#   <arm>_scr      no fine-tune: train from scratch with the base run's settings, WD on after WD_WARMUP (3000) its
#                  (the paper's protocol; densification sees the WD gradients), e.g. wdr0.3_scr
# Viewer: ENGINE (current = live SuperSplat 3.x editor on WebGPU, default; pc274 = the old PlayCanvas 2.7.4 engine),
# BGS backgrounds (default "black grey"; the first is scored for acceptance), alpha-hole fraction from viewer_shots.
# Outputs: shots $PE/<engine>/{onpath,offpath}/<arm>, metrics + montages $PE/report_<engine>.
# Acceptance (task R2): PSNR drop <= MAX_PSNR_DROP (0.3 dB) vs base AND LPIPS and DISTS both better; then look at the
# montages (sharper, not fake texture). Everything under $P/perceptual/ (SMOKE=1: $P/perceptual_smoke, 200 its, few off-path views).
# Re-running skips finished stages; jobs with different ARMS can run concurrently (summary covers all arms scored).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/perceptual-bench-%j.out sbatch/perceptual_bench.sbatch configs/<seq>.yaml
#   SMOKE=1 ARMS="wdr0.3" sbatch -A ... sbatch/perceptual_bench.sbatch configs/<seq>.yaml
#SBATCH -J perceptual-bench
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
W="${THREEDGRUT_PERCEPTUAL_DIR:-$CEAR_WS/3dgrut-perceptual}"
RUNNER=(env TORCH_EXTENSIONS_DIR="${PERCEPTUAL_TORCH_EXT:-$CEAR_WS/torch-ext-perceptual}" PYTHONPATH="$W")
PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"                    # piq (DISTS) as a --target dir: no shared venv touched
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}" PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VB="$P/viewer_bench"
REF_DS="$(get pipeline.viewer_reference_dataset)"; FRAME="$(get pipeline.export_frame_model)"
BASE_CKPT="${BASE_CKPT:-$(ls -t "$P"/edgs/on/edgs_igs4m/run/*/ckpt_last.pt | head -1)}"; BASE_DATA="${BASE_DATA:-$P/edgs/on/data}"
BASE_CFG="$(dirname "$BASE_CKPT")/parsed.yaml"; TEST="$P/onpath/test"
ARMS="${ARMS:-base ctrl wd wdr0.3 wdr0.1}"; FT_ITERS="${FT_ITERS:-10000}"; LAMBDA_WD="${LAMBDA_WD:-0.025}"; WD_SIGMA="${WD_SIGMA:-4}"
WD_WARMUP="${WD_WARMUP:-3000}"
MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"; MAX_PSNR_DROP="${MAX_PSNR_DROP:-0.3}"; EVERY_ON=1; PE="$P/perceptual"
if [ "${SMOKE:-0}" = 1 ]; then FT_ITERS=200; EVERY_OFF=200; PE="$P/perceptual_smoke"; fi
ENGINE="${ENGINE:-current}"; BGS="${BGS:-black grey}"; BG0="${BGS%% *}"   # viewer engine; first bg = the scored one
S="$PE/$ENGINE"; REP="$PE/report_$ENGINE"; mkdir -p "$REP" "$PE/ply"
bgdir() { if [ -d "$1/$2" ]; then echo "$1/$2"; else echo "$1"; fi; }    # one --bg: no per-bg subdir
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash env/make_3dgrut_perceptual.sh; bash env/make_viewer_venv.sh
[ -d "$PIQ_LIB/piq" ] || /modules/opt/linux-ubuntu24.04-x86_64/uv/uv pip install --python "$GPY" --target "$PIQ_LIB" --no-deps piq
# base run's own settings (iterations it ended at, app, Gaussian budget), so the resume continues the same model
BASE_IT="$(sed -n 's/^n_iterations: *\([0-9]*\).*/\1/p' "$BASE_CFG")"; BUDGET="$(sed -n 's/^ *budget: *\([0-9]*\).*/\1/p' "$BASE_CFG" | head -1)"
APP=apps/colmap_3dgut.yaml; grep -q 'ImprovedGSStrategy' "$BASE_CFG" && APP=apps/colmap_3dgut_improved.yaml
echo "base $BASE_CKPT (step $BASE_IT, $APP, budget ${BUDGET:-n/a}); fine-tune $FT_ITERS its; lambda_wd $LAMBDA_WD log2_sigma $WD_SIGMA"
RT="$VB/reference_dataset"                                      # text-model view of the viewer reference (off-path cams)
if [ ! -f "$RT/sparse/0/images.txt" ]; then
  mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
  "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
fi

for arm in $ARMS; do
  echo "== $arm"; date
  PLY="$PE/ply/$arm.ply"; RUN="$PE/on/$arm"
  if [ "$arm" = base ]; then CK="$BASE_CKPT"; else
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    if [ -z "$CK" ]; then
      s="$WD_SIGMA"; a="${arm%%_*}"; [[ $arm =~ _s([0-9.]+)(_|$) ]] && s="${BASH_REMATCH[1]}"
      case $a in
        ctrl) X=(loss.lambda_wd=0.0);;
        wd)   X=(loss.lambda_wd="$LAMBDA_WD" loss.lambda_rgb_scale=0.0);;
        wdr*) X=(loss.lambda_wd="$LAMBDA_WD" loss.lambda_rgb_scale="${a#wdr}");;
        *) echo "unknown arm $arm"; exit 1;;
      esac
      X+=(loss.wd_sigma="$s")
      [ -n "$BUDGET" ] && X+=(strategy.budget="$BUDGET")
      if [[ $arm == *_scr* ]]; then                            # from scratch: the base run's schedule, WD after warm-up
        X+=(loss.wd_start_iter="$WD_WARMUP" n_iterations="$BASE_IT")
      else                                                     # fine-tune: resume the base checkpoint
        X+=(loss.wd_start_iter=0 resume="$BASE_CKPT" n_iterations=$((BASE_IT + FT_ITERS)))
        [[ $arm == *_mu* ]] || ! grep -q 'ImprovedGSStrategy' "$BASE_CFG" || X+=(strategy.mu.enabled=false)
      fi
      mkdir -p "$RUN"; echo "  ${X[*]}"
      ( cd "$W" && "${RUNNER[@]}" "$GPY" train.py --config-name "$APP" path="$BASE_DATA" out_dir="$RUN" experiment_name=run \
          dataset.downsample_factor=1 dataset.test_split_interval=8 \
          loss.use_depth=false test_last=false num_workers="${SLURM_CPUS_PER_TASK:-8}" "${X[@]}" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
      CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"; grep -a -m1 "Wasserstein" "$RUN.log" || true
    fi
  fi
  [ -f "$PLY" ] || "${RUNNER[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$PLY" | tail -2
  if [ ! -f "$S/onpath/$arm/_viewer.json" ]; then              # SuperSplat engine at the held-out cameras
    "$VPY" scripts/viewer_shots.py --ply "$PLY" --dataset "$TEST" --every "$EVERY_ON" --engine "$ENGINE" --bg $BGS \
      --out-dir "$S/onpath/$arm" 2>&1 | grep -a '^\[viewer\] [0-9]* views' || true
  fi
  [ -f "$S/onpath/$arm/_viewer.json" ] || { echo "viewer_shots failed for $arm"; exit 1; }
  for bg in $BGS; do
    [ -f "$REP/${arm}_$bg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$(bgdir "$S/onpath/$arm" "$bg")" --dataset "$TEST" \
      --tag "${arm}_$bg" --json-out "$REP/${arm}_$bg.json" | tail -1
  done
  if [ ! -f "$REP/dists_$arm.json" ]; then
    PYTHONPATH="$PIQ_LIB" "$GPY" - "$(bgdir "$S/onpath/$arm" "$BG0")" "$TEST/images" "$REP/dists_$arm.json" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, out = sys.argv[1:4]
t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
m = piq.DISTS(reduction="none").cuda(); v = []
for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
    with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
json.dump({"dists": round(float(np.mean(v)), 4), "n": len(v)}, open(out, "w")); print("DISTS", round(float(np.mean(v)), 4), len(v))
PY
  fi
  if [ ! -f "$S/offpath/$arm/_viewer.json" ]; then              # off the path + editor view, in the reference's axes
    [ -f "$PE/ply/${arm}_refaxes.ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
      --gt-model "$FRAME" --in-ply "$PLY" --out-ply "$PE/ply/${arm}_refaxes.ply" | tail -1
    "$VPY" scripts/viewer_shots.py --ply "$PE/ply/${arm}_refaxes.ply" --dataset "$RT" --every "$EVERY_OFF" --move $MOVE \
      --engine "$ENGINE" --bg "$BG0" --out-dir "$S/offpath/$arm" --editor 2>&1 | grep -a '^\[viewer\] [0-9]* views\|editor' || true
  fi
done

echo "== summary + montages"
"$CPUPY" - "$S" "$REP" "$TEST/images" "$MAX_PSNR_DROP" "$ENGINE" $BGS <<'PY'
import json, os, sys, cv2, numpy as np
S, rep, gt, max_drop, engine, bgs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6:]
ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
arms = sorted({f[6:-5] for f in os.listdir(rep) if f.startswith("dists_") and f.endswith(".json")},
              key=lambda a: (a != "base", a != "ctrl", a))
bgd = lambda d, bg: os.path.join(d, bg) if os.path.isdir(os.path.join(d, bg)) else d
res = {}
for a in arms:
    r = {}
    for i, bg in enumerate(bgs):
        j = ld(os.path.join(rep, f"{a}_{bg}.json")) or {}
        r |= {(k if i == 0 else f"{k}_{bg}"): j.get(k) for k in ("psnr", "ssim", "lpips")}
    r["dists"] = (ld(os.path.join(rep, f"dists_{a}.json")) or {}).get("dists")
    r["alpha_hole_frac"] = (ld(os.path.join(S, "onpath", a, "_viewer.json")) or {}).get("alpha_hole_frac")
    res[a] = r
b = res.get("base"); g = bgs[1] if len(bgs) > 1 else None
print(f"engine {engine}, scored on bg {bgs[0]} (on-path held-out, {gt})")
print(f"{'arm':14s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'DISTS':>8s}{'dPSNR':>8s}{'dLPIPS':>8s}{'dDISTS':>8s}"
      + (f"{'PSNR_' + g:>12s}{'LPIPS_' + g:>12s}" if g else "") + f"{'holes':>8s}  accept")
for a, r in res.items():
    if b and a != "base" and None not in (r["dists"], b["dists"], r["psnr"], b["psnr"]):
        dp, dl, dd = r["psnr"] - b["psnr"], r["lpips"] - b["lpips"], r["dists"] - b["dists"]
        r.update(d_psnr=round(dp, 3), d_lpips=round(dl, 4), d_dists=round(dd, 4), accept=bool(dp >= -max_drop and dl < 0 and dd < 0))
    f = lambda k, fmt, w=8: format(r[k], fmt) if r.get(k) is not None else format("-", f">{w}")
    print(f"{a:14s}{f('psnr','8.2f')}{f('ssim','8.4f')}{f('lpips','8.4f')}{f('dists','8.4f')}{f('d_psnr','+8.2f')}"
          f"{f('d_lpips','+8.4f')}{f('d_dists','+8.4f')}" + (f"{f('psnr_' + g,'12.2f',12)}{f('lpips_' + g,'12.4f',12)}" if g else "")
          + f"{f('alpha_hole_frac','8.4f')}  {r.get('accept', '')}")
json.dump({"engine": engine, "bg": bgs, "rule": f"psnr drop <= {max_drop} dB and lpips, dists both lower than base "
           f"(SuperSplat {engine} engine, bg {bgs[0]}, on-path held-out)", "arms": res},
          open(os.path.join(rep, "summary.json"), "w"), indent=2)
def lab(x, l): cv2.putText(x, l, (6, 22), 0, 0.7, (0, 0, 255), 2); return x
def grid(sub, bg, names, real, out, crop=False):
    rows = []
    for nm in names:
        ims = [(cv2.imread(os.path.join(gt, nm)), "real")] if real else []
        ims += [(cv2.imread(p), a) for a in arms for p in [os.path.join(bgd(os.path.join(S, sub, a), bg), nm)] if os.path.exists(p)]
        if crop:                                               # 2x zoom on the image centre: texture / sharpness check
            ims = [(x[x.shape[0] // 4: 3 * x.shape[0] // 4, x.shape[1] // 4: 3 * x.shape[1] // 4], l) for x, l in ims]
        rows.append(np.hstack([lab(cv2.resize(x, (400, 300), interpolation=cv2.INTER_NEAREST if crop else cv2.INTER_AREA), l) for x, l in ims]))
    if rows: cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
d0 = bgd(os.path.join(S, "onpath", arms[0]), bgs[0])
on = sorted(f for f in os.listdir(d0) if f.endswith(".png") and not f.startswith("_"))
grid("onpath", bgs[0], on[::max(len(on) // 5, 1)][:5], True, os.path.join(rep, "onpath_montage.jpg"))
grid("onpath", bgs[0], on[::max(len(on) // 5, 1)][:5], True, os.path.join(rep, "onpath_crops.jpg"), crop=True)
d1 = bgd(os.path.join(S, "offpath", arms[0]), bgs[0])
if os.path.isdir(d1):
    off = sorted(f for f in os.listdir(d1) if f.endswith(".png") and not f.startswith("_"))
    grid("offpath", bgs[0], off[:5], False, os.path.join(rep, "offpath_montage.jpg"))
eds = [lab(cv2.resize(cv2.imread(p), (640, 400)), a) for a in arms for p in [os.path.join(S, "offpath", a, "_supersplat_editor.png")] if os.path.exists(p)]
if eds: cv2.imwrite(os.path.join(rep, "editor_montage.jpg"), np.vstack(eds)); print("editor montage ->", os.path.join(rep, "editor_montage.jpg"))
PY
echo PERCEPTUAL_BENCH_DONE
