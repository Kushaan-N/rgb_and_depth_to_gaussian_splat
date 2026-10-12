#!/bin/bash
# Viewer fine-tune of exported splats (scripts/finetune_ply_viewer.py): re-optimise the .ply through a
# differentiable copy of the viewer's rasteriser (VIEWER=pc274: PlayCanvas 2.7.4 = viewer_shots.py), then score
# with the viewer's own engine (viewer_shots.py + eval_views.py --pred-dir, as supersplat_check.sbatch).
#   1. renderer agreement: matched renderer vs viewer_shots renders of the same .ply (compare)
#   2. held-out arms (trained WITHOUT every 8th frame): base vs fine-tuned on $P/onpath/test, accept gate,
#      off-path moved views (MOVE) for visuals
#   3. all-frames deliverable: exports/edgs_igs.ply -> exports/edgs_igs_viewerft.ply (+ _refaxes)
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/finetune_viewer-%j.out sbatch/finetune_viewer.sbatch configs/<seq>.yaml
# Env: STAGE (all | compare | smoke | heldout | deliver), ARMS ("edgs_igs4m lidar_init"), ITERS (7000), VIEWER
# (pc274), RASTER (exact), DEPTH (0 = no LiDAR term), FTARGS (extra finetune flags), TAG (output suffix),
# MOVE ("0 -0.3 1.0"), EVERY_OFF (40).
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
STAGE="${STAGE:-all}"; ARMS="${ARMS:-edgs_igs4m lidar_init}"; ITERS="${ITERS:-7000}"; VIEWER="${VIEWER:-pc274}"
RASTER="${RASTER:-exact}"; DEPTH="${DEPTH:-0}"; TAG="${TAG:-viewerft}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"
F="$P/viewer_ft"; REP="$F/report"; TEST="$P/onpath/test"; TRAIN="$P/onpath/data_orig"
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
shots() {  # ply dataset outdir [extra viewer_shots args]: viewer-engine renders (skipped if complete)
  local ply="$1" ds="$2" od="$3"; shift 3
  [ -f "$od/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$ds" --out-dir "$od" "$@"
}
score() {  # tag renderdir -> $REP/<tag>.json (SuperSplat-engine metrics on the held-out frames)
  "$GPY" scripts/eval_views.py --pred-dir "$2" --dataset "$TEST" --tag "$1" --json-out "$REP/$1.json" | tail -1
}
FT() {  # in.ply out.ply dataset interval [extra]
  local in="$1" out="$2" ds="$3" iv="$4"; shift 4
  "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$in" --out "$out" --dataset "$ds" --test-split-interval "$iv" \
    --viewer "$VIEWER" --raster "$RASTER" --iters "$ITERS" --depth-lambda "$DEPTH" --eval-dataset "$TEST" ${FTARGS:-} "$@"
}

# ---- 1. renderer agreement (needs viewer renders of the base .ply at the test cameras)
if [ "$STAGE" = all ] || [ "$STAGE" = compare ] || [ "$STAGE" = smoke ]; then
  b="$(ply_of edgs_igs4m)"; shots "$b" "$TEST" "$F/onpath/edgs_igs4m"
  "$SPY" scripts/finetune_ply_viewer.py compare --ply "$b" --dataset "$TEST" --viewer-dir "$F/onpath/edgs_igs4m" \
    --every "$([ "$STAGE" = all ] && echo 1 || echo 4)" --save-dir "$F/compare" --json-out "$REP/agreement_edgs_igs4m.json"
fi
if [ "$STAGE" = smoke ]; then
  FT "$(ply_of edgs_igs4m)" "$F/smoke/edgs_igs4m_smoke.ply" "$TRAIN" 8 --iters 300 --eval-interval 0 --eval-every-frame 8
  echo FINETUNE_VIEWER_SMOKE_DONE; exit 0
fi

# ---- 2. held-out arms: base vs fine-tuned, SuperSplat engine
if [ "$STAGE" = all ] || [ "$STAGE" = heldout ]; then
  PAIRS=()
  for n in $ARMS; do
    b="$(ply_of "$n")" || continue; ft="$F/$n/${n}_$TAG.ply"
    [ -f "$ft" ] || FT "$b" "$ft" "$TRAIN" 8 2>&1 | tee "$F/${n}_$TAG.log" || true
    [ -f "$ft" ] || { echo "fine-tune of $n failed"; continue; }
    shots "$b" "$TEST" "$F/onpath/$n"; shots "$ft" "$TEST" "$F/onpath/${n}_$TAG"
    score "$n" "$F/onpath/$n"; score "${n}_$TAG" "$F/onpath/${n}_$TAG"
    shots "$b" "$TRAIN" "$F/offpath/$n" --every "$EVERY_OFF" --move $MOVE
    shots "$ft" "$TRAIN" "$F/offpath/${n}_$TAG" --every "$EVERY_OFF" --move $MOVE
    PAIRS+=(--pair "$n" "$REP/$n.json" "$REP/${n}_$TAG.json")
  done
  [ ${#PAIRS[@]} -gt 0 ] && { "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" --out "$REP/gate_$TAG.json" && echo "GATE ACCEPT" || echo "GATE REJECT (rc $?)"; }
fi

# ---- 3. all-frames deliverable (trained on every frame: its on-path score is NOT held-out)
if [ "$STAGE" = all ] || [ "$STAGE" = deliver ]; then
  EX="$OUT/exports"; b="$EX/edgs_igs.ply"; ft="$EX/edgs_igs_$TAG.ply"
  if [ -f "$b" ]; then
    [ -f "$ft" ] || FT "$b" "$ft" "$ALLDS" 0 2>&1 | tee "$F/edgs_igs_all_$TAG.log" || true
    mv -f "${ft%.ply}_ftv.json" "$F/" 2>/dev/null || true             # keep exports/ to plys
    "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$ft" \
      --out-ply "$EX/edgs_igs_${TAG}_refaxes.ply"
    shots "$b" "$TEST" "$F/onpath/edgs_igs_all"; shots "$ft" "$TEST" "$F/onpath/edgs_igs_all_$TAG"
    score edgs_igs_all "$F/onpath/edgs_igs_all"; score "edgs_igs_all_$TAG" "$F/onpath/edgs_igs_all_$TAG"
    shots "$b" "$TRAIN" "$F/offpath/edgs_igs_all" --every "$EVERY_OFF" --move $MOVE
    shots "$ft" "$TRAIN" "$F/offpath/edgs_igs_all_$TAG" --every "$EVERY_OFF" --move $MOVE
  fi
fi

# ---- summary + montages (real | base | fine-tuned per arm)
"$CPUPY" - "$F" "$REP" "$TEST/images" "$TAG" <<'PY'
import json, os, sys, cv2, numpy as np
F, rep, gt, tag = sys.argv[1:5]
print(f"{'splat':28s}{'SuperSplat-engine PSNR':>24s}{'SSIM':>8s}{'LPIPS':>8s}")
res = {}
for f in sorted(os.listdir(rep)):
    if f.endswith(".json") and not f.startswith(("gate", "agreement", "summary")):
        r = json.load(open(os.path.join(rep, f))); res[r["tag"]] = r
        print(f"{r['tag']:28s}{r['psnr']:24.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}")
json.dump(res, open(os.path.join(rep, f"summary_{tag}.json"), "w"), indent=2)
bases = [t for t in res if not t.endswith("_" + tag) and t + "_" + tag in res]
def montage(sub, names, with_gt, out):
    rows = []
    for nm in names:
        ims, labs = ([cv2.imread(os.path.join(gt, nm))], ["real"]) if with_gt else ([], [])
        for b in bases:
            for t in (b, b + "_" + tag):
                p = os.path.join(F, sub, t, nm)
                if os.path.exists(p): ims.append(cv2.imread(p)); labs.append(t)
        if not ims: continue
        ims = [cv2.resize(x, (400, 300)) for x in ims]
        for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    if rows:
        w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
if bases:
    d = os.path.join(F, "onpath", bases[0]); on = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    montage("onpath", on[::11][:5], True, os.path.join(F, f"onpath_montage_{tag}.jpg"))
    d = os.path.join(F, "offpath", bases[0])
    if os.path.isdir(d):
        off = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        montage("offpath", off[::max(1, len(off) // 5)][:5], False, os.path.join(F, f"offpath_montage_{tag}.jpg"))
PY
echo FINETUNE_VIEWER_DONE
