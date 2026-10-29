#!/bin/bash
# What a standard 3DGS viewer (SuperSplat) shows: every candidate's EXPORTED .ply rendered with gsplat's classic
# 3DGS rasteriser (scripts/eval_ply.py) on the on-path held-out frames (every 8th; no candidate trained on them),
# next to the 3DGRUT-renderer score, plus a side-by-side montage. The reference is the splat the user rates best
# in a viewer (pipeline.viewer_reference_ply, trained on the same split in its own COLMAP reconstruction).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/viewer_bench.sbatch configs/<seq>.yaml
#SBATCH -J viewer-bench
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; V="$P/viewer_bench"; REP="$V/report"
REF_PLY="$(get pipeline.viewer_reference_ply "")"; REF_DS="$(get pipeline.viewer_reference_dataset "")"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"

# candidates trained WITHOUT the held-out frames: name -> checkpoint (exported here if no ply yet)
declare -A CK=(
  [lidar_init]="$(ls -t "$P"/onpath/arm_A/run/*/ckpt_last.pt | head -1)"
  [lidar_init_depth]="$(ls -t "$P"/onpath/arm_B/run/*/ckpt_last.pt | head -1)"
  [igs]="$(ls -t "$P"/densify/onpath/igs1m/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  [igs_depth]="$(ls -t "$P"/densify/onpath/igs2m_d/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
)
for n in "${!CK[@]}"; do
  ck="${CK[$n]}"; [ -n "$ck" ] || continue
  [ -f "$V/$n.ply" ] || PYTHONPATH="$CEAR_WS/3dgrut-densify" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-densify" \
    "$GPY" scripts/export_gaussian_ply.py --checkpoint "$ck" --out "$V/$n.ply"
  "$SPY" scripts/eval_ply.py --ply "$V/$n.ply" --dataset "$P/onpath/test" --tag "$n" --json-out "$REP/$n.json" \
    --save-dir "$V/renders/$n" | tail -1
done
if [ -n "$REF_PLY" ] && [ -f "$REF_PLY" ]; then
  RT="$V/reference_dataset"                                    # text-model view of the reference reconstruction
  if [ ! -f "$RT/sparse/0/images.txt" ]; then
    mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
    "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
  fi
  "$SPY" scripts/eval_ply.py --ply "$REF_PLY" --dataset "$RT" --names-from "$P/onpath/test" --tag reference \
    --json-out "$REP/reference.json" --save-dir "$V/renders/reference" | tail -1
fi

"$CPUPY" - "$V" "$REP" "$P/onpath/test/images" "$P/onpath/report" <<'PY'
import glob, json, os, sys, cv2, numpy as np
V, rep, gt, onrep = sys.argv[1:5]
r3 = {"lidar_init": "arm_A.json", "lidar_init_depth": "arm_B.json"}
rows = sorted(glob.glob(os.path.join(rep, "*.json")))
print(f"{'splat':18s}{'viewer PSNR':>12s}{'SSIM':>8s}{'LPIPS':>8s}   (3DGRUT renderer PSNR / LPIPS)")
for f in rows:
    r = json.load(open(f)); t = r["tag"]; o = ""
    if t in r3 and os.path.exists(os.path.join(onrep, r3[t])):
        q = json.load(open(os.path.join(onrep, r3[t]))); o = f"{q['psnr']:.2f} / {q['lpips']:.4f}"
    print(f"{t:18s}{r['psnr']:12.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}   {o}")
names = sorted(os.listdir(os.path.join(V, "renders", "lidar_init")))[::9][:5]
order = ["reference", "lidar_init", "igs", "igs_depth"]
out = []
for nm in names:
    ims = [cv2.imread(os.path.join(gt, nm))] + [cv2.imread(os.path.join(V, "renders", k, nm)) for k in order
                                                if os.path.exists(os.path.join(V, "renders", k, nm))]
    labs = ["real"] + [k for k in order if os.path.exists(os.path.join(V, "renders", k, nm))]
    ims = [cv2.resize(x, (400, 300)) for x in ims]
    for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.7, (0, 0, 255), 2)
    out.append(np.hstack(ims))
cv2.imwrite(os.path.join(V, "montage.jpg"), np.vstack(out)); print("montage ->", os.path.join(V, "montage.jpg"))
PY
echo VIEWER_BENCH_DONE
