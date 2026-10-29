#!/bin/bash
# Exports in the viewer reference's axes + what a viewer shows OFF the recorded path. Converts the delivered
# splats (<out>/exports/<name>.ply, pipeline COLMAP frame) into pipeline.export_frame_model's frame
# (<out>/exports/<name>_refaxes.ply; positions, orientations, scales and SH colour), then renders them and the
# reference with gsplat's classic 3DGS rasteriser from cameras moved off the path (MOVE, camera axes:
# right down back, dataset units) -> <out>/pipeline/viewer_bench/offpath_montage.jpg.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/viewer_offpath.sbatch configs/<seq>.yaml
# Env: NAMES ("igs igs_depth"), MOVE ("0 -0.3 1.0"), EVERY (40).
#SBATCH -J viewer-offpath
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; V="$P/viewer_bench"
REF_PLY="$(get pipeline.viewer_reference_ply)"; REF_DS="$(get pipeline.viewer_reference_dataset)"
FRAME="$(get pipeline.export_frame_model)"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY="${EVERY:-40}"
hostname; date; mkdir -p "$V"
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
RT="$V/reference_dataset"
if [ ! -f "$RT/sparse/0/images.txt" ]; then
  mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
  "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
fi
TAGS="reference"
"$SPY" scripts/eval_ply.py --ply "$REF_PLY" --dataset "$RT" --every "$EVERY" --move $MOVE --save-dir "$V/offpath/reference"
for n in ${NAMES:-igs igs_depth}; do
  [ -f "$E/$n.ply" ] || continue
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
    --in-ply "$E/$n.ply" --out-ply "$E/${n}_refaxes.ply"
  "$SPY" scripts/eval_ply.py --ply "$E/${n}_refaxes.ply" --dataset "$RT" --every "$EVERY" --move $MOVE \
    --save-dir "$V/offpath/$n"
  TAGS="$TAGS $n"
done
"$CPUPY" - "$V/offpath" "$V/offpath_montage.jpg" $TAGS <<'PY'
import os, sys, cv2, numpy as np
d, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for nm in sorted(os.listdir(os.path.join(d, tags[0])))[:8]:
    ims = [cv2.resize(cv2.imread(os.path.join(d, t, nm)), (400, 300)) for t in tags]
    for x, t in zip(ims, tags): cv2.putText(x, t, (6, 22), 0, 0.7, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
PY
ls -la "$E"/*_refaxes.ply
echo VIEWER_OFFPATH_DONE
