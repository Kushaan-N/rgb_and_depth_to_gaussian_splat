#!/bin/bash
# Viewer-friendly copies (scripts/view_frame.py: recentred at the training cameras, gravity -> SuperSplat up) of
# exported PLYs, verified in the live SuperSplat: the SuperSplat editor default view (screenshot) and the held-out
# frames rendered at the SAME cameras moved into the new frame (must score as the original PLY: a rigid move).
#   PLYS="<a.ply> <b.ply>" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_view.sbatch configs/fp2_<seq>.yaml
# Env: PLYS (required; each -> <same dir>/<name>_view.ply), UP (auto), TRAIN_DATA / TEST_DATA, SCORE (1), FP2_WS
#SBATCH -J fp2-view
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
P="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)/pipeline"
DATA="${TRAIN_DATA:-$P/onpath/train}"; TEST="${TEST_DATA:-$P/onpath/test}"
hostname; date; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
bash env/make_viewer_venv.sh
for ply in ${PLYS:?}; do
  n="$(basename "$ply" .ply)"; out="$(dirname "$ply")/${n}_view.ply"; S="$P/view_check/$n"; mkdir -p "$S"
  "$CPUPY" scripts/view_frame.py --in-ply "$ply" --out-ply "$out" --dataset "$DATA" --up "${UP:-auto}"
  # the held-out cameras in the new frame: apply the stored transform to the test model
  "$CPUPY" - "$out" "$TEST" "$S/test_view" <<'PY'
import json, os, sys, shutil, numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "scripts"); from view_frame import read_cams
T = np.array(json.load(open(os.path.splitext(sys.argv[1])[0] + ".json"))["T_new_old"]); R = T[:3, :3]
src, dst = sys.argv[2], sys.argv[3]; s0 = os.path.join(dst, "sparse", "0"); shutil.rmtree(dst, ignore_errors=True); os.makedirs(s0)
shutil.copy(os.path.join(src, "sparse", "0", "cameras.txt"), s0); open(os.path.join(s0, "points3D.txt"), "w").close()
with open(os.path.join(s0, "images.txt"), "w") as f:
    for p, Rcw, t in read_cams(src):
        Rn = Rcw @ R.T; Cn = R @ (-Rcw.T @ t) + T[:3, 3]; tn = -Rn @ Cn; x, y, z, w = Rotation.from_matrix(Rn).as_quat()
        f.write(" ".join([p[0], *map(str, (w, x, y, z, *tn)), p[8], p[9]]) + "\n\n")
os.symlink(os.path.realpath(os.path.join(src, "images")), os.path.join(dst, "images"))
PY
  if [ "${SCORE:-1}" = 1 ]; then
    "$VPY" scripts/viewer_shots.py --engine current --ply "$out" --dataset "$S/test_view" --out-dir "$S/onpath" --editor
    "$GPY" scripts/eval_views.py --pred-dir "$S/onpath" --dataset "$S/test_view" --gt-dir "$TEST/images" --tag "${n}_view" --json-out "$S/eval.json" | tail -1
  else
    "$VPY" scripts/viewer_shots.py --engine current --ply "$out" --dataset "$S/test_view" --out-dir "$S/onpath" --editor --every 1000
  fi
  md5sum "$out"; ls -la "$out"
done
echo FP2_VIEW_DONE
