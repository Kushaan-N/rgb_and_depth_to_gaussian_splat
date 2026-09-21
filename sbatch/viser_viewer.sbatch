#!/bin/bash
# NAVIGABLE real-time viewer of the trained splat via 3DGRUT's viser web GUI (port 8080).
# viser is HTTP+websocket (TCP), so it forwards cleanly over SSH / VS Code (unlike Isaac's
# WebRTC/UDP). Resumes the finished checkpoint with 0 extra training, then serves the viewer.
#   sbatch -A pi_donghyunkim_umass_edu sbatch/viser_viewer.sbatch mocap1_well-lit_trot
#SBATCH -J cear-viser
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap1_well-lit_trot}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"; deactivate 2>/dev/null || true
export THREEDGRUT_DIR="${THREEDGRUT_DIR:-$CEAR_WS/3dgrut}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source "$THREEDGRUT_DIR/.venv/bin/activate"

OUT="$CEAR_OUT/$SEQ"
CKPT="$(ls -t "$OUT"/train_3dgrut/"$SEQ"/*/ckpt_last.pt 2>/dev/null | head -1)"
DATASET="$OUT/dataset_3dgrut"
[ -f "$CKPT" ] || { echo "no checkpoint for $SEQ"; exit 1; }
[ -d "$DATASET" ] || { echo "no dataset_3dgrut for $SEQ (run train_splat first)"; exit 1; }
NODE="$(hostname)"

cat <<EOF
============================================================
 NAVIGABLE viser viewer starting on node: $NODE   (port 8080)
 Connect from your laptop (TCP tunnel — works over SSH, no VPN needed):
    ssh -N -L 8080:$NODE:8080 ${USER}@unity.rc.umass.edu
 Then open in your browser:   http://localhost:8080
 (Or on the VS Code remote: run  ssh -N -L 8080:$NODE:8080 $NODE  then use the PORTS tab.)
 Drag to orbit / WASD to fly. scancel this job when done.
============================================================
EOF
nvidia-smi --query-gpu=name --format=csv,noheader

cd "$THREEDGRUT_DIR"
# resume the finished ckpt, 0 extra iters, serve the viser GUI (blocks in the render loop)
python train.py --config-name apps/colmap_3dgut.yaml \
  path="$DATASET" out_dir="$OUT/train_3dgrut" experiment_name="$SEQ" \
  dataset.downsample_factor=1 resume="$CKPT" with_viser_gui=true n_iterations=30000
echo "VISER_VIEWER_DONE"
