#!/bin/bash
# LiDAR image-based-rendering pseudo-views for a sequence's FULL training set (scripts/make_lidar_ibr_views.py): virtual
# cameras in LiDAR free space far from the robot path (0.3-1.8 m high, pitched down 0-60 deg), each a real-pixel
# re-projection of the training frames through the LiDAR geometry + masks + depth targets. Train with them through
# sbatch/export_viewer.sbatch PSEUDO_DS=<out> (measured on the off-path benchmark, sbatch/offpath_bench.sbatch: off-path
# covis PSNR 14.3 -> 20.0 dB at -0.05 dB on-path with ~23 % of the steps on pseudo-views at weight 0.3).
#   N=120 WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/ibr_views.sbatch configs/<seq>.yaml
# Env: N (views; default 0.28 x the number of frames = ~22 % of the training steps), TAG (ibr), DATA ($P/onpath/data_orig),
#      FREESPACE ($P/freespace/grid/onpath.npz), IBR_ARGS ("--single-weight 0.3 --depth-where supervised").
# Output: $P/ibr_views/<TAG>/ (COLMAP text dataset + images/<stem>{,_mask}.png, <stem>_depth.npy, ibr.json), scene.json.
#SBATCH -J ibr-views
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010,gpu035
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
OUT="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
DATA="${DATA:-$P/onpath/data_orig}"; FS="${FREESPACE:-$P/freespace/grid/onpath.npz}"; V="$P/ibr_views/${TAG:-ibr}"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
hostname; date; mkdir -p "$P/ibr_views" logs; module list 2>&1 | tail -2 || true
echo "python: $SPY ($("$SPY" --version 2>&1))"; nvidia-smi --query-gpu=name --format=csv,noheader || true
[ -f "$FS" ] || { echo "missing free-space grid $FS (scripts/build_freespace_grid.py)"; exit 1; }
NF=$(grep -c "^[0-9]" "$DATA/sparse/0/images.txt"); N="${N:-$(( NF * 28 / 100 ))}"
[ -f "$P/ibr_views/scene.json" ] || "$CPUPY" scripts/make_lidar_ibr_views.py floor --dataset "$DATA" --cloud "$CLOUD" \
  --sim3-json "$P/sim3.json" --out "$P/ibr_views/scene.json"
# shellcheck disable=SC2086
[ -f "$V/ibr.json" ] || "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DATA" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
  --scene-json "$P/ibr_views/scene.json" --freespace "$FS" --n "$N" --out "$V" ${IBR_ARGS:---single-weight 0.3 --depth-where supervised} \
  | grep -vE "Warning|torch.from_numpy" | cut -c1-1500
date; echo "IBR_VIEWS_DONE $V ($N views of $NF frames)"
