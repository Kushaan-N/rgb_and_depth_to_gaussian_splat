#!/bin/bash
# Native fisheye training (no rectification): 3DGRUT/3DGUT on the adapter's OPENCV_FISHEYE COLMAP splits (unscented
# transform through the real Kalibr equidistant model), scored natively on the held-out fisheye frames
# (scripts/eval_views.py) and exported as a standard 3DGS PLY in the same metric world frame as the pinhole sets.
#   CEAR_DATA=$MC/data/spires CEAR_OUT=$MC/outputs sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out \
#     sbatch/spires_fisheye.sbatch configs/spires_roq01.yaml
# Arms (ARMS, default "gut_mcmc"):
#   gut        apps/colmap_3dgut.yaml (3DGS densification)
#   gut_mcmc   apps/colmap_3dgut_mcmc.yaml (MCMC, LiDAR init)
#   gut_ppisp  gut_mcmc + post_processing.method=ppisp (per-camera / per-frame exposure, vignetting, CRF: the rig's
#              auto-exposure; the exported PLY is the ISP-free radiance)
#   gut_mcmc3m gut_mcmc with a 3M cap (default 1M; outdoor scenes are larger)
#   gut_edgs3m gut_mcmc3m initialised from the EDGS dense points of the rectified-pinhole export run (+ LiDAR seeds)
#   gut_mask3m gut_mcmc3m with the adapter's rig / operator masks (stage masks) excluded from the loss
#   gut_depth3m gut_mcmc3m + LiDAR ray-distance loss (patched worktree $CEAR_WS/3dgrut-depth; needs the adapter's
#              fdepth stage: <stem>_depth.npy next to the fisheye training images), LAMBDA (0.05)
# Env: ITERS (30000), EXTRA_OVR (extra hydra overrides for every arm).
# Out: <adapter.root>/run_fisheye/<arm>/ (+ <arm>_{test_in,test_out}.json), exports/<arm>.ply
#SBATCH -J spires-fisheye
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
ROOT="$(get adapter.root)"; FD="$ROOT/fisheye"; R="$ROOT/run_fisheye"; E="$R/exports"; ITERS="${ITERS:-30000}"
[ -f "$FD/train/sparse/0/images.txt" ] || { echo "missing $FD (run sbatch/spires_prepare.sbatch)"; exit 1; }
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$E"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
for arm in ${ARMS:-gut_mcmc}; do
  DIR=""; TRAIN="$FD/train"
  case "$arm" in
    gut)       APP=apps/colmap_3dgut.yaml; OVR=() ;;
    gut_mcmc)  APP=apps/colmap_3dgut_mcmc.yaml; OVR=() ;;
    gut_ppisp) APP=apps/colmap_3dgut_mcmc.yaml; OVR=(post_processing.method=ppisp) ;;
    gut_mcmc3m) APP=apps/colmap_3dgut_mcmc.yaml; OVR=(strategy.add.max_n_gaussians=3000000) ;;
    gut_edgs3m) APP=apps/colmap_3dgut_mcmc.yaml; OVR=(strategy.add.max_n_gaussians=3000000)
               # EDGS dense init (RoMa, triangulated on the rectified views: same world frame) + the LiDAR seeds
               ED="${EDGS_POINTS:-$(get paths.out_root)/pipeline/igs_full/edgs_data/sparse/0/points3D.txt}"
               [ -f "$ED" ] || { echo "no EDGS init $ED (run sbatch/spires_export.sbatch first)"; exit 1; }
               T="$FD/train_edgs"; mkdir -p "$T/sparse/0"; ln -sfn "$FD/train/images" "$T/images"
               cp "$FD/train/sparse/0/cameras.txt" "$FD/train/sparse/0/images.txt" "$T/sparse/0/"; cp "$ED" "$T/sparse/0/points3D.txt"; TRAIN="$T" ;;
    gut_mask3m) APP=apps/colmap_3dgut_mcmc.yaml; OVR=(strategy.add.max_n_gaussians=3000000)
               # rig / operator masks (adapter stage masks; 3DGRUT reads <image>_mask.png, 255 = valid -> inverted here)
               T="$FD/train_mask"; mkdir -p "$T/sparse/0" "$T/images"; cp "$FD/train/sparse/0/"*.txt "$T/sparse/0/"
               "$CPUPY" - "$FD" "$T" <<'PY'
import os, sys, cv2
FD, T = sys.argv[1:3]
for ln in open(os.path.join(FD, "train", "sparse", "0", "images.txt")):
    p = ln.split()
    if len(p) >= 10 and not ln.startswith("#"):
        n = p[9]; s = os.path.join(T, "images", n)
        if not os.path.lexists(s): os.symlink(os.path.realpath(os.path.join(FD, "train", "images", n)), s)
        m = cv2.imread(os.path.join(FD, "masks", f"cam{p[8]}.png"), 0)
        cv2.imwrite(os.path.join(T, "images", os.path.splitext(n)[0] + "_mask.png"), 255 - m)
PY
               TRAIN="$T" ;;
    gut_depth3m) APP=apps/colmap_3dgut_mcmc.yaml; DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"
               OVR=(strategy.add.max_n_gaussians=3000000 loss.use_depth=true loss.lambda_depth="${LAMBDA:-0.05}") ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  DIR="${DIR:-$THREEDGRUT_DIR}"
  RUN="$R/$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== $arm: $APP ${OVR[*]} ${EXTRA_OVR:-}, $ITERS iters on $TRAIN"
    # shellcheck disable=SC2086
    ( cd "$DIR" && "$GPY" train.py --config-name "$APP" path="$TRAIN" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 "${OVR[@]}" ${EXTRA_OVR:-} ) \
        > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  for s in test_in test_out; do
    [ -f "$R/${arm}_$s.json" ] || "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$FD/$s" --tag "${arm}_$s" \
      --json-out "$R/${arm}_$s.json" --save-dir "$R/renders_${arm}_$s" | tail -1
  done
  [ -f "$E/$arm.ply" ] || "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$E/$arm.ply"
done
ls -la "$E"; echo SPIRES_FISHEYE_DONE
