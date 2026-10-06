#!/bin/bash
# LiDAR depth supervision for 3DGRUT, on the off-path benchmark (same split as offpath_difix.sbatch).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/offpath_depthloss.sbatch configs/<seq>.yaml
#
#  1. static LiDAR cloud: remove moving things by visibility (remove_dynamic_lidar.py), then the
#     robot's own corridor (clear_robot_path.py)
#  2. per-frame LiDAR depth targets for train + held-out frames (build_depth_targets.py)
#  3. co-visibility masks of the held-out frames (covis_masks.py)
#  4. gradient check of 3DGRUT's rendered depth (check_depth_grad.py) — stops here if it fails
#  5. baseline (the stock-3DGRUT checkpoint from offpath_difix) vs depth-supervised runs at several
#     loss weights (patched 3DGRUT worktree, env/make_3dgrut_depth.sh), all scored on the held-out
#     frames: PSNR/SSIM/LPIPS, PSNR on observed vs never-observed pixels, depth error vs LiDAR.
# Results: <out>/pipeline/offpath/report/depthloss_summary.json
#SBATCH -J offpath-depth
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"
LAMBDAS="${LAMBDAS:-0.01 0.05 0.2}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; W="$O/work"; REP="$O/report"; DD="$O/depth"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"; export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$W" "$REP" "$DD"
bash env/make_3dgrut_depth.sh
[ -f "$O/split.json" ] || "$CPUPY" scripts/offpath_split.py --config "$CFG"
CK0="$(ls -t "$W"/baseline/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK0" ] || { echo "no baseline checkpoint: run sbatch/offpath_difix.sbatch first"; exit 1; }

echo "== 1. static LiDAR cloud: drop moving things (people, the robot's body) by visibility, then the robot's corridor"
[ -f "$DD/lidar_nodyn.ply" ] || "$CPUPY" scripts/remove_dynamic_lidar.py --config "$CFG" --cloud "$P/depth_cloud.ply" \
  --out "$DD/lidar_nodyn.ply"
[ -f "$DD/lidar_static.ply" ] && [ "$DD/lidar_static.ply" -nt "$DD/lidar_nodyn.ply" ] || \
  "$CPUPY" scripts/clear_robot_path.py --cloud "$DD/lidar_nodyn.ply" --gt-model "$OUT/colmap_train/sparse/0" \
  --out "$DD/lidar_static.ply"
echo "== 2. depth targets"
for ds in "$O/train_data" "$O/test"; do
  "$GPY" scripts/build_depth_targets.py --dataset "$ds" --cloud "$DD/lidar_static.ply" --sim3-json "$P/sim3.json"
done
"$CPUPY" - "$O/train_data" "$P/sim3.json" "$REP/depth_targets_preview.jpg" <<'PY'
import cv2, json, os, sys, numpy as np
ds, s = sys.argv[1], json.load(open(sys.argv[2]))["scale"]
names = sorted(n for n in os.listdir(os.path.join(ds, "images")) if n.endswith(".png") and "_" in n)[::100][:4]
rows = []
for n in names:
    im = cv2.imread(os.path.join(ds, "images", n)); d = np.load(os.path.join(ds, "images", n[:-4] + "_depth.npy")).astype(np.float32) * s
    c = cv2.applyColorMap((np.clip(d / 8, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO); c[d == 0] = 0
    rows.append(np.hstack([im, cv2.addWeighted(im, 0.45, c, 0.55, 0), c]))
cv2.imwrite(sys.argv[3], cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5)); print("preview ->", sys.argv[3])
PY
echo "== 3. co-visibility of the held-out frames"
"$GPY" scripts/covis_masks.py --train "$O/train_data" --test "$O/test"
echo "== 4. gradient check of the rendered depth"
( cd "$DEPTH_DIR" && "$GPY" -c "import threedgrut; print('trainer code:', threedgrut.__file__)" )
( cd "$DEPTH_DIR" && PYTHONPATH="$DEPTH_DIR" "$GPY" "$REPO/scripts/check_depth_grad.py" --checkpoint "$CK0" --dataset "$O/train_data" )

score() { "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "$O/test" --tag "$2" --json-out "$REP/$2.json" | tail -1; }
echo "== 5. baseline"
score "$CK0" depth_baseline
for lam in $LAMBDAS; do
  tag="depth_l${lam}"; RUN="$W/$tag"
  CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== depth-supervised, lambda_depth=$lam"
    ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$O/train_data" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
        loss.use_depth=true loss.lambda_depth="$lam" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  score "$CK" "$tag"
done

"$CPUPY" - "$REP" "$O/test/covis.json" $LAMBDAS <<'PY'
import json, os, sys
rep, covis, lams = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3:]
arms = ["depth_baseline"] + [f"depth_l{l}" for l in lams]
keys = ("psnr", "ssim", "lpips", "psnr_observed", "psnr_unobserved", "depth_absrel", "depth_within5pct")
res = {a: {k: v for k, v in json.load(open(os.path.join(rep, a + ".json"))).items() if k in keys} for a in arms}
json.dump({"covis": covis, "arms": res}, open(os.path.join(rep, "depthloss_summary.json"), "w"), indent=2)
print("held-out LiDAR pixels observed by a training camera:", covis["observed_fraction_of_lidar_pixels"])
print(f"{'arm':16s}" + "".join(f"{k:>17s}" for k in keys))
for a, r in res.items():
    print(f"{a:16s}" + "".join(f"{r.get(k, float('nan')):17.4f}" for k in keys))
PY
echo "OFFPATH_DEPTH_DONE"
