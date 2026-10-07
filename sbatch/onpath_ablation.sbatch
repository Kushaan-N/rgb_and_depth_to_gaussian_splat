#!/bin/bash
# On-path ablation (the pipeline's usual every-8th held-out split) of two candidate defaults:
#   init:  original LiDAR cloud vs the dynamic-free cloud (remove_dynamic_lidar.py + clear_robot_path.py)
#   loss:  stock 3DGRUT vs LiDAR depth loss (λ=0.05, patched worktree)
# Arms A (orig, no depth) / B (orig, depth) / C (static, no depth) / D (static, depth); each scored on the
# held-out frames (PSNR/SSIM/LPIPS + depth error vs LiDAR) and for floor coverage of the exported splat.
# Needs offpath_depthloss.sbatch to have built <out>/pipeline/offpath/depth/lidar_static.ply.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/onpath_ablation.sbatch configs/<seq>.yaml
#SBATCH -J onpath-ablation
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
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; ON="$P/onpath"; REP="$ON/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
STATIC="$P/offpath/depth/lidar_static.ply"; GT="$OUT/colmap_train/sparse/0"
[ -f "$STATIC" ] || { echo "no $STATIC: run sbatch/offpath_depthloss.sbatch first"; exit 1; }
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
mkdir -p "$ON" "$REP"; bash env/make_3dgrut_depth.sh

echo "== datasets: all frames (trainer holds out every 8th) with original / dynamic-free LiDAR init"
"$CPUPY" - "$P" "$ON" <<'PY'
import os, shutil, sys
sys.path.insert(0, "scripts")
import pycolmap
from fuse_sequences import cam_to_world, link_images, write_model
P, ON = sys.argv[1], sys.argv[2]
rec = pycolmap.Reconstruction(os.path.join(P, "dataset", "sparse", "0"))
cams = {c: (v.model.name, v.width, v.height, [float(x) for x in v.params]) for c, v in rec.cameras.items()}
imgs = sorted((i.name, i.camera_id, *cam_to_world(i)) for i in rec.images.values())
for kind in ("data_orig", "data_static"):
    d = os.path.join(ON, kind)
    if not os.path.isdir(os.path.join(d, "sparse", "0")):
        write_model(os.path.join(d, "sparse", "0"), cams, imgs, points3d=True)
        link_images(os.path.join(d, "images"), os.path.join(P, "dataset", "images"), [x[0] for x in imgs])
    held = [x for k, x in enumerate(imgs) if k % 8 == 0]                 # = 3DGRUT's test_split_interval 8
    t = os.path.join(ON, "test")
    if not os.path.isdir(t):
        write_model(os.path.join(t, "sparse", "0"), cams, held)
        link_images(os.path.join(t, "images"), os.path.join(P, "dataset", "images"), [x[0] for x in held])
shutil.copyfile(os.path.join(P, "train_data", "sparse", "0", "points3D.txt"), os.path.join(ON, "data_orig", "sparse", "0", "points3D.txt"))
print(f"{len(imgs)} frames, {len(held)} held out")
PY
"$CPUPY" scripts/lidar_to_colmap_points.py --cloud "$STATIC" --sim3-json "$P/sim3.json" \
  --out-points3d "$ON/data_static/sparse/0/points3D.txt" --max-points "$(get pipeline.init_max_points 400000)"
for ds in "$ON/data_orig" "$ON/data_static" "$ON/test"; do
  "$GPY" scripts/build_depth_targets.py --dataset "$ds" --cloud "$STATIC" --sim3-json "$P/sim3.json"
done

declare -A DATA=([A]=data_orig [B]=data_orig [C]=data_static [D]=data_static)
declare -A DEPTH=([A]=false [B]=true [C]=false [D]=true)
for arm in A B C D; do
  RUN="$ON/arm_$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== arm $arm: ${DATA[$arm]}, depth loss ${DEPTH[$arm]}"
    ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$ON/${DATA[$arm]}" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=8 \
        loss.use_depth="${DEPTH[$arm]}" loss.lambda_depth="$LAMBDA" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$ON/test" --tag "arm_$arm" --json-out "$REP/arm_$arm.json" | tail -1
  [ -f "$RUN/splat.ply" ] || "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$RUN/splat.ply"
  "$CPUPY" scripts/floor_coverage.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$RUN/splat.ply" \
    --sim3-json "$P/sim3.json" --gt-model "$GT" --out "$REP/floor_arm_$arm.png" \
    | grep '^FLOOR_COVERAGE ' | sed 's/^FLOOR_COVERAGE //' > "$REP/floor_arm_$arm.json"
done

"$CPUPY" - "$REP" <<'PY'
import json, os, sys
rep = sys.argv[1]
desc = {"A": "orig init", "B": "orig init + depth", "C": "static init", "D": "static init + depth"}
out = {}
for a in "ABCD":
    r = json.load(open(os.path.join(rep, f"arm_{a}.json"))); f = json.load(open(os.path.join(rep, f"floor_arm_{a}.json")))
    out[a] = {"arm": desc[a], **{k: r.get(k) for k in ("psnr", "ssim", "lpips", "depth_absrel", "depth_within5pct")},
              "floor": f.get("splat_has_floor")}
json.dump(out, open(os.path.join(rep, "onpath_summary.json"), "w"), indent=2)
print(f"{'arm':22s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'AbsRel':>8s}{'<5%':>8s}{'floor':>8s}")
for a, r in out.items():
    print(f"{a+' '+r['arm']:22s}{r['psnr']:8.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}{r['depth_absrel']:8.4f}{r['depth_within5pct']:8.3f}{r['floor']:8.3f}")
PY
echo "ONPATH_ABLATION_DONE"
