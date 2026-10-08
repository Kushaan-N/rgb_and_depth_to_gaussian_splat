#!/bin/bash
# Full (non-benchmark) world-model splat of one sequence — the best method from the off-path experiments
# (docs/experiments/2026-10-08_comb_replication.md), trained on ALL real frames:
#   1. our splat: LiDAR init + 3DGRUT + LiDAR depth loss (λ=0.05), every frame used for training
#   2. planned off-path views (scripts/plan_offpath_views.py): ±0.4 m sideways and ±30° turns from every
#      8th frame, at least 0.25 m from any LiDAR return
#   3. our splat rendered at those views; ArtiFixer (Wan2.1 world model) told what was measured via the
#      LiDAR co-visibility mask (not splat opacity), generates the views
#   4. ArtiFixer3D distils real frames + generated views into one splat
# Exports <out>/exports/{full_depthloss,full_worldmodel}.ply (COLMAP frame, like pipeline/splat.ply) and
# *_metric.ply (metric frame, for Isaac); a preview montage in <out>/exports/full_worldmodel_preview.jpg.
# Needs onpath_ablation.sbatch (all-frames dataset + depth targets), offpath_depthloss.sbatch
# (dynamic-free cloud) and offpath_artifixer.sbatch (captions + a base reconstruction to replace).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/worldmodel_full.sbatch configs/<seq>.yaml
#SBATCH -J worldmodel-full
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80g
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 10:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"; AFX_DIR="${AFX_DIR:-$CEAR_WS/ArtiFixer}"; APY="$AFX_VENV/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
export CHECKPOINT_PT="${CHECKPOINT_PT:-$CEAR_WS/artifixer-checkpoints/artifixer-1.3b.pt}"
export MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; SEQ="$(get sequence.name)"; GT="$OUT/colmap_train/sparse/0"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
DATA="$P/onpath/data_orig"; STATIC="$P/offpath/depth/lidar_static.ply"; PREP0="$P/offpath/artifixer/prep/$SEQ"
F="$P/worldmodel_full"; SCENE="$F/prep/$SEQ"; E="$OUT/exports"
for need in "$DATA/sparse/0/images.txt" "$STATIC" "$PREP0/split.json"; do
  [ -e "$need" ] || { echo "missing $need (run onpath_ablation / offpath_depthloss / offpath_artifixer first)"; exit 1; }
done
hostname; date; mkdir -p "$F" "$E"
SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
bash env/make_3dgrut_depth.sh

echo "== 1. our splat on all frames: LiDAR init + depth loss"
CK="$(ls -t "$F"/ours/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
if [ -z "$CK" ]; then
  ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$DATA" out_dir="$F/ours" experiment_name=run \
      dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
      loss.use_depth=true loss.lambda_depth="$LAMBDA" ) > "$F/ours.log" 2>&1 || { tail -40 "$F/ours.log"; exit 1; }
  CK="$(ls -t "$F"/ours/run/*/ckpt_last.pt | head -1)"
fi
"$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$E/full_depthloss.ply"

echo "== 2. planned off-path views"
"$CPUPY" scripts/plan_offpath_views.py --views "$DATA" --cloud "$STATIC" --sim3-json "$P/sim3.json" --out "$F/targets"
"$GPY" scripts/build_depth_targets.py --dataset "$F/targets_dataset" --cloud "$STATIC" --sim3-json "$P/sim3.json"
"$GPY" scripts/covis_masks.py --train "$DATA" --test "$F/targets_dataset"

echo "== 3. our splat at the views (+ every real frame as anchors)"
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$F/targets_dataset" --no-metrics \
  --save-dir "$F/render/traj_rgb" --save-opacity-dir "$F/render/traj_opacity" | tail -1
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$DATA" --no-metrics --save-dir "$F/render/train_rgb" | tail -1
"$CPUPY" -c "import pycolmap,sys; print('\n'.join(sorted(i.name for i in pycolmap.Reconstruction('$P/dataset/sparse/0').images.values())))" > "$F/all_images.txt"

module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
RECON0="$PREP0/$("$CPUPY" -c "import json;e=json.load(open('$PREP0/split.json'))['test']['$SEQ'];print(e['reconstruction_checkpoint'])")"
rm -rf "$SCENE"; mkdir -p "$SCENE"; cp -r "$PREP0/captions" "$SCENE/"
cd "$AFX_DIR"
"$APY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$P/dataset" --output_root "$SCENE" \
  --selected_image_names_file "$F/all_images.txt" --metric_scale "$SCALE" \
  --trajectory_path "$F/targets.json" --reconstruction_checkpoint "$RECON0"
cd "$REPO"
"$CPUPY" - "$SCENE" "$F" "$DATA" <<'PY'
import json, os, sys
import numpy as np
from PIL import Image
scene, F, data = sys.argv[1:4]
e = json.load(open(os.path.join(scene, "split.json")))["test"][os.path.basename(scene)]
rd, od = os.path.join(scene, e["render_dir"]), os.path.join(scene, e["opacity_dir"])
n = len([f for f in os.listdir(rd) if f.endswith(".png")])
traj = sorted(f for f in os.listdir(os.path.join(F, "targets_dataset", "images")) if f.startswith("traj_")
              and f.endswith(".png") and not f.endswith("_covis.png"))
tr = sorted(f for f in os.listdir(os.path.join(data, "images")) if f.endswith(".png") and not f.endswith("_covis.png"))
assert n in (len(traj), len(traj) + len(tr)), (n, len(traj), len(tr))
ref = Image.open(os.path.join(rd, "00000.png")); refo = Image.open(os.path.join(od, "00000.png"))
for i in range(n):
    if i < len(traj):
        rgb = Image.open(os.path.join(F, "render", "traj_rgb", traj[i]))
        op = np.asarray(Image.open(os.path.join(F, "render", "traj_opacity", traj[i])).convert("L")).copy()
        cv = np.asarray(Image.open(os.path.join(F, "targets_dataset", "images", traj[i][:-4] + "_covis.png")))
        op[cv == 255] = 255; op[cv == 128] = 0                      # measured visibility, not splat opacity
        opi = Image.fromarray(op, mode="L")
    else:
        rgb = Image.open(os.path.join(F, "render", "train_rgb", tr[i - len(traj)])); opi = Image.new("L", rgb.size, 255)
    rgb.convert(ref.mode).save(os.path.join(rd, f"{i:05d}.png")); opi.convert(refo.mode).save(os.path.join(od, f"{i:05d}.png"))
print(f"swapped {n} frames ({len(traj)} planned views)")
PY

echo "== 4. world model at the views, ArtiFixer3D distillation"
cd "$AFX_DIR"
"$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
  --model_id "$MODEL_ID" --save_dir "$F/run1" --split_path "$SCENE/split.json" --render_trajectory trajectory
F1="$(dirname "$(find "$F/run1" -type d -path "*frames/batch_0000/pred" | head -1)")"
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred" --phases distill
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred" --phases render \
  --render_trajectory_path "$F/targets.json"
cd "$REPO"
CKW="$(ls -t "$SCENE"/artifixer3d/runs/*/*/ours_*/ckpt_*.pt | head -1)"
"$APY" scripts/export_gaussian_ply.py --checkpoint "$CKW" --out "$E/full_worldmodel.ply"

echo "== 5. metric copies for Isaac + preview"
for n in full_depthloss full_worldmodel; do
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$GT" \
    --in-ply "$E/$n.ply" --out-ply "$E/${n}_metric.ply"
done
VR="$(find "$SCENE/artifixer3d" -type d -name renders -path "*targets*" | head -1)"
"$CPUPY" - "$F" "$F1" "$VR" "$E/full_worldmodel_preview.jpg" <<'PY'
import os, sys, cv2, numpy as np
F, f1, vr, out = sys.argv[1:5]
traj = sorted(f for f in os.listdir(os.path.join(F, "render", "traj_rgb")) if f.endswith(".png"))
rows = []
for i in np.linspace(0, len(traj) - 1, 6).astype(int):
    ims = [cv2.imread(os.path.join(F, "render", "traj_rgb", traj[i])), cv2.imread(os.path.join(f1, "pred", f"{i:05d}.png"))]
    if vr: ims.append(cv2.imread(os.path.join(vr, sorted(os.listdir(vr))[i])))
    ims = [cv2.resize(x, (320, 240)) for x in ims if x is not None]
    for x, t in zip(ims, ["our splat", "world model", "final splat"]): cv2.putText(x, t, (4, 18), 0, 0.5, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
cv2.imwrite(out, np.vstack(rows)); print("preview ->", out)
PY
ls -la "$E"
echo "WORLDMODEL_FULL_DONE"
