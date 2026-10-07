#!/bin/bash
# World model along a DENSER camera path (off-path benchmark): does generating along a smooth, dense
# path through the held-out region (more overlapping, more consistent generated views) distil into a
# sharper splat? DENSIFY=0 is the control (exactly the held-out poses, same route), DENSIFY=K inserts K
# interpolated poses between consecutive held-out views (scripts/afx_trajectory.py).
# Base = our splat (LiDAR init + depth loss); opacity = measured visibility (best so far, see
# docs/experiments/2026-10-07_artifixer_our_splat.md); distillation = ArtiFixer3D, rendered back at the
# 108 held-out poses and scored with eval_views.py.
#   sbatch -A $CEAR_SLURM_ACCOUNT --export=ALL,DENSIFY=3 -o <log> sbatch/offpath_artifixer_dense.sbatch configs/<seq>.yaml
#SBATCH -J offpath-afx-dense
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
export CHECKPOINT_PT="${CHECKPOINT_PT:-$CEAR_WS/artifixer-checkpoints/artifixer-1.3b.pt}"
export MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
K="${DENSIFY:-3}"; LAMBDA="${LAMBDA:-0.05}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; REP="$O/report"; SEQ="$(get sequence.name)"
A="$O/artifixer_dense$K"; SCENE="$A/prep/$SEQ"; PREP0="$O/artifixer/prep/$SEQ"; TAG="afx3d_dense$K"
CK="$(ls -t "$O"/work/depth_l${LAMBDA}/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK" ] || { echo "no depth-supervised checkpoint: run sbatch/offpath_depthloss.sbatch first"; exit 1; }
[ -f "$PREP0/split.json" ] || { echo "no original ArtiFixer preparation: run sbatch/offpath_artifixer.sbatch first"; exit 1; }
hostname; date; mkdir -p "$A" "$REP"
rm -rf "$A/prep" "$A/heldout"         # regenerated below; world-model output in $A/run1 is reused if complete
SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"

echo "== camera paths: generation path (densify $K) and the held-out poses (for scoring)"
"$CPUPY" scripts/afx_trajectory.py --views "$O/test" --out "$A/traj" --densify "$K" --scale "$SCALE"
"$CPUPY" scripts/afx_trajectory.py --views "$O/test" --out "$A/val" --densify 0 --scale "$SCALE"
"$CPUPY" - "$O/split.json" "$P/dataset/sparse/0" "$A/train_images.txt" <<'PY'
import json, sys, pycolmap
test = set(json.load(open(sys.argv[1]))["test_names"])
names = sorted(i.name for i in pycolmap.Reconstruction(sys.argv[2]).images.values() if i.name not in test)
open(sys.argv[3], "w").write("\n".join(names) + "\n")
PY

echo "== our splat along the path: renders, LiDAR depth, measured visibility"
"$GPY" scripts/build_depth_targets.py --dataset "$A/traj_dataset" --cloud "$O/depth/lidar_static.ply" --sim3-json "$P/sim3.json"
"$GPY" scripts/covis_masks.py --train "$O/train_data" --test "$A/traj_dataset"
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$A/traj_dataset" --no-metrics \
  --save-dir "$A/ours/traj_rgb" --save-opacity-dir "$A/ours/traj_opacity" | tail -1
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$O/train_data" --no-metrics \
  --save-dir "$A/ours/train_rgb" | tail -1

echo "== ArtiFixer preparation along the path (reusing the original run's reconstruction + captions)"
module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
RECON0="$PREP0/$("$CPUPY" -c "import json;e=json.load(open('$PREP0/split.json'))['test']['$SEQ'];print(e['reconstruction_checkpoint'])")"
mkdir -p "$SCENE"; cp -r "$PREP0/captions" "$SCENE/"
cd "$AFX_DIR"
"$APY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$P/dataset" --output_root "$SCENE" \
  --selected_image_names_file "$A/train_images.txt" --metric_scale "$SCALE" \
  --trajectory_path "$A/traj.json" --reconstruction_checkpoint "$RECON0"
cd "$REPO"

echo "== swap in our renders + visibility for every frame of the path entry"
"$CPUPY" - "$SCENE" "$A" "$O/train_data" <<'PY'
import json, os, sys
import numpy as np
from PIL import Image
scene, A, train = sys.argv[1:4]
e = json.load(open(os.path.join(scene, "split.json")))["test"][os.path.basename(scene)]
rd, od = os.path.join(scene, e["render_dir"]), os.path.join(scene, e["opacity_dir"])
n = len([f for f in os.listdir(rd) if f.endswith(".png")])
traj = sorted(f for f in os.listdir(os.path.join(A, "traj_dataset", "images")) if f.startswith("traj_") and f.endswith(".png")
              and not f.endswith("_covis.png"))
tr = sorted(f for f in os.listdir(os.path.join(train, "images")) if f.endswith(".png"))
print(f"path entry has {n} frames; path {len(traj)}, real anchors {len(tr)}")
assert n in (len(traj), len(traj) + len(tr)), "unexpected frame layout"
ref = Image.open(os.path.join(rd, "00000.png")); refo = Image.open(os.path.join(od, "00000.png"))
for i in range(n):
    if i < len(traj):
        rgb = Image.open(os.path.join(A, "ours", "traj_rgb", traj[i]))
        op = np.asarray(Image.open(os.path.join(A, "ours", "traj_opacity", traj[i])).convert("L")).copy()
        cv = np.asarray(Image.open(os.path.join(A, "traj_dataset", "images", traj[i][:-4] + "_covis.png")))
        op[cv == 255] = 255; op[cv == 128] = 0                      # measured visibility, not splat opacity
        opi = Image.fromarray(op, mode="L")
    else:
        rgb = Image.open(os.path.join(A, "ours", "train_rgb", tr[i - len(traj)]))
        opi = Image.new("L", rgb.size, 255)                         # real training view: observed
    assert rgb.size == ref.size
    rgb.convert(ref.mode).save(os.path.join(rd, f"{i:05d}.png")); opi.convert(refo.mode).save(os.path.join(od, f"{i:05d}.png"))
print("swapped", n, "frames")
PY

echo "== world model along the path, then ArtiFixer3D distillation rendered at the held-out poses"
cd "$AFX_DIR"
NPATH="$("$CPUPY" -c "import json;print(len(json.load(open('$A/traj.json'))['frames']))")"
F1="$(dirname "$(find "$A/run1" -type d -path "*frames/batch_0000/pred" 2>/dev/null | head -1)" 2>/dev/null || true)"
if [ -z "$F1" ] || [ "$(ls "$F1/pred" 2>/dev/null | wc -l)" -ne "$NPATH" ]; then
  rm -rf "$A/run1"
  "$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
    --model_id "$MODEL_ID" --save_dir "$A/run1" --split_path "$SCENE/split.json" --render_trajectory trajectory
  F1="$(dirname "$(find "$A/run1" -type d -path "*frames/batch_0000/pred" | head -1)")"
else
  echo "reusing $NPATH world-model frames in $F1"
fi
# a custom render path needs the distill and render phases called separately
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred" --phases distill
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred" --phases render \
  --render_trajectory_path "$A/val.json"
cd "$REPO"
VR="$(find "$SCENE/artifixer3d" -type d -name renders -path "*val*" | head -1)"
[ -n "$VR" ] || VR="$(find "$SCENE/artifixer3d" -type d -name renders | sort | tail -1)"
echo "held-out renders of the distilled splat: $VR ($(ls "$VR" | wc -l) frames)"
mkdir -p "$A/heldout"
"$CPUPY" - "$VR" "$O/test" "$A/heldout" <<'PY'
import os, shutil, sys
vr, test, out = sys.argv[1:4]
names = sorted(n for n in os.listdir(os.path.join(test, "images")) if n.endswith(".png") and not n.endswith("_covis.png"))
fr = sorted(f for f in os.listdir(vr) if f.endswith(".png"))
assert len(fr) == len(names), (len(fr), len(names))
for f, n in zip(fr, names):
    shutil.copyfile(os.path.join(vr, f), os.path.join(out, os.path.splitext(n)[0] + ".png"))
print(len(names), "held-out renders named")
PY
module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
"$GPY" scripts/eval_views.py --pred-dir "$A/heldout" --dataset "$O/test" --tag "$TAG" --json-out "$REP/$TAG.json" | tail -1
echo "OFFPATH_AFX_DENSE_DONE"
