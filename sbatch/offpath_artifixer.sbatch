#!/bin/bash
# Off-path benchmark with NVIDIA ArtiFixer (SIGGRAPH 2026): a Wan2.1 video world model, conditioned
# on 3DGRUT renders + opacity, generates the parts of the held-out views the reconstruction never
# saw, then ArtiFixer3D distills them back into a 3DGRUT splat. Same split as offpath_difix.sbatch.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/offpath_artifixer.sbatch configs/<seq>.yaml
#
# Arms scored on the held-out frames (ArtiFixer saves each next to its real frame):
#   afx_base     ArtiFixer's own 3DGRUT reconstruction of the training region (MCMC, 10k iters)
#   artifixer    the world model's output for each held-out view
#   artifixer3d  the distilled 3DGRUT splat (base + generated views), rendered at the held-out poses
#   artifixer3d+ the world model run again on the distilled splat's renders
# The metric scale is ours (mocap/LiDAR Sim3), not MoGe's monocular estimate.
# Needs env/make_artifixer_venv.sh. Results: <out>/pipeline/offpath/report/artifixer_summary.json
#SBATCH -J offpath-afx
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
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; REP="$O/report"
SEQ="$(get sequence.name)"; A="$O/artifixer"; SCENE="$A/prep/$SEQ"
hostname; date
module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$AFX_VENV/bin:$PATH"            # slangc for the 3DGRUT fork's kernel JIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$APY" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
mkdir -p "$A" "$REP"

[ -f "$O/split.json" ] || "$CPUPY" scripts/offpath_split.py --config "$CFG"
"$CPUPY" - "$O/split.json" "$P/dataset/sparse/0" "$A/train_images.txt" <<'PY'
import json, sys, pycolmap
test = set(json.load(open(sys.argv[1]))["test_names"])
names = sorted(i.name for i in pycolmap.Reconstruction(sys.argv[2]).images.values() if i.name not in test)
open(sys.argv[3], "w").write("\n".join(names) + "\n"); print(f"{len(names)} training images, {len(test)} held out")
PY
SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"

cd "$AFX_DIR"
echo "== prepare (3DGRUT reconstruction of the training region, renders, captions)"
"$APY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$P/dataset" --output_root "$SCENE" \
  --selected_image_names_file "$A/train_images.txt" --metric_scale "$SCALE"

echo "== ArtiFixer on the held-out views"
"$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
  --model_id "$MODEL_ID" --save_dir "$A/run1" --split_path "$SCENE/split.json" --render_trajectory val_frames
F1="$(dirname "$(find "$A/run1" -type d -path "*frames/batch_0000/pred" | head -1)")"

echo "== ArtiFixer3D (distill generated views into 3DGRUT)"
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred"

echo "== ArtiFixer3D+ (world model on the distilled splat)"
"$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
  --model_id "$MODEL_ID" --save_dir "$A/run2" --split_path "$SCENE/split_artifixer3d_plus.json" \
  --render_trajectory val_frames
F2="$(dirname "$(find "$A/run2" -type d -path "*frames/batch_0000/pred" | head -1)")"

cd "$REPO"
for arm in "afx_base:$F1/rendered:$F1/gt" "artifixer:$F1/pred:$F1/gt" \
           "artifixer3d:$F2/rendered:$F2/gt" "artifixer3d_plus:$F2/pred:$F2/gt"; do
  IFS=: read -r tag pred gt <<< "$arm"
  "$GPY" scripts/eval_views.py --pred-dir "$pred" --gt-dir "$gt" --tag "$tag" --json-out "$REP/$tag.json" | tail -1
done
"$CPUPY" - "$REP" "$F1" <<'PY'
import json, os, sys
from PIL import Image
rep, f1 = sys.argv[1], sys.argv[2]
arms = ["afx_base", "artifixer", "artifixer3d", "artifixer3d_plus"]
res = {a: json.load(open(os.path.join(rep, a + ".json"))) for a in arms}
gt = sorted(os.listdir(os.path.join(f1, "gt")))
out = {"held_out_frames": len(gt), "resolution": Image.open(os.path.join(f1, "gt", gt[0])).size,
       "arms": {a: {k: r[k] for k in ("psnr", "ssim", "lpips", "n")} for a, r in res.items()}}
json.dump(out, open(os.path.join(rep, "artifixer_summary.json"), "w"), indent=2)
print(f"{len(gt)} held-out frames at {out['resolution']}")
for a, r in out["arms"].items():
    print(f"{a:18s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}")
PY
echo "OFFPATH_ARTIFIXER_DONE"
