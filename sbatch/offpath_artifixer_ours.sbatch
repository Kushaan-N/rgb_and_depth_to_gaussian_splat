#!/bin/bash
# World model + our geometry: ArtiFixer generates the held-out views from OUR splat (LiDAR init +
# LiDAR depth loss), then the generated views are distilled two ways, on the off-path benchmark:
#   afx3d_ours_theirs  ArtiFixer's own ArtiFixer3D distillation (its 3DGRUT fork)
#   afx3d_ours_depth   our patched 3DGRUT: LiDAR init + depth loss (λ=0.05), real frames + generated
#                      views, LiDAR depth targets for both
# OPACITY=render (default) feeds the world model our splat's opacity; OPACITY=covis feeds measured
# visibility instead: training views opaque, held-out pixels no training camera saw transparent (a
# LiDAR-seeded splat is opaque everywhere, so its opacity no longer marks what was never seen).
# Needs offpath_depthloss.sbatch (the depth-supervised checkpoint + targets) and the ArtiFixer venv.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/offpath_artifixer_ours.sbatch configs/<seq>.yaml
#SBATCH -J offpath-afx-ours
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80g
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 08:00:00

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
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; REP="$O/report"; SEQ="$(get sequence.name)"
OPAC="${OPACITY:-render}"; SUF=""; [ "$OPAC" = covis ] && SUF="_covis"
A="$O/artifixer_ours$SUF"; SCENE="$A/prep/$SEQ"; ITERS="$(get pipeline.train_iterations 30000)"
APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
CK="$(ls -t "$O"/work/depth_l${LAMBDA}/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
[ -n "$CK" ] || { echo "no depth-supervised checkpoint: run sbatch/offpath_depthloss.sbatch first"; exit 1; }
hostname; date; mkdir -p "$A" "$REP"
SCALE="$("$CPUPY" -c "import json;print(json.load(open('$P/sim3.json'))['scale'])")"
"$CPUPY" - "$O/split.json" "$P/dataset/sparse/0" "$A/train_images.txt" <<'PY'
import json, sys, pycolmap
test = set(json.load(open(sys.argv[1]))["test_names"])
names = sorted(i.name for i in pycolmap.Reconstruction(sys.argv[2]).images.values() if i.name not in test)
open(sys.argv[3], "w").write("\n".join(names) + "\n")
PY

echo "== our splat's renders + opacity for every frame ($CK)"
# ArtiFixer's 3DGRUT fork cannot load our checkpoints (different gaussian feature layout), but the world
# model only consumes renders/NNNNN.png + opacity/NNNNN.png (8-bit), indexed in sorted-image order. So:
# reuse the original ArtiFixer preparation (captions, cameras, split) and swap in OUR renders.
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
rm -rf "$SCENE" "$A/run1" "$A/run2" "$A/ours_render"            # never reuse a previous attempt
PREP0="$O/artifixer/prep/$SEQ"
[ -f "$PREP0/split.json" ] || { echo "no original ArtiFixer preparation: run sbatch/offpath_artifixer.sbatch first"; exit 1; }
mkdir -p "$(dirname "$SCENE")"; cp -r "$PREP0" "$SCENE"
rm -rf "$SCENE/artifixer3d" "$SCENE/split_artifixer3d_plus.json"   # outputs of the original run: recompute
"$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$P/dataset" --no-metrics \
  --save-dir "$A/ours_render/rgb" --save-opacity-dir "$A/ours_render/opacity" | tail -1
"$CPUPY" - "$SCENE" "$P/dataset/sparse/0" "$A/ours_render" "$OPAC" "$O/test/images" <<'PY'
import json, os, shutil, sys
import numpy as np
import pycolmap
from PIL import Image
scene, model, src, mode, test_dir = sys.argv[1:6]
held = {n for n in os.listdir(test_dir) if n.endswith(".png") and not n.endswith("_covis.png")}
e = json.load(open(os.path.join(scene, "split.json")))["test"][os.path.basename(scene)]
names = sorted(i.name for i in pycolmap.Reconstruction(model).images.values())
for kind, key in (("rgb", "render_dir"), ("opacity", "opacity_dir")):
    dst = os.path.join(scene, e[key]); ref = Image.open(os.path.join(dst, "00000.png"))
    for i, n in enumerate(names):
        im = Image.open(os.path.join(src, kind, os.path.splitext(n)[0] + ".png")).convert(ref.mode)
        assert im.size == ref.size, (im.size, ref.size)
        if kind == "opacity" and mode == "covis":                  # measured visibility, not splat opacity
            if n not in held:
                im = Image.new("L", im.size, 255)                  # a training view: observed by definition
            else:
                cv = np.asarray(Image.open(os.path.join(test_dir, os.path.splitext(n)[0] + "_covis.png")))
                a = np.asarray(im).copy(); a[cv == 255] = 255; a[cv == 128] = 0
                im = Image.fromarray(a, mode="L")
        im.save(os.path.join(dst, f"{i:05d}.png"))
    print(f"replaced {len(names)} {kind} frames in {dst}")
PY
echo "base: our splat, opacity source: $OPAC" | tee "$REP/afx_ours${SUF}_base.txt"
module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
cd "$AFX_DIR"
"$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
  --model_id "$MODEL_ID" --save_dir "$A/run1" --split_path "$SCENE/split.json" --render_trajectory val_frames
F1="$(dirname "$(find "$A/run1" -type d -path "*frames/batch_0000/pred" | head -1)")"
"$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred"
"$APY" -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT" \
  --model_id "$MODEL_ID" --save_dir "$A/run2" --split_path "$SCENE/split_artifixer3d_plus.json" --render_trajectory val_frames
F2="$(dirname "$(find "$A/run2" -type d -path "*frames/batch_0000/pred" | head -1)")"

cd "$REPO"
module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
for arm in "afx_ours_base$SUF:$F1/rendered:$F1/gt" "afx_ours_views$SUF:$F1/pred:$F1/gt" "afx3d_ours_theirs$SUF:$F2/rendered:$F2/gt"; do
  IFS=: read -r tag pred gt <<< "$arm"
  "$GPY" scripts/eval_views.py --pred-dir "$pred" --gt-dir "$gt" --tag "$tag" --json-out "$REP/$tag.json" | tail -1
done

echo "== our distillation: LiDAR init + depth loss on real frames + generated views"
G="$A/generated"; rm -rf "$G"; mkdir -p "$G"
"$CPUPY" - "$P/dataset/sparse/0" "$O/test" "$F1/pred" "$G" <<'PY'
import os, sys, pycolmap
names = sorted(i.name for i in pycolmap.Reconstruction(sys.argv[1]).images.values())   # ArtiFixer frame index
test = [n for n in os.listdir(os.path.join(sys.argv[2], "images")) if n.endswith(".png") and not n.endswith("_covis.png")]
for n in test:
    os.symlink(os.path.join(sys.argv[3], f"{names.index(n):05d}.png"), os.path.join(sys.argv[4], os.path.splitext(n)[0] + ".png"))
print(f"{len(test)} generated views linked under the held-out names")
PY
"$CPUPY" scripts/difix3d_views.py trainset --train "$O/train_data" --fixed "$G" --novel "$O/test" \
  --out "$A/distill_data" --real-repeat 1
"$GPY" scripts/build_depth_targets.py --dataset "$A/distill_data" --cloud "$O/depth/lidar_static.ply" --sim3-json "$P/sim3.json"
RUN="$A/distill_depth"
( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$A/distill_data" out_dir="$RUN" experiment_name=run \
    dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
    loss.use_depth=true loss.lambda_depth="$LAMBDA" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
"$GPY" scripts/eval_views.py --checkpoint "$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)" --dataset "$O/test" \
  --tag afx3d_ours_depth$SUF --json-out "$REP/afx3d_ours_depth$SUF.json" | tail -1

"$CPUPY" - "$REP" "$SUF" <<'PY'
import json, os, sys
rep, suf = sys.argv[1], sys.argv[2]
arms = ["depth_baseline", "depth_l0.05", "artifixer3d"] + [a + suf for a in ("afx_ours_base", "afx_ours_views", "afx3d_ours_theirs", "afx3d_ours_depth")]
keys = ("psnr", "ssim", "lpips", "psnr_observed", "psnr_unobserved", "depth_absrel")
res = {a: json.load(open(os.path.join(rep, a + ".json"))) for a in arms if os.path.exists(os.path.join(rep, a + ".json"))}
json.dump({a: {k: r.get(k) for k in keys} for a, r in res.items()}, open(os.path.join(rep, f"afx_ours{suf}_summary.json"), "w"), indent=2)
print(f"{'arm':20s}" + "".join(f"{k:>16s}" for k in keys))
for a, r in res.items():
    print(f"{a:20s}" + "".join(f"{r[k]:16.4f}" if r.get(k) is not None else f"{'-':>16s}" for k in keys))
PY
echo "OFFPATH_AFX_OURS_DONE"
