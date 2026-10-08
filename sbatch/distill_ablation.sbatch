#!/bin/bash
# How should world-model views enter training? Off-path benchmark split, every 8th real training frame
# also held out so ON-path quality is scored against real images too (the world-model splats looked
# worse than the plain splats in a viewer; only off-path had been measured).
#   A  real frames only                     (our recipe: LiDAR init + LiDAR depth loss)
#   B  + generated views, full weight       (what ArtiFixer3D does)
#   C  + generated views, only never-observed pixels (per-image loss masks, build_distill_set.py)
#   D  ArtiFixer3D's own distillation (existing), rendered at the same poses — it trained on ~7/8 of
#      the on-path held-out frames, so its on-path score is optimistic
# Generated views = ArtiFixer from our splat + visibility mask (offpath_artifixer_ours.sbatch, covis).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/distill_ablation.sbatch configs/<seq>.yaml
#SBATCH -J distill-ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80g
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 05:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
AFX_VENV="${AFX_VENV:-$CEAR_WS/venv-artifixer}"; AFX_DIR="${AFX_DIR:-$CEAR_WS/ArtiFixer}"; APY="$AFX_VENV/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; O="$P/offpath"; SEQ="$(get sequence.name)"; REP="$O/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
D="$O/distill_ablation"; AO="$O/artifixer_ours_covis"; SCENE="$AO/prep/$SEQ"
F1="$(dirname "$(find "$AO/run1" -type d -path "*frames/batch_0000/pred" | head -1)")"
[ -d "$F1/pred" ] || { echo "no world-model views: run offpath_artifixer_ours.sbatch (OPACITY=covis) first"; exit 1; }
hostname; date; mkdir -p "$D" "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
bash env/make_3dgrut_depth.sh

echo "== generated views under the held-out names"
rm -rf "$D/gen"; mkdir -p "$D/gen"
"$CPUPY" - "$P/dataset/sparse/0" "$O/test" "$F1/pred" "$D/gen" <<'PY'
import os, sys, pycolmap
names = sorted(i.name for i in pycolmap.Reconstruction(sys.argv[1]).images.values())
test = [n for n in os.listdir(os.path.join(sys.argv[2], "images")) if n.endswith(".png") and "_covis" not in n]
for n in test:
    os.symlink(os.path.join(sys.argv[3], f"{names.index(n):05d}.png"), os.path.join(sys.argv[4], n))
print(len(test), "generated views")
PY
for mode in none full unobserved; do
  "$CPUPY" scripts/build_distill_set.py --real "$O/train_data" --gen-views "$O/test" --gen-images "$D/gen" \
    --gen-mode "$mode" --holdout-every 8 --out "$D/data_$mode"
done
ONT="$D/data_none_onpath_test"

score() {   # score <ckpt> <tag>
  "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "$O/test" --tag "$2_off" --json-out "$REP/dab_$2_off.json" | tail -1
  "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "$ONT" --tag "$2_on" --json-out "$REP/dab_$2_on.json" | tail -1
}
declare -A ARM=([A]=none [B]=full [C]=unobserved)
for a in A B C; do
  RUN="$D/arm_$a"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    echo "== arm $a (${ARM[$a]})"
    ( cd "$DEPTH_DIR" && "$GPY" train.py --config-name "$APP" path="$D/data_${ARM[$a]}" out_dir="$RUN" experiment_name=run \
        dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=100000 \
        loss.use_depth=true loss.lambda_depth="$LAMBDA" ) > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"
  fi
  score "$CK" "$a"
done

echo "== arm D: ArtiFixer3D (existing) rendered at the off- and on-path held-out poses"
for split in off on; do
  V="$O/test"; [ "$split" = on ] && V="$ONT"
  "$CPUPY" scripts/afx_trajectory.py --views "$V" --out "$D/afx_$split" --densify 0 --scale 1.0
done
module purge 2>/dev/null || true; module load cuda/12.8 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$AFX_VENV/bin:$PATH"
cd "$AFX_DIR"
for split in off on; do
  "$APY" -m data_processing.run_artifixer3d --scene_root "$SCENE" --artifixer_frames_dir "$F1/pred" --phases render \
    --render_trajectory_path "$D/afx_$split.json"
done
cd "$REPO"
module purge 2>/dev/null || true; module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
for split in off on; do
  V="$O/test"; [ "$split" = on ] && V="$ONT"
  VR="$(find "$SCENE/artifixer3d" -type d -name renders -path "*afx_$split*" | head -1)"
  rm -rf "$D/afx_${split}_named"; mkdir -p "$D/afx_${split}_named"
  "$CPUPY" - "$VR" "$V" "$D/afx_${split}_named" <<'PY'
import os, shutil, sys
vr, v, out = sys.argv[1:4]
names = sorted(n for n in os.listdir(os.path.join(v, "images")) if n.endswith(".png") and "_covis" not in n)
fr = sorted(f for f in os.listdir(vr) if f.endswith(".png")); assert len(fr) == len(names), (len(fr), len(names))
for f, n in zip(fr, names): shutil.copyfile(os.path.join(vr, f), os.path.join(out, n))
PY
  "$GPY" scripts/eval_views.py --pred-dir "$D/afx_${split}_named" --dataset "$V" --tag "D_$split" --json-out "$REP/dab_D_$split.json" | tail -1
done

"$CPUPY" - "$REP" <<'PY'
import json, os, sys
rep = sys.argv[1]
desc = {"A": "real only", "B": "+ generated, full weight", "C": "+ generated, unobserved pixels only", "D": "ArtiFixer3D (optimistic on-path)"}
rows = {}
for a in "ABCD":
    r = {s: json.load(open(os.path.join(rep, f"dab_{a}_{s}.json"))) for s in ("on", "off")}
    rows[a] = {"arm": desc[a], "on_psnr": r["on"]["psnr"], "on_lpips": r["on"]["lpips"], "off_psnr": r["off"]["psnr"],
               "off_lpips": r["off"]["lpips"], "off_obs": r["off"].get("psnr_observed"), "off_unobs": r["off"].get("psnr_unobserved")}
json.dump(rows, open(os.path.join(rep, "distill_ablation_summary.json"), "w"), indent=2)
print(f"{'arm':44s}{'on PSNR':>9s}{'on LPIPS':>9s}{'off PSNR':>9s}{'off LPIPS':>10s}{'off obs':>8s}{'off unobs':>10s}")
f = lambda x: f"{x:.2f}" if isinstance(x, float) else "-"
for a, r in rows.items():
    print(f"{a+' '+r['arm']:44s}{r['on_psnr']:9.2f}{r['on_lpips']:9.4f}{r['off_psnr']:9.2f}{r['off_lpips']:10.4f}{f(r['off_obs']):>8s}{f(r['off_unobs']):>10s}")
PY
echo "DISTILL_ABLATION_DONE"
