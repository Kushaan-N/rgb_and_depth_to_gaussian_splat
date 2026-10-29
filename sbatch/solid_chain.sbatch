#!/bin/bash
# The CEAR deliverable chain of <...>_clean_ft2_solid3 (the current best, 37.311 live / 37.18 QAT .compressed.ply) for ANY
# training recipe, under a new NAME, one STAGE per job (chain them with -d afterok:<previous>):
#   export  sbatch/export_viewer.sbatch: train (ARM_FLAGS, all frames) + viewer-exact WD fine-tune + LiDAR fill/inpaint +
#           haze clean -> <E>/<NAME>_clean.ply
#   ft2     a second viewer-exact WD fine-tune of the cleaned splat (scripts/finetune_ply_viewer.py, same settings as
#           export stage 2) -> <E>/<NAME>_clean_ft2.ply (+ _refaxes)
#   solid   sbatch/solidify_unseen.sbatch NAME=<NAME>_clean_ft2 TAG=solid (objects, support, build, refit, harm)
#   solid3  sbatch/solidify_unseen.sbatch TAG=solid3 JOINT=1 REFIT_ARGS="--train-zone-m 0.25" BUDGET_ARGS="--min-w 0.002
#           --iters 12" -> <E>/<NAME>_clean_ft2_solid3{,_refaxes}.ply
#   qat     sbatch/qat_view.sbatch on the solid3 _refaxes export (QITERS 8000) -> <E>/..._solid3_refaxes_qat.compressed.ply
#   score   live SuperSplat (black, grey) at $P/onpath/test of the solid3 PLY (+DISTS), gated vs REF (default the current
#           deliverable's $P/qat/report/deliv_{black,grey}.json); the QAT .compressed.ply vs REF_QAT
#           ($P/qat/report/deliv_qat8k_cply_{black,grey}.json). onpath/test is IN-SAMPLE for an all-frames export.
#   for s in export ft2 solid solid3 qat score; do J=$(STAGE=$s NAME=viewer_fa60k_wd ARM_FLAGS="..." \
#     scripts/sub.sh -A ... ${J:+-d afterok:$J} sbatch/solid_chain.sbatch configs/mocap2_well-lit_trot.yaml); done
# Env: STAGE, NAME, ARM_FLAGS (default "--strategy improved --budget 4000000 --iters 60000"), FT_ARGS (WD:
#      "--lambda-wd 0.025 --wd-log2-sigma 4"), DSCALE (1.5:4), FT_ITERS (7000), QITERS (8000), REF, REF_QAT.
#SBATCH -J solid-chain
#SBATCH -p gpu,gpu-preempt
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 06:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"
export NAME="${NAME:?NAME required}" ARM_FLAGS="${ARM_FLAGS:---strategy improved --budget 4000000 --iters 60000}"
export FT_ARGS="${FT_ARGS:---lambda-wd 0.025 --wd-log2-sigma 4}"
F2="${NAME}_clean_ft2"; S3="$E/${F2}_solid3.ply"; S3R="$E/${F2}_solid3_refaxes.ply"; R="$P/solid_chain/$NAME"; mkdir -p "$R"
hostname; date; echo "== solid_chain $NAME: STAGE=${STAGE:?}"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

case "$STAGE" in
  export) CLEAN=1 QAT=0 bash sbatch/export_viewer.sbatch "$CFG" ;;
  ft2)
    [ -f "$E/$F2.ply" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$E/${NAME}_clean.ply" --out "$E/$F2.ply" \
      --dataset "$DATA" --test-split-interval 0 --viewer ss352 --display-scale "${DSCALE:-1.5:4}" --iters "${FT_ITERS:-7000}" \
      --eval-dataset "$TEST" $FT_ARGS > "$R/ft2.log" 2>&1 || { tail -40 "$R/ft2.log"; exit 1; }
    grep -E "eval \{" "$R/ft2.log" | sed -n '1p;$p'
    [ -z "$FRAME" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
      --in-ply "$E/$F2.ply" --out-ply "$E/${F2}_refaxes.ply" ;;
  solid)  NAME="$F2" IN="$E/$F2.ply" TAG=solid bash sbatch/solidify_unseen.sbatch "$CFG" ;;
  solid3) NAME="$F2" IN="$E/$F2.ply" TAG=solid3 JOINT=1 REFIT_ARGS="--train-zone-m 0.25" BUDGET_ARGS="--min-w 0.002 --iters 12" \
            bash sbatch/solidify_unseen.sbatch "$CFG" ;;
  qat)
    V="$S3R"; [ -f "$V" ] || V="$S3"
    NAME="$NAME" VIEW_PLY="$V" SRC_PLY="$S3" TRAIN="$DATA" TESTS="onpath=$TEST" QITERS="${QITERS:-8000}" OUT="$R/qat" \
      bash sbatch/qat_view.sbatch 2>&1 | grep -E "GATE|eval \{|onpath|QAT_VIEW_DONE|rror" || echo "QAT stage FAILED" ;;
  score)
    REF="${REF:-$P/qat/report/deliv}"; REF_QAT="${REF_QAT:-$P/qat/report/deliv_qat8k_cply}"; REP="$R/report"; mkdir -p "$REP"
    [ -f "$R/shots/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black grey --ply "$S3" --dataset "$TEST" \
      --out-dir "$R/shots" | tail -1
    V="$S3R"; [ -f "$V" ] || V="$S3"; Q="${V%.ply}_qat.compressed.ply"; [ -f "$Q" ] || Q=""
    [ -z "$Q" ] || [ -f "$R/shots_qat/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black grey --ply "$Q" \
      --dataset "$TEST" --out-dir "$R/shots_qat" | tail -1
    for v in "solid3:$R/shots" "solid3_qat_cply:$R/shots_qat"; do
      t="${v%%:*}"; d="${v#*:}"; [ -d "$d/black" ] || continue
      for bg in black grey; do
        [ -f "$REP/${t}_$bg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$d/$bg" --dataset "$TEST" --tag "${t}_$bg" \
          --json-out "$REP/${t}_$bg.json" | tail -1
        PYTHONPATH="$PIQ_LIB" "$GPY" - "$d/$bg" "$TEST/images" "$REP/${t}_$bg.json" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
if "dists" not in r:
    t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
    m = piq.DISTS(reduction="none").cuda(); v = []
    for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
        with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
    r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2)
print("SCORE", json.dumps(r))
PY
      done
    done
    "$CPUPY" scripts/accept_gate.py --pair black "${REF}_black.json" "$REP/solid3_black.json" --pair grey "${REF}_grey.json" \
      "$REP/solid3_grey.json" --out "$REP/gate_solid3_vs_ref.json" | tail -1 || true
    [ -f "$REP/solid3_qat_cply_grey.json" ] && { "$CPUPY" scripts/accept_gate.py --pair black "${REF_QAT}_black.json" \
      "$REP/solid3_qat_cply_black.json" --pair grey "${REF_QAT}_grey.json" "$REP/solid3_qat_cply_grey.json" \
      --out "$REP/gate_qat_vs_ref.json" | tail -1 || true; }
    ls -la "$E/$F2"*.ply 2>/dev/null || true ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac
echo "SOLID_CHAIN_DONE $STAGE"
