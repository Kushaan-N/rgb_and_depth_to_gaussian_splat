#!/bin/bash
# Laptop-sized delivery of a splat: compress PLY -> .sog / .compressed.ply (scripts/compress_splat.py, PlayCanvas
# splat-transform; SOG encoding uses the GPU) and check that it "works properly" in the LIVE SuperSplat (viewer_shots
# --engine current, black bg): original and every compressed file are rendered in the same job at the held-out views
# of each TESTS set (the PLY's frame) and scored against the real images (eval_views --pred-dir; where the test images
# have *_mask.png person masks, also *_static scores); off-path views (--move, first test set) + the editor default
# view; DELIVER files (rigid re-framings of PLY for the user) are converted too and opened once in the editor.
# -> $OUT/check/summary.json (deltas vs original), montage_<set>.jpg (real|orig|variants||diff|x4), montage_offpath.jpg,
#    montage_editor.jpg; compressed files in $OUT.
#   NAME=cear PLY=<a.ply> TESTS="onpath=<ds>" OUT=<dir> [DELIVER="<b.ply> ..."] \
#     WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu --exclude=gpu010 -o logs/%x-%j.out sbatch/compress_check.sbatch
# Env: STEM (compressed file stem; default the PLY stem), ITERS (SOG -i, 10; several -> one .sog each, e.g. "10 50"), FORMATS ("sog compressed.ply"), MOVE ("0 -0.3 1.0"),
#      ST_GPU (splat-transform -g: cpu converts on a CPU node: -p cpu,cpu-preempt --gres=none SKIP_SCORE=1), EVERY_OFF (20), SMOKE=1 (every 13th view, no off-path), SKIP_SCORE=1 (convert only)
#SBATCH -J compress-check
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
MPY="$CEAR_WS/venv-edgs/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
NODE_BIN="$(ls -d "$HOME"/.nvm/versions/node/v2*/bin 2>/dev/null | sort -V | tail -1)"; export PATH="$NODE_BIN:$PATH"
ST="${SPLAT_TRANSFORM:-$CEAR_WS/tools/splat-transform/node_modules/.bin/splat-transform}"; export SPLAT_TRANSFORM="$ST"
: "${NAME:?}" "${PLY:?}" "${TESTS:?}" "${OUT:?}"
ITERS="${ITERS:-10}"; FORMATS="${FORMATS:-sog compressed.ply}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-20}"
C="$OUT/check"; REP="$C/report"; mkdir -p "$REP"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; module list 2>&1 | tail -2 || true
echo "node $(node --version); $("$ST" --version 2>&1 | head -1)"; "$ST" -L 2>&1 | tail -8 || true
bash env/make_viewer_venv.sh >/dev/null

stem="${STEM:-$(basename "$PLY" .ply)}"; VARS="orig=$PLY"
for it in $ITERS; do
  "$CPUPY" scripts/compress_splat.py convert --ply "$PLY" --stem "$stem" --out-dir "$OUT" --formats $FORMATS --iterations "$it" ${ST_GPU:+--gpu $ST_GPU}
done
for f in $FORMATS; do
  if [ "$f" = sog ]; then
    for it in $ITERS; do sx=""; [ "$it" = 10 ] || sx="_i$it"; VARS="$VARS sog$sx=$OUT/$stem$sx.sog"; done
  else VARS="$VARS cply=$OUT/$stem.compressed.ply"; fi
done
for d in ${DELIVER:-}; do "$CPUPY" scripts/compress_splat.py convert --ply "$d" --out-dir "$OUT" --formats $FORMATS --iterations "${ITERS%% *}" ${ST_GPU:+--gpu $ST_GPU}; done
ls -la "$OUT"
[ "${SKIP_SCORE:-0}" = 1 ] && { echo COMPRESS_CHECK_DONE; exit 0; }

STEP=""; [ "${SMOKE:-0}" = 1 ] && STEP="--every 13"
TNAMES=""; first=""
for spec in $TESTS; do
  t="${spec%%=*}"; ds="${spec#*=}"; TNAMES="$TNAMES $t"; [ -n "$first" ] || first="$ds"
  masks=0; ls "$ds"/images/*_mask.png >/dev/null 2>&1 && masks=1
  for vs in $VARS; do
    v="${vs%%=*}"; f="${vs#*=}"; [ -f "$f" ] || { echo "no $f"; continue; }
    [ -f "$C/$t/$v/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$f" --dataset "$ds" \
      --out-dir "$C/$t/$v" $STEP 2>&1 | tail -3
    "$GPY" scripts/eval_views.py --pred-dir "$C/$t/$v" --dataset "$ds" --tag "${t}_$v" --json-out "$REP/${t}_$v.json" | tail -1
    if [ "$masks" = 1 ]; then
      "$CPUPY" scripts/datasets/fusionportable_masks.py composite --dataset "$ds" --pred-dir "$C/$t/$v" --out-dir "$C/$t/${v}_static" >/dev/null
      "$GPY" scripts/eval_views.py --pred-dir "$C/$t/${v}_static" --dataset "$ds" --tag "${t}_${v}_static" \
        --json-out "$REP/${t}_${v}_static.json" | tail -1
    fi
  done
done
if [ "${SMOKE:-0}" != 1 ]; then
  for vs in $VARS; do
    v="${vs%%=*}"; f="${vs#*=}"; [ -f "$f" ] || continue
    [ -f "$C/offpath/$v/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$f" --dataset "$first" \
      --every "$EVERY_OFF" --move $MOVE --out-dir "$C/offpath/$v" --editor 2>&1 | tail -2
  done
fi
EXTRA_ED=""
for d in ${DELIVER:-}; do                      # the user-facing files: do they open in the editor? (default view only)
  ds="$(basename "$d" .ply)"
  for f in $FORMATS; do
    g="$OUT/$ds.$f"; [ -f "$g" ] || continue; e="$C/deliver/$ds.$f"
    [ -f "$e/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$g" --dataset "$first" --every 100000 \
      --out-dir "$e" --editor 2>&1 | tail -2
    EXTRA_ED="$EXTRA_ED $e/_supersplat_editor.png"
  done
done
TSPEC=""; for spec in $TESTS; do TSPEC="$TSPEC $spec"; done
"$CPUPY" scripts/compress_splat.py report --check-dir "$C" --tests $TSPEC --variants $(for vs in $VARS; do printf '%s ' "${vs%%=*}"; done) \
  --extra-editor $EXTRA_ED
cat "$OUT"/*_compress.json
echo COMPRESS_CHECK_DONE
