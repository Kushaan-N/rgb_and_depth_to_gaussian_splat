#!/bin/bash
# Quantisation-aware final fine-tune of a delivered viewer-frame PLY (<x>_view.ply from scripts/view_frame.py) for
# splat-transform's .compressed.ply (scripts/finetune_ply_viewer.py finetune --quant-format cply; see sbatch/qat_bench.sbatch
# for the CEAR study). The quantisation depends on the frame (Morton chunks, SH rotation), so the fine-tune runs IN the
# view frame: the training / test COLMAP models are moved by the view PLY's T_new_old (<x>_view.json, rigid or a sim3,
# e.g. a *_refaxes.ply with a json from `fit_sim3` below; images, masks and depth symlinked), then
#   <x>_view.ply -> <x>_view_qat.ply (float, kept for reference) -> <x>_view_qat.compressed.ply (the file to deliver)
# and both <x>_view.compressed.ply and <x>_view_qat.compressed.ply are scored in the live SuperSplat at every test set
# (black bg; *_static too where the test images carry *_mask.png person masks), gated (qat vs previous compressed, and
# vs the uncompressed view PLY), plus off-path moved views and the editor default view.
#   NAME=fiord VIEW_PLY=<exports/x_view.ply> TRAIN=<orig-frame train ds> TESTS="rect=<orig-frame test ds>" \
#     WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu --gres=gpu:1 --constraint="a100|a40|l40s" --exclude=gpu010 \
#     -o logs/%x-%j.out sbatch/qat_view.sbatch
# Env: SRC_PLY (the PLY in TRAIN's frame that VIEW_PLY is a rigid / sim3 copy of: the json is fitted by
#      scripts/fit_sim3_ply.py when missing or older than VIEW_PLY; otherwise <x>_view.json must exist),
#      TSI (0: hold-out interval within TRAIN), MASKS (1: --mask-dir <train>/images when *_mask.png exist),
#      MASK_DIR (a <stem>_mask.png dir used instead, e.g. <train>/masks_persist from fusionportable_masks.py persist; unset = as before), QITERS (3000),
#      FT_ARGS ("--lambda-wd 0.025 --wd-log2-sigma 4"), DSCALE (1.5:4), QARGS (extra), OUT (<view dir>/qat_<stem>),
#      MOVE ("0 -0.3 1.0"), EVERY_OFF (20)
#SBATCH -J qat-view
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 03:00:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
NODE_BIN="$(ls -d "$HOME"/.nvm/versions/node/v2*/bin 2>/dev/null | sort -V | tail -1)"; export PATH="$NODE_BIN:$PATH"
export SPLAT_TRANSFORM="${SPLAT_TRANSFORM:-$CEAR_WS/tools/splat-transform/node_modules/.bin/splat-transform}"
: "${NAME:?}" "${VIEW_PLY:?}" "${TRAIN:?}" "${TESTS:?}"
stem="$(basename "$VIEW_PLY" .ply)"; VD="$(dirname "$VIEW_PLY")"; O="${OUT:-$VD/qat_$stem}"; REP="$O/report"
QITERS="${QITERS:-3000}"; FT_ARGS="${FT_ARGS:---lambda-wd 0.025 --wd-log2-sigma 4}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-20}"
QPLY="$VD/${stem}_qat.ply"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
bash env/make_viewer_venv.sh | tail -1

move_ds() {  # <orig-frame COLMAP ds> <out ds>: cameras moved by T_new_old of the view PLY; images dir symlinked
  "$CPUPY" - "${VIEW_PLY%.ply}.json" "$1" "$2" <<'PY'
import json, os, sys, shutil, numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "scripts"); from view_frame import read_cams
T = np.array(json.load(open(sys.argv[1]))["T_new_old"]); sc = float(np.cbrt(np.linalg.det(T[:3, :3]))); R = T[:3, :3] / sc  # rigid or sim3
src, dst = sys.argv[2], sys.argv[3]; s0 = os.path.join(dst, "sparse", "0"); shutil.rmtree(dst, ignore_errors=True); os.makedirs(s0)
shutil.copy(os.path.join(src, "sparse", "0", "cameras.txt"), s0); open(os.path.join(s0, "points3D.txt"), "w").close()
with open(os.path.join(s0, "images.txt"), "w") as f:
    for p, Rcw, t in read_cams(src):
        Rn = Rcw @ R.T; Cn = sc * R @ (-Rcw.T @ t) + T[:3, 3]; tn = -Rn @ Cn; x, y, z, w = Rotation.from_matrix(Rn).as_quat()
        f.write(" ".join([p[0], *map(str, (w, x, y, z, *tn)), p[8], p[9]]) + "\n\n")
os.symlink(os.path.realpath(os.path.join(src, "images")), os.path.join(dst, "images"))
print(f"[qat-view] {src} -> {dst} (view frame)")
PY
}
cply() { [ -f "${1%.ply}.compressed.ply" ] || "$CPUPY" scripts/compress_splat.py convert --ply "$1" --out-dir "$(dirname "$1")" \
  --formats compressed.ply --gpu cpu | tail -1; }

if [ -n "${SRC_PLY:-}" ] && { [ ! -f "${VIEW_PLY%.ply}.json" ] || [ "$VIEW_PLY" -nt "${VIEW_PLY%.ply}.json" ]; }; then
  "$CPUPY" scripts/fit_sim3_ply.py --src "$SRC_PLY" --dst "$VIEW_PLY"     # T_new_old of the delivered copy
fi
[ -f "${VIEW_PLY%.ply}.json" ] || { echo "no ${VIEW_PLY%.ply}.json (T_new_old): set SRC_PLY=<the frame of TRAIN's PLY>"; exit 1; }
echo "== view-frame datasets"
[ -f "$O/ds/train/sparse/0/images.txt" ] || move_ds "$TRAIN" "$O/ds/train"
TNAMES=""
for spec in $TESTS; do t="${spec%%=*}"; TNAMES="$TNAMES $t"; [ -f "$O/ds/$t/sparse/0/images.txt" ] || move_ds "${spec#*=}" "$O/ds/$t"; done
first="$(echo $TNAMES | cut -d' ' -f1)"
MK=(); if [ "${MASKS:-1}" = 1 ] && ls "$TRAIN"/images/*_mask.png >/dev/null 2>&1; then MK=(--mask-dir "$O/ds/train/images"); fi
if [ -n "${MASK_DIR:-}" ]; then                     # masks are image-space (<stem>_mask.png): any dir works in the view frame
  ls "$MASK_DIR"/*_mask.png >/dev/null 2>&1 || { echo "MASK_DIR=$MASK_DIR has no *_mask.png"; exit 1; }
  MK=(--mask-dir "$MASK_DIR")
fi

echo "== QAT fine-tune ($QITERS steps, $FT_ARGS ${MK[*]})"
[ -f "$QPLY" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$VIEW_PLY" --out "$QPLY" --dataset "$O/ds/train" \
  --test-split-interval "${TSI:-0}" --viewer ss352 --display-scale "${DSCALE:-1.5:4}" --eval-display-scale 1.8 --iters "$QITERS" \
  --eval-dataset "$O/ds/$first" --eval-every-frame 4 --eval-interval 1000 --log-every 500 --quant-format cply "${MK[@]}" \
  $FT_ARGS ${QARGS:-} > "$O/qat_ft.log" 2>&1 || { tail -30 "$O/qat_ft.log"; exit 1; }
grep -E "eval \{" "$O/qat_ft.log" | sed -n '1p;$p'
cply "$VIEW_PLY"; cply "$QPLY"

echo "== live SuperSplat scores"
for t in $TNAMES; do
  ds="$O/ds/$t"; masks=0; ls "$ds"/images/*_mask.png >/dev/null 2>&1 && masks=1
  for vs in "orig=$VIEW_PLY" "cply=${VIEW_PLY%.ply}.compressed.ply" "qat_cply=${QPLY%.ply}.compressed.ply"; do
    v="${vs%%=*}"; f="${vs#*=}"; d="$O/shots/$t/$v"
    [ -f "$d/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$f" --dataset "$ds" --out-dir "$d" 2>&1 | tail -1
    [ -f "$REP/${t}_$v.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$ds" --tag "${t}_$v" --json-out "$REP/${t}_$v.json" | tail -1
    if [ "$masks" = 1 ]; then
      "$CPUPY" scripts/datasets/fusionportable_masks.py composite --dataset "$ds" --pred-dir "$d" --out-dir "${d}_static" >/dev/null
      [ -f "$REP/${t}_static_$v.json" ] || "$GPY" scripts/eval_views.py --pred-dir "${d}_static" --dataset "$ds" --tag "${t}_static_$v" \
        --json-out "$REP/${t}_static_$v.json" | tail -1
    fi
  done
done
for vs in "cply=${VIEW_PLY%.ply}.compressed.ply" "qat_cply=${QPLY%.ply}.compressed.ply"; do
  v="${vs%%=*}"; f="${vs#*=}"
  [ -f "$O/offpath/$v/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$f" --dataset "$O/ds/$first" \
    --every "$EVERY_OFF" --move $MOVE --out-dir "$O/offpath/$v" --editor 2>&1 | tail -1
done

echo "== gates"
gate() {  # name ref cand [args]
  local n="$1" a="$2" b="$3" P=(); shift 3
  for f in "$REP"/*_"$a".json; do s="$(basename "$f" "_$a.json")"; [ -f "$REP/${s}_$b.json" ] && P+=(--pair "$s" "$f" "$REP/${s}_$b.json"); done
  "$CPUPY" scripts/accept_gate.py "${P[@]}" "$@" --out "$REP/gate_$n.json" > /dev/null && echo "GATE $n: ACCEPT" || echo "GATE $n: reject"
}
gate qat_cply_vs_cply cply qat_cply; gate cply_vs_orig orig cply; gate qat_cply_vs_orig orig qat_cply
"$CPUPY" - "$REP" "$NAME" <<'PY'
import json, os, sys
rep, name = sys.argv[1:3]; rows = {}
for f in sorted(os.listdir(rep)):
    if f.endswith(".json") and not f.startswith("gate_"):
        j = json.load(open(os.path.join(rep, f))); rows[f[:-5]] = {k: j[k] for k in ("psnr", "ssim", "lpips")}
for k, v in rows.items():
    print(f"{name:10s} {k:28s} {v['psnr']:.3f} {v['ssim']:.4f} {v['lpips']:.4f}")
json.dump(rows, open(os.path.join(rep, "summary.json"), "w"), indent=1)
PY
ls -la "${QPLY%.ply}"*
echo QAT_VIEW_DONE
