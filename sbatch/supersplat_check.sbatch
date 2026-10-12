#!/bin/bash
# What SuperSplat shows, measured: each candidate's exported .ply rendered by SuperSplat's own engine (PlayCanvas
# gsplat in headless Chromium, scripts/viewer_shots.py) (1) at the on-path held-out cameras (every 8th frame; no
# candidate trained on them) and scored against the real images (eval_views.py --pred-dir), (2) from cameras
# moved off the path (MOVE), and (3) as a screenshot of the SuperSplat editor's default view. Candidates are
# compared in the viewer reference's axes (pipeline.export_frame_model) next to the reference itself.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/supersplat_check.sbatch configs/<seq>.yaml
# Env: SMOKE=1 (reference only, 4 views), MOVE ("0 -0.3 1.0"), EVERY_OFF (40).
#SBATCH -J supersplat-check
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VB="$P/viewer_bench"; S="$P/supersplat_check"; REP="$S/report"
REF_PLY="$(get pipeline.viewer_reference_ply)"; REF_DS="$(get pipeline.viewer_reference_dataset)"
FRAME="$(get pipeline.export_frame_model)"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
bash env/make_viewer_venv.sh
RT="$VB/reference_dataset"                                     # text-model view of the reference reconstruction
if [ ! -f "$RT/sparse/0/images.txt" ]; then
  mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
  "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
fi

if [ "${SMOKE:-0}" = 1 ]; then
  "$VPY" scripts/viewer_shots.py --ply "$REF_PLY" --dataset "$RT" --names-from "$P/onpath/test" --every 13 \
    --out-dir "$S/smoke" --editor
  echo SUPERSPLAT_SMOKE_DONE; exit 0
fi

# held-out models (viewer_bench.sbatch exported them, pipeline frame) and the reference
CANDS="reference"
"$VPY" scripts/viewer_shots.py --ply "$REF_PLY" --dataset "$RT" --names-from "$P/onpath/test" --out-dir "$S/onpath/reference" --editor
for n in lidar_init lidar_init_depth igs igs_depth; do
  [ -f "$VB/$n.ply" ] || continue
  "$VPY" scripts/viewer_shots.py --ply "$VB/$n.ply" --dataset "$P/onpath/test" --out-dir "$S/onpath/$n"
  [ -f "$VB/${n}_refaxes.ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
    --gt-model "$FRAME" --in-ply "$VB/$n.ply" --out-ply "$VB/${n}_refaxes.ply"
  CANDS="$CANDS $n"
done
for n in $CANDS; do
  # same real images for every candidate (the reference's held-out renders carry the same names)
  "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$n" --dataset "$P/onpath/test" --tag "$n" --json-out "$REP/$n.json" | tail -1
  ply="$VB/${n}_refaxes.ply"; [ "$n" = reference ] && ply="$REF_PLY"
  "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$RT" --every "$EVERY_OFF" --move $MOVE --out-dir "$S/offpath/$n" \
    $([ "$n" != reference ] && echo --editor)
done

"$CPUPY" - "$S" "$REP" "$P/onpath/test/images" $CANDS <<'PY'
import json, os, sys, cv2, numpy as np
S, rep, gt, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
print(f"{'splat':18s}{'SuperSplat-engine PSNR':>24s}{'SSIM':>8s}{'LPIPS':>8s}")
res = {}
for t in tags:
    f = os.path.join(rep, t + ".json")
    if os.path.exists(f):
        r = json.load(open(f)); res[t] = r; print(f"{t:18s}{r['psnr']:24.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}")
json.dump(res, open(os.path.join(rep, "summary.json"), "w"), indent=2)
def montage(sub, names, first_gt, out):
    rows = []
    for nm in names:
        ims, labs = ([cv2.imread(os.path.join(gt, nm))], ["real"]) if first_gt else ([], [])
        for t in tags:
            p = os.path.join(S, sub, t, nm)
            if os.path.exists(p): ims.append(cv2.imread(p)); labs.append(t)
        ims = [cv2.resize(x, (400, 300)) for x in ims]
        for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.7, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
on = sorted(f for f in os.listdir(os.path.join(S, "onpath", tags[0])) if not f.startswith("_"))
montage("onpath", on[::11][:5], True, os.path.join(S, "onpath_montage.jpg"))
off = sorted(f for f in os.listdir(os.path.join(S, "offpath", tags[0])) if not f.startswith("_"))
montage("offpath", off[:6], False, os.path.join(S, "offpath_montage.jpg"))
eds = [cv2.resize(cv2.imread(p), (640, 400)) for p in [os.path.join(S, d, t, "_supersplat_editor.png") for t in tags
       for d in ("onpath", "offpath")] if os.path.exists(p)]
if eds: cv2.imwrite(os.path.join(S, "editor_montage.jpg"), np.vstack(eds)); print("editor montage ->", os.path.join(S, "editor_montage.jpg"))
PY
echo SUPERSPLAT_CHECK_DONE
