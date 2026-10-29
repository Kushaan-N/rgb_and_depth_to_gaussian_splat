#!/bin/bash
# Inpainted appearance for the unseen-surface fill (scripts/fill_inpaint.py): the export's splat <name>.ply (as
# export_igs.sbatch: no floor infill) + its fill discs (<name>_complete.ply, fill_unseen_surfaces.py, made here if missing) ->
# <name>_inpainted.ply (+ _refaxes in pipeline.export_frame_model's axes). Checks, all with SuperSplat's engine
# (viewer_shots.py): (1) colour round trip rgb = 0.5 + C0 dc, (2) off-path views (MOVE, every EVERY_OFF) + editor
# view, complete vs inpainted, montage, (3) on-path test views (onpath/test) scored and pixel-diffed
# base / complete / inpainted (the fill must not change real views).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/fill-inpaint-%j.out sbatch/fill_inpaint.sbatch configs/<seq>.yaml
# Env: NAME (edgs_igs), BASE (exports/<NAME>.ply), ARM (igs_full/<ARM>_depth: edgs_igs), METHOD (lama|shiftmap|telea), MOVE ("0 -0.6 1.5"),
#      EVERY_OFF (40), TAG (inpainted), ENGINE (current = live SuperSplat on WebGPU | pc274), BGS ("black grey"), SMOKE=1 (round trip + inpaint + debug textures only).
#SBATCH -J fill-inpaint
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model)"
NAME="${NAME:-edgs_igs}"; ARM="${ARM:-edgs_igs}"; METHOD="${METHOD:-lama}"; TAG="${TAG:-inpainted}"
MOVE="${MOVE:-0 -0.6 1.5}"; EVERY_OFF="${EVERY_OFF:-40}"; BGS="${BGS:-black grey}"; BG0="${BGS%% *}"
VS=(scripts/viewer_shots.py --engine "${ENGINE:-current}")
BASE="${BASE:-$E/$NAME.ply}"; S="$P/fill_inpaint/$NAME"; RT="$P/viewer_bench/reference_dataset"; DATA="$P/onpath/data_orig"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$S"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash env/make_inpaint_venv.sh; bash env/make_viewer_venv.sh
[ -f "$BASE" ] || { echo "missing $BASE (run export_igs.sbatch)"; exit 1; }
[ -f "$RT/sparse/0/images.txt" ] || { echo "missing $RT (run supersplat_check.sbatch)"; exit 1; }

# (1) does SuperSplat's engine show rgb = 0.5 + C0 * f_dc ?
"$IPY" scripts/fill_inpaint.py --roundtrip-make "$S/roundtrip" --dataset "$RT"
"$VPY" "${VS[@]}" --bg black --ply "$S/roundtrip/roundtrip.ply" --dataset "$RT" --every 100000 --out-dir "$S/roundtrip/shots"
"$IPY" scripts/fill_inpaint.py --roundtrip-make "$S/roundtrip" --roundtrip-check "$S/roundtrip/shots"

# fill discs (as export_igs.sbatch), then their inpainted colours
if [ ! -f "$E/${NAME}_complete.ply" ]; then
  CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
  "$CPUPY" scripts/fill_unseen_surfaces.py --splat-ply "$BASE" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
    --dataset "$DATA" --splat-depth-dir "$P/igs_full/${ARM}_depth" --out-ply "$E/${NAME}_complete.ply"
fi
"$IPY" scripts/fill_inpaint.py --splat-ply "$BASE" --fill-ply "$E/${NAME}_complete.ply" \
  --sim3-json "$P/sim3.json" --method "$METHOD" --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" \
  --out-ply "$E/${NAME}_${TAG}.ply" --debug-dir "$S/debug_$TAG"
[ "${SMOKE:-0}" = 1 ] && { echo FILL_INPAINT_SMOKE_DONE; exit 0; }       # look at $S/debug_$TAG first
for v in complete "$TAG"; do
  [ -f "$E/${NAME}_${v}_refaxes.ply" ] && [ "$E/${NAME}_${v}_refaxes.ply" -nt "$E/${NAME}_$v.ply" ] && continue
  "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
    --in-ply "$E/${NAME}_$v.ply" --out-ply "$E/${NAME}_${v}_refaxes.ply"
done

# (2) off the recorded path + the editor's default view (reference axes)
for v in complete "$TAG"; do
  "$VPY" "${VS[@]}" --bg $BGS --ply "$E/${NAME}_${v}_refaxes.ply" --dataset "$RT" --every "$EVERY_OFF" --move $MOVE \
    --out-dir "$S/offpath/$v" --editor
done
# (3) held-out on-path views (pipeline frame): scores + pixel difference vs the splat without the unseen fill
for v in base complete "$TAG"; do
  ply="$E/${NAME}_$v.ply"; [ "$v" = base ] && ply="$BASE"
  "$VPY" "${VS[@]}" --bg "$BG0" --ply "$ply" --dataset "$P/onpath/test" --out-dir "$S/onpath/$v"
  "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$v" --dataset "$P/onpath/test" --tag "$v" --json-out "$S/onpath_$v.json" | tail -1
done

"$CPUPY" - "$S" "$TAG" $BGS <<'PY'
import glob, json, os, sys, cv2, numpy as np
S, tag, bgs = sys.argv[1], sys.argv[2], sys.argv[3:]; K = ("complete", tag)
def d(k, bg):                                                         # one --bg writes straight into out-dir
    return os.path.join(S, "offpath", k, bg) if len(bgs) > 1 else os.path.join(S, "offpath", k)
for bg in bgs:
    names = sorted(f for f in os.listdir(d(K[0], bg)) if f.endswith(".png") and not f.startswith("_"))
    rows = []
    for n in names:
        ims = [cv2.resize(cv2.imread(os.path.join(d(k, bg), n)), (560, 420)) for k in K]
        for x, l in zip(ims, ("blend (complete)", "inpainted")): cv2.putText(x, l, (8, 26), 0, 0.8, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    ed = [glob.glob(os.path.join(S, "offpath", k, "**", "_supersplat_editor*.png"), recursive=True) for k in K]
    if all(ed): rows.append(np.hstack([cv2.resize(cv2.imread(sorted(e)[0]), (560, 350)) for e in ed]))
    out = os.path.join(S, f"offpath_montage_{bg}.jpg"); cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
res = {}
for v in ("base", "complete", tag):
    r = json.load(open(os.path.join(S, f"onpath_{v}.json"))); res[v] = {k: r[k] for k in ("psnr", "ssim", "lpips")}
on = sorted(f for f in os.listdir(os.path.join(S, "onpath", tag)) if f.endswith(".png") and not f.startswith("_"))
for a, b in (("complete", tag), ("base", tag)):
    d = [np.abs(cv2.imread(os.path.join(S, "onpath", a, n)).astype(int) - cv2.imread(os.path.join(S, "onpath", b, n)).astype(int)) for n in on]
    res[f"diff_{a}_vs_{tag}"] = {"max": int(max(x.max() for x in d)), "mean": round(float(np.mean([x.mean() for x in d])), 5),
                                 "frac_px_gt8": round(float(np.mean([(x.max(2) > 8).mean() for x in d])), 6)}
print("FILL_INPAINT_CHECK", json.dumps(res)); json.dump(res, open(os.path.join(S, "check.json"), "w"), indent=1)
PY
echo FILL_INPAINT_DONE
