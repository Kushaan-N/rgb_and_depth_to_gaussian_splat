#!/bin/bash
# Fill v3 (scripts/fill_texture_v3.py): the unseen-surface fill discs (<name>_complete.ply) textured from the real
# training images (rectified, median-composited, colour-matched to the splat), unseen remainder by line continuation
# + LaMa, per-patch fallback to <name>_inpainted.ply (fill_inpaint.sbatch) -> <name>_fillv3.ply (+ _refaxes).
# Checks with the live SuperSplat editor (viewer_shots.py --engine current), v2 (inpainted) vs v3, own folder
# $P/fill_texture_v3/<name>: (1) off-path views for each MOVES entry (every EVERY_OFF) + editor default view,
# (2) oblique per-patch crop cameras (base | v2 | v3), (3) on-path test views scored (eval_views --pred-dir),
# pixel-diffed vs the base splat, accept_gate base vs v3 (real views must not change).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/fill-v3-%j.out sbatch/fill_texture_v3.sbatch configs/<seq>.yaml
# Env: NAME (edgs_igs), ARM (edgs_igs: igs_full/<ARM>_depth), PREV (inpainted), TAG (fillv3), BGS ("black grey"),
#      MOVES ("0 -0.6 1.5;0 -0.3 1.0"), EVERY_OFF (40), SMOKE=1 (first MAXP=15 patches, debug textures only),
#      V3_ARGS (extra fill_texture_v3.py flags).
#SBATCH -J fill-v3
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model)"
NAME="${NAME:-edgs_igs}"; ARM="${ARM:-edgs_igs}"; PREV="${PREV:-inpainted}"; TAG="${TAG:-fillv3}"
MOVES="${MOVES:-0 -0.6 1.5;0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"; BGS="${BGS:-black grey}"; BG0="${BGS%% *}"
VS=(scripts/viewer_shots.py --engine "${ENGINE:-current}")
BASE="$E/$NAME.ply"; S="$P/fill_texture_v3/$NAME"; RT="$P/viewer_bench/reference_dataset"; DATA="$P/onpath/data_orig"
DEPTH="$P/igs_full/${ARM}_depth"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$S"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash env/make_inpaint_venv.sh; bash env/make_viewer_venv.sh
for f in "$BASE" "$E/${NAME}_complete.ply" "$E/${NAME}_${PREV}.ply" "$RT/sparse/0/images.txt"; do
  [ -f "$f" ] || { echo "missing $f (run fill_inpaint.sbatch / supersplat_check.sbatch)"; exit 1; }
done
DARGS=(); [ -d "$DEPTH" ] && DARGS=(--splat-depth-dir "$DEPTH")
V3=("$IPY" scripts/fill_texture_v3.py --splat-ply "$BASE" --fill-ply "$E/${NAME}_complete.ply" --prev-ply "$E/${NAME}_${PREV}.ply"
    --sim3-json "$P/sim3.json" --dataset "$DATA" "${DARGS[@]}" --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" ${V3_ARGS:-})
if [ "${SMOKE:-0}" = 1 ]; then
  "${V3[@]}" --max-patches "${MAXP:-15}" --out-ply "$S/smoke.ply" --debug-dir "$S/debug_smoke" --debug-n "${MAXP:-15}"
  echo FILL_V3_SMOKE_DONE; exit 0
fi
"${V3[@]}" --out-ply "$E/${NAME}_${TAG}.ply" --debug-dir "$S/debug" --crops-dataset "$S/crops_dataset"
"$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
  --in-ply "$E/${NAME}_${TAG}.ply" --out-ply "$E/${NAME}_${TAG}_refaxes.ply"

# (1) off the recorded path + the editor's default view (reference axes), v2 vs v3
mi=0; IFS=';' read -ra MV <<< "$MOVES"
for mv in "${MV[@]}"; do
  for v in "$PREV" "$TAG"; do
    "$VPY" "${VS[@]}" --bg $BGS --ply "$E/${NAME}_${v}_refaxes.ply" --dataset "$RT" --every "$EVERY_OFF" --move $mv \
      --out-dir "$S/offpath$mi/$v" $([ "$mi" = 0 ] && echo --editor)
  done
  mi=$((mi + 1))
done
# (2) per-patch oblique crops (pipeline frame): base (no fill) | v2 | v3
for v in base "$PREV" "$TAG"; do
  ply="$E/${NAME}_$v.ply"; [ "$v" = base ] && ply="$BASE"
  "$VPY" "${VS[@]}" --bg "$BG0" --ply "$ply" --dataset "$S/crops_dataset" --out-dir "$S/crops/$v"
done
# (3) held-out on-path views: scores + pixel difference vs the splat without the fill
for v in base "$PREV" "$TAG"; do
  ply="$E/${NAME}_$v.ply"; [ "$v" = base ] && ply="$BASE"
  "$VPY" "${VS[@]}" --bg "$BG0" --ply "$ply" --dataset "$P/onpath/test" --out-dir "$S/onpath/$v"
  "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$v" --dataset "$P/onpath/test" --tag "$v" --json-out "$S/onpath_$v.json" | tail -1
done
"$CPUPY" scripts/accept_gate.py --pair onpath "$S/onpath_base.json" "$S/onpath_$TAG.json" --out "$S/gate_base_vs_$TAG.json" || true

"$CPUPY" - "$S" "$PREV" "$TAG" "$mi" $BGS <<'PY'
import glob, json, os, sys, cv2, numpy as np
S, prev, tag, nm, bgs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]; K = (prev, tag)
for mi in range(nm):
    for bg in bgs:
        d = lambda k: os.path.join(S, f"offpath{mi}", k, bg) if len(bgs) > 1 else os.path.join(S, f"offpath{mi}", k)
        names = sorted(f for f in os.listdir(d(K[0])) if f.endswith(".png") and not f.startswith("_"))
        rows = []
        for n in names:
            ims = [cv2.resize(cv2.imread(os.path.join(d(k), n)), (560, 420)) for k in K]
            for x, l in zip(ims, ("v2 " + prev, "v3 " + tag)): cv2.putText(x, l, (8, 26), 0, 0.8, (0, 0, 255), 2)
            rows.append(np.hstack(ims))
        ed = [glob.glob(os.path.join(S, f"offpath{mi}", k, "_supersplat_editor*.png")) for k in K]
        if all(ed): rows.append(np.hstack([cv2.resize(cv2.imread(sorted(e)[0]), (560, 350)) for e in ed]))
        cv2.imwrite(os.path.join(S, f"offpath{mi}_montage_{bg}.jpg"), np.vstack(rows))
cn = sorted(f for f in os.listdir(os.path.join(S, "crops", tag)) if f.endswith(".png") and not f.startswith("_"))
rows = []
for n in cn:
    ims = [cv2.imread(os.path.join(S, "crops", k, n)) for k in ("base", prev, tag)]
    for x, l in zip(ims, ("base (no fill) " + n[:-4], "v2 " + prev, "v3 " + tag)): cv2.putText(x, l, (8, 26), 0, 0.7, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
if rows: cv2.imwrite(os.path.join(S, "crops_montage.jpg"), np.vstack(rows))
res = {}
for v in ("base", prev, tag):
    r = json.load(open(os.path.join(S, f"onpath_{v}.json"))); res[v] = {k: r[k] for k in ("psnr", "ssim", "lpips")}
on = sorted(f for f in os.listdir(os.path.join(S, "onpath", tag)) if f.endswith(".png") and not f.startswith("_"))
for a in ("base", prev):
    dd = [np.abs(cv2.imread(os.path.join(S, "onpath", a, n)).astype(int) - cv2.imread(os.path.join(S, "onpath", tag, n)).astype(int)) for n in on]
    res[f"diff_{a}_vs_{tag}"] = {"max": int(max(x.max() for x in dd)), "mean": round(float(np.mean([x.mean() for x in dd])), 5),
                                 "frac_px_gt8": round(float(np.mean([(x.max(2) > 8).mean() for x in dd])), 6)}
print("FILL_V3_CHECK", json.dumps(res)); json.dump(res, open(os.path.join(S, "check.json"), "w"), indent=1)
PY
echo FILL_V3_DONE
