#!/bin/bash
# What the live SuperSplat (v3.5.x editor, WebGPU, scripts/viewer_shots.py --engine current) shows for FusionPortable
# v2 exports: (1) the held-out on-path frames of EVERY rig camera (<P>/onpath/test: undistorted pinhole, never
# trained on) on black and grey backgrounds, scored against the real images (eval_views.py --pred-dir; alpha-hole
# fraction from _viewer.json); (2) cameras moved off the path (MOVE, camera axes right/down/back in metres) and the
# SuperSplat editor default view, as montages to LOOK at; (3) geometry: splat gaussian centres (opacity > 0.5) vs the
# static LiDAR cloud (no GT laser map exists for the room sequences).
#   PLYS="a=<x.ply> b=<y.ply>" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_check.sbatch configs/fp2_<seq>.yaml
# Env: PLYS (required), MASKS (1: person masks + *_static scores), MOVE ("0 -0.3 1.0"), EVERY_OFF (24), TAGDIR (supersplat_check), FP2_WS
#SBATCH -J fp2-check
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
FP2_WS="${FP2_WS:-/scratch4/workspace/${USER}-multicam}"
export CEAR_DATA="${CEAR_DATA:-$FP2_WS/data/fusionportable}" CEAR_OUT="${CEAR_OUT:-$FP2_WS/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; TEST="${TEST_DATA:-$P/onpath/test}"; S="$P/${TAGDIR:-supersplat_check}"; REP="$S/report"
MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-24}"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
bash env/make_viewer_venv.sh
# transient people (operator / office workers) in the held-out frames: Mask R-CNN masks -> also score the static scene
MPY="${MASK_PY:-$CEAR_WS/venv-edgs/bin/python}"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
[ "${MASKS:-1}" = 1 ] && "$MPY" scripts/datasets/fusionportable_masks.py mask --dataset "$TEST"
NAMES=""
for spec in ${PLYS:?set PLYS=\"name=path ...\"}; do
  n="${spec%%=*}"; ply="${spec#*=}"; [ -f "$ply" ] || { echo "no $ply"; continue; }; NAMES="$NAMES $n"
  [ -f "$S/onpath/$n/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --ply "$ply" --dataset "$TEST" \
    --bg black grey --out-dir "$S/onpath/$n" --editor
  for bg in black grey; do
    "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$n/$bg" --dataset "$TEST" --tag "${n}_$bg" --json-out "$REP/${n}_$bg.json" | tail -1
    if ls "$TEST"/images/*_mask.png >/dev/null 2>&1; then
      "$CPUPY" scripts/datasets/fusionportable_masks.py composite --dataset "$TEST" --pred-dir "$S/onpath/$n/$bg" --out-dir "$S/onpath/$n/${bg}_static"
      "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$n/${bg}_static" --dataset "$TEST" --tag "${n}_${bg}_static" \
        --json-out "$REP/${n}_${bg}_static.json" | tail -1
    fi
  done
  [ -d "$S/offpath/$n" ] || "$VPY" scripts/viewer_shots.py --engine current --ply "$ply" --dataset "$P/dataset" --every "$EVERY_OFF" \
    --move $MOVE --out-dir "$S/offpath/$n"
  "$CPUPY" - "$ply" "$P/lidar_static.ply" "$REP/${n}_geom.json" <<'PY'
import json, sys, numpy as np, open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
v = PlyData.read(sys.argv[1])["vertex"]; op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
X = np.stack([v["x"], v["y"], v["z"]], 1)[op > 0.5]; L = np.asarray(o3d.io.read_point_cloud(sys.argv[2]).points)
d_sl = cKDTree(L).query(X, workers=-1)[0]; d_ls = cKDTree(X).query(L, workers=-1)[0]
r = {"gaussians_opaque": int(len(X)), "splat_to_lidar_median_m": float(np.median(d_sl)), "splat_within_5cm": float((d_sl < 0.05).mean()),
     "lidar_to_splat_median_m": float(np.median(d_ls)), "lidar_covered_10cm": float((d_ls < 0.10).mean())}
json.dump(r, open(sys.argv[3], "w"), indent=1); print("[geom]", r)
PY
done
"$CPUPY" - "$S" "$REP" "$TEST/images" $NAMES <<'PY'
import json, os, sys, cv2, numpy as np
S, rep, gt, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
print(f"{'splat':28s}{'bg':>13s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'hole%':>7s}   per-camera PSNR")
res = {}
for t in tags:
    for bg in ("black", "grey", "black_static", "grey_static"):
        f = os.path.join(rep, f"{t}_{bg}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); vj = os.path.join(S, "onpath", t, "_viewer.json")
        hole = json.load(open(vj)).get("alpha_hole_frac", float("nan")) if os.path.exists(vj) else float("nan")
        per = {}                                       # per rig camera (image names <cam>_<ns>.png)
        for nm in sorted(os.listdir(gt)):
            pr = os.path.join(S, "onpath", t, bg, os.path.splitext(nm)[0] + ".png")
            if nm.endswith(".png") and not nm.endswith("_mask.png") and os.path.exists(pr):
                a = cv2.imread(pr).astype(np.float64); b = cv2.imread(os.path.join(gt, nm)).astype(np.float64)
                per.setdefault(nm.rsplit("_", 1)[0], []).append(10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9)))
        r["psnr_per_camera"] = {c: float(np.mean(v)) for c, v in per.items()}
        pc = " ".join(f"{c}:{np.mean(v):.2f}" for c, v in sorted(per.items()))
        g = os.path.join(rep, f"{t}_geom.json"); r["geom"] = json.load(open(g)) if os.path.exists(g) else None
        r["alpha_hole_frac"] = hole; res[f"{t}_{bg}"] = r
        print(f"{t:28s}{bg:>13s}{r['psnr']:8.2f}{r['ssim']:8.4f}{r['lpips']:8.4f}{100 * hole if hole == hole else float('nan'):7.2f}   {pc}")
json.dump(res, open(os.path.join(rep, "summary.json"), "w"), indent=1)
def montage(sub, names, first_gt, out):
    rows = []
    for nm in names:
        ims, labs = ([cv2.imread(os.path.join(gt, nm))], ["real"]) if first_gt else ([], [])
        for t in tags:
            p = os.path.join(S, sub, t, "black", nm) if sub == "onpath" else os.path.join(S, sub, t, nm)
            if os.path.exists(p): ims.append(cv2.imread(p)); labs.append(t)
        if not ims: continue
        ims = [cv2.resize(x, (400, 300)) for x in ims]
        for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    if rows: cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
on = sorted(f for f in os.listdir(os.path.join(S, "onpath", tags[0], "black")) if f.endswith(".png") and not f.startswith("_"))
montage("onpath", on[:: max(1, len(on) // 6)][:6], True, os.path.join(S, "onpath_montage.jpg"))
off = sorted(f for f in os.listdir(os.path.join(S, "offpath", tags[0])) if f.endswith(".png") and not f.startswith("_"))
montage("offpath", off[:: max(1, len(off) // 6)][:6], False, os.path.join(S, "offpath_montage.jpg"))
eds = [cv2.resize(cv2.imread(p), (640, 400)) for p in [os.path.join(S, "onpath", t, "_supersplat_editor.png") for t in tags] if os.path.exists(p)]
if eds: cv2.imwrite(os.path.join(S, "editor_montage.jpg"), np.vstack(eds)); print("editor montage ->", os.path.join(S, "editor_montage.jpg"))
PY
echo FP2_CHECK_DONE
