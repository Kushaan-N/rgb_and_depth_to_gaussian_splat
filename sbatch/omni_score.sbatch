#!/bin/bash
# Fisheye / 360 datasets, scored as SuperSplat shows them: each candidate .ply (in the dataset's model frame) is
# rendered by the LIVE SuperSplat editor (scripts/viewer_shots.py --engine current, WebGPU) at the rectified held-out
# pinhole views (<out>/pipeline/rect/test, written by the dataset adapter; no candidate trained on those frames) on
# black and grey backgrounds and scored against the real (rectified) images (eval_views.py --pred-dir); then from
# cameras moved off the capture path (MOVE, default scaled to the scene: 5% down, 17% back of the median depth in
# meta.json) with the editor's default view. Where the adapter has a `geom` command (TLS ground truth), the opaque
# splat centres are also scored against the scan. Summary (all candidates scored so far, so rect and native runs land
# in one table) -> <out>/pipeline/supersplat/report/summary.json + on-path / off-path / editor montages.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o <log> sbatch/omni_score.sbatch configs/<seq>.yaml name=/path/a.ply [name2=b.ply]
# (also run inline by sbatch/omni_rect.sbatch and sbatch/omni_native.sbatch). Env: MOVE, EVERY_OFF (offpath stride, 7)
#SBATCH -J omni-score
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00

set -euo pipefail
CFG="$1"; shift
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/env/cear_env.sh" ] || REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export CEAR_DATA="${CEAR_DATA:-/scratch4/workspace/${USER}-multicam/data}" CEAR_OUT="${CEAR_OUT:-/scratch4/workspace/${USER}-multicam/outputs}"
source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; TEST="$P/rect/test"; S="$P/supersplat"; REP="$S/report"
ADAPTER="$(get sequence.adapter)"
[ -f "$TEST/sparse/0/images.txt" ] || { echo "missing $TEST (run the adapter's prepare)"; exit 1; }
MD="$("$CPUPY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('median_depth', 3.0))" "$P/meta.json")"
MOVE="${MOVE:-$("$CPUPY" -c "import sys; d=float(sys.argv[1]); print(0, -0.05*d, 0.17*d)" "$MD")}"
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
bash env/make_viewer_venv.sh >/dev/null
bash env/make_e57_venv.sh >/dev/null 2>&1 || echo "[omni] no e57 reader venv: geometry check skipped"
echo "== held-out: $(grep -c '^[0-9]' "$TEST/sparse/0/images.txt") rectified views; off-path move: $MOVE (median depth $MD)"

NAMES=""
for spec in "$@"; do
  n="${spec%%=*}"; ply="${spec#*=}"; [ -f "$ply" ] || { echo "no $ply"; continue; }
  "$VPY" scripts/viewer_shots.py --engine current --ply "$ply" --dataset "$TEST" --out-dir "$S/onpath/$n" --bg black grey
  for bg in black grey; do
    "$GPY" scripts/eval_views.py --pred-dir "$S/onpath/$n/$bg" --dataset "$TEST" --tag "$n" --json-out "$REP/${n}_$bg.json" | tail -1
  done
  "$VPY" scripts/viewer_shots.py --engine current --ply "$ply" --dataset "$TEST" --every "${EVERY_OFF:-7}" --move $MOVE \
    --out-dir "$S/offpath/$n" --editor
  if "$CPUPY" "scripts/datasets/$ADAPTER.py" -h 2>/dev/null | grep -q geom; then
    [ -f "$P/sim3_tls.json" ] || "${E57_VENV:-$CEAR_WS/venv-e57}/bin/python" "scripts/datasets/$ADAPTER.py" align "$CFG" \
      > "$P/align.log" 2>&1 || { tail -5 "$P/align.log"; echo "[omni] scan alignment failed: geometry in the model frame"; }
    "$CPUPY" "scripts/datasets/$ADAPTER.py" geom "$CFG" --ply "$ply" --json-out "$REP/${n}_geom.json" | tail -1 || true
  fi
  NAMES="$NAMES $n"
done

"$CPUPY" - "$S" "$REP" "$TEST/images" $NAMES <<'PY'
import json, os, sys, cv2, numpy as np
S, rep, gt = sys.argv[1], sys.argv[2], sys.argv[3]
tags = sorted(f[:-11] for f in os.listdir(rep) if f.endswith("_black.json"))   # every candidate scored so far
print(f"{'splat':22s}{'PSNR blk':>9s}{'SSIM':>8s}{'LPIPS':>8s}{'PSNR grey':>10s}{'holes on':>9s}{'holes off':>10s}{'geom F@5cm':>11s}")
res = {}
def hole(d):
    f = os.path.join(d, "_viewer.json")
    return json.load(open(f)).get("alpha_hole_frac") if os.path.exists(f) else None
for t in tags:
    r = {bg: json.load(open(os.path.join(rep, f"{t}_{bg}.json"))) for bg in ("black", "grey")
         if os.path.exists(os.path.join(rep, f"{t}_{bg}.json"))}
    g = os.path.join(rep, f"{t}_geom.json"); r["geom"] = json.load(open(g)) if os.path.exists(g) else None
    r["holes_onpath"], r["holes_offpath"] = hole(os.path.join(S, "onpath", t)), hole(os.path.join(S, "offpath", t))
    res[t] = r; b, gr = r.get("black", {}), r.get("grey", {})
    fmt = lambda x, p=4: "-" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))  # noqa: E731
    print(f"{t:22s}{b.get('psnr', 0):9.2f}{b.get('ssim', 0):8.4f}{b.get('lpips', 0):8.4f}{gr.get('psnr', 0):10.2f}"
          f"{fmt(r['holes_onpath']):>9s}{fmt(r['holes_offpath']):>10s}{fmt(r['geom'] and r['geom']['fscore']):>11s}")
json.dump(res, open(os.path.join(rep, "summary.json"), "w"), indent=2)
def montage(sub, names, first_gt, out, bg=""):
    rows = []
    for nm in names:
        ims, labs = ([cv2.imread(os.path.join(gt, nm))], ["real"]) if first_gt else ([], [])
        for t in tags:
            p = os.path.join(S, sub, t, bg, os.path.splitext(nm)[0] + ".png")
            if os.path.exists(p): ims.append(cv2.imread(p)); labs.append(t)
        if not ims: continue
        ims = [cv2.resize(x, (360, int(360 * x.shape[0] / x.shape[1]))) for x in ims]; h = min(x.shape[0] for x in ims)
        ims = [x[:h] for x in ims]
        for x, l in zip(ims, labs): cv2.putText(x, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack(ims))
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows)); print("montage ->", out)
on = sorted(f for f in os.listdir(gt))
montage("onpath", on[:: max(1, len(on) // 6)][:6], True, os.path.join(S, "onpath_montage.jpg"), "black")
off = sorted(f for f in os.listdir(os.path.join(S, "offpath", tags[0])) if f.endswith(".png") and not f.startswith("_"))
montage("offpath", off[:6], False, os.path.join(S, "offpath_montage.jpg"))
eds = [cv2.resize(cv2.imread(p), (640, 400)) for p in [os.path.join(S, "offpath", t, "_supersplat_editor.png") for t in tags]
       if os.path.exists(p)]
if eds: cv2.imwrite(os.path.join(S, "editor_montage.jpg"), np.vstack(eds)); print("editor montage ->", os.path.join(S, "editor_montage.jpg"))
PY
echo OMNI_SCORE_DONE
