#!/bin/bash
# Side-by-side "user-like" views of several exported splats in the LIVE SuperSplat (no ground truth; for looking at):
# the elevated orbit cameras of scripts/offpath_bench_prep.py (<EYE>/sparse/0: 1.2-1.8 m high, pitched down 30-45 deg at
# the room centre; PLYs in the pipeline / COLMAP frame) and the editor's default view of each PLY's delivered-axes copy
# (<ply>_refaxes.ply when it exists). -> <OUT>/<label>/{eyeball_NN.png,editor/_supersplat_editor.png}, <OUT>/compare.jpg
# (one row per view, one column per PLY, editor row last).
#   PLYS="old=<E>/a.ply new=<E>/b.ply" OUT=<dir> WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out \
#       sbatch/eyeball_compare.sbatch configs/<seq>.yaml
# Env: PLYS (label=path ...), OUT, EYE ($P/offpath_bench/eyeball), SORT_MODE (unset = the editor's own: radial).
#SBATCH -J eyeball-cmp
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010,gpu035
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:40:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
P="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)/pipeline"; EYE="${EYE:-$P/offpath_bench/eyeball}"
: "${PLYS:?}" "${OUT:?}"; hostname; date; mkdir -p "$OUT"; bash env/make_viewer_venv.sh | tail -1
[ -f "$EYE/sparse/0/images.txt" ] || { echo "missing $EYE (sbatch/offpath_bench.sbatch prep)"; exit 1; }
LABELS=()
for spec in $PLYS; do
  l="${spec%%=*}"; f="${spec#*=}"; LABELS+=("$l"); r="${f%.ply}_refaxes.ply"; [ -f "$r" ] || r="$f"
  [ -f "$OUT/$l/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black ${SORT_MODE:+--sort-mode $SORT_MODE} \
    --ply "$f" --dataset "$EYE" --out-dir "$OUT/$l" 2>&1 | tail -1
  [ -f "$OUT/$l/editor/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$r" --dataset "$EYE" \
    --every 1000 --editor --out-dir "$OUT/$l/editor" 2>&1 | tail -1
done
"$CPUPY" - "$OUT" "${LABELS[@]}" <<'PY'
import os, sys
from PIL import Image, ImageDraw
out, labels = sys.argv[1], sys.argv[2:]
names = sorted(f for f in os.listdir(os.path.join(out, labels[0])) if f.endswith(".png") and not f.startswith("_"))
rows = [[Image.open(os.path.join(out, l, n)).convert("RGB") for l in labels] for n in names]
w, h = rows[0][0].size
ed = [os.path.join(out, l, "editor", "_supersplat_editor.png") for l in labels]
if all(os.path.exists(e) for e in ed):
    rows.append([Image.open(e).convert("RGB").resize((w, h)) for e in ed])
M = Image.new("RGB", (w * len(labels), h * len(rows)))
for i, r in enumerate(rows):
    for j, im in enumerate(r):
        M.paste(im, (j * w, i * h))
d = ImageDraw.Draw(M)
for j, l in enumerate(labels):
    d.text((j * w + 6, 4), l, fill=(255, 255, 0))
M.save(os.path.join(out, "compare.jpg"), quality=90); print("[eyeball] ->", os.path.join(out, "compare.jpg"))
PY
echo EYEBALL_COMPARE_DONE
