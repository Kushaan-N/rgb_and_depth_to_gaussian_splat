#!/bin/bash
# Score splats in the live SuperSplat (>= 3.7.0) in BOTH Sort Modes on any held-out set(s), same session/engine for every
# PLY, so a candidate is compared to the current best rendered in the same build:
#   depth  = view-depth blend order (SuperSplat <= 3.6, and 3.7's Settings > Sort Mode > Depth): the GATE (strict and
#            accept_gate --perceptual), "never worse than today's files"
#   radial = 3.7's default (distance to the camera): reported, gated separately (bonus)
# Per test set, PLY, mode, background: viewer_shots.py --engine current --sort-mode <mode> -> eval_views.py --pred-dir (+DISTS)
#   PLYS="best=<ref.ply> new=<cand.ply> ..." TESTS="test=<COLMAP test dataset> ..." OUT=<dir> \
#     scripts/sub.sh -A ... --gres=gpu:1 ... sbatch/sort_score.sbatch
# The FIRST PLY is the reference. Env: BGS ("black grey"), MODES ("depth radial"). Outputs: <OUT>/shots/<test>/<mode>/<name>/,
# <OUT>/report/<test>_<name>_<mode>_<bg>.json, gate_<name>_<mode>{,_perc}.json (all test sets x backgrounds of that mode),
# summary.md.
#SBATCH -J sort-score
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 03:00:00
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
O="${OUT:?OUT required}"; REP="$O/report"; mkdir -p "$REP"; read -ra BG <<< "${BGS:-black grey}"
read -ra PL <<< "${PLYS:?PLYS required}"; read -ra TS <<< "${TESTS:?TESTS required}"; REF="${PL[0]%%=*}"
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true
bash env/make_viewer_venv.sh | tail -1

for ts in "${TS[@]}"; do t="${ts%%=*}"; ds="${ts#*=}"
  for m in ${MODES:-depth radial}; do
    for pl in "${PL[@]}"; do n="${pl%%=*}"; ply="${pl#*=}"; d="$O/shots/$t/$m/$n"
      [ -f "$d/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --sort-mode "$m" --bg "${BG[@]}" --ply "$ply" \
        --dataset "$ds" --out-dir "$d" | tail -1 || { sleep 60; "$VPY" scripts/viewer_shots.py --engine current --sort-mode "$m" \
        --bg "${BG[@]}" --ply "$ply" --dataset "$ds" --out-dir "$d" | tail -1; }
      for bg in "${BG[@]}"; do
        pd="$d"; [ "${#BG[@]}" -gt 1 ] && pd="$d/$bg"; js="$REP/${t}_${n}_${m}_$bg.json"
        [ -f "$js" ] || "$GPY" scripts/eval_views.py --pred-dir "$pd" --dataset "$ds" --tag "${t}_${n}_${m}_$bg" --json-out "$js" | tail -1
        PYTHONPATH="$PIQ_LIB" "$GPY" - "$pd" "$ds/images" "$js" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
if "dists" not in r:
    t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
    m = piq.DISTS(reduction="none").cuda(); v = []
    g = {os.path.splitext(f)[0]: f for f in os.listdir(gt) if f.lower().endswith((".png", ".jpg", ".jpeg"))
         and not f.endswith(("_mask.png", "_covis.png"))}                    # real frames may be .jpg: match by stem
    for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
        if os.path.splitext(n)[0] in g:
            with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, g[os.path.splitext(n)[0]]))).item())
    r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2)
PY
      done
    done
  done
done

for pl in "${PL[@]:1}"; do n="${pl%%=*}"
  for m in ${MODES:-depth radial}; do
    P=(); for ts in "${TS[@]}"; do t="${ts%%=*}"; for bg in "${BG[@]}"; do
      P+=(--pair "${t}_$bg" "$REP/${t}_${REF}_${m}_$bg.json" "$REP/${t}_${n}_${m}_$bg.json"); done; done
    "$CPUPY" scripts/accept_gate.py "${P[@]}" --out "$REP/gate_${n}_$m.json" > /dev/null && g=ACCEPT || g=reject
    "$CPUPY" scripts/accept_gate.py "${P[@]}" --perceptual --out "$REP/gate_${n}_${m}_perc.json" > /dev/null && gp=ACCEPT || gp=reject
    echo "GATE $n vs $REF [$m]: strict $g, perceptual $gp"
  done
done
"$CPUPY" - "$REP" "$REF" "${MODES:-depth radial}" "${BG[*]}" "${TS[*]}" "${PL[*]}" <<'PY'
import json, os, sys
rep, ref, modes, bgs, ts, pl = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split(), sys.argv[5].split(), sys.argv[6].split()
names = [p.split("=", 1)[0] for p in pl]; tests = [t.split("=", 1)[0] for t in ts]
L = ["| test | ply | " + " | ".join(f"{m} {b} PSNR / SSIM / LPIPS / DISTS" for m in modes for b in bgs) + " | gate depth (strict/perc) | gate radial |",
     "|---" * (3 + len(modes) * len(bgs)) + "|---|"]
for t in tests:
    for n in names:
        c = []
        for m in modes:
            for b in bgs:
                f = os.path.join(rep, f"{t}_{n}_{m}_{b}.json"); r = json.load(open(f)) if os.path.exists(f) else {}
                c.append(" / ".join(str(r.get(k, "-")) for k in ("psnr", "ssim", "lpips", "dists")))
        g = lambda m, s="": (lambda f: ("ACCEPT" if json.load(open(f))["accept"] else "reject") if os.path.exists(f) else "-")(
            os.path.join(rep, f"gate_{n}_{m}{s}.json"))
        L.append(f"| {t} | {n} | " + " | ".join(c) + (" | ref | ref |" if n == ref else f" | {g('depth')}/{g('depth', '_perc')} | {g('radial')} |"))
open(os.path.join(rep, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
echo SORT_SCORE_DONE
