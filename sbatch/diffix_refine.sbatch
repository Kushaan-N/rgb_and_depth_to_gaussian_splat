#!/bin/bash
# Difix3D+-style refinement of the viewer splat (scripts/diffix_refine.py): NVIDIA Difix (reference-conditioned,
# $DIFIX_VENV, env/make_difix_venv.sh) cleans renders at virtual cameras moving progressively off the training path,
# distilled back with the viewer-exact (SuperSplat 3.5.2) fine-tune; real frames dominate. Every arm has a matching
# control (same schedule, --pseudo-prob 0) so the fine-tune itself is not mistaken for a Difix gain.
# Scored by THE metric: live SuperSplat (viewer_shots.py --engine current, black + grey) on $P/onpath/test, + DISTS,
# accept_gate.py $GATE_FLAGS vs the base; off-path moved views ($MOVES, first one with --editor) for montages.
#   ho_*  held-out base (trained without the 53 test frames; $HO_PLY), fine-tune with --test-split-interval 8
#   dl_*  deliverable base ($DL_PLY, all frames)
#   fixtest  Difix3D+ post-render diagnostic on the held-out base's live test renders (+ Difix of the real images)
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-4 sbatch/diffix_refine.sbatch configs/<seq>.yaml
#   MODE=report sbatch -A ... -d afterany:<array> sbatch/diffix_refine.sbatch configs/<seq>.yaml
# Env: ARMS ("fixtest ho_ctrl ho_difix dl_ctrl dl_difix"), ROUNDS (0:0,0.1:5,0.2:10), IPR (iters per round, 2000),
#      DFX_ARGS (extra diffix_refine.py flags), SCORE (0 = refine only), DX (work dir, $P/difix_refine), GATE_FLAGS (--perceptual).
#SBATCH -J diffix-refine
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
DX="${DX:-$P/difix_refine}"; REP="$DX/report"; TEST="$P/onpath/test"; DATA="$P/onpath/data_orig"
HO_PLY="${HO_PLY:-$P/viewer_train_60k/igs4m_pc274/splat.ply}"; HO_REP="${HO_REP:-$P/viewer_train_60k/report}"
HO_SHOTS="${HO_SHOTS:-$P/viewer_train_60k/shots/onpath/igs4m_pc274}"
DL_PLY="${DL_PLY:-$E/viewer_igs60k_wd_clean.ply}"; DL_NAME="$(basename "$DL_PLY" .ply)"
ROUNDS="${ROUNDS:-0:0,0.1:5,0.2:10}"; IPR="${IPR:-2000}"; GATE_FLAGS="${GATE_FLAGS:---perceptual}"
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; IFS=';' read -ra MV <<< "$MOVES"; EVERY_OFF="${EVERY_OFF:-8}"
ARMS="${ARMS:-fixtest ho_ctrl ho_difix dl_ctrl dl_difix}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -x "$DIFIX_VENV/bin/python" ] || { echo "no Difix venv at $DIFIX_VENV (env/make_difix_venv.sh)"; exit 1; }

dists() {  # renderdir json: add DISTS (piq) to an eval_views json (accept_gate --perceptual uses it)
  PYTHONPATH="$PIQ_LIB" "$GPY" - "$1" "$TEST/images" "$2" <<'PY'
import json, os, sys, numpy as np, torch, piq
from PIL import Image
pred, gt, js = sys.argv[1:4]; r = json.load(open(js))
if "dists" not in r:
    t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].cuda()
    m = piq.DISTS(reduction="none").cuda(); v = []
    for n in sorted(f for f in os.listdir(pred) if f.endswith(".png") and not f.startswith("_")):
        with torch.no_grad(): v.append(m(t(os.path.join(pred, n)), t(os.path.join(gt, n))).item())
    r["dists"] = round(float(np.mean(v)), 4); json.dump(r, open(js, "w"), indent=2)
print("DISTS", r["tag"], r["dists"])
PY
}
evalp() {  # tag dir -> $REP/<tag>.json (+ DISTS)
  [ -f "$REP/$1.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$2" --dataset "$TEST" --tag "$1" --json-out "$REP/$1.json" | tail -1
  dists "$2" "$REP/$1.json"
}
shots() {  # ply outdir [viewer_shots args]; the live editor occasionally never loads: retry once
  [ -f "$2/_viewer.json" ] && return 0
  "$VPY" scripts/viewer_shots.py --engine current --ply "$1" --dataset "$TEST" --out-dir "$2" "${@:3}" | tail -1 || \
    { sleep 60; "$VPY" scripts/viewer_shots.py --engine current --ply "$1" --dataset "$TEST" --out-dir "$2" "${@:3}" | tail -1; } || echo "SHOTS FAILED $2"
}
score_ply() {  # name ply: on-path live (black, grey) -> report/<name>_<bg>.json; off-path + editor shots
  local n="$1" ply="$2" i E2
  shots "$ply" "$DX/shots/onpath/$n" --bg black grey
  for bg in black grey; do evalp "${n}_$bg" "$DX/shots/onpath/$n/$bg"; done
  for i in "${!MV[@]}"; do
    E2=(); [ "$i" = 0 ] && E2=(--editor)
    shots "$ply" "$DX/shots/off$i/$n" --bg black --every "$EVERY_OFF" --move ${MV[$i]} "${E2[@]}"
  done
}
gate() {  # cand base -> report/gate_<cand>_vs_<base>.json
  "$CPUPY" scripts/accept_gate.py --pair onpath_black "$REP/${2}_black.json" "$REP/${1}_black.json" \
    --pair onpath_grey "$REP/${2}_grey.json" "$REP/${1}_grey.json" $GATE_FLAGS --out "$REP/gate_${1}_vs_$2.json" | tail -13 || true
}
bases() {  # base scores: held-out base reuses its existing live shots; deliverable base is shot here (locked: shared)
  until mkdir "$DX/.base_lock" 2>/dev/null; do sleep 20; done
  mkdir -p "$DX/shots/onpath"
  [ -e "$DX/shots/onpath/ho_base" ] || ln -s "$HO_SHOTS" "$DX/shots/onpath/ho_base"
  for bg in black grey; do [ -f "$REP/ho_base_$bg.json" ] || cp "$HO_REP/igs4m_pc274_$bg.json" "$REP/ho_base_$bg.json"; done
  score_ply ho_base "$HO_PLY"; score_ply dl_base "$DL_PLY"; rmdir "$DX/.base_lock"
}

run_arm() {
  local arm="$1" kind typ base ply iv pp
  if [ "$arm" = fixtest ]; then                 # Difix3D+ post-render on held-out live renders; ceiling = Difix(real image)
    [ -d "$DX/fixtest/fixed_gt" ] || "$SPY" scripts/diffix_refine.py fixtest --renders "$HO_SHOTS/black" --test "$TEST" \
      --train "$DATA" --test-split-interval 8 --work "$DX/fixtest"
    evalp fixtest_render "$DX/fixtest/fixed_render"; evalp fixtest_gt "$DX/fixtest/fixed_gt"
    return 0
  fi
  [[ $arm =~ ^(ho|dl)_(ctrl|difix)(.*)$ ]] || { echo "bad arm $arm"; return 1; }
  kind="${BASH_REMATCH[1]}"; typ="${BASH_REMATCH[2]}"; pp=0.3; [ "$typ" = ctrl ] && pp=0
  if [ "$kind" = ho ]; then ply="$HO_PLY"; iv=8; else ply="$DL_PLY"; iv=0; fi
  mkdir -p "$DX/$arm"
  if [ ! -f "$DX/$arm/splat.ply" ]; then
    echo "== [$arm] refine: rounds $ROUNDS x $IPR, pseudo prob $pp ${DFX_ARGS:-}"; date
    # shellcheck disable=SC2086
    "$SPY" scripts/diffix_refine.py refine --ply "$ply" --dataset "$DATA" --out "$DX/$arm/splat.ply" --work "$DX/$arm" \
      --test-split-interval "$iv" --sim3-json "$P/sim3.json" --rounds "$ROUNDS" --iters-per-round "$IPR" \
      --pseudo-prob "$pp" --eval-dataset "$TEST" ${DFX_ARGS:-} > "$DX/$arm/refine.log" 2>&1 || { tail -40 "$DX/$arm/refine.log"; return 1; }
    grep -E '^\[dfx\] (eval|round|wrote)|^\[pseudo\]' "$DX/$arm/refine.log" | cut -c1-300 || true; date
  fi
  [ "${SCORE:-1}" = 1 ] || return 0                        # SCORE=0: smoke (refine only)
  score_ply "$arm" "$DX/$arm/splat.ply"
  base="${kind}_base"; bases
  gate "$arm" "$base"; if [ "$typ" = difix ] && [ -f "$REP/${kind}_ctrl_grey.json" ]; then gate "$arm" "${kind}_ctrl"; fi
  return 0
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra AL <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then AL=("${AL[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${AL[@]}"; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done
  [ "$MODE" = arms ] && { echo "DIFFIX_REFINE_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report"
bases
for arm in $ARMS; do
  [[ $arm =~ ^(ho|dl)_(ctrl|difix) ]] || continue; [ -f "$REP/${arm}_grey.json" ] || continue
  k="${BASH_REMATCH[1]}"; t="${BASH_REMATCH[2]}"; gate "$arm" "${k}_base" > /dev/null
  if [ "$t" = difix ] && [ -f "$REP/${k}_ctrl_grey.json" ]; then gate "$arm" "${k}_ctrl" > /dev/null; fi
done
"$CPUPY" - "$DX" "$TEST/images" "${#MV[@]}" <<'PY'
import cv2, glob, json, os, sys, numpy as np
DX, gt, nmv = sys.argv[1], sys.argv[2], int(sys.argv[3]); rep = os.path.join(DX, "report")
ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
tags = sorted({os.path.basename(f)[:-len("_black.json")] for f in glob.glob(os.path.join(rep, "*_black.json"))})
L = ["| arm | black PSNR / SSIM / LPIPS / DISTS | grey PSNR / SSIM / LPIPS / DISTS | gate vs base | gate vs ctrl |", "|---|---|---|---|---|"]
for t in tags:
    r = [ld(os.path.join(rep, f"{t}_{b}.json")) for b in ("black", "grey")]
    f = lambda j: "-" if not j else f"{j['psnr']:.3f} / {j['ssim']:.4f} / {j['lpips']:.4f} / {j.get('dists', '-')}"
    k = t.split("_")[0]; g = lambda b: (lambda j: "-" if j is None else ("ACCEPT" if j["accept"] else "reject"))(ld(os.path.join(rep, f"gate_{t}_vs_{k}_{b}.json")))
    L.append(f"| {t} | {f(r[0])} | {f(r[1])} | {g('base')} | {g('ctrl')} |")
for t in ("fixtest_render", "fixtest_gt"):
    j = ld(os.path.join(rep, f"{t}.json"))
    if j: L.append(f"| {t} | {j['psnr']:.3f} / {j['ssim']:.4f} / {j['lpips']:.4f} / {j.get('dists', '-')} | | | |")
open(os.path.join(rep, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def montage(kind, out, arms, k=4, first_gt=False):
    d = lambda t: os.path.join(DX, "shots", kind, t, *(["black"] if kind == "onpath" else []))
    arms = [a for a in arms if os.path.isdir(d(a))]
    if len(arms) < 2: return
    names = sorted(n for n in os.listdir(d(arms[0])) if n.endswith(".png") and not n.startswith("_"))
    R = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, k).astype(int)]:
        ims = ([(cv2.imread(os.path.join(gt, n)), "real")] if first_gt else []) + [(cv2.imread(os.path.join(d(t), n)), t) for t in arms]
        ims = [(cv2.resize(i, (480, 360)), l) for i, l in ims if i is not None]
        for i, l in ims: cv2.putText(i, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        R.append(np.hstack([i for i, _ in ims]))
    w = max(r.shape[1] for r in R); R = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in R]
    cv2.imwrite(out, np.vstack(R)); print("montage ->", out)
for k in ("ho", "dl"):
    arms = [f"{k}_base", f"{k}_ctrl", f"{k}_difix"]
    montage("onpath", os.path.join(DX, f"montage_{k}_onpath.jpg"), arms, 4, True)
    for i in range(nmv): montage(f"off{i}", os.path.join(DX, f"montage_{k}_off{i}.jpg"), arms)
    ed = [(cv2.imread(os.path.join(DX, "shots", "off0", t, "_supersplat_editor.png")), t) for t in arms]
    ed = [(cv2.resize(i, (640, int(640 * i.shape[0] / i.shape[1]))), t) for i, t in ed if i is not None]
    if ed:
        for i, t in ed: cv2.putText(i, t, (8, 26), 0, 0.8, (0, 0, 255), 2)
        h = min(i.shape[0] for i, _ in ed); cv2.imwrite(os.path.join(DX, f"montage_{k}_editor.jpg"), np.hstack([i[:h] for i, _ in ed]))
PY
echo DIFFIX_REFINE_DONE
