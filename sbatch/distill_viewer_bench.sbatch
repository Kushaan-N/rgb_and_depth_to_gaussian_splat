#!/bin/bash
# Self-distillation at virtual viewpoints for the viewer-matched splat. A teacher (3DGRUT EDGS + Improved-GS 4M,
# $P/edgs/on/edgs_igs4m, trained without the 53 held-out frames; very accurate near the camera path in its own
# renderer) is rendered at pseudo-cameras near -- not on -- the training path (scripts/make_pseudo_views.py: between
# consecutive training frames, optional right/down + yaw/pitch offsets, nothing within 2 frames / 8 cm of a held-out
# camera; RGB + opacity + depth via eval_views.py; pixels with opacity < 0.98, near depth edges, or > 20% nearer than
# all LiDAR returns of their 8 px block (teacher floaters; $LIDAR_PTS, skipped if absent) masked). The student
# is the deliverable recipe (train_viewer_gsplat.py igs4m_pc274, 60k) trained on real frames + pseudo-views at a
# lower masked weight (scripts/train_distill_gsplat.py wraps a snapshot of the trainer at git HEAD; trainer unchanged).
# Scored by THE metric: the PLY in live SuperSplat (viewer_shots.py --engine current, backgrounds $BGS) at
# $P/onpath/test -> eval_views.py --pred-dir; gated (accept_gate.py, $GATE_FLAGS e.g. --perceptual) against the 60k
# baseline's viewer scores ($BASE_REP/igs4m_pc274_<bg>.json). Off-path renders (no GT): base + arms at each of
# $MOVES (camera right/down/back, dataset units), black, every $EVERY_OFF-th held-out camera, alpha-hole share from
# _viewer.json; SuperSplat editor default view (--editor). Montages compare base | arms.
#   ARMS="d0r0_w0.3 d15r8_w0.3" sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%A_%a.out --array=0-1 \
#       sbatch/distill_viewer_bench.sbatch configs/<seq>.yaml          # one arm per array task
#   MODE=report ARMS="..." sbatch -A ... -d afterany:<array> sbatch/distill_viewer_bench.sbatch configs/<seq>.yaml
#   ITERS=3000 VD_OUT=$P/viewer_distill_smoke ARMS=d15r8_w0.3 sbatch ...                                  # smoke
# Arm names: d<offset cm>r<rot deg>_w<weight>[_n<views per pair>][_s<seed>][_i<k>k (iterations, default $ITERS)], e.g.
#   d15r8_w0.3_i92k = +-15 cm right/down,
#   +-8 deg yaw/pitch, weight 0.3, 1 view per training-frame pair, 92k iterations.
# Outputs under $VD (default $P/viewer_distill): pseudo/<views>/ (dataset + teacher renders), <arm>/{splat.ply,train.json,
#   train.log,renders_trainer/}, shots/{onpath,off<i>,editor}/<arm>/, report/<arm>_<bg>.json, report/gate_<arm>.json,
#   report/summary.{json,md}, montage_*.jpg.
#SBATCH -J distill-viewer
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"; DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VD="${VD_OUT:-$P/viewer_distill}"; REP="$VD/report"
DS="${VT_DATASET:-$P/edgs/on/data}"; TEST="$P/onpath/test"; TSI="${TSI:-8}"
TEACHER="${TEACHER:-$(ls -t "$P"/edgs/on/edgs_igs4m/run/*/ckpt_last.pt 2>/dev/null | head -1)}"
LIDAR_PTS="${LIDAR_PTS-$P/onpath/data_orig/sparse/0/points3D.txt}"; [ -f "$LIDAR_PTS" ] || LIDAR_PTS=""   # floater check (LiDAR seeds)
BASE_REP="${BASE_REP:-$P/viewer_train_60k/report}"; BASE_PLY="${BASE_PLY:-$P/viewer_train_60k/igs4m_pc274/splat.ply}"
ITERS="${ITERS:-60000}"; STUDENT="${STUDENT:---strategy improved --budget 4000000 --viewer pc274}"; PROF="${PROF:-pc274}"
ENGINE="${ENGINE:-current}"; BGS="${BGS:-black grey}"; read -ra BGA <<< "$BGS"; EVERY_OFF="${EVERY_OFF:-8}"
MOVES="${MOVES:-0 -0.3 1.0;0 -0.6 1.5}"; IFS=';' read -ra MV <<< "$MOVES"; GATE_FLAGS="${GATE_FLAGS:-}"
ARMS="${ARMS:-d0r0_w0.3 d15r8_w0.3}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo arms || echo all)}"
hostname; date; mkdir -p "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
[ -f "$DS/sparse/0/points3D.txt" ] || { echo "no dataset at $DS (sbatch/edgs_bench.sbatch builds it)"; exit 1; }
[ -f "$TEACHER" ] || { echo "no teacher checkpoint (TEACHER=...; sbatch/edgs_bench.sbatch ARMS=edgs_igs4m)"; exit 1; }
[ -f "$BASE_REP/igs4m_pc274_${BGA[0]}.json" ] && [ -f "$BASE_PLY" ] || { echo "no 60k baseline at $BASE_REP / $BASE_PLY"; exit 1; }
lock() { until mkdir "$1" 2>/dev/null; do sleep 20; done; }
lock "$CEAR_WS/.gsplat_viewer_lock"; trap 'rmdir "$CEAR_WS/.gsplat_viewer_lock" 2>/dev/null || true' EXIT
bash env/make_gsplat_viewer_venv.sh | grep -E 'gsplat-viewer\]|DONE'; bash env/make_viewer_venv.sh | tail -1
rmdir "$CEAR_WS/.gsplat_viewer_lock"; trap - EXIT

views_of() {  # <arm> -> V (pseudo-view set name) + PLAN (make_pseudo_views.py plan flags) + W (weight)
  [[ $1 =~ ^d([0-9.]+)r([0-9.]+)_w([0-9.]+)(_n([0-9]+))?(_s([0-9]+))?(_i([0-9]+)k)?$ ]] || { echo "bad arm name $1"; return 1; }
  local off="${BASH_REMATCH[1]}" rot="${BASH_REMATCH[2]}" n="${BASH_REMATCH[5]:-1}" s="${BASH_REMATCH[7]:-0}"
  W="${BASH_REMATCH[3]}"; V="d${off}r${rot}_n${n}_s${s}"; IT="${BASH_REMATCH[9]:+$(( BASH_REMATCH[9] * 1000 ))}"; IT="${IT:-$ITERS}"
  PLAN=(--offset-m "$("$CPUPY" -c "print($off / 100)")" --rot-deg "$rot" --per-pair "$n" --seed "$s")
}

make_views() {  # <arm>: plan -> teacher render -> masks (shared by arms with the same views; locked)
  views_of "$1"; local PD="$VD/pseudo/$V" TR="$VD/pseudo/${V}_teacher"
  mkdir -p "$VD/pseudo"; lock "$VD/pseudo/.lock_$V"
  if ! "$CPUPY" -c "import json,sys; sys.exit(0 if 'finish' in json.load(open(sys.argv[1])) else 1)" "$PD/pseudo.json" 2>/dev/null; then
    rm -rf "$PD" "$TR"; echo "== [$1] pseudo-views $V: ${PLAN[*]}"
    "$CPUPY" scripts/make_pseudo_views.py plan --dataset "$DS" --out "$PD" --test-split-interval "$TSI" \
      --sim3-json "$P/sim3.json" "${PLAN[@]}" > "$PD.plan.log" 2>&1 || { tail -20 "$PD.plan.log"; rmdir "$VD/pseudo/.lock_$V"; return 1; }
    cut -c1-400 "$PD.plan.log"
    ( export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH" TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}" PYTHONPATH="$DDIR"
      "$GPY" scripts/eval_views.py --checkpoint "$TEACHER" --dataset "$PD" --no-metrics --tag "teacher_$V" \
        --save-dir "$TR/rgb" --save-opacity-dir "$TR/opacity" --save-depth-dir "$TR/depth" ) > "$TR.log" 2>&1 \
      || { tail -30 "$TR.log"; rmdir "$VD/pseudo/.lock_$V"; return 1; }
    "$CPUPY" scripts/make_pseudo_views.py finish --out "$PD" --renders "$TR/rgb" --opacity "$TR/opacity" --depth "$TR/depth" \
      ${LIDAR_PTS:+--lidar-points "$LIDAR_PTS"} \
      || { rmdir "$VD/pseudo/.lock_$V"; return 1; }
  fi
  rmdir "$VD/pseudo/.lock_$V"
}

score_ply() {  # <name> <ply>: live SuperSplat at the held-out cameras -> report/<name>_<bg>.json; off-path + editor renders
  local n="$1" ply="$2" bg d i
  if [ ! -f "$REP/${n}_${BGA[-1]}.json" ]; then
    "$VPY" scripts/viewer_shots.py --engine "$ENGINE" --bg $BGS --ply "$ply" --dataset "$TEST" --out-dir "$VD/shots/onpath/$n" | tail -2
    for bg in "${BGA[@]}"; do
      d="$VD/shots/onpath/$n"; [ "${#BGA[@]}" -gt 1 ] && d="$d/$bg"
      "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$TEST" --tag "${n}_$bg" --json-out "$REP/${n}_$bg.json" | tail -1
    done
  fi
  for i in "${!MV[@]}"; do
    [ -f "$VD/shots/off$i/$n/_viewer.json" ] && continue
    E=(); [ "$i" = 0 ] && E=(--editor)
    "$VPY" scripts/viewer_shots.py --engine "$ENGINE" --bg black --ply "$ply" --dataset "$TEST" --every "$EVERY_OFF" \
      --move ${MV[$i]} "${E[@]}" --out-dir "$VD/shots/off$i/$n" | tail -1
  done
}

run_arm() {  # <arm>
  local arm="$1" R="$VD/$1" bg PAIRS=()
  lock "$VD/.base_lock"                                     # base = the 60k run: its on-path scores + shots are reused
  for bg in "${BGA[@]}"; do [ -f "$REP/base_$bg.json" ] || cp "$BASE_REP/igs4m_pc274_$bg.json" "$REP/base_$bg.json"; done
  mkdir -p "$VD/shots/onpath"; [ -e "$VD/shots/onpath/base" ] || ln -s "$(dirname "$BASE_REP")/shots/onpath/igs4m_pc274" "$VD/shots/onpath/base"
  score_ply base "$BASE_PLY" || true; rmdir "$VD/.base_lock"
  [ "$arm" = base ] && return 0
  make_views "$arm" || return 1; views_of "$arm"
  if [ ! -f "$R/splat.ply" ]; then
    mkdir -p "$R/_tvg"; git show HEAD:scripts/train_viewer_gsplat.py > "$R/_tvg/train_viewer_gsplat.py"   # frozen trainer
    S=(); [[ $arm =~ _s([0-9]+)(_i[0-9]+k)?$ ]] && S=(--seed "${BASH_REMATCH[1]}")
    echo "== [$arm] train: pseudo $V weight $W, $STUDENT, $IT its"; date
    PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$PROF" TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$PROF" \
      "$SPY" scripts/train_distill_gsplat.py --pseudo-dataset "$VD/pseudo/$V" --pseudo-weight "$W" --tvg "$R/_tvg/train_viewer_gsplat.py" -- \
      --dataset "$DS" --out "$R" --test-split-interval "$TSI" --eval-dataset "$TEST" --iters "$IT" $STUDENT "${S[@]}" \
      > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    grep -E '^\[distill\]|^\[vgs\] (held-out|[0-9]+ gaussians)' "$R/train.log" || true; date
  fi
  score_ply "$arm" "$R/splat.ply"
  for bg in "${BGA[@]}"; do PAIRS+=(--pair "onpath_$bg" "$REP/base_$bg.json" "$REP/${arm}_$bg.json"); done
  "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" $GATE_FLAGS --out "$REP/gate_$arm.json" | tail -$(( 3 * ${#BGA[@]} + 1 )) || true
}

if [ "$MODE" = arms ] || [ "$MODE" = all ]; then
  read -ra AL <<< "$ARMS"
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then AL=("${AL[$SLURM_ARRAY_TASK_ID]}"); fi
  rc=0
  for arm in "${AL[@]}"; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done
  [ "$MODE" = arms ] && { echo "DISTILL_VIEWER_ARMS_DONE rc=$rc"; exit $rc; }
fi

echo "== report ($ENGINE engine, backgrounds: $BGS, gate flags: ${GATE_FLAGS:-none})"
run_arm base
for arm in $ARMS; do
  [ -f "$REP/${arm}_${BGA[-1]}.json" ] || continue
  PAIRS=(); for bg in "${BGA[@]}"; do PAIRS+=(--pair "onpath_$bg" "$REP/base_$bg.json" "$REP/${arm}_$bg.json"); done
  "$CPUPY" scripts/accept_gate.py "${PAIRS[@]}" $GATE_FLAGS --out "$REP/gate_$arm.json" > /dev/null || true
done
"$CPUPY" - "$VD" "$TEST/images" "$BGS" "${#MV[@]}" base $ARMS <<'PY'
import cv2, json, os, sys, numpy as np
VD, gt, bgs, nmv, tags = sys.argv[1], sys.argv[2], sys.argv[3].split(), int(sys.argv[4]), list(dict.fromkeys(sys.argv[5:]))
rep = os.path.join(VD, "report"); ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
sub = lambda t, bg: os.path.join(VD, "shots", "onpath", t, *([bg] if len(bgs) > 1 else []))
rows = []
for t in tags:
    r = {bg: ld(os.path.join(rep, f"{t}_{bg}.json")) for bg in bgs}
    if not r[bgs[0]]: continue
    g, tj = ld(os.path.join(rep, f"gate_{t}.json")), ld(os.path.join(VD, t, "train.json"))
    row = {"arm": t, "trainer_psnr": tj.get("heldout_psnr") if tj else None, "gaussians": tj.get("n_gaussians") if tj else None,
           "train_min": round(tj["train_seconds"] / 60, 1) if tj else None, "gate": None if g is None else ("ACCEPT" if g["accept"] else "reject")}
    if tj and tj.get("distill"):
        dj = tj["distill"]; row.update(n_pseudo=dj.get("n_pseudo"), pseudo_px=dj.get("supervised_pixel_share"))
    for bg in bgs:
        if r[bg]: row.update({f"{k}_{bg}": r[bg][k] for k in ("psnr", "ssim", "lpips")})
    for i in range(nmv):                                    # off-path: share of pixels SuperSplat leaves see-through
        vj = ld(os.path.join(VD, "shots", f"off{i}", t, "_viewer.json"))
        row[f"off{i}_holes"] = round(vj["alpha_hole_frac"], 5) if vj and "alpha_hole_frac" in vj else None
    rows.append(row)
json.dump(rows, open(os.path.join(rep, "summary.json"), "w"), indent=2)
f = lambda v, p: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
L = ["| arm | " + " | ".join(f"{bg} PSNR / SSIM / LPIPS" for bg in bgs) + " | " + " | ".join(f"off{i} holes" for i in range(nmv))
     + " | pseudo views (px share) | trainer PSNR | gaussians | train min | gate vs 60k |", "|" + "---|" * (len(bgs) + nmv + 6)]
for r in rows:
    L.append(f"| {r['arm']} | " + " | ".join(f"{f(r.get('psnr_'+bg),3)} / {f(r.get('ssim_'+bg),4)} / {f(r.get('lpips_'+bg),4)}" for bg in bgs)
             + " | " + " | ".join(f(r.get(f"off{i}_holes"), 4) for i in range(nmv))
             + f" | {f(r.get('n_pseudo'),0)} ({f(r.get('pseudo_px'),3)}) | {f(r['trainer_psnr'],3)} | {f(r['gaussians'],0)} | {f(r['train_min'],1)} | {r['gate'] or '-'} |")
open(os.path.join(rep, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def montage(kind, out, k, first_gt=False):
    d = lambda t: sub(t, bgs[0]) if kind == "onpath" else os.path.join(VD, "shots", kind, t)
    if not os.path.isdir(d("base")): return
    names = sorted(n for n in os.listdir(d("base")) if n.endswith(".png") and not n.startswith("_"))
    if not names: return
    R = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, k).astype(int)]:
        ims = ([(cv2.imread(os.path.join(gt, n)), "real")] if first_gt else []) + [(cv2.imread(os.path.join(d(t), n)), t) for t in tags]
        ims = [(cv2.resize(i, (400, 300)), l) for i, l in ims if i is not None]
        for i, l in ims: cv2.putText(i, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        R.append(np.hstack([i for i, _ in ims]))
    w = max(r.shape[1] for r in R); R = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in R]
    cv2.imwrite(out, np.vstack(R)); print("montage ->", out)
montage("onpath", os.path.join(VD, "montage_onpath.jpg"), 4, True)
for i in range(nmv): montage(f"off{i}", os.path.join(VD, f"montage_off{i}.jpg"), 4)
ed = [(cv2.imread(os.path.join(VD, "shots", "off0", t, "_supersplat_editor.png")), t) for t in tags]
ed = [(cv2.resize(i, (640, int(640 * i.shape[0] / i.shape[1]))), t) for i, t in ed if i is not None]
if ed:
    for i, t in ed: cv2.putText(i, t, (8, 26), 0, 0.8, (0, 0, 255), 2)
    h = min(i.shape[0] for i, _ in ed); cv2.imwrite(os.path.join(VD, "montage_editor.jpg"), np.hstack([i[:h] for i, _ in ed]))
    print("montage ->", os.path.join(VD, "montage_editor.jpg"))
PY
echo DISTILL_VIEWER_BENCH_DONE
