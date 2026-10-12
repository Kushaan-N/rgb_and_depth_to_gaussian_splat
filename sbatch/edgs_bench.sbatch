#!/bin/bash
# EDGS dense-correspondence init (scripts/edgs_init.py) vs the LiDAR-seed init, on both benchmarks:
#   ON-PATH  train $P/onpath/data_orig (3DGRUT holds out every 8th frame), score on $P/onpath/test
#   OFF-PATH train $P/offpath/train_data (all frames),                      score on $P/offpath/test
# Arms (each trained on both sets, patched 3DGRUT worktree so the depth loss is available):
#   edgs        EDGS points + LiDAR seeds via points3D.txt (3DGRUT colmap init), no depth loss  vs arm_A / baseline
#   edgs_depth  same + LiDAR depth loss (LAMBDA)                                                 vs arm_B / depth_l0.05
#   edgs_pc     EDGS-faithful: per-point EDGS scales + opacity 0.5 (initialization=point_cloud), no opacity
#               reset, density decay 0.99 / 10 its until densify end (EDGS's schedule), no depth vs arm_A / baseline
#   edgs_igs[<N>m]  EDGS init + Improved-GS densification (budget <N>M gaussians, default IGS_BUDGET; worktree from
#                   env/make_3dgrut_densify.sh, app colmap_3dgut_improved.yaml), no depth             vs arm_A / baseline
#   edgs_igs[<N>m]_depth  same + LiDAR depth loss                                                          vs arm_B / depth_l0.05
#   <arm>_s<N>      the same arm with seed_initialization=N (retraining-noise check), e.g. edgs_s1
#   NB the IGS budget caps the TOTAL count: with a ~2.7M-point EDGS init a 1M budget means no splits at all,
#   only Improved-GS's pruning (RAP etc.); edgs_igs4m gives it room to densify.
#   (also listed in the summary: E3's igs1m / igs1m_d with the LiDAR-seed init, $P/densify/report, if present)
# Baselines are re-scored with the current eval_views.py so every metric (incl. observed/unobserved) is
# compared like for like; accept_gate.py per arm over both sets; floor coverage of each on-path model;
# preview montages (real | baseline | arms) for a few held-out frames. Everything under $P/edgs/.
# Needs env/make_edgs_venv.sh to have built $CEAR_WS/venv-edgs and the baselines of onpath_ablation /
# offpath_depthloss. Re-running skips finished stages; jobs with different ARMS can run concurrently (the
# first builds the init under a lock, the others wait for it; the summary covers every arm scored so far).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/edgs-bench-%j.out sbatch/edgs_bench.sbatch configs/<seq>.yaml
#   ARMS="edgs_igs edgs_igs_depth" sbatch -A ... sbatch/edgs_bench.sbatch configs/<seq>.yaml
#SBATCH -J edgs-bench
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; EPY="${EDGS_VENV:-$CEAR_WS/venv-edgs}/bin/python"
DEPTH_DIR="${THREEDGRUT_DEPTH_DIR:-$CEAR_WS/3dgrut-depth}"; LAMBDA="${LAMBDA:-0.05}"; ARMS="${ARMS:-edgs edgs_depth edgs_pc edgs_igs edgs_igs_depth}"
DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"; IGS_BUDGET="${IGS_BUDGET:-1000000}"
IGS_ENV=(env TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}" PYTHONPATH="$DDIR")   # its kernels differ
EDGS_ARGS="${EDGS_ARGS:-}"                                 # extra scripts/edgs_init.py flags
export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$P/edgs"; REP="$E/report"
ITERS="$(get pipeline.train_iterations 30000)"; APP="$(get pipeline.trainer_app apps/colmap_3dgut.yaml)"
DENS_END="$(sed -n 's/^ *end_iteration: *\([0-9]*\).*/\1/p' "$DEPTH_DIR/configs/strategy/gs.yaml" | head -1)"
hostname; date
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -x "$EPY" ] || { echo "no $EPY: sbatch env/make_edgs_venv.sh first"; exit 1; }
mkdir -p "$E" "$REP"; bash env/make_3dgrut_depth.sh
case " $ARMS " in *_igs*) bash env/make_3dgrut_densify.sh; export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}";; esac

declare -A SRC=([on]="$P/onpath/data_orig" [off]="$P/offpath/train_data")
declare -A TEST=([on]="$P/onpath/test" [off]="$P/offpath/test")
declare -A SPLIT=([on]=8 [off]=100000)                     # trainer's test_split_interval
declare -A EVERY=([on]=8 [off]=0)                          # frames edgs_init must leave out (same rule)
declare -A BASE_A=([on]="$(ls -t "$P"/onpath/arm_A/run/*/ckpt_last.pt | head -1)" [off]="$(ls -t "$P"/offpath/work/baseline/run/*/ckpt_last.pt | head -1)")
declare -A BASE_B=([on]="$(ls -t "$P"/onpath/arm_B/run/*/ckpt_last.pt | head -1)" [off]="$(ls -t "$P"/offpath/work/depth_l0.05/run/*/ckpt_last.pt | head -1)")
declare -A REF=([edgs]=A [edgs_depth]=B [edgs_pc]=A)
for a in $ARMS; do [ -n "${REF[$a]:-}" ] || case $a in *_depth) REF[$a]=B;; *) REF[$a]=A;; esac; done   # edgs_igs* arms

score() {  # ckpt set tag [runner...] -> $REP/<set>_<tag>.json + renders
  local t="$2_$3"; [ -f "$REP/$t.json" ] && { echo "  $t: cached"; return; }
  "${@:4}" "$GPY" scripts/eval_views.py --checkpoint "$1" --dataset "${TEST[$2]}" --tag "$t" --json-out "$REP/$t.json" \
    --save-dir "$E/renders/$t" | tail -1
}

echo "== 1. EDGS init (RoMa dense matches -> triangulation -> LiDAR / outlier filter -> merge with seeds)"
for s in on off; do
  DS="$E/$s/data"
  if [ ! -f "$DS/edgs_init.json" ] && mkdir "$E/.lock_$s" 2>/dev/null; then
    LOCKS="${LOCKS:-} $E/.lock_$s"; trap 'rmdir $LOCKS 2>/dev/null || true' EXIT
    rm -rf "$DS"; "$EPY" scripts/edgs_init.py --dataset "${SRC[$s]}" --out "$DS" --test-split-interval "${EVERY[$s]}" \
      --sim3-json "$P/sim3.json" $EDGS_ARGS > "$E/init_$s.log" 2>&1 || { tail -40 "$E/init_$s.log"; exit 1; }
    rmdir "$E/.lock_$s"
  fi
  until [ -f "$DS/edgs_init.json" ]; do [ -d "$E/.lock_$s" ] || { echo "init $s failed elsewhere (or stale lock)"; exit 1; }; sleep 30; done
  "$CPUPY" -c "import json,sys; d=json.load(open(sys.argv[1])); d.pop('args'); print('  $s', d)" "$DS/edgs_init.json"
done

echo "== 2. baselines (re-scored)"
for s in on off; do score "${BASE_A[$s]}" $s base_A; score "${BASE_B[$s]}" $s base_B; done

echo "== 3. arms"
for arm in $ARMS; do
  for s in on off; do
    RUN="$E/$s/$arm"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
    RUNNER=(); WD="$DEPTH_DIR"; A="$APP"
    case $arm in *_igs*) RUNNER=("${IGS_ENV[@]}"); WD="$DDIR"; A=apps/colmap_3dgut_improved.yaml;; esac
    if [ -z "$CK" ]; then
      X=(loss.use_depth=false)
      case $arm in
        edgs_depth) X=(loss.use_depth=true loss.lambda_depth="$LAMBDA");;
        edgs_igs*) b="$IGS_BUDGET"; [[ $arm =~ igs([0-9]+)m ]] && b=$(( BASH_REMATCH[1] * 1000000 ))
                   X=(strategy.budget="$b" loss.use_depth=false)
                   if [[ $arm == *_depth ]]; then X=(strategy.budget="$b" loss.use_depth=true loss.lambda_depth="$LAMBDA"); fi;;
        edgs_pc) X=(loss.use_depth=false initialization=point_cloud strategy.reset_density.start_iteration=-1
                    strategy.density_decay.start_iteration=0 strategy.density_decay.end_iteration="$DENS_END"
                    strategy.density_decay.frequency=10 strategy.density_decay.gamma=0.99);;
      esac
      if [[ $arm =~ _s([0-9]+)$ ]]; then X+=(seed_initialization="${BASH_REMATCH[1]}"); fi    # <arm>_s<N>: another seed
      echo "== $arm / $s: ${X[*]}"; date
      ( cd "$WD" && "${RUNNER[@]}" "$GPY" train.py --config-name "$A" path="$E/$s/data" out_dir="$RUN" experiment_name=run \
          dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval="${SPLIT[$s]}" "${X[@]}" ) \
        > "$RUN.log" 2>&1 || { tail -40 "$RUN.log"; exit 1; }
      CK="$(ls -t "$RUN"/run/*/ckpt_last.pt | head -1)"; date
    fi
    score "$CK" $s "$arm" "${RUNNER[@]}"
  done
  "$CPUPY" scripts/accept_gate.py --pair onpath "$REP/on_base_${REF[$arm]}.json" "$REP/on_$arm.json" \
    --pair offpath "$REP/off_base_${REF[$arm]}.json" "$REP/off_$arm.json" --out "$REP/gate_$arm.json" | tail -3 || true
done 2>&1

echo "== 4. floor coverage (on-path models)"
for t in base_A $ARMS; do
  [ -f "$REP/floor_$t.json" ] && continue
  if [ "$t" = base_A ]; then CK="${BASE_A[on]}"; else CK="$(ls -t "$E"/on/$t/run/*/ckpt_last.pt | head -1)"; fi
  R=(); case $t in *_igs*) R=("${IGS_ENV[@]}");; esac
  PLY="$E/on/splat_$t.ply"; [ -f "$PLY" ] || "${R[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$PLY"
  "$CPUPY" scripts/floor_coverage.py --depth-cloud "$P/depth_cloud.ply" --splat-ply "$PLY" --sim3-json "$P/sim3.json" \
    --gt-model "$OUT/colmap_train/sparse/0" --out "$REP/floor_$t.png" | grep '^FLOOR_COVERAGE ' | sed 's/^FLOOR_COVERAGE //' > "$REP/floor_$t.json"
done

echo "== 5. summary + montages"
ALL="edgs edgs_depth edgs_pc $(ls "$REP" | sed -n 's/^\(on\|off\)_\(edgs[^.]*\)\.json$/\2/p' | sort -u | tr '\n' ' ')"
"$CPUPY" - "$E" "$ALL" "${TEST[on]}" "${TEST[off]}" "$P/densify/report" <<'PY'
import cv2, json, os, sys, numpy as np
E, tests, drep = sys.argv[1], {"on": sys.argv[3], "off": sys.argv[4]}, sys.argv[5]
rep = os.path.join(E, "report"); ld = lambda f: json.load(open(os.path.join(rep, f))) if os.path.isfile(os.path.join(rep, f)) else None
arms = [a for a in dict.fromkeys(sys.argv[2].split()) if ld(f"on_{a}.json") or ld(f"off_{a}.json")]
ref = {a: "B" if a.endswith("_depth") else "A" for a in arms}
keys = ("psnr", "ssim", "lpips", "psnr_observed", "psnr_unobserved", "depth_absrel", "depth_within5pct")
out = {"init": {s: {k: v for k, v in ld_.items() if k != "args"} for s in ("on", "off")
                for ld_ in [json.load(open(os.path.join(E, s, "data", "edgs_init.json")))]}, "sets": {}, "gates": {}, "floor": {}}
for s in ("on", "off"):
    out["sets"][s] = {t: {k: r.get(k) for k in keys} for t in ["base_A", "base_B"] + arms for r in [ld(f"{s}_{t}.json")] if r}
    for t in ("igs1m", "igs1m_d"):                         # E3's Improved-GS with the LiDAR-seed init (reference only)
        f = os.path.join(drep, f"{'onpath' if s == 'on' else 'offpath'}_{t}.json")
        if os.path.isfile(f): r = json.load(open(f)); out["sets"][s]["E3_" + t] = {k: r.get(k) for k in keys}
for a in arms:
    g = ld(f"gate_{a}.json")
    if g: out["gates"][a] = {"vs": "base_" + ref[a], "accept": g["accept"], "failures": g["failures"]}
for t in ["base_A"] + arms:
    f = ld(f"floor_{t}.json"); out["floor"][t] = f.get("splat_has_floor") if f else None
json.dump(out, open(os.path.join(rep, "edgs_summary.json"), "w"), indent=2)
for s in ("on", "off"):
    print(f"--- {s}-path"); print(f"{'arm':16s}" + "".join(f"{k[:12]:>13s}" for k in keys))
    for t, r in out["sets"][s].items():
        print(f"{t:16s}" + "".join(f"{(r[k] if r[k] is not None else float('nan')):13.4f}" for k in keys))
print("gates:", json.dumps(out["gates"])); print("floor:", out["floor"])
for s in ("on", "off"):                                    # real | baseline | arms, 4 held-out frames
    names = sorted(n for n in os.listdir(os.path.join(tests[s], "images")) if n.endswith(".png") and "_depth" not in n and "_covis" not in n)
    rows = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, 4).astype(int)]:
        cols = [cv2.imread(os.path.join(tests[s], "images", n))]
        for t in ["base_A"] + arms:
            im = cv2.imread(os.path.join(E, "renders", f"{s}_{t}", n))
            if im is None: continue
            cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); cols.append(im)
        rows.append(np.hstack(cols))
    p = os.path.join(rep, f"montage_{s}.jpg"); cv2.imwrite(p, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5)); print("montage ->", p)
PY
echo "EDGS_BENCH_DONE"
