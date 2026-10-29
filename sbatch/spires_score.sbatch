#!/bin/bash
# Score exported splats (standard 3DGS PLY, adapter world frame) the CEAR way on an adapter-built rig dataset:
#   1. LIVE SuperSplat (scripts/viewer_shots.py --engine current, black + grey) at the held-out rectified-pinhole
#      frames (pinhole_center/test_in = in-sequence, test_out = out-of-sequence) -> eval_views.py --pred-dir
#   2. the dataset's own benchmark images (benchmark/test_{in,out}_cam<k>: their undistorted frames at their poses;
#      frame-weighted mean over cameras -> bench_test_{in,out}.json)
#   3. off-path views (--move, every EVERY_OFF-th test_out frame) + the SuperSplat editor's default view
#   4. geometry vs TLS: surface depth at the test_out cameras (datasets/spires_render_depth.py vs build_depth_targets.py on
#      tls.ply) and rendered-surface (training-view depth back-projected) + splat-centre precision / recall / F at 5 / 10 cm (scripts/datasets/spires_geom.py)
#   CEAR_DATA=$MC/data/spires CEAR_OUT=$MC/outputs PLYS="center=/path/a.ply fish=/path/b.ply" \
#     sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/spires_score.sbatch configs/spires_roq01.yaml
# Env: PLYS (name=ply ...), MOVE ("0 -0.3 1.0"), EVERY_OFF (10), SKIP ("" | e.g. "bench geom")
# Out: <adapter.root>/score/<name>/..., score/<name>/summary.json; score/summary.json (all names)
#SBATCH -J spires-score
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
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; SPY="$CEAR_WS/venv-gsplat/bin/python"
VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
ROOT="$(get adapter.root)"; PC="$ROOT/pinhole_center"; B="$ROOT/benchmark"; S="$ROOT/score"
MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-10}"; SKIP=" ${SKIP:-} "
hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader || true; mkdir -p "$S"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
bash env/make_viewer_venv.sh >/dev/null

# TLS depth targets at the held-out pinhole cameras (once; ray distance, metric world = identity sim3)
ID="$S/identity_sim3.json"; echo '{"scale": 1.0, "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}' > "$ID"
if [[ "$SKIP" != *" geom "* ]] && [ ! -f "$PC/test_out/depth_targets.json" ]; then
  "$GPY" scripts/build_depth_targets.py --dataset "$PC/test_out" --cloud "$ROOT/tls.ply" --sim3-json "$ID" \
    --voxel-m "$(get adapter.gt_voxel 0.02)" --max-range-m 40 | tail -2
fi

for spec in ${PLYS:?set PLYS="name=ply ..."}; do
  n="${spec%%=*}"; ply="${spec#*=}"; O="$S/$n"; mkdir -p "$O"
  [ -f "$ply" ] || { echo "missing $ply"; continue; }
  echo "== $n: $ply ($(du -h "$ply" | cut -f1))"
  for s in test_in test_out; do
    [ -f "$O/live_$s.json" ] || {
      "$VPY" scripts/viewer_shots.py --engine current --bg black grey --ply "$ply" --dataset "$PC/$s" --out-dir "$O/live_$s" \
        $([ "$s" = test_out ] && echo --editor) > "$O/live_$s.log" 2>&1 || { tail -20 "$O/live_$s.log"; exit 1; }
      for bg in black grey; do
        "$GPY" scripts/eval_views.py --pred-dir "$O/live_$s/$bg" --dataset "$PC/$s" --tag "${n}_${s}_$bg" \
          --json-out "$O/live_${s}_$bg.json" | tail -1
      done
      cp "$O/live_${s}_black.json" "$O/live_$s.json"; }
    if [[ "$SKIP" != *" bench "* ]] && [ ! -f "$O/bench_$s.json" ]; then
      for bd in "$B/${s}"_cam*; do                          # one dir per camera (their undistorted sizes differ)
        c="$(basename "$bd")"
        "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$ply" --dataset "$bd" --out-dir "$O/bench_$c" \
          > "$O/bench_$c.log" 2>&1 || { tail -20 "$O/bench_$c.log"; exit 1; }
        "$GPY" scripts/eval_views.py --pred-dir "$O/bench_$c" --dataset "$bd" --tag "${n}_bench_$c" --json-out "$O/bench_$c.json" | tail -1
      done
      "$CPUPY" -c "import json,sys,glob; r=[json.load(open(f)) for f in sorted(glob.glob(sys.argv[1]))]; n=sum(x['n'] for x in r)
json.dump({'n': n, **{k: round(sum(x[k]*x['n'] for x in r)/n, 4) for k in ('psnr','ssim','lpips')}, 'per_camera': r}, open(sys.argv[2],'w'), indent=1)" \
        "$O/bench_${s}_cam*.json" "$O/bench_$s.json"
    fi
  done
  [ -d "$O/offpath" ] || "$VPY" scripts/viewer_shots.py --engine current --bg grey --ply "$ply" --dataset "$PC/test_out" \
    --every "$EVERY_OFF" --move $MOVE --out-dir "$O/offpath" > "$O/offpath.log" 2>&1 || tail -5 "$O/offpath.log"
  if [[ "$SKIP" != *" geom "* ]]; then
    # surface depth without the >1 m sky gaussians (scripts/datasets/spires_render_depth.py explains why)
    [ -d "$O/sdepth_test_out" ] || "$SPY" scripts/datasets/spires_render_depth.py --ply "$ply" --dataset "$PC/test_out" --out-dir "$O/sdepth_test_out"
    [ -f "$O/geom_depth.json" ] || "$CPUPY" scripts/datasets/spires_geom.py depth --pred-dir "$O/sdepth_test_out" \
      --dataset "$PC/test_out" --json-out "$O/geom_depth.json"
    [ -d "$O/sdepth_train" ] || "$SPY" scripts/datasets/spires_render_depth.py --ply "$ply" --dataset "$PC/train" --out-dir "$O/sdepth_train"
    [ -f "$O/geom_points.json" ] || "$CPUPY" scripts/datasets/spires_geom.py points --ply "$ply" --tls "$ROOT/tls.ply" \
      --depth-dir "$O/sdepth_train" --depth-dataset "$PC/train" --cams "$ROOT/fisheye/train" --ref-cloud "$ROOT/lidar.ply" \
      --json-out "$O/geom_points.json"
    # their protocol: reconstruction points in the TLS octree's unknown space dropped (spires.py stage occ)
    [ ! -f "$ROOT/gt_occ_leaves.bin" ] || [ -f "$O/geom_points_occ.json" ] || "$CPUPY" scripts/datasets/spires_geom.py points \
      --tls "$ROOT/tls.ply" --depth-dir "$O/sdepth_train" --depth-dataset "$PC/train" --cams "$ROOT/fisheye/train" \
      --ref-cloud "$ROOT/lidar.ply" --occupancy "$ROOT/gt_occ_leaves.bin" --world "$ROOT/world.json" \
      --tls-align "$ROOT/tls_align.json" --json-out "$O/geom_points_occ.json"
  fi
done

"$CPUPY" - "$S" "$PC" ${PLYS} <<'PY'
import json, os, sys, cv2, numpy as np
S, PC, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
names = [s.split("=", 1)[0] for s in specs]
J = lambda n, f: json.load(open(os.path.join(S, n, f))) if os.path.exists(os.path.join(S, n, f)) else {}
allres = json.load(open(os.path.join(S, "summary.json"))) if os.path.exists(os.path.join(S, "summary.json")) else {}
print(f"{'splat':22s} {'in-seq live (blk)':>20s} {'out-seq live (blk)':>20s} {'out-seq grey':>8s} {'bench in':>9s} {'bench out':>9s} "
      f"{'dAbsRel':>8s} {'d<10cm':>7s} {'F@5cm':>6s} {'F@10cm':>6s}")
for n in names:
    r = {k: J(n, f"{k}.json") for k in ("live_test_in_black", "live_test_out_black", "live_test_out_grey", "bench_test_in",
                                         "bench_test_out", "geom_depth")}
    g = J(n, "geom_points.json"); r["geom_points"] = g; r["geom_points_occ"] = J(n, "geom_points_occ.json"); allres[n] = r
    f = lambda d, k="psnr": d.get(k, float("nan"))
    sp = g.get("splat", {})
    print(f"{n:22s} {f(r['live_test_in_black']):7.2f}/{f(r['live_test_in_black'],'ssim'):.3f}/{f(r['live_test_in_black'],'lpips'):.3f} "
          f"{f(r['live_test_out_black']):7.2f}/{f(r['live_test_out_black'],'ssim'):.3f}/{f(r['live_test_out_black'],'lpips'):.3f} "
          f"{f(r['live_test_out_grey']):8.2f} {f(r['bench_test_in']):9.2f} {f(r['bench_test_out']):9.2f} "
          f"{f(r['geom_depth'],'absrel'):8.4f} {f(r['geom_depth'],'within_10cm'):7.3f} {sp.get('fscore_5cm', float('nan')):6.3f} "
          f"{sp.get('fscore_10cm', float('nan')):6.3f}")
json.dump(allres, open(os.path.join(S, "summary.json"), "w"), indent=1)
# montage: real | each splat, a few out-of-sequence frames (black bg) + off-path (grey)
gt = os.path.join(PC, "test_out", "images"); fr = sorted(f for f in os.listdir(gt) if f.endswith(".jpg"))
rows = []
for nm in fr[:: max(1, len(fr) // 5)][:5]:
    st = os.path.splitext(nm)[0]; ims = [cv2.imread(os.path.join(gt, nm))]
    ims += [cv2.imread(os.path.join(S, n, "live_test_out", "black", st + ".png")) for n in names]
    ims = [cv2.resize(x, (480, 360)) if x is not None else np.zeros((360, 480, 3), np.uint8) for x in ims]
    for x, l in zip(ims, ["real"] + names): cv2.putText(x, l, (6, 24), 0, 0.8, (0, 0, 255), 2)
    rows.append(np.hstack(ims))
cv2.imwrite(os.path.join(S, "montage_" + "_".join(names) + ".jpg"), np.vstack(rows))
PY
echo SPIRES_SCORE_DONE
