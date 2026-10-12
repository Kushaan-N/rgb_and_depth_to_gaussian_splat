#!/bin/bash
# Depth priors on top of the current best recipe (edgs_igs4m: EDGS init + Improved-GS 4M budget, no depth loss), on the
# ON-PATH held-out split (train $P/edgs/on/data with test_split_interval=8, score on $P/onpath/test), worktree from
# env/make_3dgrut_depthmask.sh (depth loss + Improved-GS + depth mask / normal loss). Everything under $P/depthprior/.
#  prep (once, under a lock; later jobs wait for it):
#    - per-point LiDAR intensity from the bag (CPU venv; skipped when the config has no LiDAR),
#    - photometric depth: edgs_igs4m (no depth loss) rendered at every frame (eval_views --save-depth-dir),
#    - data_lidar: link copy of the EDGS dataset with LiDAR targets rebuilt without the near bias (build_depth_targets
#      --reduce surface, via depth_reliability.py --rebuild-targets) + <image>_depthmask.png reliability masks
#      (weak returns, depth edges, photometric disagreement on textured pixels) + bias diagnosis,
#    - data_pda: dense PromptDA depth (promptda_depth.py, prompted with the masked LiDAR) + its own masks (edges,
#      photometric disagreement).
#  arms (ARMS, each trained once; <arm>_s<N> = another seed):
#    lfix   LiDAR depth loss with the unbiased targets, no mask           (bias fix alone)
#    lmask  same + reliability mask                                      (Y3 a)
#    pda    PromptDA dense depth + its mask                              (Y3 b)
#    pdan   pda + normal consistency (LNORM)                             (Y3 c)
#    fp / <arm>fp  screen-footprint cap (loss.use_footprint, FPPX px, LFP): no depth / on top of <arm>, e.g. lmaskfp
#  scoring per arm and for base (edgs_igs4m) / base_depth (edgs_igs4m_depth, old LiDAR loss): the live SuperSplat
#  v3.5.2 editor (scripts/viewer_shots.py --engine current, WebGPU) on black and grey backgrounds (+ alpha-hole fraction)
#  and PlayCanvas 2.7.4 (older SuperSplat engine), scored with eval_views --pred-dir on $P/onpath/test; PSNR split by the
#  LiDAR reliability mask of the test frames (trusted / dropped / no target); 3DGRUT renders with depth AbsRel vs LiDAR
#  (old targets of $P/onpath/test and the rebuilt ones); off-path v3.5.2 shots (MOVES); accept_gate.py vs base on
#  v3.5.2 black + grey + 3DGRUT; summary + montages (CROPS="[off1/]stem:x,y,w,h:label ..." adds zoomed crops).
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/depthprior-%j.out sbatch/depthprior_bench.sbatch configs/<seq>.yaml
#   ARMS="pda pdan" sbatch ... ;  STAGE=report sbatch -p cpu ... (summary/montages only)
#SBATCH -J depthprior
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
PPY="${PROMPTDA_VENV:-$CEAR_WS/venv-promptda}/bin/python"
DMDIR="${THREEDGRUT_DEPTHMASK_DIR:-$CEAR_WS/3dgrut-depthmask}"
RUN=(env TORCH_EXTENSIONS_DIR="${DEPTHMASK_TORCH_EXT:-$CEAR_WS/torch-ext-depthmask}" PYTHONPATH="$DMDIR")
ARMS="${ARMS:-lfix lmask pda pdan}"; LAMBDA="${LAMBDA:-0.05}"; LNORM="${LNORM:-0.05}"; BUDGET="${BUDGET:-4000000}"
MOVES="${MOVES:-0 -0.3 1.0;0.5 0 0}"; STAGE="${STAGE:-all}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$P/edgs"; D="$P/depthprior"; REP="$D/report"
ITERS="$(get pipeline.train_iterations 30000)"; SRC="$E/on/data"; TEST="$P/onpath/test"
BASE_CK="$(ls -t "$E"/on/edgs_igs4m/run/*/ckpt_last.pt | head -1)"; BASED_CK="$(ls -t "$E"/on/edgs_igs4m_depth/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
STATIC="$("$CPUPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['cloud'])" "$SRC/depth_targets.json")"
hostname; date; mkdir -p "$D/on" "$REP"
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash env/make_3dgrut_depthmask.sh; bash env/make_viewer_venv.sh
VS() { "$VPY" scripts/viewer_shots.py "$@"; }

linkcopy() {  # src_ds dst_ds: sparse files + images linked (no depth / mask files)
  mkdir -p "$2/sparse/0" "$2/images"
  for f in "$1"/sparse/0/*; do ln -sfn "$f" "$2/sparse/0/"; done
  for f in "$1"/images/*; do case "$f" in *_depth.npy|*_depthmask.png|*_covis.png|*_mask.png) ;; *) ln -sfn "$f" "$2/images/";; esac; done
}

if [ "$STAGE" != report ]; then
echo "== prep"
if [ ! -f "$D/on/prep.done" ] && mkdir "$D/.lock_prep" 2>/dev/null; then
  trap 'rmdir "$D/.lock_prep" 2>/dev/null || true' EXIT
  [ -x "$PPY" ] || { echo "no $PPY: sbatch env/make_promptda_venv.sh first"; exit 1; }
  [ -f "$D/lidar_intensity.npz" ] || "$CPUPY" scripts/depth_reliability.py --intensity-from-config "$CFG" --cloud "$STATIC" \
    --intensity "$D/lidar_intensity.npz" 2>&1 | tail -4 || echo "  (no LiDAR intensity: weak-return criterion skipped)"
  INT=(); [ -f "$D/lidar_intensity.npz" ] && INT=(--intensity "$D/lidar_intensity.npz")
  [ -f "$D/on/photo/.done" ] || { "${RUN[@]}" "$GPY" scripts/eval_views.py --checkpoint "$BASE_CK" --dataset "$SRC" --no-metrics \
    --save-depth-dir "$D/on/photo" | tail -1; touch "$D/on/photo/.done"; }
  L="$D/on/data_lidar"; rm -rf "$L"; linkcopy "$SRC" "$L"
  cp "$SRC"/edgs_init.json "$SRC"/depth_targets.json "$L/"; for f in "$SRC"/images/*_depth.npy; do cp "$f" "$L/images/"; done   # old targets (diagnosis)
  "$GPY" scripts/depth_reliability.py --dataset "$L" --photo-depth-dir "$D/on/photo" --cloud "$STATIC" --sim3-json "$P/sim3.json" \
    --rebuild-targets "${INT[@]}" --edgs-points "$L/sparse/0/points3D.txt" --viz-dir "$D/viz/lidar" | tail -2
  A="$D/on/data_pda"; rm -rf "$A"; linkcopy "$SRC" "$A"; cp "$SRC"/edgs_init.json "$A/"
  HF_HOME="$HF_HOME" "$PPY" scripts/promptda_depth.py --dataset "$L" --out-dataset "$A" --sim3-json "$P/sim3.json" \
    --viz-dir "$D/viz/pda" | tail -2
  "$GPY" scripts/depth_reliability.py --dataset "$A" --photo-depth-dir "$D/on/photo" --viz-dir "$D/viz/pda_mask" | tail -1
  # held-out frames with the rebuilt (unbiased) LiDAR targets, for a second depth AbsRel
  T2="$D/on/test_new"; rm -rf "$T2"; linkcopy "$TEST" "$T2"
  "$GPY" scripts/build_depth_targets.py --dataset "$T2" --cloud "$STATIC" --sim3-json "$P/sim3.json" --reduce surface | tail -1
  touch "$D/on/prep.done"; rmdir "$D/.lock_prep"
fi
until [ -f "$D/on/prep.done" ]; do [ -d "$D/.lock_prep" ] || { echo "prep failed elsewhere (or stale lock)"; exit 1; }; sleep 30; done

score() {  # name ckpt_or_empty ply -> 3DGRUT + SuperSplat scores, off-path shots
  local n="$1" ck="$2" ply="$3"
  if [ -n "$ck" ]; then
    [ -f "$REP/g_$n.json" ] || "${RUN[@]}" "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$TEST" --tag "g_$n" \
      --json-out "$REP/g_$n.json" --save-dir "$D/renders/g_$n" | tail -1
    [ -f "$REP/g2_$n.json" ] || "${RUN[@]}" "$GPY" scripts/eval_views.py --checkpoint "$ck" --dataset "$D/on/test_new" --tag "g2_$n" \
      --json-out "$REP/g2_$n.json" | tail -1
    [ -f "$ply" ] || "${RUN[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$ck" --out "$ply"
  fi
  if [ ! -f "$REP/curgrey_$n.json" ]; then                 # primary: live SuperSplat v3.5.2 (WebGPU), black + grey bg
    VS --engine current --bg black grey --ply "$ply" --dataset "$TEST" --out-dir "$D/cur/on/$n" 2>&1 | tail -1
    for bg in black grey; do "$GPY" scripts/eval_views.py --pred-dir "$D/cur/on/$n/$bg" --dataset "$TEST" --tag "cur${bg}_$n" \
      --json-out "$REP/cur${bg}_$n.json" | tail -1; done
  fi
  if [ ! -f "$REP/ss_$n.json" ]; then                      # secondary: PlayCanvas 2.7.4 gsplat (older SuperSplat engine)
    VS --ply "$ply" --dataset "$TEST" --out-dir "$D/ss/on/$n" 2>&1 | tail -1
    "$GPY" scripts/eval_views.py --pred-dir "$D/ss/on/$n" --dataset "$TEST" --tag "ss_$n" --json-out "$REP/ss_$n.json" | tail -1
  fi
  local i=0; IFS=';' read -ra MV <<< "$MOVES"
  for m in "${MV[@]}"; do i=$((i + 1)); [ -f "$D/cur/off$i/$n/_viewer.json" ] || VS --engine current --bg black --ply "$ply" --dataset "$TEST" \
    --every 4 --move $m --out-dir "$D/cur/off$i/$n" 2>&1 | tail -1; done
}

echo "== baselines"
mkdir -p "$D/ply"
[ -f "$D/ply/base.ply" ] || { if [ -f "$E/on/splat_edgs_igs4m.ply" ]; then ln -sfn "$E/on/splat_edgs_igs4m.ply" "$D/ply/base.ply"; fi; }
score base "$BASE_CK" "$D/ply/base.ply"
if [ -n "$BASED_CK" ]; then
  [ -f "$D/ply/base_depth.ply" ] || { [ -f "$E/on/splat_edgs_igs4m_depth.ply" ] && ln -sfn "$E/on/splat_edgs_igs4m_depth.ply" "$D/ply/base_depth.ply" || true; }
  score base_depth "$BASED_CK" "$D/ply/base_depth.ply"
fi

echo "== arms"
for arm in $ARMS; do
  RD="$D/on/$arm"; CK="$(ls -t "$RD"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  X=(loss.use_depth=true loss.lambda_depth="$LAMBDA" loss.depth_mask=true)
  case $arm in
    lfix*)  DS="$D/on/data_lidar"; X=(loss.use_depth=true loss.lambda_depth="$LAMBDA" loss.depth_mask=false);;
    lmask*) DS="$D/on/data_lidar";;
    pdan*)  DS="$D/on/data_pda"; X+=(loss.use_normal=true loss.lambda_normal="$LNORM" loss.normal_start_iteration="${NSTART:-3000}");;
    pda*)   DS="$D/on/data_pda";;
    fp*)    DS="$D/on/data_lidar"; X=(loss.use_depth=false);;
    *) echo "unknown arm $arm"; exit 1;;
  esac
  case $arm in *fp*) X+=(loss.use_footprint=true loss.lambda_footprint="${LFP:-0.1}" loss.footprint_max_px="${FPPX:-10}");; esac
  if [[ $arm =~ _s([0-9]+)$ ]]; then X+=(seed_initialization="${BASH_REMATCH[1]}"); fi
  if [ -z "$CK" ]; then
    echo "== $arm: $DS ${X[*]}"; date
    ( cd "$DMDIR" && "${RUN[@]}" "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$DS" out_dir="$RD" \
        experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=8 \
        strategy.budget="$BUDGET" "${X[@]}" ) > "$RD.log" 2>&1 || { tail -40 "$RD.log"; exit 1; }
    CK="$(ls -t "$RD"/run/*/ckpt_last.pt | head -1)"; date
  fi
  score "$arm" "$CK" "$D/ply/$arm.ply"
  "$CPUPY" scripts/accept_gate.py --pair ss_black "$REP/curblack_base.json" "$REP/curblack_$arm.json" \
    --pair ss_grey "$REP/curgrey_base.json" "$REP/curgrey_$arm.json" \
    --pair grut "$REP/g_base.json" "$REP/g_$arm.json" --out "$REP/gate_$arm.json" | tail -2 || true
done 2>&1
fi

echo "== summary + montages"
"$CPUPY" - "$D" "$TEST" "${CROPS:-}" <<'PY'
import cv2, json, os, sys, numpy as np
D, test, crops = sys.argv[1], sys.argv[2], sys.argv[3].split()
rep = os.path.join(D, "report"); ld = lambda f: json.load(open(os.path.join(rep, f))) if os.path.isfile(os.path.join(rep, f)) else None
tags = ["base", "base_depth"] + sorted({f[9:-5] for f in os.listdir(rep) if f.startswith("curblack_") and f.endswith(".json")} - {"base", "base_depth"})
tags = [t for t in tags if ld(f"curblack_{t}.json")]
L = os.path.join(D, "on", "data_lidar", "images")
def region_psnr(t):                     # SuperSplat renders, PSNR within the LiDAR reliability classes of the test frames
    se = {"trusted": [0., 0], "dropped": [0., 0], "no_target": [0., 0]}
    for n in sorted(os.listdir(os.path.join(test, "images"))):
        if not n.endswith(".png") or n.endswith(("_covis.png", "_mask.png")): continue
        st = os.path.splitext(n)[0]; pr = cv2.imread(os.path.join(D, "cur", "on", t, "black", st + ".png")); m = cv2.imread(os.path.join(L, st + "_depthmask.png"), 0)
        dp = os.path.join(L, st + "_depth.npy")
        if pr is None or m is None or not os.path.isfile(dp): continue
        gt = cv2.imread(os.path.join(test, "images", n)); e = ((pr.astype(np.float32) - gt) / 255.) ** 2
        e = e.mean(-1); tgt = np.load(dp) > 0
        for k, sel in (("trusted", m > 127), ("dropped", tgt & (m <= 127)), ("no_target", ~tgt)):
            se[k][0] += e[sel].sum(); se[k][1] += int(sel.sum())
    return {k: round(float(-10 * np.log10(a / b)), 3) if b else None for k, (a, b) in se.items()}
out = {"ss_black": {}, "ss_grey": {}, "pc274": {}, "alpha_hole_frac": {}, "grut": {}, "region_psnr_ss_black": {}, "gates": {}}
for t in tags:
    for k_, f_ in (("ss_black", "curblack"), ("ss_grey", "curgrey"), ("pc274", "ss")):
        r_ = ld(f"{f_}_{t}.json"); out[k_][t] = {k: r_.get(k) for k in ("psnr", "ssim", "lpips")} if r_ else {}
    vj = os.path.join(D, "cur", "on", t, "_viewer.json")
    out["alpha_hole_frac"][t] = json.load(open(vj)).get("alpha_hole_frac") if os.path.isfile(vj) else None
    g, g2 = ld(f"g_{t}.json"), ld(f"g2_{t}.json")
    if g: out["grut"][t] = {**{k: g.get(k) for k in ("psnr", "ssim", "lpips", "depth_absrel", "depth_within5pct")},
                            **({"depth_absrel_unbiased": g2.get("depth_absrel"), "depth_within5pct_unbiased": g2.get("depth_within5pct")} if g2 else {})}
    out["region_psnr_ss_black"][t] = region_psnr(t)
    gt_ = ld(f"gate_{t}.json")
    if gt_: out["gates"][t] = {"accept": gt_["accept"], "failures": gt_["failures"]}
for f, k in (("depth_reliability.json", "lidar_mask"), ):
    p = os.path.join(D, "on", "data_lidar", f)
    if os.path.isfile(p): r = json.load(open(p)); out[k] = {kk: v for kk, v in r.items() if kk != "args"}
p = os.path.join(D, "on", "data_pda", "promptda.json")
if os.path.isfile(p): r = json.load(open(p)); out["promptda"] = {k: v for k, v in r.items() if k != "per_frame"}
p = os.path.join(D, "on", "data_pda", "depth_reliability.json")
if os.path.isfile(p): out["pda_mask"] = json.load(open(p))["of_target"]
json.dump(out, open(os.path.join(rep, "summary.json"), "w"), indent=2)
f = lambda x, w=8, d=4: f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"
print("SuperSplat v3.5.2 black / grey bg, PlayCanvas 2.7.4 (pc274), 3DGRUT (AbsRel vs LiDAR, * = unbiased targets), "
      "region PSNR on v3.5.2/black by LiDAR reliability of the test pixels")
print(f"{'arm':12s}{'blk psnr':>9s}{'ssim':>7s}{'lpips':>7s}{'grey':>7s}{'holes':>7s}{'pc274':>7s} | {'grut':>7s}{'ssim':>7s}{'lpips':>7s}"
      f"{'AbsRel':>7s}{'AbsRel*':>8s} | {'trusted':>8s}{'dropped':>8s}{'noLiDAR':>8s}")
for t in tags:
    b, gy, pc, g, r = out["ss_black"][t], out["ss_grey"][t], out["pc274"][t], out["grut"].get(t, {}), out["region_psnr_ss_black"][t]
    print(f"{t:12s}{f(b.get('psnr'), 9, 2)}{f(b.get('ssim'), 7)}{f(b.get('lpips'), 7)}{f(gy.get('psnr'), 7, 2)}{f(out['alpha_hole_frac'][t], 7)}"
          f"{f(pc.get('psnr'), 7, 2)} | {f(g.get('psnr'), 7, 2)}{f(g.get('ssim'), 7)}{f(g.get('lpips'), 7)}{f(g.get('depth_absrel'), 7)}"
          f"{f(g.get('depth_absrel_unbiased'))} | {f(r['trusted'], 8, 2)}{f(r['dropped'], 8, 2)}{f(r['no_target'], 8, 2)}")
print("gates:", json.dumps(out["gates"]))
def lab(im, s):
    cv2.putText(im, s, (6, 22), 0, 0.6, (0, 0, 0), 3); cv2.putText(im, s, (6, 22), 0, 0.6, (0, 255, 255), 1); return im
def rp(sub, t, n):                                                           # v3.5.2 black render path
    return os.path.join(D, "cur", sub, t, "black", n) if sub == "on" else os.path.join(D, "cur", sub, t, n)
for sub, first_gt in (("on", True), ("off1", False), ("off2", False)):
    base = os.path.dirname(rp(sub, "base", "x"))
    if not os.path.isdir(base): continue
    names = sorted(f for f in os.listdir(base) if f.endswith(".png") and not f.startswith("_"))
    rows = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, min(5, len(names))).astype(int)]:
        ims = [lab(cv2.imread(os.path.join(test, "images", n)), "real")] if first_gt else []
        ims += [lab(cv2.imread(rp(sub, t, n)), t) for t in tags if os.path.isfile(rp(sub, t, n))]
        rows.append(np.hstack([cv2.resize(x, (400, 300)) for x in ims]))
    p = os.path.join(rep, f"montage_{sub}.jpg"); cv2.imwrite(p, np.vstack(rows)); print("montage ->", p)
rows = []                                                                    # CROPS: zoomed regions, real | arms
for c in crops:
    st, box, name = c.split(":"); x, y, w, h = map(int, box.split(","))
    sub = "on"
    for s_ in ("on", "off1", "off2"):
        if st.startswith(s_ + "/"): sub, st = s_, st[len(s_) + 1:]
    ims = [lab(cv2.imread(os.path.join(test, "images", st + ".png"))[y:y + h, x:x + w].copy(), name)] if sub == "on" else []
    ims += [lab(cv2.imread(rp(sub, t, st + ".png"))[y:y + h, x:x + w].copy(), t) for t in tags if os.path.isfile(rp(sub, t, st + ".png"))]
    rows.append(np.hstack([cv2.resize(i, (300, int(300 * h / w)), interpolation=cv2.INTER_CUBIC) for i in ims]))
if rows:
    Wm = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, Wm - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(os.path.join(rep, "crops.jpg"), np.vstack(rows)); print("crops ->", os.path.join(rep, "crops.jpg"))
PY
echo DEPTHPRIOR_BENCH_DONE
