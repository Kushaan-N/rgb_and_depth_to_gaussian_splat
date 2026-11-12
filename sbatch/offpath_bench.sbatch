#!/bin/bash
# Viewer-era OFF-PATH benchmark: does the splat look right from where the robot never was?  One submission per arm.
# The deliverable recipe (EDGS init + scripts/train_viewer_gsplat.py Improved-GS 4M, pc274 kernels, MS-DSSIM 0.2, --sort dual,
# then the viewer-exact WD fine-tune of scripts/finetune_ply_viewer.py) is trained on the spatial hold-out's TRAINING split
# (scripts/offpath_split.py: $P/offpath/train_data, EDGS-initialised in $P/edgs/off/data) minus the usual on-path hold-out
# frames (every 8th frame of the sequence, $P/onpath/test), and the exported PLY is scored in the LIVE SuperSplat
# (scripts/viewer_shots.py --engine current, black bg, Sort Mode depth AND radial) on
#   off-path  $P/offpath/test (108 real frames ~1.1 m from the nearest training camera): full image (eval_views.py; its
#             psnr_observed / psnr_unobserved split) and restricted to content a training camera saw (covis_masks.py,
#             scripts/masked_score.py: PSNR / composite LPIPS on the mask), and on the IBR-validation mask when it exists;
#   on-path   <OB>/onpath_test (the on-path hold-out frames inside the training region);
#   eyeball   no ground truth, for LOOKING at: 8 elevated orbit views (1.2-1.8 m, pitched down 30-45 deg at the room
#             centre; scripts/offpath_bench_prep.py) + the SuperSplat editor's default view of the PLY in the delivered
#             axes -> <OB>/shots/eyeball/<arm>/_montage.jpg.
# Arm = <train>[+<post>...]:
#   train  base | ibr<w>[_n<N>][_d<lambda>][_ftp]   ibr: LiDAR image-based-rendering pseudo-views far from the path
#          (scripts/make_lidar_ibr_views.py, N views, default 200) added to training through scripts/train_distill_gsplat.py at
#          gradient weight w; d<lambda>: LiDAR depth term at the pseudo-views (--pseudo-depth, trainer --depth-lambda);
#          ftp: the WD fine-tune also sees the pseudo-views (merged dataset, unsupervised pixels masked).
#   post   shfit[<margin>]  scripts/sh_minnorm_refit.py on the parent arm's splat (closed form, no training).
#   ARMS="base base+shfit" WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/offpath_bench.sbatch configs/<seq>.yaml
# MODE (space-separated, default "arms report"): arms | ibrval (IBR generator validated at the 108 held-out poses from the
#   training frames only -> <OB>/ibr/val, report/_ibr.*.json; then every arm in ARMS is (re)scored on that mask) | report.
# Env: ITERS (30000), FT_ITERS (2000), OB_OUT ($P/offpath_bench), SORTS ("depth radial"), TRAIN_SRC ($P/edgs/off/data),
#      IBR_ARGS (extra make_lidar_ibr_views.py views flags), IBR_TAG (suffix of the IBR view / validation dirs and of
#      the *_ibrmask reports: a new generator version next to the old one), IBR_EVERY (1; validation stride), SHFIT_ARGS.
#      A train token t<label> is name-only (e.g. ibr0.3_tv2 = the IBR_TAG=_v2 views).
#SBATCH -J offpath-bench
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010,gpu035
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 03:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}" TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; FRAME="$(get pipeline.export_frame_model "")"
OB="${OB_OUT:-$P/offpath_bench}"; REP="$OB/report"; TRAIN_SRC="${TRAIN_SRC:-$P/edgs/off/data}"
OFFTEST="$P/offpath/test"; ONTEST="$OB/onpath_test"; DS="$OB/train"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
FS="${FREESPACE:-$P/freespace/grid/offpath.npz}"; IVAL="$OB/ibr/val${IBR_TAG:-}"
ITERS="${ITERS:-30000}"; FT_ITERS="${FT_ITERS:-2000}"; SORTS="${SORTS:-depth radial}"; ARMS="${ARMS:-base}"; MODE="${MODE:-arms report}"
hostname; date; mkdir -p "$REP" "$OB/arms" "$OB/ibr" logs
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
module list 2>&1 | tail -2 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "python: $SPY ($("$SPY" --version 2>&1))"
GS() { local prof="$1"; shift; env PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-$prof" \
  TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-$prof" "$SPY" "$@"; }
lock() { until mkdir "$1" 2>/dev/null; do sleep 15; done; }

prep() {
  lock "$OB/.prep_lock"
  [ -f "$OB/scene.json" ] || "$CPUPY" scripts/offpath_bench_prep.py --train-ds "$TRAIN_SRC" --onpath-test "$P/onpath/test" \
    --cloud "$CLOUD" --sim3-json "$P/sim3.json" --out "$OB" || { rmdir "$OB/.prep_lock"; return 1; }
  [ -f "$OFFTEST/covis.json" ] || "$GPY" scripts/covis_masks.py --train "$P/offpath/train_data" --test "$OFFTEST" || true
  rmdir "$OB/.prep_lock"
}

ibr_views() {  # <N> -> V: the pseudo-dataset (virtual cameras in LiDAR free space, colours from the training frames only)
  V="$OB/ibr/views${IBR_TAG:-}_n$1"; lock "$OB/ibr/.lock${IBR_TAG:-}_n$1"
  if [ ! -f "$V/ibr.json" ]; then
    echo "== IBR pseudo-views: $1 ${IBR_ARGS:-}"; date
    # shellcheck disable=SC2086
    "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DS" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
      --scene-json "$OB/scene.json" --freespace "$FS" --n "$1" --out "$V" ${IBR_ARGS:-} > "$V.log" 2>&1 \
      || { tail -30 "$V.log"; rmdir "$OB/ibr/.lock${IBR_TAG:-}_n$1"; return 1; }
  fi
  grep -E "^IBR_VIEWS" "$V.log" | cut -c1-900 || true
  rmdir "$OB/ibr/.lock${IBR_TAG:-}_n$1"
}

RECIPE=(--test-split-interval 0 --strategy improved --budget 4000000 --viewer pc274 --sort dual --msssim-lambda 0.2)
train_arm() {  # base | ibr<w>[_n<N>][_d<lambda>][_ftp]
  local arm="$1" R="$OB/arms/$1" t W="" N=200 DL="" FTP=0; IFS=_ read -ra T <<< "$1"
  for t in "${T[@]}"; do case "$t" in
    base) ;; ibr[0-9]*) W="${t#ibr}" ;; n[0-9]*) N="${t#n}" ;; d[0-9]*) DL="${t#d}" ;; ftp) FTP=1 ;; t*) ;;
    *) echo "unknown token $t in $1"; return 1 ;; esac; done
  mkdir -p "$R"
  if [ ! -f "$R/train/splat.ply" ]; then
    mkdir -p "$R/train"; date
    if [ -z "$W" ]; then
      echo "== [$arm] train ($ITERS its): ${RECIPE[*]}"
      GS pc274 scripts/train_viewer_gsplat.py --dataset "$DS" --out "$R/train" --eval-dataset "$ONTEST" --iters "$ITERS" "${RECIPE[@]}" \
        > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    else
      ibr_views "$N" || return 1
      echo "== [$arm] train ($ITERS its) + $V at weight $W${DL:+, pseudo depth lambda $DL}"
      GS pc274 scripts/train_distill_gsplat.py --pseudo-dataset "$V" --pseudo-weight "$W" --pseudo-soft ${DL:+--pseudo-depth} -- \
        --dataset "$DS" --out "$R/train" --eval-dataset "$ONTEST" --iters "$ITERS" "${RECIPE[@]}" ${DL:+--depth-lambda "$DL"} \
        > "$R/train.log" 2>&1 || { tail -40 "$R/train.log"; return 1; }
    fi
  fi
  grep -E '^\[distill\]|^\[vgs\] (held-out|[0-9]+ gaussians|[0-9]+ training views)' "$R/train.log" | cut -c1-400 || true
  if [ ! -f "$R/splat.ply" ]; then
    local FD=(--dataset "$DS")
    if [ "$FTP" = 1 ]; then
      ibr_views "$N" || return 1
      [ -f "$V/merged/sparse/0/images.txt" ] || "$SPY" scripts/make_lidar_ibr_views.py merge --dataset "$DS" --pseudo "$V" --out "$V/merged"
      FD=(--dataset "$V/merged" --mask-dir "$V/merged/ignore_masks")
    fi
    echo "== [$arm] viewer-exact WD fine-tune ($FT_ITERS its, ${FD[*]})"; date
    env PATH="$(dirname "$SPY"):$PATH" "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$R/train/splat.ply" --out "$R/splat.ply" \
      "${FD[@]}" --test-split-interval 0 --viewer ss352 --display-scale 1.5:4 --iters "$FT_ITERS" --sort dual \
      --lambda-wd 0.025 --wd-log2-sigma 4 --eval-dataset "$ONTEST" > "$R/ft.log" 2>&1 || { tail -30 "$R/ft.log"; return 1; }
  fi
  grep -E "eval \{" "$R/ft.log" | sed -n '1p;$p' || true
}

post_arm() {  # <arm> = <parent>+<post>
  local arm="$1" parent="${1%+*}" post="${1##*+}" R="$OB/arms/$1"; mkdir -p "$R"
  [ -f "$OB/arms/$parent/splat.ply" ] || run_arm "$parent" || return 1
  [ -f "$R/splat.ply" ] && return 0
  case "$post" in
    shfit*) local MG="${post#shfit}"
      echo "== [$arm] SH minimum-norm refit of $parent ${MG:+(margin $MG)} ${SHFIT_ARGS:-}"; date
      # shellcheck disable=SC2086
      GS ss352 scripts/sh_minnorm_refit.py --ply "$OB/arms/$parent/splat.ply" --dataset "$DS" --out "$R/splat.ply" \
        ${MG:+--margin "$MG"} ${SHFIT_ARGS:-} > "$R/shfit.log" 2>&1 || { tail -30 "$R/shfit.log"; return 1; }
      grep -E "^SHFIT" "$R/shfit.log" | cut -c1-1500 ;;
    *) echo "unknown post step $post in $arm"; return 1 ;;
  esac
}

shots() {  # <out dir> <viewer_shots args...>: live SuperSplat, one retry (the editor page occasionally fails to load)
  local d="$1"; shift
  [ -f "$d/_viewer.json" ] && return 0
  "$VPY" scripts/viewer_shots.py --engine current --out-dir "$d" "$@" 2>&1 | tail -1
  [ -f "$d/_viewer.json" ] || { sleep 60; "$VPY" scripts/viewer_shots.py --engine current --out-dir "$d" "$@" 2>&1 | tail -1; }
  [ -f "$d/_viewer.json" ]
}

score() {  # <arm>: live SuperSplat scores + eyeball views
  local arm="$1" ply="$OB/arms/$1/splat.ply" m d
  for m in $SORTS; do
    d="$OB/shots/offpath_$m/$arm"
    shots "$d" --bg black --sort-mode "$m" --ply "$ply" --dataset "$OFFTEST" || { echo "[$arm] viewer failed ($m)"; return 1; }
    [ -f "$REP/$arm.offpath_$m.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$OFFTEST" --tag "$arm.offpath_$m" \
      --json-out "$REP/$arm.offpath_$m.json" | tail -1
    [ -f "$REP/$arm.offpath_${m}_covis.json" ] || "$GPY" scripts/masked_score.py --pred-dir "$d" --dataset "$OFFTEST" \
      --mask-suffix _covis.png --mask-value 255 --tag "$arm.offpath_${m}_covis" --json-out "$REP/$arm.offpath_${m}_covis.json" | tail -1
    if [ -f "$IVAL/ibr.json" ] && [ ! -f "$REP/$arm.offpath_${m}_ibrmask${IBR_TAG:-}.json" ]; then
      "$GPY" scripts/masked_score.py --pred-dir "$d" --dataset "$OFFTEST" --mask-dir "$IVAL/images" --mask-suffix _mask.png \
        --tag "$arm.offpath_${m}_ibrmask${IBR_TAG:-}" --json-out "$REP/$arm.offpath_${m}_ibrmask${IBR_TAG:-}.json" | tail -1
    fi
    d="$OB/shots/onpath_$m/$arm"
    shots "$d" --bg black --sort-mode "$m" --ply "$ply" --dataset "$ONTEST" || { echo "[$arm] viewer failed ($m)"; return 1; }
    [ -f "$REP/$arm.onpath_$m.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$ONTEST" --tag "$arm.onpath_$m" \
      --json-out "$REP/$arm.onpath_$m.json" | tail -1
  done
  d="$OB/shots/eyeball/$arm"                               # what a user sees: the editor's own Sort Mode (radial since 3.7)
  shots "$d" --bg black --ply "$ply" --dataset "$OB/eyeball" || true
  local vp="$ply"
  if [ -n "$FRAME" ]; then
    vp="$OB/arms/$arm/splat_refaxes.ply"
    [ -f "$vp" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" --in-ply "$ply" \
      --out-ply "$vp" | tail -1 || vp="$ply"
  fi
  shots "$OB/shots/editor/$arm" --bg black --ply "$vp" --dataset "$OB/eyeball" --every 1000 --editor || true
  "$CPUPY" - "$d" "$OB/shots/editor/$arm/_supersplat_editor.png" "$arm" <<'PY' || true
import os, sys
from PIL import Image, ImageDraw
d, ed, arm = sys.argv[1:4]
fs = sorted(f for f in os.listdir(d) if f.startswith("eyeball_") and f.endswith(".png"))
ims = [Image.open(os.path.join(d, f)).convert("RGB") for f in fs]
if os.path.exists(ed):
    e = Image.open(ed).convert("RGB"); ims.append(e.resize((ims[0].width, int(e.height * ims[0].width / e.width))) if ims else e)
if ims:
    w, h = ims[0].size; cols = 3; rows = -(-len(ims) // cols)
    M = Image.new("RGB", (cols * w, rows * h))
    for k, im in enumerate(ims):
        M.paste(im.crop((0, 0, w, min(h, im.height))), ((k % cols) * w, (k // cols) * h))
    ImageDraw.Draw(M).text((6, 4), arm, fill=(255, 255, 0))
    M.save(os.path.join(d, "_montage.jpg"), quality=90); print("[eyeball] montage ->", os.path.join(d, "_montage.jpg"))
PY
}

run_arm() {
  if [[ "$1" == *+* ]]; then post_arm "$1" || return 1; else train_arm "$1" || return 1; fi
  score "$1"
}

ibrval() {  # the generator at the held-out real poses, colours from the training frames only: the ceiling of this supervision
  local VA="$IVAL" m
  if [ ! -f "$VA/ibr.json" ]; then
    echo "== IBR validation at $OFFTEST poses ${IBR_ARGS:-}"; date
    # shellcheck disable=SC2086
    "$SPY" scripts/make_lidar_ibr_views.py views --dataset "$DS" --cloud "$CLOUD" --sim3-json "$P/sim3.json" \
      --scene-json "$OB/scene.json" --poses-from "$OFFTEST" --every "${IBR_EVERY:-1}" --out "$VA" ${IBR_ARGS:-} > "$VA.log" 2>&1 \
      || { tail -30 "$VA.log"; return 1; }
  fi
  grep -E "^IBR_VIEWS" "$VA.log" | cut -c1-1200 || true
  [ -f "$REP/_ibr.offpath_ibrmask${IBR_TAG:-}.json" ] || "$GPY" scripts/masked_score.py --pred-dir "$VA/images" --dataset "$OFFTEST" \
    --mask-dir "$VA/images" --mask-suffix _mask.png --tag _ibr.offpath_ibrmask --json-out "$REP/_ibr.offpath_ibrmask${IBR_TAG:-}.json" | tail -1
  for arm in $ARMS; do [ -f "$OB/arms/$arm/splat.ply" ] && { score "$arm" || true; }; done
  return 0
}

report() {
  "$CPUPY" - "$OB" $ARMS <<'PY'
import json, os, sys
OB, arms = sys.argv[1], list(dict.fromkeys(sys.argv[2:])); REP = os.path.join(OB, "report"); TAG = os.environ.get("IBR_TAG", "")
known = sorted({f.split(".offpath_")[0] for f in os.listdir(REP) if ".offpath_depth.json" in f})
arms = [a for a in arms if a in known] + [a for a in known if a not in arms]
ld = lambda f: json.load(open(os.path.join(REP, f))) if os.path.isfile(os.path.join(REP, f)) else {}
g = lambda j, *ks: " / ".join("-" if j.get(k) is None else f"{j[k]:.3f}" if "psnr" in k else f"{j[k]:.4f}" for k in ks)
L = []
for m in ("depth", "radial"):
    L += [f"\n**Sort Mode {m}** (live SuperSplat, black bg)\n",
          "| arm | off-path full PSNR / SSIM / LPIPS | off-path covis PSNR / LPIPS* | off-path IBR-mask PSNR / LPIPS* | on-path held-out PSNR / SSIM / LPIPS | gaussians |",
          "|---|---|---|---|---|---|"]
    ib = ld(f"_ibr.offpath_ibrmask{TAG}.json")
    if ib:
        L.append(f"| _IBR itself (ceiling of the pseudo-views; mask coverage {ib.get('coverage')}) | - | - | {g(ib, 'psnr_mask', 'lpips_mask')} | - | - |")
    for a in arms:
        o, c, i, n = ld(f"{a}.offpath_{m}.json"), ld(f"{a}.offpath_{m}_covis.json"), ld(f"{a}.offpath_{m}_ibrmask{TAG}.json"), ld(f"{a}.onpath_{m}.json")
        if not o:
            continue
        tj = os.path.join(OB, "arms", a.split("+")[0], "train", "train.json")
        ng = json.load(open(tj)).get("n_gaussians", "-") if os.path.isfile(tj) else "-"
        L.append(f"| {a} | {g(o, 'psnr', 'ssim', 'lpips')} | {g(c, 'psnr_mask', 'lpips_mask')} | {g(i, 'psnr_mask', 'lpips_mask')} | {g(n, 'psnr', 'ssim', 'lpips')} | {ng} |")
L.append("\n*LPIPS on a mask = prediction composited over the real frame outside the mask (comparable between arms on the same mask only).")
open(os.path.join(REP, f"summary{TAG}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
}

bash env/make_viewer_venv.sh | tail -1
prep
rc=0
for md in $MODE; do
  case "$md" in
    arms) for arm in $ARMS; do run_arm "$arm" || { echo "[$arm] FAILED"; rc=1; }; done ;;
    ibrval) ibrval || { echo "[ibrval] FAILED"; rc=1; } ;;
    report) report || rc=1 ;;
    *) echo "unknown MODE $md"; rc=1 ;;
  esac
done
date; echo "OFFPATH_BENCH_DONE rc=$rc"; exit $rc
