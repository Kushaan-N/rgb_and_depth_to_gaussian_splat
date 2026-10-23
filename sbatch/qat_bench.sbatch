#!/bin/bash
# Compression-aware fidelity (the user opens splat-transform's .compressed.ply in SuperSplat): how much does the
# compressed file lose, and does a quantisation-aware fine-tune (scripts/finetune_ply_viewer.py finetune
# --quant-format cply: renders the decoded chunk-quantised values, straight-through) win it back?
#   attrib   matched-renderer (ss352 exact) attribution on the held-out frames for BASE and DELIV: original, simulated
#            .compressed.ply, each quantised attribute alone, and the real file decoded -> report/attrib_<name>.json
#   heldout  BASE (trained without the test frames) -> control fine-tune (FT_ARGS) and the same + --quant-format cply,
#            QITERS steps each, every 8th frame held out; every .ply and its .compressed.ply scored in the live
#            SuperSplat (black + grey, eval_views + DISTS), off-path moved views, gates
#   deliver  DELIV (all-frames deliverable) -> QAT fine-tune on all frames -> <DELIV stem>_qat.ply (+ _refaxes) and
#            their .compressed.ply in $E; live scores (in-sample, as the deliverable's own numbers), off-path, editor
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu --exclude=gpu010 --gres=gpu:1 --constraint="a100|a40" \
#       -o logs/%x-%j.out sbatch/qat_bench.sbatch configs/mocap2_well-lit_trot.yaml
# Env: STAGE ("attrib heldout deliver"), BASE (default $P/viewer_train_60k/igs4m_pc274/splat.ply), DELIV (default
#      $E/viewer_igs60k_wd_clean_ft2_solid3.ply), QITERS (3000), FT_ARGS ("--lambda-wd 0.025 --wd-log2-sigma 4", the
#      deliverable's WD recipe), QARGS (extra flags of the QAT arm only), DSCALE (1.5:4 train; scored at 1.8), TAG (qat),
#      Q (out dir, $P/qat), MOVE ("0 -0.3 1.0"), EVERY_OFF (20).
#SBATCH -J qat-bench
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; GPY="$THREEDGRUT_DIR/.venv/bin/python"; VPY="${VIEWER_VENV:-$CEAR_WS/venv-viewer}/bin/python"
SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"; PIQ_LIB="${PIQ_LIB:-$CEAR_WS/pylib-piq}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
NODE_BIN="$(ls -d "$HOME"/.nvm/versions/node/v2*/bin 2>/dev/null | sort -V | tail -1)"; export PATH="$NODE_BIN:$PATH"
export SPLAT_TRANSFORM="${SPLAT_TRANSFORM:-$CEAR_WS/tools/splat-transform/node_modules/.bin/splat-transform}"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; E="$OUT/exports"; FRAME="$(get pipeline.export_frame_model "")"
DATA="$P/onpath/data_orig"; TEST="$P/onpath/test"; Q="${Q:-$P/qat}"; REP="$Q/report"; TAG="${TAG:-qat}"
BASE="${BASE:-$P/viewer_train_60k/igs4m_pc274/splat.ply}"; DELIV="${DELIV:-$E/viewer_igs60k_wd_clean_ft2_solid3.ply}"
STAGE="${STAGE:-attrib heldout deliver}"; QITERS="${QITERS:-3000}"; DSCALE="${DSCALE:-1.5:4}"
FT_ARGS="${FT_ARGS:---lambda-wd 0.025 --wd-log2-sigma 4}"; MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-20}"
has() { [[ " $STAGE " == *" $1 "* ]]; }
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export TORCH_HOME="${TORCH_HOME:-$CEAR_WS/torch-cache}"
bash env/make_viewer_venv.sh | tail -1

cply() {  # <ply> -> <dir of ply>/<stem>.compressed.ply (splat-transform, CPU path)
  local d; d="$(dirname "$1")"; [ -f "${1%.ply}.compressed.ply" ] || \
    "$CPUPY" scripts/compress_splat.py convert --ply "$1" --out-dir "$d" --formats compressed.ply --gpu cpu | tail -1
}
dists() {  # renderdir json: + DISTS (accept_gate --perceptual)
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
PY
}
live() {  # <tag> <ply|compressed.ply>: live SuperSplat at the held-out cameras (black, grey) -> report/<tag>_<bg>.json
  local t="$1" f="$2" bg
  [ -f "$Q/shots/$t/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black grey --ply "$f" \
    --dataset "$TEST" --out-dir "$Q/shots/$t" 2>&1 | tail -1
  for bg in black grey; do
    [ -f "$REP/${t}_$bg.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$Q/shots/$t/$bg" --dataset "$TEST" --tag "${t}_$bg" \
      --json-out "$REP/${t}_$bg.json" | tail -1
    dists "$Q/shots/$t/$bg" "$REP/${t}_$bg.json"
  done
}
off() {  # <tag> <file> [--editor]: off-path moved views (black)
  local t="$1" f="$2"; shift 2
  [ -f "$Q/offpath/$t/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --engine current --bg black --ply "$f" --dataset "$TEST" \
    --every "$EVERY_OFF" --move $MOVE --out-dir "$Q/offpath/$t" "$@" 2>&1 | tail -1
}
gate() {  # <name> <ref tag> <cand tag> [extra accept_gate args]
  local n="$1" a="$2" b="$3"; shift 3
  "$CPUPY" scripts/accept_gate.py "$@" --pair black "$REP/${a}_black.json" "$REP/${b}_black.json" \
    --pair grey "$REP/${a}_grey.json" "$REP/${b}_grey.json" --out "$REP/gate_$n.json" > /dev/null \
    && echo "GATE $n ($b vs $a): ACCEPT" || echo "GATE $n ($b vs $a): reject"
}
FT() {  # in out interval [extra]
  local in="$1" o="$2" iv="$3"; shift 3
  [ -f "$o" ] || "$SPY" scripts/finetune_ply_viewer.py finetune --ply "$in" --out "$o" --dataset "$DATA" --test-split-interval "$iv" \
    --viewer ss352 --display-scale "$DSCALE" --eval-display-scale 1.8 --iters "$QITERS" --eval-dataset "$TEST" \
    --eval-interval 1000 --log-every 500 $FT_ARGS "$@" > "${o%.ply}.log" 2>&1 || { tail -30 "${o%.ply}.log"; return 1; }
  grep -E "eval \{" "${o%.ply}.log" | sed -n '1p;$p'
}
mrender() {  # <ply> <outdir> [render args]: matched renderer at the held-out viewer cameras, scored
  local f="$1" d="$2"; shift 2
  [ -f "$d.json" ] && return 0
  "$SPY" scripts/finetune_ply_viewer.py render --ply "$f" --dataset "$TEST" --out-dir "$d" --viewer ss352 --raster exact \
    --viewer-cam --display-scale 1.8 "$@" | tail -1
  "$GPY" scripts/eval_views.py --pred-dir "$d" --dataset "$TEST" --tag "$(basename "$d")" --json-out "$d.json" | tail -1
}

if has attrib; then
  echo "== attribution (matched renderer ss352 exact, display scale 1.8, held-out cameras)"
  for spec in "base=$BASE" "deliv=$DELIV"; do
    n="${spec%%=*}"; f="${spec#*=}"; A="$Q/attrib/$n"; mkdir -p "$A"; cply "$f"
    mrender "$f" "$A/orig"; mrender "${f%.ply}.compressed.ply" "$A/file"; mrender "$f" "$A/cply" --quant-format cply
    for part in pos rot scale color opacity sh; do mrender "$f" "$A/only_$part" --quant-format cply --quant-parts "$part"; done
    "$CPUPY" - "$A" "$REP/attrib_$n.json" <<'PY'
import json, os, sys
A, o = sys.argv[1:3]; r = {}
for v in sorted(x[:-5] for x in os.listdir(A) if x.endswith(".json")):
    j = json.load(open(os.path.join(A, v + ".json"))); r[v] = {k: j[k] for k in ("psnr", "ssim", "lpips")}
b = r["orig"]
for v in r: r[v].update({f"d_{k}": round(r[v][k] - b[k], 4) for k in ("psnr", "ssim", "lpips")})
json.dump(r, open(o, "w"), indent=1); [print(f"{v:14s} {e['psnr']:.3f} {e['ssim']:.4f} {e['lpips']:.4f}  d {e['d_psnr']:+.3f} {e['d_ssim']:+.4f} {e['d_lpips']:+.4f}") for v, e in r.items()]
PY
  done
fi

if has heldout; then
  echo "== held-out: control fine-tune vs quantisation-aware fine-tune ($QITERS steps, $FT_ARGS)"
  H="$Q/heldout"; mkdir -p "$H"
  FT "$BASE" "$H/ctrl.ply" 8 || true
  FT "$BASE" "$H/$TAG.ply" 8 --quant-format cply ${QARGS:-} || true
  cp -n "$BASE" "$H/base.ply" 2>/dev/null || true
  for n in base ctrl "$TAG"; do
    [ -f "$H/$n.ply" ] || continue; cply "$H/$n.ply"
    live "$n" "$H/$n.ply"; live "${n}_cply" "$H/$n.compressed.ply"
    off "${n}_cply" "$H/$n.compressed.ply"
  done
  off base "$H/base.ply"
  gate base_cply base base_cply; gate ctrl_cply base_cply ctrl_cply; gate "${TAG}_cply" base_cply "${TAG}_cply"
  gate "${TAG}_cply_vs_ctrl_cply" ctrl_cply "${TAG}_cply"; gate "${TAG}_cply_vs_base" base "${TAG}_cply"
  gate "${TAG}_cply_vs_base_perc" base "${TAG}_cply" --perceptual; gate "${TAG}_vs_ctrl" ctrl "$TAG"
fi

if has deliver; then
  echo "== deliverable: $DELIV -> QAT fine-tune on all frames"
  D="$Q/deliver"; mkdir -p "$D"; stem="$(basename "$DELIV" .ply)"; o="$E/${stem}_$TAG.ply"
  [ -f "$D/in.ply" ] || ln -sf "$DELIV" "$D/in.ply"
  [ -f "$o" ] || { FT "$DELIV" "$D/${stem}_$TAG.ply" 0 --quant-format cply ${QARGS:-} && cp "$D/${stem}_$TAG.ply" "$o"; }
  [ -n "$FRAME" ] && [ ! -f "${o%.ply}_refaxes.ply" ] && "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" \
    --gt-model "$FRAME" --in-ply "$o" --out-ply "${o%.ply}_refaxes.ply" | tail -1
  cply "$DELIV"; cply "$o"; [ -f "${o%.ply}_refaxes.ply" ] && cply "${o%.ply}_refaxes.ply"
  live deliv "$DELIV"; live deliv_cply "${DELIV%.ply}.compressed.ply"; live "deliv_$TAG" "$o"; live "deliv_${TAG}_cply" "${o%.ply}.compressed.ply"
  off deliv_cply "${DELIV%.ply}.compressed.ply" --editor; off "deliv_${TAG}_cply" "${o%.ply}.compressed.ply" --editor
  gate deliv_cply deliv deliv_cply; gate "deliv_${TAG}_cply" deliv_cply "deliv_${TAG}_cply"
  gate "deliv_${TAG}_cply_vs_deliv" deliv "deliv_${TAG}_cply"; gate "deliv_$TAG" deliv "deliv_$TAG"
  ls -la "$E/${stem}_$TAG"*
fi

echo "== summary"
"$CPUPY" - "$Q" "$TEST/images" <<'PY'
import cv2, json, os, sys, numpy as np
Q, gt = sys.argv[1:3]; rep = os.path.join(Q, "report")
tags = sorted({f.rsplit("_", 1)[0] for f in os.listdir(rep) if f.endswith(("_black.json", "_grey.json"))})
L = ["| arm | black PSNR / SSIM / LPIPS / DISTS | grey PSNR / SSIM / LPIPS / DISTS |", "|---|---|---|"]
for t in tags:
    c = []
    for bg in ("black", "grey"):
        f = os.path.join(rep, f"{t}_{bg}.json")
        j = json.load(open(f)) if os.path.exists(f) else None
        c.append("-" if j is None else f"{j['psnr']:.3f} / {j['ssim']:.4f} / {j['lpips']:.4f} / {j.get('dists', float('nan')):.4f}")
    L.append(f"| {t} | {c[0]} | {c[1]} |")
for f in sorted(x for x in os.listdir(rep) if x.startswith("gate_")):
    g = json.load(open(os.path.join(rep, f))); L.append(f"- {f[5:-5]}: {'ACCEPT' if g.get('accept') else 'reject'}")
open(os.path.join(Q, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def montage(sets, src, out, first_gt, k=4):          # rows: frames; cols: [real] + sets (+ |a-b| x4 of the last two)
    d0 = os.path.join(Q, src, sets[0], *(["black"] if src == "shots" else []))
    if not os.path.isdir(d0): return
    names = sorted(n for n in os.listdir(d0) if n.endswith(".png") and not n.startswith("_"))
    rows = []
    for n in [names[i] for i in np.linspace(0, len(names) - 1, k).astype(int)]:
        ims = [(cv2.imread(os.path.join(gt, n)), "real")] if first_gt else []
        ims += [(cv2.imread(os.path.join(Q, src, s, *(["black"] if src == "shots" else []), n)), s) for s in sets]
        ims = [(i, l) for i, l in ims if i is not None]
        if len(ims) >= 2:
            ims.append((np.clip(np.abs(ims[-1][0].astype(int) - ims[-2][0].astype(int)) * 4, 0, 255).astype(np.uint8), "|diff|x4"))
        ims = [(cv2.resize(i, (480, 360)), l) for i, l in ims]
        for i, l in ims: cv2.putText(i, l, (6, 22), 0, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack([i for i, _ in ims]))
    if rows:
        w = max(r.shape[1] for r in rows); cv2.imwrite(out, np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]))
        print("montage ->", out)
for name, sets, src, g in (("heldout", ["base", "base_cply", "qat_cply"], "shots", True), ("ctrl", ["ctrl_cply", "qat_cply"], "shots", True),
                           ("deliv", ["deliv", "deliv_cply", "deliv_qat_cply"], "shots", True),
                           ("offpath_heldout", ["base", "base_cply", "qat_cply"], "offpath", False),
                           ("offpath_deliv", ["deliv_cply", "deliv_qat_cply"], "offpath", False)):
    montage(sets, src, os.path.join(Q, f"montage_{name}.jpg"), g)
PY
echo QAT_BENCH_DONE
