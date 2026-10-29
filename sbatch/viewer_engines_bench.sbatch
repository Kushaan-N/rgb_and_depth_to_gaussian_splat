#!/bin/bash
# Which viewer is THE metric, and does random-background training help what users see?
# (A) every candidate .ply rendered at the on-path held-out cameras by both viewer engines of scripts/viewer_shots.py
#     (pc274 = PlayCanvas 2.7.4 WebGL gsplat; current = today's SuperSplat editor on WebGPU, its own renderer) on
#     black / grey / white backgrounds (+ current at --scale 2), scored vs the real images (eval_views.py --pred-dir);
#     reports grey-minus-black and the alpha-hole fraction (alpha < 0.98: true alpha for current, white-vs-black
#     estimate for pc274).
# (B) the best recipe (EDGS init + Improved-GS, 4M budget = edgs_bench.sbatch's edgs_igs4m) retrained on the same
#     held-out split with model.background.color=random (3DGRUT: per-pixel random colour behind the splat while
#     training, black at eval), scored by 3DGRUT (gate vs edgs_igs4m) and in (A); accepted if not worse on black and
#     better on grey/white. Montages: on-path (real | candidates per engine/bg), off-path (camera moved MOVE), editor.
# Everything under $P/viewer_engines/. Re-running skips finished stages.
#   sbatch -A $CEAR_SLURM_ACCOUNT -o logs/viewer-engines-%j.out sbatch/viewer_engines_bench.sbatch configs/<seq>.yaml
# Env: SMOKE=1 (reference + edgs_igs4m only, no training), STAGES ("B A"), CANDS (viewer_bench/<name>.ply candidates,
#      default "edgs_igs4m edgs_igs_all edgs_igs_complete"), MOVE ("0 -0.3 1.0"), EVERY_OFF (40), ARM (edgs_igs4m_rbg).
#SBATCH -J viewer-engines
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
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$CEAR_WS/playwright-browsers}"
DDIR="${THREEDGRUT_DENSIFY_DIR:-$CEAR_WS/3dgrut-densify}"
IGS_ENV=(env TORCH_EXTENSIONS_DIR="${DENSIFY_TORCH_EXT:-$CEAR_WS/torch-ext-densify}" PYTHONPATH="$DDIR")
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; VB="$P/viewer_bench"; Y="$P/viewer_engines"; REP="$Y/report"
REF_PLY="$(get pipeline.viewer_reference_ply)"; REF_DS="$(get pipeline.viewer_reference_dataset)"
FRAME="$(get pipeline.export_frame_model)"; ITERS="$(get pipeline.train_iterations 30000)"
MOVE="${MOVE:-0 -0.3 1.0}"; EVERY_OFF="${EVERY_OFF:-40}"; ARM="${ARM:-edgs_igs4m_rbg}"; STAGES="${STAGES:-B A}"
CANDS="${CANDS:-edgs_igs4m edgs_igs_all edgs_igs_complete}"; ENGINES="pc274 current"; BGS="black grey white"
TEST="$P/onpath/test"
if [ "${SMOKE:-0}" = 1 ]; then CANDS="edgs_igs4m"; STAGES="A"; Y="$Y/smoke"; REP="$Y/report"; fi
hostname; date; mkdir -p "$REP"; nvidia-smi --query-gpu=name --format=csv,noheader || true
module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"; export PATH="$THREEDGRUT_DIR/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash env/make_viewer_venv.sh
RT="$VB/reference_dataset"                                     # text-model view of the reference reconstruction
if [ ! -f "$RT/sparse/0/images.txt" ]; then
  mkdir -p "$RT/sparse/0"; ln -sfn "$REF_DS/images" "$RT/images"
  "$CPUPY" -c "import pycolmap,sys; pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[2])" "$REF_DS/sparse/0" "$RT/sparse/0"
fi

case " $STAGES " in *" B "*)
  echo "== B. $ARM: edgs_igs4m recipe + model.background.color=random (on-path split)"
  bash env/make_3dgrut_densify.sh
  RUN="$Y/train/$ARM"; CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"
  if [ -z "$CK" ]; then
    mkdir -p "$Y/train"; date
    ( cd "$DDIR" && "${IGS_ENV[@]}" "$GPY" train.py --config-name apps/colmap_3dgut_improved.yaml path="$P/edgs/on/data" \
        out_dir="$RUN" experiment_name=run dataset.downsample_factor=1 n_iterations="$ITERS" dataset.test_split_interval=8 \
        strategy.budget=4000000 loss.use_depth=false model.background.color=random ) > "$RUN.log" 2>&1 || true
    # 3DGRUT's Renderer (end-of-training test pass) only knows black/white: it asserts after ckpt_last.pt is written
    CK="$(ls -t "$RUN"/run/*/ckpt_last.pt 2>/dev/null | head -1 || true)"; date
    [ -n "$CK" ] && [ -d "$(dirname "$CK")/ours_$ITERS" ] || { tail -40 "$RUN.log"; exit 1; }
  fi
  grep -n -A2 "^  background:" "$(dirname "$CK")/parsed.yaml" || true
  # evaluation copy: 'random' is black when not training, so store it as black for Renderer.from_checkpoint
  CKE="$(dirname "$CK")/ckpt_eval_black.pt"
  [ -f "$CKE" ] || "$GPY" -c "import torch,sys; c=torch.load(sys.argv[1],weights_only=False); c['config'].model.background.color='black'; torch.save(c,sys.argv[2])" "$CK" "$CKE"
  CK="$CKE"
  [ -f "$REP/3dgrut_$ARM.json" ] || "${IGS_ENV[@]}" "$GPY" scripts/eval_views.py --checkpoint "$CK" --dataset "$TEST" \
    --tag "on_$ARM" --json-out "$REP/3dgrut_$ARM.json" --save-dir "$Y/renders_3dgrut/$ARM" | tail -1
  "$CPUPY" scripts/accept_gate.py --pair onpath_3dgrut "$P/edgs/report/on_edgs_igs4m.json" "$REP/3dgrut_$ARM.json" \
    --out "$REP/gate_3dgrut_$ARM.json" | tail -3 || true
  mkdir -p "$Y/ply"; [ -f "$Y/ply/$ARM.ply" ] || "${IGS_ENV[@]}" "$GPY" scripts/export_gaussian_ply.py --checkpoint "$CK" --out "$Y/ply/$ARM.ply"
;; esac

case " $STAGES " in *" A "*)
  echo "== A. viewer engines x backgrounds"
  declare -A PLY=([reference]="$REF_PLY") DS=([reference]="$RT")
  for n in $CANDS; do [ -f "$VB/$n.ply" ] && { PLY[$n]="$VB/$n.ply"; DS[$n]="$TEST"; }; done
  [ -f "$Y/ply/$ARM.ply" ] && [ "${SMOKE:-0}" != 1 ] && { PLY[$ARM]="$Y/ply/$ARM.ply"; DS[$ARM]="$TEST"; }
  ALL="reference $(for n in $CANDS $ARM; do [ -n "${PLY[$n]:-}" ] && [ "$n" != reference ] && printf '%s ' "$n"; done; true)"
  for n in $ALL; do
    NF=(); [ "$n" = reference ] && NF=(--names-from "$TEST")
    for e in $ENGINES; do
      od="$Y/onpath/$e/$n"
      [ -f "$od/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --ply "${PLY[$n]}" --dataset "${DS[$n]}" "${NF[@]}" \
        --engine "$e" --bg $BGS --out-dir "$od" $([ "$e" = current ] && echo --editor) 2>&1 | grep -v "^\[viewer\] .*js:" | tail -4
      for b in $BGS; do
        [ -f "$REP/${e}_${n}_$b.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$od/$b" --dataset "$TEST" \
          --tag "${e}_${n}_$b" --json-out "$REP/${e}_${n}_$b.json" | tail -1
      done
    done
    od="$Y/onpath/current_x2/$n"                               # today's SuperSplat at 2x resolution (high-DPI view)
    [ -f "$od/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --ply "${PLY[$n]}" --dataset "${DS[$n]}" "${NF[@]}" \
      --engine current --scale 2 --out-dir "$od" 2>&1 | tail -2
    [ -f "$REP/current_x2_${n}_black.json" ] || "$GPY" scripts/eval_views.py --pred-dir "$od" --dataset "$TEST" \
      --tag "current_x2_${n}_black" --json-out "$REP/current_x2_${n}_black.json" | tail -1
    # off-path: camera moved off the recorded path, in the reference's axes (today's SuperSplat, black and grey)
    ply="${PLY[$n]}"
    if [ "$n" != reference ]; then
      ply="${PLY[$n]%.ply}_refaxes.ply"
      [ -f "$ply" ] || "$CPUPY" scripts/sim3_align_splat.py --colmap-model "$P/dataset/sparse/0" --gt-model "$FRAME" \
        --in-ply "${PLY[$n]}" --out-ply "$ply"
    fi
    od="$Y/offpath/$n"
    [ -f "$od/_viewer.json" ] || "$VPY" scripts/viewer_shots.py --ply "$ply" --dataset "$RT" --every "$EVERY_OFF" --move $MOVE \
      --engine current --bg black grey --out-dir "$od" 2>&1 | tail -2
  done

  "$CPUPY" - "$Y" "$REP" "$TEST/images" "$ARM" $ALL <<'PY'
import json, os, sys, cv2, numpy as np
Y, rep, gt, arm, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
ld = lambda f: json.load(open(f)) if os.path.isfile(f) else None
out = {}
print(f"{'splat':20s}{'engine':12s}" + "".join(f"{b + ' PSNR/SSIM/LPIPS':>26s}" for b in ("black", "grey", "white"))
      + f"{'grey-blk':>9s}{'white-blk':>10s}{'holes':>7s}")
for t in tags:
    for e in ("pc274", "current", "current_x2"):
        r = {b: ld(os.path.join(rep, f"{e}_{t}_{b}.json")) for b in ("black", "grey", "white")}
        v = ld(os.path.join(Y, "onpath", e, t, "_viewer.json")) or {}
        if not any(r.values()): continue
        row = {b: {k: x[k] for k in ("psnr", "ssim", "lpips")} for b, x in r.items() if x}
        row["alpha_hole_frac"] = v.get("alpha_hole_frac"); row["renderer"] = v.get("renderer")
        out.setdefault(t, {})[e] = row
        cell = lambda b: f"{row[b]['psnr']:9.2f}/{row[b]['ssim']:.4f}/{row[b]['lpips']:.4f}" if b in row else f"{'-':>26s}"
        d = lambda b: f"{row[b]['psnr'] - row['black']['psnr']:+.2f}" if b in row and "black" in row else "-"
        h = row["alpha_hole_frac"]
        print(f"{t:20s}{e:12s}" + "".join(f"{cell(b):>26s}" for b in ("black", "grey", "white"))
              + f"{d('grey'):>9s}{d('white'):>10s}{(f'{h:.4f}' if h is not None else '-'):>7s}")
g = ld(os.path.join(rep, f"3dgrut_{arm}.json"))
if g: out["3dgrut_" + arm] = {k: g[k] for k in ("psnr", "ssim", "lpips")}; print("3DGRUT", arm, out["3dgrut_" + arm])
json.dump(out, open(os.path.join(rep, "summary.json"), "w"), indent=2)
def lab(x, s): cv2.putText(x, s, (6, 22), 0, 0.6, (0, 0, 255), 2); return x
def montage(cells_fn, rows, out_p):
    R = [np.hstack([lab(cv2.resize(im, (400, 300)), l) for im, l in cells_fn(r) if im is not None]) for r in rows]
    w = max(r.shape[1] for r in R); R = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in R]
    cv2.imwrite(out_p, np.vstack(R)); print("montage ->", out_p)
on = sorted(f for f in os.listdir(gt) if f.endswith(".png") and "_depth" not in f and "_covis" not in f)
pick = [on[i] for i in np.linspace(0, len(on) - 1, 4).astype(int)]
for e in ("pc274", "current"):                               # real | each candidate, grey bg (holes show)
    montage(lambda n: [(cv2.imread(os.path.join(gt, n)), "real")] +
            [(cv2.imread(os.path.join(Y, "onpath", e, t, "grey", n)), f"{e} {t} grey") for t in tags], pick,
            os.path.join(Y, f"onpath_{e}_grey_montage.jpg"))
cmp = [t for t in ("edgs_igs4m", arm) if t in tags]            # black vs grey vs alpha, best vs random-bg
if cmp:
    montage(lambda n: [(cv2.imread(os.path.join(gt, n)), "real")] + sum(
        [[(cv2.imread(os.path.join(Y, "onpath", "current", t, b, n)), f"{t} {b}") for b in ("black", "grey")] +
         [(cv2.imread(os.path.join(Y, "onpath", "current", t, "_alpha", n)), f"{t} alpha")] for t in cmp], []), pick,
        os.path.join(Y, "onpath_rbg_montage.jpg"))
offd = os.path.join(Y, "offpath", tags[0], "black"); off = sorted(f for f in os.listdir(offd) if f.endswith(".png"))[:5] if os.path.isdir(offd) else []
if off:
    montage(lambda n: sum([[(cv2.imread(os.path.join(Y, "offpath", t, b, n)), f"{t} {b}") for b in ("black", "grey")]
                           for t in tags], []), off, os.path.join(Y, "offpath_montage.jpg"))
eds = [(cv2.imread(p), t) for t in tags for p in [os.path.join(Y, "onpath", "current", t, "_supersplat_editor.png")] if os.path.exists(p)]
if eds: cv2.imwrite(os.path.join(Y, "editor_montage.jpg"), np.vstack([lab(cv2.resize(i, (640, 400)), t) for i, t in eds]))
PY
;; esac
echo VIEWER_ENGINES_DONE
