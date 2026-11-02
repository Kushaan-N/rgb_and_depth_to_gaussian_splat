#!/bin/bash
# Random hyper-parameter search for the viewer-matched CEAR recipe (train_viewer_gsplat.py, pc274 kernels, BASE_FLAGS)
# on a SHORT proxy that never touches the held-out test frames: the training frames of $DS (every 8th frame removed,
# 3DGRUT's rule) become a proxy dataset $HP/train, whose every 8th frame is held out again as VALIDATION ($HP/val), and
# each trial trains ITERS (default 20000; schedules scale with iters) there and is scored by the trainer's held-out PSNR
# at the viewer camera (this rasteriser) on $HP/val. Trial t draws its flags from a seeded RNG (seed t; trial 0 = the
# defaults) -> $HP/trial_<t>/{flags,train.json,train.log}; MODE=report ranks them -> $HP/ranking.md. The best flags are
# confirmed at 60k with seeds in live SuperSplat via fidelity_methods_bench.sbatch tokens hp<t> (reads $HP/trial_<t>/flags).
#   WHY=... scripts/sub.sh -A $CEAR_SLURM_ACCOUNT --array=0-23%6 sbatch/hp_search.sbatch configs/mocap2_well-lit_trot.yaml
#   MODE=report sbatch ... sbatch/hp_search.sbatch configs/<seq>.yaml        # (or run the report step on a login node: CPU only)
# Space (log-uniform unless noted): lr x [0.5, 2] for means, scales, opacities, sh0; shN x [0.25, 2]; quats x [0.5, 2];
#   --grad-threshold [1.5e-4, 6e-4]; --ssim-lambda U[0.1, 0.4]; --msssim-lambda U[0, 0.4]; --densify-stop x U[0.6, 1.4] of
#   15000 x iters/30000; --reset-mult U{0.67, 1, 1.5, 100 (= no reset)}.
#SBATCH -J hp-search
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --exclude=gpu010
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 01:30:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; SPY="${GSPLAT_VENV:-$CEAR_WS/venv-gsplat}/bin/python"
OUT="$("$CPUPY" scripts/cfg_get.py "$CFG" paths.out_root)"; P="$OUT/pipeline"
DS="${VT_DATASET:-$P/edgs/on/data}"; HP="${HP_OUT:-$P/hp_search}"; ITERS="${ITERS:-20000}"
BASE_FLAGS="${BASE_FLAGS:---strategy improved --budget 4000000 --msssim-lambda 0.2 --sort dual}"
MODE="${MODE:-$([ -n "${SLURM_ARRAY_TASK_ID:-}" ] && echo trial || echo report)}"; T="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$HP"; hostname; date

split() {  # $DS -> $HP/train (training frames only, images symlinked) and $HP/val (every 8th of those)
  "$CPUPY" - "$DS" "$HP" <<'PY'
import os, sys
ds, hp = sys.argv[1:3]; S = os.path.join(ds, "sparse", "0")
lines = [l for l in open(os.path.join(S, "images.txt")) if not l.startswith("#")]
recs = sorted([(l.split()[9], l) for l in lines[::2] if len(l.split()) >= 10])
train = [r for k, r in enumerate(recs) if k % 8]                       # 3DGRUT's rule on the full set: test removed
val = [r for k, r in enumerate(train) if k % 8 == 0]                   # the same rule on the proxy: validation
for name, sub in (("train", train), ("val", val)):
    d = os.path.join(hp, name); os.makedirs(os.path.join(d, "sparse", "0"), exist_ok=True); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    with open(os.path.join(d, "sparse", "0", "images.txt"), "w") as f:
        f.writelines(l + "\n" for _, l in sub)
    for fn in ("cameras.txt", "points3D.txt"):
        t = os.path.join(d, "sparse", "0", fn); os.path.lexists(t) or os.symlink(os.path.join(S, fn), t)
    for n, _ in sub:
        for f in (n, os.path.splitext(n)[0] + "_depth.npy"):
            s, t = os.path.join(ds, "images", f), os.path.join(d, "images", f)
            if os.path.exists(s) and not os.path.lexists(t):
                os.symlink(s, t)
print(f"[hp] split: {len(recs)} frames -> proxy train {len(train)} (val {len(val)} held out inside it)")
PY
}

flags() {  # trial t -> flags (t = 0: defaults)
  "$CPUPY" - "$1" "$ITERS" <<'PY'
import math, sys, numpy as np
t, it = int(sys.argv[1]), int(sys.argv[2])
if t == 0:
    print(""); sys.exit()
r = np.random.default_rng(1000 + t); lu = lambda a, b: float(math.exp(r.uniform(math.log(a), math.log(b))))
lr = ",".join(f"{k}={lu(a, b):.3g}" for k, a, b in (("means", .5, 2), ("scales", .5, 2), ("quats", .5, 2), ("opacities", .5, 2),
                                                    ("sh0", .5, 2), ("shN", .25, 2)))
f = [f"--lr-mult {lr}", f"--grad-threshold {lu(1.5e-4, 6e-4):.3g}", f"--ssim-lambda {r.uniform(.1, .4):.3f}",
     f"--msssim-lambda {r.uniform(0, .4):.3f}", f"--densify-stop {int(15000 * it / 30000 * r.uniform(.6, 1.4))}",
     f"--reset-mult {r.choice([0.67, 1.0, 1.5, 100.0])}"]
print(" ".join(f))
PY
}

if [ "$MODE" = trial ]; then
  until mkdir "$HP/.split_lock" 2>/dev/null; do sleep 10; done
  [ -f "$HP/val/sparse/0/images.txt" ] || split; rmdir "$HP/.split_lock"
  module load cuda/12.4.1 gcc/9.4.0 2>/dev/null || true
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
  R="$HP/trial_$T"; mkdir -p "$R"; [ -f "$R/flags" ] || flags "$T" > "$R/flags"
  echo "== trial $T: $BASE_FLAGS $(cat "$R/flags")"; which "$SPY"; "$SPY" --version
  [ -f "$R/train.json" ] || PATH="$(dirname "$SPY"):$PATH" PYTHONPATH="$CEAR_WS/gsplat-viewer-pc274" \
    TORCH_EXTENSIONS_DIR="$CEAR_WS/torch-ext-gsplat-viewer-pc274" "$SPY" scripts/train_viewer_gsplat.py --dataset "$HP/train" \
    --out "$R" --test-split-interval 8 --eval-dataset "$HP/val" --iters "$ITERS" --viewer pc274 --seed "${SEED:-0}" $BASE_FLAGS \
    $(cat "$R/flags") > "$R/train.log" 2>&1 || { tail -30 "$R/train.log"; exit 1; }
  grep -E 'held-out|gaussians ->' "$R/train.log"; echo HP_TRIAL_DONE; exit 0
fi
"$CPUPY" - "$HP" <<'PY'
import glob, json, os, sys
hp = sys.argv[1]; rows = []
for d in sorted(glob.glob(f"{hp}/trial_*")):
    f = os.path.join(d, "train.json")
    if os.path.isfile(f):
        j = json.load(open(f)); rows.append((j.get("heldout_psnr") or 0, j.get("heldout_ssim"), j.get("heldout_psnr_truecam"),
                                             j["n_gaussians"], os.path.basename(d), open(os.path.join(d, "flags")).read().strip()))
rows.sort(reverse=True)
L = ["| trial | val PSNR (viewer cam) | SSIM | PSNR true cam | gaussians | flags |", "|---|---|---|---|---|---|"]
L += [f"| {r[4]} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5] or '(defaults)'} |" for r in rows]
open(f"{hp}/ranking.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
echo HP_REPORT_DONE
