#!/usr/bin/env bash
# Run the entire CPU pipeline (Phases 1-5) end to end and STOP at the GPU boundary.
#
#   bash scripts/run_cpu_pipeline.sh --synthetic         # on a generated fixture (no download)
#   bash scripts/run_cpu_pipeline.sh --config configs/mocap1_well-lit_trot.yaml
#
# Nothing here touches a GPU. The last two stages (training, Isaac Sim) are intentionally
# NOT run — see docs/GPU_HANDOFF.md.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODE="synthetic"
CFG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --synthetic) MODE="synthetic"; shift;;
    --config) MODE="real"; CFG="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

# activate the CPU venv / scratch paths if available; else assume python is set up
if [ -f "$REPO/env/cear_env.sh" ]; then source "$REPO/env/cear_env.sh" || true; fi
PY="$(command -v python)"

if [ "$MODE" = "synthetic" ]; then
  DATA="${CEAR_DATA:-$REPO/.synthetic}"
  echo "== generating synthetic sequence into $DATA =="
  "$PY" tests/make_synthetic_sequence.py --out "$DATA" >/dev/null
  CFG="$DATA/synthetic_config.yaml"
fi
echo "== config: $CFG =="

run(){ echo; echo "########## $1 ##########"; shift; "$PY" "$@" --config "$CFG"; }

run "Phase 1 — inventory (Gate 1)"          scripts/inventory.py
run "Phase 2 — build_poses"                 scripts/build_poses.py
run "Gate 2 / 2b — verify_reprojection"     scripts/verify_reprojection.py
run "Phase 3 — build_depth_map"             scripts/build_depth_map.py
run "Gate 3 — verify_depth_map"             scripts/verify_depth_map.py
run "Phase 4 — precondition_frames"         scripts/precondition_frames.py
run "Phase 4 — gate_frames"                 scripts/gate_frames.py
run "Phase 5 — build_collider (Gate 5 proxy)" scripts/build_collider.py

cat <<EOF

================= GPU BOUNDARY =================
CPU pipeline complete (Phases 1-5). Everything above ran without a GPU.
The next steps NEED GPUs and are NOT run automatically:
  * Phase 4 training : sbatch sbatch/train_a100.sbatch   (A100/H100)
  * Phase 6 Isaac Sim: sbatch sbatch/isaac_l40s.sbatch   (L40S)
See docs/GPU_HANDOFF.md.
===============================================
EOF
