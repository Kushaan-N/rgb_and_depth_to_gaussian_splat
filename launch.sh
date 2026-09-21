#!/bin/bash
# One launcher for the CEAR -> Gaussian Splat -> Isaac Sim project.
#
#   ./launch.sh view              # build the results gallery + serve it (open the printed URL)
#   ./launch.sh open              # just (re)build results/results.html (open it in VS Code)
#   ./launch.sh robot [args...]   # GPU: render the robot navigating the photoreal splat
#   ./launch.sh walkthrough       # GPU: render the photoreal fly-through of the world
#   ./launch.sh live              # GPU: LIVE interactive Isaac Sim session (WebRTC stream)
#   ./launch.sh scene <seq>       # run the CPU pipeline on another CEAR sequence (cleaner scene)
#   ./launch.sh status            # show your queued/running jobs
#   ./launch.sh help
#
# GPU jobs run under the pi_donghyunkim_umass_edu account (high fairshare) on the 6.1.0 image.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"
SIF="${CEAR_SIF:-$CEAR_WS/isaac-sim-6.1.0.sif}"
ACCT="${CEAR_ACCT:-pi_donghyunkim_umass_edu}"
SEQ="${SEQ:-mocap1_well-lit_trot}"
CMD="${1:-help}"; shift || true

sub() {  # submit a GPU sbatch under the right account + 6.1.0 image
  local script="$1"; shift
  sbatch -A "$ACCT" --export=ALL,CEAR_SIF="$SIF" "$REPO/sbatch/$script" "$@"
}

case "$CMD" in
  open)
    python3 "$REPO/scripts/make_gallery.py" --seq "$SEQ" --out "$REPO/results/results.html"
    echo "Open in VS Code: right-click results/results.html -> 'Open with Live Preview' or Simple Browser,"
    echo "or run './launch.sh view' to serve it over a forwarded port." ;;

  view)
    python3 "$REPO/scripts/make_gallery.py" --seq "$SEQ" --out "$REPO/results/results.html"
    PORT="${PORT:-8000}"
    echo "============================================================"
    echo " Serving results at:   http://localhost:$PORT/results.html"
    echo " (VS Code Remote auto-forwards the port — Cmd/Ctrl-click the URL,"
    echo "  or check the PORTS tab. Ctrl-C to stop.)"
    echo "============================================================"
    cd "$REPO/results" && exec python3 -m http.server "$PORT" ;;

  robot)
    echo "submitting robot-in-splat render (6.1.0)…"
    sub nav_in_splat.sbatch "$SEQ" walkthrough.tum navsplat \
        --warmup 400 --cam-index 8 --look-ahead 45 --robot-count 48 --robot-mode sweep \
        --sweep-dist 0.6 --sweep-range 0.45 --robot-drop 0.12 --robot-size 0.22 "$@" ;;

  walkthrough)
    echo "submitting photoreal walkthrough render (6.1.0)…"
    sub nurec_render_test.sbatch "$SEQ" walkthrough.tum walkthrough warmup_fast.yaml "$@" ;;

  live)
    echo "submitting LIVE interactive Isaac Sim session (WebRTC)…"
    sub isaac_live.sbatch "$SEQ" "$@"
    echo "Once RUNNING, see the job's .out for the connect URL + the SSH port-forward command." ;;

  scene)
    NS="${1:?usage: ./launch.sh scene <sequence_name>}"
    echo "Run the full pipeline on '$NS'. CPU stages first:"
    echo "  source env/cear_env.sh && bash scripts/run_cpu_pipeline.sh $NS"
    echo "then GPU: sbatch/build_3dgrut, train_a100, export_usd, then './launch.sh robot' with SEQ=$NS."
    echo "See docs/RUN.md 'Cleaner scene' for the full recipe." ;;

  status)
    squeue -u "$USER" -o "%.12i %.14P %.16j %.8T %.10M %.12l %R" ;;

  help|*)
    sed -n '2,20p' "$0" ;;
esac
