# Source this before running any CPU-side script (Phases 1-5, tests).
#   source env/cear_env.sh
#
# Everything large — the venv, the CEAR download, and all outputs — lives on
# scratch, never in $HOME or /work (per lab HPC storage policy). Point tools at
# the workspace via these env vars; do not hardcode paths in scripts.
#
# The scratch workspace is a managed Unity workspace. Recreate/extend with:
#   ws_allocate cear 30      # create (max 30 days initial)
#   ws_extend  cear 30       # extend
#   ws_list -v               # inspect
# Anything precious (final REPORT.md, metrics, chosen USD) must be copied back
# to /work before the workspace expires — scratch has no snapshots.

# --- workspace root -------------------------------------------------------
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
export CEAR_WS
export CEAR_DATA="${CEAR_DATA:-$CEAR_WS/data}"      # CEAR sequences downloaded here
export CEAR_OUT="${CEAR_OUT:-$CEAR_WS/outputs}"     # all pipeline outputs here
export CEAR_VENV="${CEAR_VENV:-$CEAR_WS/venv-cpu}"  # CPU venv (Phases 1-5)

mkdir -p "$CEAR_DATA" "$CEAR_OUT"

# --- python env -----------------------------------------------------------
if [ -f "$CEAR_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$CEAR_VENV/bin/activate"
else
  echo "[cear_env] WARNING: CPU venv not found at $CEAR_VENV" >&2
  echo "[cear_env] Create it with:  bash env/make_cpu_venv.sh" >&2
fi

# Keep any stray caches off /home.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CEAR_WS/xdg-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CEAR_WS/mpl-cache}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

echo "[cear_env] CEAR_WS=$CEAR_WS"
echo "[cear_env] CEAR_DATA=$CEAR_DATA"
echo "[cear_env] CEAR_OUT=$CEAR_OUT"
echo "[cear_env] python=$(command -v python)"
