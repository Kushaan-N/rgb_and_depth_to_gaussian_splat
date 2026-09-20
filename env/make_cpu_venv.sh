#!/usr/bin/env bash
# Build the CPU venv for Phases 1-5 on scratch. CPU-only; needs no GPU and no
# job allocation — safe to run on a Unity login node.
#
#   bash env/make_cpu_venv.sh
#
# Uses uv (fast) if available, else falls back to python -m venv + pip.
set -euo pipefail

CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"
CEAR_VENV="${CEAR_VENV:-$CEAR_WS/venv-cpu}"
REQ="$(cd "$(dirname "$0")/.." && pwd)/requirements-cpu.txt"

mkdir -p "$CEAR_WS"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
unset UV_EXCLUDE_NEWER 2>/dev/null || true

# uv lives behind a module on Unity; try module, then PATH, then known location.
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  UV=/modules/opt/linux-ubuntu24.04-x86_64/uv/uv
fi

if [ -x "$UV" ] || command -v "$UV" >/dev/null 2>&1; then
  echo "[make_cpu_venv] using uv: $UV"
  "$UV" venv --python 3.11 "$CEAR_VENV"
  "$UV" pip install --python "$CEAR_VENV/bin/python" -r "$REQ"
else
  echo "[make_cpu_venv] uv not found; using python -m venv + pip"
  module load python/3.11.7 2>/dev/null || true
  python3.11 -m venv "$CEAR_VENV" || python3 -m venv "$CEAR_VENV"
  "$CEAR_VENV/bin/python" -m pip install --upgrade pip
  "$CEAR_VENV/bin/python" -m pip install -r "$REQ"
fi

echo "[make_cpu_venv] verifying imports..."
"$CEAR_VENV/bin/python" - <<'PY'
import numpy, scipy, cv2, open3d, pycolmap, PIL, matplotlib, tqdm, yaml, imageio
print("OK:", "numpy", numpy.__version__, "| open3d", open3d.__version__,
      "| pycolmap", pycolmap.__version__, "| cv2", cv2.__version__)
PY
echo "[make_cpu_venv] done -> $CEAR_VENV"
