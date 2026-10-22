#!/usr/bin/env bash
# Tiny venv to read terrestrial-laser-scan .e57 ground truth (pye57, MIT; manylinux wheel, nothing compiled):
# scripts/datasets/fiord.py geom falls back to it when the CPU venv lacks pye57; `fiord.py align` (scan <- model Sim3,
# open3d FPFH + ICP) runs in it. CPU-only, wheels only (~1 min):
#   sbatch -A $CEAR_SLURM_ACCOUNT -p cpu -c 1 --mem=2G -t 00:15:00 env/make_e57_venv.sh
set -euo pipefail
CEAR_WS="${CEAR_WS:-/scratch4/workspace/${USER}-cear}"; E57_VENV="${E57_VENV:-$CEAR_WS/venv-e57}"
export UV_CACHE_DIR="$CEAR_WS/uv-cache" UV_LINK_MODE=copy
UV="$(command -v uv || echo /modules/opt/linux-ubuntu24.04-x86_64/uv/uv)"
[ -x "$E57_VENV/bin/python" ] || "$UV" venv --python 3.11 "$E57_VENV"
"$UV" pip install --python "$E57_VENV/bin/python" "pye57>=0.4" "numpy<2" scipy plyfile open3d pyyaml pycolmap opencv-python-headless
"$E57_VENV/bin/python" -c "import pye57; print('OK pye57', pye57.__version__ if hasattr(pye57, '__version__') else '')"
