#!/bin/bash
# Completed object meshes as a 3D-consistent pseudo-view source for splat training (no mesh is swapped into the splat):
#   views   object meshes (sbatch/object_layers.sbatch stages evidence/candidates/finalize in <D>/layers_<MTAG>, cached
#           vision-language answers reused when present), the completed floor (object_layers.py floormap + LaMa inpaint),
#           mesh pseudo-views (object_layers.py meshviews: orbit cameras at viewer distances, soft mask = surface no training
#           camera saw, depth target) merged with the LiDAR IBR views (mergeviews, each source keeps its own weight)
#           -> $P/mesh_pseudo/<MTAG>/{mesh, merged}, contact sheet mesh/contact.jpg
#   Then train with the IBR recipe through sbatch/solid_chain.sbatch (export / ft2 / solid / solid3) + floor_fill.sbatch:
#     PSEUDO_DS=$P/mesh_pseudo/<MTAG>/merged PSEUDO_W=<w> PSEUDO_ARGS="--pseudo-depth --pseudo-until 0.6" NAME=<new> \
#     ARM_FLAGS="--strategy improved --budget 4000000 --iters 70000 --msssim-lambda 0.2 --sort dual --depth-lambda 0.1"
#   WHY=... scripts/sub.sh -A pi_donghyunkim_umass_edu -o logs/%x-%j.out sbatch/mesh_pseudo.sbatch configs/<seq>.yaml
# Env: NAME (splat whose objects / depth are used; default viewer_ibr70k_wd_clean_ft2_solid3_floor), MTAG (v9r), ASSET_GEN,
#      SRC_VLM (a layers dir whose vlm/ answers are copied in when this one has none), MESH_W (0.5), IBR_W (0.3), FORCE=1.
#SBATCH -J mesh-pseudo
#SBATCH -p gpu-highvram,gpu-preempt-highvram
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu010,gpu035
#SBATCH --constraint=a100|a40|l40s
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
set -euo pipefail
CFG="$1"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"; source env/cear_env.sh >/dev/null
CPUPY="$CEAR_VENV/bin/python"; TPY="${TRELLIS_VENV:-$CEAR_WS/venv-trellis}/bin/python"; IPY="${INPAINT_VENV:-$CEAR_WS/venv-inpaint}/bin/python"
get() { "$CPUPY" scripts/cfg_get.py "$CFG" "$@"; }
OUT="$(get paths.out_root)"; P="$OUT/pipeline"; NAME="${NAME:-viewer_ibr70k_wd_clean_ft2_solid3_floor}"; MTAG="${MTAG:-v9r}"
D="$P/object_priors/$NAME"; T="$D/layers_$MTAG"; DATA="$P/onpath/data_orig"; IV="$P/ibr_views"; MP="$P/mesh_pseudo/$MTAG"
CLOUD="$P/offpath/depth/lidar_static.ply"; [ -f "$CLOUD" ] || CLOUD="$P/depth_cloud.ply"
OL=scripts/object_layers/object_layers.py; needt() { [ "${FORCE:-0}" = 1 ] || [ ! -e "$1" ]; }
hostname; date; module list 2>&1 | tail -1; echo "python: $TPY ($("$TPY" --version 2>&1))"; mkdir -p "$T" "$MP"
if [ -n "${SRC_VLM:-}" ] && [ -d "$SRC_VLM/vlm" ] && [ ! -d "$T/vlm" ]; then cp -r "$SRC_VLM/vlm" "$T/vlm"; echo "vision-language answers copied from $SRC_VLM"; fi
# 1. meshes (no API calls here)
needt "$T/objects_layers.json" && HUNYUAN="${HUNYUAN:-0}" VLM_ENABLE=0 TAG="$MTAG" NAME="$NAME" STAGES="evidence candidates finalize" bash sbatch/object_layers.sbatch "$CFG" \
  | grep -E "final o|Traceback|Error|OBJ_LAYERS" | cut -c1-300
"$CPUPY" - "$T" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/candidates.json"))
for k, v in c.items():
    for n in ("prim", "gen"):
        if n in v and "watertight" in v[n]:
            print(f"[mesh] o{k} {n}: watertight {v[n]['watertight']}, size {v[n].get('size_m')}, seen share {v[n].get('seen_share')}")
PY
# 2. completed floor around the meshes
needt "$T/floor_map.npz" && "$TPY" "$OL" floormap --dir "$T" --cfg-json "$T/config.json" --dataset "$DATA" --sim3-json "$P/sim3.json" --map "$IV/floor_map.npz" --out "$T/floor_map.npz"
needt "$T/floor_map_inpainted.npz" && "$IPY" scripts/floor_fill_ibr.py inpaint --map "$T/floor_map.npz" --lama-pt "${LAMA_PT:-$CEAR_WS/torch-hub/lama/big-lama.pt}" | tail -1
# 3. mesh pseudo-views + merge with the LiDAR IBR views
needt "$MP/mesh/pseudo.json" && "$TPY" "$OL" meshviews --dir "$T" --cfg-json "$T/config.json" --sim3-json "$P/sim3.json" --dataset "$DATA" \
  --map "$T/floor_map_inpainted.npz" --cloud "$CLOUD" --out "$MP/mesh"
MW="${MESH_W:-0.5}"; SRCS=("$MP/mesh:$MW"); [ -f "$IV/ibr/sparse/0/images.txt" ] && SRCS+=("$IV/ibr:${IBR_W:-0.3}")
needt "$MP/merged/pseudo.json" && "$TPY" "$OL" mergeviews --dir "$T" --sources "${SRCS[@]}" --pseudo-weight "$MW" --out "$MP/merged"
ls "$MP/mesh/contact.jpg"; date; echo MESH_PSEUDO_DONE
