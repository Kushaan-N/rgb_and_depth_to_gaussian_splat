#!/usr/bin/env bash
# Phase 6 (export) — trained PLY -> USD, BOTH ways (docs/PLAN.md §3.4, §9).
#
#   *** Needs the 3dgrut conda env (GPU box). Not a login-node step. ***
#
# Isaac Sim can consume two 3DGS representations from the SAME PLY, with a real tradeoff
# (§3.4): NuRec USDZ = higher splat fidelity / weaker RTX integration; ParticleField =
# better RTX integration / more flicker. Export BOTH (one flag apart) and compare in-sim at
# Gate 4/6 before committing. Record the choice + reason in REPORT.md.
set -euo pipefail

SEQ="${1:-mocap1_well-lit_trot}"
: "${CEAR_OUT:?source env/cear_env.sh first}"
: "${THREEDGRUT_DIR:?set THREEDGRUT_DIR}"
OUT="$CEAR_OUT/$SEQ"
PLY="${PLY:-$OUT/train_3dgrut/export_last.ply}"     # adjust to your trainer's PLY path
[ -f "$PLY" ] || { echo "missing PLY: $PLY (set PLY=...)"; exit 1; }

cd "$THREEDGRUT_DIR"
mkdir -p "$OUT/usd"

# The exact module path is in threedgrut/export/README.md for your pinned commit.
echo "[export] ParticleField USD:"
python -m threedgrut.export.scripts.ply_to_usd "$PLY" \
  --output_file "$OUT/usd/scene_particlefield.usdz" --format particlefield || \
  echo "  (confirm the export entrypoint/flags for your commit — see export README)"

echo "[export] NuRec USDZ:"
python -m threedgrut.export.scripts.ply_to_usd "$PLY" \
  --output_file "$OUT/usd/scene_nurec.usdz" --format nurec || \
  echo "  (confirm the export entrypoint/flags for your commit — see export README)"

echo "[export] Note: PLY->USDZ produces NO mesh. The collider mesh"
echo "         ($OUT/collider/collider.ply) is composed in as the physics body by"
echo "         scripts/compose_stage.py — the splat stays visual-only (§8)."
