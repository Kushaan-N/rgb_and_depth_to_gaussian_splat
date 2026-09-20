# rgb_and_depth_to_gaussian_splat

Reconstruct a **drivable, physics-enabled Isaac Sim scene** from one CEAR quadruped
sequence using only **RGB + depth + poses + IMU** (no LiDAR — project constraint), then
walk a quadruped through it.

```
CEAR sequence (RGB + depth + poses + IMU)
  → aligned per-frame camera poses (COLMAP sparse model)      [Phase 2]
  → fused depth map ─┬→ points3D seed → 3D Gaussian Splat     [Phase 3 → 4]
                     └→ collider mesh (physics)               [Phase 5]
  → USD scene → Isaac Sim → quadruple walks                   [Phase 6]
```

Full design, rationale, and the numbered TRAPs/Gates live in
[`docs/PLAN.md`](docs/PLAN.md). **Read `docs/PLAN.md §0` first.** Two data problems —
camera-pose accuracy and motion blur — dominate success; the Gates exist to catch them.

## Where things live

- **Code** is in this repo (`scripts/`, `configs/`, `tests/`).
- **Data, venvs, and outputs** live on **scratch**, never in the repo or `$HOME`
  (Unity workspace policy). `env/cear_env.sh` wires the paths.

## CPU vs GPU split

Phases 1–5 and all tests are **CPU-only** and run on a login node. Only two things need a
GPU, and they are isolated behind their own scripts and environments:

| Stage | Needs | Script |
|---|---|---|
| Phase 4 — splat **training** | A100/H100 | `scripts/train_splat.sh` (+ `sbatch/train_a100.sbatch`) |
| Phase 6 — **Isaac Sim** | L40S | `scripts/compose_stage.py` (+ `sbatch/isaac_l40s.sbatch`) |

**This repo does not launch GPU work automatically.** The CPU pipeline runs and is tested
end-to-end on a synthetic fixture; the GPU scripts are written and ready but must be
launched deliberately.

## Quickstart (CPU)

```bash
# 1. one-time: build the CPU venv on scratch
bash env/make_cpu_venv.sh

# 2. every session
source env/cear_env.sh

# 3. prove the whole CPU pipeline works on a synthetic sequence (no download, no GPU)
bash scripts/run_cpu_pipeline.sh --synthetic
pytest -q tests/
```

`run_cpu_pipeline.sh --synthetic` generates a tiny CEAR-format sequence with **known
ground truth**, then runs inventory → pose build → Gate 2/2b → depth fusion → Gate 3 →
preconditioning → blur gating → collider mesh, asserting each gate.

## Real data

```bash
# download one sequence + calibration into $CEAR_DATA (see scripts/fetch_data.md)
source env/cear_env.sh
python scripts/inventory.py      --config configs/mocap1_well-lit_trot.yaml   # Phase 1 / Gate 1
python scripts/build_poses.py    --config configs/mocap1_well-lit_trot.yaml   # Phase 2
python scripts/verify_reprojection.py --config configs/mocap1_well-lit_trot.yaml  # Gate 2 + 2b
python scripts/build_depth_map.py     --config configs/mocap1_well-lit_trot.yaml  # Phase 3
python scripts/verify_depth_map.py    --config configs/mocap1_well-lit_trot.yaml  # Gate 3
python scripts/precondition_frames.py --config configs/mocap1_well-lit_trot.yaml  # Phase 4
python scripts/gate_frames.py         --config configs/mocap1_well-lit_trot.yaml  # Phase 4
python scripts/build_collider.py      --config configs/mocap1_well-lit_trot.yaml  # Phase 5
# optional, only if Gate 2 passes but geometry is soft (§5.3):
python scripts/refine_extrinsic.py    --config configs/mocap1_well-lit_trot.yaml
# --- GPU boundary: everything above is CPU. Stop here and hand off. ---
```

**Do not proceed past `build_collider.py` onto a GPU without explicit approval.** See
`docs/GPU_HANDOFF.md` for exactly what is ready and what the GPU steps will do.

## Status

Plan **v3.1** (paper-reconciled). CPU pipeline (Phases 1–5) implemented and tested on the
synthetic fixture — 20 pytest tests green, including the v3.1 traps: separate RGB/depth
timestamps (TRAP 7), sync-preamble exclusion (TRAP 8), clock-drift check (TRAP 3), and the
constant-extrinsic ΔT refinement (§5.3). GPU scripts (Phases 4-train, 6) written but not
run. See `REPORT.md` and `docs/GPU_HANDOFF.md`.
