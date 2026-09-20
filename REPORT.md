# REPORT — CEAR → Gaussian Splat → Isaac Sim

_This is the living record required by `docs/PLAN.md §10`. Fill each section as its Gate
passes. Sections marked TODO(GPU) require the GPU stages and are intentionally blank until
those are run._

## Versions / commits (record day one, do not upgrade mid-project)

| Component | Version / commit | Notes |
|---|---|---|
| CPU pipeline env | Python 3.11.15; numpy 2.4.6, scipy 1.17.1, opencv 5.0.0, open3d 0.20.0, pycolmap 4.2.0 | `requirements-cpu.txt` |
| 3DGRUT | TODO(GPU) — pin commit incl. cx/cy PINHOLE fix | `scripts/train_splat.sh` |
| Isaac Sim | TODO(GPU) — pin 6.0.x vs 6.1 | `scripts/compose_stage.py` |
| CUDA / GCC | TODO(GPU) — CUDA 11.8+, GCC ≤ 11 | 3DGRUT requirement |

## Sequence

- Sequence: `mocap1_well-lit_trot` (planned primary target).
- Pose source: OptiTrack (MoCap.txt) — TODO: confirm at Gate 1 on real download.
- Frame counts (RGB / depth, before → after gating): TODO.

## Phase 2 — poses (Gate 2 / 2b)

- Verified transform chain (world → RGB optical), written explicitly: TODO.
- COLMAP writer conventions (TRAP 1b): world-to-camera, scalar-first quats — round-trip
  verified at Gate 2b.
- Gate 2 reprojection images: TODO (`$CEAR_OUT/.../gate2/`).
- Gate 2b round-trip result: TODO.
- Trajectory extent + viewing-direction coverage: TODO.

## Phase 3 — depth map (Gate 3)

- Truncation range used: TODO.
- Split-half fusion agreement (floor-height diff, C2C on shared surfaces): TODO.
- Reprojection overlays: TODO.
- Coverage / holes map (unobserved-within-range fraction): TODO.

## Phase 4 — training (Gate 4) — TODO(GPU)

- Blur gating: retained frame count + ‖ω‖ histogram: TODO.
- rgb/ vs raw_rgb/ A/B (both undistorted): TODO.
- AE/AWB finding and exposure-comp route: TODO.
- PSNR / SSIM / LPIPS on held-out: TODO(GPU).
- Off-trajectory renders (lateral 0.5 m, height 0.8 m): TODO(GPU).

## Phase 5 — collider (Gate 5)

- Collider provenance (TSDF vs nvblox): TODO.
- Floor patches applied (every region logged): TODO.
- Drop-test floor height error: TODO.

## Phase 6 — Isaac Sim (Gate 6) — TODO(GPU)

- ParticleField vs NuRec in-sim comparison + chosen export: TODO(GPU).
- Walking video (≥ 5 m, no penetration): TODO(GPU).
- FPS at 640×480 on L40S: TODO(GPU).

## What does not work (the most valuable section)

- TODO: floaters, holes, above-camera-height regions, beyond-truncation surfaces,
  off-path degradation.

## CPU pipeline verification (synthetic fixture)

The full CPU pipeline (Phases 1–5, minus GPU training/sim) is exercised end-to-end on a
synthetic CEAR-format sequence with known ground truth via `scripts/run_cpu_pipeline.sh
--synthetic` and `tests/`. This validates the trap-prone math (quaternion/timestamp
conventions, transform-chain direction, COLMAP round-trip, TSDF fusion, collider floor
height) without the real download or a GPU. Results: see `tests/` output.
