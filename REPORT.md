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

## Sequence — REAL DATA (mocap1_well-lit_trot), downloaded 2026-09-20

- Pose source: OptiTrack `MoCap.txt`, **120 Hz**, 12212 poses, 102 s.
- Streams (Gate 1 PASS): RGB 5789 @ 59.6 Hz (640×480), IMU 400 Hz, joints 100 Hz; 97 s.
- **No `raw_rgb/`** in the download — only `rgb/` (processed). TRAP-5 A/B is moot here.
- `depth/` holds BOTH `*_depth_rgb` (5789, used) and `*_depth_event` projections;
  `raw_depth/` is depth-cam-frame. Depth 16-bit / 1 mm / 0=invalid confirmed; raw range up
  to 48 m (far noise) → 4 m truncation handles it; ~11% invalid pixels.
- Intrinsics (RGB): fx=381.05 fy=380.63 cx=316.61 cy=248.54, radtan dist
  [-0.0582, 0.0693, 0.00036, -0.00012, -0.0221] (OpenCV order).
- `realsense_timestamp.txt` is ONE filename per line (depth_rgb / depth_event / rgb),
  not columns — parser is format-agnostic. RGB and depth timestamps differ (TRAP 7, real).
- Also downloaded + arranged: mocap2 (4.88 GB) and mocap3 (5.77 GB) well-lit_trot
  (same room, different object arrangements — never joint-train across them).

## Resolved conventions (from real calibration + data)

- **TRAP 4 direction**: CEAR `T_rgb_marker`/`T_rgb_robot` map marker→rgb / robot→rgb, i.e.
  `marker_to_cam` / `robot_to_cam` (opposite of the synthetic default). Set in config.
- **Up-axis = Y**: MoCap xyz ranges [5.54, 0.055, 3.67] m — axis 1 (near-constant height)
  is up. `world_up: y` confirmed by data.
- **TRAP 3 offsets** match the calibration page exactly (RealSense +0.004611, IMU
  −0.004012); clock-drift check reports ~5.2 ms end-to-end (measured after the preamble) —
  flagged (> 2 ms tol) as a candidate for linear drift correction; Gate 2 still passes with
  constant offsets, so its impact is bounded.
- **TRAP 8 preamble** (mocap1): pitch swing ≈ 2–10 s, settle 10–16 s, locomotion from
  ≈ 16 s → `preamble_end ≈ 16.0 s` (auto-detected), excluded from training + fusion.

## Phase 2 — poses (Gate 2 / 2b) — REAL DATA: PASS

- Transform chain: `T_world_cam = up_axis(y→z) @ (T_world_marker(interp @ t) @ T_marker_cam)`,
  T_marker_cam = inv(T_rgb_marker); COLMAP stores inv(T_world_cam) world→cam scalar-first.
- **Gate 2 PASS**: mean grayscale warp err **5.59 / 255** over 20 pairs (k=10). Triptychs
  in `$CEAR_OUT/mocap1_well-lit_trot/gate2/` visually confirm floor/boxes/shelf reproject
  onto themselves. **Gate 2b PASS**: reloaded-model err identical, pose drift 8.8e-10.
- Trajectory: extent [5.54, 3.66, 0.056] m, path 31.2 m — healthy 2-D floor coverage
  (the tiny 3rd axis is the near-constant camera height, not 1-D motion).

## Phase 3 — depth map (Gate 3) — REAL DATA: PASS

- Truncation range 0.2–4.0 m; voxel 2.5 cm (Open3D tensor VoxelBlockGrid TSDF).
- Gated 467/5789 depth frames (‖ω‖ ≤ 0.28 rad/s + spatial spread; 3 more excluded as sync
  preamble — the ‖ω‖ gate already removes the fast swing, TRAP 8 catches the stationary
  ball-region frames that slip through). Fused ~447k points, ~824k mesh triangles.
- **Split-half fusion: floor Δz 0.4 mm, C2C median 7.9 mm (p95 39 mm)** — poses/sync are
  consistent (this is the LiDAR-cross-check substitute; a wrong pose/sync would blow it up).
- Floor plane at z≈0.02 m, thickness (std) ~11 mm (thin — real depth noise).
- Reprojection consistency 84% (real depth has more floaters than the 99% synthetic).
- Floor coverage within 4 m ≈ 46% (honest — 7.5×5.5 m room, 0.3 m camera; far floor is
  only fused when the trajectory passes near it). Coverage map in `$CEAR_OUT/.../gate3/`.

## Phase 4 — prep done (Gate 4 training is TODO(GPU))

- Blur/motion gating (REAL mocap1): kept **467/5789** frames (‖ω‖ ≤ 0.28 rad/s + laplacian
  ≥ 50 + spatial spread), split **408 train / 59 val** (hold every 8th). Train COLMAP model
  + depth-seeded points3D staged at `$CEAR_OUT/.../colmap_train/sparse/0`.
- **AE finding (REAL, important)**: undistorted 5789 frames (real distortion applied);
  mean-intensity range **64.8 / 255** over the trajectory → drift SIGNIFICANT → the trainer
  **must** use per-image exposure compensation / appearance embeddings (§3.3 swap). This
  confirms v3.1's upgrade of AE from contingency to requirement. AWB: still VERIFY.
- No `raw_rgb/` in this download, so the TRAP-5 A/B is N/A here.
- PSNR / SSIM / LPIPS on held-out + off-trajectory renders: TODO(GPU).

## Phase 5 — collider (Gate 5) — REAL DATA: PASS

- Provenance: TSDF marching-cubes mesh (823k tris) → decimated to 58k, floor holes patched
  (**4006 cells ≈ 40 m²** — 54% of the floor was unobserved within 4 m, patched with the
  RANSAC plane), then near-floor vertices flattened to the plane for a clean contact surface.
- **Gate 5 CPU drop-test proxy: PASS** — 100% ray hits (no holes), floor contact err
  p95 **16.3 mm** (< 30 mm). Real physics drop test is Phase 6 (Isaac, L40S).
- USD collider written in Phase 6 (no pxr in the CPU env).
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
--synthetic` and `tests/` (12 tests, all green). This validates the trap-prone math
without the real download or a GPU.

| Check | Result |
|---|---|
| Pose recovery vs ground truth (full chain) | 17 µm / 0.015° (pure SLERP/LERP residual) |
| Gate 1 (inventory) | PASS — frame counts self-consistent |
| Gate 2 (reprojection warp) | PASS — mean grayscale err 4.8/255 |
| Gate 2b (COLMAP round-trip) | PASS — reloaded-model pose drift 7e-10 |
| Gate 3 (depth map) | PASS — floor z=−0.006 m, thickness 3 mm, split-half Δz 0.4 mm, C2C median 2.3 mm, reproj 99.4% |
| Gate 5 (collider, CPU drop-test proxy) | PASS — 100% ray hits, floor contact err p95 3.5 mm |
| TRAP 7 (separate RGB/depth timestamps) | PASS — 4 ms gap parsed, depth fused at its own timestamp |
| TRAP 8 (sync-preamble exclusion) | PASS — 2 s preamble detected + excluded from training + fusion |
| TRAP 3 (clock-drift check) | PASS — residual IMU↔pose lag ~0 on the clean fixture |
| §5.3 (ΔT extrinsic refinement) | PASS — negligible/not-significant on a correct extrinsic |

Plan v3.1 (paper-reconciled): 20 pytest tests green. See `docs/GPU_HANDOFF.md` for full
status and exact GPU commands. Gates 4 and 6 require the GPU stages and are TODO(GPU).
