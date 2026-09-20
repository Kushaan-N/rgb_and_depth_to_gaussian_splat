# GPU handoff — what's built, what's verified, what needs a GPU

**Read this before running anything on a GPU.** Per the project constraint, no GPU work has
been run. Everything below the "GPU boundary" is written and ready; everything above it is
implemented and **verified end-to-end on a synthetic fixture** (no download, no GPU).

## TL;DR

- The full **CPU pipeline (Phases 1-5)** is implemented and passes every Gate on a
  synthetic CEAR-format sequence with known ground truth:
  `bash scripts/run_cpu_pipeline.sh --synthetic` and `pytest -q tests/` (12 tests green).
- The two **GPU stages** (splat training, Isaac Sim) are written, syntax-checked, and
  wired to SLURM, but **not run** — they need an A100/H100 and an L40S respectively, plus
  the real CEAR download.
- The trap-prone math (quaternion/timestamp conventions, the extrinsic transform chain,
  the up-axis conversion, the COLMAP round-trip) is unit-tested and recovers the synthetic
  ground-truth camera poses to **~17 µm / 0.015°** (pure interpolation residual).

## Where things live

| What | Where |
|---|---|
| Code | this repo (`scripts/`, `configs/`, `tests/`) |
| CPU venv | `/scratch4/workspace/<user>-cear/venv-cpu` (built by `env/make_cpu_venv.sh`) |
| Scratch workspace | `/scratch4/workspace/<user>-cear` (`ws_list -v`) — 30-day, extendable |
| Data / outputs | `$CEAR_DATA`, `$CEAR_OUT` (both on scratch; see `env/cear_env.sh`) |
| Design doc | `docs/PLAN.md` (the v3 plan, verbatim) |

CPU env versions (pinned in `requirements-cpu.txt`): Python 3.11.15, numpy 2.4.6,
scipy 1.17.1, opencv 5.0.0, open3d 0.20.0, pycolmap 4.2.0.

## Verified on the synthetic fixture (latest run)

| Gate | Result |
|---|---|
| Gate 1 (inventory) | PASS — frame counts self-consistent |
| Gate 2 (reprojection warp) | PASS — mean grayscale err 4.8 / 255 |
| Gate 2b (COLMAP round-trip) | PASS — reloaded-model pose drift 7e-10 |
| Gate 3 (depth map) | PASS — floor z=-0.006 m, thickness 3 mm, split-half Δz 0.4 mm, C2C median 2.3 mm, reproj 99.4% |
| Gate 5 (collider, CPU drop-test proxy) | PASS — 100% ray hits (no holes), floor contact err p95 3.5 mm |

The synthetic generator (`tests/make_synthetic_sequence.py`) writes files in CEAR's exact
on-disk format using the exact conventions the plan documents, so the same scripts run
unchanged on real data.

## Running on REAL data (still CPU, still no GPU)

```bash
bash env/make_cpu_venv.sh                 # one-time
source env/cear_env.sh
# download one sequence + calibration into $CEAR_DATA (see scripts/fetch_data.md)
bash scripts/run_cpu_pipeline.sh --config configs/mocap1_well-lit_trot.yaml
```

Before trusting the result, resolve the **VERIFY** items (they are the fields most likely
to differ from the plan's assumptions):

1. **Gate 1 depth facts** — confirm depth is 16-bit, 1 mm/unit, 0 = invalid; image 640x480.
   Set `depth.units_per_meter` / `depth.invalid_value` in the config if different.
2. **TRAP 4 pose frame** — confirm `MoCap.txt` is the *marker* frame and the RGB-Marker
   extrinsic direction. If Gate 2 shows a consistent directional offset, flip
   `frames.extrinsic.rgb_marker_direction` or `frames.pose_frame`.
3. **TRAP 3 offset signs** — if Gate 2 shows a motion-scaling smear, flip the sign of
   `timestamps.offsets_s.rgb`.
4. **Up-axis** — OptiTrack is assumed Y-up; if the fused floor doesn't land near z=0,
   check `frames.world_up`.
5. **§2.7 truncation** — tune `depth.max_range_m` (3-5 m) from the Gate 3 coverage map.
6. **TRAP 5 A/B** — run `rgb/` vs `raw_rgb/` (config `sequence.rgb_dir`); both are
   undistorted by `precondition_frames.py`.

**Gate 2 is mandatory. Do not proceed to a GPU if it fails** — the fix is a config/convention
change, not a training problem.

## ===== GPU BOUNDARY — nothing past here has been run =====

Inputs the GPU stages consume are produced by the CPU pipeline:
`$CEAR_OUT/<seq>/colmap_train/sparse/0` (gated train frames + depth-seeded points3D),
`$CEAR_OUT/<seq>/precond/<rgb_dir>` (undistorted images),
`$CEAR_OUT/<seq>/collider/collider.ply` (physics mesh).

### Phase 4 — train the splat (A100/H100)

```bash
# one-time: clone + build 3DGRUT (see env/README_gpu.md; pin a commit, GCC<=11, CUDA 11.8+)
export THREEDGRUT_DIR=$CEAR_WS/3dgrut
sbatch sbatch/train_a100.sbatch mocap1_well-lit_trot
# then: bash scripts/export_splat_usd.sh mocap1_well-lit_trot   (PLY -> ParticleField + NuRec)
```

Decisions waiting here:
- **Depth supervision is UNVERIFIED** in 3DGRUT (§7.3). With LiDAR excluded, depth is the
  only geometric signal, so the depth-regularized trainer swap is the *expected* path, not
  a fallback — the COLMAP dataset interface is trainer-agnostic (PLY is the handoff).
- Gate 4 (`eval_novel_views.py`) reports PSNR/SSIM/LPIPS on the held-out set **and**
  off-trajectory renders (lateral 0.5 m, height 0.8 m) — the off-trajectory degradation is
  the real predictor of drivability.

### Phase 6 — Isaac Sim (L40S)

```bash
export ISAAC_PYTHON=<isaac-sim>/python.sh          # pin 6.0.x vs 6.1 (see env/README_gpu.md)
sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot drop   # Gate 5 (real drop test)
sbatch sbatch/isaac_l40s.sbatch mocap1_well-lit_trot walk   # Gate 6 (walk >= 5 m)
```

Decisions waiting here:
- **ParticleField vs NuRec** — export both, compare in-sim, keep the better one (§3.4).
- **Robot asset** — Mini Cheetah USD if the lab has one, else an Isaac Lab quadruped
  (note the substitution). `compose_stage.py:gate6_walk` has a marked TODO for wiring the
  locomotion controller.

## Open questions for the PI (from PLAN §12; ask before/at download)

1. Commanded-action logs? (highest-leverage question — ask before downloading)
2. Can new sequences be collected (commands logged, less bound/pronk blur)?
3. Has anyone attempted event-camera sim already?
4. Mini Cheetah URDF/USD ready for Isaac?
5. Which environments matter most?
6. Was RealSense AE/WB fixed or automatic during capture? (feeds §7.2)
7. Does no-LiDAR forbid LiDAR *offline for validation only*? (would restore a true sensor
   cross-check at Gate 3 at zero pipeline cost)
