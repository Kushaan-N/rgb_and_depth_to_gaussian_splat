# LiDAR depth supervision for 3DGRUT (2026-10-06): 42% less depth error, appearance +0.5 dB where seen

**Question.** The model-priors review ranked direct LiDAR depth supervision as the most faithful way
to improve geometry in under-observed regions. Does it help on the off-path benchmark?

**What was built** (all committed):
- `patches/3dgrut-depth-loss.patch` + `env/make_3dgrut_depth.sh`: 3DGRUT reads `<image>_depth.npy` next
  to each image and adds a trimmed (best 95%) L1 between the target ray distance and the rendered one,
  normalised by accumulated opacity. Off by default; applied to a separate worktree (main checkout
  untouched).
- `scripts/check_depth_grad.py`: 3DGRUT marks its depth backward as unfinished, so training is gated on
  a check. PASS: gradients reach positions/rotation/scale/density (not colour); one step lowers the
  depth loss 0.440 → 0.402.
- `scripts/build_depth_targets.py`: the metric LiDAR cloud z-buffered into every frame, edge/support
  filtered; 84% of training pixels (80% held-out) get a target.
- `scripts/remove_dynamic_lidar.py`: **pipeline bug found on the way** — the accumulated LiDAR cloud
  contained the operator walking around and the robot's own body. Visibility test (Removert-style):
  a point is removed when later sweeps' beams pass through it. 16 VLP-16 rings found from the data;
  879k of 3.71M points removed; after it only ~2k ghost points remained in the robot's corridor
  (`clear_robot_path.py`; 274k before). Ghost-free targets were essential: the first attempt had
  person-shaped near-depth blobs.
- `scripts/covis_masks.py` + `eval_views.py`: held-out pixels split into observed by some training
  camera (47%) vs never observed; depth error vs LiDAR at held-out poses.

**Benchmark**: mocap2, same spatial hold-out as Difix/ArtiFixer (315 train / 108 held-out frames),
30k iterations, LiDAR-seeded init, `sbatch/offpath_depthloss.sbatch` (29 min, one A100).

| λ_depth | PSNR | LPIPS ↓ | PSNR observed | PSNR never observed | depth AbsRel ↓ | within 5% ↑ |
|---|---|---|---|---|---|---|
| 0 (baseline) | 12.99 | 0.550 | 17.01 | 9.04 | 0.106 | 33% |
| 0.01 | 13.07 | 0.544 | **17.67** | 9.08 | 0.069 | 48% |
| **0.05** | 13.06 | **0.543** | 17.48 | 9.02 | **0.061** | 52% |
| 0.2 | 12.86 | 0.551 | 17.02 | 8.94 | 0.071 | 53% |

**Findings.**
- Geometry: median relative depth error at off-path views −42% (0.106 → 0.061), share within 5% of
  LiDAR 33% → 52%. This is what the collider-from-splat, floor and Isaac depth sensors see.
- Appearance: +0.5–0.7 dB on pixels a training camera observed; no change on never-observed pixels —
  a depth loss cannot create content (ArtiFixer can; see `2026-10-06_offpath_artifixer.md`).
- λ=0.2 starts to cost colour; λ=0.05 is the best balance (λ=0.01 if appearance matters most).

**Decision.** Adopt as an option (λ=0.05) once validated on-path; the two directions are
complementary: depth loss = faithful geometry where measured, world model = plausible appearance
where never seen. Next: (1) on-path regression check (every-8th split) with the depth loss;
(2) ArtiFixer3D distillation with the depth loss and on our LiDAR-seeded splat;
(3) make the dynamic-free LiDAR cloud the pipeline default (init, floor fill, collider) as its own A/B.

**Caveats.** One sequence; on-path PSNR not yet measured; depth targets come from a cloud built with
mocap poses and the LiDAR–camera calibration, so misalignment of a few pixels at depth edges is
absorbed by the edge filter and 95% trim, not removed.
