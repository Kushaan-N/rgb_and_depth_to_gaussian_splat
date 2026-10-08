# Ideas & roadmap (evidence-ranked)

Every item here is either measured already or ranked by what the measurements imply. Anything
adopted must stay automatic and config-driven (see docs/RUN.md §6) — no per-scene fixes.

## What the experiments have settled (mocap2 held-out frames unless noted)

| Idea | Result | Status |
|---|---|---|
| COLMAP-optimized poses vs raw OptiTrack GT | PSNR 26.2 → 37.2; mocap1 26.2 → 36.5 | **adopted** |
| Refine intrinsics vs fix them | 36.96 vs 37.15 dB — intrinsics barely matter, poses do | settled |
| Denser frames into COLMAP (every 4th, ~1400) | 36.70 dB — no gain over the gated set | rejected |
| Longer training (50k iters) | +0.2 dB | not worth the GPU time |
| LiDAR-seeded init | LPIPS 0.196 → 0.165; floor coverage 51% → 81% | **adopted** |
| LiDAR collider vs RGB-D colliders | floor 7/9, walls 4/4, 0 falls (vs floor 1–5/9) | **adopted** |
| Constant marker→camera correction (ΔT) | 17.5% warp gain but 7.3°/234 mm — overfit | rejected |
| Re-syncing the RGB time offset | best offset −2.5 ms → only 6.2% warp gain | marginal |
| Exposure / colour compensation (PPISP) | colour-corrected PSNR only +0.04–0.15 dB above plain | ruled out without a run |
| Floor fill from depth | floor 78% → 83% coverage, objects left alone | **adopted (optional stage)** |
| MCMC densification | PSNR +0.8–2.3 dB but LPIPS worse and floor 83% → 64% (3/3 seqs) | rejected as default |
| Fuse mocap1–3 into one splat | held-out −9 to −12 dB: objects were rearranged between recordings | rejected (stage kept, with gates) |
| Difix3D+ (NVIDIA) on a 1 m spatial hold-out | +0.7 dB / LPIPS −0.07 off-path, but fills unseen content with wrong objects | not adopted (benchmark kept) |
| ArtiFixer world-model fill (NVIDIA) on the same hold-out | 13.0 → 17.4 dB, LPIPS 0.55 → 0.47; fills unseen regions mostly faithfully | **promising — next: on our splat + planned off-path trajectories** |
| LiDAR depth loss (patched 3DGRUT, λ=0.05) | off-path depth error −42%, +0.5 dB observed; on-path depth error 0.046 → 0.007 for −0.2 dB | **adopt as option** |
| Dynamic-point removal from the LiDAR cloud | removes the operator + robot-body ghosts (879k of 3.7M pts); as splat init: no gain (floor −1.5 pts) | used for depth targets; init unchanged |
| World model on our splat (LiDAR init + depth) | splat opacity misleads it (12.8 dB views); measured visibility instead: 16.35 dB, best SSIM/LPIPS, ramp shape right but hazy; pruning unobserved gaussians instead: worse (14.05 — loses the LiDAR anchors) | keep anchors + visibility mask; next: denser camera path for the world model |
| World model along a denser path (K=3) | 16.00 → 13.97 dB: long autoregressive generation drifts | rejected |
| Replication on mocap2_well-lit_comb | world model on our splat + visibility 17.79 → 21.97 dB (vs 17.12 own base); depth loss replicates; dynamic-free init floor −8.6 pts | **world model on our splat = best method on both** |
| World-model views in training, scored on-path too | every variant costs seen regions (−0.3–0.4 dB, worse LPIPS); ArtiFixer3D blurriest on-path | **world models dropped**; object-level completion instead |
| MCMC + opacity/scale regularisation | best LPIPS (−0.006–0.009), floor unchanged, PSNR −0.8 dB | available: `--variant mcmc_reg` |

## Next, ranked by expected value per cost

1. **SLAM-sequence pose support.** Only 18 of CEAR's 87 sequences are mocap; the rest ship
   `FasterLIO.txt` (LiDAR-inertial odometry) in a LiDAR/IMU body frame the pose chain can't read
   yet. Add a generic `frames.pose_extrinsic_chain` (same format as the LiDAR chain) and determine
   the frame / gravity axis / clock offset empirically with the Gate-2 warp error, as
   `diagnose_conventions.py` did for mocap. Unlocks ~69 sequences. (Downloads currently hit Google
   Drive's anonymous quota — needs a retry window or `GDRIVE_COOKIES`.)
2. **Room-cropped, colour-seeded init.** The 400k init points are sampled from the whole 12 m LiDAR
   sweep, most of it outside the room, and they start grey. Cropping to the room box gives ~5–9×
   denser seeding where it matters; colouring them by projecting into the camera images gives a
   better start. Cheap: prep-stage only.
3. ~~LiDAR depth loss during training~~ — done (`docs/experiments/2026-10-06_lidar_depth_loss.md`):
   −42% depth error off-path, +0.5 dB where observed. Next: combine with ArtiFixer3D distillation.
4. **Fix the Z-down metric frame at the source.** On CEAR the pipeline frame is Z-down despite
   `target_up: z`. The floor tools derive up from the data and warn; anything else that assumes
   +z = up must too. A deliberate one-time change to the pose chain plus a full rerun.
5. **Isaac as a pipeline stage.** Compose `splat_metric.ply` (NuRec) with `collider/collider.obj`
   and run the collision test automatically, writing its metrics into `summary.json`, so every
   sequence gets a physics verdict with no manual Isaac step.
6. **Neural Harmonic Textures** (`colmap_3dgut_mcmc_nht`): a newer appearance model in 3DGRUT; may
   add detail. Requires `normalize_world_space`, which interacts with the metric-frame handling.
7. **Capture guidance.** For the next recording: add a slow pass that looks *into* the centre of
   the loop at two heights; that is the only source of real texture where no camera looked.
8. **Static-only use of other recordings.** Fusion failed because movable objects differ between
   recordings (docs/experiments/2026-09-29_fusion.md). Walls/floor still agree to ~1 cm, so the other
   members' LiDAR floor could feed the floor fill/collider. Low priority: per-member floor is ~83%.

9. **Generative fill of unseen regions — anchored, not free.** Difix3D+ showed off-path quality is
   limited by content no camera saw (13 dB on a 1 m hold-out), and that a diffusion fixer fills it
   with plausible fiction. Candidates from the literature review (2026-10-02): ArtiFixer (NVIDIA,
   SIGGRAPH 2026; video diffusion that fills low-opacity regions, ships a 3DGRUT fork; needs an
   80 GB A100 and its MoGe scale step bypassed), GSFix3D (indoor, refines an existing splat,
   conditions on a mesh — our LiDAR mesh), object-level amodal completion (RecGen, SAM 3D Objects)
   for the backs of stacks vs a LiDAR cuboid baseline. Any generated gaussians must be tagged, kept
   out of the collider and checked against LiDAR free space. Evaluate with the off-path split plus a
   co-visibility mask and a consistency metric (TSED / MEt3R).
10. **Real observations beat priors.** A short extra recording that looks at the backs of the
   stacks, fused like IGFuse (multi-scan fusion that tolerates moved objects), is the faithful fix.

## Pipeline hygiene still open
- ~~Five copies of the Umeyama/Sim3 solve~~ — done: `scripts/sim3_utils.py`, and every stage reads
  the single `sim3.json` from prep (regression-tested: identical outputs).
- ~~Collider and COLMAP ran back-to-back in prep~~ — done: they now overlap (independent steps).
- COLMAP feature extraction/matching on the GPU would cut the slowest CPU step.
