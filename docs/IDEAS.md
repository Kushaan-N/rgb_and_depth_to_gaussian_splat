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
| MCMC densification, MCMC + opacity/scale reg | running (`--variant mcmc`, `mcmc_reg`) | see docs/experiments/ |

## Next, ranked by expected value per cost

1. **Fuse the mocap sequences into one splat.** mocap1–3 are the same lab recorded along different
   paths, and they already share one world frame: their LiDAR clouds overlay with a ~3.5 cm median
   nearest-neighbour gap with no alignment at all. Each path's blind spot (e.g. mocap2's loop
   centre, the source of the floor hole) is likely seen by another. Train one splat from all three —
   the "recapture" fix using data we already have. Needs joint COLMAP across sequences (cross-
   sequence pairs chosen from GT camera proximity rather than O(n²) exhaustive matching) or
   Sim3-merging per-sequence reconstructions in the shared frame. Watch for objects that moved
   between recordings (ghosting).
2. **SLAM-sequence pose support.** Only 18 of CEAR's 87 sequences are mocap; the rest ship
   `FasterLIO.txt` (LiDAR-inertial odometry) in a LiDAR/IMU body frame the pose chain can't read
   yet. Add a generic `frames.pose_extrinsic_chain` (same format as the LiDAR chain) and determine
   the frame / gravity axis / clock offset empirically with the Gate-2 warp error, as
   `diagnose_conventions.py` did for mocap. Unlocks ~69 sequences. (Downloads currently hit Google
   Drive's anonymous quota — needs a retry window or `GDRIVE_COOKIES`.)
3. **Room-cropped, colour-seeded init.** The 400k init points are sampled from the whole 12 m LiDAR
   sweep, most of it outside the room, and they start grey. Cropping to the room box gives ~5–9×
   denser seeding where it matters; colouring them by projecting into the camera images gives a
   better start. Cheap: prep-stage only.
4. **LiDAR depth loss during training.** Seeding sets where gaussians start; a depth loss keeps
   them there. Expected: fewer floaters, better geometry (and a better splat-derived collider).
   3DGRUT has no depth loss, so this needs its dataloader to serve per-frame depth and an L1 term on
   `pred_dist`. Complements #1 rather than replacing it (depth only constrains what cameras see).
5. **Fix the Z-down metric frame at the source.** On CEAR the pipeline frame is Z-down despite
   `target_up: z`. The floor tools derive up from the data and warn; anything else that assumes
   +z = up must too. A deliberate one-time change to the pose chain plus a full rerun.
6. **Isaac as a pipeline stage.** Compose `splat_metric.ply` (NuRec) with `collider/collider.obj`
   and run the collision test automatically, writing its metrics into `summary.json`, so every
   sequence gets a physics verdict with no manual Isaac step.
7. **Neural Harmonic Textures** (`colmap_3dgut_mcmc_nht`): a newer appearance model in 3DGRUT; may
   add detail. Requires `normalize_world_space`, which interacts with the metric-frame handling.
8. **Capture guidance.** For the next recording: add a slow pass that looks *into* the centre of
   the loop at two heights; that is the only source of real texture where no camera looked.

## Pipeline hygiene still open
- ~~Five copies of the Umeyama/Sim3 solve~~ — done: `scripts/sim3_utils.py`, and every stage reads
  the single `sim3.json` from prep (regression-tested: identical outputs).
- ~~Collider and COLMAP ran back-to-back in prep~~ — done: they now overlap (independent steps).
- COLMAP feature extraction/matching on the GPU would cut the slowest CPU step.
