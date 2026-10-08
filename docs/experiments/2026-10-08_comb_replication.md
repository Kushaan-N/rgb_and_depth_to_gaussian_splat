# Replication on mocap2_well-lit_comb (2026-10-08): world model on our splat is the clear winner

**Question.** Do the mocap2 (trot) findings hold on a second recording of the lab
(`mocap2_well-lit_comb`, different gait and path, 532 frames)?

**Setup.** Same jobs and settings: `offpath_split.py` (405 train / 127 held-out frames, held-out
cameras 0.71 m median / 1.34 m max from the nearest training camera; 94.5% of held-out LiDAR pixels
were seen by some training camera — an easier split than mocap2's 47%), `offpath_depthloss.sbatch`
(λ=0.05), `offpath_artifixer.sbatch`, `offpath_artifixer_ours.sbatch` (OPACITY=covis),
`onpath_ablation.sbatch`. Dynamic-point removal dropped 1.58M of 4.98M LiDAR points; depth gradcheck PASS.

**Off-path** (127 held-out frames):

| Version | PSNR | SSIM | LPIPS ↓ | PSNR observed | PSNR never observed |
|---|---|---|---|---|---|
| our baseline (LiDAR init) | 17.79 | 0.682 | 0.409 | 17.91 | 12.46 |
| + LiDAR depth loss | 18.08 | 0.696 | 0.396 | 18.25 | 12.74 |
| ArtiFixer3D, its own base (14.16 dB) | 17.12 | 0.579 | 0.460 | – | – |
| world model on our splat + visibility (views) | 20.58 | 0.743 | 0.332 | – | – |
| **world model on our splat + visibility → ArtiFixer3D** | **21.97** | **0.803** | **0.299** | – | – |
| world model on our splat + visibility → our depth distillation | 21.32 | 0.782 | 0.341 | 21.59 | **20.20** |

Depth error of rendered depth vs LiDAR (median rel.): 0.085 → 0.015 with the depth loss.

**On-path** (every-8th split):

| Arm | PSNR | SSIM | LPIPS ↓ | depth AbsRel ↓ | within 5% | floor |
|---|---|---|---|---|---|---|
| A original init | 33.59 | 0.9549 | 0.1865 | 0.057 | 48% | 0.777 |
| B + depth loss | 33.19 | 0.9542 | 0.1874 | 0.007 | 94% | 0.759 |
| C dynamic-free init | 33.59 | 0.9549 | 0.1943 | 0.048 | 52% | 0.691 |
| D dynamic-free init + depth | 33.06 | 0.9526 | 0.1960 | 0.007 | 94% | 0.707 |

**Findings across both recordings.**
- *World model on our splat + measured visibility* is the best method: here +4.2 dB / +0.12 SSIM /
  −0.11 LPIPS over our baseline and +4.9 dB over ArtiFixer on its own reconstruction; on mocap2 16.35 vs
  17.36 dB (its own base) but best SSIM/LPIPS. Visually (`img/2026-10-08_comb_world_model.jpg`) it keeps
  every measured object in place, removes the floaters on the floor, and stays sharp; ArtiFixer on its
  own base smears the floor tape and shifts objects. Our splat's LiDAR geometry is what makes it
  faithful; the visibility mask is what lets the world model repair it.
- LiDAR depth loss replicates: rendered depth ~6–8x closer to LiDAR (off- and on-path) for ±0.3 dB.
- Dynamic-free LiDAR as the splat's init hurts, now clearly (floor coverage 0.777 → 0.691): keep the
  original init; use the dynamic-free cloud only for depth targets.
- (mocap2 only) a denser generation path is worse (`2026-10-08_dense_path.md`).

**Recommended pipeline (candidate).** LiDAR init (original cloud) → 3DGRUT (+ optional depth loss for
geometry) → world-model fill: render our splat along target views with the measured-visibility mask →
ArtiFixer → ArtiFixer3D distillation. Generated content stays out of the collider (LiDAR-only).
Caveats: two recordings of one lab; targets here are held-out poses — production needs planned
off-path views; ArtiFixer weights are non-commercial.
