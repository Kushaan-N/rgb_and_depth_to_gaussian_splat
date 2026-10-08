# How should world-model views enter training? (2026-10-08) — every way costs the seen regions

**Trigger.** In a viewer the world-model splat (ArtiFixer3D on our splat + visibility) looked clearly
worse than `gaussians_colmap.ply`. Only off-path quality had been measured. This scores on- AND off-path
against real images (`sbatch/distill_ablation.sbatch`, `scripts/build_distill_set.py`): every 8th real
training frame held out (40 trot / 51 comb), same off-path split; arms A–C use our recipe (LiDAR init +
LiDAR depth loss); D is ArtiFixer3D's own distillation (its own recipe: COLMAP-point init, MCMC, LPIPS
loss, every-8th holdout) — it trained on ~7/8 of the on-path held-out frames, so its on-path PSNR is
optimistic.

**mocap2 trot**

| Arm | on PSNR | on LPIPS ↓ | off PSNR | off LPIPS ↓ | off observed | off never observed |
|---|---|---|---|---|---|---|
| A real frames only | 36.88 | **0.163** | 12.91 | 0.547 | 17.69 | 8.96 |
| B + generated views, full weight | 36.61 | 0.172 | 15.03 | 0.497 | 18.47 | 10.72 |
| C + generated views, never-observed pixels only | 36.50 | 0.172 | 14.46 | 0.510 | 19.02 | 10.68 |
| D ArtiFixer3D | 38.38* | 0.185 | 16.35 | 0.454 | 18.94 | 12.03 |

**mocap2 comb**

| Arm | on PSNR | on LPIPS ↓ | off PSNR | off LPIPS ↓ | off observed | off never observed |
|---|---|---|---|---|---|---|
| A real frames only | 34.27 | **0.191** | 17.49 | 0.411 | 17.71 | 12.51 |
| B + generated views, full weight | 33.95 | 0.197 | 21.27 | 0.341 | 21.62 | 20.08 |
| C + generated views, never-observed pixels only | 33.91 | 0.200 | 19.18 | 0.392 | 19.49 | 20.12 |
| D ArtiFixer3D | 36.27* | 0.198 | 21.97 | 0.299 | 22.25 | 21.71 |

**Findings.**
- Any use of world-model views lowers on-path quality (−0.3 to −0.4 dB, LPIPS +0.006 to +0.009 for
  B/C); ArtiFixer3D has the worst on-path LPIPS of all despite having trained on those frames — the
  blur seen in the viewer.
- Masking generated views to never-observed pixels (C) does not remove that cost and gives up part of
  the off-path gain.
- The off-path gains are real but come from regions the user does not need fixed: in the viewer the
  plain splats are already clear except the backs of unseen objects.

**Decision.** World-model fill is not used in the pipeline. The remaining defect — unseen object backs —
is addressed object by object with measured geometry (`2026-10-08_object_completion.md`, in progress).
