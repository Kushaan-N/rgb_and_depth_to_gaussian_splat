# World model on our splat (2026-10-07): visibility, not opacity, must tell it what was never seen

**Question.** ArtiFixer (Wan2.1 world model) filled never-seen regions well from its *own* weaker
reconstruction (`2026-10-06_offpath_artifixer.md`). Does it do better from *our* splat (LiDAR init +
LiDAR depth loss), and can our depth-supervised 3DGRUT be the distillation step?

**Setup.** Same off-path benchmark (mocap2, 108 held-out frames). ArtiFixer's 3DGRUT fork cannot load
our checkpoints (different gaussian feature layout), so `sbatch/offpath_artifixer_ours.sbatch` renders
all 423 frames with our 3DGRUT (`eval_views.py --save-dir --save-opacity-dir`) and swaps them in for
ArtiFixer's renders. Two opacity sources (`OPACITY=render|covis`), two distillations (ArtiFixer3D's,
ours = LiDAR init + depth loss on real + generated views).

| Version | PSNR | SSIM | LPIPS ↓ | PSNR observed | PSNR never observed |
|---|---|---|---|---|---|
| our baseline | 12.99 | 0.479 | 0.550 | 17.01 | 9.04 |
| our + depth loss | 13.06 | 0.477 | 0.543 | 17.48 | 9.02 |
| ArtiFixer3D, its own base (earlier run) | **17.36** | 0.581 | 0.467 | – | – |
| our splat, splat opacity → world model views | 12.77 | 0.519 | 0.492 | – | – |
| our splat, splat opacity → ArtiFixer3D | 14.09 | 0.585 | 0.466 | – | – |
| our splat, splat opacity → our depth distillation | 13.05 | 0.557 | 0.508 | 17.02 | 9.16 |
| our splat, **visibility** → world model views | 14.73 | 0.556 | 0.477 | – | – |
| our splat, **visibility** → ArtiFixer3D | 16.35 | **0.611** | **0.454** | – | – |
| our splat, **visibility** → our depth distillation | 15.08 | 0.594 | 0.497 | **18.44** | **10.77** |

(Depth error of the depth-distilled arms is not comparable to the others: their generated views were
supervised with LiDAR depth at the held-out poses.)

**Findings.**
- With its own opacity, our LiDAR-seeded splat makes the world model *worse*: it is opaque
  everywhere, including wrongly coloured gaussians where no camera looked, so the model keeps them.
  ArtiFixer uses low opacity as its cue to generate.
- Feeding *measured visibility* instead (training views opaque; held-out pixels no training camera saw
  transparent — `covis_masks.py`) fixes most of it: +2 dB on the generated views, +2.3 dB after
  distillation, and the best SSIM/LPIPS of any run.
- Visually (`img/2026-10-07_artifixer_our_splat.jpg`): from our splat the never-seen ramp comes out as
  an A-frame ramp in roughly the right place (its own base produced a black wedge) — structure anchored
  by LiDAR geometry — but the image is hazier, with semi-transparent doubled layers. ArtiFixer3D on its
  own base remains the sharpest and has the best PSNR.
- Our depth distillation is the first method to lift never-observed pixels (9.0 → 10.8 dB) and the
  best on observed pixels (18.4), but trails ArtiFixer's own distillation overall.

**Next.** Prune our splat's gaussians that no training camera observed before rendering for the world
model, so never-seen regions are genuinely empty (clean generation) while measured geometry stays —
expected to combine the sharpness of the first run with the faithful structure of this one.
