# World model along a denser camera path (2026-10-08): worse — long generation drifts

**Question.** On our splat + visibility mask, ArtiFixer3D left the never-seen region hazy
(`2026-10-07_artifixer_our_splat.md`). Would generating along a dense, smooth path through the
held-out region (more overlapping, hopefully more mutually consistent views) distil into a sharper splat?

**Setup** (`sbatch/offpath_artifixer_dense.sbatch`, `scripts/afx_trajectory.py`). mocap2, same off-path
split. ArtiFixer's trajectory route: generate along a path, distil (ArtiFixer3D) on the path frames +
the 315 real frames, render the distilled splat at the 108 held-out poses. Path poses use ArtiFixer's
own camera convention (verified against its transforms.json: max difference 8e-12).
DENSIFY=0 = exactly the 108 held-out poses (control); DENSIFY=3 = 3 interpolated poses between
consecutive held-out views closer than 0.5 m (423 frames).

| Path | PSNR | SSIM | LPIPS ↓ | PSNR observed | PSNR never observed |
|---|---|---|---|---|---|
| **108 held-out poses (control)** | **16.00** | **0.591** | **0.465** | 18.51 | **11.65** |
| dense, 423 frames | 13.97 | 0.580 | 0.477 | 18.52 | 9.12 |

(For reference, the val-frames route on the same inputs gave 16.35 / 0.611 / 0.454.)

**Why** (`img/2026-10-08_dense_path.jpg`: real · world-model view 108 · world-model view dense ·
distilled 108 · distilled dense): ArtiFixer generates autoregressively, and over a 423-frame sequence
its content drifts — at the same pose the dense run invents different furniture than the short run,
and neighbouring generated frames stop agreeing. Distillation averages the disagreement into blur and
loses what the short run got right; observed pixels are unchanged, never-observed pixels fall back to
the baseline (9.1 dB). Even the short run invents objects in never-seen space (a tracked vehicle where
the ramp is), but consistently enough to distil.

**Conclusion.** Keep world-model generation short and target-only; denser is not better with this
model. The haze is better attacked at distillation (real views weighted higher) than by more
generated views.
