# On-path ablation (2026-10-07): depth loss 7x more accurate geometry; dynamic-free init no gain

**Question.** Before changing defaults: does the LiDAR depth loss hurt the usual on-path quality,
and should the dynamic-free LiDAR cloud (`remove_dynamic_lidar.py` + `clear_robot_path.py`) replace
the original cloud as the splat's init?

**Setup.** mocap2, all 423 frames, the trainer's usual every-8th held-out split (53 frames), 30k
iterations, `sbatch/onpath_ablation.sbatch` (46 min). Depth targets from the dynamic-free cloud.
Floor coverage on the raw splat (before floor fill), room defined by the original cloud.

| Arm | init | depth loss | PSNR | SSIM | LPIPS ↓ | depth AbsRel ↓ | within 5% | floor |
|---|---|---|---|---|---|---|---|---|
| A | original | – | 36.34 | 0.9667 | 0.1640 | 0.0455 | 54% | 0.781 |
| B | original | λ=0.05 | 36.12 | 0.9661 | **0.1627** | **0.0068** | **95%** | 0.775 |
| C | dynamic-free | – | 36.37 | 0.9669 | 0.1676 | 0.0412 | 57% | 0.766 |
| D | dynamic-free | λ=0.05 | 35.94 | 0.9655 | 0.1676 | 0.0069 | 95% | 0.763 |

**Findings.**
- Depth loss: rendered depth at held-out views matches LiDAR ~7x better (AbsRel 0.046 → 0.007;
  95% of pixels within 5%) for −0.2 dB PSNR, LPIPS slightly better, floor coverage −0.6 points.
  Off-path it was −42% depth error and +0.5 dB where observed (`2026-10-06_lidar_depth_loss.md`).
- Dynamic-free init: no image gain (+0.03 dB), LPIPS +0.004 and floor −1.5 points worse — training
  already prunes most ghost points, and visibility removal also drops some real points. Keep the
  original cloud for init; use the dynamic-free cloud where ghosts are harmful (depth targets; the
  collider is the open question).

**Decision.** Depth loss: offer as a pipeline option (`--variant depth`), recommended for Isaac
worlds where geometry matters (collider-from-splat, depth sensors); λ=0.05. Init: unchanged.
