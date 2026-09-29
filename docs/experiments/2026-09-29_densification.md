# Densification strategy: default vs MCMC vs MCMC + regularisation (2026-09-29)

**Question.** Does 3DGRUT's MCMC densification — which relocates dead gaussians instead of pruning
them — or MCMC plus NVIDIA's opacity/scale regularisation beat the pipeline default on our data?

**Setup.** `bash scripts/run_pipeline.sh <seq> --variant {mcmc,mcmc_reg}` on mocap1/2/3 (variant
configs in `configs/variants/`). Each variant reuses that sequence's prep outputs, so within a
sequence all three arms share identical training frames, held-out frames (every 8th), COLMAP poses,
LiDAR-seeded init and 30k iterations — a clean A/B. Floor coverage is measured after the floor fill
(`floor_coverage.py`, fraction of the room floor with reconstructed floor).

| seq | variant | PSNR | SSIM | LPIPS ↓ | gaussians | floor | fill added |
|---|---|---|---|---|---|---|---|
| mocap1 | default | 34.53 | 0.9640 | 0.1718 | 950,481 | 0.823 | 550 |
| mocap1 | mcmc | **36.79** | **0.9681** | 0.1908 | 1,000,000 | 0.630 | 8,443 |
| mocap1 | mcmc_reg | 33.65 | 0.9611 | **0.1633** | 1,000,000 | 0.818 | 416 |
| mocap2 | default | 36.26 | 0.9669 | 0.1621 | 944,758 | 0.833 | 1,482 |
| mocap2 | mcmc | **37.15** | **0.9696** | 0.1795 | 1,000,000 | 0.659 | 8,336 |
| mocap2 | mcmc_reg | 35.49 | 0.9634 | **0.1558** | 1,000,000 | 0.836 | 1,097 |
| mocap3 | default | 33.87 | 0.9603 | 0.1778 | 1,005,462 | 0.842 | 444 |
| mocap3 | mcmc | **34.64** | **0.9625** | 0.1917 | 1,000,000 | 0.629 | 7,043 |
| mocap3 | mcmc_reg | 32.98 | 0.9561 | **0.1721** | 1,000,000 | 0.839 | 525 |

**Findings (same direction on all three sequences).**
- **MCMC**: PSNR +0.8 to +2.3 dB and slightly higher SSIM, but LPIPS worse by 0.014–0.019 and floor
  coverage falls from 82–84% to 63–66% even after the fill adds ~7–8k patch gaussians. MCMC moves
  its fixed budget toward well-observed surfaces — which the held-out frames (also on the robot's
  path) reward — and away from the barely-seen floor, which is exactly the navigation-relevant part.
- **MCMC + regularisation**: best LPIPS on every sequence (−0.006 to −0.009 vs default) with floor
  coverage equal to default, at a cost of 0.8–0.9 dB PSNR.

**Decision.** Keep the default (`apps/colmap_3dgut.yaml`). MCMC wins the headline metric while
making the floor worse; for a robot sim the floor matters more than trajectory-view PSNR.
`mcmc_reg` is the choice when perceptual quality is the priority (one flag: `--variant mcmc_reg`).

**Caveats.** Metrics are the trainer's own held-out views, which lie on the capture path; none of
these numbers measures quality far from the path (where MCMC's floor loss would show most). The MCMC
app also switches the initial gaussian size rule (neighbour spacing instead of camera distance), so
"mcmc" is NVIDIA's recommended MCMC setup rather than the strategy change alone.
