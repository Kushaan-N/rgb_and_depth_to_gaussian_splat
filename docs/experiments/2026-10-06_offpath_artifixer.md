# Off-path benchmark: NVIDIA ArtiFixer world-model fill (2026-10-06) — +4.4 dB, and mostly faithful

**Question.** Can a video *world model* fill the regions a robot never saw with content that is close
to reality — unlike Difix3D+, which filled them with plausible but wrong objects
(`2026-10-02_offpath_difix.md`)?

**Method.** NVIDIA ArtiFixer (SIGGRAPH 2026, nv-tlabs/ArtiFixer @a392c4d), 1.3B model on Wan2.1-T2V-1.3B.
For each held-out view it renders the 3DGRUT reconstruction plus its opacity, and the video model —
conditioned on those renders, camera rays and a Qwen3-VL scene caption — generates the view,
autoregressively along the held-out frames; generation is concentrated where opacity is low.
ArtiFixer3D then trains a 3DGRUT splat on the 315 real training frames + the 108 generated views.
Run on Unity's A100-80GB (`env/make_artifixer_venv.sh` mirrors the Docker image without Hopper-only
FlashAttention; `sbatch/offpath_artifixer.sbatch`, 43 min). The metric scale is ours (mocap/LiDAR
Sim3, 0.560), not ArtiFixer's monocular MoGe estimate.

**Same benchmark as Difix:** mocap2_well-lit_trot, one spatial cluster held out (108 frames,
1.09 m median / 1.66 m max from the nearest training camera), scored with `scripts/eval_views.py`.
Leakage checked: the distillation set's held-out frames are the world model's predictions (they
differ from the real images by 16–25 grey levels), never the real images.

| Version | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| our baseline (3DGRUT, LiDAR init, 30k) | 12.99 | 0.479 | 0.550 |
| Difix3D (best Difix arm) | 13.70 | 0.491 | 0.478 |
| ArtiFixer's own base reconstruction (MCMC, 10k, COLMAP init) | 12.00 | 0.374 | 0.612 |
| ArtiFixer (world-model views) | 15.00 | 0.487 | 0.499 |
| **ArtiFixer3D (distilled splat)** | **17.36** | **0.581** | **0.467** |
| ArtiFixer3D+ (world model on the distilled splat) | 17.29 | 0.568 | 0.468 |

**What the images show** (`img/2026-10-06_offpath_artifixer.jpg`: real · baseline · difix3d ·
artifixer · artifixer3d). Where the baseline is floaters and black space, ArtiFixer3D renders the
shelves, brick stacks and floor in the right places with the right appearance. The black/yellow
ramp — seen by no training frame — becomes a dark wedge-shaped object at the right spot rather than
vanishing (Difix replaced it with a shelf), but its stripes and the blue floor tape are missing:
plausible stand-ins anchored to the reconstruction's geometry, not exact copies. Views facing
well-observed areas are unchanged.

**Conclusion.** The world model is the first method that substantially improves never-seen regions:
+4.4 dB PSNR, +0.10 SSIM, −0.08 LPIPS over our baseline, and visibly closer to reality than Difix.
Distilling into 3D matters (+2.4 dB over the raw generated views): the splat averages the generated
views into one consistent scene. The second world-model pass (3D+) adds nothing.

**Caveats / next steps.**
- One sequence, one split; the held-out region contains objects no training frame saw.
- Its base reconstruction is weaker than ours (12.00 vs 12.99): ArtiFixer3D on *our* splat (LiDAR
  init) — `--reconstruction_checkpoint` — should do at least as well.
- Generated content is still not measured content: for Isaac, tag generated gaussians and keep them
  out of the collider; the collider stays LiDAR-only.
- In production the targets are not held-out poses but planned off-path camera paths (sideways
  offsets, other heights) via ArtiFixer's `--trajectory_path`.
- Licence: ArtiFixer weights NVIDIA One-Way Non-commercial (research use); code Apache-2.0.
