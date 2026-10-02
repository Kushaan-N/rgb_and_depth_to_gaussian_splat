# Off-path benchmark + NVIDIA Difix3D+ (2026-10-02): plausible, but not faithful

**Question.** How good is the splat away from the recorded path, and does NVIDIA's Difix3D+
(arXiv 2503.01774, CVPR 2025) improve it?

**Benchmark** (`scripts/offpath_split.py`, Difix3D+ protocol). mocap2_well-lit_trot's 423 frames are
k-means clustered by camera position (k=4, seed 0) and one whole cluster is held out: 315 train /
108 test frames. Held-out cameras are 1.09 m (median) / 1.66 m (max) from the nearest training
camera. Every arm uses the pipeline's settings (3DGRUT, LiDAR-seeded init, 30k iterations).

**Arms** (`sbatch/offpath_difix.sbatch`, one A100, 32 min):
- baseline — 3DGRUT on the 315 training frames
- baseline+ — baseline renders cleaned by Difix (`nvidia/difix_ref`, post-render only)
- difix3d — baseline resumed for 7 rounds × 1,500 iterations; each round moves 108 novel cameras
  0.3 m from the nearest training camera toward the held-out poses, renders them, fixes them with
  Difix (nearest real image as reference) and trains on them (10% of frames; real frames ×3)
  — the authors' gsplat trainer, ported to 3DGRUT (`scripts/difix3d_views.py`)
- difix3d+ — difix3d renders cleaned by Difix
- control — baseline resumed for the same 10,500 iterations on real frames only

| Arm | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| baseline | 12.99 | 0.479 | 0.550 |
| baseline+ | 13.44 | 0.477 | 0.471 |
| control | 13.04 | 0.476 | 0.550 |
| **difix3d** | **13.70** | **0.491** | 0.478 |
| difix3d+ | 13.47 | 0.468 | **0.455** |

Same-split on-path reference: the full pipeline scores ~36 dB on its usual every-8th held-out frames.

**What the numbers hide** (`img/2026-10-02_offpath_difix.jpg`: real · baseline · difix3d · difix3d+).
- Views that face areas the training region also saw render almost perfectly in every arm.
- Views that face content only the held-out region saw (the black/yellow ramp, brick stacks) are
  floaters and black space in the baseline — the whole 13 dB.
- Difix3D turns that into a clean, plausible room — **with the wrong content**: the ramp disappears
  and shelves and floor from the reference images take its place. That is where its LPIPS gain
  (−0.072) and +0.7 dB come from. Extra iterations alone (control) change nothing.

**Conclusion.** Difix3D+ works as published (gains of the same size as the paper's, e.g. DL3DV
+0.8 dB / −0.09 LPIPS), and it removes floaters. But it fills what no camera saw with plausible
*fiction*. For a robot simulator that is risky: a wrong obstacle or missing ramp is worse than a
visible artifact. Not adopted into the pipeline. Two findings carry over:
1. The off-path benchmark itself: on-path PSNR (~36 dB) says nothing about leaving the path (~13 dB
   on a 1 m spatial hold-out). It should be reported per sequence.
2. Off-path quality is limited by *unobserved content*, not render artifacts — the paper's own
   limitation ("struggles where 3D reconstruction has entirely failed"). Fixing it needs either real
   observations (another recording) or a fill method anchored to measured geometry (LiDAR), with
   generated content kept out of the collider. See IDEAS.md → generative fill.

**Caveats.** One sequence and one split. The held-out cluster contains objects no training frame
saw, so this is a harsh test of hallucination, not only of viewpoint change; a masked metric
(pixels co-visible from training) would separate the two. Novel-view targets are the held-out poses,
as in the paper.

**Reproduce.** `bash env/make_difix_venv.sh` (once, CPU job), then
`sbatch -A $CEAR_SLURM_ACCOUNT sbatch/offpath_difix.sbatch configs/mocap2_well-lit_trot.yaml`
→ `<out>/pipeline/offpath/report/summary.json`. Difix is NVIDIA-licensed for non-commercial use.
