# Object-level completion, prototype v1 (2026-10-08) — not an improvement yet

**Goal.** The plain splats are sharp except the unseen backs/sides of objects. Complete those object by
object from measured geometry and copied appearance — no generative model (the world-model route was
dropped: `2026-10-08_distill_ablation.md`).

**Pipeline** (`sbatch/object_completion.sbatch`): `find_objects.py` (6 free-standing objects in the
dynamic-free LiDAR: a block group, block stacks, the ramp) → `complete_objects.py` (2.5D surface per
object sampled every 1 cm; a sample is observed if a training camera sees it — in view, < 85° incidence,
matching that frame's LiDAR depth; unseen samples take the SH colour of the real gaussian nearest the
mirrored / same-height / nearest observed sample; unobserved real gaussians inside the object removed;
flat opaque discs added on unseen samples only) on the all-frames splat (LiDAR init + depth loss).
**Ground truth:** `mocap2_well-lit_comb` has the same object layout (object LiDAR points 1.7 cm apart),
so its frames, mapped into trot's frame (`cross_recording_views.py`), score the completed surfaces
(`object_back_masks.py`) and everything trot observed (regression gate).

| Version | comb views with completed surfaces | PSNR | LPIPS ↓ | PSNR completed surfaces | PSNR seen pixels |
|---|---|---|---|---|---|
| v1 (solid 2.5D) — base / completed | 11 | 22.14 / 20.92 | 0.204 / 0.231 | 18.45 / 12.77 | 24.33 / 22.85 |
| v3 (+ LiDAR support, free-space carving) — base / completed | 3 | 18.50 / 18.40 | 0.271 / 0.288 | 20.24 / 18.33 | 23.44 / 23.15 |

**What went wrong** (`img/2026-10-08_object_completion_v1.jpg`: real comb frame · base · completed).
- v1 treated objects as solid: it filled gaps between blocks and the open space under the A-frame ramp
  with opaque surface. Fixed in v3 by keeping only samples within 3 cm of the object's LiDAR returns and
  carving samples a camera sees through.
- The added ramp surface does not sit exactly on the splat's own ramp: a second, offset copy appears.
  LiDAR-derived geometry and the splat disagree by a few cm on thin, sloped surfaces.
- Most unseen samples fall back to "nearest observed sample" (mirror/row copy rarely applies to stacks
  and groups), giving speckled texture instead of brick courses.
- The cross-recording ground truth is thin: trot already observed ~all of what comb sees (99.8% of its
  LiDAR pixels); only 3–11 comb frames show completed surfaces (~0.5% of the image).

**Status.** Not adopted. Next: (1) evaluation with real ground truth on unseen faces — leave-arc-out
within one recording (drop a contiguous arc so some faces become unseen, complete, score the dropped
frames); (2) register the LiDAR-derived surface to the splat's own surface (ICP on the observed part)
before adding anything; (3) texture from the real images (rectified faces, period tiling) rather than
nearest-gaussian copies; (4) fall back to leaving an object untouched when its completion cannot be
validated.
