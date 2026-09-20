# CEAR → Gaussian Splat → Isaac Sim: Implementation Plan

**Status:** draft v3.1, 2026-09-20 (v3: LiDAR removed by project constraint; v3.1:
reconciled against the full RA-L paper text, arXiv 2404.04698v3)
**Owner:** Kushaan (UMass, DARoS Lab)

**Constraint (v3):** the Velodyne LiDAR stream may **not** be used. Allowed inputs: RGB,
depth, ground-truth poses, IMU, joint states. All geometry (splat initialization and
collider) comes from **RealSense depth**.

**Changes in v3.1 — paper reconciliation (arXiv 2404.04698v3):**
1. **Auto-exposure is confirmed ON** in the paper → exposure handling upgraded from
   contingency to expected requirement (§7.2); former PI question 6 resolved.
2. RGB and depth timestamps **differ by 0–8.33 ms** and are set at **mid-exposure** →
   new TRAP 7: pose each stream at its own timestamp.
3. There is **no hardware sync**; offsets come from post-hoc cross-correlation of a
   pitch-swing motion recorded at the start of **every sequence**, alongside a **thrown
   ball** → TRAP 3 gains a drift check, and new TRAP 8: trim the sync preamble.
4. Extrinsic provenance clarified: RGB↔Robot is **CAD-derived**; RealSense intrinsics and
   RGB↔Depth extrinsics are **factory onboard values** → new §5.3 constant-extrinsic
   refinement fallback.
5. MoCap runs at **120 Hz** (interpolation onto 60 Hz frames is well-conditioned).
6. The paper benchmarks Faster-LIO at ~**1–2.5° / 5–9 cm ATE** against mocap → ladder
   candidates 4–5 downgraded to stretch; their pose files are also **LiDAR-derived**,
   which may collide with the no-LiDAR constraint (§12 Q7 expanded).
7. `mocap1/2/3` confirmed as **three object arrangements of the same room** → never
   joint-train across them (§2.5).

---

## 0. Read this first

Building a **drivable simulation environment** from a real quadruped dataset:

```
CEAR sequence (RGB + depth + poses + IMU)
  → aligned per-frame camera poses (COLMAP sparse model)
  → fused depth map ──→ points3D seed ──→ 3D Gaussian Splat (visual) ─┐
                    └──→ collider mesh (physics)                     ─┴→ USD scene
                                                       → Isaac Sim → Mini Cheetah walks
```

Reference workflow: NVIDIA NuRec robot reconstruction (ROS bag → pose → depth → nvblox
mesh → 3DGRUT → USDZ with aligned mesh + occupancy, into Isaac Sim; depth-driven, no
LiDAR, novel views stay near capture height). This plan applies that to CEAR, with mocap
ground-truth poses instead of estimated ones.

Two data problems dominate: **pose accuracy** (3DGS is brutally sensitive; a 1–2° extrinsic
error yields a soft floaty splat — Phase 2 is highest-risk, do not skip its gate) and
**motion blur** (CEAR exists because RGB blurs during agile locomotion). Throughout:
**verify, do not assume** — CEAR conventions are inconsistent between files (§2.3).

## 1. Goals

**Primary:** a photorealistic, physics-enabled Isaac Sim scene from ONE indoor sequence
(RGB+depth+poses+IMU only) where a Mini Cheetah walks with correct floor contact.

**Definition of done (v1):** USD scene loads in Isaac Sim (pinned) ≥20 FPS @640×480 on
L40S; dropped body rests on floor ±3 cm; held-out real frames re-rendered with PSNR/SSIM/
LPIPS; quadruped walks ≥5 m without penetration; REPORT.md documents sequence, pose source,
metrics, pitfalls.

**Stretch (after DoD):** event-camera sim (EVIS/EsaacSim) vs real CEAR events — the
scientifically interesting part; multi-environment library; blur-aware 3DGS (§7.6).

**Non-goals:** no LiDAR (any phase); no video-gen model; splats are not collidable (§8); no
outdoor/backflip in v1; no locomotion policy (walking is validation).

## 2. The dataset: verified facts

### 2.1 Sensors
| Sensor | Model | Res | Rate | Notes |
|---|---|---|---|---|
| RGB | RealSense D455 | 640×480 | 60 Hz | 80°×65°, **rolling shutter** (§2.4) |
| Depth | RealSense D455 | 640×480 | 60 Hz | global shutter, **sole geometry source** (§2.7) |
| Event | DVXplorer / DAVIS346 | — | async | not used in v1 |
| IMU | VectorNav VN-100 | — | 400 Hz | accel+gyro+mag+quat |
| LiDAR | Velodyne VLP-16 | 16 ch | 10 Hz | **NOT USED** |
| Joints | Mini Cheetah | 12 | 100 Hz | |

GT poses: **OptiTrack 8×PrimeX-22 @ 120 Hz**, volume 7.5×5.5×2.8 m. All other sequences
(incl. non-mocap indoor) use **Faster-LIO** SLAM (~1–2.5° / 5–9 cm ATE vs mocap; §2.5).
Camera height ≈ 0.3–0.4 m — expect poor reconstruction above ~1.5 m.

### 2.2 On-disk layout
```
<sequence>/  event_*.aedat4(unused)  vectornav.txt  realsense_timestamp.txt  lidar.bag(unused)
  MoCap.txt|FasterLIO.txt  mini_cheetah_joint.txt  raw_rgb/ rgb/ raw_depth/ depth/
```
Formats: `vectornav.txt: t(us) g[3] a[3] mag[3] qw qx qy qz` (scalar-FIRST);
`MoCap/FasterLIO: t(s) x y z qx qy qz qw` (scalar-LAST); `realsense_timestamp.txt` packs
µs timestamps into filenames (separate depth and rgb columns). No ROS/bag reader needed.

### 2.3 TRAPS
- **TRAP 1** — quat order differs: vectornav scalar-FIRST, MoCap/FasterLIO scalar-LAST. Assert per file.
- **TRAP 1b** — COLMAP images.txt: world-to-camera, scalar-FIRST. Third convention; validate by round-trip (Gate 2b).
- **TRAP 2** — units: RGB/IMU/joints µs; MoCap/FasterLIO s.
- **TRAP 3** — offsets to the event clock (s): event 0; joints +0.004611; IMU −0.004012; **RealSense +0.004611 (critical)**; Velodyne +0.003044 (unused). No hardware sync — offsets are per-sequence pitch-swing cross-correlation estimates, so (a) VERIFY whether they are global or per-sequence in the download; (b) clocks may drift within a sequence — re-estimate the IMU↔pose lag on first vs last ~10 s (gyro vs mocap ω), and if it moved ≥2 ms apply a linear drift correction and report it.
- **TRAP 4** — pose frame ≠ camera frame: separate RGB-Robot / RGB-Marker extrinsics; compose; VERIFY which frame MoCap is in (T vs T⁻¹ is the classic failure).
- **TRAP 5** — two RGB folders: `rgb/` smoothed, `raw_rgb/` raw. A/B; default `raw_rgb/`; both undistorted (§7.2).
- **TRAP 6** — depth during fast motion: interpolation error smears geometry silently → fuse only motion-gated depth frames.
- **TRAP 7** — RGB and depth timestamps differ (0–8.33 ms), set at **mid-exposure** (auto-exposure varies RGB exposure). Pose each stream at its OWN timestamp column. Below Gate 2's visual threshold but not Gate 3's 2 cm split-half budget. Mid-exposure is the correct anchor for a blurred frame's pose — no further correction needed beyond using each stream's own timestamps.
- **TRAP 8** — every sequence begins with a sync preamble: a **body pitch swing + a thrown ball**. The swing is high-‖ω‖ (motion-gated anyway), but the flying/rolling ball is a **dynamic object in otherwise sharp frames** that bakes into the splat. Detect the preamble (unmistakable sinusoidal pitch in the IMU) and exclude that segment from RGB training frames (§7.1) and depth fusion (§6.1).

### 2.4 Rolling shutter
D455 RGB is rolling shutter (depth/IR global). Fast rotation (up to 750°/s in backflips)
skews lines. v1: gate hard on ‖ω‖ (§7.1); principled fix §7.6. Paper doesn't state shutter
type (does state mid-exposure timestamps); keep rolling-shutter as the working assumption.

### 2.5 Sequence selection
No indoor `Feat`; only `dininghall_*` is `Dyn` (avoid); avoid `Dark`/`Blink` in v1.
**Primary: `mocap1_well-lit_trot` (4.95 GB)** — OptiTrack poses, bounded, well-lit, small.
Ladder: mocap1/2/3 trot (OptiTrack) → classroom/lab1 trot (Faster-LIO, **stretch**) →
mocap1 comb (extra parallax, blurrier). Paper reconciliation:
- `mocap1/2/3` = **same room, three object arrangements** → joint-training across them is
  invalid; only `trot`+`comb` within one scene is legitimate. The ground objects give
  features for 3DGS and real obstacles for the collider/walk test.
- Faster-LIO candidates (4–5) are weaker than assumed (~1–2.5°/5–9 cm ATE) and their poses
  are **LiDAR-derived** (§12 Q7). Treat as stretch, not replication.
- Every sequence returns to marked start feet → start-vs-end mismatch is a free loop-closure
  drift check for any Faster-LIO sequence.

### 2.6 Trajectory coverage
A walking quadruped gives a near-linear path; 3DGS is bad away from observed views —
exactly where a drivable sim needs it. Prefer bounded mocap rooms. Quantify extent +
view-direction spread in Phase 2 and report; a straight line caps what's possible.

### 2.7 D455 depth noise vs range
Stereo-depth error grows ~quadratically with distance. Mandatory Phase 3: **truncate at a
max fusion range** (start 4 m; tune 3–5). Report the unobserved-within-range surface
fraction (honest holes). VERIFY depth validity/zero-fill and units (16-bit, 1 mm typical).

## 3. Environment setup
- **3.1 Hardware:** train 3DGS (3DGUT rasterization) + pose/depth scripts on **A100/H100**;
  Isaac Sim + splat rendering + walking on **L40S** (RTX cores; A100/H100 have none).
- **3.2 Software:** clone `nv-tlabs/3dgrut`, `install_env.sh`, conda `3dgrut`. Needs Linux,
  CUDA 11.8+, **GCC ≤ 11**, Python 3.8+. Plus numpy/scipy/opencv/open3d/pillow/tqdm/
  matplotlib/pycolmap. No ROS. CEAR is Google-Drive hosted (gdown/rclone; VERIFY egress).
- **3.3 Trainer is swappable; PLY is the interchange:** 3DGRUT ingests COLMAP-style (no raw
  pose JSON) → Phase 2 emits a COLMAP model. PLY ⇄ ParticleField ⇄ NuRec decouples trainer
  from export. Swap reasons: depth supervision (§7.3), exposure/appearance embeddings (§7.2),
  blur-aware (§7.6). Always train → PLY → `ply_to_usd`.
- **3.4 Version pinning:** Isaac 6.0 supports ParticleField + NuRec (NuRec deprecating);
  real render tradeoff from same PLY (NuRec higher fidelity/weaker RTX; ParticleField better
  RTX/more flicker) → export both, compare in-sim. Isaac 6.1 out — pin one day one. 3DGRUT
  USD export is beta. Known bug: two USDZ → wrong depth occlusion. Pin a 3DGRUT commit with
  the cx/cy PINHOLE fix. Record all commits/versions in REPORT.md day one.

## 4. Phase 1 — Acquisition and inventory
Download the sequence + GT pose + **calibration release** (intrinsics/distortion; RGB-Depth,
RGB-Robot, RGB-Marker, Robot-Links extrinsics). `inventory.py` reports counts, dims/dtype,
depth bit-depth/units/invalid (VERIFY 16-bit/1 mm/0), per-stream first/last ts + rate,
duration, pose-file type, intrinsics+distortion, shutter (rolling assumed), **AE=ON**
(paper), AWB unknown, and detects/records the sync-preamble window (TRAP 8) + clock drift
(TRAP 3). **Gate 1:** one table, all fields populated, counts self-consistent.

## 5. Phase 2 — Pose/frame alignment → COLMAP model — HIGHEST RISK
Deliverable `build_poses.py` → COLMAP sparse model (cameras/images/points3D; points seeded
in Phase 3) + `poses_meta.json`; interpolator in shared `pose_utils.py`.
Steps: (1) parse with explicit conventions, assert ‖q‖≈1, convert to event clock with TRAP 3
offsets; (2) interpolate each stream at its OWN timestamp column (TRAP 7); MoCap 120 Hz →
60 Hz is well-conditioned; LERP translation, SLERP rotation, never NN; (3) compose extrinsic
chain to world→RGB-optical, VERIFY direction (TRAP 4); (4) convert coordinate systems once
(OptiTrack Y-up → USD/Isaac Z-up, COLMAP camera convention); (5) write COLMAP world-to-cam
scalar-first (TRAP 1b), PINHOLE.

- **5.1 Gate 2 (MANDATORY, before training):** warp frame i→i+k (k≈10) via depth_i + relative
  pose; ≥20 pairs; edges land on edges. Directional offset = wrong direction/frame; rotational
  smear = quat order. Catches TRAPs 1,2,3,4.
- **5.2 Gate 2b:** reload written COLMAP model, re-run a warp subset from it (catches TRAP 1b);
  plot trajectory extent + view directions (§2.6).
- **5.3 Extrinsic provenance + ΔT fallback:** event↔RGB↔IMU via Kalibr; RGB↔Robot from CAD;
  RealSense intrinsics + RGB↔Depth are factory defaults (qualitatively validated). So a small
  CONSTANT extrinsic error is plausible even after Gate 2 passes → soft splat. Fallback:
  optimize one rigid ΔT (6 DoF), `T_world_cam = T_world_marker @ T_marker_cam @ ΔT`,
  minimizing Gate-2 warp error over the pairs. Run only if Gate 2 passes but Gates 3–4 show
  systematic softness; report ΔT magnitude (large ΔT ⇒ go back to TRAP 4). Per-image pose
  refinement is the bigger hammer — reserve for Faster-LIO sequences.

## 6. Phase 3 — Depth map: fuse, seed (depth-only)
One metric cloud/TSDF from RealSense depth; splat init now + collider later. Ahead of
training because dense metric depth beats random/SfM init.
- **6.1 Fuse:** motion-gate depth (‖ω‖, TRAP 6), exclude the preamble (TRAP 8), require
  spatial spread; each depth frame posed at its OWN timestamp (TRAP 7); drop invalid + range-
  truncate (§2.7); Open3D TSDF (voxel 2–3 cm); extract cloud (seed) + mesh (Phase-5 head
  start). Use `depth/` (RGB-frame); VERIFY at Gate 2.
- **6.2 Seed points3D.txt:** voxel-downsample (3 cm), colorize from nearest sharp RGB frames.
- **6.3 Gate 3:** reprojection overlay into 10 gated frames; **split-half fusion** (odd/even →
  two TSDFs; floor-height diff + C2C on shared surfaces, agree ~2 cm); crispness (thin walls);
  coverage/holes map (unobserved-within-range).

## 7. Phase 4 — Photometric prep, blur gating, training
- **7.1 Blur gating:** per RGB frame compute ‖ω‖ (IMU) + linear vel (mocap diff) + variance-
  of-Laplacian. Keep frames OUTSIDE the preamble (TRAP 8), low ‖ω‖ (start sharpest ~50%), high
  Laplacian, adequate spread. Expect to discard most (3DGS wants coverage/parallax). Report
  count + ‖ω‖ histogram.
- **7.2 Photometric prep:** **undistort first**, write PINHOLE. **Auto-exposure CONFIRMED ON**
  (paper) — brightness drifts by construction, upgrading exposure handling to an expected
  requirement (per-image exposure comp / appearance embeddings, §3.3). Still plot mean
  intensity over time to quantify the drift and decide if the swap is truly needed in the
  well-lit room. AWB unstated — check color-channel drift too. Apply identically to rgb/ and
  raw_rgb/.
- **7.3 Training:** 3DGRUT 3DGUT (rasterization, A100). No SfM (poses known + depth-seeded).
  Depth supervision UNVERIFIED — check configs; ladder: seed-only → depth-regularized swap
  (expected path with LiDAR excluded).
- **7.4 Hold-out:** every 8th GATED frame (split after gating).
- **7.5 Gate 4:** PSNR/SSIM/LPIPS on held-out + off-trajectory renders (lateral 0.5 m, height
  0.8 m) — degradation predicts drivability.
- **7.6 If blur dominates:** Deblur-GS / DeblurGS / **Gaussian Splatting on the Move** (models
  blur + rolling shutter; nerfstudio+gsplat; driven by velocities CEAR measures directly). A
  genuine research angle (blur trajectory is measured, not estimated); phase-2 contribution.

## 8. Phase 5 — Collider mesh (depth-only)
Splats are visual-only → build two co-registered artifacts. Option A: mesh the Phase-3 TSDF
(marching cubes / Poisson). Option B: nvblox (NVIDIA depth→mesh, what NuRec uses). Collider
need not be pretty — correct for foot contacts; simplify ≤50k tris. **Fill floor holes** along
the route with the RANSAC plane (log every patch). USD collision API + invisible.
**Gate 5:** drop a rigid sphere; rests on floor ±3 cm.

## 9. Phase 6 — Isaac Sim (L40S)
Export splat **both ways** (`ply_to_usd` → ParticleField + NuRec; no mesh in PLY→USDZ).
Compare in-sim (§3.4). Compose: splat (visual) + collider mesh (invisible) + physics scene
(gravity, Z-up). Spawn quadruped (Mini Cheetah USD or Isaac Lab fallback). **Gate 6:** walk
≥5 m through depth-observed regions, no penetration; video; ≥20 FPS @640×480 on L40S.

## 10. Phase 7 — Validation and reporting
REPORT.md: versions/commits; sequence + pose source + frame counts (RGB & depth); verified
transform chain incl. TRAP 1b writer conventions; Gate 2/2b images; Gate 3 overlays +
split-half numbers + coverage map + truncation range; trajectory/view plots; PSNR/SSIM/LPIPS
+ off-trajectory renders; rgb vs raw_rgb A/B; AE/AWB finding + exposure-comp route; collider
provenance (TSDF vs nvblox) + floor patches; ParticleField vs NuRec choice; walking video;
**a frank list of what does not work.**

## 11. Kill criteria (stop and report)
Gate 2 fail → stop, debug conventions (which trap). Gate 2b → fix writer (bounded).
Gate 3 disagree/double-walls → pose/sync error or weak gating. Gate 3 coverage → tighten
route / raise truncation / new seq. Gate 4 mush → blur (§7.6) or depth-supervised swap or new
seq. Straight-line trajectory → cannot support free nav. Gate 5 floor >5 cm → suspect Phase 2.
Gate 6 penetration → simplify collider, check up-axis/scale/route. A clean Gate-4 negative
with the coverage/blur/depth plots is a useful outcome; do not fake success.

## 12. Open questions for the PI
1. **Commanded-action logs?** (highest leverage — ask before downloading.)
2. Can new sequences be collected (commands logged, less bound/pronk blur)?
3. Has anyone attempted event-camera sim (their wall = the project)?
4. Mini Cheetah URDF/USD ready for Isaac?
5. Which environments matter most?
6. **Resolved by paper:** auto-exposure was ON (§7.2). Remaining: was auto *white balance*
   also on, and is the RealSense driver config archived?
7. Confirm the no-LiDAR scope on two fronts: (a) LiDAR offline for validation only (would
   restore a true sensor cross-check at Gate 3 for free)? (b) **every non-mocap sequence's GT
   poses are Faster-LIO, i.e. LiDAR-derived** — if barred, ladder candidates 4–5 are off the
   table and this is a mocap-sequences-only project.

## 13. Repository layout
```
scripts/  inventory.py  pose_utils.py  build_poses.py  verify_reprojection.py
          build_depth_map.py  verify_depth_map.py  precondition_frames.py  gate_frames.py
          sync_utils.py (TRAP 3/8)  gating_utils.py  image_utils.py  refine_extrinsic.py (§5.3)
          train_splat.sh  export_splat_usd.sh  eval_novel_views.py  compose_stage.py
configs/  <seq>.yaml (single source of truth for conventions)   data/ outputs/ (gitignored)
```
Commit after each gate passes, with the gate's metrics in the message.

## 14. Source references
CEAR: https://daroslab.github.io/cear/ · paper: https://arxiv.org/html/2404.04698v3 ·
3DGRUT: https://github.com/nv-tlabs/3dgrut (export README; PR #194 nerfstudio, unmerged) ·
NuRec robot (depth, no LiDAR): https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec ·
NuRec mono docs: https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html ·
Isaac NuRec/ParticleField: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_nurec.html ·
PF-vs-NuRec: https://forums.developer.nvidia.com/t/373260 · splats visual-only:
https://github.com/isaac-sim/IsaacSim/discussions/192 · depth-occlusion bug:
https://github.com/isaac-sim/IsaacSim/issues/108 · EVIS:
https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin · GS-on-the-Move:
https://arxiv.org/html/2403.13327v2 · Deblur-GS: https://github.com/Chaphlagical/Deblur-GS ·
COLMAP format: https://colmap.github.io/format.html
