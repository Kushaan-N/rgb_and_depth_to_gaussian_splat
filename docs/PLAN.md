# CEAR → Gaussian Splat → Isaac Sim: Implementation Plan

**Status:** draft v3, 2026-09-19 (supersedes v2; **LiDAR removed by project constraint**)
**Audience:** a terminal-based coding agent with shell access, CUDA GPUs, and network access.
**Owner:** Kushaan (UMass, DARoS Lab)

**Constraint (v3):** the Velodyne LiDAR stream may **not** be used. Allowed inputs: RGB,
depth, ground-truth poses, IMU, joint states. All geometry (splat initialization and
collider) now comes from **RealSense depth**.

**Changes from v2:**
1. Phase 3 is now a **fused depth map** (TSDF / back-projection from motion-gated depth
   frames), replacing the LiDAR map. It still seeds `points3D.txt` and still becomes the
   collider source (§6).
2. TRAP 6 rewritten: LiDAR deskew is moot; the analogous risk is **pose-interpolation
   error on depth frames during fast motion** → fuse only motion-gated depth frames.
3. Depth-specific mitigations added: **range truncation** for D455 noise growth, and a
   **split-half fusion cross-check** to replace the lost LiDAR-vs-depth sensor
   cross-check (§6.3, §8).
4. `rosbags` dependency dropped; `lidar.bag` and the Velodyne time offset are documented
   but unused.

Feasibility is unchanged: NVIDIA's own NuRec robot workflows are depth-driven (stereo
depth → nvblox mesh → 3DGRUT), with no LiDAR anywhere. Losing LiDAR costs two things,
both survivable indoors: long-range geometry (far walls will be noisier) and an
independent second sensor for cross-checking — replaced by internal-consistency checks.

**Changes from v1 (carried over from v2):**
1. Phase 2 deliverable is a **COLMAP sparse model**, not `poses.json` — 3DGRUT has no
   raw-pose ingestion path (§5).
2. Geometry map moved ahead of training; it seeds `points3D.txt` (§6).
3. Trainer is explicitly **swappable**, with PLY as the interchange format; 3DGRUT's
   depth supervision is treated as UNVERIFIED (§7.3).
4. Export **both** ParticleField and NuRec USDZ and compare in-sim (§3.4, §9).
5. Hardware plan for Unity HPC: **train on A100/H100, run Isaac Sim on L40S** (§3.1).
6. Photometric preconditioning: undistort-first policy, auto-exposure check (§7.2).
7. Held-out eval split sampled *after* blur gating, from gated frames only (§7.4).
8. Gates numbered 1–6 plus 2b.

---

## 0. Read this first

You are building a **drivable simulation environment** reconstructed from a real quadruped
dataset. The pipeline is:

```
CEAR sequence (RGB + depth + poses + IMU)
  → aligned per-frame camera poses (COLMAP sparse model)
  → fused depth map ──→ points3D seed ──→ 3D Gaussian Splat (visual) ─┐
                    └──→ collider mesh (physics)                     ─┴→ USD scene
                                                       → Isaac Sim → Mini Cheetah walks
```

This is not a speculative pipeline. NVIDIA's own **NuRec robot reconstruction workflows**
(Nova Carter datasets) are: ROS bag → pose estimation → **depth estimation** → mesh
generation (nvblox) → neural reconstruction (3DGRUT) → USDZ with an aligned mesh and
occupancy map, loaded into Isaac Sim — depth-driven, no LiDAR, including the caveat that
novel views should stay near the capture height, since the data came from a ground robot.
That is this plan, applied to CEAR (with the advantage that our poses are mocap ground
truth, not estimated). Use it as the reference when stuck.

Two things dominate success or failure, and both are *data* problems, not compute problems:

1. **Pose accuracy.** 3D Gaussian Splatting (3DGS) is brutally sensitive to camera pose
   error. A 1–2° extrinsic mistake does not fail loudly; it produces a soft, floaty splat
   and you will waste days blaming the trainer. **Phase 2 is the highest-risk phase in this
   document. Do not skip its verification gate.**
2. **Motion blur.** CEAR exists *because* RGB blurs during agile locomotion — that is the
   dataset's motivating claim. The RGB stream degrades exactly where ego-motion is largest.

Throughout: **verify, do not assume.** Where this document says VERIFY, that means run a
check and report the finding before proceeding. Several conventions in CEAR are
inconsistent between files (see §2.3) and guessing wrong silently corrupts everything
downstream.

---

## 1. Goals

### 1.1 Primary goal

Produce a photorealistic, physics-enabled Isaac Sim scene reconstructed from **one** CEAR
indoor sequence — using only RGB, depth, poses, and IMU — in which a Mini Cheetah URDF can
be spawned and commanded to walk, with its feet making correct contact with reconstructed
floor geometry.

### 1.2 Definition of done (v1 milestone)

- [ ] A `.usd`/`.usdz` scene loads in Isaac Sim (pinned version, §3.4) and renders the
      reconstructed environment at ≥ 20 FPS at 640×480 **on an L40S**.
- [ ] A rigid body dropped into the scene rests on the reconstructed floor at the correct
      height (± 3 cm) rather than falling through or hovering.
- [ ] A held-out set of real CEAR RGB frames (gated frames not used in training, §7.4) can
      be re-rendered from their ground-truth poses, with a reported PSNR / SSIM / LPIPS
      table.
- [ ] A quadruped (Mini Cheetah or, as fallback, any Isaac Lab quadruped) walks ≥ 5 m
      through the scene without penetrating geometry.
- [ ] `REPORT.md` documents which sequence, which pose source, all metrics, and every
      pitfall actually hit.

### 1.3 Stretch goals (do not start until 1.2 is met)

- [ ] Event camera simulation in the reconstructed scene, via EVIS or EsaacSim, compared
      against the real CEAR event stream for the same environment.
- [ ] Multiple environments reconstructed and packaged as a scene library.
- [ ] Blur-aware 3DGS exploiting CEAR's known camera velocities (§7.6).

### 1.4 Explicit non-goals and constraints

- **Do not use the LiDAR stream, in any phase, for any purpose.** `lidar.bag` stays on disk untouched.
- Do **not** train/fine-tune/post-train any video generation model (Cosmos, etc.).
- Do **not** attempt to make the Gaussian splats themselves collidable (§8).
- Do **not** reconstruct outdoor or backflip sequences in v1.
- Do **not** build a locomotion policy. Walking is a validation check.

---

## 2. The dataset: verified facts

### 2.1 Sensor specifications

| Sensor | Model | Resolution | Rate | Notes |
|---|---|---|---|---|
| RGB | Intel RealSense D455 | 640 × 480 | 60 Hz | FoV 80°×65°. **Rolling shutter** (§2.4) |
| Depth | RealSense D455 | 640 × 480 | 60 Hz | global shutter. **Sole geometry source in v3** (§2.7) |
| Event | DVXplorer Lite | 320 × 240 | async | Not used in v1 |
| Event | DAVIS346 | 346 × 260 | async | Not used in v1 |
| IMU | VectorNav VN-100 | — | 400 Hz | accel + gyro + mag + quaternion |
| LiDAR | Velodyne VLP-16 | 16 ch | 10 Hz | **NOT USED — project constraint** |
| Joints | Mini Cheetah encoders | 12 values | 100 Hz | |

Ground truth: **OptiTrack, 8×PrimeX-22, volume 7.5×5.5×2.8 m.** Outside that volume:
**Faster-LIO** SLAM poses. Camera height ≈ 0.3–0.4 m; expect poor reconstruction above
~1.5 m (barely observed).

### 2.2 On-disk layout (per sequence)

```
<sequence>/
├── event_*.aedat4            # NOT USED IN v1.
├── vectornav.txt             # IMU
├── realsense_timestamp.txt   # frame index / timestamp table
├── lidar.bag                 # NOT USED — project constraint
├── MoCap.txt   OR  FasterLIO.txt   # 6-DoF ground truth pose
├── mini_cheetah_joint.txt    # 12 joint angles, radians
├── raw_rgb/                  # unprocessed RGB
├── rgb/                      # processed (smoothed) RGB
├── raw_depth/                # depth in the DEPTH camera frame
└── depth/                    # depth projected into the RGB and EVENT camera frames
```

Line formats:

```
vectornav.txt:
  timestamp(us) gx gy gz ax ay az magx magy magz qw qx qy qz         # quat scalar-FIRST
MoCap.txt / FasterLIO.txt:
  timestamp(s) x y z qx qy qz qw                                     # quat scalar-LAST
mini_cheetah_joint.txt:
  timestamp(us) FR(abd hip knee) FL(...) HR(...) HL(...)             # radians
realsense_timestamp.txt:
  timestamp(us)_depth_rgb.png  timestamp_depth_event.png  timestamp_rgb.png  ...
```

### 2.3 TRAPS — read twice

- **TRAP 1 — quaternion convention differs.** `vectornav.txt` → `qw qx qy qz` (scalar-FIRST);
  `MoCap.txt`/`FasterLIO.txt` → `qx qy qz qw` (scalar-LAST). Assert per file.
- **TRAP 1b — COLMAP adds a third convention.** `images.txt` stores **world-to-camera**,
  **scalar-first** quats — inverse direction, other quat order. Handle in one function;
  validate by loading the written model back (Gate 2b).
- **TRAP 2 — timestamp units differ.** RGB/IMU/joints µs; MoCap/FasterLIO s. Naive merge off by 10⁶.
- **TRAP 3 — per-sensor time offsets.** Poses synced to the **event clock** (0.0 s ref).
  Offsets (s): event 0.0; MiniCheetahJoint +0.004611; VectorNav −0.004012; RealSense
  +0.004611 (**critical**); Velodyne +0.003044 (unused). ~4.6 ms is a visible pose error at trot.
- **TRAP 4 — pose frame ≠ camera frame.** Calibration gives separate `RGB-Robot` and
  `RGB-Marker` extrinsics. Compose. VERIFY which frame `MoCap.txt` is in; using T vs T⁻¹
  backwards is the most common failure.
- **TRAP 5 — two RGB folders.** `rgb/` smoothed, `raw_rgb/` unprocessed. A/B both; default
  `raw_rgb/`. Both go through undistortion (§7.2) first.
- **TRAP 6 — depth during fast motion.** Each depth frame gets one interpolated pose;
  fast-motion interpolation error smears geometry silently. **Fuse only motion-gated depth
  frames** (same ‖ω‖ criterion as §7.1).

### 2.4 Rolling shutter

D455 RGB is rolling shutter (depth/IR global). Fast rotation (up to 750°/s during
backflips) skews lines. v1 mitigation: gate hard on angular velocity (§7.1). Principled
fix in §7.6. VERIFY shutter type from calibration if stated.

### 2.5 Sequence selection

No indoor sequence is `Feat`; only `dininghall_*` is `Dyn` (avoid); avoid `Dark`/`Blink`
in v1. **Primary target: `mocap1_well-lit_trot` (4.95 GB)** — inside OptiTrack volume
(true mocap poses), bounded, well-lit, small. Candidate ladder: mocap1/2/3 trot
(OptiTrack), classroom/lab1 trot (Faster-LIO), mocap1 comb (extra parallax, blurrier —
gate hard).

### 2.6 Trajectory coverage

A walking quadruped gives a near-linear camera path; 3DGS reconstructs badly away from
observed views — exactly where a *drivable* sim needs it. Prefer bounded mocap rooms.
Quantify: plot trajectory extent + viewing-direction distribution in Phase 2; report it.

### 2.7 D455 depth noise vs range

Stereo-depth error grows ~quadratically with distance. Mandatory in Phase 3: **truncate
depth at a max fusion range** (start 4 m; tune 3–5). Report the fraction of surface never
observed within range (honest holes). VERIFY depth validity/zero-fill convention and units.

---

## 3. Environment setup

### 3.1 Hardware (Unity HPC)

| Workload | Partition | Why |
|---|---|---|
| 3DGS training (3DGUT), pose scripts, depth fusion | **A100 / H100** | rasterization; 24–48 GB ample; sweep sequences in parallel |
| Isaac Sim, splat rendering, walking validation | **L40S** | Omniverse RTX renderer needs RT cores (A100/H100 have none) |

Use 3DGRUT's **3DGUT** (rasterization) configs on A100. 3DGRT ray-tracing → L40S if ever used.

### 3.2 Software

```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut && chmod +x install_env.sh && ./install_env.sh 3dgrut && conda activate 3dgrut
```

Requires Linux, NVIDIA GPU, CUDA 11.8+, **GCC 11 or lower**, Python 3.8+. Also:
numpy, scipy, opencv-python, open3d, pillow, tqdm, matplotlib, pycolmap. No ROS in v3.
Downloads: CEAR is Google Drive-hosted — use gdown/rclone; VERIFY compute-node egress.

### 3.3 Trainer is swappable; PLY is the interchange format

3DGRUT ingests COLMAP-style / NeRF-synthetic / ScanNet++ / NCore — **no raw-pose JSON**.
Hence Phase 2 emits a COLMAP sparse model (§5). 3DGRUT converts PLY ⇄ ParticleField USD ⇄
NuRec, so trainer and Isaac export are decoupled: any trainer that reads a COLMAP model
and writes a standard 3DGS PLY can slot in. Reasons to swap: depth supervision (§7.3),
exposure/appearance embeddings (§7.2), blur-aware method (§7.6). Always: train → PLY →
`ply_to_usd`.

### 3.4 Version pinning

Isaac Sim 6.0 supports both `ParticleField3DGaussianSplat` and NuRec USDZ (NuRec slated
for deprecation). Tradeoff from the same PLY: NuRec higher fidelity/weaker RTX integration;
ParticleField better RTX integration/more flicker. **Export both, compare in-sim** (Gate
4/6). Isaac Sim 6.1 is out — pin one version day one. 3DGRUT USD export is beta (schema may
change). Known bug: two 3DGRUT USDZ files → incorrect depth occlusion. Pin a 3DGRUT commit
that includes the cx/cy PINHOLE-intrinsics fix. Record all commit hashes/versions in
`REPORT.md` day one; do not upgrade mid-project.

---

## 4. Phase 1 — Acquisition and inventory

Download `mocap1_well-lit_trot` + GT pose + **calibration release** (intrinsics/distortion
for RealSense; extrinsics for RGB-Depth, RGB-Robot, RGB-Marker, Robot Links).
`scripts/inventory.py` reports: file counts in rgb/raw_rgb/depth/raw_depth; image
dims+dtype; depth bit-depth/units/invalid convention (VERIFY 16-bit, 1 mm, 0=invalid);
first/last timestamp + inferred rate per stream; duration; pose file type; parsed
intrinsics + distortion; VERIFY shutter type and AE/AWB if stated. **Gate 1:** one table,
all fields populated, frame counts self-consistent.

## 5. Phase 2 — Pose/frame alignment → COLMAP model — HIGHEST RISK

Deliverable `scripts/build_poses.py` → COLMAP sparse model (`cameras.txt` one PINHOLE
post-undistortion camera; `images.txt` per-frame world-to-camera scalar-first + filename;
`points3D.txt` seeded in Phase 3, placeholder here) + `poses_meta.json`. Interpolator
lives in shared `scripts/pose_utils.py` (Phases 3/4/5 all use it).

Steps: (1) parse with explicit conventions, assert ‖quat‖≈1, convert timestamps to event
clock with TRAP 3 offsets; (2) interpolate poses onto RGB+depth timestamps — LERP
translation, **SLERP** rotation, never nearest-neighbour; (3) compose extrinsic chain to
world→RGB optical, VERIFY direction (TRAP 4), write the chain in a comment; (4) convert
coordinate systems once (OptiTrack Y-up → USD/Isaac Z-up, COLMAP camera convention);
(5) write COLMAP with world-to-camera scalar-first (TRAP 1b), PINHOLE.

**Gate 2 (MANDATORY, before training):** warp frame *i* into frame *i+k* (k≈10) using
frame *i* depth + relative pose; compare to actual *i+k*. ≥20 pairs across the sequence.
Pass: edges land on edges. Directional offset → wrong transform direction/frame; rotational
smear → quaternion convention error. Catches TRAPs 1,2,3,4.

**Gate 2b:** load the written COLMAP model back, reconstruct cam-to-world, re-run a warp
subset from the loaded model (catches TRAP 1b). Plot trajectory extent + view directions (§2.6).

## 6. Phase 3 — Depth map: fuse, seed (depth-only geometry)

One metric point cloud/TSDF from RealSense depth, used twice (splat init now, collider
later). Sits ahead of training because dense metric depth beats random init / SfM points.

- **6.1 Fuse:** motion-gate depth frames (‖ω‖ as §7.1), require spatial spread; drop
  invalid pixels + apply §2.7 range truncation (start 4 m); Open3D `ScalableTSDFVolume`
  (voxel 2–3 cm) posed by `pose_utils.py` interpolator at offset-corrected timestamps;
  extract cloud (→ seed) + mesh (→ collider head start). Use `depth/` (already RGB-frame);
  VERIFY at Gate 2.
- **6.2 Seed points3D.txt:** voxel-downsample (3 cm), colorize by projecting into nearest
  sharp RGB frames, write as COLMAP `points3D.txt`.
- **6.3 Gate 3:** reprojection overlay of fused points into 10 gated RGB frames;
  **split-half fusion** (odd vs even frames → two TSDFs; floor-plane height diff + C2C
  distance; agree ~2 cm); crispness (thin walls); coverage report (unobserved-within-range
  fraction, visualize holes).

## 7. Phase 4 — Photometric prep, blur gating, splat training

- **7.1 Blur gating:** ‖ω‖ from IMU + linear vel from mocap diff + variance-of-Laplacian.
  Keep frames with low ‖ω‖ (start sharpest ~50%), high Laplacian var, adequate spatial
  spread. Expect to discard most; 3DGS wants coverage/parallax, not count. Report retained
  count + ‖ω‖ histogram.
- **7.2 Photometric prep:** **undistort first**, write plain PINHOLE. AE/AWB check (plot
  mean intensity over time; drift → appearance embeddings via swap). Apply identically to
  rgb/ and raw_rgb/ (TRAP 5 A/B).
- **7.3 Training:** default 3DGRUT 3DGUT config; no SfM (poses known + depth-seeded).
  Depth supervision UNVERIFIED — check configs first; ladder: seed-only → depth-regularized
  swap. depth/ already in RGB frame (VERIFY at Gate 2).
- **7.4 Hold-out:** every 8th frame **of the gated set** (split after gating).
- **7.5 Gate 4:** PSNR/SSIM/LPIPS on held-out + rendered-vs-real; also render OFF-trajectory
  (lateral 0.5 m, height 0.8 m) — quantify degradation.
- **7.6 If blur dominates:** Deblur-GS, DeblurGS, **Gaussian Splatting on the Move**
  (models blur + rolling shutter, nerfstudio+gsplat, driven by velocities CEAR measures
  directly). Genuine research angle.

## 8. Phase 5 — Collider mesh (depth-only)

Splats are visual-only in Isaac Sim; build two co-registered artifacts (splat=appearance,
mesh=physics). Option A (default): mesh the Phase 3 TSDF (marching cubes / Poisson).
Option B: nvblox (NVIDIA depth→mesh, what NuRec robot workflow uses). Collider need not be
pretty — correct for foot contacts. Simplify ≤50k tris. **Fill floor holes** with the
RANSAC plane along the route; log patches. Apply USD collision API, mark invisible.
**Gate 5:** drop a rigid sphere; rests on floor ±3 cm.

## 9. Phase 6 — Isaac Sim integration (L40S)

Export splat **both ways** (`ply_to_usd` → ParticleField + NuRec, one flag apart).
Note: PLY→USDZ produces no mesh — add mesh in USDZ or compose separately. Compare
ParticleField vs NuRec in-sim (§3.4). Compose stage: splat (visual) + mesh (collider,
invisible) + physics scene (gravity, up-axis). Spawn quadruped (Mini Cheetah URDF or Isaac
Lab fallback). **Gate 6:** walk ≥5 m without penetration through depth-observed regions;
video; ≥20 FPS at 640×480 on L40S.

## 10. Phase 7 — Validation and reporting

`REPORT.md`: exact versions/commits; sequence + pose source + frame counts (RGB & depth);
verified transform chain incl. COLMAP writer conventions (TRAP 1b); Gate 2/2b images; Gate
3 overlays + split-half numbers + coverage map + truncation range; trajectory/view plots;
PSNR/SSIM/LPIPS + off-trajectory renders; rgb vs raw_rgb A/B; AE/AWB finding; collider
provenance (TSDF vs nvblox) + floor patches; ParticleField vs NuRec choice; walking video;
**frank list of what does not work.**

## 11. Kill criteria and decision gates

| Gate | If it fails | Do this |
|---|---|---|
| Gate 2 | transform chain wrong | **Stop.** Debug conventions; report which trap. |
| Gate 2b | writer conventions wrong (TRAP 1b) | **Stop.** Fix writer (bounded). |
| Gate 3 | split halves disagree / double walls | pose/sync error or weak gating (TRAP 6). |
| Gate 3 coverage | large unobserved route regions | tighten route / raise truncation / new seq. |
| Gate 4 | mush after gating+init | blur (§7.6) or depth-supervised swap (§7.3) or new seq. |
| Trajectory | ~straight line | cannot support free nav; new seq / new capture. |
| Gate 5 | floor wrong > 5 cm | suspect Phase 2 poses; re-check Gates 2–3. |
| Gate 6 | persistent penetration | simplify collider; check up-axis/scale; check route. |

A clean negative at Gate 4 (with coverage + blur + depth-coverage plots) is a useful
outcome. Do not fake a success.

## 12. Open questions for the PI

1. **Commanded-action logs?** (biggest lever — ask before downloading).
2. Can new sequences be collected (commands logged, diverse velocities, less bound/pronk blur)?
3. Has anyone attempted event-camera sim (their wall = the project)?
4. Mini Cheetah URDF/USD ready for Isaac?
5. Which environments matter most?
6. RealSense AE/WB fixed or automatic during capture? (§7.2)
7. Does no-LiDAR forbid LiDAR **offline for validation only**? (would restore a sensor cross-check at Gate 3)

## 13. Suggested repository layout

```
cear-world/
├── REPORT.md
├── configs/mocap1_well-lit_trot.yaml
├── scripts/
│   ├── inventory.py            # Phase 1
│   ├── pose_utils.py           # shared parsers/offsets/SLERP (Phases 2-5)
│   ├── build_poses.py          # Phase 2 → COLMAP sparse model
│   ├── verify_reprojection.py  # Gates 2 + 2b  ← before any training
│   ├── build_depth_map.py      # Phase 3
│   ├── verify_depth_map.py     # Gate 3
│   ├── precondition_frames.py  # Phase 4
│   ├── gate_frames.py          # Phase 4 gating + hold-out
│   ├── train_splat.sh          # Phase 4 (3DGUT; swappable)
│   ├── eval_novel_views.py     # Gate 4
│   ├── build_collider.py       # Phase 5
│   └── compose_stage.py        # Phase 6 (L40S)
├── data/                       # gitignored (lidar.bag present but never read)
└── outputs/                    # gitignored
```

Commit after each gate passes, with the gate's metrics in the commit message.

## 14. Source references

- CEAR: https://daroslab.github.io/cear/ · Downloads: /Downloads/ · Calibration: /Calibration/
- CEAR paper (RA-L 2024): https://arxiv.org/html/2404.04698v3
- 3DGRUT: https://github.com/nv-tlabs/3dgrut · export README `threedgrut/export/README.md`
- 3DGRUT nerfstudio PR (unmerged): https://github.com/nv-tlabs/3dgrut/pull/194
- NuRec robot datasets (depth-driven, no LiDAR): https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec
- NuRec mono docs (COLMAP inputs, GCC 11): https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html
- Isaac Sim NuRec/ParticleField: https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_nurec.html
- ParticleField vs NuRec tradeoff: https://forums.developer.nvidia.com/t/373260
- Splats visual-only: https://github.com/isaac-sim/IsaacSim/discussions/192
- Depth-occlusion bug: https://github.com/isaac-sim/IsaacSim/issues/108
- EVIS event plugin: https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin
- Gaussian Splatting on the Move: https://arxiv.org/html/2403.13327v2 · Deblur-GS: https://github.com/Chaphlagical/Deblur-GS
- COLMAP format: https://colmap.github.io/format.html
