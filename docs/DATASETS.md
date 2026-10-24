# Datasets (multi-camera / 360 extension)

One section per dataset: source, licence, what was downloaded (size, path), how to fetch, adapter.

## Oxford Spires (ori-drs) — subset: Radcliffe Observatory Quarter 01

- Source: https://huggingface.co/datasets/ori-drs/oxford_spires_dataset (project page
  https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/, code https://github.com/ori-drs/oxford_spires_dataset,
  paper Tao et al., IJRR 2025, arXiv 2411.10546). Public, not gated, no form.
- Licence: CC BY-NC-SA 4.0, non-commercial academic use (raw/LICENSE.md). Cite the paper.
- Rig: Alphasense, 3 hardware-synced global-shutter colour cameras (front / left / right, 1440x1080, Kalibr
  equidistant = COLMAP OPENCV_FISHEYE, ~122x93 deg usable), Hesai QT64 LiDAR, IMU; Leica RTC360 TLS ground-truth
  maps (mm accuracy); TLS-registered GT trajectory (gt-tum.txt, base frame, 20 Hz).
- Whole dataset: 24 sequences / 6 sites, several hundred GB (per sequence: raw images ~6 GB, raw LiDAR ~2 GB, undistorted
  clouds ~2.4 GB; per site TLS: e57 ~6 GB, 1 cm merged cloud ~2 GB). NVS benchmark: Blenheim 01+05 (553 MB), Keble 04
  (723 MB), ROQ 01 (274 MB); published trained models 4.6-5.6 GB per site (not downloaded).
- Chosen: 2024-03-13-observatory-quarter-01 (ROQ-01): smallest NVS-benchmark site, has in-sequence + out-of-sequence
  NVS splits, TLS GT and is one of the 4 reconstruction-benchmark sequences (published accuracy/completeness/F).
- Downloaded (2.68 GiB, sha256-verified against the Hub's LFS oids, unzipped next to the zips):
  `$MC/data/spires/2024-03-13-observatory-quarter-01/raw/` (`MC=/scratch4/workspace/knaskar_umass_edu-multicam`)
  - calibration/ (cam0-2.yaml, cam-lidar-imu.yaml), LICENSE.md
  - novel_view_synthesis_benchmark/training_data/nvs_roq_01.zip (274 MiB): their undistorted 1000x534 pinhole
    images with train_/eval_ prefixes, COLMAP model, transforms_train(_eval).json
  - sequences/<seq>/processed/trajectory/{gt,vilens-slam,colmap}-tum.txt, vilens-slam/{slam-poses.csv,undist-clouds.zip}
  - reconstruction_benchmark/<seq>/input.zip (690 MiB: 1 Hz original fisheye frames of all 3 cameras, VILENS
    motion-undistorted keyframe clouds, T_gt_lidar.txt), gt_cloud_merged.pcd (1.4 GiB, 122 M TLS points), gt_cloud.bt
- Fetch: `python scripts/datasets/spires_fetch.py configs/spires_roq01.yaml` (stdlib only; resumable; files listed in
  the config) or `sbatch/spires_fetch.sbatch`. Other sequences: copy configs/spires_roq01.yaml, change `sequence.name`
  and the file list.
- Adapter: `sbatch/spires_prepare.sbatch configs/spires_roq01.yaml` (scripts/datasets/spires.py) -> `$MC/outputs/spires/<seq>/`:
  fisheye/{train,test_in,test_out} (OPENCV_FISHEYE from Kalibr, cx/cy +0.5 to COLMAP convention; poses = gt-tum.txt
  x T_base_lidar x T_cam_lidar^-1; their NVS split: 225 train / 26 in-sequence (every 10th train frame) / 137
  out-of-sequence), pinhole_center (100x83 deg, 1280x960, scripts/rectify_views.py), pinhole_cube (3x60 deg native
  res), benchmark/<set>_cam<k> (their own undistorted 1000x5xx eval images at their COLMAP poses, Sim3-aligned),
  lidar.ply (VILENS keyframe clouds, 5 cm), tls.ply (TLS crop, 2 cm), checks/overlay_*.jpg.
  Checks: our camera poses vs their independent COLMAP model: 1.3 cm / 0.26 deg median. Caveat found: gt-tum.txt
  is offset ~0.23 m (mostly constant) from the TLS datum (LiDAR clouds placed with it sit 12 cm from the TLS vs
  2.5 cm with T_gt_lidar x SLAM poses); the adapter moves the TLS onto the GT datum (tls_align.json) so geometry
  scores measure the splat. `adapter.pose_source: slam_tls` uses the TLS datum directly (keyframe stamps only).

## Smaller candidates (multi-camera / 360 / fisheye; web survey 2026-10-19)

Goal: Spires/FusionPortable-like data (multi-camera or 360/fisheye, ideally LiDAR/TLS + GT poses) that is small, has no
form, and can be fetched by a script. Ranked by usefulness x size for this pipeline. Sizes are the server's
Content-Length (checked 2026-10-19); "form" = a registration/agreement the user must sign (not downloaded).

| # | Dataset | Cameras | Depth / GT | Size per scene | Access, licence | Verdict |
|---|---------|---------|------------|----------------|-----------------|---------|
| 1 | **FIORD** (Gunes et al. 2025, arXiv 2504.01732) | Insta360 One RS 1-Inch: 2 back-to-back 200 deg fisheyes (= 2-camera rig), 3264x3264, stills | Faro Focus TLS (.e57, 108 M pts), COLMAP OPENCV_FISHEYE model | 4.6-17.9 GB (indoor) | Zenodo 15181976 (4 indoor), Tampere portal (outdoor); CC-BY-4.0, md5 | **downloaded** (MeetingRoom 4.6 GB) |
| 2 | **Ricoh360** (EgoNeRF, CVPR 2023) | Ricoh Theta 360 equirect 1920x960, 12 outdoor scenes x 100 frames | SfM only (arbitrary scale) | 2.7 GB all 12 scenes | Google Drive (public folder), repo MIT | **downloaded**; ODGS/EgoNeRF benchmark |
| 3 | OmniBlender (EgoNeRF) | synthetic 360 equirect, 11 scenes | GT poses + depth | 5.2 GB all | same Drive folder | good for method sanity checks |
| 4 | 360Loc (CVPR 2024) | 360 reference + fisheye/pinhole/360 query cams | LiDAR-SLAM-derived GT poses | 4.4-7.9 GB (atrium, concourse, hall, piatrium) | hkustvgd.com/statics/360loc (direct); licence not stated | next 360+LiDAR-pose candidate |
| 5 | Holo360D (ECCV 2026) | 360 panoramas, continuous trajectories, 75 scenes | LiDAR mesh / point-cloud depth maps + poses | 7.6-80 GB (smallest test/Indoor_016) | HF ouou123/Holo360D, not gated; licence not stated | best 360+LiDAR match, but large |
| 6 | Zip-NeRF fisheye (Barron 2023) | Sony fisheye, 4 large scenes | none (SfM) | fisheye 11-22 GB, undistorted 2.7-6 GB | storage.googleapis.com/gresearch/refraw360 (direct) | 3DGUT's published fisheye benchmark |
| 7 | Hilti SLAM 2022/2023 | Alphasense 5 x 0.4 MP (grayscale) | Hesai XT-32 + TLS .e57, sparse/dense GT | 6-80 GB rosbags | HF Hilti-Research, CC BY-NC-SA 3.0 | grayscale: weak for colour splats |
| 8 | Newer College multi-cam | Alphasense 4 grayscale fisheye 126x92 deg | Ouster OS0-128, prior map, GT | rosbags | **form** (download page), CC BY-NC-SA 4.0 | stop: form |
| 9 | ScanNet++ | DSLR fisheye (1752x1168) + iPhone | Faro TLS, GT poses | ~GBs/scene | **agreement** | stop: agreement (Fisheye-GS / 3DGUT benchmark) |
| - | 360Roam | 360 indoor | SfM | - | authors stopped distribution (ODGS README, 2024-12) | unavailable |
| - | KITTI-360, Matterport3D, 2D-3D-S, Waymo, nuScenes | fisheye / 360 / multi-cam | LiDAR / mesh | large | registration / agreement | stop: form |
| - | Argoverse 2 sensor | 7-camera pinhole ring + stereo | 2 x VLP-32 LiDAR + poses | ~1 GB / 15 s log | s3 direct, CC BY-NC-SA 4.0 | multi-cam rig but driving + dynamic, pinhole |
| - | MCD (NTU multi-campus) | stereo/pinhole cams (no 360 rig) | LiDAR + survey GT | large | Google Drive | not a fisheye/360 rig: not pursued |

Picks: FIORD MeetingRoom (fisheye rig + TLS: the closest small analogue of Spires) and Ricoh360 `bricks` (real
equirect, official split, the 360-GS literature's benchmark). Workspace `MC=/scratch4/workspace/knaskar_umass_edu-multicam`.

### FIORD MeetingRoom_In

- Source: https://zenodo.org/records/15181976 (project page https://parmisian.github.io/fiord360.github.io/), CC-BY-4.0.
- Downloaded: `$MC/data/fiord/MeetingRoom_In.zip` (4,604,401,957 B, md5 9e0be45675ea632d2c653045ff8359a5 verified);
  unpacked subset `$MC/data/fiord/meetingroom/{fisheyeimages/cam1,cam2 (107+107 PNG), model/ (COLMAP bin, 203
  registered, OPENCV_FISHEYE per lens, f ~ 1000 px), meetingroom.e57 (1.6 GB Faro scan, 107.6 M pts), meetingroom.ply
  (= the 65 k SfM points)}`. Not unpacked: rawimages/ (.insp), colmap/database.db, model_undistorted/.
- Frames: the .e57 is in the scanner's frame (bounds x -9.2..6.4, y -4.3..5.3, z 48..61), NOT the model's (y up,
  cameras at y ~ 0): `fiord.py align` estimates the Sim3 (FPFH/RANSAC + ICP with scale, venv-e57).
- Published numbers: FIORD paper 3DGS 23.3-28.1 dB PSNR across scenes (90/10 split, on the undistorted model);
  arXiv 2508.06968: Fisheye-GS 20.67 / 3DGUT 20.48 dB average at 200 deg, 21.91 / 21.64 dB with a 160 deg crop.
- Fetch + adapter: `python scripts/datasets/fiord.py fetch|prepare|align|geom configs/fiord_meetingroom.yaml`
  (helpers in scripts/datasets/omni_common.py). Split: every 8th capture (both lenses) held out = 14 of 106 captures.

### Ricoh360 (EgoNeRF)

- Source: EgoNeRF (https://github.com/changwoonchoi/EgoNeRF, MIT; Drive folder 1kqLAATjSSDwfLHI5O7RTfM9NOUi7PvcK),
  file Ricoh360.zip id 1qytZCyonAwroz7r6hEJ9LDBDh0rr_xuh. No checksum published: size-checked (2,713,936,566 B).
- Downloaded: `$MC/data/ricoh360/Ricoh360.zip`; text models of all 12 scenes + images of `bricks` unpacked under
  `$MC/data/ricoh360/Ricoh360/`. Camera `OPENCV_SPHERICAL` 1920x960 = equirect; the adapter verified the pixel
  convention against the SfM tracks (median reprojection 0.84 px; the same convention ODGS's rasteriser uses).
  The SfM model holds 236 frames; only the official 50 train + 50 test are released.
- Adapter: `python scripts/datasets/ricoh360.py fetch|prepare configs/ricoh360_bricks.yaml`.

## Methods for fisheye / 360 / multi-camera Gaussian splatting (survey 2026-10-19)

| Method | Camera handling | Code | Notes for us |
|--------|-----------------|------|--------------|
| 3DGUT / 3DGRUT (NVIDIA, CVPR 2025) | unscented transform through any camera model (OPENCV_FISHEYE, rolling shutter) | github.com/nv-tlabs/3dgrut (Apache-2.0); also gsplat `with_ut` | already in $CEAR_WS/3dgrut; reads FIORD's model directly |
| gsplat fisheye | `camera_model="fisheye"` (Fisheye-GS port, PR #398) + UT path | gsplat 1.5.x | viewer-matched trainer is pinhole-only |
| Fisheye-GS (ECCVW 2024) | equidistant projection + Jacobian | github.com/zmliao/Fisheye-GS | first-order: fails at extreme distortion (3DGUT paper) |
| 3DGEER (ICLR 2026, Bosch) | exact ray integral, any camera incl. >180 deg | github.com/boschresearch/3dgeer | exact alternative to UT; not tried |
| ODGS (NeurIPS 2024) | equirect: per-gaussian tangent-plane projection | github.com/esw0116/ODGS (MIT) | native 360 arm here (env/make_odgs_venv.sh) |
| OmniGS (WACV 2025) | equirect rasteriser | github.com/liquorleaf/OmniGS (C++/libtorch) | heavier build |
| 360-GS | layout-guided panoramic, indoor | github.com/LeoDarcy/360GS | indoor layouts only |
| SC-OmniGS (ICLR 2025) | self-calibrating 360 (poses + distortion) | github.com/chenyingshu/sc-omnigs | pose/intrinsics refinement idea |
| Seam360GS (ICCV 2025) | dual-fisheye seam/lens-gap model | not found | dual-lens calibration idea |
| FIORD eval (arXiv 2508.06968) | Fisheye-GS vs 3DGUT on 200 deg | - | 160 deg crop beats 200 deg; UniK3D depth init ~ SfM |
| Per-camera appearance | per-image/per-lens colour affine, bilateral grids (gsplat) | train_viewer_gsplat --appearance | lenses/cameras differ in exposure |
| Rig-constrained BA | rigid rig extrinsics, shared per-frame pose (COLMAP 3.12 rigs) | COLMAP | fewer pose DOF for multi-cam rigs |

Best three ideas for this pipeline: (1) rectify every fisheye/equirect frame into several overlapping virtual pinhole
views kept inside ~85 deg of the lens axis (no invalid pixels, full useful FOV) so the viewer-matched pinhole
trainer + SuperSplat fine-tune run unchanged (what SuperSplat draws is a pinhole render anyway); (2) train the
geometry natively (3DGUT / ODGS) with the periphery capped at ~160 deg, then viewer-fine-tune the exported PLY on the
rectified views (hybrid: native geometry, viewer-matched appearance); (3) per-camera (per-lens) appearance
transforms + rig-constrained poses for multi-camera rigs.
Implementation: scripts/datasets/{omni_common,fiord,ricoh360}.py, configs/{fiord_meetingroom,ricoh360_bricks}.yaml,
sbatch/omni_{rect,native,score}.sbatch, env/make_{odgs,e57}_venv.sh.

## FusionPortable v2 (HKUST(GZ)) — subset: legged_room00 + handheld_room00

- Source: https://fusionportable.github.io/dataset/fusionportable_v2/ (data page .../fusionportable_v2_data/), paper
  Wei, Jiao et al., "FusionPortableV2", IJRR 2024 (arXiv 2404.08563). Data: public Google Drive folder
  1PYhnf3PlY5r0hbyzWDGTUTPxRMl6SYa- (Baidu mirror code byj8); calibration: github.com/fusionportable/calibration_files.
  No form / registration.
- Licence: none stated on the site, the Drive folder, the GitHub repos (GitHub licence: null) or the paper (arXiv
  default licence only). Treated as public research data: academic use, cite the IJRR paper; ask the authors
  (jiaojh1994 at gmail) before any redistribution or commercial use.
- Contents: 27 sequences, 4 platforms (handheld, Unitree A1 legged robot, UGV, vehicle), ROS1 bags compressed with 7z
  (3.8-~30 GB each). Sensors (handheld suite on every platform): stereo FLIR BFS-U3-31S4C global-shutter colour
  1024x768 @ 20 Hz (83 x 66 deg, plumb_bob distortion, FIXED exposure/white balance per sequence, stamps
  exposure-centred), stereo DAVIS346 event cameras (346x260), Ouster OS1-128 (10 Hz, per-point time), STIM300 IMU,
  3DM-GQ7 INS; legged: A1 joint encoders / foot contacts / kinematic odometry. GT: room/indoor sequences = Leica MS60
  total station = 3-DoF prism POSITIONS only (gt_traj.txt has identity quaternions, 20 Hz); outdoor = RTK-INS 6-DoF.
  GT maps (Leica RTC360/BLK360) only for the campus and the underground parking (not the rooms). The Drive also holds
  reference SLAM outputs per sequence (FAST-LIO2, R3LIVE, VINS-Fusion, DROID-SLAM) and a KITTI-360-format export
  of 7 sequences (sensor_data_kitti360_format/, e.g. legged_room00 6.5 GB) — not used (the bag is the full data).
- Chosen: legged_room00 (quadruped, closest to CEAR; indoor room, people walking, 174 s) and handheld_room00 (same
  room, handheld, 140 s), both with GT positions. Calibration releases: legged -> 20230912_calib, handheld ->
  20230426_calib (the page's "20230403_calib" link points at the 20230426 folder; 20230403 is a 2048x1536
  non-overlapping-camera rig that does not match these 1024x768 bags).
- Downloaded (`MC=/scratch4/workspace/knaskar_umass_edu-multicam`; md5 of the Drive files recorded by the fetcher in
  `<seq>/MANIFEST.json`, Drive publishes none; `7z t` passed, bag opened with rosbags):
  - `$MC/data/fusionportable/legged_room00/legged_room00.bag` (17.30 GB bag; .7z 6,343,861,382 B md5
    bea5907c040dd97ec0062634306183af, deleted after unpacking), gt_traj.txt, fastlio2.txt, r3live.txt
  - `$MC/data/fusionportable/handheld_room00/handheld_room00.bag` (14.78 GB; .7z 5,302,738,838 B md5
    bf65d187aa4f84112ed84776d47d7151), gt_traj.txt, fastlio2.txt, r3live.txt
  - `$MC/data/fusionportable/calibration_files/` (git clone, 41 MB)
- Fetch: `sbatch -A $CEAR_SLURM_ACCOUNT -o logs/%x-%j.out sbatch/fp2_fetch.sbatch configs/fp2_<seq>.yaml`
  (scripts/datasets/fusionportable_fetch.py: gdown + 7z; Drive ids listed in the config). Other sequences: copy a
  config, change `sequence.name`, `rig.calib_release` and the ids (all ids are in the Drive folder listing).
- Adapter: `sbatch/fp2_prep.sbatch configs/fp2_<seq>.yaml` (scripts/datasets/fusionportable.py: extract / prepare /
  sfm / lidar / split, no ROS) -> `$MC/outputs/fp2_<seq>/pipeline/{dataset,onpath/{train,test},lidar_static.ply}`;
  splat: `sbatch/fp2_export.sbatch`, live-SuperSplat score: `sbatch/fp2_check.sbatch`.

### Fisheye / 360 results (2026-10-24; live SuperSplat 3.5.2 on held-out rectified views)

| Dataset | Best arm (export) | PSNR | SSIM | LPIPS | holes | geometry vs TLS F@5/10 cm |
|---|---|---|---|---|---|---|
| FIORD MeetingRoom (135 views) | rect_igs_long_ds1 ft (rectify + EDGS + Improved-GS 6M/60k + viewer FT 1:4) | 28.42 | 0.898 | 0.264 | 0 | 0.49 / 0.66 |
| FIORD, native 3DGUT 160 deg + EDGS init + viewer FT | native_3dgut_fov80_edgs_ft | 27.96 | 0.891 | 0.284 | 0.09% | 0.38 / 0.54 |
| Ricoh360 bricks (250 views) | rect_igs_long_wd ft (6M/60k + WD FT 13k, 1:4) | 28.38 | 0.893 | 0.174 | 0.05% | - |
| Ricoh360, native ODGS + viewer FT | native_odgs_ft | 25.16 | 0.828 | 0.290 | 23% | - |

FIORD geometry uses the one-time hand-picked scan alignment stored in configs/fiord_meetingroom.yaml (`tls_align`,
user decision 2026-10-22), scored inside the scan's height range. Exports for the viewer:
`<out>/exports/<seq>_best_view.ply` (scripts/view_frame.py: recentred, gravity-up; FIORD's camera was mounted upside
down, so --up +y).
