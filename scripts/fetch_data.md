# Fetching CEAR data (Phase 1)

CEAR is hosted on Google Drive (project site: https://daroslab.github.io/cear/,
Downloads: https://daroslab.github.io/cear/Downloads/). Download **into `$CEAR_DATA`**
(scratch), never into the repo or `$HOME`.

```bash
source env/cear_env.sh          # sets $CEAR_DATA on scratch

# option A: gdown (per-file Google Drive)
pip install gdown                # into the CPU venv is fine
gdown --fuzzy "<google-drive-link-for-mocap1_well-lit_trot>" -O "$CEAR_DATA/mocap1_well-lit_trot.zip"
unzip "$CEAR_DATA/mocap1_well-lit_trot.zip" -d "$CEAR_DATA/"

# option B: rclone (if you configured a remote for the shared drive)
# rclone copy cear-drive:CEAR/mocap1_well-lit_trot "$CEAR_DATA/mocap1_well-lit_trot"
```

Also download the **calibration release** (intrinsics + distortion for RealSense;
extrinsics for RGB-Depth, RGB-Robot, RGB-Marker, Robot Links) and place the RGB intrinsics
where `configs/<seq>.yaml : intrinsics.calib_file` points, in this schema:

```yaml
# rgb_intrinsics.yaml
K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
dist: [k1, k2, p1, p2, k3]     # released RealSense coefficients
dist_model: opencv
width: 640
height: 480
# extrinsics as 4x4 row-major; direction is set in the config (frames.extrinsic.*)
T_rgb_marker: [[...],[...],[...],[0,0,0,1]]   # cam_to_marker per config default
T_rgb_robot:  [[...],[...],[...],[0,0,0,1]]
```

Then verify everything landed and is self-consistent:

```bash
python scripts/inventory.py --config configs/mocap1_well-lit_trot.yaml    # Gate 1
```

Notes:
- `lidar.bag` may be present in the download; it is **never read** (project constraint).
- VERIFY at Gate 1: depth is 16-bit, 1 mm/unit, 0 = invalid; image size 640x480;
  which pose file exists (`MoCap.txt` inside the OptiTrack volume, else `FasterLIO.txt`).
- Compute-node egress may be blocked — download from the login/data-transfer node.
