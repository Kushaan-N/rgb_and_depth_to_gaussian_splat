#!/bin/bash
# LiDAR geometry path (CPU): accumulate the Velodyne bag into a dense metric world cloud, then
# gravity-align + room-crop + Poisson + decimate it into an Isaac collider. Much better floor/wall
# coverage than the ground-level RGB-D depth (mocap2: floor 7/9, walls 4/4, 0 rovers fell).
# Needs the CEAR LiDAR extrinsics in <data>/_calib (event_rgb_extrinsic, event_lidar_extrinsic) and
# the velodyne time offset in the sequence config.
#
#   sbatch -A pi_donghyunkim_umass_edu sbatch/build_lidar.sbatch mocap2_well-lit_trot
#SBATCH -J cear-lidar
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 40:00
#SBATCH -o %x_%j.out

set -euo pipefail
SEQ="${1:-mocap2_well-lit_trot}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$REPO/env/cear_env.sh"
NAV="$CEAR_OUT/$SEQ"

echo "=== [1/2] accumulate Velodyne bag -> metric world cloud ==="
python3 "$REPO/scripts/build_lidar_cloud.py" --config "$REPO/configs/$SEQ.yaml" \
  --out "$NAV/lidar_cloud.ply" --voxel 0.02 --min-range 0.5 --max-range 12.0

echo "=== [2/2] cloud -> gravity-align + room-crop + Poisson + decimate -> collider_lidar ==="
python3 "$REPO/scripts/build_lidar_collider.py" --cloud "$NAV/lidar_cloud.ply" --out "$NAV/collider_lidar"

echo "BUILD_LIDAR_DONE $SEQ  -> $NAV/collider_lidar/collider.obj"
