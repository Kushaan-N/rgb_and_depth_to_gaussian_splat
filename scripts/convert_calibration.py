"""Convert the CEAR calibration release into the pipeline's rgb_intrinsics.yaml schema.

CEAR ships calibration as several small YAML files (one per intrinsic/extrinsic). This
merges the RealSense RGB intrinsics + the RGB↔Marker and RGB↔Robot extrinsics into the
single file `pose_utils.load_calibration` expects. The calibration is shared across
sequences captured with the same rig, so one output serves mocap1/2/3.

CEAR conventions captured here (from the file headers, verified 2026-09-20):
  * realsense_intrinsics: RGB_cam.intrinsics = [fx, fy, cx, cy]; distortion_model radtan
    with 5 coeffs = OpenCV [k1, k2, p1, p2, k3]; resolution [w, h].
  * T_rgb_marker / T_rgb_robot: "takes a point FROM marker/robot frame TO the RGB frame",
    i.e. they map marker→rgb / robot→rgb  ==  the `marker_to_cam` / `robot_to_cam`
    direction in the config (set frames.extrinsic.*_direction accordingly).

    python scripts/convert_calibration.py --calib-dir $CEAR_DATA/_calib \
        --out $CEAR_DATA/mocap1_well-lit_trot/calibration/rgb_intrinsics.yaml
"""

from __future__ import annotations

import argparse
import os

import yaml


def convert(calib_dir: str) -> dict:
    def _load(name):
        with open(os.path.join(calib_dir, name)) as f:
            return yaml.safe_load(f)

    intr = _load("realsense_intrinsics")["RGB_cam"]
    fx, fy, cx, cy = intr["intrinsics"]
    w, h = intr["resolution"]
    dist = list(intr["distortion_coeffs"])           # radtan == OpenCV k1,k2,p1,p2,k3

    T_rgb_marker = _load("rgb_marker_extrinsic")["T_rgb_marker"]
    T_rgb_robot = _load("rgb_robot_extrinsic")["T_rgb_robot"]

    return {
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "dist": dist,
        "dist_model": "opencv",       # radtan (k1,k2,p1,p2,k3) is OpenCV's plumb_bob order
        "width": int(w), "height": int(h),
        # stored maps marker->rgb / robot->rgb; config uses *_direction: marker_to_cam/robot_to_cam
        "T_rgb_marker": T_rgb_marker,
        "T_rgb_robot": T_rgb_robot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", required=True, help="dir with the CEAR calib files")
    ap.add_argument("--out", required=True, help="output rgb_intrinsics.yaml path")
    args = ap.parse_args()
    calib = convert(args.calib_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    fx = calib["K"][0][0]; fy = calib["K"][1][1]
    print(f"[convert_calibration] wrote {args.out}")
    print(f"  fx={fx} fy={fy} cx={calib['K'][0][2]} cy={calib['K'][1][2]} "
          f"{calib['width']}x{calib['height']} dist={calib['dist']}")
    print("  NOTE: set config frames.extrinsic.rgb_marker_direction=marker_to_cam, "
          "rgb_robot_direction=robot_to_cam (CEAR stores marker→rgb / robot→rgb).")


if __name__ == "__main__":
    main()
