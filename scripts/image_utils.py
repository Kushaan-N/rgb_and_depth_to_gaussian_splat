"""Image + depth IO and pinhole geometry shared across Phases 1-5.

Kept separate from pose_utils (which is pure pose/frame math) so both stay small.
All depth is handled in metres here; the raw 16-bit units and invalid convention are
read from the config (Phase-1 VERIFY items).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_rgb(path: str) -> np.ndarray:
    """Read an 8-bit image as HxWx3 RGB (uint8)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_raw(path: str) -> np.ndarray:
    """Read a depth image exactly as stored (typically uint16)."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d


def depth_to_meters(depth_raw: np.ndarray, cfg: dict,
                    truncate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw depth -> (depth_m float32, valid_mask bool).

    Applies the invalid-value convention and, if ``truncate``, the [min,max] range
    truncation from cfg.depth (§2.7). Invalid/out-of-range pixels are set to 0 and
    excluded from the mask.
    """
    dcfg = cfg["depth"]
    upm = float(dcfg["units_per_meter"])
    invalid = dcfg.get("invalid_value", 0)
    depth_m = depth_raw.astype(np.float32) / upm
    valid = depth_raw != invalid
    if truncate:
        valid &= depth_m >= float(dcfg.get("min_range_m", 0.0))
        valid &= depth_m <= float(dcfg.get("max_range_m", np.inf))
    depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
    return depth_m, valid


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def K_params(K: np.ndarray) -> Tuple[float, float, float, float]:
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def undistort_rgb(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort with the released coefficients, keeping the same K (PINHOLE)."""
    if dist is None or np.allclose(dist, 0):
        return img
    return cv2.undistort(img, K, dist, None, K)


# --------------------------------------------------------------------------- #
# Pinhole back-projection / projection
# --------------------------------------------------------------------------- #
def backproject(depth_m: np.ndarray, K: np.ndarray, valid: np.ndarray,
                rgb: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Back-project valid depth pixels to 3D points in the CAMERA frame.

    Returns (points[N,3], colors[N,3] or None, pix_idx[N,2] as (v,u)). Depth is the
    camera-frame z (standard pinhole), so X = (u-cx)/fx * z, Y = (v-cy)/fy * z, Z = z.
    """
    fx, fy, cx, cy = K_params(K)
    vs, us = np.nonzero(valid)
    z = depth_m[vs, us]
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    cols = rgb[vs, us].astype(np.uint8) if rgb is not None else None
    return pts, cols, np.stack([vs, us], axis=1)


def project(points_cam: np.ndarray, K: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to (uv[N,2] float, z[N] float). z<=0 => behind cam."""
    fx, fy, cx, cy = K_params(K)
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * points_cam[:, 0] / z + cx
        v = fy * points_cam[:, 1] / z + cy
    return np.stack([u, v], axis=1), z


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def laplacian_var(rgb: np.ndarray) -> float:
    """Variance of the Laplacian — a standard sharpness proxy (higher = sharper)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
