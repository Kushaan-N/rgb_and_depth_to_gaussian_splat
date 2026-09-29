"""Sim3 (similarity) between a COLMAP reconstruction and the metric pose frame — one shared solve.

The splat is trained in COLMAP's arbitrary-scale frame; everything metric (LiDAR, colliders, Isaac)
lives in the pose-chain frame. Both reconstructions contain the same images, so the similarity
comes from their camera centres (Umeyama). The pipeline solves it ONCE (sim3_align_splat.py writes
<out>/pipeline/sim3.json) and every later step reads that file, so all steps use the identical
transform. `get_sim3` falls back to solving from the two models when no JSON is given.

pycolmap is imported lazily, so importing this module works in the trainer venv too.
"""
from __future__ import annotations
import json
import numpy as np


def umeyama(src, dst):
    """Least-squares similarity (s, R, t) with dst ~= s * R @ src + t, for (N,3) point sets."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd((D.T @ S) / len(src))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = np.trace(np.diag(d) @ W) / ((S ** 2).sum() / len(src))
    return float(s), R, mu_d - s * R @ mu_s


def centers(model_path):
    """{image name: camera centre} of a COLMAP model."""
    import pycolmap
    return {img.name: np.asarray(img.projection_center()) for img in pycolmap.Reconstruction(model_path).images.values()}


def solve_sim3(colmap_model, gt_model, min_frames=10):
    """Solve COLMAP -> metric from shared images. Returns (s, R, t, residuals_m); raises if too few."""
    C = centers(colmap_model); G = centers(gt_model)
    common = sorted(set(C) & set(G))
    if len(common) < min_frames:
        raise ValueError(f"only {len(common)} images shared between {colmap_model} and {gt_model}")
    src = np.array([C[n] for n in common]); dst = np.array([G[n] for n in common])
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    return s, R, t, resid


def load_sim3(path):
    j = json.load(open(path))
    return float(j["scale"]), np.array(j["R"], dtype=np.float64), np.array(j["t"], dtype=np.float64)


def get_sim3(sim3_json=None, colmap_model=None, gt_model=None):
    """(s, R, t) from sim3_json if given, else solved from the two models."""
    if sim3_json:
        return load_sim3(sim3_json)
    if colmap_model and gt_model:
        return solve_sim3(colmap_model, gt_model)[:3]
    raise ValueError("need --sim3-json or both --colmap-model and --gt-model")
