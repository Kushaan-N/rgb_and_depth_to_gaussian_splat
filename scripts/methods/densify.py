"""Densification variants for scripts/train_viewer_gsplat.py's improved strategy (scripts/improved_gs_strategy.py is
patched per instance, the shared module is untouched; see scripts/methods/__init__.py).
  --pixelgs: Pixel-GS (Zhang et al., ECCV 2024, arXiv 2403.15530): the split-candidate gradient of a gaussian is the
    per-view |dL/dmean2d| averaged with its pixel footprint in that view as the weight, sum_k pix_k |g_k| / sum_k pix_k,
    instead of the plain mean over the views where it is visible. Large gaussians that cover many pixels in some views
    but are seen tiny / at the border in most others (blurry under-reconstructed areas the sparse init missed) then reach
    the threshold. Footprint = rx * ry of gsplat's screen radii (proportional to the covered area; scale cancels).
  --pixelgs-depth x (> 0): the gradient is multiplied by min(1, z / (x scene_scale)) (Pixel-GS' near-camera floater
    damping: gaussians right in front of a camera get a large footprint and would otherwise over-split).
"""
from __future__ import annotations
import types


def add_densify_args(g):
    g.add_argument("--pixelgs", action="store_true", help="Pixel-GS footprint-weighted densification gradient (improved strategy)")
    g.add_argument("--pixelgs-depth", type=float, default=0.0, help="Pixel-GS depth damping scale (x scene_scale; 0 = off)")


def patch_strategy(strat, args, scene_scale):
    if not args.pixelgs:
        return {}
    if not hasattr(strat, "_update_state") or not hasattr(strat, "edge_scores"):
        raise SystemExit("[methods] --pixelgs needs --strategy improved")
    damp = args.pixelgs_depth * scene_scale

    def _update_state(self, params, state, info):
        import torch
        g = info["means2d"].absgrad.clone() if getattr(info["means2d"], "absgrad", None) is not None else info["means2d"].grad.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if state["grad2d"] is None or len(state["grad2d"]) != n:
            state["grad2d"] = torch.zeros(n, device=g.device); state["count"] = torch.zeros(n, device=g.device)
        sel = (info["radii"] > 0).all(dim=-1)                       # [C, N]
        ids = torch.where(sel)[1]
        pix = (info["radii"][sel][:, 0].float() * info["radii"][sel][:, 1].float()).clamp_min(1.0)
        gn = g[sel].norm(dim=-1)
        if damp > 0:
            gn = gn * (info["depths"][sel].detach() / damp).clamp(0.0, 1.0)
        state["grad2d"].index_add_(0, ids, gn * pix)
        state["count"].index_add_(0, ids, pix)

    strat._update_state = types.MethodType(_update_state, strat)
    print(f"[methods] Pixel-GS densification gradient (footprint-weighted{f', depth damping {damp:.3f}' if damp > 0 else ''})", flush=True)
    return {"pixelgs": True, "pixelgs_depth_abs": round(damp, 4)}
