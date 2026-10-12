"""Improved-GS densification as a gsplat Strategy (used by scripts/train_viewer_gsplat.py; gsplat venv).

Port of our 3DGRUT port ($CEAR_WS/3dgrut-densify threedgrut/strategy/improved.py, patches/3dgrut-improved-densify.patch)
to gsplat's Strategy API, same components, switches and defaults:
  "Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering" (CVPR 2026 Findings), Deng et al.,
  https://github.com/XiaoBin2001/Improved-GS -- ImprovedGS Non-Commercial Research License (non-commercial research,
  education and evaluation use only).
  EAS  edge-aware score: per-Gaussian sum_p edge_p * T_i(p) alpha_i(p) over sampled training views (Laplacian edge
       map of the real image, min-max normalised per view, averaged) decides WHICH gradient-qualified Gaussians split
  LAS  long-axis split: two children at +-3*0.45*s_max along the longest axis, long axis * 0.55, others
       * sqrt(1 - 0.45^2), opacity * 0.6 (no clone)
  RAP  recovery-aware pruning: opacity reset to <= 0.05 every 3000 it, 300 it later the lowest-opacity 20% are
       pruned (first 2 resets); plus an opacity < 0.02 prune at it 300
  budget growth control: splits per step capped by budget * sqrt(progress); candidates = gradient-qualified only
  MU   late multi-step updates: from 20k the optimizer steps every 5th it, from 24k every 20th (the trainer asks
       should_step_optimizer())
Unlike 3DGUT, gsplat has screen-space means: the candidate gradient is AbsGS's |dL/dmean2d| summed over pixels
(rasterize_to_pixels(absgrad=True)), in NDC units (x W/2, H/2) as in the reference code, threshold 0.0003.
The trainer supplies `edge_fn(view_index) -> (per-Gaussian weight (N,), visible (N,) bool)` for EAS.

    strategy = ImprovedStrategy(budget=4_000_000); state = strategy.initialize_state(scene_scale)
    strategy.step_pre_backward(params, opts, state, step, info); loss.backward()
    if strategy.should_step_optimizer(step): opt.step() ...
    strategy.step_post_backward(params, opts, state, step, info, edge_fn=edge_fn, n_views=len(train))
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa, split
from gsplat.utils import normalized_quat_to_rotmat

_FIND_EDGES = torch.tensor([[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]]).unsqueeze(0)


def unit_range(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    lo, hi = x.amin(), x.amax()
    return torch.zeros_like(x) if float(hi - lo) <= 0.0 else (x - lo) / (hi - lo)


def edge_map(rgb_u8: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) uint8 -> (H, W) edge weights in [0, 1] (8-bit grey, Laplacian 'find edges', min-max), as the port."""
    rgb = rgb_u8.float().permute(2, 0, 1)[None, :3]
    grey = torch.round((299.0 * rgb[:, 0:1] + 587.0 * rgb[:, 1:2] + 114.0 * rgb[:, 2:3]) / 1000.0)
    e = F.conv2d(grey, _FIND_EDGES.to(grey.device), padding=1)
    return unit_range(torch.clamp(e, 0.0, 255.0) / 255.0)[0, 0]


@dataclass
class ImprovedStrategy(Strategy):
    budget: int = 1_000_000
    budget_warmup_until_offset: int = 500
    split_distance: float = 0.45
    opacity_reduction: float = 0.6
    las: bool = True
    refine_every: int = 100
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    grad_threshold: float = 0.0003
    prune_opa: float = 0.005
    eas: bool = True
    eas_sample_cams: int = 10
    eas_full_every: int = 3000
    eas_full_offset: int = 400
    eas_full_until: int = 9000
    rap: bool = True
    reset_every: int = 3000
    reset_value: float = 0.05
    rap_prune_ratio: float = 0.2
    rap_offset: int = 300
    rap_rounds: int = 2
    early_prune_iter: int = 300
    early_prune_opa: float = 0.02
    mu: bool = True
    mu_start: int = 20_000
    mu_interval: int = 5
    mu_second_start: int = 24_000
    mu_second_interval: int = 20
    verbose: bool = True
    log: list = field(default_factory=list)

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale, "rap_triggers": set(), "rap_resets": 0,
                "edge_pool": []}

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        for k in ("means", "scales", "quats", "opacities"):
            assert k in params, f"{k} is required in params but missing."

    def _in_window(self, step: int) -> bool:
        return self.refine_start_iter < step < self.refine_stop_iter

    def should_step_optimizer(self, step: int) -> bool:
        if not self.mu or step < self.mu_start:
            return True
        return step % max(self.mu_interval if step < self.mu_second_start else self.mu_second_interval, 1) == 0

    def current_budget(self, step: int) -> int:
        start, end = self.refine_start_iter, self.refine_stop_iter - self.budget_warmup_until_offset
        if end <= start:
            return int(self.budget)
        p = min(max((step - start) / float(end - start), 0.0), 1.0)
        return int(self.budget) if p >= 1.0 else max(int(p ** 0.5 * self.budget), 1)

    def _say(self, msg: str):
        self.log.append(msg)
        if self.verbose:
            print(f"[ImprovedGS] {msg}", flush=True)

    # ---------------------------------------------------------------- callbacks
    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, edge_fn: Optional[Callable] = None,
                           n_views: int = 0):
        if self._in_window(step):
            self._update_state(params, state, info)
        if self.refine_every > 0 and self._in_window(step) and step % self.refine_every == 0:
            scores = self.edge_scores(state, step, edge_fn, n_views) if (self.eas and edge_fn is not None) else None
            self.densify_and_prune(params, optimizers, state, step, scores)
        if self.rap:
            if step == self.early_prune_iter:
                self._prune(params, optimizers, state, torch.sigmoid(params["opacities"].flatten()) < self.early_prune_opa, "early")
            if step in state["rap_triggers"]:
                op = torch.sigmoid(params["opacities"].detach().flatten())
                thr = torch.quantile(op, float(self.rap_prune_ratio)) if op.numel() < 2 ** 24 else \
                    op.kthvalue(max(int(op.numel() * float(self.rap_prune_ratio)), 1)).values
                self._prune(params, optimizers, state, op < thr, "RAP")
                state["rap_triggers"].discard(step)
            if self._in_window(step) and step % self.reset_every == 0:
                reset_opa(params=params, optimizers=optimizers, state=state, value=self.reset_value)
                if self.rap_prune_ratio > 0 and state["rap_resets"] < self.rap_rounds:
                    state["rap_triggers"].add(step + int(self.rap_offset))
                state["rap_resets"] += 1
                self._say(f"it {step}: opacity reset to <= {self.reset_value}")

    def _update_state(self, params, state, info):
        g = info["means2d"].absgrad.clone() if getattr(info["means2d"], "absgrad", None) is not None else info["means2d"].grad.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if state["grad2d"] is None or len(state["grad2d"]) != n:
            state["grad2d"] = torch.zeros(n, device=g.device); state["count"] = torch.zeros(n, device=g.device)
        sel = (info["radii"] > 0).all(dim=-1)                       # [C, N]
        ids = torch.where(sel)[1]
        state["grad2d"].index_add_(0, ids, g[sel].norm(dim=-1))
        state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def _reset_buffers(self, params, state):
        n = len(params["means"]); dev = params["means"].device
        state["grad2d"] = torch.zeros(n, device=dev); state["count"] = torch.zeros(n, device=dev)

    # ---------------------------------------------------------------- EAS
    def _sample_views(self, state, step: int, n_views: int) -> list:
        k = int(self.eas_sample_cams)
        if k == -1 or k >= n_views or (step % self.eas_full_every == self.eas_full_offset and step < self.eas_full_until):
            return list(range(n_views))
        out = []
        for _ in range(k):                                          # rotating pool: every view once before repeats
            if not state["edge_pool"]:
                state["edge_pool"] = list(range(n_views))
            out.append(state["edge_pool"].pop())
        return out

    @torch.no_grad()
    def edge_scores(self, state, step, edge_fn, n_views) -> Optional[torch.Tensor]:
        ids = self._sample_views(state, step, n_views)
        imp = seen = None
        for i in ids:
            w, vis = edge_fn(i)
            if imp is None:
                imp = torch.zeros_like(w); seen = torch.zeros_like(vis)
            imp[vis] += unit_range(w)[vis] / max(len(ids), 1)
            seen |= vis
        return imp if seen is not None and bool(seen.any()) else None

    # ---------------------------------------------------------------- densify / prune
    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, state, step, scores):
        n_cur = len(params["means"])
        if state["grad2d"] is None or len(state["grad2d"]) != n_cur:
            self._reset_buffers(params, state)
        grads = state["grad2d"] / state["count"]
        grads[grads.isnan()] = 0.0
        budget = self.current_budget(step)
        min_grad = float(self.grad_threshold)
        late = max(self.refine_stop_iter - 100, self.refine_start_iter)
        if scores is None or step > late:
            scores = grads
            if n_cur < budget and step > late:
                min_grad = min_grad / 1.5
        qual = grads >= min_grad
        n_cand = int(qual.sum())
        split_budget = min(budget, n_cand + n_cur) - n_cur
        n_split = 0
        if split_budget > 0:
            if self.las:
                n_split = self.long_axis_split(params, optimizers, state, scores, split_budget, qual)
            else:                                                    # 3DGS split of the gradient-qualified Gaussians
                n_split = n_cand
                split(params=params, optimizers=optimizers, state=state, mask=qual)
                self._reset_buffers(params, state)
        self._say(f"it {step}: budget {budget}, {n_cand} candidates (grad >= {min_grad:.2e}), split {n_split} -> "
                  f"{len(params['means'])}")
        if step < late:
            self._prune(params, optimizers, state, torch.sigmoid(params["opacities"].flatten()) < self.prune_opa, "density")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def long_axis_split(self, params, optimizers, state, scores, budget, qual) -> int:
        n = len(params["means"])
        imp = torch.zeros(n, dtype=torch.float32, device=params["means"].device)
        imp[: scores.shape[0]] = scores.detach().float().clamp_min(0)
        imp[~qual] = 0
        n_pos = int((imp > 0).sum())
        if budget <= 0 or n_pos == 0:
            return 0
        budget = min(int(budget), n_pos)
        mask = torch.zeros(n, dtype=torch.bool, device=imp.device)
        if budget >= n_pos:
            mask = imp > 0
        else:                                                        # torch.multinomial caps at 2^24 categories
            if n <= 2 ** 24:
                mask[torch.multinomial(imp, budget, replacement=False)] = True
            else:                                                    # Gumbel top-k == weighted sampling w/o replacement
                key = torch.log(imp.clamp_min(1e-30)) - torch.log(-torch.log(torch.rand_like(imp).clamp_min(1e-12)))
                key[imp <= 0] = -float("inf"); mask[key.topk(budget).indices] = True
        d = float(self.split_distance)
        stds = torch.exp(params["scales"][mask])
        max_v, max_i = torch.max(stds, dim=1, keepdim=True)
        axis = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_i, True)
        off = stds * axis * 3.0 * d
        off = torch.cat([off, -off], 0)
        R = normalized_quat_to_rotmat(F.normalize(params["quats"][mask], dim=-1)).repeat(2, 1, 1)
        new_means = torch.bmm(R, off.unsqueeze(-1)).squeeze(-1) + params["means"][mask].repeat(2, 1)
        rw, rh = max(1.0 - d, 1e-6), math.sqrt(max(1.0 - d * d, 1e-6))
        new_scales = torch.log(stds.scatter(1, max_i, max_v * rw / rh).repeat(2, 1) * rh)
        new_op = torch.logit((torch.sigmoid(params["opacities"][mask]) * float(self.opacity_reduction)).clamp(1e-6, 1 - 1e-6)).repeat(2)
        k = int(mask.sum())

        def param_fn(name, p):
            if name == "means":
                child = new_means
            elif name == "scales":
                child = new_scales
            elif name == "opacities":
                child = new_op.reshape((-1,) + p.shape[1:])
            else:
                child = p[mask].repeat([2] + [1] * (p.dim() - 1))
            return torch.nn.Parameter(torch.cat([p[~mask], child.to(p.dtype)]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v[~mask], torch.zeros((2 * k, *v.shape[1:]), device=v.device, dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        self._reset_buffers(params, state)                           # reference resets its accumulators on growth
        return k

    @torch.no_grad()
    def _prune(self, params, optimizers, state, mask, label):
        p = int(mask.sum())
        if p == 0:
            return
        n = len(mask)
        if state["grad2d"] is None or len(state["grad2d"]) != n:
            self._reset_buffers(params, state)
        remove(params=params, optimizers=optimizers, state=state, mask=mask)   # indexes the per-Gaussian tensors only
        self._say(f"{label}-pruned {p} / {n} ({p / n * 100:.2f}%) gaussians")
