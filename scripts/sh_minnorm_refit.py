#!/usr/bin/env python3
"""Post-hoc SH minimum-norm refit: view-dependent colour may only do, outside the directions the training cameras saw
a gaussian from, what those directions support (gsplat venv, GPU; closed form, no optimisation).

A ground robot sees every gaussian from a narrow cone of directions. Degree-3 SH fitted there are free everywhere else
and typically explode: from above / the side the splat shows saturated "glitter". Per gaussian:
  1. observed directions: every training view whose blending weight for it (sum_px T alpha, the trainer's viewer-matched
     rasteriser) exceeds --min-w; weight w_v, direction d_v, observed colour y_v = max(SH(d_v) + 0.5, 0);
     normal equations A = sum w Y Y^T, b = sum w Y y (Y = the 16 real SH basis values), observed range [lo, hi] per channel;
  2. colour on a full sphere sampling (--sphere directions) must stay within [lo - margin, hi + margin]. Gaussians whose
     CURRENT coefficients already do are left untouched (no change on the training path at all);
  3. the others are refitted, c = argmin sum w (Y c - y)^2 + lambda c^T D c, D = diag(0 | 1 x3 | 3 x5 | 6 x7) (no penalty
     on the mean colour, growing with the band = smoothness on the sphere), with the SMALLEST lambda of --ladder that
     passes the sphere check: observed colours are kept as far as the bound allows, and outside the observed cone the
     colour relaxes smoothly towards the mean observed colour. Last rung: the weighted mean colour alone (bands 1-3 = 0).
  Gaussians no training view supports keep their DC; --unseen zero (default) clears their bands 1-3.
Writes the same PLY layout (scripts/finetune_ply_viewer.py save_ply) + <out>.json (rung shares, in-cone residuals,
sphere violations before / after).

    PYTHONPATH=$CEAR_WS/gsplat-viewer-ss352 python scripts/sh_minnorm_refit.py --ply s.ply --dataset TRAIN --out s_shfit.ply
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_ply_viewer import C0, C1, C2, C3, eval_sh, load_ply, save_ply  # noqa: E402


def sh_basis(d):
    """(M,3) unit directions -> (M,16): colour - 0.5 = basis @ [sh0 | shN] (3DGS / PlayCanvas convention, as eval_sh)."""
    import torch
    x, y, z = d.unbind(-1); xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return torch.stack([torch.full_like(x, C0), -C1 * y, C1 * z, -C1 * x,
                        C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
                        C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                        C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], -1)


def fibonacci(n, dev):
    import torch
    i = torch.arange(n, device=dev, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n; r = torch.sqrt(1 - z * z); ph = i * math.pi * (3 - math.sqrt(5))
    return torch.stack([r * torch.cos(ph), r * torch.sin(ph), z], -1).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True); ap.add_argument("--dataset", required=True, help="training frames (COLMAP text dataset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-split-interval", type=int, default=0)
    ap.add_argument("--viewer", default="ss352", choices=("ss352", "pc274"))
    ap.add_argument("--min-w", type=float, default=0.01, help="blending weight (pixel-equivalents) for a view to count")
    ap.add_argument("--margin", type=float, default=0.1, help="allowed excursion beyond the observed colour range (0-1 units)")
    ap.add_argument("--ladder", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    ap.add_argument("--sphere", type=int, default=128)
    ap.add_argument("--unseen", choices=("keep", "zero"), default="zero")
    ap.add_argument("--chunk", type=int, default=200000)
    a = ap.parse_args()
    import torch
    import train_viewer_gsplat as tvg
    dev = "cuda"; t0 = time.time()
    vertex, p = load_ply(a.ply, dev); N = len(p["means"]); K = p["shN"].shape[1]
    if K != 15:
        raise SystemExit(f"[shfit] needs SH degree 3 (15 f_rest coefficients per channel), got {K}")
    with torch.no_grad():                                           # basis self-check against the viewer's SH evaluation
        dd = torch.nn.functional.normalize(torch.randn(4096, 3, device=dev), dim=-1)
        c_all = torch.cat([p["sh0"], p["shN"]], 1)[:4096]
        dd = dd[: len(c_all)]
        mine = (torch.einsum("mk,mkc->mc", sh_basis(dd), c_all) + 0.5).clamp_min(0)
        err = float((mine - eval_sh(p["sh0"][:4096], p["shN"][:4096], dd, 3, False)).abs().max())
        assert err < 1e-3, f"SH basis mismatch {err}"
    views = tvg.load_views(a.dataset, a.test_split_interval, "train", dev)
    rast = tvg.ViewerGS(argparse.Namespace(viewer=a.viewer, eps2d=None, cull_l1=None, aa=False, sh_band2="std", min_px=0.0,
                                           near=1e-4, cap_px=1024.0, sort="depth"))
    print(f"[shfit] {a.ply}: {N} gaussians; {len(views)} training views; basis check {err:.1e}", flush=True)
    C = torch.cat([p["sh0"], p["shN"]], 1)                          # (N,16,3)
    A = torch.zeros(N, 256, device=dev); b = torch.zeros(N, 16, 3, device=dev); ws = torch.zeros(N, device=dev)
    lo = torch.full((N, 3), float("inf"), device=dev); hi = torch.full((N, 3), float("-inf"), device=dev)
    nv = torch.zeros(N, device=dev)
    for k, v in enumerate(views):
        w, _ = rast.edge_weights(p, v, torch.ones(v[3], v[2], device=dev))
        with torch.no_grad():
            sel = torch.nonzero(w > a.min_w)[:, 0]
            if len(sel) == 0:
                continue
            campos = torch.linalg.inv(v[4])[:3, 3]
            Y = sh_basis(torch.nn.functional.normalize(p["means"][sel] - campos, dim=-1))
            y = torch.einsum("mk,mkc->mc", Y, C[sel]).clamp_min(-0.5)
            wv = w[sel]
            A.index_add_(0, sel, (wv[:, None, None] * Y[:, :, None] * Y[:, None, :]).reshape(-1, 256))
            b.index_add_(0, sel, wv[:, None, None] * Y[:, :, None] * y[:, None, :])
            ws.index_add_(0, sel, wv); nv.index_add_(0, sel, torch.ones_like(wv))
            lo[sel] = torch.minimum(lo[sel], y); hi[sel] = torch.maximum(hi[sel], y)
        if k % 50 == 0:
            print(f"[shfit] view {k}/{len(views)}: {len(sel)} supported gaussians ({time.time() - t0:.0f}s)", flush=True)
    del views; torch.cuda.empty_cache()
    seen = ws > 0
    S = sh_basis(fibonacci(a.sphere, dev)).double()                 # (S,16)
    D = torch.diag(torch.tensor([0.0] + [1.0] * 3 + [3.0] * 5 + [6.0] * 7, device=dev, dtype=torch.float64))
    eye = torch.eye(16, device=dev, dtype=torch.float64)
    newC = C.clone(); rung = torch.full((N,), -2, dtype=torch.long, device=dev)       # -2 unseen, -1 kept, 0.. ladder, len = mean
    viol0 = torch.zeros(N, device=dev); viol1 = torch.zeros(N, device=dev); res = torch.zeros(N, device=dev)

    def violation(c, l, h):                                         # c (n,16,3) float64 -> max excursion beyond [l - m, h + m]
        col = torch.einsum("sk,nkc->nsc", S, c)
        return torch.maximum(col - (h[:, None] + a.margin), (l[:, None] - a.margin) - col).amax((1, 2)).clamp_min(0)

    idx_all = torch.nonzero(seen)[:, 0]
    with torch.no_grad():
        for s0 in range(0, len(idx_all), a.chunk):
            idx = idx_all[s0:s0 + a.chunk]
            Ai = (A[idx] / ws[idx, None]).double().reshape(-1, 16, 16); bi = (b[idx] / ws[idx, None, None]).double()
            l, h = lo[idx].double(), hi[idx].double(); c0 = C[idx].double()
            v0 = violation(c0, l, h); viol0[idx] = v0.float()
            done = v0 <= 0; r = torch.where(done, -1, len(a.ladder)); out = c0.clone()
            mean = torch.zeros_like(c0); mean[:, 0] = bi[:, 0] / Ai[:, 0, 0, None]     # weighted mean colour (bands 1-3 = 0)
            for li, lam in enumerate(a.ladder):
                todo = torch.nonzero(~done)[:, 0]
                if len(todo) == 0:
                    break
                M = Ai[todo] + lam * D + 1e-10 * eye
                c = torch.linalg.solve(M, bi[todo])
                ok = (violation(c, l[todo], h[todo]) <= 0) & torch.isfinite(c).all((1, 2))
                t = todo[ok]; out[t] = c[ok]; r[t] = li; done[t] = True
            rest = torch.nonzero(~done)[:, 0]; out[rest] = mean[rest]
            viol1[idx] = violation(out, l, h).float()
            # in-cone change: weighted rms of (Y c_new - Y c_old) = sqrt(dc^T A dc) (unclamped colours)
            dc = out - c0
            res[idx] = torch.sqrt(torch.einsum("nkc,nkj,njc->nc", dc, Ai, dc).clamp_min(0).mean(-1)).float()
            newC[idx] = out.float(); rung[idx] = r
        if a.unseen == "zero":
            newC[~seen, 1:] = 0
        op = torch.sigmoid(p["opacities"].flatten())
        share = lambda m: round(float(m.float().mean()), 4)
        info = {"ply": a.ply, "out": a.out, "n": N, "args": vars(a), "seconds": None,
                "unseen": share(~seen), "kept_unchanged": share(rung == -1),
                "ladder": {str(lam): share(rung == li) for li, lam in enumerate(a.ladder)}, "mean_only": share(rung == len(a.ladder)),
                "views_per_seen_gaussian_pct_10_50_90": [float(x) for x in torch.quantile(nv[seen][:: max(1, int(seen.sum()) // 1000000)],
                                                                                         torch.tensor([0.1, 0.5, 0.9], device=dev))],
                "sphere_violation_before": {"gt0": share(viol0[seen] > 0), "gt0.25": share(viol0[seen] > 0.25), "gt1": share(viol0[seen] > 1),
                                            "opacity_weighted_gt0.25": round(float((op[seen] * (viol0[seen] > 0.25)).sum() / op[seen].sum()), 4)},
                "sphere_violation_after": {"gt0": share(viol1[seen] > 1e-6), "max": round(float(viol1.max()), 4)},
                "in_cone_rms_change_of_refitted": {"mean": round(float(res[rung >= 0].mean()), 5) if bool((rung >= 0).any()) else None,
                                                   "p90": round(float(torch.quantile(res[rung >= 0][:: max(1, int((rung >= 0).sum()) // 1000000)], 0.9)), 5)
                                                   if bool((rung >= 0).any()) else None}}
    p2 = dict(p); p2["sh0"] = newC[:, :1].contiguous(); p2["shN"] = newC[:, 1:].contiguous()
    save_ply(a.out, vertex, p2)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(os.path.splitext(a.out)[0] + "_shfit.json", "w"), indent=1)
    print("SHFIT " + json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
