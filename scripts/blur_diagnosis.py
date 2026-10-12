"""Does motion blur limit fidelity? Per-frame motion-blur estimate from poses + direct image evidence.

Predicted blur (px) of every RGB frame = |image-plane flow rate| * t_exp, where the flow rate comes from the
camera pose track (mocap/SLAM poses at t +- dt through the same extrinsic chain as build_poses.py) applied to
the pixel grid at real depth (the frame's own <image>_depth.npy LiDAR target if it is a used frame, else the
temporally nearest used frame's; fallback --z-default metres). t_exp is unknown (RealSense metadata is not
shipped), so it is (a) bracketed (--texp-ms) and (b) ESTIMATED from the images themselves:

  directional-sharpness fit: a box blur of length L along phi attenuates image-gradient energy along phi more
  than across it. Per frame, per tile, the anisotropy log(E_a / E_b) for direction pairs (0,90) and (45,135)
  deg is detrended by its running temporal median (scene content cancels; gait-driven motion does not) and
  compared with the anisotropy the predicted blur vector would produce for each candidate t_exp (attenuation
  table computed exactly in Fourier space from the sharpest frames' mean power spectrum). The best t_exp
  minimises the residual; block bootstrap gives a CI. Also reported: Spearman correlation of detrended
  log-Laplacian-variance against detrended predicted flow rate (negative = blur visible), and an IMU
  cross-check of the pose-derived angular rate. Counts of frames the gate drops come from gating_meta.json.

    python scripts/blur_diagnosis.py --config configs/mocap2_well-lit_trot.yaml \
        --used-dir $P/onpath/data_orig [--out-dir $P/blur] [--texp-ms 5,10,20,30]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
import image_utils as iu
from build_poses import compute_world_cam

DIRS = {0: (1, 0), 90: (0, 1), 45: (1, 1), 135: (1, -1)}     # (du, dv) pixel steps
PAIRS = [(0, 90), (45, 135)]


def flow_rates(Twc, Tm, Tp, dt, K, uv, rng):
    """Image-plane flow rate (px/s) at pixels uv (M,2) with ray distances rng (N,M) metric. Returns (N,M,2)."""
    Ki = np.linalg.inv(K)
    d = (Ki @ np.c_[uv, np.ones(len(uv))].T).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    out = np.zeros((len(Twc), len(uv), 2))
    for i in range(len(Twc)):
        Xc = rng[i][:, None] * d                                           # camera frame at t
        Xw = Xc @ Twc[i, :3, :3].T + Twc[i, :3, 3]
        px = []
        for T in (Tm[i], Tp[i]):
            Tcw = pu.invert_T(T)
            Y = Xw @ Tcw[:3, :3].T + Tcw[:3, 3]
            p = Y @ K.T
            px.append(p[:, :2] / np.maximum(p[:, 2:3], 1e-6))
        out[i] = (px[1] - px[0]) / (2 * dt)
    return out


def tile_energies(gray, nty, ntx):
    """Gradient energy per tile and direction -> (nty*ntx, 4) in DIRS order, plus Laplacian variance."""
    import cv2
    H, W = gray.shape
    th, tw = H // nty, W // ntx
    E = np.zeros((nty * ntx, len(DIRS) + 1))                              # + 4-neighbour Laplacian energy
    lp = np.zeros_like(gray); lp[1:-1, 1:-1] = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
                                               + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    for ty in range(nty):
        for tx in range(ntx):
            E[ty * ntx + tx, -1] = (lp[ty * th + 1:(ty + 1) * th - 1, tx * tw + 1:(tx + 1) * tw - 1] ** 2).mean()
    for k, (du, dv) in enumerate(DIRS.values()):
        a = gray[max(-dv, 0):H - max(dv, 0), :W - du]
        b = gray[max(dv, 0):H + min(dv, 0), du:]
        g2 = (b - a) ** 2
        for ty in range(nty):
            for tx in range(ntx):
                E[ty * ntx + tx, k] = g2[ty * th:(ty + 1) * th - 1, tx * tw:(tx + 1) * tw - 1].mean()
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return E, lap


def attenuation_table(tiles, Ls, phis):
    """H[dir, L, phi] = log E_dir(box blur L along phi) / E_dir(sharp), from the mean power spectrum of tiles."""
    n = tiles[0].shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    P = np.mean([np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2 for t in tiles], axis=0)
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    H = np.zeros((len(DIRS) + 1, len(Ls), len(phis)))
    ops = [np.abs(np.exp(2j * np.pi * (kx * du + ky * dv)) - 1) ** 2 for du, dv in DIRS.values()]
    ops.append((2 * np.cos(2 * np.pi * kx) + 2 * np.cos(2 * np.pi * ky) - 4) ** 2)   # Laplacian
    for k, D in enumerate(ops):
        E0 = (P * D).sum()
        for j, ph in enumerate(phis):
            kp = kx * np.cos(ph) + ky * np.sin(ph)
            for i, L in enumerate(Ls):
                H[k, i, j] = np.log((P * D * np.sinc(L * kp) ** 2).sum() / E0)
    return H


def running_median(x, w):
    from scipy.ndimage import median_filter
    return x - median_filter(x, size=(2 * w + 1,) + (1,) * (x.ndim - 1), mode="nearest")


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean())} if len(a) else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--used-dir", default=None, help="dataset of frames used for training/eval (images/*.png "
                    "and optional *_depth.npy); default {out_root}/pipeline/dataset")
    ap.add_argument("--sim3", default=None, help="sim3.json (x_metric = s R x_colmap + t) to make depth metric; "
                    "default {out_root}/pipeline/sim3.json, identity if missing")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--texp-ms", default="5,10,20,30")
    ap.add_argument("--dt-ms", type=float, default=8.0, help="half-window for pose finite differences")
    ap.add_argument("--z-default", type=float, default=2.0)
    ap.add_argument("--tiles", default="4x3", help="tile grid (cols x rows) for the image fit")
    ap.add_argument("--detrend", type=int, default=6, help="running-median half window (frames)")
    ap.add_argument("--bin", type=int, default=1, help="area-bin images by this factor first (less sensor "
                    "noise in the gradient energies, which otherwise dilutes the blur signal)")
    ap.add_argument("--no-images", action="store_true", help="pose-only (skip the image evidence)")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    out_root = cfg["paths"]["out_root"]
    used_dir = args.used_dir or os.path.join(out_root, "pipeline", "dataset")
    out = args.out_dir or os.path.join(out_root, "pipeline", "blur")
    os.makedirs(out, exist_ok=True)
    sim3 = args.sim3 or os.path.join(out_root, "pipeline", "sim3.json")
    scale = json.load(open(sim3))["scale"] if os.path.exists(sim3) else 1.0

    kept, Twc, calib, interp, mocap, _ = compute_world_cam(cfg)
    ts = np.array([f.t for f in kept]); names = [f.rgb_name for f in kept]; N = len(kept)
    dt = args.dt_ms / 1e3
    Tm = pu.build_world_to_cam_track(mocap, interp, calib, cfg, np.clip(ts - dt, interp.tmin, interp.tmax))
    Tp = pu.build_world_to_cam_track(mocap, interp, calib, cfg, np.clip(ts + dt, interp.tmin, interp.tmax))
    K = calib.K; W, H = calib.width, calib.height; f = float(np.sqrt(K[0, 0] * K[1, 1]))

    used = sorted(os.path.basename(p) for p in glob.glob(os.path.join(used_dir, "images", "*.png")))
    used_set = set(used); is_used = np.array([n in used_set for n in names])
    ntx, nty = map(int, args.tiles.split("x"))
    # sample grid: 16x12 for statistics, tile centres for the image fit; depth = median of LiDAR target per cell
    gu_, gv_ = np.meshgrid((np.arange(16) + .5) * W / 16, (np.arange(12) + .5) * H / 12)
    tu_, tv_ = np.meshgrid((np.arange(ntx) + .5) * W / ntx, (np.arange(nty) + .5) * H / nty)
    uv = np.r_[np.c_[gu_.ravel(), gv_.ravel()], np.c_[tu_.ravel(), tv_.ravel()]]

    def depth_cells(npy):
        d = np.load(npy).astype(np.float32) * scale
        r = np.full(len(uv), np.nan)
        for i, (u, v) in enumerate(uv):
            if i < 192:
                y0, x0 = int(v - H / 24), int(u - W / 32); blk = d[y0:y0 + H // 12, x0:x0 + W // 16]
            else:
                y0, x0 = int(v - H / (2 * nty)), int(u - W / (2 * ntx)); blk = d[y0:y0 + H // nty, x0:x0 + W // ntx]
            blk = blk[np.isfinite(blk) & (blk > 0)]
            if blk.size > 10:
                r[i] = np.median(blk)
        return r

    used_idx = np.where(is_used)[0]; dep_used = {}
    for i in used_idx:
        npy = os.path.join(used_dir, "images", names[i][:-4] + "_depth.npy")
        if os.path.exists(npy):
            dep_used[i] = depth_cells(npy)
    if dep_used:
        zmed = float(np.nanmedian(np.concatenate(list(dep_used.values()))))
        di = np.array(sorted(dep_used))
        near = di[np.abs(ts[:, None] - ts[di][None]).argmin(1)]
        rng = np.stack([dep_used[j] for j in near])
    else:
        zmed = args.z_default; rng = np.full((N, len(uv)), zmed)
    rng = np.where(np.isfinite(rng), rng, zmed)
    print(f"[blur] {N} frames, {len(used_idx)} used, {len(dep_used)} with LiDAR depth (median {zmed:.2f} m), "
          f"f={f:.1f}px, dt=+-{args.dt_ms}ms")

    flow = flow_rates(Twc, Tm, Tp, dt, K, uv, rng)                        # px/s
    sp = np.linalg.norm(flow, axis=2)
    rate_med = np.median(sp[:, :192], axis=1) / 1e3                         # px per ms of exposure
    rate_p90 = np.percentile(sp[:, :192], 90, axis=1) / 1e3
    # rotation-only rate and IMU cross-check
    Rrel = np.einsum("nji,njk->nik", Tm[:, :3, :3], Tp[:, :3, :3])
    ang = np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)) / (2 * dt)
    import gating_utils as gu
    w_imu = gu.omega_magnitude_at(cfg, ts)
    import sync_utils as su
    pre_end = su.detect_sync_preamble(cfg) if cfg.get("gating", {}).get("exclude_preamble", True) else None
    not_pre = ts > pre_end if pre_end is not None else np.ones(N, bool)

    texps = [float(x) for x in args.texp_ms.split(",")]
    res = {"n_frames": N, "n_used": int(is_used.sum()), "n_used_with_depth": len(dep_used),
           "focal_px": f, "median_depth_m": zmed, "dt_ms": args.dt_ms, "preamble_end_s": pre_end,
           "frame_period_ms": float(np.median(np.diff(ts)) * 1e3),
           "omega_mocap_rad_s": {"all": stats(ang), "used": stats(ang[is_used])},
           "omega_imu_rad_s": {"all": stats(w_imu), "used": stats(w_imu[is_used])},
           "omega_mocap_vs_imu_pearson": float(np.corrcoef(ang[not_pre], w_imu[not_pre])[0, 1]),
           "flow_rate_px_per_ms": {"all": stats(rate_med), "all_no_preamble": stats(rate_med[not_pre]),
                                   "used": stats(rate_med[is_used])},
           "blur_px_by_texp": {}}
    for te in texps:
        res["blur_px_by_texp"][f"{te:g}ms"] = {
            "median_over_image": {"all": stats(rate_med * te), "all_no_preamble": stats(rate_med[not_pre] * te),
                                  "used": stats(rate_med[is_used] * te)},
            "p90_over_image": {"all": stats(rate_p90 * te), "used": stats(rate_p90[is_used] * te)}}
    gm = os.path.join(out_root, "gating", "gating_meta.json")
    if os.path.exists(gm):
        g = json.load(open(gm))
        res["gate"] = {k: g.get(k) for k in ("n_frames_in", "n_gated", "n_excluded_preamble", "n_train", "n_val",
                                             "omega_threshold_rad_s", "discard_fraction")}
        res["gate"]["n_dropped"] = g["n_frames_in"] - g["n_gated"]

    if not args.no_images:
        root = cfg["sequence"]["data_root"]
        pre = os.path.join(out_root, "precond", cfg["sequence"]["rgb_dir"])
        src = pre if os.path.isdir(pre) else os.path.join(root, cfg["sequence"]["rgb_dir"])
        import cv2
        E = np.zeros((N, ntx * nty, len(DIRS) + 1)); lap = np.zeros(N); bn = max(1, args.bin)
        order = np.argsort(rate_med); sharp_ids = set(order[:40].tolist()); sharp_tiles = []
        th, tw = H // bn // nty, W // bn // ntx
        for i, n in enumerate(names):
            g = cv2.imread(os.path.join(src, n), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if bn > 1:
                g = cv2.resize(g, (W // bn, H // bn), interpolation=cv2.INTER_AREA)
            E[i], lap[i] = tile_energies(g, nty, ntx)
            if i in sharp_ids:
                sharp_tiles += [g[ty * th:ty * th + min(th, tw), tx * tw:tx * tw + min(th, tw)]
                                for ty in range(nty) for tx in range(ntx)]
            if i % 1000 == 0:
                print(f"[blur] images {i}/{N}", flush=True)
        Ls = np.arange(0, 24.01, 0.25); phis = np.deg2rad(np.arange(0, 180, 5))
        Htab = attenuation_table(sharp_tiles, Ls, phis)
        keys = list(DIRS) + ["lap"]
        tfl = flow[:, 192:] / 1e3 / bn                                           # px/ms at tile centres (N,T,2)
        tsp = np.linalg.norm(tfl, axis=2); tph = np.mod(np.arctan2(tfl[..., 1], tfl[..., 0]), np.pi)
        jph = np.round(tph / np.deg2rad(5)).astype(int) % len(phis)
        yobs = np.stack([np.log(E[..., keys.index(a)] / E[..., keys.index(b)]) for a, b in PAIRS], -1)
        ok = np.all(np.isfinite(yobs), axis=(1, 2)) & not_pre
        yo = running_median(yobs, args.detrend)[ok]

        ylo = np.log(E[..., -1:]); okl = np.all(np.isfinite(ylo), axis=(1, 2)) & not_pre
        ylo = running_median(ylo, args.detrend)[okl]

        def ypred(te, lapl=False):
            L = np.clip(tsp * te, 0, Ls[-1]); iL = L / 0.25
            i0 = np.floor(iL).astype(int); i1 = np.minimum(i0 + 1, len(Ls) - 1); w = iL - i0
            h = lambda k: (1 - w) * Htab[k, i0, jph] + w * Htab[k, i1, jph]
            if lapl:                                                         # magnitude: Laplacian energy drop
                return running_median(h(len(keys) - 1)[..., None], args.detrend)[okl]
            y = np.stack([h(keys.index(a)) - h(keys.index(b)) for a, b in PAIRS], -1)
            return running_median(y, args.detrend)[ok]

        tgrid = np.arange(0, 20.01, 0.25)
        preds = np.stack([ypred(t) for t in tgrid])
        sse = ((preds - yo[None]) ** 2).sum(axis=(1, 2, 3))
        tbest = float(tgrid[sse.argmin()])
        # block bootstrap over time (blocks of 60 frames ~ 1 s)
        rs = np.random.default_rng(0); nb = max(1, len(yo) // 60); boots = []
        blocks = np.array_split(np.arange(len(yo)), nb)
        r2 = (preds - yo[None]) ** 2
        per_block = np.stack([r2[:, b].sum(axis=(1, 2, 3)) for b in blocks], 1)   # (T, nb)
        for _ in range(500):
            boots.append(tgrid[per_block[:, rs.integers(0, nb, nb)].sum(1).argmin()])
        p = preds[sse.argmin()].ravel(); o = yo.ravel()
        predl = np.stack([ypred(t, True) for t in tgrid]); r2l = (predl - ylo[None]) ** 2
        ssel = r2l.sum(axis=(1, 2, 3)); tbl = float(tgrid[ssel.argmin()])
        pbl = np.stack([r2l[:, b].sum(axis=(1, 2, 3)) for b in np.array_split(np.arange(len(ylo)), nb)], 1)
        bootl = [tgrid[pbl[:, rs.integers(0, nb, nb)].sum(1).argmin()] for _ in range(500)]
        from scipy.stats import spearmanr
        rl = running_median(np.log(lap)[:, None], args.detrend)[:, 0]
        rr = running_median(np.log(np.maximum(rate_med, 1e-6))[:, None], args.detrend)[:, 0]
        res["image_evidence"] = {
            "source": src, "tiles": args.tiles, "detrend_half_window": args.detrend, "n_frames_fit": int(ok.sum()),
            "texp_fit_ms": tbest, "texp_ci90_ms": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "sse_ratio_best_vs_zero_blur": float(sse.min() / sse[0]),
            "pearson_pred_vs_obs_anisotropy_at_best": float(np.corrcoef(p, o)[0, 1]) if tbest > 0 else 0.0,
            "pearson_pred_vs_obs_anisotropy_at_10ms": float(np.corrcoef(preds[np.searchsorted(tgrid, 10)].ravel(), o)[0, 1]),
            "texp_fit_laplacian_ms": tbl, "texp_ci90_laplacian_ms": [float(np.percentile(bootl, 5)),
                                                                    float(np.percentile(bootl, 95))],
            "sse_ratio_laplacian_best_vs_zero_blur": float(ssel.min() / ssel[0]), "bin": bn,
            "spearman_loglap_vs_lograte_detrended": float(spearmanr(rl[ok], rr[ok]).correlation),
            "laplacian_var": {"all": stats(lap), "used": stats(lap[is_used])}}
        if tbest > 0:
            res["blur_px_at_fit_texp"] = {"used": stats(rate_med[is_used] * tbest),
                                          "all": stats(rate_med * tbest),
                                          "all_no_preamble": stats(rate_med[not_pre] * tbest),
                                          "used_p90_over_image": stats(rate_p90[is_used] * tbest)}
        np.savez(os.path.join(out, "image_fit.npz"), tgrid=tgrid, sse=sse, lap=lap, E=E)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
            ax[0].plot(tgrid, sse / sse[0]); ax[0].axvline(tbest, ls="--", c="r")
            ax[0].set_xlabel("t_exp (ms)"); ax[0].set_ylabel("SSE / SSE(no blur)"); ax[0].set_title("directional fit")
            s = slice(0, min(600, int(ok.sum())))
            ax[1].plot(o.reshape(yo.shape)[s, :, 0].mean(1), lw=.8, label="observed (0/90 anis.)")
            ax[1].plot(p.reshape(yo.shape)[s, :, 0].mean(1), lw=.8, label=f"predicted @ {tbest:g} ms")
            ax[1].legend(fontsize=7); ax[1].set_xlabel("frame"); ax[1].set_title("detrended anisotropy, tile mean")
            te = tbest if tbest > 0 else 10
            ax[2].hist(rate_med * te, bins=60, alpha=.6, label="all"); ax[2].hist(rate_med[is_used] * te, bins=60, alpha=.8, label="used")
            ax[2].set_xlabel(f"blur px @ {te:g} ms (median over image)"); ax[2].legend(); ax[2].set_title("predicted blur")
            fig.tight_layout(); fig.savefig(os.path.join(out, "blur_diagnosis.png"), dpi=110); plt.close(fig)
        except Exception as e:                                              # plotting is optional
            print("[blur] plot failed:", e)

    with open(os.path.join(out, "per_frame.csv"), "w") as fh:
        fh.write("name,t,used,rate_px_per_ms_med,rate_px_per_ms_p90,omega_mocap,omega_imu\n")
        for i in range(N):
            fh.write(f"{names[i]},{ts[i]:.6f},{int(is_used[i])},{rate_med[i]:.5f},{rate_p90[i]:.5f},{ang[i]:.4f},{w_imu[i]:.4f}\n")
    pu.save_json(os.path.join(out, "blur_diagnosis.json"), res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
