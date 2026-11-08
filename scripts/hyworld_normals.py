"""Per-frame normal priors for the HY-World 2.0 depth-to-normal loss (scripts/methods/hyworld.py --hy-normal).
HY-World 2.0 (arXiv 2604.14268) supervises its splat's depth-derived normals with predicted normals. Two sources:
  worldmirror  WorldMirror 2.0 (the paper's feed-forward reconstructor), run over chunks of consecutive training frames
               WITH our priors: camera-to-world poses + intrinsics from the COLMAP model and the LiDAR depth target
               (<image>_depth.npy, ray distance -> z) as its depth prior (cond_flags [1, 1, 1]); every frame keeps the
               prediction of the chunk it is most central in.
  moge2        MoGe-2 (Ruicheng/moge-2-vitl-normal), per frame -- what the official 3DGS stage
               (hyworld2/worldgen/gen_gs_data.py) feeds world_gs_trainer.py's normal loss.
Output: <out>/<image stem>.npy, float16 (H,W,3) unit normals at the dataset resolution in the OpenCV camera frame
(x right, y down, z forward), facing the camera, zeros where invalid (confidence below --conf-pct percentile, outside
the model's crop). The model's axis convention is not assumed: of the 8 axis-sign flips (and world->camera rotation
for WorldMirror) the one agreeing best with normals of the LiDAR depth targets is used, and the median angle to them
is logged (<out>/_normals.json) -- also an honest quality check of the prior. Only training frames (every
--test-split-interval-th frame held out) are used as model input. Run in a GPU job (sbatch/hyworld_normals.sbatch):
  python scripts/hyworld_normals.py --dataset $P/edgs/on/data --out $P/hyworld/normals --source worldmirror
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_poses_gsplat import read_model, intrinsics  # noqa: E402


def depth_normals(z, K, s=1, smooth=0):
    """z (H,W) z-depth (0 = none) -> camera-frame normals (H,W,3), valid (H,W); central differences over +-s px of the
    (optionally smooth x smooth box-filtered, valid-weighted) depth, camera-facing. The LiDAR targets are float16 point
    renders: per-pixel differences are mostly quantisation noise, so the reference check uses s=4, smooth=7."""
    H, W = z.shape
    ok = z > 0
    if smooth > 1:
        import cv2
        m = ok.astype(np.float32); zs = cv2.blur(z.astype(np.float32) * m, (smooth, smooth)) / np.maximum(cv2.blur(m, (smooth, smooth)), 1e-6)
        z = np.where(ok, zs, 0)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    dx = np.zeros_like(X); dy = np.zeros_like(X)
    dx[:, s:-s] = X[:, 2 * s:] - X[:, :-2 * s]; dy[s:-s] = X[2 * s:] - X[:-2 * s]
    n = np.cross(dx, dy); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    ok = ok & np.roll(ok, s, 0) & np.roll(ok, -s, 0) & np.roll(ok, s, 1) & np.roll(ok, -s, 1) & (nn[..., 0] > 0)
    zz = np.where(ok, z, np.nan)                     # depth edges: relative jump > 5 % per px -> no normal
    for ax in (0, 1):
        ok &= ~(np.abs(np.roll(zz, s, ax) - np.roll(zz, -s, ax)) > 0.05 * s * zz)
    ok[:s] = ok[-s:] = False; ok[:, :s] = ok[:, -s:] = False
    n = np.where((n * X).sum(-1, keepdims=True) > 0, -n, n)
    return n, ok


def face(n, K):
    H, W = n.shape[:2]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return np.where((n * ray).sum(-1, keepdims=True) > 0, -n, n)


def angle_deg(a, b, m):
    return np.degrees(np.arccos(np.clip((a[m] * b[m]).sum(-1), -1, 1)))


def lidar_ref(ds, name, K, W, H):
    f = os.path.join(ds, "images", os.path.splitext(name)[0] + "_depth.npy")
    if not os.path.exists(f):
        return None
    r = np.load(f).astype(np.float32)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rn = np.sqrt(((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2 + 1)
    return r / rn


def back_to_full(x, pp, W, H):
    """model-resolution map (h,w,C) on the resized + centre-cropped grid -> (H,W,C) at the dataset resolution (nearest)."""
    import cv2
    can = np.zeros((pp["new_h"], pp["new_w"], x.shape[-1]), np.float32)
    can[pp["crop_y"]: pp["crop_y"] + x.shape[0], pp["crop_x"]: pp["crop_x"] + x.shape[1]] = x
    return cv2.resize(can, (W, H), interpolation=cv2.INTER_NEAREST).reshape(H, W, -1)


def run_worldmirror(a, frames, ds, dev):
    import torch
    sys.path.insert(0, a.hyworld_dir)
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import prepare_images_to_tensor, compute_preprocessing_transform
    pipe = WorldMirrorPipeline.from_pretrained("tencent/HY-World-2.0", subfolder="HY-WorldMirror-2.0",
                                               enable_bf16=True, disable_heads=["gs", "points"])
    C, ov = a.chunk, a.overlap; st = max(C - ov, 1)
    starts = list(range(0, max(len(frames) - C, 0) + 1, st))
    if starts[-1] + C < len(frames):
        starts.append(len(frames) - C)
    best = {}                                                       # frame index -> (centrality, normals, conf)
    for s0 in starts:
        idx = list(range(s0, min(s0 + C, len(frames))))
        paths = [frames[i]["path"] for i in idx]
        pp = compute_preprocessing_transform(paths, a.target_size)
        imgs = prepare_images_to_tensor(paths, target_size=a.target_size, resize_strategy="crop").to(dev)
        h, w = imgs.shape[-2:]
        c2w = np.stack([frames[i]["c2w"] for i in idx]).astype(np.float32)
        c2w = np.linalg.inv(c2w[0])[None] @ c2w                     # relative to the chunk's first frame (as the pipeline)
        Ks = []
        for i in idx:
            K = frames[i]["K"].copy(); K[0] *= pp["scale_x"]; K[1] *= pp["scale_y"]; K[0, 2] -= pp["crop_x"]; K[1, 2] -= pp["crop_y"]
            Ks.append(K)
        views = {"img": imgs, "camera_poses": torch.from_numpy(c2w)[None].to(dev),
                 "camera_intrs": torch.from_numpy(np.stack(Ks).astype(np.float32))[None].to(dev)}
        flags = [1, 0, 1]
        if not a.no_depth_prior and all(frames[i]["z"] is not None for i in idx):
            import cv2
            zs = []
            for i in idx:
                z = cv2.resize(frames[i]["z"], (pp["new_w"], pp["new_h"]), interpolation=cv2.INTER_NEAREST)
                zs.append(z[pp["crop_y"]: pp["crop_y"] + h, pp["crop_x"]: pp["crop_x"] + w])
            views["depthmap"] = torch.from_numpy(np.stack(zs).astype(np.float32))[None].to(dev); flags[1] = 1
        with torch.no_grad():
            pr = pipe.model(views=views, cond_flags=flags, is_inference=True)
        N = pr["normals"][0].float().cpu().numpy()                  # (S,h,w,3)
        Cf = pr.get("normals_conf")
        Cf = Cf[0].float().cpu().numpy().reshape(len(idx), h, w) if Cf is not None else np.ones(N.shape[:3], np.float32)
        for j, i in enumerate(idx):
            cen = -abs(j - (len(idx) - 1) / 2)
            if i not in best or cen > best[i][0]:
                best[i] = (cen, N[j], Cf[j], pp, c2w[j])
        print(f"[hyn] worldmirror chunk {s0}..{idx[-1]} ({len(idx)} frames, {w}x{h}, flags {flags})", flush=True)
    out = {}
    for i, (_, n, cf, pp, c2w_rel) in best.items():
        W, H = frames[i]["W"], frames[i]["H"]
        out[i] = {"n": back_to_full(n, pp, W, H), "conf": back_to_full(cf[..., None], pp, W, H)[..., 0],
                  "R_w2c": np.linalg.inv(c2w_rel)[:3, :3]}         # in case the model returns world-frame normals
    return out


def run_moge2(a, frames, dev):
    import torch
    from moge.model.v2 import MoGeModel
    m = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    from PIL import Image
    out = {}
    for i, f in enumerate(frames):
        x = torch.from_numpy(np.asarray(Image.open(f["path"]).convert("RGB"), np.float32) / 255).permute(2, 0, 1).to(dev)
        fov_x = float(np.degrees(2 * np.arctan(f["W"] / (2 * f["K"][0, 0]))))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = m.infer(x, fov_x=fov_x)
        n = p["normal"].float().cpu().numpy()
        msk = p["mask"].cpu().numpy().astype(np.float32) if "mask" in p else np.ones(n.shape[:2], np.float32)
        out[i] = {"n": n, "conf": msk, "R_w2c": np.eye(3)}
        if i % 50 == 0:
            print(f"[hyn] moge2 {i}/{len(frames)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=("worldmirror", "moge2"), default="worldmirror")
    ap.add_argument("--test-split-interval", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--target-size", type=int, default=518)
    ap.add_argument("--no-depth-prior", action="store_true")
    ap.add_argument("--conf-pct", type=float, default=10.0, help="zero the least-confident x %% of each frame (WorldMirror)")
    ap.add_argument("--hyworld-dir", default=os.environ.get("HYWORLD_DIR", ""))
    a = ap.parse_args()
    import torch
    dev = "cuda"; t0 = time.time()
    cams, imgs = read_model(os.path.join(a.dataset, "sparse", "0"))
    frames = []
    for k, (name, _iid, cid, c2w) in enumerate(imgs):
        if a.test_split_interval > 0 and k % a.test_split_interval == 0:
            continue
        K, W, H = intrinsics(cams[cid]); K = np.asarray(K, np.float64)
        frames.append({"name": name, "path": os.path.join(a.dataset, "images", name), "K": K, "W": int(W), "H": int(H),
                       "c2w": c2w, "z": lidar_ref(a.dataset, name, K, int(W), int(H))})
    print(f"[hyn] {len(frames)} training frames, source {a.source}", flush=True)
    res = run_worldmirror(a, frames, a.dataset, dev) if a.source == "worldmirror" else run_moge2(a, frames, dev)
    # axis convention: pick the sign flip (x world->camera rotation) that best matches the LiDAR depth normals
    cand = [(s, rot) for s in itertools.product((1, -1), repeat=3) for rot in (False, True)]
    score = {c: [] for c in cand}
    probe = [i for i in sorted(res)[:: max(len(res) // 24, 1)] if frames[i]["z"] is not None]
    refs = {}
    for i in probe:
        rn, ok = depth_normals(frames[i]["z"], frames[i]["K"], 4, 7); refs[i] = (rn, ok)
        for s, rot in cand:
            n = res[i]["n"] @ res[i]["R_w2c"].T if rot else res[i]["n"]
            n = face(n * np.array(s), frames[i]["K"])
            m = ok & (np.linalg.norm(res[i]["n"], axis=-1) > 0.5)
            if m.any():
                score[(s, rot)].append(np.median(angle_deg(n, rn, m)))
    if probe:
        (sg, rot), med = min(((c, float(np.mean(v))) for c, v in score.items() if v), key=lambda x: x[1])
    else:
        (sg, rot), med = ((1, 1, 1), False), None
    print(f"[hyn] convention: axis signs {sg}, world->camera {rot}; median angle to LiDAR normals {med} deg", flush=True)
    os.makedirs(a.out, exist_ok=True); errs, cover = [], []
    for i, r in res.items():
        n = r["n"] @ r["R_w2c"].T if rot else r["n"]
        n = n * np.array(sg); nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = face(n / np.maximum(nn, 1e-8), frames[i]["K"])
        bad = nn[..., 0] < 0.5
        if a.source == "worldmirror" and a.conf_pct > 0:
            cf = r["conf"]; bad |= cf < np.percentile(cf[~bad], a.conf_pct) if (~bad).any() else bad
        else:
            bad |= r["conf"] < 0.5
        n[bad] = 0; cover.append(float(1 - bad.mean()))
        np.save(os.path.join(a.out, os.path.splitext(frames[i]["name"])[0] + ".npy"), n.astype(np.float16))
        if i in refs:
            m = refs[i][1] & ~bad
            if m.any():
                errs.append(float(np.median(angle_deg(n, refs[i][0], m))))
    rec = {"source": a.source, "frames": len(res), "axis_signs": list(sg), "world_to_camera": rot, "convention_score_deg": med,
           "median_angle_to_lidar_deg": round(float(np.median(errs)), 2) if errs else None,
           "coverage": round(float(np.mean(cover)), 4), "depth_prior": not a.no_depth_prior, "chunk": a.chunk,
           "target_size": a.target_size, "seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    json.dump(rec, open(os.path.join(a.out, "_normals.json"), "w"), indent=1)
    if refs:                                                        # one side-by-side for looking at: rgb | prior | LiDAR
        from PIL import Image
        i = sorted(refs)[len(refs) // 2]
        n = np.load(os.path.join(a.out, os.path.splitext(frames[i]["name"])[0] + ".npy")).astype(np.float32)
        rgb = np.asarray(Image.open(frames[i]["path"]).convert("RGB"))
        vis = lambda x: ((x * [1, -1, -1] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        ln = refs[i][0] * refs[i][1][..., None]
        Image.fromarray(np.concatenate([rgb, vis(n), vis(ln)], 1)).save(os.path.join(a.out, "_preview.png"))
    print(f"[hyn] {json.dumps(rec)}", flush=True); print("HYWORLD_NORMALS_DONE", flush=True)


if __name__ == "__main__":
    main()
