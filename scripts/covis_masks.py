"""Co-visibility masks for held-out views (3DGRUT venv; GPU if present).

Splits each held-out view into pixels some training camera actually observed and pixels none did,
so a method's effect on observed geometry is not confused with what it does to never-seen content.
A held-out pixel is OBSERVED if its LiDAR surface point (from <image>_depth.npy) projects inside a
training camera and agrees, within --tol-rel, with that camera's own LiDAR depth at that pixel
(i.e. the point is not occluded there). Pixels without a LiDAR target are left unclassified.

Writes <test images>/<stem>_covis.png: 255 observed, 128 not observed, 0 unknown.
Needs build_depth_targets.py run on both datasets first.

    python scripts/covis_masks.py --train <offpath>/train_data --test <offpath>/test
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model


def cams_of(ds):
    cams, imgs = read_text_model(os.path.join(ds, "sparse", "0"))
    out = []
    for name, cid, q, tcw in sorted(imgs):
        W, H, (fx, fy, cx, cy) = cams[cid]
        dep = os.path.splitext(os.path.join(ds, "images", name))[0] + "_depth.npy"
        out.append((name, W, H, fx, fy, cx, cy, quat_to_R(q), tcw, dep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tol-rel", type=float, default=0.05)
    args = ap.parse_args()
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda a: torch.tensor(np.asarray(a, dtype=np.float64), device=dev)

    train = [(c, T(np.load(c[9]).astype(np.float32))) for c in cams_of(args.train)]
    fr = []
    for name, W, H, fx, fy, cx, cy, Rcw, tcw, dep in cams_of(args.test):
        D = T(np.load(dep).astype(np.float32))
        v, u = torch.nonzero(D > 0, as_tuple=True)
        r = torch.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(u, dtype=torch.float64)], 1)
        r = r / torch.linalg.norm(r, dim=1, keepdim=True)
        Rcw_t, tcw_t = T(Rcw), T(tcw)
        X = (r * D[v, u][:, None] - tcw_t) @ Rcw_t                            # cam -> world: R^T (x - t)
        seen = torch.zeros(len(X), dtype=torch.bool, device=dev)
        for (_, W2, H2, fx2, fy2, cx2, cy2, R2, t2, _), D2 in train:
            Pc = X @ T(R2).T + T(t2)
            z = Pc[:, 2]
            uu = torch.floor(fx2 * Pc[:, 0] / z + cx2).long(); vv = torch.floor(fy2 * Pc[:, 1] / z + cy2).long()
            ok = (z > 0.05) & (uu >= 0) & (uu < W2) & (vv >= 0) & (vv < H2)
            idx = torch.nonzero(ok, as_tuple=True)[0]
            dt = D2[vv[idx], uu[idx]]
            rng = torch.linalg.norm(Pc[idx], dim=1)
            seen[idx[(dt > 0) & (torch.abs(rng - dt) < args.tol_rel * dt)]] = True
        M = np.zeros((H, W), np.uint8)
        M[v.cpu().numpy(), u.cpu().numpy()] = np.where(seen.cpu().numpy(), 255, 128)
        Image.fromarray(M).save(os.path.splitext(os.path.join(args.test, "images", name))[0] + "_covis.png")
        fr.append(float(seen.float().mean()) if len(seen) else 0.0)
    info = {"frames": len(fr), "observed_fraction_of_lidar_pixels": {"median": round(float(np.median(fr)), 3),
                                                                    "mean": round(float(np.mean(fr)), 3)}}
    json.dump(info, open(os.path.join(args.test, "covis.json"), "w"), indent=2)
    print(f"[covis] {len(fr)} held-out frames; LiDAR pixels observed by a training camera: "
          f"median {info['observed_fraction_of_lidar_pixels']['median']:.1%}", flush=True)


if __name__ == "__main__":
    main()
