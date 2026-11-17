"""Diagnose a pseudo-view training run (gsplat venv, GPU, minutes): did the pseudo-view content get into the splat?
  - masked PSNR of each --ply at every pseudo-view of --pseudo (target = the view's image, mask = its _mask.png > 0.05),
    both sort modes; contact sheet of --n views: target | mask | each PLY (radial sort)
  - gaussians near each object's UNSEEN mesh surface (<= --near-m of a vertex no training camera saw) per PLY
  - scale / opacity statistics per PLY (large gaussians: max scale > --big-m)
    python scripts/object_layers/diag_pseudo.py --pseudo PDS --layers T --sim3-json S --ply name=path ... --out DIR
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo", required=True); ap.add_argument("--layers", required=True); ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--ply", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--near-m", type=float, default=0.02); ap.add_argument("--big-m", type=float, default=0.1)
    a = ap.parse_args()
    import cv2, torch
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from finetune_ply_viewer import ViewerRaster, cam_args, frames, load_ply, sh_deg, to_t
    from object_layers import final_meshes, sim3
    os.makedirs(a.out, exist_ok=True); dev = "cuda"; s, R, t = sim3(a.sim3_json)
    fr = frames(a.pseudo, 0, "all"); fr = [f for f in fr if f[0].startswith("msh_")]
    pick = set(f[0] for f in [fr[i] for i in np.linspace(0, len(fr) - 1, a.n).astype(int)])
    tgt = {n_: cv2.imread(os.path.join(a.pseudo, "images", n_))[..., ::-1].astype(np.float32) / 255 for n_, _, _ in fr}
    msk = {n_: cv2.imread(os.path.join(a.pseudo, "images", n_[:-4] + "_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for n_, _, _ in fr}
    ms = final_meshes(a.layers); un = [((m["V"][~m["seen"]] - t) @ R) / s for m in ms]; rep = {"views": len(fr), "plys": {}}; sheets = {}
    for spec in a.ply:
        nm, path = spec.split("=", 1); _, p = load_ply(path, dev); deg = sh_deg(p); r = rep["plys"][nm] = {}
        X = p["means"].cpu().numpy().astype(np.float64); sc = np.exp(p["scales"].cpu().numpy()).max(1) * s; op = 1 / (1 + np.exp(-p["opacities"].cpu().numpy()))
        r["n"] = int(len(X)); r["scale_max_m_q"] = {q: round(float(np.quantile(sc, q)), 4) for q in (0.5, 0.9, 0.99, 0.999)}
        r["big"] = int((sc > a.big_m).sum()); r["big_opaque"] = int(((sc > a.big_m) & (op > 0.5)).sum()); r["opacity_q"] = {q: round(float(np.quantile(op, q)), 3) for q in (0.1, 0.5, 0.9)}
        tree = cKDTree(X); r["near_unseen"] = {str(m["id"]): int(len(set(i for l in tree.query_ball_point(U, a.near_m / s) for i in l))) for m, U in zip(ms, un)}
        for mode in ("radial", "depth"):
            Rv = ViewerRaster("ss352", "exact", display_scale=1.8, sort=mode) if "sort" in ViewerRaster.__init__.__code__.co_varnames else ViewerRaster("ss352", "exact", display_scale=1.8)
            ps = []
            with torch.no_grad():
                for n_, cam, c2w in fr:
                    img = Rv.render(p, to_t(np.linalg.inv(c2w), dev), *cam_args(cam), sh_degree=deg)[0].clamp(0, 1).cpu().numpy()
                    w = msk[n_] > 0.05
                    if w.sum() > 50:
                        ps.append(float(-10 * np.log10(max(((img[w] - tgt[n_][w]) ** 2).mean(), 1e-10))))
                    if mode == "radial" and n_ in pick:
                        sheets.setdefault(n_, {})[nm] = img
            r[f"masked_psnr_{mode}"] = round(float(np.mean(ps)), 2) if ps else None
            if "sort" not in ViewerRaster.__init__.__code__.co_varnames:
                break
        print(f"[diag] {nm}: " + json.dumps(r), flush=True)
    rows = []
    lab = lambda im, tx: cv2.putText(np.ascontiguousarray((np.clip(im, 0, 1) * 255).astype(np.uint8)), tx, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for n_ in sorted(sheets):
        tiles = [lab(cv2.resize(tgt[n_], (320, 240)), "target " + n_[:-4]), lab(cv2.resize(np.dstack([msk[n_]] * 3), (320, 240)), "mask")]
        tiles += [lab(cv2.resize(sheets[n_][k], (320, 240)), k) for k in sheets[n_]]
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(a.out, "contact.jpg"), np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(rep, open(os.path.join(a.out, "diag.json"), "w"), indent=1); print("DIAG_DONE " + a.out, flush=True)


if __name__ == "__main__":
    main()
