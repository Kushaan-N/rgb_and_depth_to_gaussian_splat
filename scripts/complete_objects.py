"""Complete the unseen surfaces of free-standing objects in a 3DGRUT splat — measured geometry, copied
appearance, no generative model (3DGRUT venv; GPU if present).

For each object from find_objects.py (metric LiDAR points of blocks, boxes, ramps standing on the floor):
  1. Shape: a 2.5D surface over the object's footprint (heights from its LiDAR points on a --hcell grid,
     axes aligned to the object by PCA): flat tops, vertical walls at the footprint edge and at steps.
     Stepped brick stacks, boxes and ramps are all 2.5D.
  2. Surface samples every --cell, each with a normal — kept only where the object's LiDAR returns are
     within --support-m (measured surface) and no camera sees through them (free-space carving).
  3. Observed?: a sample is observed if some training camera sees it — inside the image, facing the
     camera (< --max-incidence), and not occluded (its ray distance matches that frame's LiDAR depth
     target within tolerance; where the frame has no LiDAR target it counts as observed if it is in
     view and facing, so the real splat keeps authority whenever in doubt).
  4. Appearance for unseen samples, copied from observed ones, in order:
       mirror  — the point mirrored across the object's centre on the opposite, parallel face
       row     — the nearest observed sample at the same height on a face with the same orientation
       nearest — the nearest observed sample of the same kind (wall/top)
     The copied value is the SH colour of the real gaussian nearest the source sample, so colour
     encoding and exposure match the splat exactly.
  5. Splat edit: gaussians inside the object's volume that no training camera observed are removed
     (junk in never-seen space); flat, opaque, matte gaussians are added on the unseen samples only.
     Observed surfaces keep their real gaussians.

Writes a new checkpoint (render/export it like any other), <out>_report.json, and
<out>_unseen.npz (unseen sample positions + normals, dataset frame) for cross-recording evaluation.

    python scripts/complete_objects.py --objects objects --checkpoint ckpt.pt --dataset <all frames with *_depth.npy> \
        --sim3-json <pipeline>/sim3.json --out completed.pt
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covis_masks import cams_of


def object_surface(P, n, d, hcell, cell, min_h=0.01):
    """2.5D surface samples of one object (metric): positions, normals, kind (0 top / 1 wall), face key."""
    from scipy import ndimage
    h = P @ n + d
    base = P - np.outer(h, n)                                          # foot points on the floor plane
    c0 = base.mean(0)
    e = np.linalg.svd((base - c0) - np.outer((base - c0) @ n, n), full_matrices=False)[2]
    a1 = e[0] - (e[0] @ n) * n; a1 /= np.linalg.norm(a1); a2 = np.cross(n, a1)
    uv = np.stack([(base - c0) @ a1, (base - c0) @ a2], 1)
    lo = uv.min(0) - hcell; sh = np.ceil((uv.max(0) + hcell - lo) / hcell).astype(int) + 1
    ij = np.floor((uv - lo) / hcell).astype(int)
    H = np.zeros(sh); cnt = np.zeros(sh)
    keep = h > min_h
    order = np.argsort(h[keep])
    H[ij[keep][order, 0], ij[keep][order, 1]] = h[keep][order]       # max height per cell
    cnt[ij[keep, 0], ij[keep, 1]] += 1
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(cnt > 0, iterations=1))
    if (mask & (cnt == 0)).any():                                     # fill empty cells from nearest measured
        idx = ndimage.distance_transform_edt(cnt == 0, return_distances=False, return_indices=True)
        H = np.where(cnt > 0, H, H[idx[0], idx[1]])
    H = np.where(mask, ndimage.median_filter(H, 3), 0)
    k = max(1, int(round(hcell / cell))); sub = (np.arange(k) + 0.5) / k
    pos, nor, kind, face = [], [], [], []
    to3 = lambda u, v, hh: c0 + u * a1 + v * a2 + hh * n
    for i, j in zip(*np.nonzero(mask)):
        u0, v0 = lo + np.array([i, j]) * hcell
        for su in sub:
            for sv in sub:                                             # top
                pos.append(to3(u0 + su * hcell, v0 + sv * hcell, H[i, j])); nor.append(n); kind.append(0); face.append(-1)
        for f, (di, dj, vec) in enumerate(((1, 0, a1), (-1, 0, -a1), (0, 1, a2), (0, -1, -a2))):
            ni, nj = i + di, j + dj
            inside = 0 <= ni < sh[0] and 0 <= nj < sh[1] and mask[ni, nj]
            h2 = H[ni, nj] if inside else 0.0
            if H[i, j] - h2 < hcell:
                continue
            for s in sub:                                              # wall on the shared edge, from h2 up
                eu = u0 + (1.0 if di == 1 else 0.0 if di == -1 else s) * hcell
                ev = v0 + (1.0 if dj == 1 else 0.0 if dj == -1 else s) * hcell
                for hh in np.arange(h2 + cell / 2, H[i, j], cell):
                    pos.append(to3(eu, ev, hh)); nor.append(vec); kind.append(1); face.append(f)
    pos = np.array(pos); local = np.stack([(pos - c0) @ a1, (pos - c0) @ a2, pos @ n + d], 1)
    centre = np.array([(local[:, 0].min() + local[:, 0].max()) / 2, (local[:, 1].min() + local[:, 1].max()) / 2])
    return pos, np.array(nor), np.array(kind), np.array(face), local, centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True, help="find_objects.py output prefix")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="training frames (COLMAP text) with <image>_depth.npy")
    ap.add_argument("--sim3-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.01, help="surface sample spacing (m)")
    ap.add_argument("--hcell", type=float, default=0.02, help="height-map cell (m)")
    ap.add_argument("--max-incidence", type=float, default=85.0)   # tops are only ever seen at grazing angles
    ap.add_argument("--depth-tol-m", type=float, default=0.03)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--prune-margin-m", type=float, default=0.04)
    ap.add_argument("--support-m", type=float, default=0.03, help="surface samples need a LiDAR return this close")
    args = ap.parse_args()
    import torch
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(args.objects + ".json")); pts = np.load(args.objects + ".npz")
    n = np.array(meta["floor"]["n"]); d = float(meta["floor"]["d"])
    j = json.load(open(args.sim3_json)); s = float(j["scale"]); R = np.array(j["R"]); t = np.array(j["t"])
    to_ds = lambda X: ((X - t) @ R) / s                               # metric -> dataset (COLMAP) frame
    cams = cams_of(args.dataset)[::args.frame_stride]
    depth = [torch.tensor(np.load(c[9]).astype(np.float32), device=dev) if os.path.exists(c[9]) else None for c in cams]
    cosmax = np.cos(np.deg2rad(args.max_incidence)); tol = args.depth_tol_m / s
    T = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float64)

    def observed(Xd, Nd=None, through_out=None):
        """per point: seen by some camera (in view, facing if normals given, not occluded). With through_out,
        also flags points some camera sees THROUGH (its LiDAR depth lies clearly beyond the point: free space)."""
        X = T(Xd); seen = torch.zeros(len(X), dtype=torch.bool, device=dev)
        thru = torch.zeros(len(X), dtype=torch.bool, device=dev)
        Nn = T(Nd) if Nd is not None else None
        for (name, W, H, fx, fy, cx, cy, Rcw, tcw, _), D in zip(cams, depth):
            Rc = T(Rcw); Pc = X @ Rc.T + T(tcw); z = Pc[:, 2]
            u = torch.floor(fx * Pc[:, 0] / z + cx).long(); v = torch.floor(fy * Pc[:, 1] / z + cy).long()
            ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if through_out is not None and D is not None:
                ia = torch.nonzero(ok, as_tuple=True)[0]
                if len(ia):
                    dt = D[v[ia], u[ia]].double(); r = torch.linalg.norm(Pc[ia], dim=1)
                    thru[ia[(dt > 0) & (dt - r > torch.clamp(0.05 * dt, min=2 * tol))]] = True
            if Nn is not None:
                Cw = -Rc.T @ T(tcw); vd = Cw[None] - X; vd = vd / torch.linalg.norm(vd, dim=1, keepdim=True)
                ok &= (vd * Nn).sum(1) > cosmax
            idx = torch.nonzero(ok, as_tuple=True)[0]
            if D is not None and len(idx):
                dt = D[v[idx], u[idx]].double(); r = torch.linalg.norm(Pc[idx], dim=1)
                vis = (dt <= 0) | ((r - dt).abs() < torch.clamp(0.03 * dt, min=tol))
                idx = idx[vis]
            seen[idx] = True
        if through_out is not None:
            through_out.append(thru.cpu().numpy())
        return seen.cpu().numpy()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    G = ck["positions"].detach().double().numpy(); N0 = len(G)
    opac = torch.sigmoid(ck["density"].detach()).numpy().ravel()
    gtree = cKDTree(G); alb = ck["features_albedo"].detach().numpy()
    remove = np.zeros(N0, bool); new_pos, new_nor, new_alb, rep, unseen_all = [], [], [], [], []
    for o in meta["objects"]:
        P = pts[f"obj{o['id']}"]
        pos, nor, kind, face, local, centre = object_surface(P, n, d, args.hcell, args.cell)
        # keep only MEASURED surface: within --support-m of this object's LiDAR returns (the LiDAR scans 360
        # degrees, so real backs have returns; gaps between blocks and the space under a ramp do not)
        sup = cKDTree(P).query(pos)[0] < args.support_m
        pos, nor, kind, face, local = pos[sup], nor[sup], kind[sup], face[sup], local[sup]
        posd, nord = to_ds(pos), nor @ R                               # rotate normals into the dataset frame
        th = []
        seen = observed(posd, nord, through_out=th)
        free = th[0]                                                   # a camera sees through it: not a surface
        pos, nor, kind, face, local, posd, nord, seen = (a[~free] for a in (pos, nor, kind, face, local, posd, nord, seen))
        # appearance of observed samples = SH colour of the nearest opaque real gaussian
        dist, gi = gtree.query(posd, k=8)
        src = np.full(len(pos), -1)
        for c in range(8):
            ok = (src < 0) & (dist[:, c] < 0.015 / s) & (opac[gi[:, c]] > 0.5)
            src[ok] = gi[ok, c]
        have = seen & (src >= 0)
        fill = np.full(len(pos), -1); how = np.zeros(len(pos), int)          # 1 mirror, 2 row, 3 nearest
        un = np.nonzero(~seen)[0]
        if have.any() and len(un):
            hv = np.nonzero(have)[0]
            for f_un, f_op in ((0, 1), (1, 0), (2, 3), (3, 2)):           # mirror onto the opposite face
                a = un[(kind[un] == 1) & (face[un] == f_un)]; b = hv[(kind[hv] == 1) & (face[hv] == f_op)]
                if len(a) and len(b):
                    m = local[a].copy(); ax = 0 if f_un in (0, 1) else 1; m[:, ax] = 2 * centre[ax] - m[:, ax]
                    dd, ii = cKDTree(local[b]).query(m)
                    ok = dd < 0.03; fill[a[ok]] = b[ii[ok]]; how[a[ok]] = 1
            for f in range(4):                                           # same height, same orientation
                a = un[(kind[un] == 1) & (face[un] == f) & (fill[un] < 0)]; b = hv[(kind[hv] == 1) & (face[hv] == f)]
                if len(a) and len(b):
                    dd, ii = cKDTree(np.c_[local[b][:, :2] * 0.05, local[b][:, 2]]).query(np.c_[local[a][:, :2] * 0.05, local[a][:, 2]])
                    ok = np.abs(local[b][ii, 2] - local[a][:, 2]) < args.cell; fill[a[ok]] = b[ii[ok]]; how[a[ok]] = 2
            for kd in (0, 1):                                            # nearest observed of the same kind
                a = un[(kind[un] == kd) & (fill[un] < 0)]; b = hv[kind[hv] == kd]
                if not len(b):
                    b = hv
                if len(a) and len(b):
                    _, ii = cKDTree(local[b]).query(local[a]); fill[a] = b[ii]; how[a] = 3
        add = un[fill[un] >= 0]
        new_pos.append(posd[add]); new_nor.append(nord[add]); new_alb.append(alb[src[fill[add]]])
        unseen_all.append(np.c_[posd[un], nord[un]])
        # junk in never-seen space inside the object's volume: unobserved real gaussians there
        hi = local.max(0)
        Gm = G @ (s * R).T + t                                             # gaussians in metres
        hG = Gm @ n + d
        inside = np.nonzero((hG > 0.01) & (hG < hi[2] + args.prune_margin_m) &
                            (np.linalg.norm(Gm - np.array(o["centre_m"]), axis=1) < 0.5 * np.hypot(*o["footprint_m"]) + args.prune_margin_m))[0]
        if len(inside):
            gseen = observed(G[inside])
            remove[inside[~gseen]] = True
        rep.append({"id": o["id"], "samples": int(len(pos)), "dropped_unsupported": int((~sup).sum()), "dropped_free_space": int(free.sum()), "unseen": int(len(un)), "added": int(len(add)),
                    "fill": {"mirror": int((how == 1).sum()), "row": int((how == 2).sum()), "nearest": int((how == 3).sum())},
                    "unseen_fraction": round(len(un) / max(len(pos), 1), 3)})
        print(f"[complete] object {o['id']}: {len(pos)} surface samples, {len(un)} unseen ({len(un)/max(len(pos),1):.0%}); "
              f"filled mirror {rep[-1]['fill']['mirror']} / row {rep[-1]['fill']['row']} / nearest {rep[-1]['fill']['nearest']}", flush=True)
    NP = np.concatenate(new_pos) if new_pos else np.zeros((0, 3)); NN = np.concatenate(new_nor) if new_nor else np.zeros((0, 3))
    NA = np.concatenate(new_alb) if new_alb else np.zeros((0, alb.shape[1]))
    # new flat gaussians: tangent frame from the normal, in-plane sigma ~0.6 cell, 1 mm thick, opaque, matte
    tan = np.cross(NN, np.array([0.0, 0.0, 1.0])); bad = np.linalg.norm(tan, axis=1) < 1e-3
    tan[bad] = np.cross(NN[bad], np.array([1.0, 0.0, 0.0])); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    bit = np.cross(NN, tan)
    q = Rotation.from_matrix(np.stack([tan, bit, NN], 2)).as_quat()[:, [3, 0, 1, 2]] if len(NP) else np.zeros((0, 4))   # wxyz
    keep = ~remove
    def cat(key, extra):
        v = ck[key]; base = v.detach()[torch.from_numpy(keep)]
        out = torch.cat([base, torch.as_tensor(extra, dtype=base.dtype)], 0)
        ck[key] = torch.nn.Parameter(out, requires_grad=v.requires_grad) if isinstance(v, torch.nn.Parameter) else out
    m = len(NP)
    cat("positions", NP)
    cat("rotation", q)
    cat("scale", np.tile(np.log([0.6 * args.cell / s, 0.6 * args.cell / s, 0.001 / s]), (m, 1)))
    cat("density", np.full((m, 1), np.log(0.98 / 0.02)))
    cat("features_albedo", NA)
    cat("features_specular", np.zeros((m, ck["features_specular"].shape[1])))
    for k_, v in list(ck.items()):                                     # any other per-gaussian tensor: pad
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N0 and k_ not in ("positions", "rotation", "scale", "density",
                                                                                "features_albedo", "features_specular"):
            ck[k_] = torch.cat([v.detach()[torch.from_numpy(keep)], torch.zeros((m,) + tuple(v.shape[1:]), dtype=v.dtype)], 0)
    ck.pop("optimizer", None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    base = os.path.splitext(args.out)[0]
    np.savez_compressed(base + "_unseen.npz", unseen=np.concatenate(unseen_all) if unseen_all else np.zeros((0, 6)))
    info = {"gaussians_in": N0, "removed_unobserved_in_objects": int(remove.sum()), "added": m, "objects": rep,
            "params": {k_: v for k_, v in vars(args).items() if k_ in ("cell", "hcell", "max_incidence", "depth_tol_m", "prune_margin_m")}}
    json.dump(info, open(base + "_report.json", "w"), indent=2)
    print(f"[complete] {len(rep)} objects: removed {remove.sum():,} unobserved gaussians, added {m:,} surface gaussians "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
