"""GT-aware visibility budget inside object zones (gsplat venv, GPU): solidify_unseen.py budget with the seen-view rule
relaxed for object-prior edits only (user decision 2026-11-02).

Per training view, renders of the input (ref), the candidate (img) and the real image (gt):
  * a pixel is WORSE if the candidate moved it (> --thr off ref) and, averaged over its --patch x --patch patch, it is
    further from the real image than the input was by > --tol (|img-gt| > |ref-gt| + tol);
  * zone rows (new rows + input rows in an object zone, object_priors.py zone.npz) are reverted / dropped only where
    they blend > --min-w into a WORSE pixel - a change that brings a seen pixel closer to the real image is kept;
  * every other row keeps solidify_unseen.py budget's strict rule (visible cluster > --vis-thr, or > --thr in a view
    that lost > --view-db);
  * per-view sanity cap: in a view whose PSNR vs the real image dropped by > --view-db, every changed pixel counts as
    worse (zone rows too);
  * removed input rows the input's render shows in a flagged pixel are restored. Repeated until nothing changes.
Output / src semantics as solidify_unseen.py budget (untouched rows byte-identical to the input).

    python scripts/object_priors/zone_budget.py --input in.ply --cand refit.ply --src refit_src.npz --zone T/zone.npz \
        --dataset <train ds> <test ds> --out budget.ply
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from solidify_unseen import pair_weights, all_frames, gt_image, write_vertex  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--zone", required=True, help="object_priors.py zone.npz (input rows in an object zone)")
    ap.add_argument("--dataset", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=2 / 255); ap.add_argument("--vis-thr", type=float, default=8 / 255)
    ap.add_argument("--tol", type=float, default=1 / 255, help="patch may get at most this much further from the real image")
    ap.add_argument("--patch", type=int, default=4); ap.add_argument("--view-db", type=float, default=0.3)
    ap.add_argument("--min-w", type=float, default=0.002); ap.add_argument("--display-scale", type=float, default=1.8)
    ap.add_argument("--iters", type=int, default=12)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from finetune_ply_viewer import ViewerRaster, cam_args, load_ply, sh_deg, to_t
    from refine_poses_gsplat import psnr
    dev = "cuda"
    vi, pi = load_ply(a.input, dev); vc, pc = load_ply(a.cand, dev); N = len(pi["means"])
    src = torch.tensor(np.load(a.src)["src"], device=dev); isin = src >= 0
    zin = torch.tensor(np.load(a.zone)["zone"], device=dev); assert len(zin) == N, "zone mask is for another input"
    ci = torch.full((N,), -1, dtype=torch.long, device=dev); ci[src[isin]] = torch.nonzero(isin)[:, 0]
    present = ci >= 0; use_c = present.clone(); di = torch.nonzero(~isin)[:, 0]; dkeep = torch.ones(len(di), dtype=torch.bool, device=dev)
    deg = sh_deg(pi); Rv = ViewerRaster("ss352", "exact", display_scale=a.display_scale)
    cams = [(n, to_t(np.linalg.inv(c2w), dev), cam_args(cam)) for n, cam, c2w in all_frames(a.dataset)]
    pool = lambda x: F.avg_pool2d(x[None, None], a.patch, 1, a.patch // 2)[0, 0][:x.shape[0], :x.shape[1]]

    def build():
        q = {}
        for k in pi:
            b = pi[k].clone(); u = present & use_c
            b[u] = pc[k][ci[u]]
            q[k] = torch.cat([b[present], pc[k][di[dkeep]]])
        return q
    hist = []
    with torch.no_grad():
        ref = {n: Rv.render(pi, w2c, *ca, sh_degree=deg)[0].half() for n, w2c, ca in cams}
        gts = {n: gt_image(a.dataset, n, dev) for n, _, _ in cams}
        for it in range(a.iters):
            q = build(); pidx = torch.nonzero(present)[:, 0]; npres = len(pidx)
            zrow = torch.cat([zin[pidx], torch.ones(int(dkeep.sum()), dtype=torch.bool, device=dev)])   # zone flag per q row
            rev = torch.zeros(N, dtype=torch.bool, device=dev); drop = torch.zeros(len(di), dtype=torch.bool, device=dev)
            res_ = torch.zeros(N, dtype=torch.bool, device=dev); drops, nw, ns, better = [], 0, 0, 0
            for n, w2c, ca in cams:
                img = Rv.render(q, w2c, *ca, sh_degree=deg)[0]; r = ref[n].float(); g = gts[n]
                d = (img - r).abs().amax(-1); moved = d > a.thr
                if not moved.any():
                    drops.append(0.0); continue
                db = psnr(r, g.float()) - psnr(img, g.float()) if g is not None else 0.0
                drops.append(db)
                if g is not None:
                    en, eo = pool((img - g.float()).abs().mean(-1)), pool((r - g.float()).abs().mean(-1))
                    worse = moved & (en > eo + a.tol); better += int((moved & (en < eo - a.tol)).sum())
                else:
                    worse = moved
                if db > a.view_db:
                    worse = moved                                               # per-view sanity cap
                big = (d > a.vis_thr).float()[None, None]
                strict = F.max_pool2d(-F.max_pool2d(-big, 3, 1, 1), 3, 1, 1)[0, 0] > 0
                if db > a.view_db:
                    strict = strict | moved
                if not (worse.any() or strict.any()):
                    continue
                nw += int(worse.sum()); ns += int(strict.sum())
                wf, sf = worse.reshape(-1), strict.reshape(-1)
                gid, pid, w = pair_weights(Rv, q, w2c, *ca); hit = w > a.min_w
                zr = zrow[gid]
                h = gid[hit & ((zr & wf[pid]) | (~zr & sf[pid]))]
                hi = h[h < npres]; rev[pidx[hi]] = True
                hd = h[h >= npres] - npres; full = torch.nonzero(dkeep)[:, 0]; drop[full[hd]] = True
                gid, pid, w = pair_weights(Rv, pi, w2c, *ca); hit = w > a.min_w
                res_[gid[hit & ((zin[gid] & wf[pid]) | (~zin[gid] & sf[pid]))]] = True
            rev |= res_ & present; res_ &= ~present; rev &= use_c
            hist.append({"iter": it, "worse_px": nw, "strict_px": ns, "better_px": better,
                         "views_over_db": int(sum(x > a.view_db for x in drops)), "mean_view_drop_db": float(np.mean(drops)),
                         "max_view_drop_db": float(np.max(drops)), "revert": int(rev.sum()), "drop_new": int(drop.sum()), "restore": int(res_.sum())})
            print(f"[zone-budget] {hist[-1]}", flush=True)
            if not (rev.any() or drop.any() or res_.any()):
                break
            use_c &= ~rev; dkeep &= ~drop; present |= res_
    pr, uc = present.cpu().numpy(), use_c.cpu().numpy(); cin = ci.cpu().numpy()
    rows = vi.data.copy(); u = pr & uc; rows[u] = vc.data[cin[u]]
    out = np.concatenate([rows[pr], vc.data[di.cpu().numpy()][dkeep.cpu().numpy()]])
    write_vertex(a.out, out)
    r = {"input": N, "present": int(pr.sum()), "cand_rows_kept": int(u.sum()), "reverted_to_input": int((pr & ~uc).sum()),
         "new_in": int(len(di)), "new_kept": int(dkeep.sum()), "discs_in": int(len(di)), "discs_kept": int(dkeep.sum()),
         "removed": int((~pr).sum()), "out": int(len(out)), "iterations": hist, "args": vars(a)}
    json.dump(r, open(os.path.splitext(a.out)[0] + "_budget.json", "w"), indent=1)
    print("ZONE_BUDGET", json.dumps({k: v for k, v in r.items() if k not in ("iterations", "args")}), flush=True)


if __name__ == "__main__":
    main()
