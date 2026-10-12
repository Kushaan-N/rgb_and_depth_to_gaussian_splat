"""Change masks for fusing several recordings of one place (CL-Splats-style change detection, DINOv2 features).

A fused splat (scripts/fuse_sequences.py) is supervised by every recording, so whatever differs between them
(moved objects, people, a door, small items) is averaged into one blurred compromise and makes each recording's
own views worse. The REFERENCE recording (fuse.members[0] unless --reference) stays fully supervised; every other
recording's training frame gets <image>_valid.png that switches the photometric loss off (3DGRUT fusion patch,
loss.valid_mask=true) where it shows something the reference splat disagrees with:
  1. the reference recording's own splat is rendered at the other recording's frames (poses mapped fused frame ->
     reference COLMAP frame via members/<ref>.json and the reference's sim3.json; done by the sbatch with
     eval_views.py from the datasets `prep` writes),
  2. render and real image are compared with DINOv2 (ViT-S/14, HF facebook/dinov2-small) patch cosine similarity
     after per-channel colour normalisation (render's mean/std matched to the image: exposure / white balance
     differences between recordings are not changes),
  3. a patch is CHANGED if its similarity is below tau. tau is calibrated per reference splat: the --calib-q
     quantile of the same similarity on the reference's own held-out frames (novel views of the same scene, no
     change; fuse_sequences.py's eval/<ref>/single), clipped to [--tau-min, --tau-max]. Isolated single patches
     are dropped,
  4. only where the reference actually OBSERVED the surface: the pixel's LiDAR point (<image>_depth.npy, fused
     units) is seen un-occluded by >= --min-views reference training cameras (covis_masks.py's test); pixels
     without LiDAR fall back to the reference render's opacity > --opacity-min. Surfaces only the other recording
     saw (the reason to fuse) are never masked,
  5. dilated by --dilate px; valid = 1 - changed (255/0 PNG).
--symmetric also masks the reference's frames against each other recording's splat (then a change is supervised
by neither recording where both saw it). The images themselves are never modified.

  prep   (CPU venv)  training dataset <out>/data (fused train frames, LiDAR depth in fused units, LiDAR seeds),
                     cross-render datasets <out>/xr/<owner>__at__<target> and <out>/change_plan.json
  masks  (GPU venv with torch + transformers, e.g. venv-difix) after the renders: <image>_valid.png in <out>/data,
                     <out>/change_masks.json (tau, masked fractions) and <out>/change_preview.jpg

    python scripts/change_masks.py prep  --fused <fused>/pipeline --out <work> [--reference <member>] [--ckpt m=glob]
    (render every plan pair: eval_views.py --checkpoint <ckpt> --dataset <xr>/<pair> --no-metrics
        --save-dir <xr>/<pair>/render --save-opacity-dir <xr>/<pair>/opacity)
    python scripts/change_masks.py masks --out <work> [--calib-q 0.02 --dilate 12 --min-views 2]
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model


def load_sim3(p):
    j = json.load(open(p))
    return float(j.get("scale", 1.0)), np.asarray(j["R"], float), np.asarray(j["t"], float)


def write_images(path, entries):
    """COLMAP images.txt from [(name, cid, R_cw, t_cw)]."""
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        for i, (name, cid, Rcw, tcw) in enumerate(sorted(entries, key=lambda e: e[0]), 1):
            q = Rotation.from_matrix(Rcw).as_quat()                                 # x, y, z, w
            f.write(f"{i} {q[3]:.12g} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} " + " ".join(f"{x:.12g}" for x in tcw)
                    + f" {cid} {name}\n\n")


def member_depth(MP, stem):
    for d in ("onpath/data_orig/images", "train_data/images", "dataset/images"):   # build_depth_targets.py output
        p = os.path.join(MP, d, stem + "_depth.npy")
        if os.path.exists(p):
            return p
    return None


def prep(a):
    P = a.fused; fz = json.load(open(os.path.join(P, "fuse.json")))
    members = list(fz["members"]); ref = a.reference or members[0]
    if ref not in members:
        raise SystemExit(f"[change] --reference {ref} not in {members}")
    info = {m: json.load(open(os.path.join(P, "members", f"{m}.json"))) for m in members}
    ck_over = dict(s.split("=", 1) for s in a.ckpt)
    ckpt = {}
    for m in members:
        g = ck_over.get(m, info[m]["ckpt_glob"]); c = sorted(glob.glob(g), key=os.path.getmtime)
        ckpt[m] = c[-1] if c else None
    src = os.path.join(P, "train_data", "sparse", "0")
    cams, imgs = read_text_model(src)
    D = os.path.join(a.out, "data"); os.makedirs(os.path.join(D, "sparse", "0"), exist_ok=True)
    for f in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copyfile(os.path.join(src, f), os.path.join(D, "sparse", "0", f))
    # training images (real per-member dirs: the change masks are written next to them) + LiDAR depth, fused units
    n_depth = 0
    for name, *_ in imgs:
        m, n = name.split("/", 1); stem = os.path.splitext(n)[0]
        os.makedirs(os.path.join(D, "images", m), exist_ok=True)
        dst = os.path.join(D, "images", m, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(P, "dataset", "images", name)), dst)
        dd = os.path.join(D, "images", m, stem + "_depth.npy")
        sd = member_depth(info[m]["pipeline"], stem)
        if sd and not os.path.exists(dd):            # member COLMAP units -> metres (fused frame = rigid of metric)
            s_m = load_sim3(os.path.join(info[m]["pipeline"], "sim3.json"))[0]
            np.save(dd, (np.load(sd).astype(np.float32) * s_m).astype(np.float16))
        n_depth += bool(sd)
    json.dump({"units": "ray distance, fused (metric) units", "frames": len(imgs), "with_depth": n_depth},
              open(os.path.join(D, "depth_targets.json"), "w"), indent=2)
    # cross renders: owner's splat (its own COLMAP frame) at target's training frames
    pairs = [(ref, m) for m in members if m != ref] + ([(m, ref) for m in members if m != ref] if a.symmetric else [])
    plan = {"fused": P, "reference": ref, "members": members, "ckpt": ckpt, "data": D, "pairs": [], "calib": {}}
    for o, tg in pairs:
        if not ckpt[o]:
            raise SystemExit(f"[change] no checkpoint for {o} ({info[o]['ckpt_glob']}); pass --ckpt {o}=<path>")
        s, R, t = load_sim3(os.path.join(info[o]["pipeline"], "sim3.json"))
        Tf = json.load(open(os.path.join(P, "members", f"{o}.json"))); Rf, tf = np.asarray(Tf["R"]), np.asarray(Tf["t"])
        d = os.path.join(a.out, "xr", f"{o}__at__{tg}"); os.makedirs(os.path.join(d, "sparse", "0"), exist_ok=True)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        ent = []
        for name, cid, q, tcw in imgs:
            if not name.startswith(tg + "/"):
                continue
            Rcw = quat_to_R(q); Cf = -Rcw.T @ tcw                       # fused frame camera
            Cm = Tf["scale"] * Rf @ Cf + tf if "scale" in Tf else Rf @ Cf + tf   # -> owner metric
            Co = R.T @ (Cm - t) / s; Rwc_o = R.T @ Rf @ Rcw.T                    # -> owner COLMAP frame
            n = name.split("/", 1)[1]; Rcw_o = Rwc_o.T
            ent.append((n, cid, Rcw_o, -Rcw_o @ Co))
            if not os.path.lexists(os.path.join(d, "images", n)):
                os.symlink(os.path.realpath(os.path.join(P, "dataset", "images", name)), os.path.join(d, "images", n))
        with open(os.path.join(d, "sparse", "0", "cameras.txt"), "w") as f:
            for cid in sorted({e[1] for e in ent}):
                W, H, p = cams[cid]; f.write(f"{cid} PINHOLE {W} {H} " + " ".join(f"{x:.10g}" for x in p) + "\n")
        write_images(os.path.join(d, "sparse", "0", "images.txt"), ent)
        open(os.path.join(d, "sparse", "0", "points3D.txt"), "w").close()
        plan["pairs"].append({"owner": o, "target": tg, "dataset": d, "ckpt": ckpt[o], "frames": len(ent)})
        cal = os.path.join(P, "eval", o, "single")                       # owner's own held-out frames, its frame
        if os.path.isdir(cal):
            plan["calib"][o] = {"dataset": cal, "ckpt": ckpt[o], "out": os.path.join(a.out, "xr", f"{o}__calib")}
    json.dump(plan, open(os.path.join(a.out, "change_plan.json"), "w"), indent=2)
    print(f"[change] reference {ref}; {len(imgs)} training frames ({n_depth} with LiDAR depth) -> {D}")
    for p in plan["pairs"]:
        print(f"[change] render {p['owner']} at {p['target']}: {p['frames']} frames  ({p['ckpt']})")
    print("CHANGE_PREP " + json.dumps({"pairs": len(plan["pairs"]), "calib": list(plan["calib"])}))


# ---------------------------------------------------------------------------------------------------------- masks
class Dino:
    def __init__(self, name, size, dev):
        import torch
        from transformers import AutoModel
        self.torch, self.dev, self.size = torch, dev, size                    # (H, W), multiples of 14
        self.m = AutoModel.from_pretrained(name).to(dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def __call__(self, x):                                                     # (H, W, 3) [0,1] -> (h, w, C) unit
        t = self.torch; F = t.nn.functional
        x = F.interpolate(x.permute(2, 0, 1)[None], size=self.size, mode="bilinear", align_corners=False)
        with t.no_grad():
            f = self.m(pixel_values=(x - self.mean) / self.std).last_hidden_state[0, 1:]
        h, w = self.size[0] // 14, self.size[1] // 14
        return F.normalize(f[-h * w:].float(), dim=-1).view(h, w, -1)


def masks(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    plan = json.load(open(os.path.join(a.out, "change_plan.json"))); D = plan["data"]
    dino = Dino(a.dino, (a.dino_h, a.dino_w), dev)
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).to(dev)
    loadg = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0).to(dev)

    def cnorm(r, g, w):                                  # render's per-channel mean/std -> the image's (weights w)
        w = w[..., None]; sw = w.sum().clamp_min(1.0)
        mr, mg = (r * w).sum((0, 1)) / sw, (g * w).sum((0, 1)) / sw
        sr = (((r - mr) ** 2 * w).sum((0, 1)) / sw).sqrt().clamp_min(1e-3); sg = (((g - mg) ** 2 * w).sum((0, 1)) / sw).sqrt()
        return ((r - mr) / sr * sg + mg).clamp(0, 1)

    def sim(real, rend, op):
        rn = cnorm(rend, real, (op > a.opacity_min).float())
        return (dino(real) * dino(rn)).sum(-1), rn                          # (h, w) cosine

    # 1. tau per owner from its own held-out views (no change there)
    taus = {}
    for o, c in plan["calib"].items():
        vals = []
        for n in sorted(os.listdir(os.path.join(c["out"], "render"))):
            real = os.path.join(c["dataset"], "images", n)
            if not os.path.exists(real):
                continue
            op = loadg(os.path.join(c["out"], "opacity", n))
            s, _ = sim(load(real), load(os.path.join(c["out"], "render", n)), op)
            opp = F.adaptive_avg_pool2d(op[None, None], s.shape)[0, 0]
            vals.append(s[opp > a.opacity_min].flatten().cpu())
        v = torch.cat(vals).numpy() if vals else np.array([a.tau_min])
        taus[o] = {"tau": float(np.clip(np.quantile(v, a.calib_q), a.tau_min, a.tau_max)), "q": a.calib_q,
                   "calib_patches": int(v.size), "calib_median": round(float(np.median(v)), 4)}
        print(f"[change] {o}: held-out patch similarity median {np.median(v):.3f}, q{a.calib_q} -> tau {taus[o]['tau']:.3f}")
    if a.tau is not None:
        taus = {o: {"tau": a.tau, "fixed": True} for o in plan["members"]}

    # 2. per pair: change x observed-by-owner -> dilate -> <image>_valid.png
    cams, imgs = read_text_model(os.path.join(D, "sparse", "0"))
    def cam(e):
        name, cid, q, tcw = e; W, H, (fx, fy, cx, cy) = cams[cid]
        return name, W, H, fx, fy, cx, cy, torch.tensor(quat_to_R(q), device=dev, dtype=torch.float32), \
            torch.tensor(tcw, device=dev, dtype=torch.float32)
    dpath = lambda name: os.path.join(D, "images", os.path.splitext(name)[0] + "_depth.npy")
    report, prev, valid_all = {"taus": taus, "pairs": []}, [], {}
    for pr in plan["pairs"]:
        o, tg, xr = pr["owner"], pr["target"], pr["dataset"]
        tau = taus.get(o, {"tau": a.tau_min})["tau"]
        oc = [cam(e) for e in imgs if e[0].startswith(o + "/") and os.path.exists(dpath(e[0]))]
        od = [torch.from_numpy(np.load(dpath(c[0])).astype(np.float32)).to(dev) for c in oc]
        fr = {"changed": [], "masked": [], "observed": []}
        tgt = [e for e in imgs if e[0].startswith(tg + "/")]
        show = set(np.linspace(0, len(tgt) - 1, a.previews).astype(int).tolist()) if tgt else set()
        for fi, e in enumerate(tgt):
            name, W, H, fx, fy, cx, cy, Rcw, tcw = cam(e); n = name.split("/", 1)[1]
            real, rend = load(os.path.join(D, "images", name)), load(os.path.join(xr, "render", n))
            op = loadg(os.path.join(xr, "opacity", n))
            s, rn = sim(real, rend, op)
            ch = (s < tau).float()
            nb = F.conv2d(ch[None, None], torch.ones(1, 1, 3, 3, device=dev), padding=1)[0, 0] - ch
            ch = ch * (nb >= a.min_neighbours).float()                                  # drop isolated patches
            chp = F.interpolate(ch[None, None], size=(H, W), mode="nearest")[0, 0] > 0
            # observed by the owner: LiDAR point seen un-occluded by >= min_views owner cameras
            obs = op > a.opacity_min                                                    # fallback: no LiDAR
            if os.path.exists(dpath(name)) and oc:
                Dt = torch.from_numpy(np.load(dpath(name)).astype(np.float32)).to(dev)
                st = a.covis_stride
                v, u = torch.meshgrid(torch.arange(st // 2, H, st, device=dev), torch.arange(st // 2, W, st, device=dev), indexing="ij")
                d = Dt[v, u]
                r = torch.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones_like(d)], -1)
                X = (F.normalize(r, dim=-1) * d[..., None] - tcw) @ Rcw                 # world (fused) points
                cnt = torch.zeros_like(d)
                for (_, W2, H2, fx2, fy2, cx2, cy2, R2, t2), D2 in zip(oc, od):
                    Pc = X @ R2.T + t2; z = Pc[..., 2]
                    uu = torch.floor(fx2 * Pc[..., 0] / z.clamp_min(1e-6) + cx2).long()
                    vv = torch.floor(fy2 * Pc[..., 1] / z.clamp_min(1e-6) + cy2).long()
                    ok = (z > 0.05) & (uu >= 0) & (uu < W2) & (vv >= 0) & (vv < H2)
                    dt = D2[vv.clamp(0, H2 - 1), uu.clamp(0, W2 - 1)]; rg = Pc.norm(dim=-1)
                    cnt += (ok & (dt > 0) & ((rg - dt).abs() < a.covis_tol * dt)).float()
                seen = F.interpolate(cnt[None, None], size=(H, W), mode="nearest")[0, 0] >= a.min_views
                hasd = F.interpolate((d > 0).float()[None, None], size=(H, W), mode="nearest")[0, 0] > 0
                obs = torch.where(hasd, seen, obs)
            chg = (chp & obs).float()
            k = 2 * a.dilate + 1
            chg = F.max_pool2d(chg[None, None], k, stride=1, padding=a.dilate)[0, 0] if a.dilate > 0 else chg
            valid = 1 - chg
            vb = (valid > 0.5).cpu()                                     # several owners per target: AND
            valid_all[name] = vb if name not in valid_all else valid_all[name] & vb
            fr["changed"].append(float(chp.float().mean())); fr["observed"].append(float(obs.float().mean()))
            fr["masked"].append(float(1 - valid.mean()))
            if fi in show:
                sm = F.interpolate(s[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                heat = torch.stack([1 - sm, sm, torch.zeros_like(sm)], -1).clamp(0, 1)
                over = real * (0.35 + 0.65 * valid[..., None]) + torch.tensor([0.6, 0, 0], device=dev) * chg[..., None]
                prev.append(torch.cat([real, rn, heat, over.clamp(0, 1)], 1))
        r = {"owner": o, "target": tg, "tau": tau, "frames": len(fr["masked"]),
             "masked_fraction_mean": round(float(np.mean(fr["masked"])), 4),
             "masked_fraction_p90": round(float(np.quantile(fr["masked"], 0.9)), 4),
             "dino_changed_fraction_mean": round(float(np.mean(fr["changed"])), 4),
             "observed_by_owner_mean": round(float(np.mean(fr["observed"])), 4)}
        report["pairs"].append(r); print("[change] " + json.dumps(r), flush=True)
    for name, vb in valid_all.items():
        Image.fromarray(vb.numpy().astype(np.uint8) * 255).save(os.path.join(D, "images", os.path.splitext(name)[0] + a.suffix))
    report["frames_with_mask"] = len(valid_all)
    json.dump(report, open(os.path.join(a.out, "change_masks.json"), "w"), indent=2)
    if prev:
        im = torch.cat(prev, 0); im = F.interpolate(im.permute(2, 0, 1)[None], scale_factor=0.5, mode="area")[0]
        Image.fromarray((im.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()).save(os.path.join(a.out, "change_preview.jpg"))
    print("CHANGE_MASKS " + json.dumps(report["pairs"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fused", required=True, help="fused <out>/pipeline (fuse_sequences.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None, help="member kept fully supervised (default: fuse.members[0])")
    p.add_argument("--ckpt", nargs="*", default=[], help="member=<ckpt or glob> (default: the member's pipeline splat)")
    p.add_argument("--symmetric", action="store_true", help="also mask the reference's frames")
    m = sub.add_parser("masks")
    m.add_argument("--out", required=True)
    m.add_argument("--dino", default="facebook/dinov2-small")
    m.add_argument("--dino-h", type=int, default=504); m.add_argument("--dino-w", type=int, default=672)
    m.add_argument("--tau", type=float, default=None, help="fixed similarity threshold (default: calibrated)")
    m.add_argument("--calib-q", type=float, default=0.02, help="quantile of the owner's held-out similarities")
    m.add_argument("--tau-min", type=float, default=0.3); m.add_argument("--tau-max", type=float, default=0.8)
    m.add_argument("--min-neighbours", type=int, default=1, help="changed patches need this many changed neighbours")
    m.add_argument("--opacity-min", type=float, default=0.5)
    m.add_argument("--min-views", type=int, default=2); m.add_argument("--covis-tol", type=float, default=0.05)
    m.add_argument("--covis-stride", type=int, default=4)
    m.add_argument("--dilate", type=int, default=12)
    m.add_argument("--suffix", default="_valid.png")
    m.add_argument("--previews", type=int, default=3, help="preview rows per pair")
    a = ap.parse_args()
    return prep(a) if a.cmd == "prep" else masks(a)


if __name__ == "__main__":
    sys.exit(main())
