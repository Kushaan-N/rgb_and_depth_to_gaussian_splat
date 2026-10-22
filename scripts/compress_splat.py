"""Compress a standard 3DGS .ply for laptop viewing (PlayCanvas splat-transform: SOG = webp-packed, k-means SH,
~10-20x smaller; .compressed.ply = PlayCanvas chunk-quantised PLY, ~4x smaller) and check the result as SuperSplat
shows it (sbatch/compress_check.sbatch renders original vs compressed in the live editor and calls `report`).

  convert: <ply> -> <out-dir>/<stem>.sog (+ <stem>_i<N>.sog when --iterations != 10) and <stem>.compressed.ply;
           sizes / gaussian counts / seconds -> <out-dir>/<stem>_compress.json. SOG compression runs on a GPU
           (WebGPU via Dawn; `splat-transform -L` lists adapters) - run on a GPU node; --gpu cpu works but is slow.
  report:  per test set and variant: eval_views JSONs (+ alpha-hole fraction) -> summary.json with deltas vs the
           original, a table, and montages real | original | variants | |original - variant| x DIFF_GAIN.

    python scripts/compress_splat.py convert --ply a.ply --out-dir comp/ [--formats sog compressed.ply] [--iterations 10]
    python scripts/compress_splat.py report --check-dir comp/check --tests onpath=<ds> --variants orig sog cply
Env: SPLAT_TRANSFORM (binary; default $CEAR_WS/tools/splat-transform/node_modules/.bin/splat-transform; needs node>=20 on PATH)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ST = os.environ.get("SPLAT_TRANSFORM") or os.path.join(os.environ.get("CEAR_WS", "/scratch4/workspace/knaskar_umass_edu-cear"),
                                                        "tools/splat-transform/node_modules/.bin/splat-transform")
DIFF_GAIN = 4


def n_gaussians(ply):
    """Vertex count from a .ply header (None for .sog / other)."""
    if not ply.endswith(".ply"):
        return None
    with open(ply, "rb") as f:
        for line in f.read(65536).split(b"\n"):
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return None


def out_name(stem, fmt, iters):
    return f"{stem}{'_i%d' % iters if fmt == 'sog' and iters != 10 else ''}.{fmt}"


def convert(a):
    os.makedirs(a.out_dir, exist_ok=True)
    stem = a.stem or os.path.basename(a.ply)[:-4]
    jf = os.path.join(a.out_dir, f"{stem}_compress.json")
    res = json.load(open(jf)) if os.path.exists(jf) else {}
    res["input"] = {"path": os.path.abspath(a.ply), "bytes": os.path.getsize(a.ply), "gaussians": n_gaussians(a.ply)}
    for fmt in a.formats:
        out = os.path.join(a.out_dir, out_name(stem, fmt, a.iterations))
        if os.path.exists(out) and not a.force:
            print(f"[compress] exists: {out}"); continue
        cmd = [ST, "-w", *(["-g", a.gpu] if a.gpu else []), *(["-i", str(a.iterations)] if fmt == "sog" else []), a.ply, out]
        print("[compress]", " ".join(cmd), flush=True); t0 = time.time()
        subprocess.run(cmd, check=True)
        r = {"path": out, "bytes": os.path.getsize(out), "seconds": round(time.time() - t0, 1), "gaussians": n_gaussians(out),
             **({"iterations": a.iterations} if fmt == "sog" else {})}
        r["ratio"] = round(res["input"]["bytes"] / r["bytes"], 2)
        res[os.path.basename(out)] = r
        print(f"[compress] {os.path.basename(out)}: {r['bytes'] / 1e6:.1f} MB (x{r['ratio']} smaller) in {r['seconds']} s", flush=True)
    json.dump(res, open(jf, "w"), indent=1)
    print(f"[compress] {os.path.basename(a.ply)} {res['input']['bytes'] / 1e6:.1f} MB, {res['input']['gaussians']} gaussians -> {jf}")


def report(a):
    import cv2, numpy as np
    C = a.check_dir; rep = os.path.join(C, "report"); tests = dict(t.split("=", 1) for t in a.tests)
    J = lambda f: json.load(open(f)) if os.path.exists(f) else None   # noqa: E731
    hole = lambda d: (J(os.path.join(d, "_viewer.json")) or {}).get("alpha_hole_frac")   # noqa: E731
    res, keys = {}, ("psnr", "ssim", "lpips")
    print(f"{'set':16s}{'variant':10s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}{'dPSNR':>8s}{'dSSIM':>8s}{'dLPIPS':>8s}{'holes':>8s}")
    for t in tests:
        for suf in ("", "_static"):
            base = J(os.path.join(rep, f"{t}_orig{suf}.json"))
            if base is None:
                continue
            for v in a.variants:
                r = J(os.path.join(rep, f"{t}_{v}{suf}.json"))
                if r is None:
                    continue
                e = {k: r[k] for k in keys}; e.update({f"d_{k}": round(r[k] - base[k], 4) for k in keys})
                e["alpha_hole_frac"] = hole(os.path.join(C, t, v)); res[f"{t}{suf}/{v}"] = e
                h = e["alpha_hole_frac"]
                print(f"{t + suf:16s}{v:10s}{r['psnr']:8.3f}{r['ssim']:8.4f}{r['lpips']:8.4f}{e['d_psnr']:+8.3f}{e['d_ssim']:+8.4f}"
                      f"{e['d_lpips']:+8.4f}{'-' if h is None else f'{h:.5f}':>8s}")
    for v in a.variants:                                              # off-path: compressed vs original pixels (no GT)
        d0, d1 = os.path.join(C, "offpath", "orig"), os.path.join(C, "offpath", v)
        if v == "orig" or not os.path.isdir(d1):
            continue
        ps = []
        for f in sorted(os.listdir(d0)):
            if f.endswith(".png") and not f.startswith("_") and os.path.exists(os.path.join(d1, f)):
                x, y = cv2.imread(os.path.join(d0, f)).astype(np.float64), cv2.imread(os.path.join(d1, f)).astype(np.float64)
                ps.append(10 * np.log10(255 ** 2 / max(((x - y) ** 2).mean(), 1e-9)))
        res[f"offpath/{v}"] = {"psnr_vs_orig": float(np.mean(ps)) if ps else None, "n": len(ps),
                               "alpha_hole_frac": hole(d1), "alpha_hole_frac_orig": hole(d0)}
        print(f"offpath {v}: PSNR vs original render {res[f'offpath/{v}']['psnr_vs_orig']:.2f} dB over {len(ps)} views; "
              f"holes {hole(d1)} (orig {hole(d0)})")
    json.dump(res, open(os.path.join(C, "summary.json"), "w"), indent=1)

    def panel(p, lab, size):
        x = cv2.imread(p) if p and os.path.exists(p) else None
        x = np.zeros((size[1], size[0], 3), np.uint8) if x is None else cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        cv2.putText(x, lab, (6, 22), 0, 0.6, (0, 0, 255), 2); return x

    def montage(rows_spec, out):
        rows = [np.hstack(r) for r in rows_spec if r]
        if rows:
            cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]); print("montage ->", out)
    vs = [v for v in a.variants if v != "orig"]
    for t, ds in tests.items():                                      # held-out: real | orig | variants | diffs
        d0 = os.path.join(C, t, "orig"); fr = sorted(f for f in os.listdir(d0) if f.endswith(".png") and not f.startswith("_"))
        rows = []
        for f in fr[:: max(1, len(fr) // a.rows)][:a.rows]:
            im0 = cv2.imread(os.path.join(d0, f)); W = 480; size = (W, int(W * im0.shape[0] / im0.shape[1]))
            gt = [g for g in (os.path.join(ds, "images", os.path.splitext(f)[0] + e) for e in (".png", ".jpg", ".JPG", ".jpeg")) if os.path.exists(g)]
            row = [panel(gt[0] if gt else None, "real", size), panel(os.path.join(d0, f), "original", size)]
            row += [panel(os.path.join(C, t, v, f), v, size) for v in vs]
            for v in vs:
                p = os.path.join(C, t, v, f)
                if os.path.exists(p):
                    d = np.clip(np.abs(cv2.imread(p).astype(np.int16) - im0.astype(np.int16)) * DIFF_GAIN, 0, 255).astype(np.uint8)
                    row.append(cv2.resize(d, size)); cv2.putText(row[-1], f"|orig-{v}|x{DIFF_GAIN}", (6, 22), 0, 0.6, (0, 0, 255), 2)
            rows.append(row)
        montage(rows, os.path.join(C, f"montage_{t}.jpg"))
    d0 = os.path.join(C, "offpath", "orig")
    if os.path.isdir(d0):
        fr = sorted(f for f in os.listdir(d0) if f.endswith(".png") and not f.startswith("_")); rows = []
        for f in fr[:: max(1, len(fr) // a.rows)][:a.rows]:
            im0 = cv2.imread(os.path.join(d0, f)); size = (480, int(480 * im0.shape[0] / im0.shape[1]))
            row = [panel(os.path.join(d0, f), "original (off-path)", size)] + [panel(os.path.join(C, "offpath", v, f), v, size) for v in vs]
            for v in vs:
                p = os.path.join(C, "offpath", v, f)
                if os.path.exists(p):
                    d = np.clip(np.abs(cv2.imread(p).astype(np.int16) - im0.astype(np.int16)) * DIFF_GAIN, 0, 255).astype(np.uint8)
                    row.append(cv2.resize(d, size)); cv2.putText(row[-1], f"|orig-{v}|x{DIFF_GAIN}", (6, 22), 0, 0.6, (0, 0, 255), 2)
            rows.append(row)
        montage(rows, os.path.join(C, "montage_offpath.jpg"))
    eds = [(v, os.path.join(C, "offpath", v, "_supersplat_editor.png")) for v in a.variants]
    eds += [(os.path.basename(os.path.dirname(p)), p) for p in a.extra_editor]
    eds = [panel(p, lab, (640, 400)) for lab, p in eds if os.path.exists(p)]
    if eds:
        cv2.imwrite(os.path.join(C, "montage_editor.jpg"), np.vstack([np.hstack(eds[i:i + 2]) if i + 1 < len(eds) else
                    np.hstack([eds[i], eds[i] * 0]) for i in range(0, len(eds), 2)])); print("montage ->", os.path.join(C, "montage_editor.jpg"))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("convert"); c.add_argument("--ply", required=True); c.add_argument("--out-dir", required=True)
    c.add_argument("--stem", default=None); c.add_argument("--formats", nargs="+", default=["sog", "compressed.ply"])
    c.add_argument("--iterations", type=int, default=10, help="SOG SH k-means iterations (splat-transform -i)")
    c.add_argument("--gpu", default=None, help="splat-transform -g: adapter index or 'cpu'"); c.add_argument("--force", action="store_true")
    r = sp.add_parser("report"); r.add_argument("--check-dir", required=True)
    r.add_argument("--tests", nargs="+", required=True, help="name=<dataset dir> (held-out sets rendered under <check-dir>/<name>/<variant>)")
    r.add_argument("--variants", nargs="+", required=True, help="'orig' first, then compressed variants")
    r.add_argument("--rows", type=int, default=4); r.add_argument("--extra-editor", nargs="*", default=[])
    a = ap.parse_args()
    {"convert": convert, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
