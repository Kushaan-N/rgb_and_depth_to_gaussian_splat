"""One table of every scored splat across all sequences -> results/LEADERBOARD.md (login-node safe; reads JSON only).

Walks each outputs root (bounded depth) for report JSONs and collects every record with numeric psnr/ssim/lpips
(eval_views.py / viewer_shots scoring format, also nested inside summary.json files). Records are grouped by
sequence and by evaluation set (sequence + scored view count n + set name such as live_test_out / black / grey),
deduplicated by pred_dir, and sorted by PSNR; LPIPS-best is marked. Only comparable rows share a table.

    python scripts/leaderboard.py [--roots DIR ...] [--out results/LEADERBOARD.md] [--top 15]
"""
from __future__ import annotations
import argparse, glob, json, os, time

DEFAULT_ROOTS = ["/scratch4/workspace/knaskar_umass_edu-cear/outputs", "/scratch4/workspace/knaskar_umass_edu-multicam/outputs"]


def records(obj, chain=()):
    if isinstance(obj, dict):
        if all(isinstance(obj.get(k), (int, float)) for k in ("psnr", "ssim", "lpips")):
            yield chain, obj
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from records(v, chain + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from records(v, chain + (str(i),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "LEADERBOARD.md"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--max-depth", type=int, default=7)
    args = ap.parse_args()
    groups, seen = {}, set()
    for root in args.roots:
        files = []
        for d in range(1, args.max_depth + 1):
            files += glob.glob(os.path.join(root, *(["*"] * d)) + ".json")
        for f in files:
            if "report" not in f and "score" not in f:
                continue
            try:
                data = json.load(open(f))
            except Exception:
                continue
            rel = os.path.relpath(f, root).split(os.sep)
            seq = rel[0]
            for chain, r in records(data):
                if "per_camera" in chain or any(c.isdigit() for c in chain):
                    continue
                key = r.get("pred_dir") or (f, chain)
                if key in seen:
                    continue
                seen.add(key)
                tag = r.get("tag") or "/".join(chain) or os.path.splitext(rel[-1])[0]
                setname = "/".join(c for c in chain[1:] if c not in (tag,)) or (rel[-3] if len(rel) >= 3 else "")
                hint = f"{r.get('pred_dir', '')}/{rel[-1]}/{tag}"
                bg = "grey" if "grey" in hint else "black" if "black" in hint else ""
                for suf in ("_black", "_grey"):
                    tag = tag.replace(suf, "")
                setname = "/".join(x for x in setname.replace("_black", "").replace("_grey", "").split("/") if x not in ("black", "grey"))
                setname = f"{setname} [{bg or 'bg?'}]"
                g = groups.setdefault((seq, r.get("n", "?"), setname), [])
                g.append((r["psnr"], r["ssim"], r["lpips"], tag, os.path.relpath(f, root), os.path.getmtime(f)))
    out = [f"# Leaderboard\n\nGenerated {time.strftime('%F %H:%M')} by scripts/leaderboard.py from {', '.join(args.roots)}.",
           "Rows within one table share sequence, scored-view count and evaluation set; PSNR/SSIM higher, LPIPS lower.\n"]
    for (seq, n, setname), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: -x[0])
        best_l = min(r[2] for r in rows)
        out.append(f"## {seq} — n={n} {setname}\n\n| # | arm | PSNR | SSIM | LPIPS | source | date |\n|---|---|---|---|---|---|---|")
        for i, (p, s, l, tag, src, mt) in enumerate(rows[:args.top], 1):
            out.append(f"| {i} | {tag} | {p:.2f} | {s:.4f} | {l:.4f}{' *' if l == best_l else ''} | {src} | {time.strftime('%m-%d', time.localtime(mt))} |")
        out.append("")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(out) + "\n")
    print(f"[leaderboard] {sum(len(v) for v in groups.values())} records, {len(groups)} groups -> {args.out}")


if __name__ == "__main__":
    main()
