"""Download an Oxford Spires subset from Hugging Face (ori-drs/oxford_spires_dataset), checksum-verified.

Stdlib only (urllib + hashlib + zipfile), so it runs in any venv. The files to fetch are listed in the sequence
config (`download.files`, Hugging Face repo paths); each lands at <data_root>/raw/<repo path>. LFS files are verified
against the sha256 the Hub reports (paths-info API), small text files against their size; `.zip` files are unpacked
next to themselves (into <name>/) and kept, so a re-run skips everything already verified. Single stream, resumable
(HTTP Range), so it is gentle enough for a login node for small sets; use sbatch/spires_fetch.sbatch for GB-scale.

    python scripts/datasets/spires_fetch.py configs/spires_roq01.yaml [--only nvs_roq_01.zip] [--dry-run] [--extra]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, urllib.request, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pose_utils as pu  # noqa: E402


def paths_info(repo: str, rev: str, paths: list[str]) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/paths-info/{rev}",
                                 data=json.dumps({"paths": paths}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return {e["path"]: e for e in json.load(r)}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: str, size: int):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    for attempt in range(5):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                t0, n = time.time(), 0
                for b in iter(lambda: r.read(1 << 22), b""):
                    f.write(b); n += len(b)
                print(f"  {n / 2**20:.0f} MiB in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:                                   # noqa: BLE001 - retry any network error
            print(f"  retry {attempt + 1}: {e}", flush=True); time.sleep(10)
    os.replace(part, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--only", nargs="*", help="substrings: fetch only matching files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", action="store_true", help="also download.files_extra (e.g. the 6 GB raw 20 Hz frames)")
    a = ap.parse_args()
    cfg = pu.load_config(a.config); dl = cfg["download"]
    root = os.path.join(cfg["sequence"]["data_root"], "raw")
    allf = dl["files"] + (dl.get("files_extra") or [])        # files_extra: only with --extra (large, optional)
    files = [f for f in (allf if a.extra else dl["files"]) if not a.only or any(s in f for s in a.only)]
    info = paths_info(dl["repo"], dl.get("revision", "main"), files)
    total = sum(info[f]["size"] for f in files if f in info)
    print(f"[fetch] {len(files)} files, {total / 2**30:.2f} GiB -> {root}")
    for f in files:
        if f not in info:
            sys.exit(f"[fetch] not on the Hub: {f}")
        e, dst = info[f], os.path.join(root, f)
        want = (e.get("lfs") or {}).get("oid")
        ok = os.path.exists(dst) and os.path.getsize(dst) == e["size"]
        if ok and want and os.path.exists(dst + ".sha256"):
            ok = open(dst + ".sha256").read().strip() == want
        elif ok and want:
            ok = sha256(dst) == want
            if ok:
                open(dst + ".sha256", "w").write(want)
        print(f"[fetch] {'have' if ok else 'get '} {f} ({e['size'] / 2**20:.0f} MiB)", flush=True)
        if a.dry_run:
            continue
        if not ok:
            url = f"https://huggingface.co/datasets/{dl['repo']}/resolve/{dl.get('revision', 'main')}/{f}"
            download(url, dst, e["size"])
            if os.path.getsize(dst) != e["size"]:
                sys.exit(f"[fetch] size mismatch {f}")
            if want:
                got = sha256(dst)
                if got != want:
                    os.remove(dst); sys.exit(f"[fetch] sha256 mismatch {f}: {got} != {want}")
                open(dst + ".sha256", "w").write(want)
        if f.endswith(".zip"):
            out = dst[:-4]
            if not os.path.exists(os.path.join(out, ".unzipped")):
                with zipfile.ZipFile(dst) as z:
                    z.extractall(out)
                open(os.path.join(out, ".unzipped"), "w").write("ok\n")
                print(f"[fetch]   unzipped -> {out}", flush=True)
    print("SPIRES_FETCH_DONE")


if __name__ == "__main__":
    main()
