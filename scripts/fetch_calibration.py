"""Fetch a dataset's calibration release and generate the pipeline's per-sequence calibration file.

Replaces the manual "download the calibration release" step. Everything comes from the config:

  calibration.dir      where the release files live
  calibration.fetch    [{file, gdrive|url, expect}] — downloaded if missing; `expect` is text that
                       must appear in the file, so a wrong/corrupt/HTML download fails loudly
  calibration.convert  converter script in scripts/ (default convert_calibration.py) that turns the
                       release into intrinsics.calib_file; run only if that file is missing

Idempotent: existing files are validated, not re-downloaded (use --force to refetch).

    python scripts/fetch_calibration.py --config configs/<seq>.yaml
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download(entry: dict) -> bytes:
    if "url" in entry:
        return _get(entry["url"])
    gid = entry["gdrive"]
    data = _get(f"https://drive.google.com/uc?export=download&id={gid}")
    # large files get an interstitial "can't scan for viruses" page carrying a confirm token
    if data[:512].lower().lstrip().startswith((b"<!doctype html", b"<html")):
        m = re.search(rb'name="confirm"\s+value="([^"]+)"', data) or re.search(rb"confirm=([0-9A-Za-z_-]+)", data)
        if m:
            data = _get(f"https://drive.usercontent.google.com/download?id={gid}&export=download&confirm={m.group(1).decode()}")
    return data


def valid(data: bytes, expect: str) -> bool:
    head = data[:512].lower().lstrip()
    return bool(data) and not head.startswith((b"<!doctype html", b"<html")) and expect.encode() in data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dir", default=None, help="override calibration.dir (e.g. to verify into a scratch dir)")
    ap.add_argument("--no-convert", action="store_true")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    cal = cfg.get("calibration") or {}
    cdir = args.dir or cal.get("dir")
    if not cdir or not cal.get("fetch"):
        print("[calib] no calibration.fetch in config — nothing to do"); return 0
    os.makedirs(cdir, exist_ok=True)

    bad = []
    for e in cal["fetch"]:
        path = os.path.join(cdir, e["file"])
        if os.path.exists(path) and not args.force:
            if valid(open(path, "rb").read(), e["expect"]):
                print(f"[calib] ok       {e['file']}"); continue
            print(f"[calib] invalid  {e['file']} — refetching")
        try:
            data = download(e)
        except Exception as ex:  # network / 404
            bad.append(f"{e['file']}: {ex}"); continue
        if not valid(data, e["expect"]):
            bad.append(f"{e['file']}: downloaded content lacks {e['expect']!r} (wrong id or HTML page)"); continue
        with open(path, "wb") as f:
            f.write(data)
        print(f"[calib] fetched  {e['file']} ({len(data)} B)")
    if bad:
        print("[calib] FAILED:\n  " + "\n  ".join(bad)); return 1

    target = (cfg.get("intrinsics") or {}).get("calib_file")
    if target and not args.no_convert and (args.force or not os.path.exists(target)):
        conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), cal.get("convert", "convert_calibration.py"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        subprocess.run([sys.executable, conv, "--calib-dir", cdir, "--out", target], check=True)
        print(f"[calib] generated {target}")
    elif target:
        print(f"[calib] present  {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
