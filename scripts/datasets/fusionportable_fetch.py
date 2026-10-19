"""Download a FusionPortable v2 sequence subset (rosbag .7z + GT trajectory + reference odometry) and the calibration repo.

The dataset is public (Google Drive folder 1PYhnf3PlY5r0hbyzWDGTUTPxRMl6SYa-, no form/registration); files to fetch are
listed in the sequence config as `download.files: {<name in data_root>: <gdrive id>}`. Downloads with gdown (single
stream, resumable), unpacks `.7z` with the system 7z next to itself (the .7z is deleted only after `7z t` passes and the
bag opens), and records size + md5 of every file in <data_root>/MANIFEST.json (Drive publishes no checksums, so the
manifest is what later runs and docs/DATASETS.md compare against). Re-runs skip everything already present.
Calibration: `git clone` of download.calib_repo into download.calib_dir (tiny).

    python scripts/datasets/fusionportable_fetch.py configs/fp2_legged_room00.yaml [--only gt_traj.txt] [--keep-7z]
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pose_utils as pu  # noqa: E402


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gdown_file(fid: str, dst: str) -> None:
    import gdown
    tmp = dst + ".part"
    for attempt in range(5):
        try:
            gdown.download(id=fid, output=tmp, quiet=False, resume=True)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, dst); return
        except Exception as e:  # Drive quota / transient errors: back off and resume
            print(f"[fp2-fetch] {os.path.basename(dst)} attempt {attempt + 1} failed: {e}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise SystemExit(f"[fp2-fetch] could not download {fid} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", nargs="*", default=None, help="substrings: fetch only matching files")
    ap.add_argument("--keep-7z", action="store_true")
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    root = cfg["sequence"]["data_root"]; dl = cfg["download"]; os.makedirs(root, exist_ok=True)
    cdir = dl.get("calib_dir")
    if cdir and not os.path.isdir(os.path.join(cdir, ".git")):
        subprocess.run(["git", "clone", "-q", "--depth", "1", dl["calib_repo"], cdir], check=True)
        print(f"[fp2-fetch] calibration repo -> {cdir}")
    man_p = os.path.join(root, "MANIFEST.json")
    man = json.load(open(man_p)) if os.path.exists(man_p) else {}
    for name, fid in (dl.get("files") or {}).items():
        if args.only and not any(s in name for s in args.only):
            continue
        dst = os.path.join(root, name)
        bag = os.path.join(root, name[:-3] + ".bag") if name.endswith(".7z") else None
        if (bag and os.path.exists(bag) and name in man) or (not bag and os.path.exists(dst) and name in man):
            print(f"[fp2-fetch] have {name}"); continue
        if not os.path.exists(dst):
            print(f"[fp2-fetch] downloading {name} (gdrive {fid})", flush=True)
            gdown_file(fid, dst)
        man[name] = {"gdrive": fid, "bytes": os.path.getsize(dst), "md5": md5(dst)}
        if bag:
            subprocess.run(["7z", "t", dst], check=True, stdout=subprocess.DEVNULL)
            if not os.path.exists(bag):
                subprocess.run(["7z", "x", "-y", f"-o{root}", dst], check=True, stdout=subprocess.DEVNULL)
            got = glob.glob(os.path.join(root, "**", "*.bag"), recursive=True)
            if not os.path.exists(bag) and len(got) == 1:   # archives hold <seq>/<seq>.bag: move it up
                os.replace(got[0], bag)
                try:
                    os.rmdir(os.path.dirname(got[0]))
                except OSError:
                    pass
            from rosbags.highlevel import AnyReader
            from pathlib import Path
            with AnyReader([Path(bag)]) as r:
                topics = {c.topic: c.msgcount for c in r.connections}
                man[name]["bag"] = {"file": os.path.basename(bag), "bytes": os.path.getsize(bag),
                                    "duration_s": (r.end_time - r.start_time) * 1e-9, "topics": topics}
            print(f"[fp2-fetch] {os.path.basename(bag)}: {os.path.getsize(bag) / 1e9:.2f} GB, "
                  f"{man[name]['bag']['duration_s']:.1f} s, {len(topics)} topics", flush=True)
            if not args.keep_7z:
                os.remove(dst)
        json.dump(man, open(man_p, "w"), indent=2)
    print(json.dumps(man, indent=2)[:4000])
    print(f"FP2_FETCH_DONE {root}")


if __name__ == "__main__":
    main()
