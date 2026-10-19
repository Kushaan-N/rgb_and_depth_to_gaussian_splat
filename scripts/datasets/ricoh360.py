"""Ricoh360 adapter (EgoNeRF, Choi et al. CVPR 2023; MIT): real 360 equirectangular captures (Ricoh Theta, 1920x960,
100 frames per scene, COLMAP-like SfM poses in arbitrary units, official 50/50 train/test lists). CPU venv.

  fetch    download Ricoh360.zip (Google Drive, public; `download.drive_id`, size-checked, resumable) into
           $CEAR_DATA/ricoh360 and unpack the configured scene's text model + images only
  prepare  scripts/datasets/omni_common.prepare(): the official test list is the held-out set;
           rect/{train,test}: each panorama -> virtual pinhole views (`rect.views`, cube-map style, default 4
           horizontal 100 deg views + up/down 90 deg) for the CEAR pinhole recipe and for SuperSplat scoring;
           native/all: the equirect frames + train.txt/test.txt exactly as ODGS reads them (sbatch/omni_native.sbatch);
           the equirect pixel convention is verified against the SfM tracks (meta.json `conventions`).

    python scripts/datasets/ricoh360.py fetch   configs/ricoh360_bricks.yaml
    python scripts/datasets/ricoh360.py prepare configs/ricoh360_bricks.yaml [--frames 2] [--workers 4]
"""
from __future__ import annotations
import argparse, os, sys, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu  # noqa: E402
import omni_common as oc  # noqa: E402


def cfg_paths(cfg):
    data = os.path.expandvars(cfg["paths"].get("data_root", "${CEAR_DATA}/ricoh360"))
    scene = cfg["sequence"]["scene"]
    return data, scene, os.path.join(data, "Ricoh360", scene), os.path.join(cfg["paths"]["out_root"], "pipeline")


def fetch(cfg, args):
    data, scene, sdir, _ = cfg_paths(cfg)
    d = cfg["download"]
    z = oc.fetch(f"https://drive.usercontent.google.com/download?id={d['drive_id']}&export=download&confirm=t",
                 os.path.join(data, d.get("file", "Ricoh360.zip")), d.get("md5"), d.get("bytes"))
    with zipfile.ZipFile(z) as zf:
        want = [n for n in zf.namelist() if n.startswith(f"Ricoh360/{scene}/") and not n.endswith("/")]
        todo = [n for n in want if not os.path.exists(os.path.join(data, n))]
        for n in todo:
            zf.extract(n, data)
    print(f"[ricoh360] {scene}: {len(want)} files ({len(todo)} new) -> {sdir}")


def prepare(cfg, args):
    data, scene, sdir, P = cfg_paths(cfg)
    model = os.path.join(sdir, "output_dir", "colmap")
    test = {f"{int(x)}.png" for x in open(os.path.join(sdir, "test.txt")).read().split()}
    r = cfg.get("rect", {})
    rigs = {"default": r.get("views", [{"yaw": a, "pitch": 0, "fov": 100, "name": f"h{a}"} for a in (0, 90, 180, 270)]
                       + [{"yaw": 0, "pitch": 90, "fov": 90, "name": "up"}, {"yaw": 0, "pitch": -90, "fov": 90, "name": "dn"}])}
    # the SfM model holds more frames than were released: keep the official train + test frames
    train = {f"{int(x)}.png" for x in open(os.path.join(sdir, "train.txt")).read().split()}
    _, imgs, _ = oc.read_model(model)
    tr = [n for n, *_ in imgs if n in train]; te = [n for n, *_ in imgs if n in test]
    frames = tr[: args.frames] + te[: args.frames] if args.frames else tr + te
    meta = oc.prepare(P, model, lambda n: os.path.join(sdir, "imgs", n), test, rigs, rect_scale=r.get("scale", 1.0),
                      rect_size=r.get("size"), ext=r.get("ext", ".png"), workers=args.workers, frames=frames)
    print(f"[ricoh360] {meta['n_frames']} frames, {len(meta['test'])} held out, median depth {meta.get('median_depth', 0):.3f}"
          f" (SfM units), valid fractions {meta['valid_fraction']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fetch", "prepare"))
    ap.add_argument("config")
    ap.add_argument("--frames", type=int, default=0, help="smoke test: only the first N train + N test frames")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = pu.load_config(args.config)
    {"fetch": fetch, "prepare": prepare}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
