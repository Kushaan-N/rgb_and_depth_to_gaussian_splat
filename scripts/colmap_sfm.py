"""COLMAP SfM on the gated frames -> a 3DGRUT dataset, for the GT-vs-COLMAP fidelity experiment.

Baseline uses the provided intrinsics + OptiTrack GT poses (no optimization). This runs COLMAP
structure-from-motion on the SAME gated frames: it estimates poses and *refines* the intrinsics via
bundle adjustment (the "optimization" arm). Training a splat on this and comparing held-out PSNR to
the GT baseline answers empirically whether COLMAP optimization beats using the intrinsics + GT.

Undistorted PINHOLE frames in, known K as the mapper's init. Writes <out>/sparse/0 + images/ so
3DGRUT can train on it directly.

    python scripts/colmap_sfm.py --config configs/mocap2_well-lit_trot.yaml --out <out>/dataset_colmap
"""
from __future__ import annotations
import argparse, os, sys, shutil, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--refine-intrinsics", type=int, default=1, help="1=let BA refine focal/pp (the optimization arm)")
    args = ap.parse_args()
    import pycolmap

    cfg = pu.load_config(args.config)
    root = cfg["sequence"]["data_root"]
    out_root = cfg["paths"]["out_root"]
    precond = os.path.join(out_root, "precond", cfg["sequence"].get("rgb_dir", "rgb"))
    # gated set used by the baseline = train_frames.txt + val_frames.txt (same frames -> fair compare)
    names = []
    for f in ["train_frames.txt", "val_frames.txt"]:
        p = os.path.join(out_root, "gating", f)
        if os.path.exists(p):
            names += [ln.strip() for ln in open(p) if ln.strip()]
    names = sorted(set(os.path.basename(n) for n in names))
    print(f"[colmap] {len(names)} gated frames; image dir {precond}", flush=True)

    # known intrinsics as init
    import yaml
    K = np.array(yaml.safe_load(open(cfg["intrinsics"]["calib_file"]))["K"], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    work = args.out + "_work"
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "database.db")
    if os.path.exists(db):
        os.remove(db)
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"
    ro.camera_params = f"{fx},{fy},{cx},{cy}"
    print("[colmap] extract_features (CPU SIFT)...", flush=True)
    pycolmap.extract_features(db, precond, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, device=pycolmap.Device.cpu)
    print(f"[colmap] match ({args.matcher})...", flush=True)
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_sequential(db, device=pycolmap.Device.cpu)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = bool(args.refine_intrinsics)
    opts.ba_refine_principal_point = bool(args.refine_intrinsics)
    sparse = os.path.join(work, "sparse")
    os.makedirs(sparse, exist_ok=True)
    print("[colmap] incremental_mapping...", flush=True)
    recs = pycolmap.incremental_mapping(db, precond, sparse, options=opts)
    if not recs:
        print("[colmap] FAILED: no reconstruction produced"); return 2
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"[colmap] {len(recs)} model(s); best registers {best.num_reg_images()}/{len(names)} images, "
          f"{best.num_points3D()} points", flush=True)
    cam = list(best.cameras.values())[0]
    print(f"[colmap] refined camera: {cam.model.name} params={np.round(cam.params,2).tolist()}  "
          f"(init fx,fy,cx,cy={fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})", flush=True)

    # assemble a 3DGRUT dataset: sparse/0 (model) + images/ (symlinks to the gated frames)
    os.makedirs(os.path.join(args.out, "sparse", "0"), exist_ok=True)
    best.write(os.path.join(args.out, "sparse", "0"))
    img_out = os.path.join(args.out, "images")
    os.makedirs(img_out, exist_ok=True)
    for n in names:
        dst = os.path.join(img_out, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(precond, n), dst)
    print(f"[colmap] wrote 3DGRUT dataset -> {args.out}  (sparse/0 + {len(names)} images)", flush=True)
    print(f"COLMAP_SFM_DONE reg={best.num_reg_images()}/{len(names)}")


if __name__ == "__main__":
    main()
