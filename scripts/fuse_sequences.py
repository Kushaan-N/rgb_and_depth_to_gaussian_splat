"""Fuse several already-prepped sequences of the same place into ONE training set (CPU).

Each member's prep already has COLMAP-refined poses (sharp) plus the Sim3 that maps them into its
metric frame, and a metric depth cloud. If the members were recorded in one world frame (e.g. the
same mocap room), fusing needs no joint SfM: every member's cameras go to the metric frame through
its own Sim3, and the remaining cross-recording offset is removed by ICP between the depth clouds
(member 0 is the reference; a large or poorly-fitting correction fails loudly instead of fusing
two different places). Output is a standard <out>/pipeline, so train/post run unchanged:

  dataset/     all frames of all members, metric frame (images/<member>/<name>)
  train_data/  training frames only (each member's held-out frames are excluded) + depth init
  sim3.json    identity (the fused model IS metric)
  lidar_cloud.ply / depth_cloud.ply   union of the member clouds (voxel-thinned)
  eval/<member>/{fused,single}  the member's held-out frames at its COLMAP poses, in the fused
               frame and in the member's own frame -> same frames, same pose source for both models
  members/<member>.json  fused->member transform (for per-member floor coverage) + alignment stats

    python scripts/fuse_sequences.py --config configs/fused_<name>.yaml
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from sim3_utils import load_sim3

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def member_config(name):
    for p in (name, os.path.join(REPO, "configs", f"{name}.yaml")):
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit(f"[fuse] no config for member '{name}'")


def cam_to_world(img):
    cfw = img.cam_from_world
    M = np.asarray((cfw() if callable(cfw) else cfw).matrix())
    Rcw, tcw = M[:, :3], M[:, 3]
    return Rcw.T, -Rcw.T @ tcw                                   # R_wc, camera centre


def to_reference(Rwc, C, s, R, t, T):
    """A member camera (COLMAP frame) -> member metric frame (Sim3 s,R,t) -> reference frame (rigid T)."""
    A, b = T[:3, :3], T[:3, 3]
    return A @ R @ Rwc, A @ (s * R @ C + t) + b


def icp_align(src_path, ref_path, voxel, max_m, max_deg, min_fitness):
    """Rigid src->ref correction from the depth clouds, coarse-to-fine point-to-plane ICP."""
    import open3d as o3d
    src, ref = o3d.io.read_point_cloud(src_path), o3d.io.read_point_cloud(ref_path)
    T = np.eye(4)
    for v, corr in ((4 * voxel, 12 * voxel), (voxel, 3 * voxel)):
        s, r = src.voxel_down_sample(v), ref.voxel_down_sample(v)
        for pc in (s, r):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * v, max_nn=30))
        res = o3d.pipelines.registration.registration_icp(
            s, r, corr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = res.transformation
    s = src.voxel_down_sample(voxel)
    d0 = np.asarray(s.compute_point_cloud_distance(ref))
    d1 = np.asarray(s.transform(T).compute_point_cloud_distance(ref))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    st = {"fitness": round(float(res.fitness), 3), "rmse_m": round(float(res.inlier_rmse), 4),
          "translation_m": round(float(np.linalg.norm(T[:3, 3])), 4), "rotation_deg": round(ang, 3),
          "nn_median_before_m": round(float(np.median(d0)), 4), "nn_median_after_m": round(float(np.median(d1)), 4)}
    if st["translation_m"] > max_m or ang > max_deg or res.fitness < min_fitness:
        raise SystemExit(f"[fuse] {src_path} does not align with the reference ({st}); members must share "
                         f"a world frame (limits: {max_m} m, {max_deg} deg, fitness >= {min_fitness})")
    return T, st


def write_model(d, cams, imgs, points3d=None):
    """COLMAP text model: cams {id: (model, w, h, params)}, imgs [(name, cam_id, R_wc, C)]."""
    from scipy.spatial.transform import Rotation
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cameras.txt"), "w") as f:
        for cid, (model, w, h, prm) in sorted(cams.items()):
            f.write(f"{cid} {model} {w} {h} " + " ".join(f"{p:.10g}" for p in prm) + "\n")
    with open(os.path.join(d, "images.txt"), "w") as f:
        for i, (name, cid, Rwc, C) in enumerate(sorted(imgs, key=lambda x: x[0]), 1):
            Rcw = Rwc.T; tcw = -Rcw @ C
            q = Rotation.from_matrix(Rcw).as_quat()                # x,y,z,w
            f.write(f"{i} {q[3]:.12g} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} "
                    f"{tcw[0]:.12g} {tcw[1]:.12g} {tcw[2]:.12g} {cid} {name}\n\n")
    if points3d is None:
        open(os.path.join(d, "points3D.txt"), "w").close()


def link_images(dst_dir, src_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    for n in names:
        dst = os.path.join(dst_dir, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src_dir, n), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import pycolmap, open3d as o3d

    cfg = pu.load_config(args.config)
    fz = cfg.get("fuse") or {}
    members = fz.get("members") or []
    if len(members) < 2:
        raise SystemExit("[fuse] fuse.members needs at least two sequences")
    P = os.path.join(cfg["paths"]["out_root"], "pipeline")
    interval = int(fz.get("holdout_interval", 8))            # = the trainer's test_split_interval
    align = fz.get("align", "icp")
    os.makedirs(os.path.join(P, "members"), exist_ok=True)

    cams, all_imgs, train_imgs, clouds, info = {}, [], [], [], {}
    ref_cloud = None
    for k, m in enumerate(members):
        mcfg = pu.load_config(member_config(m))
        seq = mcfg["sequence"]["name"]; MP = os.path.join(mcfg["paths"]["out_root"], "pipeline")
        rec = pycolmap.Reconstruction(os.path.join(MP, "dataset", "sparse", "0"))
        s, R, t = load_sim3(os.path.join(MP, "sim3.json"))
        cloud = os.path.join(MP, "depth_cloud.ply")
        cloud = cloud if os.path.exists(cloud) else None

        T, st = np.eye(4), {"reference": True} if k == 0 else {"method": "none"}
        if k == 0:
            ref_cloud = cloud
        elif align == "icp" and cloud and ref_cloud:
            T, st = icp_align(cloud, ref_cloud, float(fz.get("icp_voxel_m", 0.05)),
                              float(fz.get("max_align_m", 0.3)), float(fz.get("max_align_deg", 5.0)),
                              float(fz.get("min_icp_fitness", 0.5)))
        print(f"[fuse] {seq}: {rec.num_reg_images()} frames  align={st}", flush=True)

        cid_map = {}
        for cid, c in sorted(rec.cameras.items()):
            cid_map[cid] = len(cams) + 1
            cams[cid_map[cid]] = (c.model.name, c.width, c.height, [float(x) for x in c.params])
        names = sorted(i.name for i in rec.images.values())
        val = set(names[::interval])                          # the trainer's held-out frames for this member
        single_val, fused_val = [], []
        for img in rec.images.values():
            Rwc, C = cam_to_world(img)
            Rwc_m, C_m = to_reference(Rwc, C, s, R, t, T)
            e = (f"{seq}/{img.name}", cid_map[img.camera_id], Rwc_m, C_m)
            all_imgs.append(e)
            if img.name in val:
                fused_val.append((img.name, e[1], Rwc_m, C_m)); single_val.append((img.name, img.camera_id, Rwc, C))
            else:
                train_imgs.append(e)
        link_images(os.path.join(P, "dataset", "images", seq), os.path.join(MP, "dataset", "images"), names)
        for kind, lst, cm in (("fused", fused_val, {cid_map[c]: cams[cid_map[c]] for c in rec.cameras}),
                              ("single", single_val, {c: (v.model.name, v.width, v.height, [float(x) for x in v.params])
                                                      for c, v in rec.cameras.items()})):
            d = os.path.join(P, "eval", seq, kind)
            write_model(os.path.join(d, "sparse", "0"), cm, lst)
            link_images(os.path.join(d, "images"), os.path.join(MP, "dataset", "images"), [x[0] for x in lst])
        if cloud:
            pc = o3d.io.read_point_cloud(cloud); pc.transform(T); clouds.append(pc)
        # fused(reference) -> member metric frame, in sim3.json form (floor coverage per member)
        inv = np.linalg.inv(T)
        info[seq] = {"config": member_config(m), "pipeline": MP, "frames": len(names), "held_out": len(val),
                     "alignment": st, "ckpt_glob": os.path.join(MP, "train", "*", "*", "ckpt_last.pt")}
        json.dump({"scale": 1.0, "R": inv[:3, :3].tolist(), "t": inv[:3, 3].tolist(), **info[seq]},
                  open(os.path.join(P, "members", f"{seq}.json"), "w"), indent=2)

    write_model(os.path.join(P, "dataset", "sparse", "0"), cams, all_imgs)
    tdir = os.path.join(P, "train_data", "sparse", "0")
    write_model(tdir, cams, train_imgs, points3d=True)          # points3D.txt comes from the depth init
    tim = os.path.join(P, "train_data", "images")
    if not os.path.lexists(tim):
        os.symlink(os.path.join(P, "dataset", "images"), tim)
    json.dump({"scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "n_frames": len(all_imgs),
               "residual_mean_m": 0.0, "residual_p95_m": 0.0, "note": "fused dataset is already metric"},
              open(os.path.join(P, "sim3.json"), "w"), indent=2)
    if clouds:
        fused = clouds[0]
        for pc in clouds[1:]:
            fused += pc
        n0 = len(fused.points)
        fused = fused.voxel_down_sample(float(fz.get("cloud_voxel_m", 0.01)))
        o3d.io.write_point_cloud(os.path.join(P, "lidar_cloud.ply"), fused)
        print(f"[fuse] depth cloud: {n0:,} -> {len(fused.points):,} points", flush=True)
    json.dump({"members": info, "frames_total": len(all_imgs), "frames_train": len(train_imgs)},
              open(os.path.join(P, "fuse.json"), "w"), indent=2)
    print(f"[fuse] {len(all_imgs)} frames ({len(train_imgs)} train) from {len(members)} members -> {P}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
