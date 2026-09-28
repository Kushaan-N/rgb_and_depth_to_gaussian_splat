"""Accumulate a LiDAR bag into a dense metric world point cloud (CPU). Dataset-agnostic.

LiDAR gives dense, accurate, long-range geometry — a much stronger source for the collider, splat
init and floor fill than a ground-level RGB-D stream. Every scan is registered into the SAME Z-up
metric world frame the rest of the pipeline uses:

    p_world = T_world_cam(t) @ T_cam_lidar @ p_lidar      t = t_scan + offsets_s[<time_offset_key>]

Everything sensor-specific comes from the config's `lidar:` section, so any robot dataset works
with a config change only:

  lidar.bag              bag path relative to data_root (glob ok) — rosbag1 or rosbag2, auto-detected
  lidar.topic            sensor_msgs/PointCloud2 topic
  lidar.time_offset_key  REQUIRED key in timestamps.offsets_s (a missing offset is an error, never 0)
  lidar.extrinsic_chain  T_cam_lidar composed left-to-right from calibration.dir files
                         [{file, key, invert}]  (a single entry for rigs that publish T_cam_lidar)
  lidar.min_range_m / max_range_m / voxel_m

T_world_cam(t) is the pipeline's own pose chain (build_world_to_cam_track). If the config has no
`lidar:` section or the bag is absent, the stage prints SKIP and exits 0.

    python scripts/build_lidar_cloud.py --config configs/<seq>.yaml --out <out>/lidar_cloud.ply
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu

_PF_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}   # PointField.datatype


def load_T(path, key):
    import yaml
    return np.array(yaml.safe_load(open(path))[key], dtype=np.float64).reshape(4, 4)


def cam_from_lidar(chain, calib_dir):
    """Compose T_cam_lidar left-to-right from [{file, key, invert}] calibration entries."""
    T = np.eye(4)
    for step in chain:
        M = load_T(os.path.join(calib_dir, step["file"]), step["key"])
        T = T @ (np.linalg.inv(M) if step.get("invert") else M)
    return T


def read_xyz(msg, min_r, max_r):
    """Parse any PointCloud2 -> (N,3) float64 xyz, dropping invalid / out-of-range returns."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in "xyz"):
        raise ValueError("PointCloud2 has no x/y/z fields")
    end = ">" if msg.is_bigendian else "<"
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = raw.reshape(msg.height, msg.row_step)[:, : msg.width * msg.point_step]
    buf = np.ascontiguousarray(rows).reshape(-1, msg.point_step)

    def col(name):
        f = fields[name]; dt = np.dtype(end + _PF_DTYPE[f.datatype])
        return buf[:, f.offset: f.offset + dt.itemsize].copy().view(dt).ravel().astype(np.float64)

    x, y, z = col("x"), col("y"), col("z")
    r = np.sqrt(x * x + y * y + z * z)
    ok = np.isfinite(r) & (r > min_r) & (r < max_r)
    return np.stack([x[ok], y[ok], z[ok]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bag", default=None, help="override lidar.bag")
    ap.add_argument("--calib-dir", default=None, help="override calibration.dir")
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--min-range", type=float, default=None)
    ap.add_argument("--max-range", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan (smoke tests)")
    args = ap.parse_args()

    cfg = pu.load_config(args.config)
    lid = cfg.get("lidar")
    if not lid:
        print("[lidar] SKIP: config has no lidar section"); return 0
    root = cfg["sequence"]["data_root"]
    pattern = args.bag or lid.get("bag", "")
    hits = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(root, pattern)))
    if not hits:
        print(f"[lidar] SKIP: no bag matching {pattern!r} under {root}"); return 0
    bag = hits[0]

    key = lid.get("time_offset_key")
    offsets = (cfg.get("timestamps") or {}).get("offsets_s") or {}
    if not key or offsets.get(key) is None:
        print(f"[lidar] ERROR: timestamps.offsets_s.{key} is not set. Set it explicitly (0.0 if the "
              f"LiDAR is hardware-synced) — a silent default would misalign every scan.")
        return 2
    voff = float(offsets[key])
    calib_dir = args.calib_dir or (cfg.get("calibration") or {}).get("dir")
    T_cam_lidar = cam_from_lidar(lid["extrinsic_chain"], calib_dir)
    min_r = args.min_range if args.min_range is not None else float(lid.get("min_range_m", 0.5))
    max_r = args.max_range if args.max_range is not None else float(lid.get("max_range_m", 12.0))
    voxel = args.voxel if args.voxel is not None else float(lid.get("voxel_m", 0.02))
    print(f"[lidar] bag={os.path.basename(bag)} topic={lid['topic']} |t_cam_lidar|="
          f"{np.linalg.norm(T_cam_lidar[:3, 3]):.3f} m  offset={voff*1000:.3f} ms", flush=True)

    import open3d as o3d
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    stamps, scans = [], []
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.LATEST)) as r:
        conns = [c for c in r.connections if c.topic == lid["topic"]]
        if not conns:
            print(f"[lidar] ERROR: topic {lid['topic']} not in bag (topics: {sorted({c.topic for c in r.connections})})")
            return 2
        for i, (conn, _t, raw) in enumerate(r.messages(connections=conns)):
            if i % args.stride:
                continue
            m = r.deserialize(raw, conn.msgtype)
            stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 + voff)
            scans.append(read_xyz(m, min_r, max_r))
    stamps = np.array(stamps)
    print(f"[lidar] {len(scans)} scans; time [{stamps.min():.2f},{stamps.max():.2f}]", flush=True)

    mocap = pu.parse_mocap(os.path.join(root, cfg["sequence"]["pose_file"]), cfg)
    interp = pu.PoseInterpolator(mocap)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)
    inr = interp.in_range(stamps)
    print(f"[lidar] {int(inr.sum())}/{len(stamps)} scans within the pose time range", flush=True)
    if not inr.any():
        print("[lidar] ERROR: no scan overlaps the poses — check the time offset / units"); return 2
    Twc = pu.build_world_to_cam_track(mocap, interp, calib, cfg, stamps[inr])

    pts = []
    for k, i in enumerate(np.nonzero(inr)[0]):
        if len(scans[i]):
            T = Twc[k] @ T_cam_lidar
            pts.append(scans[i] @ T[:3, :3].T + T[:3, 3])
    W = np.concatenate(pts)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W)).voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, pc)
    V = np.asarray(pc.points)
    print(f"[lidar] wrote {args.out}: {len(V):,} pts ({len(W):,} raw, voxel {voxel} m), "
          f"bbox {V.min(0).round(2)}..{V.max(0).round(2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
