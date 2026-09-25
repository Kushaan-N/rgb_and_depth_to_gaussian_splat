"""Accumulate the Velodyne LiDAR into a dense metric world cloud (CPU).

CEAR ships a Velodyne VLP-16 bag (/velodyne_points, ~10 Hz) that the RGB+depth pipeline
deliberately ignored. LiDAR gives dense, accurate, long-range 3D geometry — a much stronger
source for the physics collider than the ground-level RGB-D depth (which left floor holes /
coverage gaps). This registers every scan into the SAME Z-up metric world frame the rest of the
pipeline uses:

    p_world = T_world_rgb(t) @ T_rgb_lidar @ p_lidar
    T_rgb_lidar = inv(T_event_rgb) @ T_event_lidar          (from the CEAR extrinsics)
    t_event     = t_scan + offsets_s.velodyne               (temporal sync)

T_world_rgb(t) is the existing mocap pose chain (build_world_to_cam_track), so the LiDAR cloud
lands co-registered with the splat, the depth cloud, and the collider frame.

    python scripts/build_lidar_cloud.py --config configs/mocap2_well-lit_trot.yaml \
        --out $CEAR_OUT/mocap2_well-lit_trot/lidar_cloud.ply
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu
from build_poses import compute_world_cam  # noqa: F401  (imports the shared pose machinery)


def load_T(path, key):
    import yaml
    return np.array(yaml.safe_load(open(path))[key], dtype=np.float64)


def rgb_from_lidar(calib_dir):
    """T_rgb_lidar = inv(T_event_rgb) @ T_event_lidar  (a point in LiDAR frame -> RGB frame)."""
    T_event_rgb = load_T(os.path.join(calib_dir, "event_rgb_extrinsic"), "T_event_rgb")
    T_event_lidar = load_T(os.path.join(calib_dir, "event_lidar_extrinsic"), "T_event_lidar")
    return np.linalg.inv(T_event_rgb) @ T_event_lidar


def read_scan_xyz(msg, min_r, max_r):
    """Parse PointCloud2 -> (N,3) float32 xyz, dropping invalid / out-of-range returns."""
    off = {f.name: f.offset for f in msg.fields}
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    def col(name):
        o = off[name]
        return buf[:, o:o + 4].copy().view("<f4").ravel()
    x, y, z = col("x"), col("y"), col("z")
    r = np.sqrt(x * x + y * y + z * z)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (r > min_r) & (r < max_r)
    return np.stack([x[ok], y[ok], z[ok]], axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dir", default=None, help="dir with event_rgb/event_lidar extrinsics (default: <data>/_calib or <data>/../_calib)")
    ap.add_argument("--voxel", type=float, default=0.02, help="final voxel-downsample size (m)")
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--max-range", type=float, default=15.0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth scan")
    args = ap.parse_args()

    import open3d as o3d
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)

    cfg = pu.load_config(args.config)
    seq = cfg["sequence"]; root = seq["data_root"]
    calib_dir = args.calib_dir or (os.path.join(os.path.dirname(root), "_calib"))
    T_rgb_lidar = rgb_from_lidar(calib_dir)
    voff = float(cfg.get("timestamps", {}).get("offsets_s", {}).get("velodyne", 0.0))
    print(f"[lidar] T_rgb_lidar |t|={np.linalg.norm(T_rgb_lidar[:3,3]):.3f} m ; velodyne offset {voff*1000:.3f} ms", flush=True)

    mocap = pu.parse_mocap(os.path.join(root, seq["pose_file"]), cfg)
    interp = pu.PoseInterpolator(mocap)
    calib = pu.load_calibration(cfg["intrinsics"]["calib_file"], cfg)

    bag = os.path.join(root, "lidar.bag")
    stamps, scans = [], []
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == "/velodyne_points"]
        n = 0
        for conn, _t, raw in r.messages(connections=conns):
            n += 1
            if (n - 1) % args.stride:
                continue
            m = ts.deserialize_ros1(raw, conn.msgtype)
            t_scan = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            stamps.append(t_scan + voff)
            scans.append(read_scan_xyz(m, args.min_range, args.max_range))
    stamps = np.array(stamps)
    print(f"[lidar] {len(scans)} scans read; time [{stamps.min():.2f},{stamps.max():.2f}]", flush=True)

    inr = interp.in_range(stamps)
    print(f"[lidar] {inr.sum()}/{len(stamps)} scans within mocap pose range", flush=True)
    Twr = pu.build_world_to_cam_track(mocap, interp, calib, cfg, stamps[inr])  # T_world_rgb per used scan

    used = np.where(inr)[0]
    allpts = []
    for k, i in enumerate(used):
        P = scans[i]
        if not len(P):
            continue
        T = Twr[k] @ T_rgb_lidar                      # T_world_lidar
        Ph = P @ T[:3, :3].T + T[:3, 3]
        allpts.append(Ph)
    W = np.concatenate(allpts, axis=0)
    print(f"[lidar] {len(W):,} world points before downsample; bbox {W.min(0).round(2)}..{W.max(0).round(2)}", flush=True)

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W))
    pc = pc.voxel_down_sample(args.voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    o3d.io.write_point_cloud(args.out, pc)
    V = np.asarray(pc.points)
    print(f"[lidar] wrote {args.out}: {len(V):,} pts (voxel {args.voxel} m), bbox {V.min(0).round(2)}..{V.max(0).round(2)}", flush=True)


if __name__ == "__main__":
    main()
