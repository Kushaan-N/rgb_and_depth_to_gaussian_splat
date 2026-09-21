#!/usr/bin/env python3
"""Render a robot navigating the photoreal NuRec splat world (Isaac Sim 6.x, in-container).

This reuses NVIDIA's OWN tested NuRec render machinery (open_stage -> setup_for_rendering ->
RenderTargetFactory / CameraRenderer.render_at_pose, from isaacsim.replicator.nurec_utils)
and injects a moving robot mesh into the opened stage. Everything is in the splat's own
(COLMAP) frame, so the robot is placed along the RECORDED camera trajectory — guaranteed to be
in the reconstructed region and on the drivable path. A fixed viewing camera (an early recorded
pose) watches the robot drive away down the path.

Why not compose_stage.py's World+collider here: the collider is in a different (Z-up metric)
frame than the splat (COLMAP), and nesting the USDZ under a prim breaks its authored
RenderProduct->camera wiring. Physics validity (Gate 5, rover drive) is proven separately in
compose_stage.py; this script is the photoreal VISUAL of navigation.

    ./python.sh nav_in_splat.py --stage scene_nurec.usdz --tum walkthrough.tum \
        --output <dir> --warmup 300 --view-index 0 --robot-start 6 --robot-drop 0.25
"""
from __future__ import annotations

import argparse
import os
import sys


def read_tum(path):
    """Read a TUM file -> list of (ts, [tx,ty,tz,qx,qy,qz,qw])."""
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 8:
            continue
        out.append((p[0], [float(x) for x in p[1:8]]))
    return out


def look_at_pose(cam, target, up):
    """Build a TUM pose [tx,ty,tz,qx,qy,qz,qw] for a USD camera at `cam` looking at `target`.

    USD cameras look down local -Z with +Y up. We solve the look-at rotation directly (no
    OpenCV/COLMAP convention guessing), so the camera reliably frames `target`. This is what
    fixes the earlier half-black views + invisible robot: aim along the path where coverage and
    the robot are, instead of trusting the raw recorded quaternion.
    """
    import numpy as np
    f = np.asarray(target, float) - np.asarray(cam, float)
    f = f / (np.linalg.norm(f) + 1e-9)                 # forward: camera -Z points here
    up = np.asarray(up, float)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:                        # forward ~parallel to up
        up = np.array([1.0, 0.0, 0.0]); r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f); u = u / (np.linalg.norm(u) + 1e-9)
    R = np.column_stack([r, u, -f])                    # cam-to-world: X=right, Y=up, Z=-forward
    t = np.trace(R)
    if t > 0:
        s = (t + 1.0) ** 0.5 * 2; qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s; qy = (R[0, 2] - R[2, 0]) / s; qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2; qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2; qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2; qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    c = np.asarray(cam, float)
    return [float(c[0]), float(c[1]), float(c[2]), float(qx), float(qy), float(qz), float(qw)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="NuRec USDZ")
    ap.add_argument("--tum", required=True, help="TUM trajectory (camera-to-world, splat frame)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--camera", default="camera_0", help="authored camera name in the USDZ")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--warmup", type=int, default=300, help="RTPT accumulation ticks per frame")
    ap.add_argument("--cam-index", type=int, default=8, help="TUM index for the FIXED viewing camera position")
    ap.add_argument("--look-ahead", type=int, default=45,
                    help="TUM index offset the camera AIMS at (down the path, well-covered forward view)")
    ap.add_argument("--up-sign", type=float, default=1.0, help="sign of the world up axis (flip to -1 if upside down)")
    ap.add_argument("--robot-start", type=int, default=4, help="robot's first TUM index offset from cam-index")
    ap.add_argument("--robot-stride", type=int, default=1)
    ap.add_argument("--robot-count", type=int, default=60)
    ap.add_argument("--robot-drop", type=float, default=0.25,
                    help="lower the robot from camera height toward the floor along the up axis (m)")
    ap.add_argument("--robot-size", type=float, default=0.25)
    ap.add_argument("--robot-mode", choices=["sweep", "path"], default="sweep",
                    help="sweep: foreground lateral pass (stays visible); path: drive along the trajectory")
    ap.add_argument("--sweep-dist", type=float, default=1.3, help="robot distance in front of the camera (m)")
    ap.add_argument("--sweep-range", type=float, default=0.9, help="half-width of the lateral sweep (m)")
    ap.add_argument("--cam-mode", choices=["fixed", "follow"], default="fixed",
                    help="fixed: stationary look-at camera; follow: camera moves along the path (upright "
                         "walkthrough via look-at — avoids the ~90deg roll of the raw recorded quaternions)")
    ap.add_argument("--cam-stride", type=int, default=2, help="path steps per frame in follow mode")
    ap.add_argument("--frames", type=int, default=60, help="number of frames in follow mode")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    W, H = (int(x) for x in args.resolution.lower().split("x"))

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False,
                         "extra_args": ["--enable", "isaacsim.replicator.nurec_utils",
                                        "--/renderer/multiGpu/enabled=false"]})

    import numpy as np
    import carb  # noqa: F401
    from pxr import UsdGeom, UsdLux, Gf, Vt
    from isaacsim.replicator.nurec_utils.usd_utils import open_stage  # returns the Stage (core.utils one returns bool)
    from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering, enable_omni_rtx_spg
    from isaacsim.replicator.nurec_utils.render import (
        RenderTargetFactory, _resolve_camera_targets, _open_camera_renderer,
    )

    app.update()
    try:
        enable_omni_rtx_spg(app)
    except Exception as e:  # noqa: BLE001
        print(f"[nav] enable_omni_rtx_spg: {e}", flush=True)

    stage = open_stage(args.stage)
    if stage is None:
        print("[nav] failed to open stage", flush=True); app.close(); return 2
    ok, _, has_spg, _ = setup_for_rendering(stage)
    print(f"[nav] setup_for_rendering ok={ok} has_spg={has_spg}", flush=True)
    app.update()  # must be after setup, before first render tick

    tum = read_tum(args.tum)
    if not tum:
        print("[nav] empty TUM", flush=True); app.close(); return 2

    # camera centres along the recorded path; the path is ~planar at constant height along one
    # world axis (the up axis) — detect it from the spread of centres.
    centres = np.array([p[1][:3] for p in tum])
    spread = centres.max(0) - centres.min(0)
    up_axis = int(np.argmin(spread))                 # smallest spread = height/up axis
    up_vec = np.zeros(3); up_vec[up_axis] = args.up_sign

    # FIXED viewing camera: sit at a recorded position and AIM along the path (look-at), so the
    # view faces the well-covered forward direction and frames the robot. Not the raw recorded
    # quaternion (that gave half-black views + an off-screen robot).
    ci = max(0, min(args.cam_index, len(tum) - 1))
    li = max(0, min(ci + args.look_ahead, len(tum) - 1))
    cam_pos = centres[ci].copy()
    look_target = centres[li].copy(); look_target[up_axis] -= args.robot_drop  # aim at floor-ish, down the path
    view_pose = look_at_pose(cam_pos, look_target, up_vec)
    print(f"[nav] up_axis={up_axis} up_sign={args.up_sign} cam_idx={ci} look_idx={li} "
          f"cam@{cam_pos.round(2).tolist()} aim@{look_target.round(2).tolist()}", flush=True)

    # Robot path. mocap1 is a dense foliage volume (no open corridor), so driving "forward" sends
    # the robot behind the gaussians. Default: a FOREGROUND lateral sweep — hold the robot a fixed
    # distance in front of the camera and move it across the view, staying visible against the
    # photoreal backdrop. (`--robot-mode path` drives along the recorded trajectory instead.)
    front = look_target - cam_pos; front = front / (np.linalg.norm(front) + 1e-9)
    right = np.cross(front, up_vec); right = right / (np.linalg.norm(right) + 1e-9)
    def sweep_pos(k, n):
        s = (k / max(n - 1, 1)) * 2.0 - 1.0                       # -1 .. +1
        return (cam_pos + front * args.sweep_dist
                + right * (s * args.sweep_range) - up_vec * args.robot_drop)
    idxs = [ci + args.robot_start + i * args.robot_stride for i in range(args.robot_count)]
    idxs = [i for i in idxs if 0 <= i < len(tum)]

    # dome light so meshes are lit (the gaussians carry their own radiance)
    UsdLux.DomeLight.Define(stage, "/World/navLight").CreateIntensityAttr(600.0)
    r_t = None
    if args.robot_count > 0:                          # robot optional (0 = clean walkthrough)
        robot = UsdGeom.Cube.Define(stage, "/World/robot")
        robot.CreateSizeAttr(1.0)
        robot.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.45, 1.0)]))
        rx = UsdGeom.Xformable(robot.GetPrim()); rx.ClearXformOpOrder()
        r_t = rx.AddTranslateOp()
        rx.AddScaleOp().Set(Gf.Vec3d(args.robot_size, args.robot_size, args.robot_size))

    targets = _resolve_camera_targets(stage, has_spg, {args.camera})
    if not targets:
        print(f"[nav] no render target for {args.camera}; cams available differ", flush=True)
        app.close(); return 2
    factory = RenderTargetFactory(has_spg, resolution=(W, H))
    cap = _open_camera_renderer(factory, stage, args.camera, app, targets[args.camera],
                                warmup_steps=args.warmup, force_identity_exposure=has_spg)

    import cv2
    n_written = 0
    # frame count: follow mode walks the path; fixed mode steps through the robot positions
    N = args.frames if args.cam_mode == "follow" else len(idxs)
    for k in range(N):
        if args.cam_mode == "follow":
            # UPRIGHT walkthrough: camera moves along the path, look-at the point ahead (up=world-up)
            c = min(ci + k * args.cam_stride, len(tum) - 1)
            t = min(c + args.look_ahead, len(tum) - 1)
            tgt = centres[t].copy(); tgt[up_axis] -= args.robot_drop
            pose = look_at_pose(centres[c], tgt, up_vec)
        else:
            pose = view_pose
        if r_t is not None:
            if args.robot_mode == "sweep" and args.cam_mode == "fixed":
                pos = sweep_pos(k, N)
            else:
                i = idxs[min(k, len(idxs) - 1)] if idxs else ci
                pos = centres[i].copy(); pos[up_axis] -= args.robot_drop
            r_t.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        rgb = cap.render_at_pose(pose)
        if rgb is None:
            print(f"[nav] frame {k} produced no data", flush=True); continue
        bgr = cv2.cvtColor(np.asarray(rgb)[..., :3].astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output, f"nav_{k:04d}.png"), bgr)
        n_written += 1
        if k % 10 == 0:
            print(f"[nav] frame {k}/{N} std={np.asarray(rgb)[...,:3].std():.1f}", flush=True)
    try:
        cap.close()
    except Exception:  # noqa: BLE001
        pass
    print(f"[nav] wrote {n_written}/{len(idxs)} frames to {args.output}", flush=True)
    app.close()
    return 0 if n_written else 2


if __name__ == "__main__":
    sys.exit(main())
