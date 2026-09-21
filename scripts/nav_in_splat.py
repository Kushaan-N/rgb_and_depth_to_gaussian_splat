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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="NuRec USDZ")
    ap.add_argument("--tum", required=True, help="TUM trajectory (camera-to-world, splat frame)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--camera", default="camera_0", help="authored camera name in the USDZ")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--warmup", type=int, default=300, help="RTPT accumulation ticks per frame")
    ap.add_argument("--view-index", type=int, default=0, help="TUM index for the FIXED viewing camera")
    ap.add_argument("--robot-start", type=int, default=6, help="first TUM index used as a robot position")
    ap.add_argument("--robot-stride", type=int, default=1)
    ap.add_argument("--robot-count", type=int, default=60)
    ap.add_argument("--robot-drop", type=float, default=0.25,
                    help="lower the robot from camera height toward the floor along the up axis (m)")
    ap.add_argument("--robot-size", type=float, default=0.25)
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
    view_pose = tum[min(args.view_index, len(tum) - 1)][1]

    # robot positions along the recorded path (camera centres), lowered toward the floor. The
    # trajectory is ~planar at constant height along one world axis (the up axis) — detect that
    # axis from the spread of camera centres and drop the robot along it.
    centres = np.array([p[1][:3] for p in tum])
    spread = centres.max(0) - centres.min(0)
    up_axis = int(np.argmin(spread))                 # smallest spread = height/up axis
    # floor is below the cameras along the up axis; subtract to drop the robot toward it. We
    # don't know the up SIGN a priori — pass a negative --robot-drop to flip if it goes up.
    print(f"[nav] up_axis={up_axis} spread={spread.round(3).tolist()} drop={args.robot_drop}", flush=True)

    idxs = [args.robot_start + i * args.robot_stride for i in range(args.robot_count)]
    idxs = [i for i in idxs if 0 <= i < len(tum)]

    # robot mesh + a dome light so it's lit (the gaussians carry their own radiance)
    UsdLux.DomeLight.Define(stage, "/World/navLight").CreateIntensityAttr(600.0)
    robot = UsdGeom.Cube.Define(stage, "/World/robot")
    robot.CreateSizeAttr(1.0)
    robot.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.45, 1.0)]))
    rx = UsdGeom.Xformable(robot.GetPrim()); rx.ClearXformOpOrder()
    r_t = rx.AddTranslateOp(); r_s = rx.AddScaleOp()
    r_s.Set(Gf.Vec3d(args.robot_size, args.robot_size, args.robot_size))

    targets = _resolve_camera_targets(stage, has_spg, {args.camera})
    if not targets:
        print(f"[nav] no render target for {args.camera}; cams available differ", flush=True)
        app.close(); return 2
    factory = RenderTargetFactory(has_spg, resolution=(W, H))
    cap = _open_camera_renderer(factory, stage, args.camera, app, targets[args.camera],
                                warmup_steps=args.warmup, force_identity_exposure=has_spg)

    import cv2
    n_written = 0
    for k, i in enumerate(idxs):
        pos = centres[i].copy()
        pos[up_axis] -= args.robot_drop                      # drop toward floor (negative to flip)
        r_t.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        rgb = cap.render_at_pose(view_pose)
        if rgb is None:
            print(f"[nav] frame {k} (tum {i}) produced no data", flush=True); continue
        bgr = cv2.cvtColor(np.asarray(rgb)[..., :3].astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output, f"nav_{k:04d}.png"), bgr)
        n_written += 1
        if k % 10 == 0:
            print(f"[nav] frame {k}/{len(idxs)} robot@{pos.round(3).tolist()} std={np.asarray(rgb)[...,:3].std():.1f}", flush=True)
    try:
        cap.close()
    except Exception:  # noqa: BLE001
        pass
    print(f"[nav] wrote {n_written}/{len(idxs)} frames to {args.output}", flush=True)
    app.close()
    return 0 if n_written else 2


if __name__ == "__main__":
    sys.exit(main())
