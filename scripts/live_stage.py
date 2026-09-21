#!/usr/bin/env python3
"""Launch a LIVE, navigable Isaac Sim session on the NuRec splat world (WebRTC streaming).

Boots Isaac Sim headless with the WebRTC livestream extension, opens the splat, applies the
NuRec render setup, drops in a lit robot, and idles — so you can fly the camera / look around
the reconstructed world in real time from your browser. The camera is yours (mouse), not a
scripted path.

    ./python.sh live_stage.py --stage scene_nurec.usdz
Connect: forward the node's port 8211 over SSH, then open the Isaac WebRTC client (see RUN.md).
"""
from __future__ import annotations
import argparse, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--robot-size", type=float, default=0.25)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    # headless + WebRTC livestream: the stream IS the display. NuRec render setup as usual.
    app = SimulationApp({"headless": True, "multi_gpu": False, "extra_args": [
        "--enable", "omni.kit.livestream.webrtc",
        "--enable", "isaacsim.replicator.nurec_utils",
        "--/app/livestream/enabled=true",
        "--/app/window/drawMouse=true",
        "--/renderer/multiGpu/enabled=false",
    ]})

    import numpy as np  # noqa: F401
    from pxr import UsdGeom, UsdLux, Gf, Vt
    from isaacsim.replicator.nurec_utils.usd_utils import open_stage
    from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering, enable_omni_rtx_spg

    app.update()
    try:
        enable_omni_rtx_spg(app)
    except Exception as e:  # noqa: BLE001
        print(f"[live] enable_omni_rtx_spg: {e}", flush=True)
    stage = open_stage(args.stage)
    if stage is None:
        print("[live] failed to open stage", flush=True); app.close(); return 2
    ok, _, has_spg, _ = setup_for_rendering(stage)
    print(f"[live] setup_for_rendering ok={ok} has_spg={has_spg}", flush=True)

    UsdLux.DomeLight.Define(stage, "/World/navLight").CreateIntensityAttr(600.0)
    robot = UsdGeom.Cube.Define(stage, "/World/robot")
    robot.CreateSizeAttr(1.0)
    robot.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.45, 1.0)]))
    rx = UsdGeom.Xformable(robot.GetPrim()); rx.ClearXformOpOrder()
    rx.AddScaleOp().Set(Gf.Vec3d(args.robot_size, args.robot_size, args.robot_size))

    print("[live] STREAMING — connect the Isaac WebRTC client to <node>:8211 (see RUN.md). "
          "Ctrl-C / job end to stop.", flush=True)
    try:
        while app.is_running():
            app.update()
    except KeyboardInterrupt:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
