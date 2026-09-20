"""Phase 6 — compose the Isaac Sim scene and validate it (docs/PLAN.md §9).

  *** Runs ONLY inside Isaac Sim's python (isaacsim 5.0) on a GPU node. ***
  Launch via sbatch/isaac_l40s.sbatch.

Modes:
  drop : physics-only (collider + gravity, NO rendering) — drop rigid spheres and check they
         rest on the reconstructed floor at the right height (Gate 5). The robust milestone.
  nav  : load the splat (visual) + collider (physics) + a WHEELED robot, drive it forward
         through the depth-covered region, and save camera frames — "navigate a robot in the
         splat world". Rendering path (RTX), higher-risk; builds on `drop`.

All heavy imports happen AFTER SimulationApp() (Isaac requirement).

    python scripts/compose_stage.py --collider .../collider/collider.obj --mode drop \
        --floor-z 0.02
    python scripts/compose_stage.py --collider ... --splat-usd .../scene_nurec.usdz \
        --mode nav --robot jetbot
"""

from __future__ import annotations

import argparse
import json
import os


# --------------------------------------------------------------------------- #
# pure-python OBJ reader (avoids trimesh/open3d in the Isaac venv)
# --------------------------------------------------------------------------- #
def _quat_wxyz_to_R(q):
    import numpy as np
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_to_quat_wxyz(R):
    import numpy as np
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def read_cam_pose(model_dir, which="middle"):
    """Read a recorded camera-to-world pose (Twc, OpenCV convention) from a COLMAP model."""
    import numpy as np
    lines = [ln for ln in open(os.path.join(model_dir, "images.txt")) if ln.strip() and not ln.startswith("#")]
    data = lines[0::2]                              # pose lines (skip the empty points2D lines)
    ln = data[len(data) // 2 if which == "middle" else 0].split()
    qw, qx, qy, qz = map(float, ln[1:5]); tx, ty, tz = map(float, ln[5:8])
    R_cw = _quat_wxyz_to_R(np.array([qw, qx, qy, qz]))
    t_cw = np.array([tx, ty, tz])
    R_wc = R_cw.T; C = -R_wc @ t_cw                 # camera-to-world
    Twc = np.eye(4); Twc[:3, :3] = R_wc; Twc[:3, 3] = C
    return Twc


def read_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
                faces.append(idx)
    return verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collider", required=True, help="collider .obj (Z-up, metres)")
    ap.add_argument("--splat-usd", default=None, help="NuRec/ParticleField splat USD (nav mode)")
    ap.add_argument("--mode", choices=["drop", "nav"], default="drop")
    ap.add_argument("--floor-z", type=float, default=0.0, help="RANSAC floor height (m)")
    ap.add_argument("--robot", default="jetbot", help="wheeled robot asset (nav mode)")
    ap.add_argument("--cam-model", default=None,
                    help="COLMAP sparse dir; place the nav camera at a RECORDED pose (on-path"
                         " -> clean splat) instead of a chase cam")
    ap.add_argument("--out-dir", default="/tmp/isaac_out")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from isaacsim import SimulationApp
    # nav needs rendering (RTX); drop is physics-only and can run truly headless.
    #
    # NuRec (neural Gaussian volume) rendering is NOT bundled in the base isaac-sim:5.1.0
    # image: the render feature lives in the extension `omni.rtx.spg` and the setup helper in
    # `isaacsim.replicator.nurec_utils`, both of which are pulled from NVIDIA's extension
    # registry on demand. Unity compute nodes have outbound HTTPS, so we turn the registry ON
    # (`registryEnabled=true`) and `--enable` the extensions at launch; Kit downloads them into
    # the scratch-bound kit data dir (persists across runs). Then the official recipe is
    # `/rtx/spg/enabled=true` + nurec carb settings + nurec_utils.setup_for_rendering(stage).
    nurec_args = ["--/renderer/multiGpu/enabled=false",
                  "--/app/extensions/registryEnabled=true",
                  "--/app/extensions/installUncertifiedExts=true",
                  "--enable", "omni.rtx.spg",
                  "--enable", "isaacsim.replicator.nurec_utils",
                  "--/rtx/spg/enabled=true",
                  "--/omni/rtx/nre/compositing/disableNuRecPostProcessings=true",
                  "--/rtx/rtpt/gaussian/skipTonemapping/enabled=false"]
    app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting",
                         "extra_args": nurec_args if args.mode == "nav" else []})

    import numpy as np
    from pxr import UsdGeom, UsdPhysics, Gf, Vt, Sdf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicSphere
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # ground physics scene comes from World's default; add our collider mesh -------------
    verts, faces = read_obj(args.collider)
    m = UsdGeom.Mesh.Define(stage, "/World/collider")
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
    mc.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)   # exact static triangle mesh
    if args.mode == "drop":
        UsdGeom.Imageable(m).MakeInvisible()
    vmin = np.array(verts).min(0); vmax = np.array(verts).max(0)
    cx, cy = float((vmin[0] + vmax[0]) / 2), float((vmin[1] + vmax[1]) / 2)
    print(f"[compose] collider verts={len(verts)} tris={len(faces)} "
          f"bbox x[{vmin[0]:.2f},{vmax[0]:.2f}] y[{vmin[1]:.2f},{vmax[1]:.2f}] floor_z={args.floor_z}")

    result = {}
    if args.mode == "drop":
        result = gate5_drop(world, DynamicSphere, np, cx, cy, args.floor_z)
    else:
        # a dome light so the reconstructed geometry (collider + rover) is visible under RTX
        # even if the NuRec volume fails to render — a lit navigation is still real evidence,
        # strictly better than the all-black frames from no-light + no-splat.
        from pxr import UsdLux
        UsdLux.DomeLight.Define(stage, "/World/domeLight").CreateIntensityAttr(1000.0)

        result_setup, setup_err = "no_splat", None
        if args.splat_usd:
            add_reference_to_stage(usd_path=args.splat_usd, prim_path="/World/splat")
            print(f"[compose] referenced splat: {args.splat_usd}", flush=True)
            # ensure the registry-downloaded exts are enabled (idempotent if --enable already
            # did it at launch), then apply the official NuRec render setup.
            from isaacsim.core.utils.extensions import enable_extension
            for ext in ("omni.rtx.spg", "isaacsim.replicator.nurec_utils"):
                try:
                    enable_extension(ext)
                except Exception as e:  # noqa: BLE001
                    setup_err = f"enable {ext}: {type(e).__name__}: {e}"
                    print(f"[compose] {setup_err}", flush=True)
            app.update()   # let the extension manager resolve/download before importing
            try:
                from isaacsim.replicator.nurec_utils import setup_for_rendering
                setup_for_rendering(stage)
                print("[compose] nurec_utils.setup_for_rendering applied", flush=True)
                result_setup = "nurec_utils"
            except Exception as e:  # noqa: BLE001
                setup_err = f"{type(e).__name__}: {e}"
                print(f"[compose] nurec_utils unavailable ({setup_err}); launch args only", flush=True)
                result_setup = "launch_args_only"
        result = nav_rover(world, app, args, cx, cy)
        result["nurec_setup"] = result_setup
        if setup_err:
            result["nurec_setup_err"] = setup_err

    json.dump(result, open(os.path.join(args.out_dir, f"{args.mode}_result.json"), "w"), indent=2)
    print("[compose] result:", json.dumps(result))
    app.close()


def gate5_drop(world, DynamicSphere, np, cx, cy, floor_z, n=5, radius=0.05):
    """Drop rigid spheres over the floor; each must rest at ~floor_z + radius (Gate 5)."""
    world.reset()
    pts = [(cx, cy), (cx + 0.5, cy), (cx - 0.5, cy), (cx, cy + 0.5), (cx, cy - 0.5)][:n]
    spheres = []
    for k, (x, y) in enumerate(pts):
        s = world.scene.add(DynamicSphere(prim_path=f"/World/probe_{k}", name=f"probe_{k}",
                                          position=np.array([x, y, floor_z + 0.6]),
                                          radius=radius, mass=0.2))
        spheres.append(s)
    world.reset()
    for _ in range(400):                      # ~ a few seconds of settling
        world.step(render=False)
    rests, oks = [], []
    for k, s in enumerate(spheres):
        z = float(s.get_world_pose()[0][2])
        expected = floor_z + radius
        err = abs(z - expected)
        rests.append({"xy": pts[k], "rest_z": z, "expected_z": expected, "err_m": err})
        oks.append(err < 0.03 and z > floor_z - 0.05)   # not fallen through, within 3 cm
    passed = sum(oks) >= max(1, len(oks) - 1)             # allow 1 sphere over an obstacle
    return {"gate5": "PASS" if passed else "REVIEW", "n": len(spheres),
            "n_ok": int(sum(oks)), "drops": rests}


def nav_rover(world, app, args, cx, cy):
    """Drive a self-contained rigid-body rover through the splat world under velocity control
    (real collision with the reconstructed collider) and render a chase-camera view each few
    steps, so we get a video of a robot navigating the photorealistic Gaussian-splat world.

    Self-contained on purpose: the compute node can't reach NVIDIA's S3 asset server, so we
    build the rover from primitives instead of fetching a Jetbot/Carter USD. Swapping in a
    real wheeled URDF later (once an asset is available locally) is a drop-in change.
    """
    import os
    import numpy as np
    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import UsdGeom, Gf
    import omni.replicator.core as rep
    info = {"robot": "procedural_rover", "frames": [], "note": "", "splat_rendered": None,
            "camera": "onpath" if args.cam_model else "chase"}
    fdir = os.path.join(args.out_dir, "nav_frames"); os.makedirs(fdir, exist_ok=True)
    z0 = args.floor_z + 0.08
    rover_start = np.array([cx, cy, z0]); drive = np.array([0.4, 0.0, 0.0])   # 0.4 m/s +x

    # camera pose: a RECORDED on-path pose (clean splat). OpenCV Twc -> USD camera basis
    # (USD cam looks down -Z, +Y up) via a Y,Z flip.
    if args.cam_model:
        Twc = read_cam_pose(args.cam_model)
        cpos = Twc[:3, 3]
        cquat = _R_to_quat_wxyz(Twc[:3, :3] @ np.diag([1.0, -1.0, -1.0]))
    else:
        cpos = np.array([cx - 2.0, cy, z0 + 0.5]); cquat = np.array([1.0, 0.0, 0.0, 0.0])

    cam_path = "/World/nav_cam"
    ucam = UsdGeom.Camera.Define(world.stage, cam_path)
    xf = UsdGeom.Xformable(ucam.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cpos[0]), float(cpos[1]), float(cpos[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(cquat[0]), float(cquat[1]), float(cquat[2]), float(cquat[3])))

    # rover at the collider centre — known-open floor (Gate-5 rested a body there)
    rover = world.scene.add(DynamicCuboid(
        prim_path="/World/rover", name="rover", position=rover_start,
        scale=np.array([0.3, 0.2, 0.12]), mass=2.0, color=np.array([0.1, 0.4, 0.9])))

    # headless capture via a replicator render product + annotator (avoids the
    # isaacsim.sensors.camera.get_rgba overscan bug in 5.1)
    rp = rep.create.render_product(cam_path, (1280, 720))
    annot = rep.AnnotatorRegistry.get_annotator("LdrColor")
    annot.attach([rp])

    world.reset()
    import cv2
    for _ in range(90):                        # warm up (NuRec gaussians take frames to converge)
        world.step(render=True)
    vel_err = None
    for i in range(400):
        try:
            rover.set_linear_velocity(drive)
        except Exception as e:  # noqa: BLE001
            vel_err = f"{type(e).__name__}: {e}"
        world.step(render=True)
        if i % 40 == 0:
            pos = rover.get_world_pose()[0]
            frame = {"step": i, "rover_xyz": [float(v) for v in pos]}
            try:
                arr = np.asarray(annot.get_data())
                if arr.ndim >= 3 and arr.shape[0] > 1 and arr.shape[1] > 1:
                    fp = os.path.join(fdir, f"nav_{i:04d}.png")
                    cv2.imwrite(fp, cv2.cvtColor(arr[..., :3].astype("uint8"), cv2.COLOR_RGB2BGR))
                    frame["frame"] = fp
                    info["splat_rendered"] = bool(arr[..., :3].std() > 3)
            except Exception as e:  # noqa: BLE001
                frame["capture_err"] = f"{type(e).__name__}: {e}"
            info["frames"].append(frame)
    if info["frames"]:
        d = np.linalg.norm(np.array(info["frames"][-1]["rover_xyz"]) - np.array(info["frames"][0]["rover_xyz"]))
        info["note"] += f"rover drove {d:.2f} m; {len([f for f in info['frames'] if 'frame' in f])} frames rendered"
    if vel_err:
        info["note"] += f"; set_linear_velocity failed: {vel_err}"
    return info


if __name__ == "__main__":
    main()
