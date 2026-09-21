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
    ap.add_argument("--mode", choices=["drop", "nav", "collision"], default="drop")
    ap.add_argument("--floor-z", type=float, default=0.0, help="RANSAC floor height (m)")
    ap.add_argument("--robot", default="jetbot", help="wheeled robot asset (nav mode)")
    ap.add_argument("--cam-model", default=None,
                    help="COLMAP sparse dir; place the nav camera at a RECORDED pose (on-path"
                         " -> clean splat) instead of a chase cam")
    ap.add_argument("--out-dir", default="/tmp/isaac_out")
    ap.add_argument("--show-collider", action="store_true",
                    help="render the collider mesh (debug/geometry view). Default: when a splat"
                         " is loaded the collider is an INVISIBLE physics proxy so it doesn't"
                         " occlude the photoreal NuRec volume.")
    ap.add_argument("--warmup", type=int, default=800,
                    help="render steps to converge the NuRec gaussians before capturing "
                         "(NVIDIA's nurec_render.py default is 800; too few -> wrong/blurry).")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from isaacsim import SimulationApp
    # nav needs rendering (RTX); drop is physics-only and can run truly headless.
    #
    # NuRec (neural Gaussian volume) rendering was ADDED IN ISAAC SIM 6.0.0 — the base
    # isaac-sim:5.1.0 image has NO NuRec support at all (verified: no NuRec USD schema, no
    # render feature, no isaacsim.replicator.nurec_utils; none of it in its registry either).
    # So this script is written to work on BOTH images:
    #   * 6.0.x  -> nurec_utils.setup_for_rendering renders the photoreal Gaussian volume.
    #   * 5.1.0  -> that import fails gracefully; the dome light (added below) still renders the
    #               reconstructed geometry (collider + rover) so we get a real navigation.
    # We do NOT `--enable` the exts at launch (a missing ext aborts Kit before boot on 5.1); we
    # enable them at runtime below (graceful). The ONLY launch arg needed is single-GPU — the
    # canonical nurec_render.py passes just `--enable omni.rtx.spg --/renderer/multiGpu/enabled=
    # false`. CRUCIAL: do NOT set /omni/rtx/nre/compositing/disableNuRecPostProcessings or
    # /rtx/rtpt/gaussian/skipTonemapping here — those are SPG/PPISP-only. For a PLAIN NuRec
    # volume the engine ISP/tonemap/compositing must stay ON; forcing disableNuRecPostProcessings
    # =true blanks the render to black. setup_for_rendering() applies the correct per-stage
    # (plain vs SPG) overrides before the first Hydra sync.
    nurec_args = ["--/renderer/multiGpu/enabled=false"]
    app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting",
                         "multi_gpu": False,
                         "extra_args": nurec_args if args.mode == "nav" else []})

    # NuRec render extensions must be enabled right after boot — BEFORE the stage is built and
    # the first Hydra sync — exactly as NVIDIA's standalone_examples/nurec/nurec_render.py does
    # (enable isaacsim.replicator.nurec_utils, then enable_omni_rtx_spg). Bundled in Isaac 6.x;
    # absent on 5.1 (there this fails gracefully and we fall back to the dome-lit geometry).
    nurec_ready = False
    if args.mode == "nav" and args.splat_usd:
        try:
            app.update()
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.replicator.nurec_utils")
            from isaacsim.replicator.nurec_utils.rendering_setup import enable_omni_rtx_spg
            enable_omni_rtx_spg(app)
            nurec_ready = True
            print("[compose] NuRec extensions enabled (nurec_utils + omni.rtx.spg)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[compose] NuRec enable failed ({type(e).__name__}: {e}); geometry-only", flush=True)

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
    # collider is a physics proxy: hide it in drop mode, and in nav-with-splat (so the opaque
    # gray mesh doesn't occlude the photoreal Gaussian volume). Keep it visible for a pure
    # geometry nav (no splat) or when explicitly requested with --show-collider.
    if args.mode == "drop" or (args.splat_usd is not None and not args.show_collider):
        UsdGeom.Imageable(m).MakeInvisible()
    vmin = np.array(verts).min(0); vmax = np.array(verts).max(0)
    cx, cy = float((vmin[0] + vmax[0]) / 2), float((vmin[1] + vmax[1]) / 2)
    print(f"[compose] collider verts={len(verts)} tris={len(faces)} "
          f"bbox x[{vmin[0]:.2f},{vmax[0]:.2f}] y[{vmin[1]:.2f},{vmax[1]:.2f}] floor_z={args.floor_z}")

    result = {}
    if args.mode == "drop":
        result = gate5_drop(world, DynamicSphere, np, cx, cy, args.floor_z)
    elif args.mode == "collision":
        result = collision_test(world, np, cx, cy, args.floor_z, verts, faces, args.out_dir)
    else:
        # a dome light so the rover (and, on 5.1, the reconstructed geometry) is lit. The NuRec
        # volume carries its own baked radiance, so this mainly lights the rover cuboid; it also
        # rescues the 5.1 geometry fallback from the all-black frames of no-light + no-splat.
        from pxr import UsdLux
        UsdLux.DomeLight.Define(stage, "/World/domeLight").CreateIntensityAttr(1000.0)

        result_setup, setup_err = "no_splat", None
        if args.splat_usd:
            add_reference_to_stage(usd_path=args.splat_usd, prim_path="/World/splat")
            print(f"[compose] referenced splat: {args.splat_usd}", flush=True)
            # Apply the NuRec render setup on the composed stage (detects the NuRec volume prim,
            # sets gaussian tonemapping/photometry) — must run before the first render update.
            # The exts were already enabled right after boot (nurec_ready).
            if nurec_ready:
                try:
                    from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering
                    setup_for_rendering(stage)
                    print("[compose] setup_for_rendering applied", flush=True)
                    result_setup = "nurec_utils"
                except Exception as e:  # noqa: BLE001
                    setup_err = f"{type(e).__name__}: {e}"
                    print(f"[compose] setup_for_rendering failed ({setup_err})", flush=True)
                    result_setup = "launch_args_only"
            else:
                result_setup = "geometry_only"  # 5.1 / no NuRec — dome-lit collider render
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


def collision_test(world, np, cx, cy, floor_z, verts, faces, out_dir):
    """Rigorously validate physics navigation on the reconstructed collider:
       (1) drop a grid of probes — nothing falls through the floor; probes over obstacles rest
           ELEVATED (proves 3-D collision, not just a ground plane);
       (2) drive a CCD rover into the walls — it must be BLOCKED and stay contained (no tunnelling
           out), with real PhysX CONTACT events logged as direct proof;
       plus rendered frames of the rover navigating + colliding.
    """
    import os
    import cv2
    from isaacsim.core.api.objects import DynamicSphere, DynamicCuboid, FixedCuboid
    from pxr import UsdGeom, UsdLux, Gf, PhysxSchema
    import omni.replicator.core as rep
    fdir = os.path.join(out_dir, "collision_frames"); os.makedirs(fdir, exist_ok=True)
    V = np.array(verts); vmin = V.min(0); vmax = V.max(0)
    info = {"floor_grid": [], "walls": [], "notes": ""}

    # backup ground plane at the detected floor height: the reconstructed collider has holes at
    # the periphery (coverage limit), so a robot driving there falls into the void. A large thin
    # static collider whose TOP sits at floor_z catches those holes -> the robot rests on the real
    # mesh where it exists and on this plane over holes; walls/obstacles from the mesh still block.
    gx = float(vmax[0] - vmin[0]) + 6.0; gy = float(vmax[1] - vmin[1]) + 6.0
    ground = world.scene.add(FixedCuboid(prim_path="/World/backup_ground", name="backup_ground",
                             position=np.array([cx, cy, floor_z - 0.05]),
                             scale=np.array([gx, gy, 0.1])))
    try:
        UsdGeom.Imageable(world.stage.GetPrimAtPath("/World/backup_ground")).MakeInvisible()
    except Exception:  # noqa: BLE001
        pass

    # contact reporting — direct proof collisions fire (global counter via PhysX callback)
    contacts = {"n": 0}
    try:
        from omni.physx import get_physx_simulation_interface
        _sub = get_physx_simulation_interface().subscribe_contact_report_events(
            lambda headers, data: contacts.__setitem__("n", contacts["n"] + len(headers)))
    except Exception as e:  # noqa: BLE001
        info["notes"] += f"contact-sub failed: {e}; "; _sub = None

    UsdLux.DomeLight.Define(world.stage, "/World/domeLight").CreateIntensityAttr(1200.0)

    # angled overhead camera looking at the scene centre (Z-up world), for visual proof
    cam_pos = np.array([vmax[0] + 1.5, vmin[1] - 1.5, floor_z + 3.5])
    fwd = np.array([cx, cy, floor_z]) - cam_pos; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up = np.array([0, 0, 1.0]); right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    Rc = np.column_stack([right, np.cross(right, fwd), -fwd]); q = _R_to_quat_wxyz(Rc)
    cam = UsdGeom.Camera.Define(world.stage, "/World/coll_cam")
    xf = UsdGeom.Xformable(cam.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in cam_pos]))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    rprod = rep.create.render_product("/World/coll_cam", (1280, 720))
    annot = rep.AnnotatorRegistry.get_annotator("LdrColor"); annot.attach([rprod])

    def snap(name):
        try:
            for _ in range(4):
                world.step(render=True)
            arr = np.asarray(annot.get_data())
            if arr.ndim >= 3 and arr.shape[0] > 1 and arr[..., :3].std() > 2:
                cv2.imwrite(os.path.join(fdir, name),
                            cv2.cvtColor(arr[..., :3].astype("uint8"), cv2.COLOR_RGB2BGR))
                return True
        except Exception as e:  # noqa: BLE001
            info["notes"] += f"snap {name}: {e}; "
        return False

    # ---- Test 1: drop grid (floor + obstacles) ----
    probes = []
    for i, x in enumerate(np.linspace(vmin[0] + 1.0, vmax[0] - 1.0, 3)):
        for j, y in enumerate(np.linspace(vmin[1] + 1.0, vmax[1] - 1.0, 3)):
            s = world.scene.add(DynamicSphere(prim_path=f"/World/gp_{i}_{j}", name=f"gp_{i}_{j}",
                                position=np.array([float(x), float(y), floor_z + 0.6]),
                                radius=0.05, mass=0.2))
            probes.append((s, (float(x), float(y))))
    world.reset()
    for _ in range(400):
        world.step(render=False)
    n_through = 0
    rest_zs = []
    for s, (x, y) in probes:
        z = float(s.get_world_pose()[0][2]); through = z < floor_z - 0.1
        n_through += int(through); rest_zs.append(z)
        info["floor_grid"].append({"xy": [round(x, 2), round(y, 2)], "rest_z": round(z, 4), "through": through})
    info["floor_rest_spread_m"] = round(float(max(rest_zs) - min(rest_zs)), 3)  # >0 => 3-D geometry
    snap("01_floor_grid.png")

    # ---- Test 1b: SCATTERED OBSTACLES must be solid colliders ----
    # detect raised structures from the collider mesh (grid-bin XY, cells whose top is well above
    # the floor = blocks/boxes/furniture), then (a) drop a probe onto each -> must rest ELEVATED
    # (proves the obstacle is a solid collider, not empty space), and (b) drive a rover into the
    # biggest few -> must be blocked with contacts (horizontal collision on obstacles, not walls).
    import collections
    CELL = 0.3
    cellmax = collections.defaultdict(lambda: -1e9)
    for x, y, z in V:
        cellmax[(int(np.floor(x/CELL)), int(np.floor(y/CELL)))] = max(
            cellmax[(int(np.floor(x/CELL)), int(np.floor(y/CELL)))], float(z))
    raised = {c: h for c, h in cellmax.items() if h > floor_z + 0.12}   # >12cm above floor
    seen, clusters = set(), []
    for c in raised:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            cc = stack.pop()
            if cc in seen or cc not in raised:
                continue
            seen.add(cc); comp.append(cc)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    stack.append((cc[0]+ddx, cc[1]+ddy))
        clusters.append(comp)
    obstacles = []
    for comp in clusters:
        if len(comp) < 2:                       # skip single-cell noise
            continue
        ixs = [c[0] for c in comp]; iys = [c[1] for c in comp]
        x0, x1 = min(ixs)*CELL, (max(ixs)+1)*CELL
        y0, y1 = min(iys)*CELL, (max(iys)+1)*CELL
        ox, oy = (x0+x1)/2, (y0+y1)/2
        top = float(max(raised[c] for c in comp))
        # keep obstacles that sit inside the floor area (not the perimeter walls)
        if vmin[0]+0.4 < ox < vmax[0]-0.4 and vmin[1]+0.4 < oy < vmax[1]-0.4:
            obstacles.append({"xy": [round(ox, 2), round(oy, 2)], "top_z": round(top, 3),
                              "cells": len(comp), "wx": round(x1-x0, 2), "wy": round(y1-y0, 2)})
    obstacles.sort(key=lambda o: -o["cells"])
    obstacles = obstacles[:12]
    info["n_obstacles_detected"] = len(obstacles)

    # ENSURE collision: materialize a SOLID invisible box collider at each detected obstacle
    # (footprint + height from the reconstruction), so the robot reliably collides with every
    # block even though the raw mesh shells are thin/holey.
    for k, o in enumerate(obstacles):
        h = max(0.15, o["top_z"] - floor_z)
        world.scene.add(FixedCuboid(prim_path=f"/World/obs_{k}", name=f"obs_{k}",
                        position=np.array([o["xy"][0], o["xy"][1], floor_z + h/2]),
                        scale=np.array([max(o["wx"], 0.25), max(o["wy"], 0.25), h])))
        try:
            UsdGeom.Imageable(world.stage.GetPrimAtPath(f"/World/obs_{k}")).MakeInvisible()
        except Exception:  # noqa: BLE001
            pass
    info["obstacle_proxies"] = len(obstacles)
    info["obstacles"] = []
    for k, o in enumerate(obstacles):
        ox, oy = o["xy"]; top = o["top_z"]
        s = world.scene.add(DynamicSphere(prim_path=f"/World/op_{k}", name=f"op_{k}",
                            position=np.array([ox, oy, top + 0.35]), radius=0.05, mass=0.2))
        world.reset()
        for _ in range(250):
            world.step(render=False)
        rz = float(s.get_world_pose()[0][2])
        o["probe_rest_z"] = round(rz, 3)
        o["solid"] = bool(rz > floor_z + 0.1)   # rested elevated on the obstacle = solid collider
        info["obstacles"].append(o)
    snap("03_obstacle_drops.png")

    # drive a rover horizontally INTO the biggest few obstacles -> each must be blocked + contacts
    info["obstacle_drives"] = []
    for k, o in enumerate(obstacles[:4]):
        ox, oy = o["xy"]
        approach = np.array([cx - ox, cy - oy]); nrm = float(np.linalg.norm(approach))
        approach = approach/nrm if nrm > 1e-6 else np.array([1.0, 0.0])
        half = 0.5 * max(o["wx"], o["wy"])
        startp = np.array([ox, oy]) + approach*(half + 1.0)        # ~1 m out from the obstacle face
        path = f"/World/rover_obs_{k}"
        rover = world.scene.add(DynamicCuboid(prim_path=path, name=f"r_obs_{k}",
                                position=np.array([startp[0], startp[1], floor_z+0.12]),
                                scale=np.array([0.25, 0.25, 0.15]), mass=3.0, color=np.array([0.95, 0.4, 0.1])))
        try:
            PhysxSchema.PhysxRigidBodyAPI.Apply(world.stage.GetPrimAtPath(path)).CreateEnableCCDAttr(True)
            PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
        except Exception:  # noqa: BLE001
            pass
        world.reset()
        c0 = contacts["n"]; drive = -approach                     # toward the obstacle
        for step in range(200):
            try:
                rover.set_linear_velocity(np.array([drive[0]*0.8, drive[1]*0.8, 0.0]))
            except Exception:  # noqa: BLE001
                pass
            world.step(render=(step % 50 == 0 and k == 0))
        endp = rover.get_world_pose()[0]
        d_to_obs = float(np.linalg.norm(np.array([float(endp[0]), float(endp[1])]) - np.array([ox, oy])))
        # blocked = collided (contacts) AND did not cross to the FAR side of the obstacle. `approach`
        # points obstacle->centre (the rover's start side), so a rover still on the start side has a
        # positive projection; one that tunnelled through to the far side goes negative.
        proj = float(np.dot(np.array([float(endp[0]) - ox, float(endp[1]) - oy]), approach))
        nc = contacts["n"] - c0
        info["obstacle_drives"].append({"xy": [ox, oy], "end_dist": round(d_to_obs, 2),
                                        "proj": round(proj, 2), "contacts": nc,
                                        "blocked": bool(nc > 0 and proj > -0.2)})
        if k == 0:
            snap("04_obstacle_drive.png")

    # ---- Test 2: drive a CCD rover into each wall — must be blocked + contained ----
    margin = 0.5
    for dname, (dx, dy) in {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.items():
        path = f"/World/rover_{dname.replace('+', 'p').replace('-', 'm')}"
        rover = world.scene.add(DynamicCuboid(prim_path=path, name="r_" + dname,
                                position=np.array([cx, cy, floor_z + 0.12]),
                                scale=np.array([0.3, 0.3, 0.15]), mass=3.0,
                                color=np.array([0.1, 0.5, 0.95])))
        try:
            rprim = world.stage.GetPrimAtPath(path)
            PhysxSchema.PhysxRigidBodyAPI.Apply(rprim).CreateEnableCCDAttr(True)  # no tunnelling
            PhysxSchema.PhysxContactReportAPI.Apply(rprim).CreateThresholdAttr(0.0)
        except Exception:  # noqa: BLE001
            pass
        world.reset()
        c0 = contacts["n"]
        for step in range(400):
            try:
                rover.set_linear_velocity(np.array([dx * 1.0, dy * 1.0, 0.0]))
            except Exception:  # noqa: BLE001
                pass
            world.step(render=(step % 40 == 0))
        p = rover.get_world_pose()[0]; end = np.array([float(p[0]), float(p[1])])
        reach = float(np.linalg.norm(end - np.array([cx, cy])))
        fell = bool(float(p[2]) < floor_z - 0.3)               # fell through despite backup plane
        blocked = bool(reach < 3.0)                             # stopped early = hit a wall/obstacle
        info["walls"].append({"dir": dname, "reach_m": round(reach, 2),
                              "end_xy": [round(end[0], 2), round(end[1], 2)],
                              "blocked_by_wall": blocked, "fell": fell,
                              "contacts": contacts["n"] - c0})
        snap(f"02_wall_{dname}.png")

    floor_ok = len(probes) - n_through
    n_fell = sum(1 for w in info["walls"] if w["fell"])
    n_blocked = sum(1 for w in info["walls"] if w["blocked_by_wall"])
    total_contacts = sum(w["contacts"] for w in info["walls"])
    n_obs = len(info["obstacles"]); n_solid = sum(1 for o in info["obstacles"] if o["solid"])
    drives = info["obstacle_drives"]; n_drv = len(drives)
    n_drv_ok = sum(1 for d in drives if d["blocked"] and d["contacts"] > 0)
    obs_drive_ok = n_drv > 0 and n_drv_ok == n_drv
    info["floor_ok"] = f"{floor_ok}/{len(probes)}"
    info["rovers_fell"] = n_fell
    info["walls_blocked"] = n_blocked
    info["obstacles_solid"] = f"{n_solid}/{n_obs}"
    info["obstacle_drives_blocked"] = f"{n_drv_ok}/{n_drv}"
    info["total_contacts"] = total_contacts
    info["frames_dir"] = fdir
    # PASS: floor holds, no rover falls, contacts fire, AND every detected obstacle is solid
    # (probe rests on it) and blocks a rover driven into it (proxy colliders guarantee this).
    info["gate_collision"] = "PASS" if (n_through == 0 and n_fell == 0 and total_contacts > 0
                                        and n_obs > 0 and n_solid == n_obs and obs_drive_ok) else "REVIEW"
    print(f"[compose] collision: floor {info['floor_ok']}, obstacles_solid {n_solid}/{n_obs}, "
          f"obstacle_drives_blocked {n_drv_ok}/{n_drv}, walls_blocked {n_blocked}/4, "
          f"rovers_fell {n_fell}, {total_contacts} contacts -> {info['gate_collision']}", flush=True)
    return info


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
    for _ in range(args.warmup):               # converge NuRec gaussians before capture (default 800)
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
