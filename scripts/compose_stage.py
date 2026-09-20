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
    ap.add_argument("--out-dir", default="/tmp/isaac_out")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from isaacsim import SimulationApp
    # nav needs rendering (RTX); drop is physics-only and can run truly headless
    app = SimulationApp({"headless": not args.gui, "renderer": "RayTracedLighting"})

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
        if args.splat_usd:
            add_reference_to_stage(usd_path=args.splat_usd, prim_path="/World/splat")
            print(f"[compose] referenced splat: {args.splat_usd}")
        result = nav_wheeled(world, app, args, cx, cy)

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


def nav_wheeled(world, app, args, cx, cy):
    """Spawn a wheeled robot and drive it forward through the covered region; save frames."""
    import numpy as np
    info = {"robot": args.robot, "frames": [], "note": ""}
    try:
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
        assets = get_assets_root_path()
        if assets is None:
            info["note"] = "no asset server reachable from compute node; robot skipped"
            return info
        asset_path = f"{assets}/Isaac/Robots/Jetbot/jetbot.usd"
        robot = world.scene.add(WheeledRobot(
            prim_path="/World/robot", name="robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True, usd_path=asset_path,
            position=np.array([cx, cy, args.floor_z + 0.05])))
        ctrl = DifferentialController(name="diff", wheel_radius=0.03, wheel_base=0.1125)
        world.reset()
        for i in range(600):
            robot.apply_wheel_actions(ctrl.forward(command=[0.3, 0.0]))  # 0.3 m/s forward
            world.step(render=True)
            if i % 60 == 0:
                info["frames"].append({"step": i,
                                       "base_xy": robot.get_world_pose()[0][:2].tolist()})
        info["note"] = "drove forward 0.3 m/s; base trajectory logged (see frames)"
    except Exception as e:  # noqa: BLE001
        info["note"] = f"nav error: {type(e).__name__}: {e}"
    return info


if __name__ == "__main__":
    main()
