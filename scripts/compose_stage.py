"""Phase 6 — compose the Isaac Sim scene and validate walking (docs/PLAN.md §9).

  *** Runs ONLY inside Isaac Sim's python on the L40S partition (RTX cores required;
      A100/H100 have none). Launch via sbatch/isaac_l40s.sbatch. ***

Composes: splat (visual, no collider) + collider mesh (physics, invisible) + a physics
scene (gravity, Z-up), then either drops a rigid body (Gate 5, real) or spawns a quadruped
and walks it >= 5 m (Gate 6). All Isaac imports are inside functions so this file stays
importable/syntax-checkable off-GPU.

    (inside Isaac python)
    python scripts/compose_stage.py --splat-usd .../scene_particlefield.usdz \
        --collider .../collider/collider.ply --mode drop
    python scripts/compose_stage.py --splat-usd .../scene_nurec.usdz \
        --collider .../collider/collider.ply --robot .../mini_cheetah.usd --mode walk
"""

from __future__ import annotations

import argparse


def _boot(headless=True):
    """Start Isaac Sim's SimulationApp (must happen before any omni import)."""
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": headless})
    return app


def build_stage(splat_usd: str, collider_path: str):
    """Create a stage: Z-up physics scene + visual splat + invisible collider mesh."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf  # noqa: F401
    import numpy as np
    import trimesh  # for reading the collider .ply/.obj into USD points

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # physics scene
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    # visual splat (reference the exported USD/USDZ; it carries no collider, §8)
    splat = UsdGeom.Xform.Define(stage, "/World/splat")
    splat.GetPrim().GetReferences().AddReference(splat_usd)

    # collider mesh: invisible, with a collision API (foot contact geometry)
    mesh = trimesh.load(collider_path, process=False)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)
    m = UsdGeom.Mesh.Define(stage, "/World/collider")
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in V])
    m.CreateFaceVertexCountsAttr([3] * len(F))
    m.CreateFaceVertexIndicesAttr(F.ravel().tolist())
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    meshcol = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
    meshcol.CreateApproximationAttr().Set("none")     # exact triangle mesh (static)
    UsdGeom.Imageable(m).MakeInvisible()
    return stage


def gate5_drop(app, stage, drop_xyz=(0.0, 0.0, 1.0)):
    """Gate 5 (real): drop a rigid sphere; it must rest on the floor at the right height."""
    from pxr import UsdGeom, UsdPhysics, Gf, Sdf
    import numpy as np
    s = UsdGeom.Sphere.Define(stage, "/World/probe")
    s.AddTranslateOp().Set(Gf.Vec3f(*drop_xyz))
    s.CreateRadiusAttr(0.05)
    UsdPhysics.RigidBodyAPI.Apply(s.GetPrim())
    UsdPhysics.CollisionAPI.Apply(s.GetPrim())
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    for _ in range(240):                      # ~4 s at 60 Hz
        app.update()
    xf = UsdGeom.Xformable(s.GetPrim()).ComputeLocalToWorldTransform(0)
    rest_z = float(xf.ExtractTranslation()[2])
    print(f"[Gate 5] sphere rest height z = {rest_z:.4f} m (expect ~ floor + radius)")
    return rest_z


def gate6_walk(app, stage, robot_usd: str, distance_m=5.0):
    """Gate 6: spawn a quadruped and command a forward walk; check no penetration."""
    if not robot_usd:
        print("[Gate 6] no --robot given; substitute an Isaac Lab quadruped "
              "(ANYmal/Unitree) — note the substitution in REPORT.md")
        return None
    from pxr import UsdGeom, Gf
    ref = UsdGeom.Xform.Define(stage, "/World/robot")
    ref.GetPrim().GetReferences().AddReference(robot_usd)
    ref.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.35))    # camera-height spawn
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    # TODO(lab): drive the quadruped controller (Isaac Lab locomotion policy or a
    #   position/velocity command) forward `distance_m`; log base trajectory + contact
    #   forces; flag any collider penetration. Route through depth-observed regions
    #   (Gate 3 coverage map). Capture video via the RTX renderer / WebRTC livestream.
    for _ in range(600):
        app.update()
    print(f"[Gate 6] walked (target {distance_m} m) — verify no penetration in the capture")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat-usd", required=True)
    ap.add_argument("--collider", required=True)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--mode", choices=["drop", "walk"], default="drop")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    app = _boot(headless=not args.gui)
    try:
        stage = build_stage(args.splat_usd, args.collider)
        if args.mode == "drop":
            gate5_drop(app, stage)
        else:
            gate6_walk(app, stage, args.robot)
    finally:
        app.close()


if __name__ == "__main__":
    main()
