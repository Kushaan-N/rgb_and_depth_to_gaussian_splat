"""End-to-end CPU pipeline test on the synthetic fixture (Phases 1-5).

Runs each phase's build() in order against the generated sequence and asserts the Gates.
Because the fixture has known ground truth (floor at z=0, a box, a curved trajectory), a
convention/transform regression anywhere upstream shows up here as a failed Gate.
"""

import os
import tempfile

import numpy as np

import build_poses
import build_depth_map
import verify_depth_map
import precondition_frames
import gate_frames
import build_collider
import verify_reprojection as vr


def test_full_cpu_pipeline(synthetic):
    cfg = synthetic["cfg"]

    # Phase 2
    meta_p = build_poses.build(cfg)
    assert meta_p["n_frames_kept"] >= 30
    model = os.path.join(cfg["paths"]["out_root"], "colmap", "sparse", "0", "images.txt")
    assert os.path.exists(model)

    # Gate 2 + 2b
    kept, Twc, calib, *_ = build_poses.compute_world_cam(cfg)
    with tempfile.TemporaryDirectory() as td:
        errs = vr.run_gate2(cfg, kept, Twc, calib, k=8, n_pairs=8, out_dir=td,
                            tag="e2e", save=False)
    assert np.nanmean(errs) < 20.0, f"Gate 2 warp error too high: {np.nanmean(errs)}"

    # Phase 3
    meta_d = build_depth_map.build(cfg)
    assert meta_d["n_seed_points"] > 1000
    assert meta_d["n_mesh_triangles"] > 1000

    # Gate 3
    meta_g3 = verify_depth_map.build(cfg)
    assert meta_g3["gate3_pass"], f"Gate 3 failed: {meta_g3['checks']}"
    assert abs(meta_g3["floor_height_full_m"]) < 0.03, "fused floor not near z=0"
    assert meta_g3["split_floor_dz_m"] < 0.02

    # Phase 4
    precondition_frames.precondition(cfg)
    meta_gf = gate_frames.build(cfg)
    assert meta_gf["n_train"] > 0 and meta_gf["n_val"] > 0

    # Phase 5 + Gate 5 proxy
    meta_c = build_collider.build(cfg)
    assert meta_c["gate5_proxy_pass"], f"Gate 5 proxy failed: {meta_c['gate5_proxy']}"
    assert abs(meta_c["floor_z_m"]) < 0.03, "collider floor not near z=0"
    assert meta_c["gate5_proxy"]["hit_fraction"] > 0.99, "collider has holes"
