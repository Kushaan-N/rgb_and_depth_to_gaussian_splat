"""§5.3 — constant-extrinsic ΔT refinement. On the synthetic fixture the extrinsic is
exact, so refinement should find nothing worth applying (small ΔT, sub-5% gain), and the
ΔT plumbing through pose_utils must actually change the composed pose."""

import numpy as np

import pose_utils as pu
import refine_extrinsic as rex


def test_delta_extrinsic_plumbing():
    assert np.allclose(pu.delta_extrinsic({}), np.eye(4))
    dT = pu.make_T(np.eye(3), [0.1, 0.0, 0.0])
    cfg = {"frames": {"extrinsic": {"delta_T": dT.tolist()}}}
    assert np.allclose(pu.delta_extrinsic(cfg), dT)


def test_refine_on_perfect_extrinsic_is_negligible(synthetic):
    r = rex.refine(synthetic["cfg"], k=6, n_pairs=5)
    # a correct extrinsic => tiny rotation and no meaningful warp-error gain
    assert r["rotation_deg"] < 1.0, f"unexpected large ΔT rotation: {r['rotation_deg']} deg"
    assert not r["significant"], f"spurious 'significant' ΔT: gain {r['relative_gain']}"
