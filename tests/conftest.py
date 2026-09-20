"""Shared pytest fixtures. Puts scripts/ and tests/ on sys.path and builds one
synthetic CEAR-format sequence per test session (see make_synthetic_sequence.py)."""

import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
for p in (os.path.join(_ROOT, "scripts"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from make_synthetic_sequence import make_synthetic_sequence  # noqa: E402
import pose_utils as pu  # noqa: E402


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """Generate a synthetic sequence once; return paths, loaded config, ground truth."""
    out = str(tmp_path_factory.mktemp("cear_synth"))
    paths = make_synthetic_sequence(out, "synthetic", n_frames=40, seed=0)
    cfg = pu.load_config(paths["config"])
    gt = json.load(open(paths["ground_truth"]))
    return dict(paths=paths, cfg=cfg, gt=gt, seq_dir=paths["seq_dir"], out=out)


@pytest.fixture(scope="session")
def synthetic_preamble(tmp_path_factory):
    """A sequence WITH a 2 s sync preamble (pitch swing), for TRAP-8 exclusion tests."""
    out = str(tmp_path_factory.mktemp("cear_synth_pre"))
    paths = make_synthetic_sequence(out, "synthetic", n_frames=48, seed=1, preamble_s=2.0)
    cfg = pu.load_config(paths["config"])
    gt = json.load(open(paths["ground_truth"]))
    return dict(paths=paths, cfg=cfg, gt=gt, seq_dir=paths["seq_dir"], out=out)
