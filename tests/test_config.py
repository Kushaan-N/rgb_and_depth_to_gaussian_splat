"""Config inheritance: a sequence config = dataset base + a few overrides, with {seq} paths."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pose_utils as pu  # noqa: E402


def _write(p, text):
    p.write_text(text)
    return str(p)


def test_base_inheritance_and_seq_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT_T", "/data")
    (tmp_path / "datasets").mkdir()
    _write(tmp_path / "datasets" / "ds.yaml", """
sequence:
  data_root: "${DATA_ROOT_T}/{seq}"
  rgb_dir: rgb
timestamps:
  offsets_s: {event: 0.0, rgb: 0.004611, velodyne: 0.003044}
paths:
  out_root: "/out/{seq}"
lidar:
  topic: /velodyne_points
  extrinsic_chain: [{file: a, key: A}, {file: b, key: B}]
""")
    seq = _write(tmp_path / "seq.yaml", """
base: datasets/ds.yaml
sequence:
  name: run7
timestamps:
  offsets_s: {rgb: 0.005}
lidar:
  extrinsic_chain: [{file: c, key: C}]
""")
    cfg = pu.load_config(seq)
    assert cfg["sequence"]["name"] == "run7"
    assert cfg["sequence"]["rgb_dir"] == "rgb"                      # inherited
    assert cfg["sequence"]["data_root"] == "/data/run7"              # env + {seq}
    assert cfg["paths"]["out_root"] == "/out/run7"
    assert cfg["timestamps"]["offsets_s"] == {"event": 0.0, "rgb": 0.005, "velodyne": 0.003044}  # deep merge
    assert cfg["lidar"]["topic"] == "/velodyne_points"
    assert cfg["lidar"]["extrinsic_chain"] == [{"file": "c", "key": "C"}]  # lists are replaced, not merged
    assert "base" not in cfg


def test_config_without_base_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("X_T", "/x")
    p = _write(tmp_path / "plain.yaml", "sequence: {name: s1, data_root: '${X_T}/d'}\nother: '{seq} stays literal only with a name'\n")
    cfg = pu.load_config(p)
    assert cfg["sequence"]["data_root"] == "/x/d"
    assert cfg["other"] == "s1 stays literal only with a name"


def test_base_list_merges_in_order(tmp_path):
    """A generated variant config = [sequence config, variant overlay] + its own run_dir."""
    _write(tmp_path / "seq.yaml", "sequence: {name: s}\npipeline: {trainer_app: a.yaml, iters: 30000}\npaths: {out_root: '/o/{seq}'}\n")
    _write(tmp_path / "var.yaml", "pipeline: {trainer_app: b.yaml, trainer_overrides: [x=1]}\n")
    run = _write(tmp_path / "run.yaml", f"base: [{tmp_path/'seq.yaml'}, {tmp_path/'var.yaml'}]\npipeline: {{run_dir: /o/s/v}}\n")
    cfg = pu.load_config(run)
    assert cfg["pipeline"] == {"trainer_app": "b.yaml", "iters": 30000, "trainer_overrides": ["x=1"], "run_dir": "/o/s/v"}
    assert cfg["paths"]["out_root"] == "/o/s"
