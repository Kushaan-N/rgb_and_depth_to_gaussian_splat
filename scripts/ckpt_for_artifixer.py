"""Make one of our 3DGRUT checkpoints loadable by ArtiFixer's 3DGRUT fork (ArtiFixer venv, CPU).

The fork's renderer reads config keys our 3DGRUT does not have (e.g. `selected_indices_file`), so it
refuses our checkpoints. This composes the fork's own app config (default `apps/colmap_3dgut_sparse`,
the 3DGUT renderer like ours) and lays our checkpoint's config on top: every value we trained with is
kept, only keys the fork expects and we lack are filled with the fork's defaults. Gaussians untouched.

    python scripts/ckpt_for_artifixer.py --checkpoint ckpt_last.pt --out ckpt_artifixer.pt \
        [--fork-configs $AFX_DIR/thirdparty/3DGRUT-ArtiFixer/configs]
"""
from __future__ import annotations
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fork-configs", default=os.path.join(os.environ.get("AFX_DIR", ""), "thirdparty",
                                                           "3DGRUT-ArtiFixer", "configs"))
    ap.add_argument("--app", default="apps/colmap_3dgut_sparse")
    args = ap.parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=os.path.abspath(args.fork_configs), version_base=None):
        fork = compose(config_name=args.app)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ours = ck["config"]
    OmegaConf.set_struct(fork, False); OmegaConf.set_struct(ours, False)
    merged = OmegaConf.merge(fork, ours)                       # ours wins; fork-only keys filled in
    added = sorted(set(OmegaConf.to_container(fork).keys()) - set(OmegaConf.to_container(ours).keys()))
    ck["config"] = merged
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    print(f"[ckpt_for_artifixer] {args.checkpoint} -> {args.out}; top-level keys added from the fork: {added}", flush=True)


if __name__ == "__main__":
    main()
