"""Fix rendered views with NVIDIA Difix (Difix3D+, CVPR 2025) — runs in the Difix venv (GPU).

Each render is cleaned by the single-step diffusion model, conditioned on a real reference image
(`nvidia/difix_ref`), with the settings of the authors' own trainer: prompt "remove degradation",
one step at timestep 199, no guidance, output resized back to the render's size (Lanczos).

    python scripts/difix_images.py --in-dir renders/ --out-dir fixed/ --refs refs.json
        refs.json = {"<render name>": "<path of its real reference image>", ...}

Needs DIFIX_DIR (the Difix3D checkout, for pipeline_difix) — set by env/cear_env.sh.
Difix code and weights are NVIDIA-licensed for non-commercial (research/evaluation) use.
"""
from __future__ import annotations
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--refs", default=None, help="json name -> reference image; omit for the reference-free model")
    ap.add_argument("--model", default=None, help="default: nvidia/difix_ref with --refs, else nvidia/difix")
    ap.add_argument("--timestep", type=int, default=199)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.environ.get("DIFIX_DIR", ""), "src"))
    import torch
    from PIL import Image
    from pipeline_difix import DifixPipeline

    refs = json.load(open(args.refs)) if args.refs else None
    model = args.model or ("nvidia/difix_ref" if refs else "nvidia/difix")
    pipe = DifixPipeline.from_pretrained(model, trust_remote_code=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    names = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith((".png", ".jpg")))
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for n in names:
        img = Image.open(os.path.join(args.in_dir, n)).convert("RGB")
        kw = {"ref_image": Image.open(refs[n]).convert("RGB")} if refs else {}
        with torch.no_grad():
            out = pipe("remove degradation", image=img, num_inference_steps=1, timesteps=[args.timestep],
                       guidance_scale=0.0, **kw).images[0]
        out.resize(img.size, Image.LANCZOS).save(os.path.join(args.out_dir, n))
    dt = (time.time() - t0) / max(len(names), 1)
    print(f"[difix] {model}: fixed {len(names)} views -> {args.out_dir} ({dt * 1000:.0f} ms/view)", flush=True)


if __name__ == "__main__":
    main()
