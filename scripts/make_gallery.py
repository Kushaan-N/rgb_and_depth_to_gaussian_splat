#!/usr/bin/env python3
"""Build a single self-contained results.html — codec-free JS frame players + still frames.

Browsers won't reliably play OpenCV's mp4v-encoded .mp4 in an HTML5 <video>, so instead of
embedding video we embed the PNG frames (subsampled, resized, JPEG-compressed, base64) and
cycle them with a tiny JS player (play/pause + scrubber). Works in any browser, over http.server
or opened directly. No external assets.

    python scripts/make_gallery.py --seq mocap1_well-lit_trot --out results/results.html
"""
from __future__ import annotations
import argparse, base64, glob, os


def load_seq(frame_dir, n_max=48, width=560, quality=82, min_mean=20.0):
    """Subsample frames in `frame_dir`, drop near-black (uncovered) ones, JPEG-embed -> data URIs.

    mocap1's coverage is sparse, so a moving camera produces some black frames (view points into
    unreconstructed space). We skip those (mean brightness < min_mean) so the player only shows
    usable frames.
    """
    import cv2
    fs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    fs = [f for f in fs if (cv2.imread(f) is not None and cv2.imread(f).mean() >= min_mean)]
    if not fs:
        return []
    if len(fs) > n_max:
        step = len(fs) / n_max
        fs = [fs[int(i * step)] for i in range(n_max)]
    uris = []
    for f in fs:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        if w > width:
            im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            uris.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())
    return uris


def b64_img(path, width=720, quality=88):
    import cv2
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def find(*cands):
    for c in cands:
        hits = sorted(glob.glob(c))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="mocap1_well-lit_trot")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    repo, seq = args.repo, args.seq
    ws = os.environ.get("CEAR_WS", "/scratch4/workspace/%s-cear" % os.environ.get("USER", ""))
    nav = f"{ws}/outputs/{seq}"
    out = args.out or os.path.join(repo, "results", "results.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    gsplat_dir = find(f"{nav}/train_3dgrut/*/*/ours_30000/renders")       # 3DGRUT native (correct convention)
    robot_dir = find(f"{nav}/navsplat_hq", f"{nav}/navsplat")           # prefer HQ (1280x720)
    walk_dir  = find(f"{nav}/walkthrough_follow", f"{nav}/walkthrough_hq/camera_0",
                     f"{nav}/walkthrough/camera_0")                       # prefer upright follow-cam
    hero      = find(f"{nav}/navsplat_hq/nav_0011.png", f"{repo}/docs/figures/robot_in_splat_hero.png",
                     f"{nav}/navsplat/nav_0012.png")
    gt_img    = find(f"{repo}/docs/figures/gt_lab.png")
    broken    = find(f"{repo}/docs/figures/isaac_convention_bug.png")
    proof     = find(f"{repo}/docs/figures/nurec_render_proof.png")
    onoff     = find(f"{repo}/docs/figures/mocap1_onpath_vs_offpath.png")

    players, pid = [], 0

    def player_card(title, desc, frame_dir):
        nonlocal pid
        if not frame_dir:
            return ""
        uris = load_seq(frame_dir)
        if not uris:
            return ""
        pid += 1
        i = f"p{pid}"
        arr = "[" + ",".join(f'"{u}"' for u in uris) + "]"
        return f"""<section class="card"><h2>{title}</h2><p>{desc}</p>
          <img id="{i}img" class="frame">
          <div class="ctl"><button id="{i}btn">⏸ Pause</button>
            <input id="{i}rng" type="range" min="0" max="{len(uris)-1}" value="0">
            <span id="{i}lbl" class="lbl">1/{len(uris)}</span></div>
          <script>(function(){{const F={arr};let k=0,play=true;
            const img=document.getElementById("{i}img"),rng=document.getElementById("{i}rng"),
                  btn=document.getElementById("{i}btn"),lbl=document.getElementById("{i}lbl");
            function show(x){{k=(x+F.length)%F.length;img.src=F[k];rng.value=k;lbl.textContent=(k+1)+"/"+F.length;}}
            show(0);setInterval(()=>{{if(play)show(k+1);}},{int(1000/args.fps)});
            btn.onclick=()=>{{play=!play;btn.textContent=play?"⏸ Pause":"▶ Play";}};
            rng.oninput=()=>{{play=false;btn.textContent="▶ Play";show(+rng.value);}};}})();</script>
        </section>"""

    def img_card(title, desc, path):
        if not path or not os.path.exists(path):
            return ""
        u = b64_img(path)
        return f'<section class="card"><h2>{title}</h2><p>{desc}</p><img class="frame" src="{u}"></section>' if u else ""

    cards = [
        player_card("✅ The actual reconstruction — 3DGRUT native render (mocap1 = an indoor robotics lab)",
                    "51 held-out views rendered by 3DGRUT itself (the correct camera convention). This is "
                    "the true splat quality: a sharp, recognizable lab — walls, ceiling lights, shelving "
                    "with brick/box stacks, tiled floor. PSNR 26.2 / SSIM 0.84.", gsplat_dir),
        img_card("Ground truth (what the camera saw)", "A raw CEAR frame — the lab the splat reconstructs.", gt_img),
        img_card("⚠️ The Isaac render bug", "The SAME pose rendered through my Isaac/NuRec path: rotated ~90° "
                 "and smeared. The splat is fine — my COLMAP→NuRec camera-pose conversion is wrong, so it "
                 "renders from the wrong viewpoint. This is the bug to fix.", broken),
        player_card("🤖 Robot in Isaac (composite works; camera convention WIP)",
                    "A lit 3D robot composited into the NuRec volume in Isaac Sim 6.1.0 — proves mesh+splat "
                    "compositing works. The background is mangled by the same pose-convention bug above.", robot_dir),
        img_card("NuRec render proof", "Our .usdz rendered by NVIDIA's nurec_render.py — the asset loads and "
                 "renders (quality here is limited by the same pose convention).", proof),
    ]
    body = "\n".join(c for c in cards if c) or "<p class='card'>No frames found — run a render (see docs/RUN.md).</p>"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CEAR → Splat → Isaac</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; margin:0; background:#0f1115; color:#e8e8ea; }}
  header {{ padding:26px 24px; background:linear-gradient(135deg,#1b2333,#0f1115); border-bottom:1px solid #222; }}
  header h1 {{ margin:0 0 6px; font-size:22px; }} header p {{ margin:0; color:#9aa4b2; font-size:14px; }}
  main {{ max-width:860px; margin:0 auto; padding:18px; }}
  .card {{ background:#151922; border:1px solid #232a36; border-radius:12px; padding:18px; margin:18px 0; }}
  .card h2 {{ margin:0 0 6px; font-size:17px; }} .card p {{ margin:0 0 12px; color:#9aa4b2; font-size:13.5px; line-height:1.5; }}
  .frame {{ width:100%; border-radius:8px; background:#000; display:block; }}
  .ctl {{ display:flex; align-items:center; gap:12px; margin-top:10px; }}
  .ctl button {{ background:#2a3142; color:#e8e8ea; border:none; padding:7px 12px; border-radius:7px; cursor:pointer; font-size:13px; }}
  .ctl input[type=range] {{ flex:1; }} .lbl {{ color:#9aa4b2; font-size:12px; min-width:52px; text-align:right; }}
  footer {{ text-align:center; color:#6b7280; font-size:12px; padding:24px; }}
</style></head><body>
<header><h1>CEAR → Gaussian Splat → Isaac Sim</h1>
<p>mocap1 quadruped sequence reconstructed from RGB+depth+poses (no LiDAR), rendered with NuRec in Isaac Sim 6.1.0.</p></header>
<main>{body}</main>
<footer>Self-contained frame players (no video codec needed). Generated by scripts/make_gallery.py</footer>
</body></html>"""
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out)//1024} KB, {pid} players)")


if __name__ == "__main__":
    main()
