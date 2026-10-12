"""Render a splat .ply exactly as SuperSplat does: PlayCanvas's gsplat renderer (SuperSplat's engine) in headless
Chromium (venv-viewer, GPU node), at the cameras of a COLMAP text dataset (PLY in that dataset's frame).

Writes <out-dir>/<image stem>.png per camera (score them with `eval_views.py --pred-dir <out-dir> --dataset
<dataset>`); --move renders from cameras moved off the recorded path (camera axes right/down/back, dataset
units). --editor also saves a screenshot of the SuperSplat editor itself (https://superspl.at/editor) with
the PLY loaded at its default view. Prints the WebGL renderer in use (GPU via ANGLE, else SwiftShader).

    python scripts/viewer_shots.py --ply splat.ply --dataset <ds> --out-dir shots/ [--every 8] [--move 0 -0.3 1]
"""
from __future__ import annotations
import argparse, base64, functools, http.server, json, os, sys, tempfile, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model

PC_URL = "https://cdn.jsdelivr.net/npm/playcanvas@2.7.4/build/playcanvas.mjs"
PAGE = """<!doctype html><html><body style="margin:0;background:#000">
<canvas id="c" width="%(W)d" height="%(H)d" style="width:%(W)dpx;height:%(H)dpx"></canvas>
<script type="module">
import * as pc from '%(PC)s';
const canvas = document.getElementById('c');
const app = new pc.Application(canvas, {graphicsDeviceOptions: {preserveDrawingBuffer: true, antialias: false, alpha: false}});
app.setCanvasFillMode(pc.FILLMODE_NONE); app.setCanvasResolution(pc.RESOLUTION_FIXED, %(W)d, %(H)d);
app.start();
const cam = new pc.Entity('cam');
cam.addComponent('camera', {clearColor: new pc.Color(0, 0, 0, 1), nearClip: 0.01, farClip: 1000, fov: 60});
app.root.addChild(cam);
const gl = app.graphicsDevice.gl;
const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
window.glRenderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : String(app.graphicsDevice.deviceType);
const asset = new pc.Asset('splat', 'gsplat', {url: '/scene.ply'});
app.assets.add(asset);
asset.ready(() => { const e = new pc.Entity('splat'); e.addComponent('gsplat', {asset}); app.root.addChild(e); window.splatReady = true; });
asset.on('error', (err) => { window.splatError = String(err); });
app.assets.load(asset);
window.shot = async (p, q, fov, waitMs) => {
  cam.setPosition(p[0], p[1], p[2]); cam.setRotation(new pc.Quat(q[0], q[1], q[2], q[3])); cam.camera.fov = fov;
  await new Promise(r => setTimeout(r, waitMs));          // let the async depth sort settle
  return canvas.toDataURL('image/png');
};
</script></body></html>"""

FLAG_SETS = [  # GPU first (ANGLE on Vulkan / EGL), software last
    ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"],
    ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"],
    ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
]


def serve(root):
    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*"); super().end_headers()
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--names-from", default=None, help="only image names present in this dataset's model")
    ap.add_argument("--move", type=float, nargs=3, default=None, metavar=("RIGHT", "DOWN", "BACK"))
    ap.add_argument("--wait-ms", type=int, default=600)
    ap.add_argument("--editor", action="store_true", help="also screenshot the SuperSplat editor's default view")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from scipy.spatial.transform import Rotation
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    imgs = sorted(imgs, key=lambda i: i[0])
    if args.names_from:
        keep = {i[0] for i in read_text_model(os.path.join(args.names_from, "sparse", "0"))[1]}
        imgs = [i for i in imgs if i[0] in keep]
    imgs = imgs[args.offset::args.every]
    W, H, _ = cams[imgs[0][1]]
    root = tempfile.mkdtemp(prefix="viewer_")
    os.symlink(os.path.abspath(args.ply), os.path.join(root, "scene.ply"))
    open(os.path.join(root, "index.html"), "w").write(PAGE % {"W": W, "H": H, "PC": PC_URL})
    port = serve(root); os.makedirs(args.out_dir, exist_ok=True)
    with sync_playwright() as pw:
        page = renderer = None
        for flags in FLAG_SETS:
            browser = pw.chromium.launch(headless=True, args=flags)
            page = browser.new_page(viewport={"width": W, "height": H})
            page.on("console", lambda m: m.type == "error" and print("[viewer] js:", m.text, flush=True))
            page.goto(f"http://127.0.0.1:{port}/index.html")
            try:
                page.wait_for_function("window.splatReady === true || window.splatError !== undefined", timeout=300000)
            except Exception as e:
                print(f"[viewer] {flags[0]}: no splat ({e.__class__.__name__})", flush=True); browser.close(); continue
            renderer = page.evaluate("window.glRenderer"); err = page.evaluate("window.splatError || null")
            if err is None and renderer:
                break
            print(f"[viewer] {flags[0]}: {err or 'no WebGL'}", flush=True); browser.close()
        print(f"[viewer] WebGL renderer: {renderer}", flush=True)
        t0 = time.time()
        for name, cid, q, t in imgs:
            _, _, (fx, fy, cx, cy) = cams[cid]
            Rcw = quat_to_R(q); C = -Rcw.T @ np.asarray(t)
            if args.move:
                C = C + Rcw.T @ np.array(args.move)
            Rpc = Rcw.T @ np.diag([1.0, -1.0, -1.0])                     # OpenCV camera -> PlayCanvas (x right, y up, -z fwd)
            qq = Rotation.from_matrix(Rpc).as_quat()                      # x, y, z, w
            fov = float(np.degrees(2 * np.arctan(H / (2 * fy))))          # vertical fov (principal point assumed central)
            url = page.evaluate("([p,q,f,w]) => window.shot(p,q,f,w)", [C.tolist(), qq.tolist(), fov, args.wait_ms])
            open(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".png"), "wb").write(
                base64.b64decode(url.split(",", 1)[1]))
        print(f"[viewer] {len(imgs)} views in {time.time() - t0:.0f}s -> {args.out_dir}", flush=True)
        if args.editor:                                                   # the real SuperSplat app, default view
            ed = page.context.browser.new_page(viewport={"width": 1280, "height": 800})
            ed.goto(f"https://superspl.at/editor?load=http://127.0.0.1:{port}/scene.ply")
            time.sleep(60)
            ed.screenshot(path=os.path.join(args.out_dir, "_supersplat_editor.png"))
            print("[viewer] editor screenshot ->", os.path.join(args.out_dir, "_supersplat_editor.png"), flush=True)
    json.dump({"renderer": renderer, "ply": args.ply, "n": len(imgs), "move": args.move},
              open(os.path.join(args.out_dir, "_viewer.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
