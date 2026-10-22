"""Render a splat .ply as SuperSplat shows it, in headless Chromium (venv-viewer, GPU node), at the cameras of a
COLMAP text dataset (PLY in that dataset's frame). Two engines (--engine):

  pc274    (default) PlayCanvas engine 2.7.4's gsplat component on WebGL (ANGLE/Vulkan): the engine older SuperSplat
           2.x shipped (band-2 SH bug, ~0.075 px^2 effective dilation, sub-pixel cull, 1/255 discard).
  current  today's SuperSplat itself (the live editor, --supersplat-url, 3.x): WebGPU only, its own compute splat
           renderer (0.3 px^2 dilation, normalised kernel cut at d^2=8, 2 px size cull, 8-bit opacity, correct SH).
           The editor is driven through its window.scene: settled sorted frames (stochastic mode off), camera via
           scene.camera.setPoseOverride, pixels via its 'render.offscreen' (what Render > Image uses) - premultiplied
           RGBA, so the true accumulated alpha is read and any --bg is composited exactly from one render.

Writes <out-dir>/<image stem>.png per camera (score with `eval_views.py --pred-dir <out-dir> --dataset <dataset>`);
with several --bg values, <out-dir>/<bg>/<stem>.png per background. --bg black|grey|white|R,G,B (0-1; SuperSplat's
default background is black, scene-config.ts DEFAULT_BG_CLR). --scale K renders at K x the resolution with the same
FOV and area-downsamples (a high-DPI screen / zoomed-in window). --move renders from cameras moved off the recorded
path (camera axes right/down/back, dataset units). --editor also screenshots the SuperSplat editor (WebGPU build)
with the PLY loaded at its default view (viewport pixels from the editor's own offscreen render: headless WebGPU
canvases are not composited into screenshots). Alpha: `current` saves <out-dir>/_alpha/<stem>.png and the alpha-hole
fraction (pixels with accumulated alpha < 0.98) in _viewer.json; `pc274` estimates it when both black and white are
rendered (alpha = 1 - max channel difference).

    python scripts/viewer_shots.py --ply splat.ply --dataset <ds> --out-dir shots/ [--every 8] [--move 0 -0.3 1]
    python scripts/viewer_shots.py ... --engine current --bg black grey white --scale 2
"""
from __future__ import annotations
import argparse, base64, functools, http.server, io, json, os, sys, tempfile, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_depth_targets import quat_to_R, read_text_model

PC_URL = "https://cdn.jsdelivr.net/npm/playcanvas@2.7.4/build/playcanvas.mjs"
SUPERSPLAT_URL = "https://superspl.at/editor"
BGS = {"black": (0.0, 0.0, 0.0), "grey": (0.5, 0.5, 0.5), "white": (1.0, 1.0, 1.0)}
HOLE_ALPHA = 0.98
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
window.shot = async (p, q, fov, waitMs, bg) => {
  cam.setPosition(p[0], p[1], p[2]); cam.setRotation(new pc.Quat(q[0], q[1], q[2], q[3])); cam.camera.fov = fov;
  if (bg) cam.camera.clearColor = new pc.Color(bg[0], bg[1], bg[2], 1);
  await new Promise(r => setTimeout(r, waitMs));          // let the async depth sort settle
  return canvas.toDataURL('image/png');
};
</script></body></html>"""

FLAG_SETS = [  # GPU first (ANGLE on Vulkan / EGL), software last
    ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"],
    ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"],
    ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
]
# WebGPU on the A100 nodes: new headless mode + Dawn on Vulkan (old headless / ANGLE-only flags give no adapter)
WEBGPU_FLAGS = ["--headless=new", "--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
                "--use-vulkan", "--use-angle=vulkan", "--disable-vulkan-surface", "--ignore-gpu-blocklist", "--enable-gpu",
                "--disable-features=PrivateNetworkAccessSendPreflights,BlockInsecurePrivateNetworkRequests"]

# today's SuperSplat: find the frame exposing window.scene, then render settled frames at a given pose
SS_FIND = """() => { const s = window.scene; if (!s || !s.events) return null;
  const sp = s.elements.filter(e => e.type === 'splat');
  return {splats: sp.length, n: sp.reduce((a, e) => a + (e.numSplats || 0), 0),
          webgpu: !!s.graphicsDevice.isWebGPU, gpu: String(s.events.invoke('scene.gpu') || ''),
          version: ((document.body.innerText || '').match(/SUPERSPLAT\\s*v[0-9.]+/i) || [''])[0],
          M: sp.length ? Array.from(sp[0].entity.getWorldTransform().data) : null,
          bands: s.events.invoke('view.bands'), minPixelSize: s.events.invoke('view.minPixelSize'),
          targetSize: s.targetSize ? {width: s.targetSize.width, height: s.targetSize.height} : null,
          devicePixelRatio: window.devicePixelRatio}; }"""
SS_SETUP = """() => { const s = window.scene; s.events.fire('view.setStochastic', 'disabled'); return s.events.invoke('view.stochastic'); }"""
SS_SHOT = """async ([p, q, fov, W, H]) => {
  const s = window.scene, V = s.camera.position.constructor, Q = s.camera.mainCamera.getRotation().constructor;
  const pos = new V(p[0], p[1], p[2]), rot = new Q(q[0], q[1], q[2], q[3]);
  const fwd = rot.transformVector(new V(0, 0, -1)), R = s.bound.halfExtents.length();
  const d = new V().sub2(s.bound.center, pos).dot(fwd);             // viewport's fitClippingPlanes
  const far = Math.max(d + R, 1e-2), near = Math.min(1.0, Math.max(d - R, far / (1024 * 16)));
  s.camera.setPoseOverride({position: pos, rotation: rot, fov, near, far});
  const px = await s.events.invoke('render.offscreen', W, H);        // RGBA8 premultiplied, top row first
  let b = ''; for (let i = 0; i < px.length; i += 0x8000) b += String.fromCharCode.apply(null, px.subarray(i, i + 0x8000));
  return btoa(b); }"""


def serve(root):
    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Allow-Headers", "*"); super().end_headers()
        def do_OPTIONS(self):
            self.send_response(204); self.end_headers()
        def log_message(self, *a):
            pass
    class S(http.server.ThreadingHTTPServer):
        def handle_error(self, *a):                                       # browser-cancelled transfers
            pass
    srv = S(("127.0.0.1", 0), functools.partial(H, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def parse_bg(s):
    if s in BGS:
        return s, BGS[s]
    v = tuple(float(x) for x in s.split(","))
    assert len(v) == 3, f"--bg {s}: black|grey|white|R,G,B"
    return s.replace(",", "_"), v


SCENE = "scene.ply"   # served name; keeps the input's extension so SuperSplat picks the right loader (.sog, .compressed.ply, .spz)


def scene_name(path):
    for ext in (".compressed.ply", ".ply", ".sog", ".spz", ".splat", ".ksplat"):
        if path.lower().endswith(ext):
            return "scene" + ext
    return "scene.ply"


def open_editor(pw, url, port, timeout_s=600):
    """Today's SuperSplat (WebGPU) with /scene.ply loaded -> (browser, page, frame, info)."""
    browser = pw.chromium.launch(headless=False, args=WEBGPU_FLAGS)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.on("console", lambda m: m.type == "error" and print("[viewer] editor js:", m.text[:300], flush=True))
    page.goto(f"{url}?load=http://127.0.0.1:{port}/{SCENE}", timeout=120000)
    t0, info = time.time(), None
    while time.time() - t0 < timeout_s:
        for fr in page.frames:
            try:
                info = fr.evaluate(SS_FIND)
            except Exception:
                info = None
            if info and info["n"] > 0:
                time.sleep(3); return browser, page, fr, fr.evaluate(SS_FIND)
        time.sleep(2)
    raise RuntimeError(f"SuperSplat at {url}: no splat loaded after {timeout_s}s (last {info})")


SS_VIEW = """async () => { const s = window.scene, c = s.canvas, r = c.getBoundingClientRect(), W = c.width, H = c.height;
  s.camera.setPoseOverride(null); const px = await s.events.invoke('render.offscreen', W, H);
  const k = s.events.invoke('bgClr');
  let b = ''; for (let i = 0; i < px.length; i += 0x8000) b += String.fromCharCode.apply(null, px.subarray(i, i + 0x8000));
  return {rect: [r.x, r.y, r.width, r.height], W, H, bg: [k.r, k.g, k.b], px: btoa(b)}; }"""


def editor_shot(page, frame, path):
    """Screenshot of the editor UI with its viewport filled by the editor's own render of the current (default) view:
    headless Chromium's WebGPU canvas never reaches the compositor (Dawn needs --disable-vulkan-surface here), so
    the raw viewport is black; the pixels come from 'render.offscreen' (no grid/gizmo overlays), over its bg colour."""
    from PIL import Image
    v = frame.evaluate(SS_VIEW)
    shot = np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB")).astype(np.float64)
    rgba = np.frombuffer(base64.b64decode(v["px"]), np.uint8).reshape(v["H"], v["W"], 4).astype(np.float64)
    view = rgba[..., :3] + (255.0 - rgba[..., 3:]) * np.asarray(v["bg"])
    x, y, w, h = (int(round(t)) for t in v["rect"])
    h, w = min(h, shot.shape[0] - y), min(w, shot.shape[1] - x)
    view = np.asarray(Image.fromarray(np.clip(view, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float64)
    reg = shot[y:y + h, x:x + w]; empty = reg.sum(-1, keepdims=True) < 8              # viewport pixels not under UI
    shot[y:y + h, x:x + w] = np.where(empty, view, reg)
    Image.fromarray(np.clip(np.rint(shot), 0, 255).astype(np.uint8)).save(path)


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
    ap.add_argument("--engine", choices=("pc274", "current"), default="pc274")
    ap.add_argument("--supersplat-url", default=SUPERSPLAT_URL, help="--engine current: the editor build to drive")
    ap.add_argument("--bg", nargs="+", default=["black"], help="black|grey|white|R,G,B; several -> <out-dir>/<bg>/")
    ap.add_argument("--scale", type=int, default=1, help="render at K x resolution (same FOV), area-downsample")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from scipy.spatial.transform import Rotation
    from PIL import Image
    bgs = [parse_bg(b) for b in args.bg]; K = max(1, args.scale)
    cams, imgs = read_text_model(os.path.join(args.dataset, "sparse", "0"))
    imgs = sorted(imgs, key=lambda i: i[0])
    if args.names_from:
        keep = {i[0] for i in read_text_model(os.path.join(args.names_from, "sparse", "0"))[1]}
        imgs = [i for i in imgs if i[0] in keep]
    imgs = imgs[args.offset::args.every]
    # the engine's projection is centred: render a canvas centred on the principal point, then crop W x H.
    # Per camera, so rigs whose cameras have different sizes / principal points are cropped correctly
    # (--engine current renders each shot at its own canvas size; pc274's page has one fixed viewport).
    def canvas(cid):
        W, H, (_, _, cx, cy) = cams[cid]
        CW = int(np.ceil(2 * max(cx, W - cx))); CH = int(np.ceil(2 * max(cy, H - cy)))
        return W, H, CW, CH, int(round(CW / 2 - cx)), int(round(CH / 2 - cy))
    geo = {cid: canvas(cid) for cid in {i[1] for i in imgs}}
    W, H, CW, CH, x0, y0 = geo[imgs[0][1]]
    if args.engine == "pc274" and len(set(geo.values())) > 1:
        raise SystemExit("[viewer] pc274 engine: cameras differ in size/principal point; use --engine current")
    root = tempfile.mkdtemp(prefix="viewer_")
    global SCENE
    SCENE = scene_name(args.ply)
    if args.engine == "pc274" and SCENE != "scene.ply":
        raise SystemExit("[viewer] pc274 engine loads plain .ply only; use --engine current")
    os.symlink(os.path.abspath(args.ply), os.path.join(root, SCENE))
    open(os.path.join(root, "index.html"), "w").write(PAGE % {"W": CW * K, "H": CH * K, "PC": PC_URL})
    port = serve(root); os.makedirs(args.out_dir, exist_ok=True)
    odir = {n: (args.out_dir if len(bgs) == 1 else os.path.join(args.out_dir, n)) for n, _ in bgs}
    for d in odir.values():
        os.makedirs(d, exist_ok=True)
    adir = os.path.join(args.out_dir, "_alpha")

    def finish(arr, g=None):  # K x canvas (h, w, c) -> crop -> area-downsample -> W x H
        W_, H_, _, _, x0_, y0_ = g or (W, H, CW, CH, x0, y0)
        a = arr[y0_ * K:(y0_ + H_) * K, x0_ * K:(x0_ + W_) * K].astype(np.float64)
        return a.reshape(H_, K, W_, K, -1).mean((1, 3)) if K > 1 else a

    holes, meta = [], {}
    with sync_playwright() as pw:
        page = renderer = frame = None
        if args.engine == "pc274":
            for flags in FLAG_SETS:
                browser = pw.chromium.launch(headless=True, args=flags)
                page = browser.new_page(viewport={"width": CW * K, "height": CH * K})
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
        else:
            browser, page, frame, info = open_editor(pw, args.supersplat_url, port)
            meta = {k: v for k, v in info.items() if k != "M"}; meta["stochastic"] = frame.evaluate(SS_SETUP)
            renderer = f"SuperSplat {info['version'] or '?'} WebGPU={info['webgpu']} on {info['gpu']}"
            print(f"[viewer] {renderer}; {info['n']} splats, bands {info['bands']}, min px {info['minPixelSize']}, "
                  f"stochastic {meta['stochastic']}", flush=True)
            if not info["webgpu"]:
                raise SystemExit("[viewer] SuperSplat is not on WebGPU: not today's renderer")
            M = np.asarray(info["M"], dtype=np.float64).reshape(4, 4).T   # splat entity world transform (column-major)
            sM = np.cbrt(abs(np.linalg.det(M[:3, :3]))); RM = M[:3, :3] / sM
            if args.editor:                                               # default view, before any pose override
                editor_shot(page, frame, os.path.join(args.out_dir, "_supersplat_editor.png"))
        t0 = time.time()
        for name, cid, q, t in imgs:
            _, _, (fx, fy, cx, cy) = cams[cid]
            Rcw = quat_to_R(q); C = -Rcw.T @ np.asarray(t)
            if args.move:
                C = C + Rcw.T @ np.array(args.move)
            Rpc = Rcw.T @ np.diag([1.0, -1.0, -1.0])                     # OpenCV camera -> PlayCanvas (x right, y up, -z fwd)
            stem = os.path.splitext(name)[0] + ".png"
            if args.engine == "pc274":
                qq = Rotation.from_matrix(Rpc).as_quat()                  # x, y, z, w
                fov = float(np.degrees(2 * np.arctan(CH / (2 * fy))))     # vertical fov of the centred canvas
                outs = {}
                for n, c in bgs:
                    url = page.evaluate("([p,q,f,w,b]) => window.shot(p,q,f,w,b)",
                                        [C.tolist(), qq.tolist(), fov, args.wait_ms, list(c)])
                    outs[n] = finish(np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")))
                if "black" in outs and "white" in outs:                    # alpha from the white-vs-black difference
                    alpha = 1.0 - (outs["white"] - outs["black"]).max(-1) / 255.0
                    holes.append(float((alpha < HOLE_ALPHA).mean()))
            else:
                Cw = M[:3, :3] @ C + M[:3, 3]; Rw = RM @ Rpc               # into the editor's world (its import rotation)
                qq = Rotation.from_matrix(Rw).as_quat()
                # SuperSplat's camera: horizontal fov when the target is wider than tall (camera.ts horizontalFov)
                g = geo[cid]; cw, ch = g[2], g[3]
                fov = float(np.degrees(2 * np.arctan(cw / (2 * fx)) if cw > ch else 2 * np.arctan(ch / (2 * fy))))
                b64 = frame.evaluate(SS_SHOT, [Cw.tolist(), qq.tolist(), fov, cw * K, ch * K])
                rgba = finish(np.frombuffer(base64.b64decode(b64), np.uint8).reshape(ch * K, cw * K, 4), g)
                a = rgba[..., 3:] / 255.0
                outs = {n: rgba[..., :3] + (1.0 - a) * 255.0 * np.asarray(c) for n, c in bgs}   # premultiplied over bg
                holes.append(float((a[..., 0] < HOLE_ALPHA).mean())); os.makedirs(adir, exist_ok=True)
                Image.fromarray(np.clip(np.rint(rgba[..., 3]), 0, 255).astype(np.uint8)).save(os.path.join(adir, stem))
            for n, im in outs.items():
                Image.fromarray(np.clip(np.rint(im), 0, 255).astype(np.uint8)).save(os.path.join(odir[n], stem))
        print(f"[viewer] {len(imgs)} views in {time.time() - t0:.0f}s -> {args.out_dir}"
              + (f"; alpha-hole fraction {np.mean(holes):.4f}" if holes else ""), flush=True)
        if args.editor and args.engine == "pc274":                       # the real SuperSplat app (WebGPU), default view
            try:
                eb, ed, efr, info = open_editor(pw, args.supersplat_url, port)
                editor_shot(ed, efr, os.path.join(args.out_dir, "_supersplat_editor.png")); eb.close()
                print(f"[viewer] editor ({info['version']}, WebGPU={info['webgpu']}) screenshot ->",
                      os.path.join(args.out_dir, "_supersplat_editor.png"), flush=True)
            except Exception as e:                                        # no WebGPU: whatever the site serves
                print(f"[viewer] WebGPU editor failed ({str(e)[:200]}); plain screenshot", flush=True)
                ed = page.context.browser.new_page(viewport={"width": 1280, "height": 800})
                ed.goto(f"{args.supersplat_url}?load=http://127.0.0.1:{port}/{SCENE}")
                time.sleep(60)
                ed.screenshot(path=os.path.join(args.out_dir, "_supersplat_editor.png"))
    json.dump({"renderer": renderer, "engine": args.engine, "ply": args.ply, "n": len(imgs), "move": args.move,
               "bg": {n: c for n, c in bgs}, "scale": K, **({"editor": meta} if meta else {}),
               **({"alpha_hole_frac": float(np.mean(holes)), "alpha_hole_per_view": holes} if holes else {})},
              open(os.path.join(args.out_dir, "_viewer.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
