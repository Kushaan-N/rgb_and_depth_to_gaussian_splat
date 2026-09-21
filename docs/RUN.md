# Run & view — CEAR → Gaussian Splat → Isaac Sim

One launcher drives everything: **`./launch.sh <command>`** (run from the repo root on Unity).
GPU jobs auto-use the `pi_donghyunkim_umass_edu` account (high fairshare) and the Isaac Sim
6.1.0 image (`$CEAR_WS/isaac-sim-6.1.0.sif`).

```
./launch.sh view          # build the results gallery + serve it (open the printed URL)
./launch.sh robot         # GPU: render the robot navigating the photoreal splat
./launch.sh walkthrough   # GPU: render the photoreal fly-through of the world
./launch.sh live          # GPU: LIVE interactive Isaac session (WebRTC stream)
./launch.sh scene <seq>   # run the pipeline on another CEAR sequence (cleaner scene)
./launch.sh status        # your queued/running jobs
```

---

## 1. See the results (works today, no GPU)

```
cd /work/pi_donghyunkim_umass_edu/kushaan/rgb_and_depth_to_gaussian_splat
./launch.sh view
```
This builds a **single self-contained** `results/results.html` (all videos + frames embedded)
and serves it. Because you're on VS Code Remote-SSH, the port is auto-forwarded:

- **Cmd/Ctrl-click** the `http://localhost:8000/results.html` line it prints, **or** open the
  **PORTS** tab in VS Code and click the globe next to 8000.
- Prefer a file? `./launch.sh open` just writes `results/results.html` — right-click it in the
  VS Code explorer → *Open Preview* / *Simple Browser*, or download it and open locally (it's
  fully portable — every asset is inlined).

What's in it: the robot-in-splat clip + hero frame, the world walkthrough, the NuRec render
proof, and the on-path/off-path reconstruction-quality figure.

---

## 2. Render more (GPU, ~4 min each)

```
./launch.sh robot                      # foreground sweep of the robot through the scene
./launch.sh robot --robot-mode path --robot-start 3 --robot-count 25   # drive down the path
./launch.sh walkthrough                # 90-frame photoreal fly-through
./launch.sh status                     # watch the job
```
Frames land in `$CEAR_OUT/<seq>/navsplat/` (robot) and `.../walkthrough/camera_0/`. Re-run
`./launch.sh view` to rebuild the gallery with the new output. Tunables for `robot`:
`--cam-index` (viewpoint), `--look-ahead` (aim), `--sweep-dist`/`--sweep-range`,
`--robot-size`, `--robot-drop`, `--up-sign` (flip if upside down), `--warmup` (quality).

---

## 3. Navigate the splat live (recommended: viser)

```
./launch.sh viser         # submit + auto-start the login-node relay; prints the PORTS step
```
This launches 3DGRUT's **viser** web viewer (real-time, server-side rendered) and starts a
relay on the login node so the viewer appears as a local port. Then, entirely in **VS Code**:

1. Open the **PORTS** tab (bottom panel, next to TERMINAL).
2. **Forward a Port** → `8090` → Enter.
3. Click the **🌐 globe** on that row → browser opens the viewer.
4. **Drag** to orbit, **WASD** + scroll to fly. `scancel <job>` when done.

Why this and not WebRTC: viser is **HTTP+websocket (TCP)**, so it forwards over your existing
SSH / VS Code connection — no VPN, no keys, no laptop terminal. (The relay exists because the
viewer runs on a compute node; login1 can reach it directly and VS Code forwards the login-node
port.) If the page dies, the `gpu-preempt` job was likely preempted — re-run `./launch.sh viser`.

### Alternative: Isaac WebRTC (`./launch.sh live`)
Full Isaac viewport (robot + physics), but WebRTC media is **UDP**, which a plain SSH tunnel
can't carry — only works on the campus **VPN** (connect VPN, then Isaac WebRTC client →
`<node>:8211`). Prefer `viser` unless you specifically need the Isaac scene.

---

## 4. Cleaner scene (another CEAR sequence)

mocap1 is a dense-foliage patch, so the splat is soft/hazy. A more open, better-covered
sequence reconstructs more crisply. Full recipe for `<seq>` (e.g. `mocap2_well-lit_trot`):
```
source env/cear_env.sh
bash scripts/run_cpu_pipeline.sh <seq>                 # poses, depth, collider, COLMAP, dataset
sbatch -A pi_donghyunkim_umass_edu sbatch/build_3dgrut.sbatch     # (once) build 3DGRUT env
sbatch -A pi_donghyunkim_umass_edu sbatch/train_a100.sbatch <seq> # train the splat
sbatch -A pi_donghyunkim_umass_edu sbatch/export_usd.sbatch <seq> # -> scene_nurec.usdz
python3 scripts/colmap_to_tum.py $CEAR_OUT/<seq>/colmap/sparse/0/images.txt \
        $CEAR_WS/walkthrough.tum 90 3                  # walkthrough trajectory
SEQ=<seq> ./launch.sh robot                            # robot in the new world
```
Data for mocap2/3 is already downloaded on scratch; only the GPU train/export are new.

---

## 5. Validated physics rover — status & path to unify

Physics **is** validated, just in a separate stage from the photoreal render:

- **Proven now:** `scripts/compose_stage.py --mode drop` (Gate 5) drops rigid bodies onto the
  reconstructed **collider** and they rest at the true floor height (err ~2e-8 m); nav mode
  drives a velocity-controlled rover on it (1.42 m). This confirms the world is drivable with
  real PhysX collisions.
- **Why it's separate from the photoreal frame:** the collider is in a Z-up metric frame while
  the splat/`scene_nurec.usdz` is in the COLMAP frame (exported `--no-transform` so the recorded
  camera poses line up with it). The photoreal robot in `nav_in_splat.py` is therefore driven
  **kinematically** along the recorded path.
- **To unify (real physics *in* the photoreal stage):** in the opened splat stage, add a PhysX
  ground plane at the floor height in COLMAP-frame (the trajectory is planar along the detected
  up axis) + a dynamic rover, step PhysX between render ticks, and capture with the same look-at
  camera. The frame reconciliation (collider COLMAP↔Z-up via the pipeline's up-axis rotation)
  is the remaining piece; see `scripts/pose_utils.py` for the transform.

---

## Files
- `launch.sh` — entry point.
- `scripts/make_gallery.py` — self-contained results viewer.
- `scripts/nav_in_splat.py` + `sbatch/nav_in_splat.sbatch` — robot in the photoreal world.
- `scripts/live_stage.py` + `sbatch/isaac_live.sbatch` — live WebRTC session.
- `sbatch/nurec_render_test.sbatch` — photoreal render at recorded poses (walkthrough / proof).
- `scripts/colmap_to_tum.py` — camera trajectory → TUM for the renderer.
- `scripts/compose_stage.py` — physics (Gate 5 drop) + geometry nav.
