# GPU environments (Phases 4-train and 6)

The CPU pipeline (Phases 1-5) uses the venv from `env/make_cpu_venv.sh`. The two GPU stages
have **separate** environments and must not be mixed into that venv.

## 1. 3DGRUT (training + USD export) — A100/H100

```bash
# on a GPU node (or a node with CUDA + GCC<=11 available)
git clone --recursive https://github.com/nv-tlabs/3dgrut.git $CEAR_WS/3dgrut
cd $CEAR_WS/3dgrut
git checkout <PIN_A_COMMIT>          # pin one that includes the cx/cy PINHOLE-intrinsics fix
chmod +x install_env.sh && ./install_env.sh 3dgrut
conda activate 3dgrut
export THREEDGRUT_DIR=$CEAR_WS/3dgrut
```

Requirements (from 3DGRUT docs): Linux, NVIDIA GPU, **CUDA 11.8+**, **GCC 11 or lower**,
Python 3.8+. On Unity:

```bash
module avail gcc                     # look for a GCC 11 module
# if none, install GCC 11 inside the conda env (route NVIDIA's NuRec docs recommend):
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

Use the **3DGUT (rasterization)** config on the A100s (`train_splat.sh` calls it) — it
needs no RT cores. The 3DGRT ray-tracing mode wants an RTX-class GPU (→ L40S).

**Pin and record** (REPORT.md, day one): the 3dgrut commit, CUDA version, GCC version.
Do not upgrade mid-project. The USD export is a beta feature whose schema may change.

## 2. Isaac Sim (compose + walk) — L40S

Isaac Sim wants RTX cores; the A100/H100 have none, so this runs on the L40S. Install Isaac
Sim (pin **one** version day one — 6.0.x vs 6.1; check 6.1 release notes for ParticleField
fixes, §3.4), then point the sbatch at its bundled python:

```bash
export ISAAC_PYTHON=<isaac-sim>/python.sh
```

`compose_stage.py` also uses `trimesh` to read the collider mesh — install it into Isaac's
python if missing (`$ISAAC_PYTHON -m pip install trimesh`).

Known Isaac issues to watch (§3.4): importing two 3DGRUT USDZ files can give incorrect
depth occlusion (reported fixed — first suspect if you see depth weirdness); ParticleField
vs NuRec render tradeoff (export both, compare in-sim).

## 3. Cluster egress

CEAR is Google-Drive hosted and 3dgrut/Isaac pull dependencies from the network. VERIFY
whether **compute nodes** have internet egress or whether downloads must go through a
login/data-transfer node (the login node here does have egress). Do large downloads and
`install_env.sh` where egress exists, onto `$CEAR_WS` (scratch).
