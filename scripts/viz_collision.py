"""Offline visualization of a collider + its Isaac collision-test outcome (CPU/matplotlib).

The in-Isaac collision frames put the camera inside the mesh (useless for review). This renders
a clear top-down + oblique view of the collider point cloud with the collision-test results
overlaid: floor-drop probes (green=held / red=fell-through), the detected obstacle footprint,
the drive-into path, and the 4 wall-drive endpoints. Pure CPU, no GPU/Isaac needed.

    python viz_collision.py --collider <dir>/collider.obj --result <dir>/collision_result.json \
        --out fig.png --title "2DGS collider"
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collider", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="collider")
    ap.add_argument("--max-pts", type=int, default=40000)
    args = ap.parse_args()

    import open3d as o3d
    m = o3d.io.read_triangle_mesh(args.collider)
    V = np.asarray(m.vertices)
    if len(V) > args.max_pts:
        V = V[np.random.default_rng(0).choice(len(V), args.max_pts, replace=False)]
    d = json.load(open(args.result))
    fg = d.get("floor_grid", []); obs = d.get("obstacles", [])
    drives = d.get("obstacle_drives", []); walls = d.get("walls", [])

    fig = plt.figure(figsize=(15, 6.6), facecolor="white")
    # ---- panel 1: top-down (x-y), colored by height, with overlays ----
    ax = fig.add_subplot(1, 2, 1)
    order = np.argsort(V[:, 2])
    ax.scatter(V[order, 0], V[order, 1], c=V[order, 2], s=1.2, cmap="viridis", alpha=0.5, linewidths=0)
    for p in fg:
        x, y = p["xy"]; held = not p.get("through")
        ax.scatter([x], [y], c=("#18a558" if held else "#e23b3b"), s=170, marker="o",
                   edgecolors="black", linewidths=1.4, zorder=5)
    for o in obs:
        x, y = o["xy"]; wx, wy = o.get("wx", 0.4), o.get("wy", 0.4)
        ax.add_patch(plt.Rectangle((x - wx / 2, y - wy / 2), wx, wy, fill=False,
                                   edgecolor="#ff8c00", lw=2.2, zorder=4))
    for w in walls:
        ex, ey = w.get("end_xy", [0, 0])
        ax.annotate("", xy=(ex, ey), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=("#b30000" if w.get("fell") else "#0057d8"), lw=2))
        ax.scatter([ex], [ey], marker="x", c=("#b30000" if w.get("fell") else "#0057d8"), s=90, zorder=6)
    ax.set_aspect("equal"); ax.set_title(f"{args.title} — top-down", fontsize=13)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    held = sum(1 for p in fg if not p.get("through"))
    ax.text(0.02, 0.98, f"floor held {held}/{len(fg)}\ngreen=held  red=fell\norange=obstacle  arrows=wall drives",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))

    # ---- panel 2: oblique 3D of the collider ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(V[:, 0], V[:, 1], V[:, 2], c=V[:, 2], s=0.8, cmap="viridis", alpha=0.45, linewidths=0)
    for p in fg:
        x, y = p["xy"]; held = not p.get("through")
        ax2.scatter([x], [y], [0.05], c=("#18a558" if held else "#e23b3b"), s=60,
                    edgecolors="black", linewidths=0.8)
    ax2.set_title(f"{args.title} — oblique 3D", fontsize=13)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z (m)")
    ax2.view_init(elev=22, azim=-60)
    try:
        ax2.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), max(np.ptp(V[:, 2]), 0.5)))
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[viz] wrote {args.out}  (floor held {held}/{len(fg)}, {len(obs)} obstacles, {len(walls)} walls)")


if __name__ == "__main__":
    main()
