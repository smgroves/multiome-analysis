"""Secondary RadViz-style plot (companion to plot_radviz_archetype_projection.py): same
hexagon of 6 archetype anchors, positioned by Hamming distance to bobaT's fitted archetype
average states (see that script's docstring for the full method + the documented
divergence from the reference figure). This version puts organoid_shGFP's REAL cells back
on as a reference, but summarized as a per-predicted.id KDE contour rather than a raw
scatter cloud -- the goal is a single readable question: does organoid's own "nonNE"-type
labeling actually land near GEMM's nonNE anchors (and same for the other labels), not
whether individual dots are legible.

Method for each organoid predicted.id group:
  1. Binarize each cell, project onto the hexagon exactly as in the primary script.
  2. Fit a 2D Gaussian KDE (scipy) over that group's projected points.
  3. Evaluate the KDE on a shared grid (same grid for every group, so contours are
     directly comparable), then draw the 50% highest-density-region (HDR) contour -- the
     smallest region containing 50% of that group's estimated probability mass. One
     contour per group avoids the clutter of many overlapping density lines while still
     showing each group's "core" location and rough shape.

No walk trajectories on this plot -- that comparison lives in the primary (walks-only)
plot. This one is purely: real organoid archetype labels vs. GEMM's fixed archetype
anchors.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_radviz_archetype_kde_reference.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import gaussian_kde

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"
T = 10.0  # softmax temperature (bits) -- matches the primary RadViz script
HDR_MASS = 0.5  # fraction of each group's estimated probability mass enclosed by its contour
MIN_GROUP_N = 20
GRID_N = 200

ANCHOR_ORDER = ["nonNE1", "Generalist_nonNE", "NE1", "NE2", "Secretory", "nonNE2"]
ANCHOR_ARC_MAP = {
    "nonNE1": "Arc_4", "Generalist_nonNE": "Generalist_nonNE", "NE1": "Arc_5",
    "NE2": "Arc_6", "Secretory": "Arc_3", "nonNE2": "Arc_2",
}
# Colors matched to the user's reference figure (ref_hex).
ANCHOR_COLORS = {
    "nonNE1": "tab:red", "Generalist_nonNE": "lightcoral", "NE1": "tab:purple",
    "NE2": "darkred", "Secretory": "tab:green", "nonNE2": "orange",
}
EXTRA_ARCHETYPE_MARKERS = [("Generalist_NE", "0.4"), ("Arc_1", "tab:blue")]

# organoid's own predicted.id -> KDE contour color, matched to ANCHOR_COLORS/
# EXTRA_ARCHETYPE_MARKERS above (i.e. ref_hex) wherever a same-named archetype exists.
# Stress has no ref_hex counterpart, so it keeps a distinct neutral color.
PREDICTED_ID_COLORS = {
    "Generalist NE": "0.4", "Intermediate": "tab:blue", "Neuroendocrine1": "tab:purple",
    "Neuroendocrine2": "darkred", "nonNE1": "tab:red", "Generalist nonNE": "lightcoral",
    "Stress": "tab:gray",
}


def load_archetype_indices(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    return {a: int("".join(str(int(v)) for v in row), 2) for a, row in avg_states.iterrows()}


def hexagon_anchor_xy():
    angles = {name: np.pi / 2 + np.pi / 6 - i * (2 * np.pi / 6) for i, name in enumerate(ANCHOR_ORDER)}
    return {name: (np.cos(a), np.sin(a)) for name, a in angles.items()}


def project(state_idx, archetype_idx, anchor_xy, t=T):
    dists = np.array([(state_idx ^ archetype_idx[ANCHOR_ARC_MAP[name]]).bit_count() for name in ANCHOR_ORDER])
    w = np.exp(-dists / t)
    w = w / w.sum()
    x = sum(w[i] * anchor_xy[name][0] for i, name in enumerate(ANCHOR_ORDER))
    y = sum(w[i] * anchor_xy[name][1] for i, name in enumerate(ANCHOR_ORDER))
    return x, y


def hdr_level(kde, grid_xy, cell_area, mass=HDR_MASS):
    """Density threshold whose super-level set encloses `mass` fraction of probability."""
    dens = kde(grid_xy)
    order = np.argsort(dens)[::-1]
    cum_mass = np.cumsum(dens[order]) * cell_area
    idx = np.searchsorted(cum_mass, mass)
    idx = min(idx, len(dens) - 1)
    return dens[order][idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    archetype_idx = load_archetype_indices(nodes)
    anchor_xy = hexagon_anchor_xy()

    # --- Real organoid_shGFP cells, binarized per-cell, projected ---
    data = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    clusters = pd.read_csv(f"{DIR_PREFIX}/data/organoid/organoid_clusters.csv", index_col=0).reindex(data.index)

    binaries = bb.utils.binarize_data_df(data, nodes, threshold=0.5)
    cell_idx = binaries.apply(lambda row: int("".join(str(int(v)) for v in row), 2), axis=1)

    print(f"Projecting {len(cell_idx)} real organoid_shGFP cells...")
    real_xy = np.array([project(idx, archetype_idx, anchor_xy) for idx in cell_idx])
    real_df = pd.DataFrame(real_xy, columns=["x", "y"], index=data.index)
    real_df["predicted_id"] = clusters["predicted.id"].values

    group_sizes = real_df["predicted_id"].value_counts()
    print(f"predicted.id group sizes:\n{group_sizes}")
    all_groups = {pid: sub for pid, sub in real_df.groupby("predicted_id") if pid in PREDICTED_ID_COLORS}
    groups = {pid: sub for pid, sub in all_groups.items() if len(sub) >= MIN_GROUP_N}
    # Groups too small for a reliable KDE (e.g. n=1-16 here, since organoid_shGFP is the
    # untreated control and is overwhelmingly NE-like) are still shown, as raw points rather
    # than a contour that would misrepresent how little data backs it.
    scatter_groups = {pid: sub for pid, sub in all_groups.items() if len(sub) < MIN_GROUP_N}
    if scatter_groups:
        print(f"Too few cells for a KDE (<{MIN_GROUP_N}), plotting as scatter instead: "
              f"{ {pid: len(sub) for pid, sub in scatter_groups.items()} }")

    # Shared grid (full hexagon extent) so every group's KDE/contour is evaluated identically.
    gx = np.linspace(-1.7, 1.7, GRID_N)
    gy = np.linspace(-1.55, 1.65, GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.vstack([GX.ravel(), GY.ravel()])
    cell_area = (gx[1] - gx[0]) * (gy[1] - gy[0])

    # Zoom region for the inset: bounding box of the real cells' 50%-HDR contours' extent
    # is unknown ahead of time, so use the full cell cloud (all groups combined) instead --
    # honest and simple, computed from the actual data.
    pad_x = 0.08 * (real_df["x"].max() - real_df["x"].min())
    pad_y = 0.08 * (real_df["y"].max() - real_df["y"].min())
    zoom_xlim = (real_df["x"].min() - pad_x, real_df["x"].max() + pad_x)
    zoom_ylim = (real_df["y"].min() - pad_y, real_df["y"].max() + pad_y)
    print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")

    kdes = {}
    for pid, sub in groups.items():
        kde = gaussian_kde(np.vstack([sub["x"], sub["y"]]))
        level = hdr_level(kde, grid_xy, cell_area)
        dens_grid = kde(grid_xy).reshape(GX.shape)
        kdes[pid] = (dens_grid, level)
        print(f"  {pid}: n={len(sub)}, 50%-HDR density level={level:.4f}")

    def draw_content(target_ax, contour_lw=2.0, archetype_s=140, scatter_s=40):
        for pid, (dens_grid, level) in kdes.items():
            color = PREDICTED_ID_COLORS[pid]
            target_ax.contour(GX, GY, dens_grid, levels=[level], colors=[color], linewidths=contour_lw, zorder=3)
            target_ax.contourf(GX, GY, dens_grid, levels=[level, dens_grid.max()], colors=[color], alpha=0.12, zorder=1)

        for pid, sub in scatter_groups.items():
            target_ax.scatter(sub["x"], sub["y"], s=scatter_s, color=PREDICTED_ID_COLORS[pid],
                               edgecolor="black", linewidth=0.5, marker="x" if len(sub) == 1 else "o",
                               alpha=0.9, zorder=3)

        hexagon_pts = [anchor_xy[name] for name in ANCHOR_ORDER] + [anchor_xy[ANCHOR_ORDER[0]]]
        hx, hy = zip(*hexagon_pts)
        target_ax.plot(hx, hy, color="black", linewidth=1.2, zorder=4)
        for name in ANCHOR_ORDER:
            x, y = anchor_xy[name]
            target_ax.scatter([x], [y], s=200, color=ANCHOR_COLORS[name], edgecolor="black", zorder=5)
        for arc_name, color in EXTRA_ARCHETYPE_MARKERS:
            x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
            target_ax.scatter([x], [y], s=archetype_s, color=color, edgecolor="black", marker="o", zorder=5)

    fig = plt.figure(figsize=(14, 10.5))
    ax = fig.add_axes([0.03, 0.38, 0.44, 0.55])
    axins = fig.add_axes([0.52, 0.38, 0.44, 0.55])

    draw_content(ax)
    for name in ANCHOR_ORDER:
        x, y = anchor_xy[name]
        ax.text(x * 1.28, y * 1.28, name, fontsize=10, ha="center", va="center", fontweight="bold")
    for arc_name, color in EXTRA_ARCHETYPE_MARKERS:
        x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
        label = "Intermediate (Arc_1)" if arc_name == "Arc_1" else arc_name
        ax.text(x, y - 0.06, label, fontsize=8.5, ha="center", va="top", fontweight="bold", color=color)

    zoom_rect = Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                           linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--", zorder=8)
    ax.add_patch(zoom_rect)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.55, 1.65)
    ax.set_title("Full projection (true scale)\ndashed box = region magnified at right", fontsize=10)

    draw_content(axins, contour_lw=2.5, archetype_s=220)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    axins.set_title("Zoomed inset -- same data, magnified", fontsize=10)

    fig.suptitle(
        "Hamming-distance RadViz projection: organoid_shGFP real cells (by predicted.id, 50% HDR contours)\n"
        "vs. GEMM's fixed archetype anchors",
        fontsize=12, y=0.99,
    )

    legend_elements = [
        Line2D([0], [0], color=PREDICTED_ID_COLORS[pid], lw=2.0,
               label=f"organoid \"{pid}\" (n={len(groups[pid])}), 50% HDR contour")
        for pid in groups
    ] + [
        Line2D([0], [0], marker=("x" if len(sub) == 1 else "o"), color="w", markerfacecolor=PREDICTED_ID_COLORS[pid],
               markeredgecolor="black" if len(sub) > 1 else PREDICTED_ID_COLORS[pid], markeredgewidth=1.5 if len(sub) == 1 else 1,
               markersize=8, label=f"organoid \"{pid}\" (n={len(sub)}, too few for KDE -- shown as points)")
        for pid, sub in scatter_groups.items()
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black", markersize=9,
               label=f"GEMM {arc_name} (archetype average state, color-matched to organoid \"{'Generalist NE' if arc_name == 'Generalist_NE' else 'Intermediate'}\")")
        for arc_name, color in EXTRA_ARCHETYPE_MARKERS
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.30), fontsize=7.5, ncol=3, frameon=True)

    note = (
        "Note: each contour is the smallest region containing 50% of that organoid predicted.id group's estimated probability mass (2D Gaussian KDE\n"
        "over the group's Hamming-distance projection), evaluated on a shared grid so contours are directly comparable across groups. Groups with\n"
        f"<{MIN_GROUP_N} cells are shown as raw points instead, since a KDE from that few cells would be unreliable. Hexagon vertices + the 2 circles\n"
        "are GEMM's own fitted archetype average states, colored to match their same-named organoid group where one exists (Secretory and nonNE2\n"
        "have no organoid predicted.id counterpart here, so keep distinct colors). See plot_radviz_archetype_projection.py for the full projection\n"
        "method and the confirmed Generalist_NE centering divergence from the original reference figure."
    )
    fig.text(0.5, 0.09, note, ha="center", va="center", fontsize=7.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/radviz_archetype_kde_reference.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/radviz_archetype_kde_reference.{{png,pdf}}")


if __name__ == "__main__":
    main()
