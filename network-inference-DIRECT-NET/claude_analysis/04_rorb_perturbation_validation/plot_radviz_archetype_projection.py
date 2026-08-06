"""RadViz-style projection of cells/states onto a hexagon of 6 archetype anchors
(nonNE1/Arc_4, Generalist_nonNE, NE1/Arc_5, NE2/Arc_6, Secretory/Arc_3, nonNE2/Arc_2),
matching the layout of the user's reference figure -- but positioned by HAMMING DISTANCE
to each archetype's average discrete state (bobaT's own representation), not whatever
continuous archetype-signature score the original figure used.

Method: for a state/cell's binarized 53-gene vector, compute Hamming distance to each of
the 6 anchor archetypes, convert to softmax weights over -distance/T (closer archetypes
get more weight), then position = weighted sum of the 6 anchors' hexagon coordinates
(standard barycentric/RadViz combination).

KNOWN DIVERGENCE FROM THE REFERENCE FIGURE (confirmed, not a bug): this method pulls
Generalist_NE's own archetype average state toward the NE1/NE2 edge (it really is
Hamming-close to both, 4-5 bits, and far from everything else), not to the center where
the reference figure places it -- the original figure's method evidently scores
"Generalist" cells as non-specifically low across all 6 named archetypes (centering them),
which is a different construction than literal Hamming closeness. Kept as-is per explicit
decision -- an honest, different, and itself informative view of the discrete state space,
not force-fit to visually match the reference.

Plots organoid_shGFP's real cells (colored by their own predicted.id) as background,
overlaid with the mean walk trajectory (organoid 'Neuroendocrine1' start, knockdown vs.
unperturbed) as a time-colored path.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_radviz_archetype_projection.py
"""

import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
NE1_START_IDX = 5500048749758430
STEP_STRIDE = 20
T = 10.0  # softmax temperature (bits); tunable -- see sanity check in session notes

ANCHOR_ORDER = ["nonNE1", "Generalist_nonNE", "NE1", "NE2", "Secretory", "nonNE2"]
ANCHOR_ARC_MAP = {
    "nonNE1": "Arc_4", "Generalist_nonNE": "Generalist_nonNE", "NE1": "Arc_5",
    "NE2": "Arc_6", "Secretory": "Arc_3", "nonNE2": "Arc_2",
}
ANCHOR_COLORS = {
    "nonNE1": "tab:red", "Generalist_nonNE": "lightcoral", "NE1": "tab:purple",
    "NE2": "darkred", "Secretory": "tab:green", "nonNE2": "orange",
}
# organoid's own predicted.id -> plotting color. Deliberately AVOIDS purple/grey/black --
# those are reserved for the knockdown/unperturbed path colormaps (Purples/Greys), so a
# real-cell color can't be mistaken for a path color.
PREDICTED_ID_COLORS = {
    "Generalist NE": "gold", "Intermediate": "tab:blue", "Neuroendocrine1": "tab:green",
    "Neuroendocrine2": "tab:brown", "nonNE1": "tab:red", "Generalist nonNE": "lightcoral",
    "Stress": "tab:orange",
}


def load_archetype_indices(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    return {a: int("".join(str(int(v)) for v in row), 2) for a, row in avg_states.iterrows()}


def hexagon_anchor_xy():
    angles = {name: np.pi / 2 - i * (2 * np.pi / 6) for i, name in enumerate(ANCHOR_ORDER)}
    return {name: (np.cos(a), np.sin(a)) for name, a in angles.items()}


def project(state_idx, archetype_idx, anchor_xy, t=T):
    dists = np.array([(state_idx ^ archetype_idx[ANCHOR_ARC_MAP[name]]).bit_count() for name in ANCHOR_ORDER])
    w = np.exp(-dists / t)
    w = w / w.sum()
    x = sum(w[i] * anchor_xy[name][0] for i, name in enumerate(ANCHOR_ORDER))
    y = sum(w[i] * anchor_xy[name][1] for i, name in enumerate(ANCHOR_ORDER))
    return x, y


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


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

    # --- Simulated walk trajectories: organoid Neuroendocrine1 start ---
    walks_kd = parse_walk_file(f"{WALK_PATH}/{NE1_START_IDX}/results_RORA_RORB_kd.csv")
    walks_unpert = parse_walk_file(f"{WALK_PATH}/{NE1_START_IDX}/results.csv")

    def mean_path(walks):
        steps = walks[0][::STEP_STRIDE]
        n_steps = len(steps)
        xy_sum = np.zeros((n_steps, 2))
        for w in walks:
            substates = w[::STEP_STRIDE]
            for i, s in enumerate(substates):
                xy_sum[i] += project(s, archetype_idx, anchor_xy)
        return xy_sum / len(walks)

    print("Projecting simulated walk trajectories (mean path)...")
    kd_path = mean_path(walks_kd)
    unpert_path = mean_path(walks_unpert)

    # Zoom region for the inset: bounding box of the real cells + both paths, padded --
    # computed from the ACTUAL data, not eyeballed, so the zoom box is honest about what
    # it covers relative to the full hexagon.
    all_x = np.concatenate([real_df["x"].values, kd_path[:, 0], unpert_path[:, 0]])
    all_y = np.concatenate([real_df["y"].values, kd_path[:, 1], unpert_path[:, 1]])
    pad_x = 0.08 * (all_x.max() - all_x.min())
    pad_y = 0.08 * (all_y.max() - all_y.min())
    zoom_xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    zoom_ylim = (all_y.min() - pad_y, all_y.max() + pad_y)
    print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")

    def draw_content(target_ax, point_size=4, path_lw=2.5, marker_s=80, marker_s_end=120):
        for pid, color in PREDICTED_ID_COLORS.items():
            sub = real_df[real_df["predicted_id"] == pid]
            if len(sub) == 0:
                continue
            target_ax.scatter(sub["x"], sub["y"], s=point_size, alpha=0.35, color=color,
                               label=f"organoid cells: {pid}", zorder=1)

        hexagon_pts = [anchor_xy[name] for name in ANCHOR_ORDER] + [anchor_xy[ANCHOR_ORDER[0]]]
        hx, hy = zip(*hexagon_pts)
        target_ax.plot(hx, hy, color="black", linewidth=1.2, zorder=2)
        for name in ANCHOR_ORDER:
            x, y = anchor_xy[name]
            target_ax.scatter([x], [y], s=200, color=ANCHOR_COLORS[name], edgecolor="black", zorder=4)

        for arc_name, color in [("Generalist_NE", "0.4"), ("Arc_1", "tab:blue")]:
            x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
            target_ax.scatter([x], [y], s=140, color=color, edgecolor="black", marker="*", zorder=5)

        for path, cmap in [(kd_path, plt.cm.Purples), (unpert_path, plt.cm.Greys)]:
            n = len(path)
            colors = cmap(np.linspace(0.3, 1.0, n))
            for i in range(n - 1):
                target_ax.plot(path[i:i+2, 0], path[i:i+2, 1], color=colors[i], linewidth=path_lw, zorder=6)
            target_ax.scatter(*path[0], color=colors[0], s=marker_s, edgecolor="black", zorder=7, marker="o")
            target_ax.scatter(*path[-1], color=colors[-1], s=marker_s_end, edgecolor="black", zorder=7, marker="s")

    # --- Plot: full hexagon (honest, true-scale context) + zoomed inset (readable paths) ---
    # Explicit figure-fraction axes placement (not ax.inset_axes/tight_layout, which fought
    # with the manually-sized legend/note text below and caused overlap) -- main hexagon on
    # the left, zoomed panel on the right, legend + note in their own reserved band at the
    # bottom so nothing can collide regardless of content size.
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.03, 0.22, 0.44, 0.68])
    axins = fig.add_axes([0.52, 0.22, 0.44, 0.68])

    draw_content(ax)
    for name in ANCHOR_ORDER:
        x, y = anchor_xy[name]
        ax.text(x * 1.28, y * 1.28, name, fontsize=10, ha="center", va="center", fontweight="bold")
    for arc_name, color in [("Generalist_NE", "0.4"), ("Arc_1", "tab:blue")]:
        x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
        label = "Intermediate (Arc_1)" if arc_name == "Arc_1" else arc_name
        ax.text(x, y - 0.06, label, fontsize=8.5, ha="center", va="top", fontweight="bold", color=color)

    # Rectangle marking exactly what the inset zooms into -- drawn from the same computed
    # zoom_xlim/zoom_ylim used for the inset's own axis limits, so it cannot misrepresent
    # the inset's extent.
    zoom_rect = Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                           linewidth=1.5, edgecolor="black", facecolor="none", linestyle="--", zorder=8)
    ax.add_patch(zoom_rect)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.55, 1.65)
    ax.set_title("Full projection (true scale)\ndashed box = region magnified at right", fontsize=10)

    draw_content(axins, point_size=10, path_lw=3.5, marker_s=140, marker_s_end=200)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    axins.set_title("Zoomed inset -- same data, magnified\n(paths/cells only legible at this scale)", fontsize=10)

    fig.suptitle(
        "Hamming-distance RadViz projection onto 6 archetype anchors\n"
        "organoid_shGFP real cells (background) + simulated RORA_RORB knockdown walk trajectory (overlay)",
        fontsize=12, y=0.99,
    )

    legend_elements = [
        Line2D([0], [0], color="tab:purple", lw=2.5, label="organoid 'Neuroendocrine1' start, RORA_RORB knockdown (light->dark = early->late)"),
        Line2D([0], [0], color="0.5", lw=2.5, label="organoid 'Neuroendocrine1' start, unperturbed (light->dark = early->late)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path end (step 4000)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.145), fontsize=8.5, ncol=2, frameon=True)

    note = (
        "Note: this projection uses Hamming distance to each archetype's discrete Boolean average state (softmax, T=10), not the continuous\n"
        "archetype-signature scores used in the original reference figure. Confirmed divergence: Generalist_NE's own state projects toward the\n"
        "NE1/NE2 edge here (Hamming-close to both), not to the center as in the reference figure -- shown honestly via the grey star in both panels."
    )
    fig.text(0.5, 0.04, note, ha="center", va="center", fontsize=7.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/radviz_archetype_projection_Neuroendocrine1.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/radviz_archetype_projection_Neuroendocrine1.{{png,pdf}}")


if __name__ == "__main__":
    main()
