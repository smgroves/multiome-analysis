"""RadViz-style projection of simulated walk trajectories onto a hexagon of 6 archetype
anchors (nonNE1/Arc_4, Generalist_nonNE, NE1/Arc_5, NE2/Arc_6, Secretory/Arc_3,
nonNE2/Arc_2), matching the layout of the user's reference figure -- but positioned by
HAMMING DISTANCE to each archetype's average discrete state (bobaT's own representation),
not whatever continuous archetype-signature score the original figure used.

Method: for a state's binarized 53-gene vector, compute Hamming distance to each of the 6
anchor archetypes, convert to softmax weights over -distance/T (closer archetypes get more
weight), then position = weighted sum of the 6 anchors' hexagon coordinates (standard
barycentric/RadViz combination).

Every labeled point in this figure -- the 6 hexagon vertices AND the 2 circles inside
(Generalist_NE, Intermediate/Arc_1) -- is one of bobaT's 8 fitted archetype average states
(`6667/attractors/average_states.txt`), projected by this same Hamming-distance method.
The hexagon vertices are 6 of those 8 states, chosen (per the reference figure's layout) to
define the hexagon's corners; Generalist_NE and Arc_1/Intermediate are the remaining 2 of
the 8 and are plotted as circles rather than corners, using the identical projection -- the
circle marker is deliberately the same shape as the vertex markers to signal they're the
same *kind* of reference point (a fitted archetype average state), just not used to anchor
the hexagon's geometry.

KNOWN DIVERGENCE FROM THE REFERENCE FIGURE (confirmed, not a bug): this method pulls
Generalist_NE's own archetype average state toward the NE1/NE2 edge (it really is
Hamming-close to both, 4-5 bits, and far from everything else), not to the center where
the reference figure places it -- the original figure's method evidently scores
"Generalist" cells as non-specifically low across all 6 named archetypes (centering them),
which is a different construction than literal Hamming closeness. Kept as-is per explicit
decision -- an honest, different, and itself informative view of the discrete state space,
not force-fit to visually match the reference.

Plots simulated walk trajectories only (organoid 'Neuroendocrine1' start, knockdown vs.
unperturbed): all 100 individual walks per condition (thin, translucent) plus the bold
mean path (time-colored, light->dark = early->late).

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
# The 2 of bobaT's 8 archetype average states not used as hexagon vertices (states #7-8,
# using the ordering: Arc_2..Arc_6 + Generalist_nonNE = the 6 vertices = states #1-6).
EXTRA_ARCHETYPE_MARKERS = [("Generalist_NE", "0.4"), ("Arc_1", "tab:blue")]


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

    # --- Simulated walk trajectories: organoid Neuroendocrine1 start ---
    walks_kd = parse_walk_file(f"{WALK_PATH}/{NE1_START_IDX}/results_RORA_RORB_kd.csv")
    walks_unpert = parse_walk_file(f"{WALK_PATH}/{NE1_START_IDX}/results.csv")

    def project_all(walks):
        return [np.array([project(s, archetype_idx, anchor_xy) for s in w[::STEP_STRIDE]]) for w in walks]

    def mean_path(paths):
        return np.mean(np.stack(paths, axis=0), axis=0)

    print(f"Projecting {len(walks_kd)} knockdown + {len(walks_unpert)} unperturbed walks...")
    kd_paths = project_all(walks_kd)
    unpert_paths = project_all(walks_unpert)
    kd_mean = mean_path(kd_paths)
    unpert_mean = mean_path(unpert_paths)

    # Zoom region for the inset: bounding box of all individual walk paths (both conditions),
    # padded -- computed from the ACTUAL data, not eyeballed, so the zoom box is honest about
    # what it covers relative to the full hexagon.
    all_x = np.concatenate([p[:, 0] for p in kd_paths + unpert_paths])
    all_y = np.concatenate([p[:, 1] for p in kd_paths + unpert_paths])
    pad_x = 0.08 * (all_x.max() - all_x.min())
    pad_y = 0.08 * (all_y.max() - all_y.min())
    zoom_xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    zoom_ylim = (all_y.min() - pad_y, all_y.max() + pad_y)
    print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")

    def draw_content(target_ax, path_lw=2.5, spaghetti_lw=0.6, marker_s=80, marker_s_end=120, archetype_s=140):
        hexagon_pts = [anchor_xy[name] for name in ANCHOR_ORDER] + [anchor_xy[ANCHOR_ORDER[0]]]
        hx, hy = zip(*hexagon_pts)
        target_ax.plot(hx, hy, color="black", linewidth=1.2, zorder=2)
        for name in ANCHOR_ORDER:
            x, y = anchor_xy[name]
            target_ax.scatter([x], [y], s=200, color=ANCHOR_COLORS[name], edgecolor="black", zorder=4)

        for arc_name, color in EXTRA_ARCHETYPE_MARKERS:
            x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
            target_ax.scatter([x], [y], s=archetype_s, color=color, edgecolor="black", marker="o", zorder=5)

        # Individual walks: thin, translucent spaghetti showing the full spread.
        for p in unpert_paths:
            target_ax.plot(p[:, 0], p[:, 1], color="0.5", linewidth=spaghetti_lw, alpha=0.08, zorder=3)
        for p in kd_paths:
            target_ax.plot(p[:, 0], p[:, 1], color="tab:purple", linewidth=spaghetti_lw, alpha=0.08, zorder=3)

        # Bold mean path per condition, time-colored light->dark = early->late.
        for path, cmap in [(kd_mean, plt.cm.Purples), (unpert_mean, plt.cm.Greys)]:
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
    for arc_name, color in EXTRA_ARCHETYPE_MARKERS:
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

    draw_content(axins, path_lw=3.5, spaghetti_lw=1.0, marker_s=140, marker_s_end=200, archetype_s=220)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    axins.set_title("Zoomed inset -- same data, magnified\n(individual walks only legible at this scale)", fontsize=10)

    fig.suptitle(
        "Hamming-distance RadViz projection onto 6 archetype anchors\n"
        "Simulated RORA_RORB knockdown vs. unperturbed walk trajectories (organoid 'Neuroendocrine1' start)",
        fontsize=12, y=0.99,
    )

    legend_elements = [
        Line2D([0], [0], color="tab:purple", lw=2.5, label="RORA_RORB knockdown: mean path (light->dark = early->late)"),
        Line2D([0], [0], color="0.5", lw=2.5, label="Unperturbed: mean path (light->dark = early->late)"),
        Line2D([0], [0], color="tab:purple", lw=1.0, alpha=0.4, label="RORA_RORB knockdown: 100 individual walks"),
        Line2D([0], [0], color="0.5", lw=1.0, alpha=0.4, label="Unperturbed: 100 individual walks"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path end (step 4000)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="Generalist_NE (archetype average state, not a hexagon vertex)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="black", markersize=9, label="Intermediate/Arc_1 (archetype average state, not a hexagon vertex)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.145), fontsize=8, ncol=2, frameon=True)

    note = (
        "Note: every labeled point (6 hexagon vertices + the 2 circles) is one of bobaT's 8 fitted archetype average states, projected by Hamming\n"
        "distance to the 6 vertex archetypes (softmax, T=10) -- not the continuous archetype-signature scores used in the original reference figure.\n"
        "Confirmed divergence: Generalist_NE's own state projects toward the NE1/NE2 edge here (Hamming-close to both), not to the center as in the\n"
        "reference figure -- shown honestly via its own circle marker rather than forced to the center."
    )
    fig.text(0.5, 0.04, note, ha="center", va="center", fontsize=7.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/radviz_archetype_projection_Neuroendocrine1.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/radviz_archetype_projection_Neuroendocrine1.{{png,pdf}}")


if __name__ == "__main__":
    main()
