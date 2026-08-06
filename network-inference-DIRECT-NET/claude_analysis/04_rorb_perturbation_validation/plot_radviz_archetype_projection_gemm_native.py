"""Same walks-only RadViz plot as plot_radviz_archetype_projection.py (see that file's
docstring for the full projection method), but for GEMM's own NATIVE long walks instead of
organoid-seeded ones -- i.e. walks that start already inside one of GEMM's own fitted
attractor basins, not from an organoid-derived starting state.

Starting archetype: Arc_5 (NE1) -- the direct GEMM-native counterpart to the organoid
'Neuroendocrine1' start used in the companion plot, so the two are a matched pair (same
target biological identity, GEMM-native start vs. organoid-derived start).

GEMM's filtered attractor basins are not single states -- each archetype's basin
(`bb.utils.get_attractor_dict(..., filtered=True)`) is a *set* of individually-discovered
attractor member states (Arc_5 has 11), and `long_random_walks` was originally run
separately from each member (`6667/walks/long_walks/4000_step_walks/<member_idx>/`, 100
iters per member for most members). This script pools all of Arc_5's member-seeded walks
together into one set of individual walks per condition, exactly analogous to organoid's
100 pooled walks in the companion plot.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_radviz_archetype_projection_gemm_native.py
"""

import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"
WALK_PATH = f"{DIR_PREFIX}/6667/walks/long_walks/4000_step_walks"
ATTRACTOR_DIR = f"{DIR_PREFIX}/6667/attractors/attractors_threshold_0.5"
START_ARCHETYPE = "Arc_5"  # NE1 -- GEMM-native counterpart to organoid's 'Neuroendocrine1'
STEP_STRIDE = 20
T = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0  # softmax temperature (bits); pass as CLI arg to override
OUT_SUFFIX = f"_T{sys.argv[1]}" if len(sys.argv) > 1 else ""  # keeps the default-T output file untouched

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


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    archetype_idx = load_archetype_indices(nodes)
    anchor_xy = hexagon_anchor_xy()

    attractor_dict = bb.utils.get_attractor_dict(ATTRACTOR_DIR, filtered=True)
    member_idxs = attractor_dict[START_ARCHETYPE]
    print(f"{START_ARCHETYPE} basin has {len(member_idxs)} member states: {member_idxs}")

    walks_kd, walks_unpert = [], []
    for member_idx in member_idxs:
        member_dir = f"{WALK_PATH}/{member_idx}"
        kd_file = f"{member_dir}/results_RORA_RORB_kd.csv"
        unpert_file = f"{member_dir}/results.csv"
        if not (os.path.exists(kd_file) and os.path.exists(unpert_file)):
            print(f"  skipping member {member_idx}: missing walk output")
            continue
        member_kd = parse_walk_file(kd_file)
        member_unpert = parse_walk_file(unpert_file)
        print(f"  member {member_idx}: {len(member_kd)} kd walks, {len(member_unpert)} unperturbed walks")
        walks_kd.extend(member_kd)
        walks_unpert.extend(member_unpert)

    def project_all(walks):
        return [np.array([project(s, archetype_idx, anchor_xy) for s in w[::STEP_STRIDE]]) for w in walks]

    def mean_path(paths):
        # Individual walks can differ in length by a step or two -- truncate to the
        # shortest so the mean is computed over a consistent number of time-points.
        min_len = min(len(p) for p in paths)
        return np.mean(np.stack([p[:min_len] for p in paths], axis=0), axis=0)

    print(f"Pooled: {len(walks_kd)} knockdown + {len(walks_unpert)} unperturbed walks across all {START_ARCHETYPE} members...")
    kd_paths = project_all(walks_kd)
    unpert_paths = project_all(walks_unpert)
    kd_mean = mean_path(kd_paths)
    unpert_mean = mean_path(unpert_paths)

    all_x = np.concatenate([p[:, 0] for p in kd_paths + unpert_paths])
    all_y = np.concatenate([p[:, 1] for p in kd_paths + unpert_paths])
    pad_x = 0.08 * (all_x.max() - all_x.min())
    pad_y = 0.08 * (all_y.max() - all_y.min())
    zoom_xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    zoom_ylim = (all_y.min() - pad_y, all_y.max() + pad_y)
    print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")

    def draw_content(target_ax, path_lw=2.5, spaghetti_lw=0.5, marker_s=80, marker_s_end=120, archetype_s=140):
        hexagon_pts = [anchor_xy[name] for name in ANCHOR_ORDER] + [anchor_xy[ANCHOR_ORDER[0]]]
        hx, hy = zip(*hexagon_pts)
        target_ax.plot(hx, hy, color="black", linewidth=1.2, zorder=2)
        for name in ANCHOR_ORDER:
            x, y = anchor_xy[name]
            target_ax.scatter([x], [y], s=200, color=ANCHOR_COLORS[name], edgecolor="black", zorder=4)

        for arc_name, color in EXTRA_ARCHETYPE_MARKERS:
            x, y = project(archetype_idx[arc_name], archetype_idx, anchor_xy)
            target_ax.scatter([x], [y], s=archetype_s, color=color, edgecolor="black", marker="o", zorder=5)

        for p in unpert_paths:
            target_ax.plot(p[:, 0], p[:, 1], color="0.5", linewidth=spaghetti_lw, alpha=0.05, zorder=3)
        for p in kd_paths:
            target_ax.plot(p[:, 0], p[:, 1], color="tab:purple", linewidth=spaghetti_lw, alpha=0.05, zorder=3)

        for path, cmap in [(kd_mean, plt.cm.Purples), (unpert_mean, plt.cm.Greys)]:
            n = len(path)
            colors = cmap(np.linspace(0.3, 1.0, n))
            for i in range(n - 1):
                target_ax.plot(path[i:i+2, 0], path[i:i+2, 1], color=colors[i], linewidth=path_lw, zorder=6)
            target_ax.scatter(*path[0], color=colors[0], s=marker_s, edgecolor="black", zorder=7, marker="o")
            target_ax.scatter(*path[-1], color=colors[-1], s=marker_s_end, edgecolor="black", zorder=7, marker="s")

    fig = plt.figure(figsize=(14, 9.5))
    ax = fig.add_axes([0.03, 0.32, 0.44, 0.60])
    axins = fig.add_axes([0.52, 0.32, 0.44, 0.60])

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

    draw_content(axins, path_lw=3.5, spaghetti_lw=0.8, marker_s=140, marker_s_end=200, archetype_s=220)
    axins.set_xlim(*zoom_xlim)
    axins.set_ylim(*zoom_ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    axins.set_title("Zoomed inset -- same data, magnified\n(individual walks only legible at this scale)", fontsize=10)

    fig.suptitle(
        f"Hamming-distance RadViz projection onto 6 archetype anchors (softmax T={T:g})\n"
        f"Simulated RORA_RORB knockdown vs. unperturbed walk trajectories (GEMM-native {START_ARCHETYPE}/NE1 start)",
        fontsize=12, y=0.99,
    )

    legend_elements = [
        Line2D([0], [0], color="tab:purple", lw=2.5, label="RORA_RORB knockdown: mean path (light->dark = early->late)"),
        Line2D([0], [0], color="0.5", lw=2.5, label="Unperturbed: mean path (light->dark = early->late)"),
        Line2D([0], [0], color="tab:purple", lw=1.0, alpha=0.4, label=f"RORA_RORB knockdown: {len(kd_paths)} individual walks (pooled across {len(member_idxs)} {START_ARCHETYPE} attractor members)"),
        Line2D([0], [0], color="0.5", lw=1.0, alpha=0.4, label=f"Unperturbed: {len(unpert_paths)} individual walks (pooled across {len(member_idxs)} {START_ARCHETYPE} attractor members)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path end (step 4000)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="Generalist_NE (archetype average state, not a hexagon vertex)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="black", markersize=9, label="Intermediate/Arc_1 (archetype average state, not a hexagon vertex)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.215), fontsize=8, ncol=2, frameon=True)

    note = (
        f"Note: starts are GEMM's own {START_ARCHETYPE} basin -- {len(member_idxs)} individually-discovered attractor member states, each seeding its own\n"
        f"set of long walks; all pooled here (unlike the organoid-seeded companion plot, which starts from one representative state). Every labeled\n"
        f"point (6 hexagon vertices + the 2 circles) is one of bobaT's 8 fitted archetype average states, projected by Hamming distance to the 6 vertex\n"
        f"archetypes (softmax, T={T:g}) -- not the continuous archetype-signature scores used in the original reference figure."
    )
    fig.text(0.5, 0.06, note, ha="center", va="center", fontsize=7.5,
              bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/radviz_archetype_projection_gemm_{START_ARCHETYPE}{OUT_SUFFIX}.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/radviz_archetype_projection_gemm_{START_ARCHETYPE}{OUT_SUFFIX}.{{png,pdf}}")


if __name__ == "__main__":
    main()
