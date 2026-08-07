"""Hexagon RadViz plot (same method as plot_radviz_archetype_projection.py -- see that
file's docstring) for each of the 5 new real-cell-seeded starting points: GEMM NE1 (item
3, unperturbed vs. RORA_RORB knockdown), GEMM's 3 nonNE archetypes (item 4, unperturbed
vs. RORA_RORB overexpression -- does forcing RORB back on push back toward NE?), and
organoid's real shRORB "Generalist nonNE" cells (item 7, same overexpression/rescue
question but starting from real perturbed-experiment cells instead of a GEMM archetype).

Only 2 conditions per plot here (unperturbed + one perturbation) -- no ASCL1 comparison,
since these starts aren't part of that comparison.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_new_start_hexagon.py
"""

import ast
import os

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
NEW_WALK_DIR = f"{DIR_PREFIX}/6667/new_starts_seeded/walks/long_walks/4000_step_walks"
ORGANOID_WALK_DIR = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
STEP_STRIDE = 20
T = 10.0

# (out_subdir, out_name, label, walk_dir, start_idx, pert_file, pert_display, pert_color)
SPECS = [
    ("gemm_ne1_start", "hexagon_GEMM_NE1", "GEMM NE1 real cell", NEW_WALK_DIR, 8914049663948766,
     "results_RORA_RORB_kd.csv", "RORA_RORB knockdown", "tab:purple"),
    ("gemm_nonne_start_oe", "hexagon_GEMM_Generalist_nonNE", "GEMM Generalist_nonNE real cell", NEW_WALK_DIR, 1222330852650596,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("gemm_nonne_start_oe", "hexagon_GEMM_Arc4_nonNE1", "GEMM nonNE1 (Arc_4) real cell", NEW_WALK_DIR, 3474165076405860,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("gemm_nonne_start_oe", "hexagon_GEMM_Arc2_nonNE2", "GEMM nonNE2 (Arc_2) real cell", NEW_WALK_DIR, 3474165059628640,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression", "tab:green"),
    ("organoid_shrorb_nonne_oe", "hexagon_organoid_shRORB_Generalist_nonNE", "organoid shRORB Generalist_nonNE", ORGANOID_WALK_DIR, 1371881672649956,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression (rescue?)", "tab:green"),
]

ANCHOR_ORDER = ["nonNE1", "Generalist_nonNE", "NE1", "NE2", "Secretory", "nonNE2"]
ANCHOR_ARC_MAP = {
    "nonNE1": "Arc_4", "Generalist_nonNE": "Generalist_nonNE", "NE1": "Arc_5",
    "NE2": "Arc_6", "Secretory": "Arc_3", "nonNE2": "Arc_2",
}
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
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    archetype_idx = load_archetype_indices(nodes)
    anchor_xy = hexagon_anchor_xy()

    for out_subdir, out_name, label, walk_dir, start_idx, pert_file, pert_display, pert_color in SPECS:
        out_dir = f"{DIR_PREFIX}/comparisons/organoid_walks/{out_subdir}"
        os.makedirs(out_dir, exist_ok=True)

        walks_pert = parse_walk_file(f"{walk_dir}/{start_idx}/{pert_file}")
        walks_unpert = parse_walk_file(f"{walk_dir}/{start_idx}/results.csv")

        def project_all(walks):
            return [np.array([project(s, archetype_idx, anchor_xy) for s in w[::STEP_STRIDE]]) for w in walks]

        def mean_path(paths):
            return np.mean(np.stack(paths, axis=0), axis=0)

        print(f"[{label}] Projecting {len(walks_pert)} {pert_display} + {len(walks_unpert)} unperturbed walks...")
        pert_paths = project_all(walks_pert)
        unpert_paths = project_all(walks_unpert)
        pert_mean = mean_path(pert_paths)
        unpert_mean = mean_path(unpert_paths)

        all_x = np.concatenate([p[:, 0] for p in pert_paths + unpert_paths])
        all_y = np.concatenate([p[:, 1] for p in pert_paths + unpert_paths])
        pad_x = 0.08 * (all_x.max() - all_x.min())
        pad_y = 0.08 * (all_y.max() - all_y.min())
        zoom_xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
        zoom_ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

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

            for p in unpert_paths:
                target_ax.plot(p[:, 0], p[:, 1], color="0.5", linewidth=spaghetti_lw, alpha=0.08, zorder=3)
            for p in pert_paths:
                target_ax.plot(p[:, 0], p[:, 1], color=pert_color, linewidth=spaghetti_lw, alpha=0.08, zorder=3)

            for path, cmap in [(pert_mean, plt.cm.Greens if pert_color == "tab:green" else plt.cm.Purples), (unpert_mean, plt.cm.Greys)]:
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
            arc_label = "Intermediate (Arc_1)" if arc_name == "Arc_1" else arc_name
            ax.text(x, y - 0.06, arc_label, fontsize=8.5, ha="center", va="top", fontweight="bold", color=color)

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
        axins.set_title("Zoomed inset -- same data, magnified", fontsize=10)

        fig.suptitle(
            f"Hamming-distance RadViz projection onto 6 archetype anchors (softmax T={T:g})\n"
            f"{pert_display} vs. unperturbed walk trajectories ({label} start)",
            fontsize=12, y=0.99,
        )

        pert_cmap_name = "Greens" if pert_color == "tab:green" else "Purples"
        legend_elements = [
            Line2D([0], [0], color=pert_color, lw=2.5, label=f"{pert_display}: mean path (light->dark = early->late)"),
            Line2D([0], [0], color="0.5", lw=2.5, label="Unperturbed: mean path (light->dark = early->late)"),
            Line2D([0], [0], color=pert_color, lw=1.0, alpha=0.4, label=f"{pert_display}: {len(pert_paths)} individual walks"),
            Line2D([0], [0], color="0.5", lw=1.0, alpha=0.4, label=f"Unperturbed: {len(unpert_paths)} individual walks"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path start"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", markeredgecolor="black", markersize=8, label="path end (step 4000)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markeredgecolor="black", markersize=9, label="Generalist_NE (archetype average state)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="black", markersize=9, label="Intermediate/Arc_1 (archetype average state)"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.215), fontsize=8, ncol=2, frameon=True)

        note = (
            f"Note: start is a single real GEMM/organoid cell (Hamming distance 0 from its archetype's average state, confirmed directly)\n"
            f"-- not a pooled/averaged population. Hexagon vertices + the 2 circles are GEMM's own fitted archetype average states, projected\n"
            f"by Hamming distance to the 6 vertex archetypes (softmax, T={T:g}); colors matched to the user's reference figure (ref_hex)."
        )
        fig.text(0.5, 0.065, note, ha="center", va="center", fontsize=7.5,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

        for ext in ["png", "pdf"]:
            fig.savefig(f"{out_dir}/{out_name}.{ext}", dpi=150)
        plt.close(fig)
        print(f"  Wrote {out_dir}/{out_name}.{{png,pdf}}")


if __name__ == "__main__":
    main()
