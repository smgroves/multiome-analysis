"""Per-target Hamming-distance-over-time plot (same method as
diagnose_walk_temporal_ordering.py's combined figure) for each of the 5 new real-cell
-seeded starting points -- see plot_new_start_hexagon.py's docstring for what each one is.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_new_start_temporal.py
"""

import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from scipy.stats import wilcoxon

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
GEMM_WALK_DIR = f"{DIR_PREFIX}/6667/walks/long_walks/4000_step_walks"
STEP_STRIDE = 10
NE_GROUP = ["Generalist_NE", "Arc_5", "Arc_6"]
NONNE_GROUP = ["Generalist_nonNE", "Arc_2", "Arc_4"]
INTERMEDIATE = "Arc_1"

SPECS = [
    ("gemm_ne1_start", "temporal_GEMM_NE1_true_attractor", "GEMM NE1 (true attractor state)", GEMM_WALK_DIR, 7788149757105118,
     "results_RORA_RORB_kd.csv", "RORA_RORB knockdown"),
    ("gemm_nonne_start_oe", "temporal_GEMM_nonNE_true_attractor", "GEMM nonNE (true attractor, shared by Generalist_nonNE/Arc_4)", GEMM_WALK_DIR, 95331434180192,
     "results_RORA_RORB_act.csv", "RORA_RORB overexpression"),
]


def load_archetype_indices(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    return {archetype: int("".join(str(int(v)) for v in row), 2) for archetype, row in avg_states.iterrows()}


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    archetype_idx = load_archetype_indices(nodes)
    ne_group_idx = {k: archetype_idx[k] for k in NE_GROUP}
    nonne_group_idx = {k: archetype_idx[k] for k in NONNE_GROUP}
    intermediate_idx = archetype_idx[INTERMEDIATE]

    for out_subdir, out_name, label, walk_dir, start_idx, pert_file, pert_display in SPECS:
        out_dir = f"{DIR_PREFIX}/comparisons/organoid_walks/{out_subdir}"
        os.makedirs(out_dir, exist_ok=True)

        curves_by_condition = {}
        ordering = {}
        for condition, fname, color, linestyle_set in [
            ("unperturbed", "results.csv", "0.6", ("-", "--", ":")),
            ("perturbed", pert_file, "tab:red", ("-", "-", "-")),
        ]:
            walks = parse_walk_file(f"{walk_dir}/{start_idx}/{fname}")
            ne_curves, int_curves, nonne_curves = [], [], []
            arc1_argmin_steps, nonne_argmin_steps = [], []
            for walk in walks:
                ne_dists = np.array([min((s ^ idx).bit_count() for idx in ne_group_idx.values()) for s in walk[::STEP_STRIDE]])
                int_dists = np.array([(s ^ intermediate_idx).bit_count() for s in walk[::STEP_STRIDE]])
                nonne_dists = np.array([min((s ^ idx).bit_count() for idx in nonne_group_idx.values()) for s in walk[::STEP_STRIDE]])
                ne_curves.append(ne_dists)
                int_curves.append(int_dists)
                nonne_curves.append(nonne_dists)
                arc1_argmin_steps.append(np.argmin(int_dists) * STEP_STRIDE)
                nonne_argmin_steps.append(np.argmin(nonne_dists) * STEP_STRIDE)

            step_axis = np.arange(len(ne_curves[0])) * STEP_STRIDE
            curves_by_condition[condition] = {
                "ne": ne_curves, "int": int_curves, "nonne": nonne_curves,
                "ne_mean": np.mean(ne_curves, axis=0), "int_mean": np.mean(int_curves, axis=0), "nonne_mean": np.mean(nonne_curves, axis=0),
                "step_axis": step_axis,
            }
            arc1_steps, nonne_steps = np.array(arc1_argmin_steps), np.array(nonne_argmin_steps)
            diff = nonne_steps - arc1_steps
            try:
                _, p = wilcoxon(diff)
            except ValueError:
                p = np.nan
            ordering[condition] = {
                "mean_step_closest_to_Arc1": arc1_steps.mean(), "mean_step_closest_to_nonNE": nonne_steps.mean(),
                "frac_walks_Arc1_before_nonNE": (diff > 0).mean(), "wilcoxon_p": p,
            }

        unpert = curves_by_condition["unperturbed"]
        pert = curves_by_condition["perturbed"]
        step_axis = unpert["step_axis"]

        plt.figure(figsize=(9, 5.5))
        series = [
            ("NE cluster, perturbed", pert["ne"], pert["ne_mean"], "tab:red", "-"),
            ("Intermediate, perturbed", pert["int"], pert["int_mean"], "tab:orange", "-"),
            ("nonNE cluster, perturbed", pert["nonne"], pert["nonne_mean"], "tab:blue", "-"),
            ("NE cluster, unperturbed", unpert["ne"], unpert["ne_mean"], "0.6", "-"),
            ("Intermediate, unperturbed", unpert["int"], unpert["int_mean"], "0.6", "--"),
            ("nonNE cluster, unperturbed", unpert["nonne"], unpert["nonne_mean"], "0.6", ":"),
        ]
        for legend_label, curves, mean_curve, color, linestyle in series:
            for curve in curves:
                plt.plot(step_axis, curve, color=color, alpha=0.06, linewidth=0.8, zorder=1, linestyle=linestyle)
            plt.plot(step_axis, mean_curve, color=color, linewidth=2.5, label=legend_label, zorder=2, linestyle=linestyle)
        plt.xlabel("Walk step")
        plt.ylabel("Hamming distance (100 individual walks, thin; mean, bold)")
        plt.title(f"Distance to archetype clusters over time\n{label} start: {pert_display} (color) vs. unperturbed (grey)")
        plt.legend(loc="lower right", fontsize=8)

        stats_text = (
            "Wilcoxon signed-rank, step of closest approach: Arc_1 (Intermediate) vs. nonNE cluster (n=100 walks)\n"
            f"{pert_display}: mean step {ordering['perturbed']['mean_step_closest_to_Arc1']:.0f} vs. {ordering['perturbed']['mean_step_closest_to_nonNE']:.0f}; "
            f"Arc_1 first in {ordering['perturbed']['frac_walks_Arc1_before_nonNE']*100:.0f}% of walks; p={ordering['perturbed']['wilcoxon_p']:.1e}\n"
            f"Unperturbed: mean step {ordering['unperturbed']['mean_step_closest_to_Arc1']:.0f} vs. {ordering['unperturbed']['mean_step_closest_to_nonNE']:.0f}; "
            f"Arc_1 first in {ordering['unperturbed']['frac_walks_Arc1_before_nonNE']*100:.0f}% of walks; p={ordering['unperturbed']['wilcoxon_p']:.1e}"
        )
        plt.gcf().text(0.5, -0.02, stats_text, ha="center", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        for ext in ["png", "pdf"]:
            plt.savefig(f"{out_dir}/{out_name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out_dir}/{out_name}.{{png,pdf}}")


if __name__ == "__main__":
    main()
