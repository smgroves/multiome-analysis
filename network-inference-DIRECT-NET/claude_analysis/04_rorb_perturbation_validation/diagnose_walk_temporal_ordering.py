"""Does a random walk starting from an NE-like organoid state, under RORA_RORB knockdown,
pass THROUGH the Intermediate archetype (Arc_1) on its way toward the nonNE cluster
(Generalist_nonNE/Arc_2/Arc_4), or does it just end up closer to nonNE without any
temporal ordering? diagnose_walk_archetype_distance.py only looked at whole-trajectory
mean/min distance, which can't distinguish "passes through Intermediate en route" from
"drifts straight to nonNE, coincidentally passing near Intermediate's position along the
NE<->nonNE axis without visiting it in a temporally ordered way."

For each walk (all 4000 states, fixed length -- confirmed no early termination), computes
a per-step distance time series to three targets: the NE cluster (min distance to
Generalist_NE/Arc_5/Arc_6), Arc_1 ("Intermediate"), and the nonNE cluster (min distance to
Generalist_nonNE/Arc_2/Arc_4). Averages across the 100 walks per condition to get
canonical distance-vs-step curves, and directly tests temporal ordering: per walk, is the
step of closest approach to Arc_1 earlier than the step of closest approach to the nonNE
cluster (Wilcoxon signed-rank, paired within each walk)?

Default (no CLI args) reproduces the original RORA_RORB-knockdown-vs-unperturbed-only
plot/CSV; pass "ascl1" to additionally include the ASCL1-knockout positive control (a
canonical NE master regulator, so its knockout should drive an unambiguous, large
NE -> nonNE shift) -- written to separate, differently-named outputs so neither version
overwrites the other.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_walk_temporal_ordering.py [ascl1]
"""

import ast
import os
import sys

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
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks"
PLOT_DIR = f"{OUT_DIR}/temporal_distance_plots"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
ORGANOID_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Intermediate": 2308064086923296,
    "Neuroendocrine2": 17609338899731,
}
NE_GROUP = ["Generalist_NE", "Arc_5", "Arc_6"]
NONNE_GROUP = ["Generalist_nonNE", "Arc_2", "Arc_4"]
INTERMEDIATE = "Arc_1"
STEP_STRIDE = 10  # subsample every 10th step for compute/plot tractability (4000 -> 400 points)
WITH_ASCL1 = "ascl1" in [a.lower() for a in sys.argv[1:]]
OUT_SUFFIX = "_with_ASCL1" if WITH_ASCL1 else ""


def load_archetype_indices(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    return {archetype: int("".join(str(int(v)) for v in row), 2) for archetype, row in avg_states.iterrows()}


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def distance_series(walk, targets_idx):
    """Per-step distance to each named archetype index, subsampled every STEP_STRIDE steps."""
    steps = walk[::STEP_STRIDE]
    return {name: np.array([(s ^ idx).bit_count() for s in steps]) for name, idx in targets_idx.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    archetype_idx = load_archetype_indices(nodes)

    ne_group_idx = {k: archetype_idx[k] for k in NE_GROUP}
    nonne_group_idx = {k: archetype_idx[k] for k in NONNE_GROUP}
    intermediate_idx = archetype_idx[INTERMEDIATE]

    NE_STARTING_LABELS = ["Neuroendocrine1", "Generalist NE", "Neuroendocrine2"]

    conditions = [("unperturbed", "results.csv"), ("RORA_RORB_kd", "results_RORA_RORB_kd.csv")]
    if WITH_ASCL1:
        conditions.append(("ASCL1_kd", "results_ASCL1_kd.csv"))

    ordering_rows = []
    for label, start_idx in ORGANOID_STARTS.items():
        curves_by_condition = {}
        for condition, fname in conditions:
            walks = parse_walk_file(f"{WALK_PATH}/{start_idx}/{fname}")
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

            ne_mean_curve = np.mean(ne_curves, axis=0)
            int_mean_curve = np.mean(int_curves, axis=0)
            nonne_mean_curve = np.mean(nonne_curves, axis=0)
            step_axis = np.arange(len(ne_mean_curve)) * STEP_STRIDE
            curves_by_condition[condition] = {
                "ne": ne_curves, "int": int_curves, "nonne": nonne_curves,
                "ne_mean": ne_mean_curve, "int_mean": int_mean_curve, "nonne_mean": nonne_mean_curve,
                "step_axis": step_axis,
            }

            arc1_steps = np.array(arc1_argmin_steps)
            nonne_steps = np.array(nonne_argmin_steps)
            diff = nonne_steps - arc1_steps  # positive => Arc_1 reached (closest) before nonNE
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                stat, p = np.nan, np.nan  # all-zero differences

            ordering_rows.append({
                "organoid_start": label, "condition": condition,
                "mean_step_closest_to_Arc1": arc1_steps.mean(), "mean_step_closest_to_nonNE": nonne_steps.mean(),
                "frac_walks_Arc1_before_nonNE": (diff > 0).mean(),
                "frac_walks_simultaneous": (diff == 0).mean(),
                "wilcoxon_p": p,
            })

        if label in NE_STARTING_LABELS:
            # Combined figure: RORA_RORB knockdown (colored: all 100 individual walks thin
            # + bold mean) overlaid with the unperturbed negative control (greyed out, same
            # spaghetti+mean treatment, distinguished by linestyle since color is reused
            # for "condition" here instead of "target").
            kd = curves_by_condition["RORA_RORB_kd"]
            unpert = curves_by_condition["unperturbed"]
            step_axis = kd["step_axis"]

            plt.figure(figsize=(9, 6.1) if WITH_ASCL1 else (9, 5.5))
            # Color = target identity, same hue family across conditions but a visibly
            # lighter tint for ASCL1 ko (rather than relying on the linestyle alone, which
            # was hard to distinguish at legend-swatch size) so RORB kd (saturated, solid)
            # and ASCL1 ko (light, dash-dot) read apart at a glance; unperturbed stays grey.
            # ASCL1 ko: mean only (no per-walk spaghetti) -- it's a reference positive
            # control here, not the primary comparison, so its 100 individual walks would
            # just add clutter without adding information beyond the mean.
            series = []
            if WITH_ASCL1:
                ascl1 = curves_by_condition["ASCL1_kd"]
                series += [
                    ("GEMM NE cluster, organoid start, ASCL1 ko (positive control, mean only)", None, ascl1["ne_mean"], "lightcoral", "-."),
                    ("GEMM Intermediate, organoid start, ASCL1 ko (positive control, mean only)", None, ascl1["int_mean"], "navajowhite", "-."),
                    ("GEMM nonNE cluster, organoid start, ASCL1 ko (positive control, mean only)", None, ascl1["nonne_mean"], "lightskyblue", "-."),
                ]
            series += [
                ("GEMM NE cluster (Generalist_NE/NE1/NE2), organoid start, knockdown", kd["ne"], kd["ne_mean"], "tab:red", "-"),
                ("GEMM Intermediate, organoid start, knockdown", kd["int"], kd["int_mean"], "tab:orange", "-"),
                ("GEMM nonNE cluster (Generalist_nonNE/nonNE1/nonNE2), organoid start, knockdown", kd["nonne"], kd["nonne_mean"], "tab:blue", "-"),
                ("GEMM NE cluster, organoid start, unperturbed (negative control)", unpert["ne"], unpert["ne_mean"], "0.6", "-"),
                ("GEMM Intermediate, organoid start, unperturbed (negative control)", unpert["int"], unpert["int_mean"], "0.6", "--"),
                ("GEMM nonNE cluster, organoid start, unperturbed (negative control)", unpert["nonne"], unpert["nonne_mean"], "0.6", ":"),
            ]
            for legend_label, curves, mean_curve, color, linestyle in series:
                if curves is not None:
                    for curve in curves:
                        plt.plot(step_axis, curve, color=color, alpha=0.06, linewidth=0.8, zorder=1, linestyle=linestyle)
                plt.plot(step_axis, mean_curve, color=color, linewidth=2.5, label=legend_label, zorder=2, linestyle=linestyle)
            plt.xlabel("Walk step")
            plt.ylabel("Hamming distance (100 individual walks, thin; mean, bold)")
            title_condition = (
                "RORA_RORB kd (bold, solid) vs. ASCL1 ko (light, dash-dot)\nvs. unperturbed (grey)"
                if WITH_ASCL1 else
                "knockdown (color) vs. unperturbed negative control (grey)"
            )
            plt.title(f"Distance to archetype clusters over time\norganoid '{label}' start: {title_condition}",
                      fontsize=11 if WITH_ASCL1 else 12)
            plt.legend(loc="lower right", fontsize=7 if WITH_ASCL1 else 8)

            # Annotate the actual test statistics (Wilcoxon signed-rank on per-walk
            # closest-approach step, Arc_1 vs. nonNE cluster) directly on the figure.
            kd_row = next(r for r in ordering_rows if r["organoid_start"] == label and r["condition"] == "RORA_RORB_kd")
            unpert_row = next(r for r in ordering_rows if r["organoid_start"] == label and r["condition"] == "unperturbed")
            stats_text = (
                "Wilcoxon signed-rank, step of closest approach: Arc_1 (Intermediate) vs. nonNE cluster (n=100 walks)\n"
                f"{'RORA_RORB kd' if WITH_ASCL1 else 'Knockdown'}: mean step {kd_row['mean_step_closest_to_Arc1']:.0f} vs. {kd_row['mean_step_closest_to_nonNE']:.0f}; "
                f"Arc_1 first in {kd_row['frac_walks_Arc1_before_nonNE']*100:.0f}% of walks; p={kd_row['wilcoxon_p']:.1e}\n"
            )
            if WITH_ASCL1:
                ascl1_row = next(r for r in ordering_rows if r["organoid_start"] == label and r["condition"] == "ASCL1_kd")
                stats_text += (
                    f"ASCL1 ko: mean step {ascl1_row['mean_step_closest_to_Arc1']:.0f} vs. {ascl1_row['mean_step_closest_to_nonNE']:.0f}; "
                    f"Arc_1 first in {ascl1_row['frac_walks_Arc1_before_nonNE']*100:.0f}% of walks; p={ascl1_row['wilcoxon_p']:.1e}\n"
                )
            stats_text += (
                f"Unperturbed: mean step {unpert_row['mean_step_closest_to_Arc1']:.0f} vs. {unpert_row['mean_step_closest_to_nonNE']:.0f}; "
                f"Arc_1 first in {unpert_row['frac_walks_Arc1_before_nonNE']*100:.0f}% of walks; p={unpert_row['wilcoxon_p']:.1e}"
            )
            plt.gcf().text(0.5, -0.02, stats_text, ha="center", va="top", fontsize=7.5,
                            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
            plt.tight_layout(rect=[0, 0.13, 1, 0.92] if WITH_ASCL1 else [0, 0.08, 1, 1])
            for ext in ["png", "pdf"]:
                plt.savefig(f"{PLOT_DIR}/walk_temporal_distance_{label.replace(' ', '_')}_combined{OUT_SUFFIX}.{ext}", dpi=150, bbox_inches="tight")
            plt.close()

    ordering_df = pd.DataFrame(ordering_rows)
    ordering_df.to_csv(f"{OUT_DIR}/walk_temporal_ordering{OUT_SUFFIX}.csv", index=False)
    pd.set_option("display.width", 160)
    print(ordering_df.to_string(index=False))

    print(f"\nWrote {OUT_DIR}/walk_temporal_ordering{OUT_SUFFIX}.csv and "
          f"walk_temporal_distance_<label>_combined{OUT_SUFFIX}.{{png,pdf}} for {NE_STARTING_LABELS}")


if __name__ == "__main__":
    main()
