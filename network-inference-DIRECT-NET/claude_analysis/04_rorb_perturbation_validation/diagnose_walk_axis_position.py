"""Collapses the 3-target distance view (diagnose_walk_temporal_ordering.py) into a
single number per step: position along the NE<->nonNE combinatorial axis itself (the 46
genes where Generalist_NE and Generalist_nonNE actually differ), expressed as %NE-like
(100 = exactly matches Generalist_NE on all 46 axis genes, 0 = exactly matches
Generalist_nonNE). Arc_1 ("Intermediate") sits at 65.2% along this axis -- marked
explicitly on the y-axis. Plots this position over walk steps for all 100 individual
walks (thin) + mean (bold), knockdown (color) vs. unperturbed negative control (grey),
for each of the three NE-starting organoid populations.

Default (no CLI args) reproduces the original RORA_RORB-knockdown-vs-unperturbed-only
plot; pass "ascl1" to additionally overlay the ASCL1-knockout positive control (a
canonical NE master regulator, so its knockout should drive an unambiguous, large
NE -> nonNE shift) -- written to separate, differently-named outputs so neither version
overwrites the other.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/diagnose_walk_axis_position.py [ascl1]
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
from scipy.stats import mannwhitneyu

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks/axis_position_plots/raw_hamming_axis"
WALK_PATH = f"{DIR_PREFIX}/6667/organoid_seeded/walks/long_walks/4000_step_walks"
ORGANOID_STARTS = {
    "Neuroendocrine1": 5500048749758430,
    "Generalist NE": 7861313821741716,
    "Neuroendocrine2": 17609338899731,
}
STEP_STRIDE = 10
WITH_ASCL1 = "ascl1" in [a.lower() for a in sys.argv[1:]]
OUT_SUFFIX = "_with_ASCL1" if WITH_ASCL1 else ""
# Other archetypes' known axis positions (from FINDINGS.md sec 11), for reference lines.
# Arc_N -> biological name mapping per user: Arc_1=Intermediate, Arc_2=nonNE2,
# Arc_3=Secretory, Arc_4=nonNE1, Arc_5=NE1, Arc_6=NE2.
ARCHETYPE_POSITIONS = {
    "Generalist_NE": 100.0, "NE2 (Arc_6)": 95.7, "NE1 (Arc_5)": 91.3, "Intermediate (Arc_1)": 65.2,
    "Secretory (Arc_3)": 58.7, "nonNE1 (Arc_4)": 6.5, "nonNE2 (Arc_2)": 4.3, "Generalist_nonNE": 0.0,
}


def load_axis(nodes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    ne_row = avg_states.loc["Generalist_NE"]
    nonne_row = avg_states.loc["Generalist_nonNE"]
    ne_idx = int("".join(str(int(v)) for v in ne_row), 2)

    axis_genes = [g for g in nodes if int(ne_row[g]) != int(nonne_row[g])]
    n = len(nodes)
    # bitmask with 1s at the bit positions (MSB-first, matching idx2binary/state2idx
    # convention) corresponding to axis_genes
    mask = 0
    for i, g in enumerate(nodes):
        if g in axis_genes:
            mask |= 1 << (n - 1 - i)
    return ne_idx, mask, len(axis_genes)


def parse_walk_file(path):
    with open(path) as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]


def position_series(walk, ne_idx, mask, n_axis_genes):
    steps = walk[::STEP_STRIDE]
    mismatches = np.array([((s ^ ne_idx) & mask).bit_count() for s in steps])
    return 100.0 * (n_axis_genes - mismatches) / n_axis_genes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    ne_idx, mask, n_axis_genes = load_axis(nodes)
    print(f"NE<->nonNE axis: {n_axis_genes} genes")

    for label, start_idx in ORGANOID_STARTS.items():
        walks_kd = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_RORA_RORB_kd.csv")
        walks_unpert = parse_walk_file(f"{WALK_PATH}/{start_idx}/results.csv")

        kd_pos = [position_series(w, ne_idx, mask, n_axis_genes) for w in walks_kd]
        unpert_pos = [position_series(w, ne_idx, mask, n_axis_genes) for w in walks_unpert]
        step_axis = np.arange(len(kd_pos[0])) * STEP_STRIDE

        kd_mean = np.mean(kd_pos, axis=0)
        unpert_mean = np.mean(unpert_pos, axis=0)

        # Test statistics: knockdown vs. unperturbed, both on final-step position (where
        # each walk ends up after 4000 steps) and on each walk's own whole-trajectory mean
        # position (Mann-Whitney U, unpaired -- these are two independent sets of 100 walks).
        kd_final = np.array([c[-1] for c in kd_pos])
        unpert_final = np.array([c[-1] for c in unpert_pos])
        kd_walkmean = np.array([c.mean() for c in kd_pos])
        unpert_walkmean = np.array([c.mean() for c in unpert_pos])
        _, p_final = mannwhitneyu(kd_final, unpert_final, alternative="two-sided")
        _, p_walkmean = mannwhitneyu(kd_walkmean, unpert_walkmean, alternative="two-sided")

        if WITH_ASCL1:
            walks_ascl1 = parse_walk_file(f"{WALK_PATH}/{start_idx}/results_ASCL1_kd.csv")
            ascl1_pos = [position_series(w, ne_idx, mask, n_axis_genes) for w in walks_ascl1]
            ascl1_mean = np.mean(ascl1_pos, axis=0)
            ascl1_final = np.array([c[-1] for c in ascl1_pos])
            ascl1_walkmean = np.array([c.mean() for c in ascl1_pos])
            _, p_final_ascl1 = mannwhitneyu(ascl1_final, unpert_final, alternative="two-sided")
            _, p_walkmean_ascl1 = mannwhitneyu(ascl1_walkmean, unpert_walkmean, alternative="two-sided")

        plt.figure(figsize=(9.5, 5.5))
        if WITH_ASCL1:
            for curve in ascl1_pos:
                plt.plot(step_axis, curve, color="tab:red", alpha=0.08, linewidth=0.8, zorder=1)
            plt.plot(step_axis, ascl1_mean, color="tab:red", linewidth=2.5,
                      label=f"organoid '{label}' start, ASCL1 knockout (positive control, mean)", zorder=3)
        for curve in kd_pos:
            plt.plot(step_axis, curve, color="tab:purple", alpha=0.08, linewidth=0.8, zorder=1)
        plt.plot(step_axis, kd_mean, color="tab:purple", linewidth=2.5,
                  label=f"organoid '{label}' start, RORA_RORB knockdown (mean)", zorder=3)
        for curve in unpert_pos:
            plt.plot(step_axis, curve, color="0.6", alpha=0.06, linewidth=0.8, zorder=1)
        plt.plot(step_axis, unpert_mean, color="0.5", linewidth=2.5, linestyle="--",
                  label=f"organoid '{label}' start, unperturbed (negative control, mean)", zorder=2)

        # Right-side reference archetypes are GEMM's own (distinct from the organoid-derived
        # trace above) -- headed and prefixed explicitly so the two can't be conflated.
        plt.text(step_axis[-1] * 1.03, 108, "GEMM archetypes:", fontsize=8, va="center", ha="left", fontweight="bold")
        for arc_label, pos in ARCHETYPE_POSITIONS.items():
            plt.axhline(pos, color="black", linewidth=0.6, linestyle=":", alpha=0.5, zorder=0)
            plt.text(step_axis[-1] * 1.03, pos, f"GEMM {arc_label}", fontsize=7.5, va="center", ha="left")

        plt.xlabel("Walk step")
        plt.ylabel("Position on NE <-> nonNE axis\n(%NE-like; 100=Generalist_NE, 0=Generalist_nonNE)")
        plt.title(f"organoid '{label}' start: position along NE<->nonNE axis over time")
        plt.xlim(0, step_axis[-1] * 1.42)
        plt.ylim(-5, 112)
        plt.legend(loc="lower left", fontsize=8.5)

        if WITH_ASCL1:
            stats_text = (
                f"Mann-Whitney U vs. unperturbed (n=100 walks each) -- RORA_RORB kd: final-step "
                f"{kd_final.mean():.1f} vs. {unpert_final.mean():.1f}, p={p_final:.1e}; whole-walk mean "
                f"{kd_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, p={p_walkmean:.1e}\n"
                f"ASCL1 ko (positive control): final-step {ascl1_final.mean():.1f} vs. {unpert_final.mean():.1f}, "
                f"p={p_final_ascl1:.1e}; whole-walk mean {ascl1_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, "
                f"p={p_walkmean_ascl1:.1e}"
            )
            rect = [0, 0.10, 1, 1]
            text_y = -0.04
        else:
            stats_text = (
                f"Mann-Whitney U, knockdown vs. unperturbed (n=100 walks each): "
                f"final-step position {kd_final.mean():.1f} vs. {unpert_final.mean():.1f}, p={p_final:.1e}; "
                f"whole-walk mean position {kd_walkmean.mean():.1f} vs. {unpert_walkmean.mean():.1f}, p={p_walkmean:.1e}"
            )
            rect = [0, 0.06, 1, 1]
            text_y = -0.02
        plt.gcf().text(0.5, text_y, stats_text, ha="center", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
        plt.tight_layout(rect=rect)
        for ext in ["png", "pdf"]:
            plt.savefig(f"{OUT_DIR}/walk_axis_position_{label.replace(' ', '_')}{OUT_SUFFIX}.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        extra = f", ascl1 mean start={ascl1_mean[0]:.1f}, end={ascl1_mean[-1]:.1f}, p_final_ascl1={p_final_ascl1:.2e}, p_walkmean_ascl1={p_walkmean_ascl1:.2e}" if WITH_ASCL1 else ""
        print(f"Wrote walk_axis_position_{label.replace(' ', '_')}{OUT_SUFFIX}.{{png,pdf}} "
              f"(kd mean start={kd_mean[0]:.1f}, end={kd_mean[-1]:.1f}; "
              f"unpert mean start={unpert_mean[0]:.1f}, end={unpert_mean[-1]:.1f}; "
              f"p_final={p_final:.2e}, p_walkmean={p_walkmean:.2e}{extra})")


if __name__ == "__main__":
    main()
