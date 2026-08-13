"""Secondary diagnostic: is a state's "flip probability" (per bb.utils.get_flip_probs --
for each of the 53 genes, the probability the fitted rule favors flipping it given the
CURRENT state) elevated for real cells labeled Intermediate/Arc_1 as a whole population,
not just for Arc_1's own archetype average state (already checked directly and found to
be the least stable of all 8 archetypes -- see FINDINGS.md)?

For every real GEMM cell (binarized, same convention as average_states.txt/attractors):
compute its own mean flip probability across all 53 genes (0 = every gene agrees with the
rule's prediction given current context, i.e. maximally stable; 1 = every gene disagrees).
Boxplot this per-cell distribution, grouped by the cell's own phenotype label
(`S_0.5threshold_splitgen`).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/plot_flip_probability_by_archetype.py
"""

import os

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
OUT_DIR = f"{DIR_PREFIX}/comparisons/organoid_walks/flip_probability_diagnostic"
GEMM_DATA_PATH = f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
GEMM_CLUSTERS_PATH = f"{DIR_PREFIX}/data/AA_clusters_splitgen.csv"
GEMM_NORM = 0.3

# Biological names for axis labels; plotted in this NE -> nonNE order for readability.
ARCHETYPE_ORDER = ["Arc_5", "Arc_6", "Generalist_NE", "Arc_3", "Arc_1", "Arc_2", "Arc_4", "Generalist_nonNE"]
ARCHETYPE_LABELS = {
    "Arc_5": "NE1\n(Arc_5)", "Arc_6": "NE2\n(Arc_6)", "Generalist_NE": "Generalist_NE",
    "Arc_3": "Secretory\n(Arc_3)", "Arc_1": "Intermediate\n(Arc_1)", "Arc_2": "nonNE2\n(Arc_2)",
    "Arc_4": "nonNE1\n(Arc_4)", "Generalist_nonNE": "Generalist_nonNE",
}
ARCHETYPE_COLORS = {
    "Arc_5": "tab:purple", "Arc_6": "darkred", "Generalist_NE": "0.4", "Arc_3": "tab:green",
    "Arc_1": "tab:blue", "Arc_2": "orange", "Arc_4": "tab:red", "Generalist_nonNE": "lightcoral",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    node_indices = dict(zip(nodes, range(len(nodes))))

    data = bb.load.load_data(GEMM_DATA_PATH, nodes, norm=GEMM_NORM, delimiter=",", log1p=False,
                              transpose=True, sample_order=False, fillna=0)
    binaries = bb.utils.binarize_data_df(data, nodes, threshold=0.5)
    cell_idx = binaries.apply(lambda row: int("".join(str(int(v)) for v in row), 2), axis=1)

    clusters = pd.read_csv(GEMM_CLUSTERS_PATH, index_col=0).reindex(data.index)
    df = pd.DataFrame({"cell_idx": cell_idx, "phenotype": clusters["S_0.5threshold_splitgen"].values}, index=data.index)
    df = df.dropna(subset=["phenotype"])

    print(f"Computing flip probability for {len(df)} real GEMM cells...")
    unique_idx = df["cell_idx"].unique()
    print(f"  ({len(unique_idx)} unique discrete states among them)")
    mean_cache, strict_cache = {}, {}
    for i, idx in enumerate(unique_idx):
        flips = np.array(bb.utils.get_flip_probs(int(idx), rules, regulators_dict, nodes, node_indices))
        mean_cache[idx] = flips.mean()
        strict_cache[idx] = int((flips > 0.5).sum())
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(unique_idx)} unique states done")
    df["mean_flip_prob"] = df["cell_idx"].map(mean_cache)
    df["n_genes_flip_gt_0.5"] = df["cell_idx"].map(strict_cache)

    n_true_attractors = (df["n_genes_flip_gt_0.5"] == 0).sum()
    print(f"\n{n_true_attractors}/{len(df)} real cells ({100*n_true_attractors/len(df):.1f}%) are genuine "
          f"fixed points (0/53 genes with flip probability >0.5) under the fitted rules.")

    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    avg_flip_vals = {}
    for a in ARCHETYPE_ORDER:
        avg_idx = int("".join(str(int(v)) for v in avg_states.loc[a]), 2)
        flips = np.array(bb.utils.get_flip_probs(avg_idx, rules, regulators_dict, nodes, node_indices))
        avg_flip_vals[a] = {"mean_flip_prob": flips.mean(), "n_genes_flip_gt_0.5": int((flips > 0.5).sum())}

    def make_boxplot(column, ylabel, title, out_name, avg_marker_label, ylim, is_int=False):
        groups = [df.loc[df["phenotype"] == a, column].values for a in ARCHETYPE_ORDER]
        ns = [len(g) for g in groups]
        print(f"\nPer-archetype {column} (across all real cells, not just the average state):")
        summary = df.groupby("phenotype")[column].agg(["mean", "median", "std", "count"]).reindex(ARCHETYPE_ORDER)
        print(summary)

        arc1_vals = df.loc[df["phenotype"] == "Arc_1", column].values
        print(f"\nMann-Whitney U, Arc_1 (Intermediate) vs. each other archetype ({column}):")
        for a in ARCHETYPE_ORDER:
            if a == "Arc_1":
                continue
            other = df.loc[df["phenotype"] == a, column].values
            _, p = mannwhitneyu(arc1_vals, other, alternative="greater")
            print(f"  Arc_1 > {a}: p={p:.2e} (Arc_1 n={len(arc1_vals)}, {a} n={len(other)})")

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(groups, positions=range(len(ARCHETYPE_ORDER)), widths=0.6, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black"))
        for patch, a in zip(bp["boxes"], ARCHETYPE_ORDER):
            patch.set_facecolor(ARCHETYPE_COLORS[a])
            patch.set_alpha(0.6)
        rng = np.random.default_rng(0)
        for i, (a, g) in enumerate(zip(ARCHETYPE_ORDER, groups)):
            if len(g) == 0:
                continue
            jitter = rng.uniform(-0.3, 0.3, size=len(g)) if is_int else rng.uniform(-0.15, 0.15, size=len(g))
            ax.scatter(np.full(len(g), i) + jitter, g, s=4, color=ARCHETYPE_COLORS[a], alpha=0.15, zorder=1)

        ax.set_xticks(range(len(ARCHETYPE_ORDER)))
        ax.set_xticklabels([f"{ARCHETYPE_LABELS[a]}\n(n={n})" for a, n in zip(ARCHETYPE_ORDER, ns)], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        for i, a in enumerate(ARCHETYPE_ORDER):
            ax.scatter([i], [avg_flip_vals[a][column]], marker="X", s=120, color="black", zorder=5,
                       label=avg_marker_label if i == 0 else None)

        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(*ylim)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(f"{OUT_DIR}/{out_name}.{ext}", dpi=150)
        plt.close(fig)
        summary.to_csv(f"{OUT_DIR}/{out_name}_summary.csv")
        print(f"Wrote {OUT_DIR}/{out_name}.{{png,pdf}}, {out_name}_summary.csv")

    make_boxplot(
        "mean_flip_prob",
        "Mean flip probability across all 53 genes\n(per real cell's own binarized state; 0=fully stable, 1=every gene disagrees with the rule)",
        "Per-cell flip probability by GEMM phenotype label\n(archetype AVERAGE states' own values shown as X markers, for reference)",
        "flip_probability_by_archetype", "archetype average state's own flip probability", (-0.02, 1.02),
    )
    make_boxplot(
        "n_genes_flip_gt_0.5",
        "# of 53 genes with flip probability >0.5\n(per real cell's own binarized state; 0 = genuine fixed point / true attractor, by the strict criterion)",
        "Per-cell count of unstable genes (flip prob. >0.5) by GEMM phenotype label\n(0 = a true fixed point; archetype AVERAGE states' own values shown as X markers)",
        "flip_probability_strict_by_archetype", "archetype average state's own count", (-0.5, max(df["n_genes_flip_gt_0.5"].max(), 1) + 0.5),
        is_int=True,
    )

    df[["phenotype", "mean_flip_prob", "n_genes_flip_gt_0.5"]].to_csv(f"{OUT_DIR}/flip_probability_per_cell.csv")
    print(f"\nWrote {OUT_DIR}/flip_probability_per_cell.csv")


if __name__ == "__main__":
    main()
