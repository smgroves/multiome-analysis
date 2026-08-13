"""Refit BoBa-T rules on GEMM's own training data (identical split, network, and
node_normalization=0.3 as barcode 6667), changing exactly one thing: the pseudo-observation
prior is anchored to each gene's own marginal rate (empirical on/off frequency across all
training cells) instead of a flat 0.5.

Motivation: single-cell data is rarely balanced -- most genes are not naturally on/off in
50% of cells. Pulling an under-evidenced leaf toward 0.5 is itself a systematic bias for a
gene that's globally rare (or globally common): the "safe" fallback when we don't have
enough evidence for a specific regulator combination shouldn't be "no idea," it should be
"probably behaves like this gene usually does." This is an empirical-Bayes shrinkage target
change, not a change to how much weight the pseudo-observation gets (that's still the
original max_heat formula, unchanged from 6667) -- isolating this one variable from the
pseudocount_mode="aggregate" idea already tried (barcode 7777), which changed the weight
formula instead of the anchor and showed no real benefit once a network-loading confound
was fixed.

Uses bb.tl.get_rules directly (the real package function, now with a pseudocount_target
option added -- see bobaT/tl.py) rather than a standalone reimplementation, specifically to
avoid the class of setup mistake that confounded barcode 7777 (an accidental
remove_selfloops mismatch vs 6667's original fitting script). remove_selfloops=True here,
matching main_all_data_remove_selfloops_6667.py exactly.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/09_marginal_rate_prior_8888/fit_rules_marginal_rate_8888.py
"""

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
TRAIN_PATH = f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv"
OUT_RULES_PATH = f"{DIR_PREFIX}/8888/rules/rules_8888.txt"
THRESHOLD = 0


def main():
    # remove_selfloops=True matches 6667's original fitting script exactly (see the 7777
    # postmortem in 7777/README.md for why this specific flag matters).
    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    print("Loading GEMM training data (identical split to 6667)...")
    train_data = bb.load.load_data(
        TRAIN_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    print(f"Training on {train_data.shape[0]} cells, {train_data.shape[1]} genes.")

    rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
        data=train_data,
        vertex_dict=vertex_dict,
        plot=False,
        threshold=THRESHOLD,
        pseudocount_mode="max_heat",
        pseudocount_target="marginal_rate",
    )

    bb.tl.save_rules(rules, regulators_dict, fname=OUT_RULES_PATH)
    print(f"\nWrote {OUT_RULES_PATH}")

    # Same audit as 7777: confirm regulator SETS didn't change vs 6667 (order-insensitive),
    # and quantify the rule-value shift for genes where they didn't.
    rules_6667, regulators_dict_6667 = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    diffs = []
    n_same_set = 0
    n_diff_set = 0
    for gene in nodes:
        r6 = regulators_dict_6667.get(gene)
        r8 = regulators_dict[gene]
        if set(r6) == set(r8):
            n_same_set += 1
            if r6 == r8:
                diffs.append({"gene": gene, "mean_abs_diff": np.mean(np.abs(rules[gene] - rules_6667[gene])),
                              "n_regulators": len(r8), "note": "same set, same order"})
            else:
                diffs.append({"gene": gene, "mean_abs_diff": np.nan, "n_regulators": len(r8),
                              "note": "same set, reordered (behaviorally identical)"})
        else:
            n_diff_set += 1
            diffs.append({"gene": gene, "mean_abs_diff": np.nan, "n_regulators": len(r8),
                          "note": f"DIFFERENT SET: 6667={r6} 8888={r8}"})
    diff_df = pd.DataFrame(diffs).sort_values("mean_abs_diff", ascending=False)
    diff_df.to_csv(f"{DIR_PREFIX}/claude_analysis/09_marginal_rate_prior_8888/rule_value_shift_vs_6667.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"\nRegulator sets identical to 6667: {n_same_set}/{len(nodes)}  (genuinely different: {n_diff_set}/{len(nodes)})")
    print("\n=== Genes whose fitted rule values shifted the most vs. 6667 (same set & order only) ===")
    print(diff_df.head(15).to_string(index=False))
    print(f"\nMean |diff| across directly-comparable genes: {diff_df['mean_abs_diff'].mean():.4f}")


if __name__ == "__main__":
    main()
