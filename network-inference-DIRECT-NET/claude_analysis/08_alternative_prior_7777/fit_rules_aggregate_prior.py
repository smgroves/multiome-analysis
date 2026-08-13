"""Refit BoBa-T rules on GEMM's own training data (identical split, network, and
node_normalization=0.3 as barcode 6667) but with a DIFFERENT pseudo-observation prior --
the "aggregate" alternative stubbed but never executed in
claude_analysis/BoBa-T_hyperparameters.md sec 7.

Today's default (bobaT/tl.py get_rules, unchanged, not edited here per this project's
"only edit in multiome-analysis, not BoBa-T" convention): every leaf gets a pseudo-
observation of weight `1 - max(heat[:, leaf])` at probability 0.5 -- i.e. a leaf only
escapes the pull-to-0.5 if AT LEAST ONE cell is a confident (near-1 heat) match. This
ignores aggregate evidence: a leaf with 500 cells each at heat=0.4 (no single confident
match, but a lot of collective weight) still gets pulled hard toward 0.5 under the default
scheme, exactly as hard as a leaf with 0 cells at all.

This script's alternative: pseudo-observation weight = C / (C + sum(heat[:, leaf])) -- a
Beta-style pseudocount tied to the leaf's TOTAL weighted evidence rather than its single
best cell. C=1.0 (default here) means a leaf needs roughly one fully-confident-cell's worth
of aggregate evidence before the prior's pull meaningfully fades; well-populated leaves
(tens of confident-equivalent cells) get a near-negligible pull, while genuinely
under-observed leaves still get pulled hard toward 0.5, same as before.

Reuses bobaT's own `detect_irrelevant_regulator`/`reorder_binary_decision_tree` (imported,
unmodified -- only the pseudocount computation itself changes) for regulator selection, so
the two rule sets differ in exactly one respect. `threshold=0` (no regulator pruning),
matching every other rule-fitting run in this repo.

Run in bobaT_env_py3.13 (needs bobaT for helper functions + utils):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/fit_rules_aggregate_prior.py
"""

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import bobaT.tl as tl
import bobaT.utils as ut
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
TRAIN_PATH = f"{DIR_PREFIX}/6667/data_split/train_t0combined.csv"
OUT_RULES_PATH = f"{DIR_PREFIX}/7777/rules/rules_7777.txt"
THRESHOLD = 0
PSEUDOCOUNT_C = 1.0


def get_rules_aggregate_prior(data, vertex_dict, threshold=0, pseudocount_c=1.0):
    """Reimplementation of bobaT.tl.get_rules with the aggregate-evidence pseudocount
    prior in place of the default max-heat one. Identical control flow (iterative
    irrelevant-regulator pruning, importance reordering) -- see module docstring for what's
    actually different (just the two `prob_01[...] += ...` lines)."""
    v_names = {v: k for k, v in vertex_dict.items()}
    nodes = list(vertex_dict)
    rules = {}
    regulators_dict = {}
    total_nodes = len(nodes)

    for xx, gene in enumerate(nodes):
        print(f"Fitting {xx}/{total_nodes}: {gene}")
        irrelevant = []
        n_irrelevant_new = 0
        regulators = [v_names[v] for v in vertex_dict[gene].in_neighbors() if v_names[v] not in irrelevant]

        while True:
            n_irrelevant_old = n_irrelevant_new
            regulators_dict[gene] = regulators
            n = len(regulators)

            if n > 0:
                prob_01 = np.zeros((2, 2**n))
                heat = np.ones((data.shape[0], 2**n))

                for leaf in range(2**n):
                    binary = ut.idx2binary(leaf, len(regulators))
                    binary = [{"0": False, "1": True}[i] for i in binary]
                    for i, idx in enumerate(data.index):
                        df = data.loc[idx]
                        val = float(data.loc[idx, gene])
                        for col, on in enumerate(binary):
                            regulator = regulators[col]
                            if on:
                                heat[i, leaf] *= float(df[regulator])
                            else:
                                heat[i, leaf] *= 1 - float(df[regulator])
                        prob_01[0, leaf] += val * heat[i, leaf]
                        prob_01[1, leaf] += (1 - val) * heat[i, leaf]

                # --- The one substantive change vs. bobaT.tl.get_rules ---
                total_heat = np.sum(heat, axis=0)
                pseudo_weight = pseudocount_c / (pseudocount_c + total_heat)
                prob_01[0, :] += pseudo_weight * 0.5
                prob_01[1, :] += pseudo_weight * 0.5
                # --- end change ---

                rules[gene] = prob_01[0, :] / np.sum(prob_01, axis=0)
                max_regulator_relevance, tot_regulator_relevance, signed_tot_regulator_relevance = (
                    tl.detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold, heat=heat)
                )

                old_regulator_order = list(regulators)
                regulators = sorted(regulators, key=lambda x: max_regulator_relevance[x], reverse=True)
                if max_regulator_relevance[regulators[-1]] < threshold:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(regulators, key=lambda x: tot_regulator_relevance[x], reverse=True)
                regulators_dict[gene] = regulators
                n_irrelevant_new = len(irrelevant)

            if len(regulators) == 0 and gene not in irrelevant:
                regulators = [gene]
                regulators_dict[gene] = [gene]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0:
                break

        if len(regulators) > 0:
            importance_order = tl.reorder_binary_decision_tree(old_regulator_order, regulators)
            rules[gene] = rules[gene][importance_order]

    return rules, regulators_dict


def main():
    # remove_selfloops=True matches 6667's original fitting script
    # (main_all_data_remove_selfloops_6667.py) exactly -- an earlier version of this script
    # used remove_selfloops=False by mistake, which let 13 genes' self-edges into the
    # candidate-regulator set here when 6667 never had them as candidates at all (not
    # something 6667's fit considered and pruned). That confounded the pseudocount
    # comparison with an unrelated network-loading difference; fixed here so the only
    # difference between 6667 and 7777 is the pseudocount.
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    print("Loading GEMM training data (identical split to 6667)...")
    train_data = bb.load.load_data(
        TRAIN_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    print(f"Training on {train_data.shape[0]} cells, {train_data.shape[1]} genes.")

    rules, regulators_dict = get_rules_aggregate_prior(train_data, vertex_dict, threshold=THRESHOLD, pseudocount_c=PSEUDOCOUNT_C)

    bb.tl.save_rules(rules, regulators_dict, fname=OUT_RULES_PATH)
    print(f"\nWrote {OUT_RULES_PATH}")

    # Quick comparison against 6667's rules -- how much did the fitted values actually move?
    rules_6667, regulators_dict_6667 = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    diffs = []
    for gene in nodes:
        if regulators_dict[gene] == regulators_dict_6667.get(gene):
            diffs.append({"gene": gene, "mean_abs_diff": np.mean(np.abs(rules[gene] - rules_6667[gene])),
                          "n_regulators": len(regulators_dict[gene])})
        else:
            diffs.append({"gene": gene, "mean_abs_diff": np.nan, "n_regulators": len(regulators_dict[gene]),
                          "note": "regulator set changed vs 6667"})
    diff_df = pd.DataFrame(diffs).sort_values("mean_abs_diff", ascending=False)
    diff_df.to_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/rule_value_shift_vs_6667.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== Genes whose fitted rule values shifted the most vs. 6667 ===")
    print(diff_df.head(15).to_string(index=False))
    print(f"\nMean |diff| across all genes: {diff_df['mean_abs_diff'].mean():.4f}")


if __name__ == "__main__":
    main()
