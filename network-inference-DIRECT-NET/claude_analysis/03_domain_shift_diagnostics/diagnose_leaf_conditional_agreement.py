"""Disambiguates two very different explanations for §4's "42% of edges flip sign"
finding, prompted directly by the question: if organoid_shGFP genuinely varies along the
same NE/Intermediate/ASCL1 genes as GEMM (per §8), why would the specific relationships
between those genes differ -- shouldn't the same regulatory logic apply?

Marginal pairwise correlation (what §4 measured) is a summary of the FULL population's
joint distribution -- for any gene with more than one regulator, BoBa-T's own model says
a gene's relationship to regulator A can flip sign depending on the state of regulator B
(that's what a truth table encodes: conditional, not simple pairwise, logic). If GEMM and
organoid cells have a different MIXTURE of full regulator-combination states (a
composition effect), the marginal correlation between any single regulator pair can look
completely different between datasets even if every individual fitted truth-table entry
(the actual "logic") is identical and unchanged. This is a population-composition
artifact, not evidence of rewiring, and marginal correlation cannot distinguish it from
genuine rewiring.

Direct disambiguating test: hard-binarize each cell (>0.5) into its exact combinatorial
leaf (the same leaf indexing get_rules/parent_heatmap uses) for a few genes, and compare
organoid_shGFP's mean ACTUAL target value within each leaf against GEMM's OWN fitted rule
value for that exact leaf -- conditioning on the full regulator combination, not just one
pairwise relationship. Good leaf-conditional agreement despite bad marginal-correlation/R2
would mean the logic itself transfers and the R2 shortfall is a composition effect; poor
leaf-conditional agreement would mean the logic genuinely differs (real rewiring).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_leaf_conditional_agreement.py
"""

import os
import pickle

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import bobaT.utils as ut
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples"
TEST_GENES = ["LMX1B", "ASCL1", "NFIB", "TCF7L1", "EHF"]
MIN_CELLS_PER_LEAF = 10


def leaf_index_for_cell(row, regulators):
    bits = "".join("1" if row[r] > 0.5 else "0" for r in regulators)
    return ut.state2idx(bits)


def leaf_conditional_table(data, gene, regulators, rule):
    leaves = data.apply(lambda row: leaf_index_for_cell(row, regulators), axis=1)
    df = pd.DataFrame({"leaf": leaves, "actual": data[gene]})
    grouped = df.groupby("leaf")["actual"].agg(["mean", "count"])
    grouped["gemm_rule_value"] = [rule[leaf] for leaf in grouped.index]
    grouped["residual"] = grouped["mean"] - grouped["gemm_rule_value"]
    return grouped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    test_cellids = set(split_indices["test_cellID"])

    gemm_test = bb.load.load_data(
        f"{DIR_PREFIX}/6667/data_split/test_t0combined.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    shgfp = bb.load.load_data(
        f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    all_rows = []
    for gene in TEST_GENES:
        regulators = regulators_dict[gene]
        rule = rules[gene]
        print(f"\n=== {gene} ({len(regulators)} regulators, {2**len(regulators)} leaves) ===")

        gemm_table = leaf_conditional_table(gemm_test, gene, regulators, rule)
        gemm_well_populated = gemm_table[gemm_table["count"] >= MIN_CELLS_PER_LEAF]
        print(f"GEMM held-out test (positive control): {len(gemm_well_populated)} well-populated leaves "
              f"(>= {MIN_CELLS_PER_LEAF} cells), mean |residual| = {gemm_well_populated['residual'].abs().mean():.4f}")

        org_table = leaf_conditional_table(shgfp, gene, regulators, rule)
        org_well_populated = org_table[org_table["count"] >= MIN_CELLS_PER_LEAF]
        print(f"organoid_shGFP: {len(org_well_populated)} well-populated leaves (>= {MIN_CELLS_PER_LEAF} cells), "
              f"mean |residual| = {org_well_populated['residual'].abs().mean():.4f}")
        print(org_well_populated.sort_values("count", ascending=False).head(8))

        for leaf, row in org_well_populated.iterrows():
            all_rows.append({
                "gene": gene, "leaf": leaf, "n_cells_organoid": row["count"],
                "organoid_actual_mean": row["mean"], "gemm_rule_value": row["gemm_rule_value"],
                "residual": row["residual"],
            })

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(f"{OUT_DIR}/leaf_conditional_agreement_organoid_shgfp.csv", index=False)

    print(f"\n=== Overall: organoid_shGFP leaf-conditional residuals across all {len(TEST_GENES)} test genes ===")
    print(f"Mean |residual| (weighted by leaf, unweighted by cell count): {all_df['residual'].abs().mean():.4f}")
    print(f"Fraction of well-populated leaves with |residual| < 0.15: {(all_df['residual'].abs() < 0.15).mean():.3f}")

    print(f"\nWrote {OUT_DIR}/leaf_conditional_agreement_organoid_shgfp.csv")


if __name__ == "__main__":
    main()
