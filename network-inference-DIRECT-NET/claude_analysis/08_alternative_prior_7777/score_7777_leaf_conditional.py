"""Score barcode 7777's rules (aggregate-evidence prior, fit_rules_aggregate_prior.py)
against the same 44-sample population already scored for 6667
(leaf_conditional_transferability_with_ireland2025.py's output), using the identical
leaf-conditional-transferability metric, for a fast, apples-to-apples comparison before
committing to the much slower full fit_validation/R2/AUC run.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/score_7777_leaf_conditional.py
"""

import os
import sys

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_domain_shift_diagnostics"))
from diagnose_leaf_conditional_agreement_full import gather_sample_paths as base_gather_sample_paths
from diagnose_leaf_conditional_agreement_full import leaf_conditional_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "07_batch_effect_diagnostics"))
from leaf_conditional_transferability_with_ireland2025 import IRELAND_PATHS

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"
SELF_LOOP_GENES = {"TFDP1", "NFYC", "CREB1", "TCF4", "ZEB1", "ESR1", "STAT1", "RBPJ", "JUND", "NR6A1", "SOX9"}


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules_7777, regulators_dict_7777 = bb.load.load_rules(fname=f"{DIR_PREFIX}/7777/rules/rules_7777.txt")

    sample_paths = base_gather_sample_paths()
    sample_paths.update(IRELAND_PATHS)

    rows = []
    for name, path in sample_paths.items():
        if not os.path.exists(path):
            continue
        d = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        for gene in nodes:
            regulators = regulators_dict_7777[gene]
            if len(regulators) == 0:
                continue
            t, mar, n_leaves = leaf_conditional_score(d, gene, regulators, rules_7777[gene])
            rows.append({"gene": gene, "sample": name, "transferability": t})
        print(f"scored {name}")

    result_7777 = pd.DataFrame(rows)
    result_7777.to_csv(f"{OUT_DIR}/leaf_conditional_transferability_7777.csv", index=False)

    result_6667 = pd.read_csv(f"{DIR_PREFIX}/comparisons/ireland2025_external_validation/leaf_conditional_transferability_matrix.csv")
    result_6667 = result_6667[result_6667["sample"] != "GEMM_held_out_test"]

    merged = result_7777.merge(result_6667[["gene", "sample", "transferability"]], on=["gene", "sample"], suffixes=("_7777", "_6667"))
    merged["diff"] = merged["transferability_7777"] - merged["transferability_6667"]
    merged.to_csv(f"{OUT_DIR}/leaf_conditional_7777_vs_6667.csv", index=False)

    ext = merged[~merged["gene"].isin(SELF_LOOP_GENES)]
    print(f"\nMean transferability (excl self-loops): 6667={ext['transferability_6667'].mean():.4f}, "
          f"7777={ext['transferability_7777'].mean():.4f}, diff={ext['diff'].mean():.4f}")
    print(f"Mean |diff| across all (gene,sample) pairs: {ext['diff'].abs().mean():.4f}")
    print(f"Fraction of pairs improved (7777 > 6667): {(ext['diff'] > 0).mean():.3f}")

    per_gene = ext.groupby("gene")["diff"].mean().sort_values()
    pd.set_option("display.width", 160)
    print("\n=== Genes most HURT by the aggregate prior (7777 < 6667) ===")
    print(per_gene.head(10))
    print("\n=== Genes most HELPED by the aggregate prior (7777 > 6667) ===")
    print(per_gene.tail(10))

    print(f"\nWrote {OUT_DIR}/leaf_conditional_transferability_7777.csv, {OUT_DIR}/leaf_conditional_7777_vs_6667.csv")


if __name__ == "__main__":
    main()
