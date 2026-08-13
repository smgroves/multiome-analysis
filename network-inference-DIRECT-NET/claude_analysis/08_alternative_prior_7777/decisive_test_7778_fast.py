"""Fast leaf-conditional check for barcode 7778 (rescale mean-shift + marginalize
variance-collapse) across the 44-sample population, mirroring decisive_test_all_samples_v2.py's
structure exactly so the two are directly comparable -- 7778 and 7779 differ only in the
variance_collapse branch.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/decisive_test_7778_fast.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "07_batch_effect_diagnostics"))
from leaf_conditional_transferability_with_ireland2025 import IRELAND_PATHS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_rescale_v2 import classify_genes
from adaptive_hybrid_7778 import build_hybrid_data_and_rules

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
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    sample_paths = base_gather_sample_paths()
    sample_paths.update(IRELAND_PATHS)

    rows = []
    for name, path in sample_paths.items():
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path, index_col=0)
        raw.columns = [c.upper() for c in raw.columns]
        normed = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        classification = classify_genes(raw, nodes)
        adaptive, rules_hybrid, regs_hybrid, nodes_hybrid = build_hybrid_data_and_rules(
            raw, normed, nodes, rules, regulators_dict, classification=classification,
        )
        dropped = set(nodes) - set(nodes_hybrid)

        for gene in nodes:
            if gene in SELF_LOOP_GENES or gene in dropped:
                continue
            regs_naive = regulators_dict[gene]
            if len(regs_naive) == 0:
                continue
            regs_affected = [r for r in regs_naive if classification.get(r) is not None] or (
                [gene] if classification.get(gene) is not None else []
            )
            if not regs_affected:
                continue  # nothing changed for this gene in this sample

            t_naive, resid_naive, _ = leaf_conditional_score(normed, gene, regs_naive, rules[gene])
            regs_hybrid_gene = regs_hybrid[gene]
            t_hybrid, resid_hybrid, _ = leaf_conditional_score(adaptive, gene, regs_hybrid_gene, rules_hybrid[gene])
            if np.isnan(t_naive) or np.isnan(t_hybrid):
                continue

            binary_diff = t_hybrid - t_naive
            resid_diff = resid_naive - resid_hybrid
            if binary_diff > 0.02 and resid_diff > 0:
                verdict = "improved"
            elif binary_diff < -0.02 and resid_diff < 0:
                verdict = "worsened"
            elif abs(binary_diff) <= 0.02 and abs(resid_diff) < 0.01:
                verdict = "no_change"
            else:
                verdict = "ambiguous"

            rows.append({
                "gene": gene, "sample": name, "n_regulators_dropped": len(regs_naive) - len(regs_hybrid_gene),
                "naive_transferability": t_naive, "hybrid_transferability": t_hybrid, "binary_diff": binary_diff,
                "naive_residual": resid_naive, "hybrid_residual": resid_hybrid, "resid_diff": resid_diff,
                "verdict": verdict,
            })
        print(f"{name}: {len(dropped)} genes dropped as targets (variance_collapse)")

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/decisive_test_7778_fast.csv", index=False)

    print(f"\n{len(result)} (gene,sample) pairs affected")
    print(result["verdict"].value_counts())
    print(f"\nMean binary_diff: {result['binary_diff'].mean():.4f}, mean resid_diff: {result['resid_diff'].mean():.4f}")

    per_gene = result.groupby("gene")["verdict"].value_counts().unstack(fill_value=0)
    per_gene["n_total"] = per_gene.sum(axis=1)
    per_gene = per_gene.sort_values("n_total", ascending=False)
    pd.set_option("display.width", 160)
    print("\n=== Per-gene verdict summary ===")
    print(per_gene.head(20).to_string())

    print(f"\nWrote {OUT_DIR}/decisive_test_7778_fast.csv")


if __name__ == "__main__":
    main()
