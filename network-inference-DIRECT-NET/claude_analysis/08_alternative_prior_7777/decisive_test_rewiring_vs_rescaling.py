"""THE decisive test: is organoid_shGFP's originally-documented "genuine rewiring"
(domain_shift_diagnostic_and_organoid_walks/FINDINGS.md sec 9 -- LMX1B, ASCL1, NFIB,
TCF7L1, EHF hard-binarized into exact combinatorial leaves, compared against GEMM's fitted
rule value for that exact leaf) actually genuine, or is it partly/wholly an artifact of
per-sample quantile-clip miscalibration for scale-mismatched genes?

Sec 9's original test used bobaT's standard norm=0.3 to decide each cell's leaf membership
-- it controlled for the POPULATION-MIXTURE confound (grouping by average state vs exact
combinatorial state) but never controlled for the SCALE-CALIBRATION confound this session's
diagnose_per_gene_scale_collapse.py found pervasively. This script reruns the exact same
leaf-conditional-agreement test for those same 5 genes on organoid_shGFP, once with the
original norm=0.3 data (reproducing sec 9) and once with adaptive_gmm_rescale.py's
GEMM-referenced-GMM rescaling applied to any of that gene's regulators (and the gene
itself) that are flagged as scale-mismatched in organoid_shGFP specifically. If the
disagreement shrinks substantially under the adaptive version, sec 9's rewiring conclusion
was partly a rescaling artifact; if it persists, it's real.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/decisive_test_rewiring_vs_rescaling.py
"""

import os
import sys

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_domain_shift_diagnostics"))
from diagnose_leaf_conditional_agreement_full import leaf_conditional_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_gmm_rescale import build_adaptive_data, flag_genes

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
SAMPLE_PATH = f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv"
SEC9_GENES = ["LMX1B", "ASCL1", "NFIB", "TCF7L1", "EHF"]
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    raw = pd.read_csv(SAMPLE_PATH, index_col=0)
    raw.columns = [c.upper() for c in raw.columns]
    normed = bb.load.load_data(SAMPLE_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)

    flagged = flag_genes(raw, nodes)
    print(f"organoid_shGFP: {len(flagged)}/{len(nodes)} genes flagged as scale-mismatched vs GEMM")
    print(f"Flagged: {sorted(flagged)}\n")

    adaptive, _ = build_adaptive_data(raw, normed, nodes, flagged_genes=flagged)

    print("=== sec 9's original 5 genes: naive (norm=0.3) vs adaptive (GEMM-GMM-rescaled flagged genes) ===")
    rows = []
    for gene in SEC9_GENES:
        regs = regulators_dict[gene]
        regs_flagged = [r for r in regs if r in flagged]
        t_naive, resid_naive, n_leaves_naive = leaf_conditional_score(normed, gene, regs, rules[gene])
        t_adapt, resid_adapt, n_leaves_adapt = leaf_conditional_score(adaptive, gene, regs, rules[gene])
        rows.append({
            "gene": gene, "n_regulators": len(regs), "regulators_flagged": ";".join(regs_flagged),
            "gene_itself_flagged": gene in flagged,
            "naive_transferability": t_naive, "naive_mean_abs_residual": resid_naive,
            "adaptive_transferability": t_adapt, "adaptive_mean_abs_residual": resid_adapt,
            "improvement": t_adapt - t_naive if not (np.isnan(t_adapt) or np.isnan(t_naive)) else np.nan,
        })
        print(f"{gene}: regulators_flagged={regs_flagged or 'none'}, gene_itself_flagged={gene in flagged}")
        print(f"  naive:    transferability={t_naive:.3f}  mean|residual|={resid_naive:.3f}")
        print(f"  adaptive: transferability={t_adapt:.3f}  mean|residual|={resid_adapt:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/decisive_test_organoid_shGFP_sec9_genes.csv", index=False)

    print("\n=== Full network: all genes, organoid_shGFP, naive vs adaptive ===")
    all_rows = []
    for gene in nodes:
        regs = regulators_dict[gene]
        if len(regs) == 0:
            continue
        t_naive, _, _ = leaf_conditional_score(normed, gene, regs, rules[gene])
        t_adapt, _, _ = leaf_conditional_score(adaptive, gene, regs, rules[gene])
        all_rows.append({"gene": gene, "naive": t_naive, "adaptive": t_adapt})
    all_df = pd.DataFrame(all_rows).dropna()
    all_df.to_csv(f"{OUT_DIR}/decisive_test_organoid_shGFP_all_genes.csv", index=False)
    print(f"Mean transferability across all {len(all_df)} scoreable genes: naive={all_df['naive'].mean():.4f}, "
          f"adaptive={all_df['adaptive'].mean():.4f}")
    print(f"Fraction improved: {(all_df['adaptive'] > all_df['naive']).mean():.3f}")

    print(f"\nWrote {OUT_DIR}/decisive_test_organoid_shGFP_{{sec9_genes,all_genes}}.csv")


if __name__ == "__main__":
    main()
