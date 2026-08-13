"""Extended version of decisive_test_rewiring_vs_rescaling.py: run the naive-vs-adaptive
leaf-conditional comparison across EVERY gene x every sample in the 44-sample population
(33 original + 11 Ireland 2025), not just organoid_shGFP's original 5 sec-9 genes. Reports
both the binary "transferability" metric (matches sec 9's original methodology) and the
continuous mean_abs_residual (more robust -- see EHF's threshold-sensitivity case in the
single-sample version, where binary transferability dropped to 0 while the continuous
residual actually improved).

Produces a per-(gene,sample) verdict: "unaffected" (no flags -- any poor transferability
here is NOT explained by rescaling, i.e. more likely genuine rewiring), "improved"
(rescaling substantially helped, i.e. was at least partly a calibration artifact), or
"ambiguous" (binary metric moved one way, continuous residual moved the other -- exactly
EHF's case, don't trust either direction confidently).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/decisive_test_all_samples.py
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
from adaptive_gmm_rescale import build_adaptive_data, flag_genes

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"
SELF_LOOP_GENES = {"TFDP1", "NFYC", "CREB1", "TCF4", "ZEB1", "ESR1", "STAT1", "RBPJ", "JUND", "NR6A1", "SOX9"}


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
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
        flagged = flag_genes(raw, nodes)
        adaptive, _ = build_adaptive_data(raw, normed, nodes, flagged_genes=flagged)

        for gene in nodes:
            if gene in SELF_LOOP_GENES:
                continue
            regs = regulators_dict[gene]
            if len(regs) == 0:
                continue
            regs_flagged = [r for r in regs if r in flagged]
            if not regs_flagged and gene not in flagged:
                continue  # nothing to compare -- naive==adaptive by construction, skip for speed

            t_naive, resid_naive, _ = leaf_conditional_score(normed, gene, regs, rules[gene])
            t_adapt, resid_adapt, _ = leaf_conditional_score(adaptive, gene, regs, rules[gene])
            if np.isnan(t_naive) or np.isnan(t_adapt):
                continue

            binary_diff = t_adapt - t_naive
            resid_diff = resid_naive - resid_adapt  # positive = residual shrank = improved
            if binary_diff > 0.02 and resid_diff > 0:
                verdict = "improved"
            elif binary_diff < -0.02 and resid_diff < 0:
                verdict = "worsened"
            elif abs(binary_diff) <= 0.02 and abs(resid_diff) < 0.01:
                verdict = "no_change"
            else:
                verdict = "ambiguous"  # binary and continuous metrics disagree in direction

            rows.append({
                "gene": gene, "sample": name, "n_regulators_flagged": len(regs_flagged),
                "gene_itself_flagged": gene in flagged,
                "naive_transferability": t_naive, "adaptive_transferability": t_adapt, "binary_diff": binary_diff,
                "naive_residual": resid_naive, "adaptive_residual": resid_adapt, "resid_diff": resid_diff,
                "verdict": verdict,
            })
        print(f"{name}: {len(flagged)} genes flagged")

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/decisive_test_all_samples.csv", index=False)

    print(f"\n{len(result)} (gene,sample) pairs where rescaling changed at least one regulator or the gene itself")
    print("\n=== Verdict breakdown ===")
    print(result["verdict"].value_counts())
    print(f"\nMean binary_diff: {result['binary_diff'].mean():.4f}, mean resid_diff: {result['resid_diff'].mean():.4f}")

    print("\n=== Per-gene verdict summary (how often each gene improves vs worsens vs is ambiguous, when affected) ===")
    per_gene = result.groupby("gene")["verdict"].value_counts().unstack(fill_value=0)
    per_gene["n_total"] = per_gene.sum(axis=1)
    per_gene = per_gene.sort_values("n_total", ascending=False)
    pd.set_option("display.width", 160)
    print(per_gene.head(20).to_string())

    print(f"\nWrote {OUT_DIR}/decisive_test_all_samples.csv")


if __name__ == "__main__":
    main()
