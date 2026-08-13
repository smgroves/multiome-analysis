"""Deliberate regulator ablation: for every network gene with >=2 regulators, marginalize
out EACH of its regulators in turn (using marginalize_rules.marginalize_rule, built for the
missing-gene case but reused here on purpose even though we HAVE data for the dropped
regulator), and check whether the resulting smaller rule's leaf-conditional transferability
(same metric as diagnose_leaf_conditional_agreement_full.py) is higher than the full rule's
-- i.e. is some specific regulator actively HURTING a gene's cross-context transferability,
such that dropping it (not just failing to help) produces a more portable rule?

Scored against the full 44-sample population (original 33 + 11 Ireland 2025 condition
splits) for a stable, well-powered average per (gene, dropped_regulator) pair. Self-loop-
only genes (1 regulator) are skipped -- ablating a gene's only regulator leaves nothing to
condition on.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/test_regulator_ablation_transferability.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "06_marginal_rule_validation"))
from marginalize_rules import marginalize_rule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_conditional_transferability_with_ireland2025 import IRELAND_PATHS

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/ireland2025_external_validation"
SELF_LOOP_GENES = {"TFDP1", "NFYC", "CREB1", "TCF4", "ZEB1", "ESR1", "STAT1", "RBPJ", "JUND", "NR6A1", "SOX9"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    sample_paths = base_gather_sample_paths()
    sample_paths.update(IRELAND_PATHS)

    print(f"Loading {len(sample_paths)} samples...")
    sample_data = {}
    for name, path in sample_paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        sample_data[name] = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
    print(f"Loaded {len(sample_data)} samples.")

    rows = []
    for gene in nodes:
        if gene in SELF_LOOP_GENES:
            continue
        regulators = regulators_dict[gene]
        if len(regulators) < 2:
            continue

        # Baseline: full rule, all regulators.
        baseline_scores = []
        for name, d in sample_data.items():
            t, _, _ = leaf_conditional_score(d, gene, regulators, rules[gene])
            if not np.isnan(t):
                baseline_scores.append(t)
        baseline_mean = np.mean(baseline_scores) if baseline_scores else np.nan
        rows.append({"gene": gene, "dropped_regulator": None, "n_regulators": len(regulators),
                      "mean_transferability": baseline_mean, "n_samples_scored": len(baseline_scores)})

        for reg_to_drop in regulators:
            marg_rule, marg_regs = marginalize_rule(rules[gene], regulators, [reg_to_drop])
            scores = []
            for name, d in sample_data.items():
                t, _, _ = leaf_conditional_score(d, gene, marg_regs, marg_rule)
                if not np.isnan(t):
                    scores.append(t)
            mean_t = np.mean(scores) if scores else np.nan
            rows.append({"gene": gene, "dropped_regulator": reg_to_drop, "n_regulators": len(marg_regs),
                         "mean_transferability": mean_t, "n_samples_scored": len(scores)})
        print(f"{gene}: baseline={baseline_mean:.3f}, {len(regulators)} regulators tested")

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/regulator_ablation_transferability.csv", index=False)

    # For each gene, find the best ablation (if any) vs its own baseline.
    summary_rows = []
    for gene, sub in result.groupby("gene"):
        baseline = sub[sub["dropped_regulator"].isna()]["mean_transferability"].iloc[0]
        ablations = sub[sub["dropped_regulator"].notna()]
        if len(ablations) == 0:
            continue
        best = ablations.loc[ablations["mean_transferability"].idxmax()]
        summary_rows.append({
            "gene": gene, "baseline_transferability": baseline,
            "best_dropped_regulator": best["dropped_regulator"],
            "best_ablation_transferability": best["mean_transferability"],
            "improvement": best["mean_transferability"] - baseline,
            "n_regulators_remaining": best["n_regulators"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("improvement", ascending=False)
    summary.to_csv(f"{OUT_DIR}/regulator_ablation_best_per_gene.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Genes most improved by dropping their single worst regulator ===")
    print(summary.head(15).to_string(index=False))
    print(f"\n{(summary['improvement'] > 0.05).sum()}/{len(summary)} genes improve by >0.05 transferability from a single-regulator ablation")
    print(f"Mean improvement across all genes: {summary['improvement'].mean():.4f}")
    print(f"\nWrote {OUT_DIR}/regulator_ablation_transferability.csv, {OUT_DIR}/regulator_ablation_best_per_gene.csv")


if __name__ == "__main__":
    main()
