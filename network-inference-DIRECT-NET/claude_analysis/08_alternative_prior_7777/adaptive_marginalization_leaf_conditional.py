"""Prototype: instead of a single FIXED decision about which regulators to marginalize out
network-wide (test_regulator_ablation_transferability.py's per-gene best-single-drop), make
the marginalization decision ADAPTIVELY, per external sample, using the ahead-of-time scale
diagnostic (diagnose_per_gene_scale_collapse.py) as the trigger.

For each sample S: flag any gene g as an "unreliable regulator in S" if its raw distribution
in S deviates sharply from GEMM's own (same thresholds as the scale diagnostic: |mean shift|
> 3 GEMM-SDs, or within-sample std collapses to <10% of GEMM's own std). For every target
gene whose regulator set includes a flagged gene, marginalize that regulator out
(marginalize_rules.marginalize_rule) JUST for scoring sample S -- other samples where that
same gene isn't flagged keep the regulator. This is the natural generalization of the
already-validated single-best-drop ablation result: instead of guessing (or brute-forcing)
which regulator to drop, use the diagnostic to decide, per sample, whether a drop is even
warranted.

Uses the fast leaf-conditional transferability metric (not full fit_validation) for this
first pass, matching this session's own pattern (cheap check before committing to the slow
confirmatory R2/AUC run).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/adaptive_marginalization_leaf_conditional.py
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "07_batch_effect_diagnostics"))
from leaf_conditional_transferability_with_ireland2025 import IRELAND_PATHS

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"
SELF_LOOP_GENES = {"TFDP1", "NFYC", "CREB1", "TCF4", "ZEB1", "ESR1", "STAT1", "RBPJ", "JUND", "NR6A1", "SOX9"}
MEAN_SHIFT_THRESHOLD = 8.0
STD_RATIO_THRESHOLD = 0.02


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    nodes_cols = [n for n in nodes if n != "RORA_RORB"] + ["RORA_RORB"]

    gemm = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm.columns = [c.upper() for c in gemm.columns]
    gemm_mean = gemm[nodes_cols].mean()
    gemm_std = gemm[nodes_cols].std()

    # Same raw (un-normalized) CSVs used for the scale diagnostic -- need these for the
    # flagging decision -- AND the norm=0.3 loaded version for leaf_conditional_score.
    sample_paths = base_gather_sample_paths()
    sample_paths.update(IRELAND_PATHS)

    rows_baseline = []
    rows_adaptive = []
    n_flags_total = 0

    for name, path in sample_paths.items():
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path, index_col=0)
        raw.columns = [c.upper() for c in raw.columns]
        d_mean = raw[[c for c in nodes_cols if c in raw.columns]].mean()
        d_std = raw[[c for c in nodes_cols if c in raw.columns]].std()

        flagged = set()
        for gene in nodes_cols:
            if gene not in d_mean.index:
                continue
            shift = abs((d_mean[gene] - gemm_mean[gene]) / gemm_std[gene])
            ratio = d_std[gene] / gemm_std[gene]
            if shift > MEAN_SHIFT_THRESHOLD or ratio < STD_RATIO_THRESHOLD:
                flagged.add(gene)
        n_flags_total += len(flagged)

        d_normed = bb.load.load_data(path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)

        for gene in nodes:
            if gene in SELF_LOOP_GENES:
                continue
            regs = regulators_dict[gene]
            if len(regs) == 0:
                continue

            t_base, _, _ = leaf_conditional_score(d_normed, gene, regs, rules[gene])
            if not np.isnan(t_base):
                rows_baseline.append({"gene": gene, "sample": name, "transferability": t_base})

            gene_flagged_regs = [r for r in regs if r in flagged]
            if gene_flagged_regs:
                marg_rule, marg_regs = marginalize_rule(rules[gene], regs, gene_flagged_regs)
                t_adapt, _, _ = leaf_conditional_score(d_normed, gene, marg_regs, marg_rule)
            else:
                t_adapt = t_base
            if not np.isnan(t_adapt):
                rows_adaptive.append({"gene": gene, "sample": name, "transferability": t_adapt,
                                       "n_regulators_dropped": len(gene_flagged_regs)})
        print(f"{name}: {len(flagged)} genes flagged as scale-mismatched regulators")

    baseline_df = pd.DataFrame(rows_baseline)
    adaptive_df = pd.DataFrame(rows_adaptive)
    merged = baseline_df.merge(adaptive_df, on=["gene", "sample"], suffixes=("_baseline", "_adaptive"))
    merged.to_csv(f"{OUT_DIR}/adaptive_marginalization_leaf_conditional_strict.csv", index=False)

    print(f"\nTotal (gene, sample) flags across population: {n_flags_total}")
    print(f"Mean transferability: baseline={merged['transferability_baseline'].mean():.4f}, "
          f"adaptive={merged['transferability_adaptive'].mean():.4f}")
    changed = merged[merged["n_regulators_dropped"] > 0]
    print(f"\n{len(changed)}/{len(merged)} (gene,sample) pairs had >=1 regulator adaptively dropped")
    if len(changed):
        print(f"Among those: mean baseline={changed['transferability_baseline'].mean():.4f}, "
              f"mean adaptive={changed['transferability_adaptive'].mean():.4f}, "
              f"mean diff={( changed['transferability_adaptive']-changed['transferability_baseline']).mean():.4f}")
        print(f"Fraction improved: {(changed['transferability_adaptive'] > changed['transferability_baseline']).mean():.3f}")

    print(f"\nWrote {OUT_DIR}/adaptive_marginalization_leaf_conditional.csv")


if __name__ == "__main__":
    main()
