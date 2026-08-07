"""External validation for a dataset that's missing 1+ of the 6667 network's 53 genes,
comparing two ways of handling the missing gene(s) as regulators for other, present genes:

  (a) "zero-fill" baseline: insert a column of 0 for each missing gene, then run bobaT's
      unmodified fit_validation against the FULL, unmodified rules. Per parent_heatmap's
      leaf-weighting, a forced value of exactly 0 makes every leaf where that regulator is
      "on" get weight 0 and every leaf where it's "off" keep full weight -- i.e. this
      deterministically assumes the missing regulator is OFF, not "unknown".
  (b) "marginal": use marginalize_rules.build_marginal_rules to collapse each missing gene
      out of every rule that used it as a regulator (average its ON/OFF entries, a uniform
      prior), and drop it (and any other missing gene) entirely from the scored `nodes`
      list (no ground truth for it, so it can't be scored regardless of approach). No
      zero-filled column needed for these genes at all.

Both arms use bobaT's own unmodified fit_validation/parent_heatmap/plot_accuracy -- only
the `rules`/`regulators_dict`/`nodes` (marginal arm) or the input CSV (zero-fill arm) differ.
Scoring uses fixed_get_sklearn_metrics.get_sklearn_metrics_fixed so RORA_RORB isn't
mislabeled RORA (see claude_analysis/misc/fixed_get_sklearn_metrics.py).

Run in bobaT_env_py3.13 (same numpy shim as main_external_validation_organoid_mets.py):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python main_external_validation_marginal.py <sample>
"""

import os
import sys

import numpy as np
import pandas as pd

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import bobaT as bb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marginalize_rules import build_marginal_rules

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "misc"))
from fixed_get_sklearn_metrics import get_sklearn_metrics_fixed

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)

# Filled in once step-2 inspection of the Ireland 2025 datasets identifies which of the 53
# network genes (if any) are actually absent from a given dataset. Populate per sample as:
#   "sample_name": (["adata_..._v3_RORA_RORB_ave.csv" path, relative to DIR_PREFIX], [missing gene names])
DATASETS = {
    # "cgrp_k5": ("data/cgrp_k5/adata_cgrp_k5_v3_RORA_RORB_ave.csv", ["SOME_MISSING_GENE"]),
}


def zero_fill_csv(orig_path, missing_genes, tmp_path):
    """Write a copy of the dataset's CSV with a column of 0 added for each missing gene,
    reproducing the "assume regulator = OFF" baseline without bobaT.load.load_data's
    normal hard-fail on a wholly-absent node."""
    df = pd.read_csv(orig_path, index_col=0)
    for g in missing_genes:
        df[g] = 0.0
    df.to_csv(tmp_path)


def run_arm(label, data_path, data_nodes, score_nodes, regulators_dict, rules, val_dir, norm):
    """`data_nodes` is passed to load_data (must include every gene needed as a column --
    for the zero-fill arm that's the full 53, including the zero-filled missing ones, since
    they're still needed as regulator inputs; for the marginal arm it's just the present
    genes). `score_nodes` is what fit_validation actually iterates over/scores -- always
    the present-gene list, since a missing gene has no ground truth to validate against
    either way."""
    data_test = bb.load.load_data(
        data_path, data_nodes, norm=norm, delimiter=",", log1p=False, transpose=True,
        sample_order=False, fillna=0,
    )
    os.makedirs(val_dir, exist_ok=True)
    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
        data_test, data_test_t1=None, nodes=score_nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=val_dir, plot=True, show_plots=False, save_df=True, fname=label,
    )
    bb.tl.save_auc_by_gene(area_all, score_nodes, val_dir)
    summary_stats = get_sklearn_metrics_fixed(val_dir, score_nodes)
    summary_stats.to_csv(f"{val_dir}/summary_stats_fixed.csv", index=False)
    return summary_stats


def main(sample, brcd="6667", norm=0.3):
    if sample not in DATASETS:
        raise ValueError(f"{sample} not in DATASETS -- populate its path + missing-gene list first")
    rel_data_path, missing_genes = DATASETS[sample]
    data_path = f"{DIR_PREFIX}/{rel_data_path}"

    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False,
    )
    v_names, nodes_full = bb.utils.get_nodes(vertex_dict, graph)
    rules_full, regulators_dict_full = bb.load.load_rules(fname=f"{DIR_PREFIX}/{brcd}/rules/rules_{brcd}.txt")

    print(f"{sample}: {len(missing_genes)} missing network genes: {missing_genes}")

    # Both arms only ever SCORE present target genes (a missing gene has no ground-truth
    # "actual" column, so it can't be meaningfully validated as a target regardless of how
    # its own regulators are handled) -- they differ only in how a missing gene is treated
    # when it appears as a REGULATOR of some other, present target gene. `nodes_marg` (the
    # present-gene node list) is reused for both arms for this reason.
    rules_marg, regulators_dict_marg, nodes_marg = build_marginal_rules(
        rules_full, regulators_dict_full, nodes_full, missing_genes,
    )

    # (a) zero-fill baseline: full (unmarginalized) rules/regulator lists (need the full
    # 53-column data, with missing genes zero-filled, since parent_heatmap looks up every
    # regulator in regulators_dict_full[node] including the missing ones), but SCORED only
    # over present targets (nodes_marg) -- a missing gene has no ground truth either way.
    baseline_dir = f"{DIR_PREFIX}/{brcd}/validation/external_validation_marginal/{sample}_zerofill"
    tmp_csv = f"/tmp/{sample}_zerofilled.csv"
    zero_fill_csv(data_path, missing_genes, tmp_csv)
    baseline_stats = run_arm(
        f"{sample}_zerofill", tmp_csv, nodes_full, nodes_marg, regulators_dict_full, rules_full, baseline_dir, norm,
    )

    # (b) marginal: same scored nodes, but missing regulator genes are marginalized out of
    # every rule/regulator list that used them, so the data only ever needs present-gene
    # columns (data_nodes == score_nodes == nodes_marg here).
    marginal_dir = f"{DIR_PREFIX}/{brcd}/validation/external_validation_marginal/{sample}_marginal"
    marginal_stats = run_arm(
        f"{sample}_marginal", data_path, nodes_marg, nodes_marg, regulators_dict_marg, rules_marg, marginal_dir, norm,
    )

    # Compare, restricted to genes actually affected by a missing regulator (unaffected
    # genes should score identically in both arms -- a consistency check).
    affected = [
        n for n in nodes_marg
        if any(g in set(missing_genes) for g in regulators_dict_full[n])
    ]
    merged = baseline_stats.merge(marginal_stats, on="gene", suffixes=("_zerofill", "_marginal"))
    merged["affected_by_missing_regulator"] = merged["gene"].isin(affected)
    out_path = f"{DIR_PREFIX}/{brcd}/validation/external_validation_marginal/{sample}_comparison.csv"
    merged.to_csv(out_path, index=False)

    print(f"\n{sample}: {len(affected)} genes affected by missing regulator(s) {missing_genes}")
    cols = ["gene", "r2_zerofill", "r2_marginal", "roc_auc_score_zerofill", "roc_auc_score_marginal"]
    print(merged.loc[merged["affected_by_missing_regulator"], [c for c in cols if c in merged.columns]])
    print(f"-> {out_path}")


if __name__ == "__main__":
    sample_arg = sys.argv[1]
    brcd_arg = sys.argv[2] if len(sys.argv) > 2 else "6667"
    norm_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    main(sample_arg, brcd_arg, norm_arg)
