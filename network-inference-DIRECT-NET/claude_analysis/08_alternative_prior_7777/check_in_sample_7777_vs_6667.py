"""In-sample check: score both 6667's and 7777's rules against GEMM's own held-out TEST
split (6667/data_split/test_t0combined.csv -- never touched during fitting for either
barcode), to see whether 7777's external-validation gains (43/43 samples improved) come
with any in-sample cost, or whether it's a strictly better fit.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/check_in_sample_7777_vs_6667.py
"""

import os
import sys

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "misc"))
from fixed_get_sklearn_metrics import get_sklearn_metrics_fixed

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
TEST_PATH = f"{DIR_PREFIX}/6667/data_split/test_t0combined.csv"


def score(brcd, nodes, regulators_dict, rules, test_data):
    VAL_DIR = f"{DIR_PREFIX}/{brcd}/validation/in_sample_test_check"
    os.makedirs(VAL_DIR, exist_ok=True)
    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
        test_data, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=f"insample_{brcd}",
    )
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
    stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)
    stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)
    return stats


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    test_data = bb.load.load_data(
        TEST_PATH, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    print(f"GEMM held-out test set: {test_data.shape[0]} cells")

    rules_6667, regs_6667 = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")
    rules_7777, regs_7777 = bb.load.load_rules(fname=f"{DIR_PREFIX}/7777/rules/rules_7777.txt")

    print("\nScoring 6667 in-sample...")
    stats_6667 = score("6667", nodes, regs_6667, rules_6667, test_data)
    print("Scoring 7777 in-sample...")
    stats_7777 = score("7777", nodes, regs_7777, rules_7777, test_data)

    print(f"\n6667 in-sample: mean R2={stats_6667['r2'].mean():.4f}, median R2={stats_6667['r2'].median():.4f}, mean AUC={stats_6667['roc_auc_score'].mean():.4f}")
    print(f"7777 in-sample: mean R2={stats_7777['r2'].mean():.4f}, median R2={stats_7777['r2'].median():.4f}, mean AUC={stats_7777['roc_auc_score'].mean():.4f}")

    merged = stats_6667[["gene", "r2", "roc_auc_score"]].merge(
        stats_7777[["gene", "r2", "roc_auc_score"]], on="gene", suffixes=("_6667", "_7777"),
    )
    merged["r2_diff"] = merged["r2_7777"] - merged["r2_6667"]
    merged.to_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/in_sample_6667_vs_7777.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"\nGenes improved in-sample: {(merged['r2_diff'] > 0).sum()}/{len(merged)}")
    print("\n=== Genes with the biggest in-sample R2 changes ===")
    print(merged.sort_values("r2_diff").to_string(index=False))


if __name__ == "__main__":
    main()
