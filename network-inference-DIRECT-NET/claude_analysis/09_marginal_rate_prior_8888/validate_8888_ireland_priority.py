"""Validate the corrected 8888 refit (remove_selfloops=True, matching 6667) against just
the Ireland et al. 2025 samples first -- these are flagged as most important, so check them
before spending time on the full 43-sample population. Ordered smallest-first for fast
incremental feedback.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "07_batch_effect_diagnostics"))
from leaf_conditional_transferability_with_ireland2025 import IRELAND_PATHS

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = ("networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
                "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv")
BRCD = "8888"
NORM = 0.3

# smallest-first (by cell count, checked via wc -l on each CSV)
IRELAND_ORDER = [
    "organoid_celltag_RPM", "tbo_allograft_5khvg_RPM_CTpreCre", "organoid_celltag_WT",
    "tbo_allograft_5khvg_RPM_CTpostCre", "organoid_celltag_RPMA", "cgrp_k5_CGRP",
    "rpr2_allograft_RPR2", "celltag_fate_dpt_RPMA", "rpr2_allograft_RPM",
    "celltag_fate_dpt_RPM", "cgrp_k5_K5",
]


def gather_baseline(name):
    c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
    if not os.path.exists(c):
        c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"
    if not os.path.exists(c):
        return np.nan, np.nan, np.nan
    df = pd.read_csv(c)
    return df["r2"].mean(), df["r2"].median(), df["roc_auc_score"].mean()


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/{BRCD}/rules/rules_{BRCD}.txt")

    rows = []
    for name in IRELAND_ORDER:
        path = IRELAND_PATHS[name]
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"\n=== {name} ===")
        data_test = bb.load.load_data(path, nodes, norm=NORM, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        VAL_DIR = f"{DIR_PREFIX}/{BRCD}/validation/external_validation/{name}"
        os.makedirs(VAL_DIR, exist_ok=True)

        _, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
            save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=name,
        )
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
        stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)
        stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)

        baseline_r2, baseline_r2_median, baseline_auc = gather_baseline(name)
        row = {
            "name": name, "n_cells": data_test.shape[0],
            "baseline_r2": baseline_r2, "r2_8888": stats["r2"].mean(),
            "baseline_r2_median": baseline_r2_median, "r2_8888_median": stats["r2"].median(),
            "baseline_auc": baseline_auc, "auc_8888": stats["roc_auc_score"].mean(),
        }
        rows.append(row)
        print(f"{name} (n={row['n_cells']}): R2 mean {baseline_r2:.3f}->{row['r2_8888']:.3f}, "
              f"median {baseline_r2_median:.3f}->{row['r2_8888_median']:.3f}, "
              f"AUC {baseline_auc:.3f}->{row['auc_8888']:.3f}")
        pd.DataFrame(rows).to_csv(f"{DIR_PREFIX}/claude_analysis/09_marginal_rate_prior_8888/8888_refit_vs_6667_ireland.csv", index=False)

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n=== Ireland 2025 comparison (corrected 8888 refit) ===")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
