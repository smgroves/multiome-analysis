"""Score BoBa-T's 6667 rules against the ComBat-corrected + pooled-quantile-normalized
network-gene data (combat_correct_network_genes.py output), using norm=None (the data is
already appropriately scaled -- see that script's docstring for why the standard per-sample
norm=0.3 would erase the correction). Compares mean R2/AUC before (already-scored, standard
pipeline) vs after (ComBat) for each of the 18 non-GEMM datasets in scope.

Explicit guard against the numerical failure mode that broke the earlier global-reference-
norm prototype (R2=-1.8e14, see combat_correct_network_genes.py docstring): flags any
gene whose "actual" column has near-zero std (<1e-3) in a given dataset as degenerate,
excludes it from that dataset's mean R2/AUC, and reports the exclusion count explicitly
rather than letting it silently blow up an average.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/combat_run_validation_and_compare.py
"""

import glob
import os
import re

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
COMBAT_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics/combat_corrected"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics"
BRCD = "6667"
DEGENERATE_STD_THRESHOLD = 1e-3

sys_path_misc = f"{DIR_PREFIX}/claude_analysis/misc"
import sys
sys.path.insert(0, sys_path_misc)
from fixed_get_sklearn_metrics import get_sklearn_metrics_fixed


def gather_combat_paths():
    paths = {}
    for f in glob.glob(f"{COMBAT_DIR}/adata_*_combat_globalnorm.csv"):
        name = re.search(r"adata_(.+)_combat_globalnorm", f).group(1)
        if name == "GEMM_train":
            continue
        paths[name] = f
    return paths


def gather_baseline_r2_auc(name):
    if name.startswith("allograft_"):
        c = f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"
    elif name == "organoid_combined":
        c = f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv"
    else:
        c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
        if not os.path.exists(c):
            c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"
    if not os.path.exists(c):
        return np.nan, np.nan
    df = pd.read_csv(c)
    return df["r2"].mean(), df["roc_auc_score"].mean()


def run_one(name, path, nodes, regulators_dict, rules):
    data_test = bb.load.load_data(
        path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    VAL_DIR = f"{DIR_PREFIX}/{BRCD}/validation/external_validation_combat/{name}"
    os.makedirs(VAL_DIR, exist_ok=True)

    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
        data_test, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=name,
    )
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
    stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)

    # Guard against the degenerate-variance failure mode.
    degenerate = []
    for gene in stats["gene"]:
        val_csv = f"{VAL_DIR}/accuracy_plots/{gene}_validation.csv"
        if os.path.exists(val_csv):
            actual_std = pd.read_csv(val_csv)["actual"].std()
            if actual_std < DEGENERATE_STD_THRESHOLD:
                degenerate.append(gene)

    stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)
    clean = stats[~stats["gene"].isin(degenerate)]
    return stats, clean, degenerate


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/{BRCD}/rules/rules_{BRCD}.txt")

    combat_paths = gather_combat_paths()
    print(f"{len(combat_paths)} datasets to revalidate under ComBat correction: {list(combat_paths.keys())}")

    rows = []
    for name, path in combat_paths.items():
        print(f"\n=== {name} ===")
        stats, clean, degenerate = run_one(name, path, nodes, regulators_dict, rules)
        baseline_r2, baseline_auc = gather_baseline_r2_auc(name)
        row = {
            "name": name,
            "baseline_r2": baseline_r2,
            "combat_r2_all_genes": stats["r2"].mean(),
            "combat_r2_excl_degenerate": clean["r2"].mean(),
            "baseline_auc": baseline_auc,
            "combat_auc_all_genes": stats["roc_auc_score"].mean(),
            "combat_auc_excl_degenerate": clean["roc_auc_score"].mean(),
            "n_genes_scored": len(stats),
            "n_degenerate_excluded": len(degenerate),
            "degenerate_genes": ";".join(degenerate),
        }
        rows.append(row)
        print(f"{name}: baseline R2={baseline_r2:.3f} -> combat R2={clean['r2'].mean():.3f} "
              f"({len(degenerate)} degenerate genes excluded: {degenerate})")

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/combat_vs_baseline_comparison.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Full comparison ===")
    print(result[["name", "baseline_r2", "combat_r2_excl_degenerate", "baseline_auc", "combat_auc_excl_degenerate", "n_degenerate_excluded"]].to_string(index=False))

    valid = result.dropna(subset=["baseline_r2", "combat_r2_excl_degenerate"])
    print(f"\nMean baseline R2: {valid['baseline_r2'].mean():.4f}")
    print(f"Mean ComBat-corrected R2 (excl. degenerate genes): {valid['combat_r2_excl_degenerate'].mean():.4f}")
    print(f"Mean baseline AUC: {valid['baseline_auc'].mean():.4f}")
    print(f"Mean ComBat-corrected AUC (excl. degenerate genes): {valid['combat_auc_excl_degenerate'].mean():.4f}")
    n_improved = (valid["combat_r2_excl_degenerate"] > valid["baseline_r2"]).sum()
    print(f"Datasets where ComBat improved mean R2: {n_improved}/{len(valid)}")

    print(f"\nWrote {OUT_DIR}/combat_vs_baseline_comparison.csv")


if __name__ == "__main__":
    main()
