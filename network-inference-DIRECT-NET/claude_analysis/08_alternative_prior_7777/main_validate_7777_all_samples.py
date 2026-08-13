"""Full R2/AUC validation of barcode 7777 (aggregate-evidence prior,
fit_rules_aggregate_prior.py) against the complete external-validation population: 12
allografts, 16 human tumors, organoid (combined + 3 conditions), mets_compiled, and the 16
Ireland et al. 2025 samples (5 whole + 11 condition splits) -- same population used
throughout this investigation. Generalizes main_external_validation_organoid_mets.py's
brcd-parameterized pattern to the allograft/human-tumor samples too (the older
main_external_validation.py/_copy.py hardcode brcd=6667, not reused here).

The decisive test after the leaf-conditional transferability check (which showed a
promising but threshold-sensitive improvement, mean 0.379->0.455 excl. self-loops) --
R2/AUC are continuous and don't share that sensitivity to a value crossing 0.15 by chance.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/main_validate_7777_all_samples.py
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
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
BRCD = "7777"
NORM = 0.3
ALLOGRAFTS = ["1L", "2L", "2LR", "3L", "5B", "TKO-luc", "mt2", "mt3", "mt4", "mt4Rf", "mt5", "mt6"]
HUMAN_TUMORS = [
    "PleuralEffusion", "RU1065", "RU1066", "RU1080", "RU1108", "RU1124", "RU1144", "RU1145",
    "RU1152", "RU1181", "RU1195", "RU1215", "RU1229", "RU1231", "RU1293", "RU1311", "RU426", "RU779",
]


def gather_all_paths():
    paths = {}
    for a in ALLOGRAFTS:
        paths[f"allograft_{a}"] = f"{DIR_PREFIX}/data/allografts/adata_{a}_allografts_v3_RORA_RORB_ave.csv"
    for h in HUMAN_TUMORS:
        paths[f"human_{h}"] = f"{DIR_PREFIX}/data/human_tumor_MSK/adata_{h}_v3_RORA_RORB_ave.csv"
    paths["organoid_combined"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_v3_RORA_RORB_ave.csv"
    paths["organoid_shGFP"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB1"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB2"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv"
    paths["mets_compiled"] = f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv"
    paths.update(IRELAND_PATHS)
    return paths


def gather_baseline(name):
    if name.startswith("allograft_"):
        c = f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"
    elif name.startswith("human_"):
        c = f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv"
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


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/{BRCD}/rules/rules_{BRCD}.txt")

    paths = gather_all_paths()
    rows = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"\n=== {name} ===")
        data_test = bb.load.load_data(path, nodes, norm=NORM, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        VAL_DIR = f"{DIR_PREFIX}/{BRCD}/validation/external_validation/{name}"
        os.makedirs(VAL_DIR, exist_ok=True)

        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
            save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=name,
        )
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
        stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)
        stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)

        baseline_r2, baseline_auc = gather_baseline(name)
        row = {
            "name": name, "baseline_r2": baseline_r2, "r2_7777": stats["r2"].mean(),
            "baseline_auc": baseline_auc, "auc_7777": stats["roc_auc_score"].mean(),
        }
        rows.append(row)
        print(f"{name}: R2 {baseline_r2:.3f} -> {row['r2_7777']:.3f}, AUC {baseline_auc:.3f} -> {row['auc_7777']:.3f}")

        pd.DataFrame(rows).to_csv(
            f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/7777_vs_6667_full_validation.csv", index=False,
        )  # write incrementally so partial progress is visible/recoverable

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n=== Full comparison ===")
    print(result.to_string(index=False))
    valid = result.dropna(subset=["baseline_r2"])
    print(f"\nMean baseline R2: {valid['baseline_r2'].mean():.4f}  Mean 7777 R2: {valid['r2_7777'].mean():.4f}")
    print(f"Mean baseline AUC: {valid['baseline_auc'].mean():.4f}  Mean 7777 AUC: {valid['auc_7777'].mean():.4f}")
    print(f"Datasets improved (R2): {(valid['r2_7777'] > valid['baseline_r2']).sum()}/{len(valid)}")
    print(f"Datasets improved (AUC): {(valid['auc_7777'] > valid['baseline_auc']).sum()}/{len(valid)}")


if __name__ == "__main__":
    main()
