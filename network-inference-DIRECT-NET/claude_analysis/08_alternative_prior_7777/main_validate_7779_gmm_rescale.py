"""Barcode 7779: full R2/AUC validation of the split-by-failure-type adaptive rescaling --
rescale mean-shift-only genes (location+scale correction toward GEMM's reference,
adaptive_rescale_v2.rescale_mean_shift_gene), and (REVISED, see adaptive_rescale_v2.py's
module docstring for the full story) recenter-only (no scale) variance-collapse genes
against GEMM's same fixed window (adaptive_rescale_v2.rescale_collapse_gene). Built on
6667's ORIGINAL rules. Differs from 7778 in exactly one respect: how the variance_collapse
branch is handled (recenter-only rescale here; marginalize in 7778).

An earlier version of the variance_collapse branch used GEMM's reference GMM to force a
discrete ON/OFF pole assignment, then excluded those genes as scoring targets because that
forcing made their "actual" value fabricated rather than real. The revised branch doesn't
force an outcome, so that exclusion is gone too: every sample scores the same 53 genes as
6667's baseline. If a gene is genuinely single-class in some sample, get_sklearn_metrics_fixed
already catches the resulting ValueError from sklearn's roc_auc_score and reports NaN --
correctly excluded from the AUC mean by pandas' default skipna behavior, not from the
gene's analysis.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/main_validate_7779_gmm_rescale.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_rescale_v2 import classify_genes, build_adaptive_data

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
BRCD_OUT = "7779"
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
    """No gene-subset restriction needed -- 7779 scores the same full 53-gene set as
    6667's baseline now (the revised variance_collapse branch no longer forces an
    artificial ground truth, so there's no reason to exclude those genes as targets)."""
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
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    paths = gather_all_paths()
    rows = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"\n=== {name} ===")
        raw = pd.read_csv(path, index_col=0)
        raw.columns = [c.upper() for c in raw.columns]
        normed = bb.load.load_data(path, nodes, norm=NORM, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        classification = classify_genes(raw, nodes)
        n_mean_shift = sum(1 for v in classification.values() if v == "mean_shift")
        n_collapse = sum(1 for v in classification.values() if v == "variance_collapse")
        print(f"{n_mean_shift} mean_shift (rescaled), {n_collapse} variance_collapse (GMM-pole)")

        data_test, _ = build_adaptive_data(raw, normed, nodes, classification=classification)

        VAL_DIR = f"{DIR_PREFIX}/{BRCD_OUT}/validation/external_validation/{name}"
        os.makedirs(VAL_DIR, exist_ok=True)

        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
            save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=name,
        )
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
        stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)
        stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)

        n_nan_auc = stats["roc_auc_score"].isna().sum()
        if n_nan_auc:
            print(f"  {n_nan_auc} genes have undefined AUC (genuinely single-class ground truth in this sample) -- excluded from the AUC mean only")

        baseline_r2, baseline_auc = gather_baseline(name)
        row = {
            "name": name, "n_mean_shift_rescaled": n_mean_shift, "n_collapse_gmm_pole": n_collapse,
            "baseline_r2": baseline_r2, "r2_7779": stats["r2"].mean(),
            "baseline_auc": baseline_auc, "auc_7779": stats["roc_auc_score"].mean(),
        }
        rows.append(row)
        print(f"{name}: R2 {baseline_r2:.3f} -> {row['r2_7779']:.3f}, AUC {baseline_auc:.3f} -> {row['auc_7779']:.3f}")

        pd.DataFrame(rows).to_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/7779_vs_6667_full_validation.csv", index=False)

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n=== Full comparison ===")
    print(result.to_string(index=False))
    valid = result.dropna(subset=["baseline_r2"])
    print(f"\nMean baseline R2: {valid['baseline_r2'].mean():.4f}  Mean 7779 R2: {valid['r2_7779'].mean():.4f}")
    print(f"Mean baseline AUC: {valid['baseline_auc'].mean():.4f}  Mean 7779 AUC: {valid['auc_7779'].mean():.4f}")
    print(f"Datasets improved (R2): {(valid['r2_7779'] > valid['baseline_r2']).sum()}/{len(valid)}")
    print(f"Datasets improved (AUC): {(valid['auc_7779'] > valid['baseline_auc']).sum()}/{len(valid)}")


if __name__ == "__main__":
    main()
