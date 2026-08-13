"""Barcode 7778: full R2/AUC validation of the hybrid adaptive approach -- rescale
mean-shift-only genes (location+scale correction toward GEMM's reference,
adaptive_rescale_v2.rescale_mean_shift_gene), MARGINALIZE variance-collapse genes
(claude_analysis/06_marginal_rule_validation/marginalize_rules.py's "treat as missing"
convention). Built on 6667's ORIGINAL rules. Differs from 7779 in exactly one respect: how
the variance_collapse branch is handled (marginalize here; GMM-pole in 7779) -- confirmed
via the fast leaf-conditional metric to be the better of the two (decisive_test_7778_fast.py:
223 improved/147 worsened, mean binary_diff +0.0105, mean resid_diff +0.0067 -- all better
than 7779's 209/201, +0.0050, -0.0030).

Since the variance_collapse branch changes which regulators each rule uses (marginalized
out), and per sample too, this can't be saved as a single static rules_7778.txt the way
7777 can -- the "barcode" here is the PROCEDURE (this script + 6667's base rules + GEMM's
reference stats + the flagging thresholds), reproducible on any new sample without any
dependence on this project's specific validation population.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/main_validate_7778_adaptive_marginalization.py
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
from adaptive_rescale_v2 import classify_genes
from adaptive_hybrid_7778 import build_hybrid_data_and_rules

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
BRCD_OUT = "7778"
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


def gather_baseline(name, nodes_subset):
    """Restricted to `nodes_subset` (the same genes 7778 actually scored for this sample)
    -- comparing 7778's mean over a smaller gene set against 6667's full-53-gene mean would
    be unfair, since the dropped genes are disproportionately the hard-to-score ones."""
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
    df = df[df["gene"].isin(nodes_subset)]
    return df["r2"].mean(), df["roc_auc_score"].mean()


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes_full = bb.utils.get_nodes(vertex_dict, graph)
    rules_full, regulators_dict_full = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    paths = gather_all_paths()
    rows = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"\n=== {name} ===")
        raw = pd.read_csv(path, index_col=0)
        raw.columns = [c.upper() for c in raw.columns]
        normed = bb.load.load_data(path, nodes_full, norm=NORM, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        classification = classify_genes(raw, nodes_full)
        n_mean_shift = sum(1 for v in classification.values() if v == "mean_shift")
        n_collapse = sum(1 for v in classification.values() if v == "variance_collapse")
        print(f"{n_mean_shift} mean_shift (rescaled), {n_collapse} variance_collapse (marginalized)")

        data_test, rules_adapted, regulators_dict_adapted, nodes_adapted = build_hybrid_data_and_rules(
            raw, normed, nodes_full, rules_full, regulators_dict_full, classification=classification,
        )
        n_flagged = n_mean_shift + n_collapse

        VAL_DIR = f"{DIR_PREFIX}/{BRCD_OUT}/validation/external_validation/{name}"
        os.makedirs(VAL_DIR, exist_ok=True)

        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test, data_test_t1=None, nodes=nodes_adapted, regulators_dict=regulators_dict_adapted, rules=rules_adapted,
            save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=name,
        )
        bb.tl.save_auc_by_gene(area_all, nodes_adapted, VAL_DIR)
        stats = get_sklearn_metrics_fixed(VAL_DIR, nodes_adapted)
        stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)

        baseline_r2, baseline_auc = gather_baseline(name, nodes_adapted)
        row = {
            "name": name, "n_mean_shift_rescaled": n_mean_shift, "n_collapse_marginalized": n_collapse,
            "n_genes_scored": len(nodes_adapted),
            "baseline_r2": baseline_r2, "r2_7778": stats["r2"].mean(),
            "baseline_auc": baseline_auc, "auc_7778": stats["roc_auc_score"].mean(),
        }
        rows.append(row)
        print(f"{name}: R2 {baseline_r2:.3f} -> {row['r2_7778']:.3f} ({n_flagged} genes flagged), "
              f"AUC {baseline_auc:.3f} -> {row['auc_7778']:.3f}")

        pd.DataFrame(rows).to_csv(f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/7778_vs_6667_full_validation.csv", index=False)

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n=== Full comparison ===")
    print(result.to_string(index=False))
    valid = result.dropna(subset=["baseline_r2"])
    print(f"\nMean baseline R2: {valid['baseline_r2'].mean():.4f}  Mean 7778 R2: {valid['r2_7778'].mean():.4f}")
    print(f"Mean baseline AUC: {valid['baseline_auc'].mean():.4f}  Mean 7778 AUC: {valid['auc_7778'].mean():.4f}")
    print(f"Datasets improved (R2): {(valid['r2_7778'] > valid['baseline_r2']).sum()}/{len(valid)}")
    print(f"Datasets improved (AUC): {(valid['auc_7778'] > valid['baseline_auc']).sum()}/{len(valid)}")


if __name__ == "__main__":
    main()
