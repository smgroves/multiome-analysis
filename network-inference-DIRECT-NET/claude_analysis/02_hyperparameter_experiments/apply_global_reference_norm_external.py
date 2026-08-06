"""Score an external validation sample against 6670's rules (fit under the
global-reference normalization prototype) using GEMM's OWN fitted reference (gene
medians + shared scale, saved by build_global_reference_norm.py) -- NOT a freshly
recomputed reference from the external sample itself.

This is the direct answer to "is there a normalization setting that accounts for the
diversity<->R2 relationship in diagnose_sample_diversity.py": since this scheme's [0,1]
squash is referenced to GEMM's own scale rather than each dataset's own quantiles, a
low-diversity external sample's raw values simply won't deviate much from GEMM's gene
medians in absolute terms, so it will correctly land near 0.5 (appropriately uncertain)
rather than being artificially stretched to fill [0,1] the way node_normalization=0.3
forces every dataset to ~60% exact-0/1 regardless of its own true diversity. Whether that
raises R2 (rather than just making wrong predictions less confidently wrong) is an
empirical question this script tests directly.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/02_hyperparameter_experiments/apply_global_reference_norm_external.py <sample>
    e.g. organoid_shGFP, organoid_shRORB1, organoid_shRORB2, organoid, mets_compiled
"""

import os
import sys

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
BRCD = "6670"

DATA_PATHS = {
    "organoid": "data/organoid/adata_organoid_v3_RORA_RORB_ave.csv",
    "mets_compiled": "data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv",
    "organoid_shGFP": "data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv",
    "organoid_shRORB1": "data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv",
    "organoid_shRORB2": "data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv",
}


def main(sample: str):
    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False,
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    raw = bb.load.load_data(
        f"{DIR_PREFIX}/{DATA_PATHS[sample]}", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    params = pd.read_csv(f"{DIR_PREFIX}/data/global_reference_norm_params_gemm.csv", index_col=0)
    gene_medians = params["gene_median"]
    shared_scale = params["shared_scale"].iloc[0]
    K = params["K"].iloc[0]
    print(f"Applying GEMM's reference (shared_scale={shared_scale:.4f}, K={K}) to {sample}'s raw data...")

    centered = raw - gene_medians
    scaled = 1.0 / (1.0 + np.exp(-centered / (K * shared_scale)))
    frac_extreme = ((scaled > 0.95) | (scaled < 0.05)).mean().mean()
    print(f"{sample}: mean fraction confidently-binarized under GEMM-referenced scaling: {frac_extreme:.4f}")

    tmp_path = f"{DIR_PREFIX}/data/_tmp_{sample}_globalnorm_gemmref.csv"
    scaled.index.name = "CellID"
    scaled.to_csv(tmp_path)

    data_test_t0 = bb.load.load_data(
        tmp_path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    os.remove(tmp_path)

    print("Reading in pre-generated rules for brcd 6670...")
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/{BRCD}/rules/rules_{BRCD}.txt")

    VAL_DIR = f"{DIR_PREFIX}/{BRCD}/validation/external_validation/{sample}"
    os.makedirs(VAL_DIR, exist_ok=True)

    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
        data_test_t0, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=True, show_plots=False, save_df=True, fname=sample,
    )
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)

    summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
    print(f"\n{sample} vs 6670 (GEMM-referenced global norm): {len(summary_stats)} genes scored")
    print(f"Mean r2: {summary_stats['r2'].mean():.4f}  Mean roc_auc: {summary_stats['roc_auc_score'].mean():.4f}")
    print(f"-> {VAL_DIR}")


if __name__ == "__main__":
    main(sys.argv[1])
