"""External validation of boba-T's fitted 6667 rules on the organoid and mets_compiled
datasets, same shape as main_external_validation.py's allograft loop: load the same
228-edge LASSO network, load the new external dataset (already restricted to the network
gene panel + RORA_RORB averaged, see preprocess_organoid.py / preprocess_mets_compiled.R),
load the already-fit 6667 rules, run fit_validation, score with get_sklearn_metrics.

organoid has a real experimental grouping worth splitting by: `condition`
(shGFP/shRORB1/shRORB2, a RORB-knockdown experiment) -- per-condition CSVs
(adata_organoid_<condition>_v3_RORA_RORB_ave.csv) are cut directly from the full
organoid export by condition and scored the same way, into separate VAL_DIRs. No
equivalent grouping variable was found for mets_compiled (checked obs columns, barcode
patterns, Idents(), Project(), @misc, @commands -- no sample/location field survived
whatever merge produced this compiled object), so it's only ever run as one group.

Run in bobaT_env_py3.13 (brcd/norm default to 6667/0.3, so a bare `<sample>` call is
identical to before; pass a second arg to score a different brcd's rules, e.g. the 6668
node_normalization=0.4 hyperparameter-test run -- norm here must match whatever
node_normalization that brcd's rules were fit with, since it controls how this script's
own load_data() binarizes the external data before scoring):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py organoid
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py mets_compiled
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py organoid_shGFP
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py organoid_shRORB1
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py organoid_shRORB2
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py organoid_shGFP 6668 0.4
"""

import os
import sys

import numpy as np

# bobaT/tl.py's roc() calls np.trapz, removed in this env's numpy (2.4.6) -- restored as
# an alias to np.trapezoid here rather than pinning numpy down (which breaks bobaT's own
# graph_tool/scipy import chain, which needs numpy>=2.0) or editing the bobaT package.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import bobaT as bb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "misc"))
from fixed_get_sklearn_metrics import get_sklearn_metrics_fixed

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)

DATA_PATHS = {
    "organoid": "data/organoid/adata_organoid_v3_RORA_RORB_ave.csv",
    "mets_compiled": "data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv",
    "organoid_shGFP": "data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv",
    "organoid_shRORB1": "data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv",
    "organoid_shRORB2": "data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv",
    # Ireland et al. 2025 Zenodo record 15857303 (see preprocess_zenodo_ireland2025.py)
    "cgrp_k5": "data/cgrp_k5/adata_cgrp_k5_v3_RORA_RORB_ave.csv",
    "cgrp_k5_CGRP": "data/cgrp_k5/adata_cgrp_k5_CGRP_v3_RORA_RORB_ave.csv",
    "cgrp_k5_K5": "data/cgrp_k5/adata_cgrp_k5_K5_v3_RORA_RORB_ave.csv",
    "organoid_celltag": "data/organoid_celltag/adata_organoid_celltag_v3_RORA_RORB_ave.csv",
    "organoid_celltag_RPM": "data/organoid_celltag/adata_organoid_celltag_RPM_v3_RORA_RORB_ave.csv",
    "organoid_celltag_RPMA": "data/organoid_celltag/adata_organoid_celltag_RPMA_v3_RORA_RORB_ave.csv",
    "organoid_celltag_WT": "data/organoid_celltag/adata_organoid_celltag_WT_v3_RORA_RORB_ave.csv",
    "tbo_allograft_5khvg": "data/tbo_allograft_5khvg/adata_tbo_allograft_5khvg_v3_RORA_RORB_ave.csv",
    "tbo_allograft_5khvg_RPM_CTpostCre": "data/tbo_allograft_5khvg/adata_tbo_allograft_5khvg_RPM_CTpostCre_v3_RORA_RORB_ave.csv",
    "tbo_allograft_5khvg_RPM_CTpreCre": "data/tbo_allograft_5khvg/adata_tbo_allograft_5khvg_RPM_CTpreCre_v3_RORA_RORB_ave.csv",
    "rpr2_allograft": "data/rpr2_allograft/adata_rpr2_allograft_v3_RORA_RORB_ave.csv",
    "rpr2_allograft_RPM": "data/rpr2_allograft/adata_rpr2_allograft_RPM_v3_RORA_RORB_ave.csv",
    "rpr2_allograft_RPR2": "data/rpr2_allograft/adata_rpr2_allograft_RPR2_v3_RORA_RORB_ave.csv",
    "celltag_fate_dpt": "data/celltag_fate_dpt/adata_celltag_fate_dpt_v3_RORA_RORB_ave.csv",
    "celltag_fate_dpt_RPM": "data/celltag_fate_dpt/adata_celltag_fate_dpt_RPM_v3_RORA_RORB_ave.csv",
    "celltag_fate_dpt_RPMA": "data/celltag_fate_dpt/adata_celltag_fate_dpt_RPMA_v3_RORA_RORB_ave.csv",
}


def main(sample: str, brcd: str = "6667", norm: float = 0.3):
    validation_fname = f"validation/external_validation/{sample}"

    graph, vertex_dict = bb.load.load_network(
        f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False,
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    print(f"Reading in {sample} data...")
    data_test_t0 = bb.load.load_data(
        f"{DIR_PREFIX}/{DATA_PATHS[sample]}", nodes,
        norm=norm, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    print(f"Reading in pre-generated rules for brcd {brcd}...")
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/{brcd}/rules/rules_{brcd}.txt")

    VAL_DIR = f"{DIR_PREFIX}/{brcd}/{validation_fname}"
    os.makedirs(VAL_DIR, exist_ok=True)

    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
        data_test_t0, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=True, show_plots=False, save_df=True, fname=sample,
    )
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)

    # get_sklearn_metrics_fixed instead of bb.tl.get_sklearn_metrics -- the installed
    # bobaT's gene-name parsing truncates at the first underscore, mislabeling RORA_RORB
    # as a nonexistent "RORA" (see claude_analysis/misc/fixed_get_sklearn_metrics.py).
    summary_stats = get_sklearn_metrics_fixed(VAL_DIR, nodes)
    summary_stats.to_csv(f"{VAL_DIR}/summary_stats_fixed.csv", index=False)
    print(f"\n{sample} (brcd {brcd}, norm {norm}): {len(summary_stats)} genes scored")
    print(summary_stats[["gene", "r2", "roc_auc_score", "f1"]].describe())
    print(f"-> {VAL_DIR}")


if __name__ == "__main__":
    sample_arg = sys.argv[1]
    brcd_arg = sys.argv[2] if len(sys.argv) > 2 else "6667"
    norm_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    main(sample_arg, brcd_arg, norm_arg)
