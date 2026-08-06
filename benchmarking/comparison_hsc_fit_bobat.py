"""Fit boba-T on the BoolODE-simulated HSC data (see benchmarking/README.md, "Combinatorial
regulatory logic: the HSC ground truth"). Candidate network + BoolODE-simulated expression
come from hsc_ground_truth.py / prepare_hsc_bobat_input.py.

Run in bobaT_env:
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_fit_bobat.py
"""

import os

import numpy as np

if not hasattr(np, "trapz"):  # see main_external_validation_organoid_mets.py's note
    np.trapz = np.trapezoid

import bobaT as bb

GT_DIR = "data/hsc_ground_truth"
BRCD = "hsc"
DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/benchmarking"


def main():
    graph, vertex_dict = bb.load.load_network(
        f"{GT_DIR}/candidate_network.csv", remove_sinks=False, remove_selfloops=False, remove_sources=False,
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    print(f"{len(nodes)} nodes: {sorted(nodes)}")

    data_t0 = bb.load.load_data(
        f"{GT_DIR}/expr_bobat.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    clusters = bb.utils.get_clusters(
        data_t0, cellID_table=f"{GT_DIR}/clusters_bobat.csv", cluster_header_list=["class"],
    )

    os.makedirs(f"{DIR_PREFIX}/{BRCD}/data_split", exist_ok=True)
    (data_train_t0, data_test_t0, _, _, clusters_train, clusters_test) = bb.utils.split_train_test(
        data_t0, None, clusters, f"{DIR_PREFIX}/{BRCD}/data_split", suffix="hsc",
    )

    rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
        data=data_train_t0, vertex_dict=vertex_dict, plot=False, threshold=0,
    )
    os.makedirs(f"{DIR_PREFIX}/{BRCD}/rules", exist_ok=True)
    bb.tl.save_rules(rules, regulators_dict, fname=f"{DIR_PREFIX}/{BRCD}/rules/rules_{BRCD}.txt")
    signed_strengths.to_csv(f"{DIR_PREFIX}/{BRCD}/rules/signed_strengths.csv")
    strengths.to_csv(f"{DIR_PREFIX}/{BRCD}/rules/strengths.csv")

    VAL_DIR = f"{DIR_PREFIX}/{BRCD}/validation"
    os.makedirs(VAL_DIR, exist_ok=True)
    bb.tl.fit_validation(
        data_test_t0, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname="hsc",
    )
    summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
    print(summary_stats[["gene", "r2", "roc_auc_score", "f1"]])

    print("\nFitted rules (regulators_dict, per gene):")
    for gene, regs in regulators_dict.items():
        print(f"  {gene}: regulators = {regs}")


if __name__ == "__main__":
    main()
