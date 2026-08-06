"""Fit boba-T on the BoolODE-simulated HSC data using a ChEA-derived candidate network
instead of the literal ground-truth edges (see hsc_chea_candidate_network.py, and
benchmarking/README.md's "Combinatorial regulatory logic" section). Unlike
comparison_hsc_fit_bobat.py, the candidate network here is real, independently-sourced,
and partly wrong/incomplete relative to this specific Boolean model -- so, unlike that
script, regulator-set recovery here is a genuine test, not guaranteed by construction.

threshold=0, matching the real 6667 network's own convention (main_all_data_remove_
selfloops_6667.py: `node_threshold = 0  # don't remove any parents`) rather than bobaT's
package default of 0.1 -- this project never actually uses bobaT's own irrelevant-regulator
pruning step in its real runs, so Track 3 stays consistent with that rather than
introducing pruning behavior that isn't used anywhere else in the project. This means
Track 3's "does boba-T discover structure" test is really "does boba-T discover structure
THE WAY THIS PROJECT ACTUALLY RUNS IT" (no internal pruning) -- any regulator-set recovery
has to come from the candidate network itself, not from bobaT's own relevance filtering.

Run in bobaT_env:
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_fit_bobat_chea.py
"""

import os

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import bobaT as bb

GT_DIR = "data/hsc_ground_truth"
BRCD = "hsc_chea"
DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/benchmarking"


def main():
    graph, vertex_dict = bb.load.load_network(
        f"{GT_DIR}/candidate_network_chea.csv", remove_sinks=False, remove_selfloops=False, remove_sources=False,
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
        data_t0, None, clusters, f"{DIR_PREFIX}/{BRCD}/data_split", suffix=BRCD,
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
        save=True, save_dir=VAL_DIR, plot=False, show_plots=False, save_df=True, fname=BRCD,
    )
    summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
    print(summary_stats[["gene", "r2", "roc_auc_score", "f1"]])

    print("\nFitted rules (regulators_dict, per gene) vs. how many ChEA candidates each started with:")
    for gene, regs in regulators_dict.items():
        print(f"  {gene}: fitted regulators = {regs}")


if __name__ == "__main__":
    main()
