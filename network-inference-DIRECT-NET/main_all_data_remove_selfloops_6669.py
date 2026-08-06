# Phase 2, item 2 hyperparameter test run: identical to main_all_data_remove_selfloops_6667
# copy.py (node_normalization=0.3, same as 6667 -- NOT the 6668 norm sweep, kept separate so
# this test isolates the regulator-cap variable) except using the regulator-capped candidate
# network (build_capped_network.py: caps every gene at 6 regulators max, dropping the
# lowest-fitted-relevance regulator(s) per over-cap gene per 6667's own strengths.csv,
# affecting the 10 genes that had 7-8 regulators -- see BoBa-T_hyperparameters.md sec 3 for
# the sizing heuristic). Does not touch or overwrite anything under 6667/ or 6668/.
import random
import time

import numpy as np
import seaborn as sns
import bobaT as bb
import os
import os.path as op
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from graph_tool import all as gt
from graph_tool import GraphView
from bb_utils import *
import sys
from datetime import timedelta
import glob
import json
import platform, multiprocessing

if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")
    customPalette = sns.color_palette("tab10")

    # =============================================================================
    # Set variables and csvs
    # =============================================================================
    print_graph_information = True
    plot_network = True
    split_train_test = True
    write_binarized_data = False
    fit_rules = True
    run_validation = True
    validation_averages = True
    find_average_states = False
    find_attractors = False
    tf_basin = 2
    filter_attractors = False
    perturbations = False
    stability = False
    walk_to_basin = False
    plot_walk_to_basin = False
    on_nodes = []
    off_nodes = []

    ## Set variables for computation
    remove_sinks = False
    remove_selfloops = True
    remove_sources = False

    node_normalization = 0.3  # same as 6667; the capped network is the only variable changed here
    node_threshold = 0  # don't remove any parents (capping already handled regulator count)
    transpose = True

    validation_fname = "validation/in_sample_validation/"
    fname = "combined"
    notes_for_log = (
        "6669: same node_normalization/data/split as 6667, but using the regulator-capped "
        "candidate network (cap=6, build_capped_network.py) instead of the original "
        "228-edge network, to test whether capping truth-table size on the 10 "
        "previously-over-cap genes improves fit robustness."
    )
    ## Set paths
    dir_prefix = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
    network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined_cap6.csv"
    data_path = "/data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
    t1 = False
    data_t1_path = None

    ## Set metadata information
    cellID_table = "data/AA_clusters_splitgen.csv"
    cluster_header_list = ["class"]

    ## Set brcd
    brcd = str(6669)
    print(brcd)
    data_train_t0_path = f"{brcd}/data_split/train_t0{fname}.csv"
    data_train_t1_path = None
    data_test_t0_path = f"{brcd}/data_split/test_t0{fname}.csv"
    data_test_t1_path = None

    job_brcd = str(random.randint(0, 99999))
    print(f"Job barcode: {job_brcd}")

    random_state = 1234

    #########################################

    # =============================================================================
    # Start timer and check paths
    # =============================================================================

    if not os.path.exists(f"{dir_prefix}/{brcd}"):
        os.makedirs(f"{dir_prefix}/{brcd}")

    time1 = time.time()

    if dir_prefix[-1] != os.sep:
        dir_prefix = dir_prefix + os.sep
    if not network_path.endswith(".csv") or not os.path.isfile(
        dir_prefix + network_path
    ):
        raise Exception(
            "Network path must be a .csv file.  Check file name and location"
        )
    if not data_path.endswith(".csv") or not os.path.isfile(dir_prefix + data_path):
        raise Exception("data path must be a .csv file.  Check file name and location")
    if cellID_table is not None:
        if not cellID_table.endswith(".csv") or not os.path.isfile(
            dir_prefix + cellID_table
        ):
            raise Exception(
                "CellID path must be a .csv file.  Check file name and location"
            )

    # =============================================================================
    # Load the network
    # =============================================================================
    graph, vertex_dict = bb.load.load_network(
        f"{dir_prefix}/{network_path}",
        remove_sinks=remove_sinks,
        remove_selfloops=remove_selfloops,
        remove_sources=remove_sources,
    )

    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    if print_graph_information:
        print_graph_info(
            graph,
            vertex_dict,
            nodes,
            fname,
            brcd=brcd,
            dir_prefix=dir_prefix,
            plot=True,
            add_edge_weights=False,
        )

    # =============================================================================
    # Load the data and clusters
    # =============================================================================
    print("Reading in data")
    print(f"{dir_prefix}/{data_path}")
    data_t0 = bb.load.load_data(
        f"{dir_prefix}/{data_path}",
        nodes,
        norm=node_normalization,
        delimiter=",",
        log1p=False,
        transpose=transpose,
        sample_order=False,
        fillna=0,
    )
    data_t1 = None

    clusters = bb.utils.get_clusters(
        data_t0,
        cellID_table=f"{dir_prefix}/{cellID_table}",
        cluster_header_list=cluster_header_list,
    )

    if not os.path.exists(f"{dir_prefix}/{brcd}/data_split"):
        os.makedirs(f"{dir_prefix}/{brcd}/data_split")

    (
        data_train_t0,
        data_test_t0,
        data_train_t1,
        data_test_t1,
        clusters_train,
        clusters_test,
    ) = bb.utils.split_train_test(
        data_t0, data_t1, clusters, f"{dir_prefix}/{brcd}/data_split", suffix=fname
    )

    # =============================================================================
    # Read in binarized data
    # =============================================================================
    print("Binarizing data")
    if write_binarized_data:
        save = True
    else:
        save = False
    if not os.path.exists(f"{dir_prefix}/{brcd}/binarized_data"):
        os.makedirs(f"{dir_prefix}/{brcd}/binarized_data")

    binarized_data_t0 = bb.proc.binarize_data(
        data_t0,
        phenotype_labels=clusters,
        save=save,
        save_dir=f"{dir_prefix}/{brcd}/binarized_data",
        fname=f"binarized_data_t0_{fname}",
    )

    binarized_data_train_t0 = bb.proc.binarize_data(
        data_train_t0,
        phenotype_labels=clusters,
        save=save,
        save_dir=f"{dir_prefix}/{brcd}/binarized_data",
        fname=f"binarized_data_train_t0_{fname}",
    )
    binarized_data_train_t1 = None

    print("Binarizing test data")
    binarized_data_test = bb.proc.binarize_data(
        data_test_t0,
        phenotype_labels=clusters,
        save=save,
        save_dir=f"{dir_prefix}/{brcd}/binarized_data",
        fname=f"binarized_data_test_t0_{fname}",
    )
    binarized_data_test_t1 = None

    # =============================================================================
    # Fit rules with the training dataset
    # =============================================================================
    if fit_rules:
        if not os.path.exists(f"{dir_prefix}/{brcd}/rules"):
            os.makedirs(f"{dir_prefix}/{brcd}/rules")
        print("Running classic BooleaBayes rule fitting with a single timepoint...")
        rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
            data=data_train_t0,
            vertex_dict=vertex_dict,
            plot=False,
            threshold=node_threshold,
        )
        bb.tl.save_rules(
            rules, regulators_dict, fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt"
        )
        strengths.to_csv(f"{dir_prefix}/{brcd}/rules/strengths.csv")
        signed_strengths.to_csv(f"{dir_prefix}/{brcd}/rules/signed_strengths.csv")
        draw_grn(
            graph,
            vertex_dict,
            rules,
            regulators_dict,
            f"{dir_prefix}/{brcd}/{fname}_network.pdf",
            save_edge_weights=True,
            edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv",
        )
    else:
        try:
            print("Reading in pre-generated rules...")
            rules, regulators_dict = bb.load.load_rules(
                fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt"
            )
        except FileNotFoundError:
            print(
                "Rules file not found. Please set fit_rules to True to generate rules."
            )

    # =============================================================================
    # Calculate AUC for test dataset for a true error calculation
    # =============================================================================

    if run_validation:
        print("Running validation step...")
        VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"
        os.makedirs(VAL_DIR, exist_ok=True)

        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test_t0,
            data_test_t1=None,
            nodes=nodes,
            regulators_dict=regulators_dict,
            rules=rules,
            save=True,
            save_dir=VAL_DIR,
            plot=True,
            show_plots=False,
            save_df=True,
            fname=fname,
        )
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)
    else:
        print("Skipping validation step...")

    if validation_averages:
        print("Calculating validation averages...")
        VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"

        if run_validation == False:
            tprs_all, fprs_all, area_all = bb.tl.roc_from_file(
                f"{VAL_DIR}/accuracy_plots", nodes, save=True, save_dir=VAL_DIR
            )

        aucs = pd.read_csv(f"{VAL_DIR}/aucs.csv", header=None, index_col=0)
        print("AUC means: ", aucs.mean())

        bb.plot.plot_aucs(VAL_DIR, save=True, show_plot=False)

        bb.plot.plot_validation_avgs(
            fprs_all,
            tprs_all,
            len(nodes),
            area_all,
            save=True,
            save_dir=VAL_DIR,
            show_plot=False,
            vertex_dict=vertex_dict, graph=graph, remove_sources=True
        )

        summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
        bb.plot.plot_sklearn_metrics(VAL_DIR)
        bb.plot.plot_sklearn_summ_stats(
            summary_stats.drop("max_error", axis=1), VAL_DIR, fname=""
        )
    else:
        print("Skipping validation averaging...")

    time2 = time.time()
    time_for_job = (time2 - time1) / 60.0
    print("Time for job: ", time_for_job, " minutes")

    log_job(
        dir_prefix,
        brcd,
        random_state,
        network_path,
        data_path,
        data_t1_path,
        cellID_table,
        node_normalization,
        node_threshold,
        split_train_test,
        write_binarized_data,
        fit_rules,
        run_validation,
        validation_averages,
        find_average_states,
        find_attractors,
        tf_basin,
        filter_attractors,
        on_nodes,
        off_nodes,
        perturbations,
        stability,
        time=time_for_job,
        job_barcode=job_brcd,
        notes_for_job=notes_for_log,
    )
