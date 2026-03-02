# uses network based off networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks.csv (6666) except with RORA/RORB combined into one node.
import random
import time

import numpy as np
import seaborn as sns
import booleabayes as bb
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
    # To modulate which parts of the pipeline need to be computed, use the following variables
    # =============================================================================
    print_graph_information = True  # whether to print graph info to {brcd}.txt
    plot_network = True
    split_train_test = False
    write_binarized_data = False
    fit_rules = False
    run_validation = False
    validation_averages = False
    find_average_states = False
    find_attractors = False
    tf_basin = 2  # if -1, use average distance between clusters for search basin for attractors.
    # otherwise use the same size basin for all phenotypes. For single cell data, there may be so many samples that average distance is small.
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

    node_normalization = 0.3
    node_threshold = 0  # don't remove any parents
    transpose = True

    # sample = sys.argv[1]

    validation_fname = "validation/"
    # fname = f"{sample}"
    fname = "combined"
    notes_for_log = "Validation and attractor finding for updated DIRECT-NET network with 2020db and indpendent LASSO models, wo sinks, RORA/RORB combined"
    ## Set paths
    dir_prefix = "/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
    network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
    data_path = "/data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
    t1 = False
    data_t1_path = None  # if no T1 (i.e. single dataset), replace with None

    ## Set metadata information
    cellID_table = "data/AA_clusters_splitgen.csv"
    # cellID_table = f'data/human_tumors/{sample}_clusters.csv'
    # Assign headers to cluster csv, with one called "class"
    # cluster_header_list = ['class']

    # cluster headers with "identity" replaced with "class"
    cluster_header_list = ["class"]

    # the below headers go with metadata_final
    # cluster_header_list = ["orig.ident","nCount_RNA","nFeature_RNA","nCount_ATAC","nFeature_ATAC","nucleosome_signal",
    #                        "nucleosome_percentile","TSS.enrichment","TSS.percentile","barcode","sample","ATAC_snn_res.0.5",
    #                        "seurat_clusters","nCount_peaks","nFeature_peaks","peaks_snn_res.0.5","percent.mt","nCount_SCT",
    #                        "nFeature_SCT","SCT_snn_res.0.5","SCT.weight","peaks.weight","nCount_Imputed_counts",
    #                        "nFeature_Imputed_counts","nCount_gene_activity","nFeature_gene_activity","NE_score1",
    #                        "class","non.NE_score1","comb.score","S.Score","G2M.Score","Phase","old.ident","wsnn_res.0.5"
    #                        ]

    ## Set brcd and train/test data if rerun
    brcd = str(6667)  # correspond to LASSO network with RORA/RORB combined
    print(brcd)
    # if rerunning a brcd and data has already been split into training and testing sets, use the below code
    # Otherwise, these settings are ignored
    data_train_t0_path = f"{brcd}/data_split/train_t0{fname}.csv"
    data_train_t1_path = None  # if no T1, replace with None
    data_test_t0_path = f"{brcd}/data_split/test_t0{fname}.csv"
    data_test_t1_path = None  # if no T1, replace with None

 

    job_brcd = str(
        random.randint(0, 99999)
    )  # use a job brcd to keep track of multiple jobs for the same brcd
    print(f"Job barcode: {job_brcd}")

    random_state = 1234

    # Append the results to a MasterResults file

    #########################################

    # =============================================================================
    # Start timer and check paths
    # =============================================================================

    if not os.path.exists(f"{dir_prefix}/{brcd}"):
        # Create a new directory because it does not exist
        os.makedirs(f"{dir_prefix}/{brcd}")

    # sys.stdout = open(f'{dir_prefix}/{brcd}/jobs/{job_brcd}_log.txt','wt')

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
    if t1 == True:
        if split_train_test == True:
            if data_t1_path is None:
                raise Exception("t1 is set to True, but no data_t1_path given.")
        else:
            if data_train_t1_path is None or data_test_t1_path is None:
                raise Exception(
                    "t1 is set to True, but no data_[train/test]_t1_path is given."
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
    if split_train_test:
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
        if data_t1_path is not None:
            data_t1 = bb.load.load_data(
                f"{dir_prefix}/{data_t1_path}",
                nodes,
                norm=node_normalization,
                delimiter=",",
                log1p=False,
                transpose=transpose,
                sample_order=False,
                fillna=0,
            )
        else:
            data_t1 = None

        # Only need to pass 'data_t0' since this data is not split into train/test
        # TODO: change the below code so that you can input which column should be
        # replaced with "class" instead of full cluster_header_list
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
        # random_state = random_state)
    else:  # load the data
        data_train_t0 = bb.load.load_data(
            f"{dir_prefix}/{data_train_t0_path}",
            nodes,
            norm=node_normalization,
            delimiter=",",
            log1p=False,
            transpose=True,
            sample_order=False,
            fillna=0,
        )
        if t1:
            data_train_t1 = bb.load.load_data(
                f"{dir_prefix}/{data_train_t1_path}",
                nodes,
                norm=node_normalization,
                delimiter=",",
                log1p=False,
                transpose=True,
                sample_order=False,
                fillna=0,
            )
        else:
            data_train_t1 = None

        data_test_t0 = bb.load.load_data(
            f"{dir_prefix}/{data_test_t0_path}",
            nodes,
            norm=node_normalization,
            delimiter=",",
            log1p=False,
            transpose=True,
            sample_order=False,
            fillna=0,
        )

        if t1:
            data_test_t1 = bb.load.load_data(
                f"{dir_prefix}/{data_test_t1_path}",
                nodes,
                norm=node_normalization,
                delimiter=",",
                log1p=False,
                transpose=True,
                sample_order=False,
                fillna=0,
            )
        else:
            data_test_t1 = None

        clusters = bb.utils.get_clusters(
            data_train_t0,
            data_test=data_test_t0,
            is_data_split=True,
            cellID_table=f"{dir_prefix}/{cellID_table}",
            cluster_header_list=cluster_header_list,
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
        # Create a new directory because it does not exist
        os.makedirs(f"{dir_prefix}/{brcd}/binarized_data")

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
    if t1:
        binarized_data_train_t1 = bb.proc.binarize_data(
            data_train_t1,
            phenotype_labels=clusters,
            save=save,
            save_dir=f"{dir_prefix}/{brcd}/binarized_data",
            fname=f"binarized_data_train_t1_{fname}",
        )
    else:
        binarized_data_train_t1 = None

    print("Binarizing test data")
    binarized_data_test = bb.proc.binarize_data(
        data_test_t0,
        phenotype_labels=clusters,
        save=save,
        save_dir=f"{dir_prefix}/{brcd}/binarized_data",
        fname=f"binarized_data_test_t0_{fname}",
    )

    if t1:
        binarized_data_test_t1 = bb.proc.binarize_data(
            data_test_t1,
            phenotype_labels=clusters,
            save=save,
            save_dir=f"{dir_prefix}/{brcd}/binarized_data",
            fname=f"binarized_data_test_t1_{fname}",
        )
    else:
        binarized_data_test_t1 = None

    # =============================================================================
    # Re-fit rules with the training dataset
    # =============================================================================
    if fit_rules:
        if not os.path.exists(f"{dir_prefix}/{brcd}/rules"):
            # Create a new directory because it does not exist
            os.makedirs(f"{dir_prefix}/{brcd}/rules")
        if t1:
            print("Running time-series-adapted BooleaBayes rule fitting...")
            rules, regulators_dict, strengths, signed_strengths = (
                bb.tl.get_rules_scvelo(
                    data=data_train_t0,
                    data_t1=data_train_t1,
                    vertex_dict=vertex_dict,
                    plot=False,
                    threshold=node_threshold,
                )
            )
        else:
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
        )  # , gene2color = gene2color)
    else:
        try:
            print("Reading in pre-generated rules...")
            rules, regulators_dict = bb.load.load_rules(
                fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt"
            )
            genegroups = {1:["ASCL1", "MEIS2", "NFATC2", "PROX1", "LMX1B", "RORA_RORB","ICAM1","CUX2","KMT2A","TBX15","ZBTB20","PKNOX2","SOX11","CREB1","SIX1","EGR1","NCAM1","TFDP1","PBX1","ESR1","NFIB","BACH1","ZEB1","TCF4","HSF2","ESR1","SIX4","NR6A1","ETS1","EPCAM"],
            2:["JUN", "FOS", "JUND", "GRHL2", "NFIA", "EHF", "THRB", "JUNB","HES1","NFYC", "FOXO3", "SOX9", "SMAD3", "STAT2", "TCF7L1", "TCF7L2", "RUNX1", "NFIA", "STAT1", "NFIX", "FOXO3", "NFYC", "REST", "BACH2", "RBPJ", "EHF", "THRB"]}
            #flip genegroups dict to gene2group dict for use in plotting
            gene2group = {}
            gene2color = {}
            for group in genegroups:
                for gene in genegroups[group]:
                    gene2group[gene] = group
                    if group == 1:
                        gene2color[gene] = customPalette[0]
                    elif group == 2:
                        gene2color[gene] = customPalette[1]
            draw_grn_alt(
                    graph,
                    vertex_dict,
                    rules,
                    regulators_dict,
                    f"{dir_prefix}/{brcd}/{fname}_network.pdf",
                    save_edge_weights=True,
                    edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv",
                    gene2group=gene2group,
                    gene2color=gene2color,
                    mu =1.5,
                    C = 0.5,
                    k = 1,
                    p = 2,
                    gamma = 1
                ) 
            
                        # mu=mu,        # increase from 0.5 — this is the repulsive force strength
        # eweight=edge_weights,
        # max_iter=2000,
        # C=C,         # edge length constant — increase to spread connected nodes apart
        # K=K,         # preferred edge length — try 0.5–2.0
        # p=p,         # repulsion exponent — higher = stronger short-range repulsion
        # gamma=gamma,     # cooling schedule — lower = more careful optimization
        except FileNotFoundError:
            print(
                "Rules file not found. Please set fit_rules to True to generate rules."
            )


for mu in [1.5, 3.0]:
    for C in [0.5, 1.0]:
        for K in [1.0, 2.0]:
            for p in [2, 3]:
                for gamma in [1, 0.1]:
                    draw_grn_alt(graph,
                                vertex_dict,
                                rules,
                                regulators_dict,
                                f"{dir_prefix}/{brcd}/{fname}_network_{mu}_{C}_{K}_{p}_{gamma}.pdf",
                                save_edge_weights=False,
                                gene2group=gene2group,
                                gene2color=gene2color,
                                mu = mu,
                                C = C,
                                K = K,
                                p = p,
                                gamma = gamma
                            ) 