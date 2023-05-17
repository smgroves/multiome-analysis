# This code is a copy of main_all_data.py, and is used to rerun the network rule fitting on the human tumor data.
# We will then compare the rules fit here to the actual rules on combined M1/M2 data to see which are similar.

# The main difference with this code is that we split the data using 5-fold cross-validation and run the rule fitting on
# each fold to generate an ensemble of rules to be compared to the original.

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

customPalette = sns.color_palette('tab10')

# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
print_graph_information = False #whether to print graph info to {brcd}.txt
plot_network = False
split_train_test = False
write_binarized_data = False
fit_rules = True
run_validation = True
validation_averages = True


## Set variables for computation
remove_sinks=False
remove_selfloops=False
remove_sources=False

node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True

# sample = sys.argv[1]

fname = f"human_tumors"
notes_for_log = "Fitting rules to human tumors for comparison"

## Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'
data_path = f'data/adata_human_tumors_MSK.csv'
t1 = False
data_t1_path = None #if no T1 (i.e. single dataset), replace with None

## Set metadata information
# cellID_table = 'data/AA_clusters.csv'
cellID_table = f'data/human_tumors_MSK_clusters.csv'
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
# brcd = str(random.randint(0,99999))
brcd = "9999-human-tumor"
print(brcd)

job_brcd = str(random.randint(0,99999)) #use a job brcd to keep track of multiple jobs for the same brcd
print(f"Job barcode: {job_brcd}")

# brcd =str(35468)
# random_state = str(random.Random.randint(0,99999)) #for train-test split
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
if not network_path.endswith('.csv') or not os.path.isfile(dir_prefix + network_path):
    raise Exception('Network path must be a .csv file.  Check file name and location')
if not data_path.endswith('.csv') or not os.path.isfile(dir_prefix + data_path):
    raise Exception('data path must be a .csv file.  Check file name and location')
if cellID_table is not None:
    if not cellID_table.endswith('.csv') or not os.path.isfile(dir_prefix + cellID_table):
        raise Exception('CellID path must be a .csv file.  Check file name and location')


# =============================================================================
# Load the network
# =============================================================================
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                              remove_sources=remove_sources)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

if print_graph_information:
    print_graph_info(graph, vertex_dict, nodes,  fname, brcd = brcd, dir_prefix = dir_prefix,plot = False)

# =============================================================================
# Load the data and clusters
# =============================================================================
print('Reading in data')


if split_train_test:
    data_t0 = bb.load.load_data(f'{dir_prefix}/{data_path}', nodes, norm=node_normalization,
                        delimiter=',', log1p=False, transpose=transpose,
                        sample_order=False, fillna=0)
    if data_t1_path is not None:
        data_t1 = bb.load.load_data(f'{dir_prefix}/{data_t1_path}', nodes, norm=node_normalization,
                            delimiter=',', log1p=False, transpose=transpose,
                            sample_order=False, fillna=0)
    else:
        data_t1 = None

    # Only need to pass 'data_t0' since this data is not split into train/test
    #TODO: change the below code so that you can input which column should be
    # replaced with "class" instead of full cluster_header_list
    clusters = bb.utils.get_clusters(data_t0, cellID_table=f"{dir_prefix}/{cellID_table}",
                               cluster_header_list=cluster_header_list)

    if not os.path.exists(f"{dir_prefix}/{brcd}/data_split"):
        os.makedirs(f"{dir_prefix}/{brcd}/data_split")

    split_train_test_crossval(data_t0, data_t1, clusters, f"{dir_prefix}/{brcd}/data_split", fname=fname)


# iterate through datasets

for idx in range(1, 5):
    # if idx == 0:
    #     fit_rules = False

    fname = f"human_tumors_{idx}"
    print(fname)
    validation_fname = f'validation/{fname}/'
    data_train_t0_path = f'{brcd}/data_split/train_t0_{fname}.csv'
    data_test_t0_path = f'{brcd}/data_split/test_t0_{fname}.csv'
    print(f'{dir_prefix}/{data_train_t0_path}')
    data_train_t0 = bb.load.load_data(f'{dir_prefix}/{data_train_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=True, sample_order=False, fillna=0)

    data_train_t1 = None

    data_test_t0 = bb.load.load_data(f'{dir_prefix}/{data_test_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                      log1p=False, transpose=True, sample_order=False, fillna = 0)

    data_test_t1 = None

    clusters = bb.utils.get_clusters(data_train_t0,data_test=data_test_t0, is_data_split=True,
                                                          cellID_table=f"{dir_prefix}/{cellID_table}",
                                                          cluster_header_list=cluster_header_list)
    # =============================================================================
    # Read in binarized data
    # =============================================================================
    print('Binarizing data')
    if write_binarized_data: save = True
    else: save = False
    if not os.path.exists(f"{dir_prefix}/{brcd}/binarized_data"):
        # Create a new directory because it does not exist
        os.makedirs(f"{dir_prefix}/{brcd}/binarized_data")

    data_t0 = bb.load.load_data(f'{dir_prefix}/{data_path}', nodes, norm=node_normalization,
                                delimiter=',', log1p=False, transpose=transpose,
                                sample_order=False, fillna=0)
    binarized_data_t0 = bb.proc.binarize_data(data_t0, phenotype_labels=clusters, save = save,
                                                    save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_t0_{fname}')

    binarized_data_train_t0 = bb.proc.binarize_data(data_train_t0, phenotype_labels=clusters, save = save,
                                           save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_train_t0_{fname}')
    if t1:
        binarized_data_train_t1 = bb.proc.binarize_data(data_train_t1, phenotype_labels=clusters, save = save,
                                              save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_train_t1_{fname}')
    else: binarized_data_train_t1 = None

    print('Binarizing test data')
    binarized_data_test = bb.proc.binarize_data(data_test_t0, phenotype_labels=clusters, save = save,
                                                save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_test_t0_{fname}')

    if t1:
        binarized_data_test_t1 = bb.proc.binarize_data(data_test_t1, phenotype_labels=clusters, save = save,
                                                   save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_test_t1_{fname}')
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
            rules, regulators_dict,strengths, signed_strengths = bb.tl.get_rules_scvelo(data = data_train_t0, data_t1 = data_train_t1,
                                                                                        vertex_dict=vertex_dict,
                                                                                        plot=False,
                                                                                        threshold=node_threshold)
        else:
            print("Running classic BooleaBayes rule fitting with a single timepoint...")
            rules, regulators_dict,strengths, signed_strengths = bb.tl.get_rules(data = data_train_t0,
                                                                                 vertex_dict=vertex_dict,
                                                                                 plot=False,
                                                                                 threshold=node_threshold)
        bb.tl.save_rules(rules, regulators_dict, fname=f"{dir_prefix}/{brcd}/rules/rules_{fname}.txt")
        strengths.to_csv(f"{dir_prefix}/{brcd}/rules/strengths_{fname}.csv")
        signed_strengths.to_csv(f"{dir_prefix}/{brcd}/rules/signed_strengths_{fname}.csv")
        draw_grn(graph,vertex_dict,rules, regulators_dict,f"{dir_prefix}/{brcd}/{fname}_network.pdf", save_edge_weights=True,
                 edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights_{fname}.csv")#, gene2color = gene2color)
    else:
        print("Reading in pre-generated rules...")
        rules, regulators_dict = bb.load.load_rules(fname=f"{dir_prefix}/{brcd}/rules/rules_{fname}.txt")

        if plot_network:
            draw_grn(graph,vertex_dict,rules, regulators_dict,f"{dir_prefix}/{brcd}/{fname}_network.pdf", save_edge_weights=True,
                 edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv")#, gene2color = gene2color)
    # =============================================================================
    # Calculate AUC for test dataset for a true error calculation
    # =============================================================================

    if run_validation:
        print("Running validation step...")
        try:
            os.mkdir(f"{dir_prefix}{brcd}/validation")
        except FileExistsError:
            pass

        VAL_DIR = f"{dir_prefix}{brcd}/{validation_fname}"
        try:
            os.mkdir(VAL_DIR)
        except FileExistsError:
            pass

        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(data_test_t0, data_test_t1 = None, nodes = nodes,
                                                                        regulators_dict=regulators_dict, rules = rules,
                                                                        save=True, save_dir=VAL_DIR,plot = True,
                                                                        show_plots=False, save_df=True, fname = fname)
        # Saves auc values for each gene (node) in the passed directory as 'aucs.csv'
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)


    else:
        print("Skipping validation step...")

    if validation_averages:
        print("Calculating validation averages...")
        VAL_DIR = f"{dir_prefix}{brcd}/{validation_fname}"

        if run_validation == False:
            # Function to calculate roc and tpr, fpr, area from saved validation files
            # if validation == False, read in values from files instead of from above
            tprs_all, fprs_all, area_all = bb.tl.roc_from_file(f'{VAL_DIR}/accuracy_plots', nodes, save=True, save_dir=VAL_DIR)

        aucs = pd.read_csv(f'{VAL_DIR}/aucs.csv', header=None, index_col=0)
        print("AUC means: ",aucs.mean())

        # bb.plot.plot_aucs(aucs, save=True, save_dir=VAL_DIR, show_plot=True)
        bb.plot.plot_aucs(VAL_DIR, save=True, show_plot=True) #once BB > 0.0.7, change to this line


        bb.plot.plot_validation_avgs(fprs_all, tprs_all, len(nodes), area_all, save=True, save_dir=VAL_DIR, show_plot=True)


        ## bb version > 0.1.7
        summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
        bb.plot.plot_sklearn_metrics(VAL_DIR)
        bb.plot.plot_sklearn_summ_stats(summary_stats.drop("max_error", axis = 1), VAL_DIR, fname = "")

    else:
        print("Skipping validation averaging...")
    # =============================================================================
    # Get attractors and set phenotypes using nearest neighbors
    # =============================================================================

time2 = time.time()
time_for_job = (time2 - time1) / 60.
print("Time for job: ", time_for_job)

log_job(dir_prefix, brcd, random_state, network_path, data_path, None, cellID_table, node_normalization,
        node_threshold, split_train_test, write_binarized_data,fit_rules,run_validation,validation_averages,
        False,False,2,False,[],[],False, False,
        time = time_for_job, job_barcode= job_brcd,
        notes_for_job=notes_for_log)