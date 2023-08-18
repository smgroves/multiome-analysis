#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 9 2023

@author: smgroves

Code to build Booleabayes rules for a given network and dataset
For use with parallelization
Inspired by parallel_rule_builder_Xval from Christian Meyer
"""

from datetime import timedelta
import random
import time
import sys
import numpy as np
import booleabayes as bb
import pandas as pd
import os
import sklearn.model_selection as ms
import resource
import matplotlib as plt

plt.rcParams.update({'figure.max_open_warning': 0})

# =============================================================================
# If running from command line, Read in the arguments
# =============================================================================
if True:
    if len(sys.argv) < 5 or len(sys.argv) > 9:
        raise OSError(
            'Wrong number of arguments.  Must pass <directory> <network file path from parent directory> <data file path from parent directory> <cellID-table> <node_normalization (optional default=.5) node_threshold (optional default=.1)>')
    dir_prefix = sys.argv[1]
    network_path = sys.argv[2]
    data_path = sys.argv[3]
    cellID_table = sys.argv[4]
    try:
        node_normalization = .5 if len(sys.argv) <= 5 else float(sys.argv[5])
    except:
        node_normalization = .5 if len(sys.argv) <= 5 else sys.argv[5]
    node_threshold = .1 if len(sys.argv) <= 6 else float(sys.argv[6])
    remove_sinks = False if len(sys.argv) <= 7 else bool(sys.argv[7])
    remove_selfloops = True if len(sys.argv) <= 8 else bool(sys.argv[8])
    remove_sources = False if len(sys.argv) <= 9 else bool(sys.argv[9])

    brcd = str(np.random.randint(0, 10001))
else:  # Use the example data
    dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/parallel'
    network_path = 'networks/test_network.csv'
    data_path = 'data/adata_imputed_combined_v3.csv'
    cellID_table = 'data/AA_clusters_splitgen.csv'
    node_normalization = 0.5
    node_threshold = 0.1
    brcd = str(300)

    remove_sinks = False
    remove_selfloops = True
    remove_sources = False

cluster_header_list = ["class"]
transpose = True

print(os.path.isfile(dir_prefix + network_path))
print(dir_prefix + network_path)
print(node_normalization,type(node_normalization), node_threshold, type(node_threshold))

if dir_prefix[-1] != os.sep:
    dir_prefix = dir_prefix + os.sep
if not network_path.endswith('.csv') or not os.path.isfile(dir_prefix + network_path):
    raise Exception('Network path must be a .csv file.  Check file name and location')
if not data_path.endswith('.csv') or not os.path.isfile(dir_prefix + data_path):
    raise Exception('data path must be a .csv file.  Check file name and location')
if cellID_table is not None:
    if not cellID_table.endswith('.csv') or not os.path.isfile(dir_prefix + cellID_table):
        raise Exception('CellID path must be a .csv file.  Check file name and location')

# Test if the barcode is in use.  If it is generate a new one.
while True:
    if os.path.isdir(dir_prefix + brcd):
        brcd = str(np.random.randint(0, 10001))
    else:
        os.mkdir(dir_prefix + brcd)
        break
# =============================================================================
# Start the timer and generate a barcode identifier for this job
# =============================================================================
t1 = time.time()

# Set random seed for reproducibility
np.random.seed(1)
print(brcd)
# =============================================================================
# Load the network
# =============================================================================
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                              remove_sources=remove_sources)
v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
n = len(nodes)
bb.utils.print_graph_info(graph, vertex_dict, nodes, "", brcd=brcd, dir_prefix=dir_prefix, plot=False)

# =============================================================================
# Load the data and cell clusters
# =============================================================================
data = bb.load.load_data(f'{dir_prefix}/{data_path}', nodes, norm=node_normalization,
                        delimiter=',', log1p=False, transpose=transpose,
                        sample_order=False, fillna=0)
clusters = bb.utils.get_clusters(data, cellID_table=f"{dir_prefix}/{cellID_table}",
                           cluster_header_list=cluster_header_list)
binarized_data = bb.proc.binarize_data(data, phenotype_labels=clusters, save = True,
                                                save_dir=f"{dir_prefix}/{brcd}",fname=f'binarized_data')


# =============================================================================
# Generate and save rules with 5-fold cross validation
# Do density dependent sampling
# =============================================================================

n_splits = 5
kf = ms.StratifiedKFold(n_splits=n_splits)
test_set = 'validation_set'
try:
    os.mkdir(f"{dir_prefix}{brcd}/{test_set}/")
except FileExistsError:
    pass
i = 0  # Index of fold

for train_index, test_index in kf.split(data.index, clusters.loc[data.index, 'class']):
    print(f'Generating Rules for K-Fold {i}')

    try:
        os.mkdir(f"{dir_prefix}{brcd}/{test_set}/{i}/")
    except FileExistsError:
        pass
    # save rules for each barcode and cross-validation fold
    rules, regulators_dict,strengths, signed_strengths = bb.tl.get_rules(data = data.iloc[train_index],
                                                                             vertex_dict=vertex_dict,
                                                                             plot=False,
                                                                             threshold=node_threshold)
    bb.tl.save_rules(rules, regulators_dict, fname=f"{dir_prefix}/{brcd}/{test_set}/{i}/rules_{brcd}_{i}.txt")
    # strengths.to_csv(f"{dir_prefix}/{brcd}/{test_set}/{i}/strengths_{brcd}_{i}.csv")
    signed_strengths.to_csv(f"{dir_prefix}/{brcd}/{test_set}/{i}/signed_strengths_{brcd}_{i}.csv")

    # =============================================================================
    # Calculate AUC for test dataset for a true error calculation
    # =============================================================================

    VAL_DIR = f"{dir_prefix}/{brcd}/{test_set}/{i}/validation"
    try:
        os.mkdir(VAL_DIR)
    except FileExistsError:
        pass

    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(data.iloc[test_index], data_test_t1 = None, nodes = nodes,
                                                                    regulators_dict=regulators_dict, rules = rules,
                                                                    save=True, save_dir=VAL_DIR,plot = True,
                                                                    show_plots=False, save_df=True, fname = '')
    # Saves auc values for each gene (node) in the passed directory as 'aucs.csv'
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)


    print("Calculating validation averages...")

    aucs = pd.read_csv(f'{VAL_DIR}/aucs.csv', header=None, index_col=0)
    # bb.plot.plot_aucs(VAL_DIR, save=True, show_plot=False)
    # bb.plot.plot_validation_avgs(fprs_all, tprs_all, len(nodes), area_all, save=True, save_dir=VAL_DIR, show_plot=False)
    ## bb version > 0.1.7
    summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
    # bb.plot.plot_sklearn_metrics(VAL_DIR)
    # bb.plot.plot_sklearn_summ_stats(summary_stats.drop("max_error", axis = 1), VAL_DIR, fname = "")

    i += 1




# =============================================================================
# Write out information about the this job
# =============================================================================
# Append the results to a MasterResults file
T = {}
t2 = time.time()
T['time_minutes'] = (t2 - t1) / 60.
# How much memory did I use?   Only can use on linux platform
if os.name == 'posix':
    T['memory_Mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
else:
    T['memory_Mb'] = np.nan
T['barcode'] = brcd
T['dir_prefix'] = dir_prefix
T['network_path'] = network_path
T['data_path'] = data_path
T['cellID_table'] = cellID_table
T['node_normalization'] = node_normalization
T['node_threshold'] = node_threshold
T['remove_sinks'] = False
T['remove_selfloops'] = True
T['remove_sources'] = False


T = pd.DataFrame([T])
if not os.path.isfile(dir_prefix + 'Job_specs.csv'):
    T.to_csv(dir_prefix + 'Job_specs.csv')
else:
    with open(dir_prefix + 'Job_specs.csv', 'a') as f:
        T.to_csv(f, header=False)