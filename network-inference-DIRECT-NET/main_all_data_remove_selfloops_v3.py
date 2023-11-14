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
fit_rules = False
run_validation = False
validation_averages = False
find_average_states = True
find_attractors = True
tf_basin = 2 # if -1, use average distance between clusters for search basin for attractors.
# otherwise use the same size basin for all phenotypes. For single cell data, there may be so many samples that average distance is small.
filter_attractors = True
perturbations = False
stability = False
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

validation_fname = f'validation/'
# fname = f"{sample}"
fname = "combined"
notes_for_log = "Attractors for updated DIRECT-NET network with 2020db and indpendent LASSO models, wo sinks"

## Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks.csv'
data_path = f'data/adata_imputed_combined_v3.csv'
t1 = False
data_t1_path = None #if no T1 (i.e. single dataset), replace with None

## Set metadata information
cellID_table = 'data/AA_clusters_splitgen.csv'
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
# brcd = str(random.randint(0,99999))
brcd = str(6666)
print(brcd)
# if rerunning a brcd and data has already been split into training and testing sets, use the below code
# Otherwise, these settings are ignored
data_train_t0_path = f'{brcd}/data_split/train_t0_{fname}.csv'
data_train_t1_path = None #if no T1, replace with None
data_test_t0_path = f'{brcd}/data_split/test_t0_{fname}.csv'
data_test_t1_path = None #if no T1, replace with None

# data_train_t0_path = data_path
# data_test_t0_path = data_path

## Set job barcode and random_state
# temp = sys.stdout

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
if t1 == True:
    if split_train_test == True:
        if data_t1_path is None:
            raise Exception('t1 is set to True, but no data_t1_path given.')
    else:
        if data_train_t1_path is None or data_test_t1_path is None:
            raise Exception("t1 is set to True, but no data_[train/test]_t1_path is given.")


# =============================================================================
# Load the network
# =============================================================================
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                              remove_sources=remove_sources)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

if print_graph_information:
    print_graph_info(graph, vertex_dict, nodes,  fname, brcd = brcd, dir_prefix = dir_prefix,plot = True,
                     add_edge_weights=False)

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

    data_train_t0, data_test_t0, data_train_t1, data_test_t1, clusters_train, clusters_test =  bb.utils.split_train_test(data_t0, data_t1, clusters,
                                                                                                        f"{dir_prefix}/{brcd}/data_split", fname=fname)
                                                                                                        # random_state = random_state)
else: #load the data
    data_train_t0 = bb.load.load_data(f'{dir_prefix}/{data_train_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=True, sample_order=False, fillna=0)
    if t1:
        data_train_t1 = bb.load.load_data(f'{dir_prefix}/{data_train_t1_path}', nodes, norm=node_normalization,
                                    delimiter=',',
                                    log1p=False, transpose=True, sample_order=False, fillna=0)
    else: data_train_t1 = None

    data_test_t0 = bb.load.load_data(f'{dir_prefix}/{data_test_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                      log1p=False, transpose=True, sample_order=False, fillna = 0)

    if t1:
        data_test_t1 = bb.load.load_data(f'{dir_prefix}/{data_test_t1_path}', nodes, norm=node_normalization, delimiter=',',
                                         log1p=False, transpose=True, sample_order=False, fillna = 0)
    else: data_test_t1 = None

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
    bb.tl.save_rules(rules, regulators_dict, fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")
    strengths.to_csv(f"{dir_prefix}/{brcd}/rules/strengths.csv")
    signed_strengths.to_csv(f"{dir_prefix}/{brcd}/rules/signed_strengths.csv")
    draw_grn(graph,vertex_dict,rules, regulators_dict,f"{dir_prefix}/{brcd}/{fname}_network.pdf", save_edge_weights=True,
             edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv")#, gene2color = gene2color)
else:
    try:
        print("Reading in pre-generated rules...")
        rules, regulators_dict = bb.load.load_rules(fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")

        if plot_network:
            draw_grn(graph,vertex_dict,rules, regulators_dict,f"{dir_prefix}/{brcd}/{fname}_network.pdf", save_edge_weights=True,
                 edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv")#, gene2color = gene2color)
    except FileNotFoundError:
        print("Rules file not found. Please set fit_rules to True to generate rules.")
# =============================================================================
# Calculate AUC for test dataset for a true error calculation
# =============================================================================

if run_validation:
    print("Running validation step...")
    VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"
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
    VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"

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
n = len(nodes)
n_states = 2 ** n

if find_average_states:
    print("Finding average states...")
    ATTRACTOR_DIR = f"{dir_prefix}{brcd}/attractors"
    try:
        os.mkdir(ATTRACTOR_DIR)
    except FileExistsError:
        pass
    # Find average states from binarized data and write the avg state index files
    average_states = bb.tl.find_avg_states(binarized_data_t0, nodes, save_dir=ATTRACTOR_DIR)
    print('Average states: ', average_states)

    # Plot average state for each subtype
    bb.plot.plot_attractors(f'{ATTRACTOR_DIR}/average_states.txt', save_dir="")

else:
    print("Skipping finding average states...")

if find_attractors:
    print("Finding attractors...")
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
    try:
        os.mkdir(ATTRACTOR_DIR)
    except FileExistsError:
        pass

    start = time.time()

    attractor_dict = bb.tl.find_attractors(binarized_data_t0, rules, nodes, regulators_dict, tf_basin=tf_basin,threshold=.5,
                                    save_dir=ATTRACTOR_DIR, on_nodes=on_nodes, off_nodes=off_nodes)
    end = time.time()
    print('Time to find attractors: ', str(timedelta(seconds=end-start)))

    outfile = open(f'{ATTRACTOR_DIR}/attractors_unfiltered.txt', 'w+')
    bb.tl.write_attractor_dict(attractor_dict, nodes, outfile)

    # Plot average state for each subtype
    bb.plot.plot_attractors(f'{ATTRACTOR_DIR}/attractors_unfiltered.txt', save_dir="")

else:
    print("Skipping finding attractors...")

if filter_attractors:
    print("Filtering attractors...")
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
    AVE_STATES_DIR = f"{dir_prefix}/{brcd}/attractors/"

    average_states = {}
    attractor_dict = {}
    average_states_df = pd.read_csv(f'{AVE_STATES_DIR}/average_states.txt', sep=',', header=0, index_col=0)
    for i,r in average_states_df.iterrows():
        s = ""
        cnt = 0
        for letter in list(r):
            if cnt > 0:
                s = s+str(letter)
            cnt += 1
        average_states[i] = bb.utils.state2idx(s)
    print(average_states)

    for phen in clusters['class'].unique():
        d = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_{phen}.txt', sep = ',', header = 0)
        attractor_dict[f'{phen}'] =  list(np.unique(d['attractor']))

    print(attractor_dict)
        #     ##### Below code compares each attractor to average state for each subtype instead of closest single binarized data point
    a = attractor_dict.copy()
    # attractor_dict = a.copy()
    for p in attractor_dict.keys():
        print(p)
        for q in attractor_dict.keys():
            print("q", q)
            if p == q: continue
            n_same = list(set(attractor_dict[p]).intersection(set(attractor_dict[q])))
            if len(n_same) != 0:
                for x in n_same:
                    p_dist = bb.utils.hamming_idx(x, average_states[p], len(nodes))
                    q_dist = bb.utils.hamming_idx(x, average_states[q], len(nodes))
                    if p_dist < q_dist:
                        a[q].remove(x)
                    elif q_dist < p_dist:
                        a[p].remove(x)
                    else:
                        a[q].remove(x)
                        a[p].remove(x)
                        try:
                            a[f'{q}_{p}'].append(x)
                        except KeyError:
                            a[f'{q}_{p}'] = [x]
    attractor_dict = a
    print(attractor_dict)
    file = open(f"{ATTRACTOR_DIR}/attractors_filtered.txt", 'w+')
    # plot attractors
    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in attractor_dict.keys():
        att = [bb.utils.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            file.write(f"{k}")
            for c in i:
                file.write(f",{c}")
            file.write("\n")

    file.close()
    # bb.plot.plot_attractors(f'{ATTRACTOR_DIR}/attractors_filtered.txt', save_dir="")

    # added to Booleabayes version > 0.1.9
    def plot_attractors_clustermap(fname, sep = ","):
        att = pd.read_table(fname, sep=sep, header=0, index_col=0)
        att = att.transpose()
        plt.figure(figsize=(20, 12))
        clust = sorted(att.columns.unique())
        lut = dict(zip(clust, sns.color_palette("tab20")))
        column_series = pd.Series(att.columns)
        row_colors = column_series.map(lut)
        g = sns.clustermap(att.T.reset_index().drop('index', axis = 1).T,linecolor="lightgrey",
                           linewidths=1, figsize = (20,12),cbar_pos = None,
                           cmap = 'binary', square = True, row_cluster = True, col_cluster = False, yticklabels = True, xticklabels = False,col_colors = row_colors)
        markers = []
        for i in lut.keys():
            markers.append(plt.Line2D([0,0],[0,0],color=lut[i], marker='o', linestyle=''))
        lgd = plt.legend(markers, lut.keys(), numpoints=1, loc = 'upper left', bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout()
        plt.savefig(f"{fname.split('.')[0]}_clustered.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

    plot_attractors_clustermap(fname = f"{ATTRACTOR_DIR}/attractors_filtered.txt")

    make_jaccard_heatmap(f'{ATTRACTOR_DIR}/attractors_filtered.txt', cmap='viridis',
                         set_color={"Generalist":'lightgrey'},
                         clustered=True,
                         figsize=(10, 10), save=True)

    # with open(f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5/attractor_dict.txt", 'w') as convert_file:
    #     convert_file.write(json.dumps(attractor_dict))

else:
    print("Skipping filtering attractors...")
# =============================================================================
# Perform random walks for calculating stability and identifying destabilizers
# =============================================================================

# record random walk from each attractor in each phenotype with different perturbations
# will make plots for each perturbation for each starting state

dir_prefix_walks = op.join(dir_prefix, brcd)


if perturbations:
    print("Running TF perturbations...")
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []

    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

    bb.rw.random_walks(attractor_dict,
                       rules,
                       regulators_dict,
                       nodes,
                       save_dir = f"{dir_prefix}/{brcd}/",
                       radius=2,
                       perturbations=True,
                       iters=500,
                       max_steps=500,
                       stability=False,
                       reach_or_leave="leave",
                       random_start=False,
                       on_nodes=[],
                       off_nodes=[],
                       overwrite_walks=False
                       )
    perturbations_dir = f"{dir_prefix}/{brcd}/perturbations"

    bb.tl.perturbations_summary(attractor_dict,perturbations_dir, show = False, save = True, plot_by_attractor = True,
                                save_dir = "clustered_perturb_plots", save_full = True, significance = 'both', fname = "",
                                ncols = 5, mean_threshold = -0.3)

    ## gene dict and plots with threshold = -0.2
    perturb_dict, full = bb.utils.get_perturbation_dict(attractor_dict, perturbations_dir, significance = 'both', save_full=False,
                                               mean_threshold=-0.2)
    perturb_gene_dict = bb.utils.reverse_perturb_dictionary(perturb_dict)
    bb.plot.plot_perturb_gene_dictionary(perturb_gene_dict, full,perturbations_dir,show = False, save = True, ncols = 5, fname = "_0.2")
else:
    "Skipping perturbations..."

if stability:
    print("Running stability...")
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []
    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

    subset = set(attractor_dict.keys()).difference({'NE_1','Club cells_2'})
    attractor_dict_edit = {k:v for k,v in attractor_dict.items() if k in subset}

    bb.rw.random_walks(attractor_dict_edit,
                       rules,
                       regulators_dict,
                       nodes,
                       save_dir = f"{dir_prefix}/{brcd}/",
                       radius=[3,5,6],
                       perturbations=False,
                       iters=500,
                       max_steps=500,
                       stability=True,
                       reach_or_leave="leave",
                       random_start=50,
                       on_nodes=[],
                       off_nodes=[],
                       )
else:
    print("Skipping stability...")

# =============================================================================
# Calculate and plot stability of each attractor
# =============================================================================
if False:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
    attractor_dict = bb.utils.get_attractor_dict(ATTRACTOR_DIR, filtered = True)

    bb.rw.random_walks(attractor_dict,
                       rules,
                       regulators_dict,
                       nodes,
                       save_dir = f"{dir_prefix}/{brcd}/",
                       radius=[2,3,4,5,6],
                       perturbations=False,
                       iters=500,
                       max_steps=500,
                       stability=True,
                       reach_or_leave="leave",
                       random_start=50,
                       on_nodes=[],
                       off_nodes=[],
                       overwrite_walks=False
                       )


    walks_dir = f"{dir_prefix}/{brcd}/walks"
    bb.plot.plot_stability(attractor_dict, walks_dir, palette = sns.color_palette("tab20"), rescaled = True,
                   show = False, save = True)

# =============================================================================
# Calculate likelihood of reaching other attractors
# =============================================================================

# record random walk from one attractor to another for each combination of attractors
# give list of perturbation nodes and repeat walks with perturbed nodes to record #
# that make it from one attractor to another


time2 = time.time()
time_for_job = (time2 - time1) / 60.
print("Time for job: ", time_for_job, " minutes")

log_job(dir_prefix, brcd, random_state, network_path, data_path, data_t1_path, cellID_table, node_normalization,
        node_threshold, split_train_test, write_binarized_data,fit_rules,run_validation,validation_averages,
        find_average_states,find_attractors,tf_basin,filter_attractors,on_nodes,off_nodes,perturbations, stability,
        time = time_for_job, job_barcode= job_brcd,
        notes_for_job=notes_for_log)