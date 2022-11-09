import random
import time
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

customPalette = sns.color_palette('tab10')

# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
print_graph_information = True #whether to print graph info to {brcd}.txt

split_train_test = False
write_binarized_data = False
fit_rules = False
run_validation = False
validation_averages = False
find_average_states = False
find_attractors = False
tf_basin = 2 # if -1, use average distance between clusters for search basin for attractors.
# otherwise use the same size basin for all phenotypes. For single cell data, there may be so many samples that average distance is small.
filter_attractors = False
perturbations = False
stability = False
on_nodes = []
off_nodes = []

## Set variables for computation
remove_sinks=False
remove_selfloops=False
remove_sources=False

node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True
validation_fname = 'validation_set'
fname = 'M2'
notes_for_log = "Plotting perturbations for attractors 0.5"

## Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks.csv'
data_path = 'data/adata_04_nodubs_imputed_M2.csv'
t1 = False
data_t1_path = None #if no T1 (i.e. single dataset), replace with None

## Set metadata information
cellID_table = 'data/M2_clusters.csv'
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
brcd = str(3000)
print(brcd)
# if rerunning a brcd and data has already been split into training and testing sets, use the below code
# Otherwise, these settings are ignored
data_train_t0_path = f'{brcd}/data_split/train_t0_{fname}.csv'
data_train_t1_path = None #if no T1, replace with None
data_test_t0_path = f'{brcd}/data_split/test_t0_{fname}.csv'
data_test_t1_path = None #if no T1, replace with None


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

if not os.path.exists(f"{dir_prefix}/{brcd}/jobs"):
    # Create a new directory because it does not exist
    os.makedirs(f"{dir_prefix}/{brcd}/jobs")

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
    print_graph_info(graph, nodes,  fname, brcd = brcd, dir_prefix = dir_prefix,plot = False)

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
                                                                                                        f"{dir_prefix}/{brcd}/data_split", fname='M2')
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
    print("Reading in pre-generated rules...")
    rules, regulators_dict = bb.load.load_rules(fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")
#
# colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
# color_palette = sns.xkcd_palette(colors)
# attract = pd.read_csv(op.join(dir_prefix, f'{brcd}/attractors_filtered.txt'), sep = ',', header = 0, index_col = 0)
# gene2color = {}
# vertex_group = {}
# for g in attract:
#     print(g)
#     if attract.loc['10AMutant'][g] == 1:
#         print('Mutant')
#         gene2color[g] = [0.21568627450980393, 0.47058823529411764, 0.7490196078431373, 1]
#         vertex_group[g] = 0
#     elif attract.loc['10AParental'][g] == 0:
#         print('Control')
#         gene2color[g] = [0.6588235294117647, 0.6431372549019608, 0.5843137254901961,1]
#         vertex_group[g] = 1
#     else:
#         gene2color[g] = [0.5098039215686274, 0.37254901960784315, 0.5294117647058824, 1]
#         vertex_group[g] = 1
# # print(gene2color)




# =============================================================================
# Calculate AUC for test dataset for a true error calculation
# =============================================================================

if run_validation:
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
    # n = len(nodes)-2
    # aucs = pd.read_csv(f"{dir_prefix}{brcd}/{validation_fname}/auc_2364_0.csv", header = None, index_col=0)
    # print(aucs.mean(axis = 1))
    # aucs.columns = ['auc']
    # plt.figure()
    # plt.bar(height=aucs['auc'], x = aucs.index)
    # plt.xticks(rotation = 90)
    # plt.savefig(f"{dir_prefix}/{brcd}/aucs.pdf")
    #
    # ind = [i for i in np.linspace(0, 1, 50)]
    # tpr_all = pd.DataFrame(index=ind)
    # fpr_all = pd.DataFrame(index=ind)
    # area_all = []
    #
    # for g in nodes:
    #     if g in ['NEUROD1','SIX5']: continue
    #     validation = pd.read_csv(
    #         f'{dir_prefix}{brcd}/{validation_fname}/{g}_validation.csv',
    #         index_col=0, header=0)
    #     tprs, fprs, area = bb.utils.roc(validation, g, n_thresholds=50, save_plots='', save = False, plot = False)
    #     tpr_all[g] = tprs
    #     fpr_all[g] = fprs
    #     area_all.append(area)
    # print(area_all)
    #
    # plt.figure()
    # ax = plt.subplot()
    # plt.plot(fpr_all.sum(axis=1) / n, tpr_all.sum(axis=1) / n, '-o')
    # ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.ylabel("True Positive Rate")
    # plt.xlabel("False Positive Rate")
    # plt.title(f"ROC Curve Data \n {np.sum(area_all) / n}")
    # plt.savefig(f'{dir_prefix}/{brcd}/{validation_fname}/ROC_AUC_average.pdf')
else:
    print("Skipping validation averaging...")
# =============================================================================
# Get attractors and set phenotypes using nearest neighbors
# =============================================================================
n = len(nodes)
n_states = 2 ** n

if find_average_states:
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
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.7"
    try:
        os.mkdir(ATTRACTOR_DIR)
    except FileExistsError:
        pass

    start = time.time()

    attractor_dict = bb.tl.find_attractors(binarized_data_t0, rules, nodes, regulators_dict, tf_basin=tf_basin,threshold=.7,
                                    save_dir=ATTRACTOR_DIR, on_nodes=on_nodes, off_nodes=off_nodes)
    end = time.time()
    print('Time to find attractors: ', str(timedelta(seconds=end-start)))

    outfile = open(f'{ATTRACTOR_DIR}/attractors_unfiltered.txt', 'w+')
    bb.tl.write_attractor_dict(attractor_dict, nodes, outfile)

    # Plot average state for each subtype
    bb.plot.plot_attractors(f'{ATTRACTOR_DIR}/attractors_unfiltered.txt', save_dir="")

else:
    print("Skipping finding attractors")

if filter_attractors:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.7"
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
    att = pd.read_table(f"{ATTRACTOR_DIR}/attractors_filtered.txt", sep=',', header=0, index_col=0)
    att = att.transpose()
    plt.figure(figsize=(20, 12))
    clust = att.columns.unique()
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
    plt.savefig(f"{ATTRACTOR_DIR}/attractors_filtered_clustered_x.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

else:
    print("Skipping filtering attractors")
# =============================================================================
# Perform random walks for calculating stability and identifying destabilizers
# =============================================================================

# record random walk from each attractor in each phenotype with different perturbations
# will make plots for each perturbation for each starting state

dir_prefix_walks = op.join(dir_prefix, brcd)


if perturbations:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []

    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

    print("Perturbations...")
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
                       )
    perturbations_dir = f"{dir_prefix}/{brcd}/perturbations"

    #when bb version > 0.1.2, uncomment this code
    # bb.plot.plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True)


#perturbation summary plots. In future, merge this chunk with perturbations chunk above

## plot barplot of destabilization scores for each TF for all attractors
## one plot per perturbation type and per cluster type


def plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = True):
    for k in attractor_dict.keys():
        print(k)
        if clustered:
            try:
                os.mkdir(f"{perturbations_dir}/clustered_perturb_plots")
            except FileExistsError:
                pass
            results = pd.DataFrame(columns = ['attr','gene','perturb','score'])
            for attr in attractor_dict[k]:
                tmp = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                tmp.columns = ["attractor_dir","cluster","gene","perturb","score"]
                for i,r in tmp.iterrows():
                    results = results.append(pd.Series([attr, r['gene'],r['perturb'],r['score']],
                                                       index = ['attr','gene','perturb','score']), ignore_index=True)
            results_act = results.loc[results["perturb"] == 'activate']
            plt.figure()
            # my_order = results_act.sort_values(by = 'score')['gene'].values
            my_order = results_act.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
            plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')

            if len(attractor_dict[k]) == 1:
                sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order)
            else:
                sns.boxplot(data = results_act, x = 'gene', y = 'score', order = my_order)
            plt.xticks(rotation = 90, fontsize = 8)
            plt.xlabel("Gene")
            plt.ylabel("Stabilization Score")
            plt.title(f"Destabilization by TF Activation for {k} Attractors \n {len(attractor_dict[k])} Attractors")
            plt.legend([],[], frameon=False)
            plt.tight_layout()
            if show:
                plt.show()
            if save:
                plt.savefig(f"{perturbations_dir}/clustered_perturb_plots/{k}_activation_scores.pdf")
                plt.close()
                results_act = results.loc[results["perturb"] == 'activate']

            results_kd = results.loc[results["perturb"] == 'knockdown']

            plt.figure()
            # my_order = results_act.sort_values(by = 'score')['gene'].values
            my_order = results_kd.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
            plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')
            if len(attractor_dict[k]) == 1:
                sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
            else:
                sns.boxplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
            plt.xticks(rotation = 90, fontsize = 8)
            plt.xlabel("Gene")
            plt.ylabel("Stabilization Score")
            plt.title(f"Destabilization by TF Knockdown for {k} Attractors \n {len(attractor_dict[k])} Attractors")
            plt.legend([],[], frameon=False)
            plt.tight_layout()
            if show:
                plt.show()
            if save:
                plt.savefig(f"{perturbations_dir}/clustered_perturb_plots/{k}_knockdown_scores.pdf")
                plt.close()
        else:
            for attr in attractor_dict[k]:
                results = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                results.columns = ["attractor_dir","cluster","gene","perturb","score"]
                #activation plot
                results_act = results.loc[results["perturb"] == 'activate']
                colormat=list(np.where(results_act['score']>0, 'g','r'))
                results_act['color'] = colormat

                plt.figure()
                my_order = results_act.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Activation")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/activation_scores.pdf")
                    plt.close()

                #knockdown plot
                results_kd = results.loc[results["perturb"] == 'knockdown']
                colormat=list(np.where(results_kd['score']>0, 'g','r'))
                results_kd['color'] = colormat

                plt.figure()
                my_order = results_kd.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Knockdown")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/knockdown_scores.pdf")
                    plt.close()


if True:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []

    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

    perturbations_dir = f"{dir_prefix}/{brcd}/perturbations"

    plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = True)




if stability:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
    # attractor_dict = {}
    # for phen in clusters['class'].unique():
    #     d = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_{phen}.txt', sep = ',', header = 0)
    #     attractor_dict[f'{phen}'] =  list(np.unique(d['attractor']))

    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []

    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))
    # # record random walk from each attractor in each phenotype with different radii for stability analysis
    #
    # for k in attractor_dict.keys():
    #     steady_states = attractor_dict[k]
    #     dir_prefix_walks = op.join(dir_prefix, brcd)
    #     for radius in [1,2,3,5,6,7,8]:
    #         random_walks(steady_states, radius, rules, regulators_dict,
    #                      nodes, dir_prefix=dir_prefix_walks,
    #                      perturbations=False, iters=1000, max_steps=500)
    print("Stability...")
    bb.rw.random_walks(attractor_dict,
                       rules,
                       regulators_dict,
                       nodes,
                       save_dir = f"{dir_prefix}/{brcd}/",
                       radius=[2,3,4,5,6,7,8],
                       perturbations=False,
                       iters=1000,
                       max_steps=500,
                       stability=True,
                       reach_or_leave="leave",
                       random_start=False,
                       on_nodes=[],
                       off_nodes=[],
                       )
    # record random walk from random states to compare to for stability
    # for random walks to compare stability of attractor states
    # try:
    #     os.mkdir(op.join(dir_prefix, f'{brcd}/walks/random'))
    # except FileExistsError:
    #     pass
    # dir_prefix_random = op.join(dir_prefix, f'{brcd}/walks/random')
    #
    # length = 100
    # random_list = []
    # for i in range(length):
    #     rand_state = random.choices([0,1], k=n)
    #     rand_idx = bb.utils.state_bool2idx(rand_state)
    #     random_list.append(rand_idx)

    # for radius in [1,2,3,4,5,6,7,8]:
    #     random_walks(random_list, radius, rules, regulators_dict,
    #                  nodes, dir_prefix=dir_prefix_walks,
    #                  perturbations=False, iters=1000, max_steps=500)

# =============================================================================
# Calculate and plot stability of each attractor
# =============================================================================

# =============================================================================
# Calculate likelihood of reaching other attractors
# =============================================================================

# record random walk from one attractor to another for each combination of attractors
# give list of perturbation nodes and repeat walks with perturbed nodes to record #
# that make it from one attractor to another


time2 = time.time()
time_for_job = (time2 - time1) / 60.
print("Time for job: ", time_for_job)

log_job(dir_prefix, brcd, random_state, network_path, data_path, data_t1_path, cellID_table, node_normalization,
        node_threshold, split_train_test, write_binarized_data,fit_rules,run_validation,validation_averages,
        find_average_states,find_attractors,tf_basin,filter_attractors,on_nodes,off_nodes,perturbations, stability,
        time = time_for_job, job_barcode= job_brcd,
        notes_for_job=notes_for_log)