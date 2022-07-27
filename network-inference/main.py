#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 2020 4:31:33
Follow-up code after parallel_rule_builder_Xval.py
- Plots range of AUC values for cross-validated rule sets
- Re-fits rules with the entire training dataset
--- alternative would be to average the rules from the cross-validation folds
- Calculates AUC for test dataset (which has not been used to fit the rules) for a true error calculation
- Gets attractors and sets the phenotype of each using nearest neighbors from single cell dataset
- Calculates stability of each attractor and destabilizers/stabilizers using random walks through STG
- Calculates likelihood of reaching other attractors under different perturbations
"""
# =============================================================================
# If running from command line, Read in the arguments
# This code can be edited to input a barcode and read the arguments
# from the Job_specs.csv file
# =============================================================================
# =============================================================================
# Import packages.
# See spec-file.txt for packages
# To recreate the environment : conda create --name booleabayes --file spec-file.txt
# =============================================================================
from src import graph_fit, graph_utils
import time
import seaborn as sns
from src.graph_utils import *

customPalette = sns.color_palette('tab10')

def idx2binary(idx, n):
    binary = "{0:b}".format(idx)
    return "0" * (n - len(binary)) + binary

# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
split_train_test = True
write_binarized_data = True
fit_rules = True
validation = True
validation_averages = True
find_average_states = True
find_attractors = True
tf_basin = -1 # if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
filter_attractors = True
on_nodes = []
off_nodes = []

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference'
network_path = '_0_network.csv'
data_path = 'data/t0_M2.csv'
data_t1_path = 'data/t1_M2.csv'
data_test_path = '2364/test_t0_M2.csv'
data_test_t1_path = '2364/test_t1_M2.csv'
cellID_table = 'data/M2_clusters.csv'
#########################################
brcd = str(0000)
cluster_header_list = ['class'] #don't rename this; replace header with "class" for whichever cluster ID column you are using
# cluster_header_list = ["cell.line","source","branch_col","class","subtype_v2","phenotype","nphenotype"]
node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True
test_set = 'validation_set'
fname = 'M2'


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
# Write out information about the this job
# =============================================================================
# Append the results to a MasterResults file
T = {}
# t2 = time.time()
# T['time'] = (t2 - t1) / 60.
# # How much memory did I use?   Only can use on linux platform
# if os.name == 'posix':
#     T['memory_Mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
# else:
#     T['memory_Mb'] = np.nan
T['barcode'] = brcd
T['dir_prefix'] = dir_prefix
T['network_path'] = network_path
T['data_path'] = data_path
T['cellID_table'] = cellID_table
T['node_normalization'] = node_normalization
T['node_threshold'] = node_threshold

T = pd.DataFrame([T])
if not os.path.isfile(dir_prefix + 'Job_specs.csv'):
    T.to_csv(dir_prefix + 'Job_specs.csv')
else:
    with open(dir_prefix + 'Job_specs.csv', 'a') as f:
        T.to_csv(f, header=False)

if not os.path.exists(dir_prefix + brcd):
    # Create a new directory because it does not exist
    os.makedirs(dir_prefix + brcd)

# =============================================================================
# Start the timer and generate a barcode identifier for this job
# =============================================================================
t1 = time.time()
print(brcd)

# =============================================================================
# Load the network
# =============================================================================
graph, vertex_dict = graph_utils.load_network(f'{dir_prefix + network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)
v_names = graph.vertex_properties['name']  # Map of vertex -> name (basically the inverse of vertex_dict)
nodes = sorted(vertex_dict.keys())
print('Reading in data')

# =============================================================================
# Load the data and clusters
# =============================================================================
if split_train_test:
    data = graph_utils.load_data(op.join(dir_prefix, f"{data_path}"), nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=transpose, sample_order=False, fillna=0)
    data_t1 = graph_utils.load_data(op.join(dir_prefix, f"{data_t1_path}"), nodes, norm=node_normalization,
                                    delimiter=',',
                                    log1p=False, transpose=transpose, sample_order=False, fillna=0)
    print('Reading in cell cluster labels')
    if cellID_table is not None:
        clusters = pd.read_csv(op.join(dir_prefix, f"{dir_prefix + cellID_table}"), index_col=0, header=0,
                               delimiter=',')
        # clusters.columns = ["cell.stage", "class", "cell.protocol"]
        # clusters.columns = ["class", "pheno_color","Tuft_score","nonNE_score","NE_score","NEv2_score","NEv1_score","treatment"]
        clusters.columns = cluster_header_list
    else:
        clusters = pd.DataFrame([0] * len(data.index), index=data.index, columns=['class'])
    data, data_test, data_t1, data_test_t1, clusters_train, clusters_test = graph_utils.split_train_test(data, data_t1,
                                                                                                         clusters,
                                                                                                         dir_prefix,
                                                                                                         fname=fname)
else:
    data = graph_utils.load_data(op.join(dir_prefix, f"{data_path}"), nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=transpose, sample_order=False, fillna=0)
    data_t1 = graph_utils.load_data(op.join(dir_prefix, f"{data_t1_path}"), nodes, norm=node_normalization,
                                    delimiter=',',
                                    log1p=False, transpose=transpose, sample_order=False, fillna=0)

    data_test = graph_utils.load_data(op.join(dir_prefix, f"{data_test_path}"), nodes, norm=node_normalization, delimiter=',',
                                      log1p=False, transpose=transpose, sample_order=False, fillna = 0)
    data_test_t1 = graph_utils.load_data(op.join(dir_prefix, f"{data_test_t1_path}"), nodes, norm=node_normalization, delimiter=',',
                                         log1p=False, transpose=transpose, sample_order=False, fillna = 0)

    print('Reading in cell cluster labels')
    if cellID_table is not None:
        clusters = pd.read_csv(op.join(dir_prefix, f"{dir_prefix+cellID_table}"), index_col=0, header=0, delimiter=',')
        # clusters.columns = ["cell.stage", "class", "cell.protocol"]
        # clusters.columns = ["class", "pheno_color","Tuft_score","nonNE_score","NE_score","NEv2_score","NEv1_score","treatment"]
        clusters.columns = cluster_header_list
        clusters_train = clusters.loc[data.index]
        clusters_test = clusters.loc[data_test.index]
    else:
        clusters = pd.DataFrame([0]*len(data.index), index = data.index, columns=['class'])

# =============================================================================
# Read in binarized data
# =============================================================================
print('Binarizing data')
binarized_data = graph_utils.binarize_data(data, phenotype_labels=clusters)
binarized_data_t1 = graph_utils.binarize_data(data_t1, phenotype_labels=clusters)
print('Binarizing test data')
binarized_data_test = graph_utils.binarize_data(data_test, phenotype_labels=clusters)
binarized_data_t1_test = graph_utils.binarize_data(data_test_t1, phenotype_labels=clusters)

if write_binarized_data:
    with open(dir_prefix + brcd + os.sep + 'binarized_data_' + brcd + '.csv', 'w+') as outfile:
        for k in binarized_data.keys():
            outfile.write(f"{k}: {binarized_data[k]}\n")
    with open(dir_prefix + brcd + os.sep + 'binarized_data_t1_' + brcd + '.csv', 'w+') as outfile:
        for k in binarized_data_t1.keys():
            outfile.write(f"{k}: {binarized_data_t1[k]}\n")
    with open(dir_prefix + brcd + os.sep + 'binarized_data_test' + brcd + '.csv', 'w+') as outfile:
        for k in binarized_data_test.keys():
            outfile.write(f"{k}: {binarized_data_test[k]}\n")
    with open(dir_prefix + brcd + os.sep + 'binarized_data_t1_test' + brcd + '.csv', 'w+') as outfile:
        for k in binarized_data_t1_test.keys():
            outfile.write(f"{k}: {binarized_data_t1_test[k]}\n")


# =============================================================================
# Re-fit rules with the entire training dataset
# =============================================================================
if fit_rules:
    try:
        os.mkdir(dir_prefix + brcd + os.sep + test_set)
    except FileExistsError:
        print(f"Folder {brcd}/{test_set} already exists. Rewriting any data inside folder.")


    rules, regulators_dict,strengths, signed_strengths = graph_fit.get_rules_scvelo(data, data_t1, vertex_dict,
                                                                                    directory=dir_prefix + brcd + os.sep + "rules_" + brcd,
                                                                                    plot=False, threshold=node_threshold)
    graph_fit.save_rules(rules, regulators_dict, fname=f"{dir_prefix}{brcd}/{test_set}/rules_{brcd}.txt")
    strengths.to_csv(f'{dir_prefix}{brcd}/{test_set}/strengths.csv')
    signed_strengths.to_csv(f'{dir_prefix}{brcd}/{test_set}/signed_strengths.csv')
else:
    rules, regulators_dict = graph_fit.load_rules(fname=f"{dir_prefix}{brcd}/{test_set}/rules_{brcd}.txt")
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
draw_grn(graph,vertex_dict,rules, regulators_dict,op.join(dir_prefix,f"M2_network_0.pdf"))#, gene2color = gene2color)


# =============================================================================
# Calculate AUC for test dataset for a true error calculation
# =============================================================================

if validation:
    outfile = open(f"{dir_prefix}{brcd}/tprs_fprs_{brcd}.csv", 'w+')
    # data_test = data
    ind = [x for x in np.linspace(0, 1, 50)]
    tpr_all = pd.DataFrame(index=ind)
    fpr_all = pd.DataFrame(index=ind)
    area_all = []

    outfile.write(f",,")
    for j in ind:
        outfile.write(str(j)+',')
    outfile.write('\n')
    for g in nodes:
        print(g)

        validation = graph_utils.plot_accuracy_scvelo(data_test, data_test_t1, g, regulators_dict, rules, save_plots='test',
                                                      plot=True, plot_clusters=False, save_df=True,
                                                      dir_prefix=dir_prefix + brcd + os.sep + str(test_set) + os.sep)
        tprs, fprs, area = graph_utils.roc(validation, g, n_thresholds=50, save_plots='test', plot=True, save=True,
                                           dir_prefix=dir_prefix + brcd + os.sep + str(test_set) + os.sep)
        tpr_all[g] = tprs
        fpr_all[g] = fprs
        outfile.write(f"{g},tprs,{tprs}\n")
        outfile.write(f"{g},fprs,{fprs}\n")
        area_all.append(area)
    outfile.close()

    # # save AUC values by gene
    outfile = open(f"{dir_prefix}{brcd}/aucs.csv", 'w+')
    for n, a in enumerate(area_all):
        outfile.write(f"{nodes[n]},{a} \n")
    outfile.close()

if validation_averages:
    n = len(nodes)-2
    aucs = pd.read_csv(f"{dir_prefix}{brcd}/{test_set}/auc_2364_0.csv", header = None, index_col=0)
    print(aucs.mean(axis = 1))
    aucs.columns = ['auc']
    plt.figure()
    plt.bar(height=aucs['auc'], x = aucs.index)
    plt.xticks(rotation = 90)
    plt.savefig(f"{dir_prefix}/{brcd}/aucs.pdf")

    ind = [i for i in np.linspace(0, 1, 50)]
    tpr_all = pd.DataFrame(index=ind)
    fpr_all = pd.DataFrame(index=ind)
    area_all = []

    for g in nodes:
        if g in ['NEUROD1','SIX5']: continue
        validation = pd.read_csv(
            f'{dir_prefix}{brcd}/{test_set}/{g}_validation.csv',
            index_col=0, header=0)
        tprs, fprs, area = graph_utils.roc(validation, g, n_thresholds=50, save_plots='', save = False, plot = False)
        tpr_all[g] = tprs
        fpr_all[g] = fprs
        area_all.append(area)
    print(area_all)

    plt.figure()
    ax = plt.subplot()
    plt.plot(fpr_all.sum(axis=1) / n, tpr_all.sum(axis=1) / n, '-o')
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(f"ROC Curve Data \n {np.sum(area_all) / n}")
    plt.savefig(f'{dir_prefix}/{brcd}/{test_set}/ROC_AUC_average.pdf')
# =============================================================================
# Get attractors and set phenotypes using nearest neighbors
# =============================================================================
n = len(nodes)
n_states = 2 ** n

if find_average_states:

    # #cluster name will be appended to this name to form 1 txt file for each phenotype in binarized_data
    # #average state of each subtype
    average_states = dict()

    for k in binarized_data.keys():
        ave = graph_utils.average_state(binarized_data[k], n)
        state = ave.copy()
        state[state < 0.5] = 0
        state[state >= 0.5] = 1
        state = [int(i) for i in state]
        idx = graph_utils.state2idx(''.join(["%d" % i for i in state]))
        average_states[k] = idx
    print(average_states)

    file = open(op.join(dir_prefix,f'{brcd}/average_states.txt'), 'w+')
    #plot average state for each subtype
    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in average_states.keys():
        file_idx = open(op.join(dir_prefix, f'{brcd}/average_states_idx_{k}.txt'), 'w+')
        file_idx.write(',average_state')
        att = graph_utils.idx2binary(average_states[k], n)
        file.write(f"{k}")
        for i in att:
            file.write(f",{i}")
            file_idx.write(f"{i}\n")
        file.write("\n")
        file_idx.close()
    file.close()
    graph_utils.plot_attractors(op.join(dir_prefix, f'{brcd}/average_states.txt'))


if find_attractors:
    outfile_name = f'{dir_prefix}{brcd}/attractors'
    if tf_basin < 0:
        dist_dict = dict()
        # find average minimum distance between binarized data points within each cluster
        # and use this to inform the size of tf_basin to search for attractors
        for k in sorted(binarized_data.keys()):
            print(k)
            distances = []
            for s in binarized_data[k]:
                if len(binarized_data[k]) == 1:
                    distances = [4]
                else:
                    min_dist = 20
                    for t in binarized_data[k]:
                        if s == t: pass
                        else:
                            dist = graph_utils.hamming_idx(s, t, n)
                            if dist < min_dist: min_dist = dist
                    distances.append(min_dist)
            try:
                dist_dict[k] = int(np.ceil(np.mean(distances)))
                print(dist_dict[k])

            except ValueError: print("Not enough data in group to find distances.")
        attractor_dict = graph_utils.find_attractors(binarized_data, rules, nodes, regulators_dict, outfile_name, tf_basin=dist_dict, on_nodes=on_nodes, off_nodes=off_nodes)

    else:
        attractor_dict = graph_utils.find_attractors(binarized_data, rules, nodes, regulators_dict, outfile_name, tf_basin=tf_basin, on_nodes=on_nodes, off_nodes=off_nodes)


    file = open(op.join(dir_prefix, f'{brcd}/attractors_unfiltered.txt'), 'w+')
    # plot average state for each subtype
    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in attractor_dict.keys():
        att = [graph_utils.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            file.write(f"{k}")
            for c in i:
                file.write(f",{c}")
            file.write("\n")

    file.close()
    graph_utils.plot_attractors(op.join(dir_prefix, f'{brcd}/attractors_unfiltered.txt'))


if filter_attractors:
    average_states = {"SCLC-A": [], "SCLC-A2": [], 'SCLC-Y': [], 'SCLC-P': [], 'SCLC-N': [], 'SCLC-uncl': []}
    attractor_dict = {"SCLC-A": [], "SCLC-A2": [], 'SCLC-Y': [], 'SCLC-P': [], 'SCLC-N': [], 'SCLC-uncl': []}

    for phen in ['SCLC-A', 'SCLC-N', 'SCLC-A2', 'SCLC-P', 'SCLC-Y']:
        d = pd.read_csv(op.join(dir_prefix, f'{brcd}/average_states_idx_{phen}.txt'), sep=',', header=0)
        average_states[f'{phen}'] = list(np.unique(d['average_state']))
        d = pd.read_csv(op.join(dir_prefix, f'{brcd}/attractors_{phen}.txt'), sep = ',', header = 0)
        attractor_dict[f'{phen}'] =  list(np.unique(d['attractor']))

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
                        p_dist = graph_utils.hamming_idx(x, average_states[p], len(nodes))
                        q_dist = graph_utils.hamming_idx(x, average_states[q], len(nodes))
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
    file = open(op.join(dir_prefix,f'{brcd}/attractors_filtered.txt'), 'w+')
    # plot attractors
    for j in nodes:
        file.write(f",{j}")
    file.write("\n")
    for k in attractor_dict.keys():
        att = [graph_utils.idx2binary(x, len(nodes)) for x in attractor_dict[k]]
        for i, a in zip(att, attractor_dict[k]):
            file.write(f"{k}")
            for c in i:
                file.write(f",{c}")
            file.write("\n")

    file.close()
    graph_utils.plot_attractors(op.join(dir_prefix, f'{brcd}/attractors_filtered.txt'))

# =============================================================================
# Perform random walks for calculating stability and identifying destabilizers
# =============================================================================

# record random walk from each attractor in each phenotype with different perturbations
# will make plots for each perturbation for each starting state

dir_prefix_walks = op.join(dir_prefix, brcd)
radius = 4

# attractor_dict = {'A2': [32136863904], 'P': [], 'A': [21349304528], 'uncl': [1446933], 'Y': [14897871719], 'N': [15703179135,17045356415]} #walks from original attractors

if False:
    for k in attractor_dict.keys():
        steady_states = attractor_dict[k]
        random_walks(steady_states, radius, rules, regulators_dict,
                     nodes, dir_prefix=dir_prefix_walks, phenotype_name = k,
                     perturbations = True,  iters=1000, max_steps = 500)

    # # record random walk from each attractor in each phenotype with different radii for stability analysis

    for k in attractor_dict.keys():
        steady_states = attractor_dict[k]
        dir_prefix_walks = op.join(dir_prefix, brcd)
        for radius in [1,2,3,5,6,7,8]:
            random_walks(steady_states, radius, rules, regulators_dict,
                         nodes, dir_prefix=dir_prefix_walks,
                         perturbations=False, iters=1000, max_steps=500)

    # record random walk from random states to compare to for stability
    # for random walks to compare stability of attractor states
    try:
        os.mkdir(op.join(dir_prefix, f'{brcd}/walks/random'))
    except FileExistsError:
        pass
    dir_prefix_random = op.join(dir_prefix, f'{brcd}/walks/random')

    length = 100
    random_list = []
    for i in range(length):
        rand_state = random.choices([0,1], k=n)
        rand_idx = state_bool2idx(rand_state)
        random_list.append(rand_idx)

    for radius in [1,2,3,4,5,6,7,8]:
        random_walks(random_list, radius, rules, regulators_dict,
                     nodes, dir_prefix=dir_prefix_walks,
                     perturbations=False, iters=1000, max_steps=500)

# =============================================================================
# Calculate and plot stability of each attractor
# =============================================================================

# =============================================================================
# Calculate likelihood of reaching other attractors
# =============================================================================

# record random walk from one attractor to another for each combination of attractors
# give list of perturbation nodes and repeat walks with perturbed nodes to record #
# that make it from one attractor to another

# =============================================================================
# Write out information about the this job
# =============================================================================
# Append the results to a MasterResults file
T = {}
t2 = time.time()
T['time'] = (t2 - t1) / 60.
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

T = pd.DataFrame([T])
if not os.path.isfile(dir_prefix + 'Job_specs_post.csv'):
    T.to_csv(dir_prefix + 'Job_specs_post.csv')
else:
    with open(dir_prefix + 'Job_specs_post.csv', 'a') as f:
        T.to_csv(f, header=False)


