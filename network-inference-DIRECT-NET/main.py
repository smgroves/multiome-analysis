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

customPalette = sns.color_palette('tab10')

def draw_grn(G, gene2vertex, rules, regulators_dict, fname, gene2group=None, gene2color=None, type = "", B_min = 5):
    vertex2gene = G.vertex_properties['name']

    vertex_group = None
    if gene2group is not None:
        vertex_group = G.new_vertex_property("int")
        for gene in gene2group.keys():
            vertex_group[gene2vertex[gene]] = gene2group[gene]

    vertex_colors = [0.4, 0.2, 0.4, 1]
    if gene2color is not None:
        vertex_colors = G.new_vertex_property("vector<float>")
        for gene in gene2color.keys():
            vertex_colors[gene2vertex[gene]] = gene2color[gene]

    edge_weight_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))
    edge_binary_df = pd.DataFrame(index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys()))

    edge_markers = G.new_edge_property("string")
    edge_weights = G.new_edge_property("float")
    edge_colors = G.new_edge_property("vector<float>")
    for edge in G.edges():
        edge_colors[edge] = [0., 0., 0., 0.3]
        edge_markers[edge] = "arrow"
        edge_weights[edge] = 0.2

    for edge in G.edges():
        vs, vt = edge.source(), edge.target()
        source = vertex2gene[vs]
        target = vertex2gene[vt]
        regulators = regulators_dict[target]
        if source in regulators:
            i = regulators.index(source)
            n = 2 ** len(regulators)

            rule = rules[target]
            off_leaves, on_leaves = bb.tl.get_leaves_of_regulator(n, i)
            if rule[off_leaves].mean() < rule[on_leaves].mean():  # The regulator is an activator
                edge_colors[edge] = [0., 0.3, 0., 0.8]
                edge_binary_df.loc[target,source] = 1
            else:
                edge_markers[edge] = "bar"
                edge_colors[edge] = [0.88, 0., 0., 0.5]
                edge_binary_df.loc[target,source] = -1

            edge_weights[edge] = rule[on_leaves].mean() - rule[off_leaves].mean() + 0.2
            edge_weight_df.loc[target, source] = rule[on_leaves].mean() - rule[off_leaves].mean()
    G.edge_properties["edge_weights"] = edge_weights
    pos = gt.sfdp_layout(G, groups=vertex_group,mu = 1, eweight=edge_weights, max_iter=1000)
    # pos = gt.arf_layout(G, max_iter=100, dt=1e-4)
    eprops = {"color": edge_colors, "pen_width": 2, "marker_size": 15, "end_marker": edge_markers}
    vprops = {"text": vertex2gene, "shape": "circle", "size": 20, "pen_width": 1, 'fill_color': vertex_colors}
    if type == 'circle':
        state = gt.minimize_nested_blockmodel_dl(G, B_min = B_min)
        state.draw(vprops=vprops, eprops=eprops)  # mplfig=ax[0,1])
    else:
        gt.graph_draw(G, pos=pos, output=fname, vprops=vprops, eprops=eprops, output_size=(1000, 1000))
    return G, edge_weight_df, edge_binary_df


# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
split_train_test = True
write_binarized_data = True
fit_rules = False
validation = False
validation_averages = False
find_average_states = False
find_attractors = False
tf_basin = -1 # if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
filter_attractors = False
on_nodes = []
off_nodes = []

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2.csv'
data_path = 'data/Cytotrace_DGE_2_SCT.csv' #after split train test is done, these should point to training data, not full data
t1 = False

data_t1_path = None #if no T1 (i.e. single dataset), replace with None

cellID_table = 'data/metadata_final_nodashes.csv'
# Assign headers to cluster csv, with one called "class"
# cluster_header_list = ['class']

# cluster headers with "identity" replaced with "class"
cluster_header_list = ["orig.ident","nCount_RNA","nFeature_RNA","nCount_ATAC","nFeature_ATAC","nucleosome_signal",
                       "nucleosome_percentile","TSS.enrichment","TSS.percentile","barcode","sample","ATAC_snn_res.0.5",
                       "seurat_clusters","nCount_peaks","nFeature_peaks","peaks_snn_res.0.5","percent.mt","nCount_SCT",
                       "nFeature_SCT","SCT_snn_res.0.5","SCT.weight","peaks.weight","nCount_Imputed_counts",
                       "nFeature_Imputed_counts","nCount_gene_activity","nFeature_gene_activity","NE_score1",
                       "class","non.NE_score1","comb.score","S.Score","G2M.Score","Phase","old.ident","wsnn_res.0.5"
                       ]
#########################################
# brcd = str(random.Random.randint(0,99999))
brcd = str(1000)
node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = False
external_validation = 'validation_set'
fname = 'M2'

# if rerunning a brcd and data has already been split into training and testing sets, use the below code
# Otherwise, these settings are ignored
data_train_t0_path = f'{brcd}/data_split/train_t0_M2.csv'
data_train_t1_path = f'{brcd}/data_split/train_t1_M2.csv' #if no T1, replace with None
data_test_t0_path = f'{brcd}/data_split/test_t0_M2.csv'
data_test_t1_path = f'{brcd}/data_split/test_t1_M2.csv' #if no T1, replace with None

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
T['data_t1_path'] = data_t1_path
T['cellID_table'] = cellID_table
T['node_normalization'] = node_normalization
T['node_threshold'] = node_threshold
T['split_train_test'] = split_train_test
T['split_train_test'] = write_binarized_data
T['split_train_test'] = fit_rules
T['split_train_test'] = validation
T['split_train_test'] = validation_averages
T['split_train_test'] = find_average_states
T['split_train_test'] = find_attractors
T['split_train_test'] = tf_basin = -1 # if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
T['split_train_test'] = filter_attractors
T['split_train_test'] = on_nodes = []
T['split_train_test'] = off_nodes = []

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
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
print(nodes)
print("Number of nodes:", len(nodes))
print(graph)

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
    clusters = bb.utils.get_clusters(data_t0, cellID_table=f"{dir_prefix}/{cellID_table}",
                               cluster_header_list=cluster_header_list)

    if not os.path.exists(f"{dir_prefix}/{brcd}/data_split"):
        os.makedirs(f"{dir_prefix}/{brcd}/data_split")

    data_train_t0, data_test_t0, data_train_t1, data_test_t1, clusters_train, clusters_test =  bb.utils.split_train_test(data_t0, data_t1, clusters,
                                                                                                        f"{dir_prefix}/{brcd}/data_split", fname='M2')
else: #load the binarized data
    data_train_t0 = bb.load.load_data(f'{dir_prefix}/{data_train_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=transpose, sample_order=False, fillna=0)
    if t1:
        data_train_t1 = bb.load.load_data(f'{dir_prefix}/{data_train_t1_path}', nodes, norm=node_normalization,
                                    delimiter=',',
                                    log1p=False, transpose=transpose, sample_order=False, fillna=0)
    else: data_train_t1 = None

    data_test_t0 = bb.load.load_data(f'{dir_prefix}/{data_test_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                      log1p=False, transpose=transpose, sample_order=False, fillna = 0)

    if t1:
        data_test_t1 = bb.load.load_data(f'{dir_prefix}/{data_test_t1_path}', nodes, norm=node_normalization, delimiter=',',
                                         log1p=False, transpose=transpose, sample_order=False, fillna = 0)
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
else: binarized_data_test_t1 = None


# =============================================================================
# Re-fit rules with the training dataset
# =============================================================================
if fit_rules:
    if not os.path.exists(f"{dir_prefix}/{brcd}/rules"):
        # Create a new directory because it does not exist
        os.makedirs(f"dir_prefix/{brcd}/rules")
    if t1:
        rules, regulators_dict,strengths, signed_strengths = bb.tl.get_rules_scvelo(data_train_t0, data_train_t1, vertex_dict,
                                                                                    directory=dir_prefix + brcd + os.sep + "rules_" + brcd,
                                                                                    plot=False, threshold=node_threshold)
    else:
        rules, regulators_dict,strengths, signed_strengths = bb.tl.get_rules(data_train_t0, vertex_dict,
                                                                                    directory=dir_prefix + brcd + os.sep + "rules_" + brcd,
                                                                                    plot=False, threshold=node_threshold)
    bb.tl.save_rules(rules, regulators_dict, fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")
    strengths.to_csv(f"{dir_prefix}/{brcd}/rules/strengths.csv")
    signed_strengths.to_csv(f"{dir_prefix}/{brcd}/rules/signed_strengths.csv")
else:
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

        validation = bb.utils.plot_accuracy_scvelo(data_test, data_test_t1, g, regulators_dict, rules, save_plots='test',
                                                      plot=True, plot_clusters=False, save_df=True,
                                                      dir_prefix=dir_prefix + brcd + os.sep + str(test_set) + os.sep)
        tprs, fprs, area = bb.utils.roc(validation, g, n_thresholds=50, save_plots='test', plot=True, save=True,
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
        tprs, fprs, area = bb.utils.roc(validation, g, n_thresholds=50, save_plots='', save = False, plot = False)
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
        ave = bb.utils.average_state(binarized_data[k], n)
        state = ave.copy()
        state[state < 0.5] = 0
        state[state >= 0.5] = 1
        state = [int(i) for i in state]
        idx = bb.utils.state2idx(''.join(["%d" % i for i in state]))
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
        att = bb.utils.idx2binary(average_states[k], n)
        file.write(f"{k}")
        for i in att:
            file.write(f",{i}")
            file_idx.write(f"{i}\n")
        file.write("\n")
        file_idx.close()
    file.close()
    bb.utils.plot_attractors(op.join(dir_prefix, f'{brcd}/average_states.txt'))


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
                            dist = bb.utils.hamming_idx(s, t, n)
                            if dist < min_dist: min_dist = dist
                    distances.append(min_dist)
            try:
                dist_dict[k] = int(np.ceil(np.mean(distances)))
                print(dist_dict[k])

            except ValueError: print("Not enough data in group to find distances.")
        attractor_dict = bb.utils.find_attractors(binarized_data, rules, nodes, regulators_dict, outfile_name, tf_basin=dist_dict, on_nodes=on_nodes, off_nodes=off_nodes)

    else:
        attractor_dict = bb.utils.find_attractors(binarized_data, rules, nodes, regulators_dict, outfile_name, tf_basin=tf_basin, on_nodes=on_nodes, off_nodes=off_nodes)


    file = open(op.join(dir_prefix, f'{brcd}/attractors_unfiltered.txt'), 'w+')
    # plot average state for each subtype
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
    bb.utils.plot_attractors(op.join(dir_prefix, f'{brcd}/attractors_unfiltered.txt'))


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
    file = open(op.join(dir_prefix,f'{brcd}/attractors_filtered.txt'), 'w+')
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
    bb.utils.plot_attractors(op.join(dir_prefix, f'{brcd}/attractors_filtered.txt'))

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
# T = {}
# t2 = time.time()
# T['time'] = (t2 - t1) / 60.
# # How much memory did I use?   Only can use on linux platform
# if os.name == 'posix':
#     T['memory_Mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
# else:
#     T['memory_Mb'] = np.nan
# # T['barcode'] = brcd
# # T['dir_prefix'] = dir_prefix
# # T['network_path'] = network_path
# # T['data_path'] = data_path
# # T['cellID_table'] = cellID_table
# # T['node_normalization'] = node_normalization
# # T['node_threshold'] = node_threshold
#
# T = pd.DataFrame([T])
# if not os.path.isfile(dir_prefix + 'Job_specs_post.csv'):
#     T.to_csv(dir_prefix + 'Job_specs_post.csv')
# else:
#     with open(dir_prefix + 'Job_specs_post.csv', 'a') as f:
#         T.to_csv(f, header=False)


