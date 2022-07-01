import pandas as pd
from graph_tool import all as gt
from graph_tool import GraphView
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from graph_tool.topology import label_components
# from extra import graph_fit_edits
import graph_fit
import seaborn as sns
import os.path as op
import resource
import os
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from src import graph_sim
import pickle
import sklearn.model_selection as ms


def load_network(fname, remove_sources=False, remove_sinks=True, remove_selfloops=True, add_selfloops_to_sources=True,
                 header=None):
    G = gt.Graph()
    infile = pd.read_csv(fname, header=header, dtype="str")
    vertex_dict = dict()
    vertex_names = G.new_vertex_property('string')
    vertex_source = G.new_vertex_property('bool')
    vertex_sink = G.new_vertex_property('bool')
    for tf in set(list(infile[0]) + list(infile[1])):
        v = G.add_vertex()
        vertex_dict[tf] = v
        vertex_names[v] = tf

    for i in infile.index:
        if (not remove_selfloops or infile.loc[i, 0] != infile.loc[i, 1]):
            v1 = vertex_dict[infile.loc[i, 0]]
            v2 = vertex_dict[infile.loc[i, 1]]
            if v2 not in v1.out_neighbors(): G.add_edge(v1, v2)

    G.vertex_properties["name"] = vertex_names

    if (remove_sources or remove_sinks):
        G = prune_network(G, remove_sources=remove_sources, remove_sinks=remove_sinks)
        vertex_dict = dict()
        for v in G.vertices():
            vertex_dict[vertex_names[v]] = v

    for v in G.vertices():
        if v.in_degree() == 0:
            if add_selfloops_to_sources: G.add_edge(v, v)
            vertex_source[v] = True
        else:
            vertex_source[v] = False
        if v.out_degree() == 0:
            vertex_sink[v] = True
        else:
            vertex_sink[v] = False

    G.vertex_properties['sink'] = vertex_sink
    G.vertex_properties['source'] = vertex_source

    return G, vertex_dict


def prune_network(G, remove_sources=True, remove_sinks=False):
    oneStripped = True
    while (oneStripped):

        vfilt = G.new_vertex_property('bool');
        oneStripped = False
        for v in G.vertices():
            if (remove_sources and v.in_degree() == 0) or (remove_sinks and v.out_degree() == 0):
                vfilt[v] = False
                oneStripped = True
            else:
                vfilt[v] = True

        G = GraphView(G, vfilt)
    return G


# Reads dataframe. File must have rows=genes, cols=samples. Returned dataframe is transposed.
# If norm is one of "gmm" (data are normalized via 2-component gaussian mixture model),
# "minmax" (data are linearly normalized from 0(min) to 1(max)) or no normalization is done
def load_data(filename, nodes, log=False, log1p=False, sample_order=None, delimiter=",", norm="gmm", index_col=0,
              transpose = False, fillna = None, fill_missing = True):
    data = pd.read_csv(filename, index_col=index_col, delimiter=delimiter, na_values=['null', 'NULL'])
    if transpose: data = data.transpose()
    if index_col > 0: data = data[data.columns[index_col:]]
    data.index = [str(i).upper() for i in data.index]
    missing_nodes = [i for i in nodes if not i in data.index]
    if fill_missing:
        if len(missing_nodes) > 0: print("Missing nodes: %s" % repr(missing_nodes))
        for i in missing_nodes:
            data.loc[i] = [0]*len(data.columns)
    else:
        if len(missing_nodes) > 0: raise Warning("Missing nodes: %s" % repr(missing_nodes))
    data = data.loc[nodes]

    if log1p:
        data = np.log(data + 1)
    elif log:
        data = np.log(data)

    df = data.transpose()  # Now: rows=samples, columns=genes
    data = pd.DataFrame(index=df.index, columns=nodes, dtype = float)
    for node in nodes:
        if type(df[node]) == pd.Series:
            data[node] = df[node]
        else:
            data[node] = df[node].mean(axis=1)

    if type(norm) == str:
        if norm.lower() == "gmm":
            gm = GaussianMixture(n_components=2)
            for gene in data.columns:
                d = data[gene].values.reshape(data.shape[0], 1)
                gm.fit(d)

                # Figure out which cluster is ON
                idx = 0
                if gm.means_[0][0] < gm.means_[1][0]: idx = 1

                data[gene] = gm.predict_proba(d)[:, idx]
        elif norm.lower() == "minmax":
            data = (data - data.min()) / (data.max() - data.min())
    elif type(norm) == float:
        if norm > 0 and norm < 1:
            lq = data.quantile(q=norm)
            uq = data.quantile(q=1 - norm)
            data = (data - lq) / (uq - lq)
            data[data < 0] = 0
            data[data > 1] = 1
    if fillna is not None:
        data = data.fillna(fillna)

    if sample_order is None:
        cluster_linkage = linkage(data)
        cluster_dendro = dendrogram(cluster_linkage, no_plot=True)
        cluster_leaves = [data.index[i] for i in cluster_dendro['leaves']]
        data = data.loc[cluster_leaves]
    elif type(sample_order) != bool:  # If sample_order=False, don't sort at all
        data = data.loc[sample_order]

    return data


# filenames is a list of filenames, nodes gives the only genes we are reading, log is True/False, or list of [True, False, True...], delimiter is string, or list of strings
def load_data_multiple(filenames, nodes, log=False, delimiter=",", norm="gmm"):
    datasets = []
    for i, filename in enumerate(filenames):
        if type(log) == list:
            log_i = log[i]
        else:
            log_i = log
        if type(delimiter) == list:
            delimiter_i = delimiter[i]
        else:
            delimiter_i = delimiter

        datasets.append(load_data(filename, nodes, log=log_i, sample_order=False, delimiter=delimiter_i, norm=norm))

    data = pd.concat(datasets)

    cluster_linkage = linkage(data)
    cluster_dendro = dendrogram(cluster_linkage, no_plot=True)
    cluster_leaves = [data.index[i] for i in cluster_dendro['leaves']]
    data = data.loc[cluster_leaves]

    return data


def binarize_data(data, phenotype_labels=None, threshold=0.5):
    if phenotype_labels is None:
        binaries = set()
    else:
        binaries = dict()
        for c in phenotype_labels['class'].unique(): binaries[c] = set()

    f = np.vectorize(lambda x: '0' if x < threshold else '1')
    for sample in data.index:
        b = state2idx(''.join(f(data.loc[sample])))

        if phenotype_labels is None:
            binaries.add(b)
        else:
            binaries[phenotype_labels.loc[sample, 'class']].add(b)
    return binaries


def idx2binary(idx, n):
    binary = "{0:b}".format(idx)
    return "0" * (n - len(binary)) + binary


def state2idx(state):
    return int(state, 2)


# Returns 0 if state is []
def state_bool2idx(state):
    n = len(state) - 1
    d = dict({True: 1, False: 0})
    idx = 0
    for s in state:
        idx += d[s] * 2 ** n
        n -= 1
    return idx


# Hamming distance between 2 states
def hamming(x, y):
    s = 0
    for i, j in zip(x, y):
        if i != j: s += 1
    return s


# Hamming distance between 2 states, where binary states are given by decimal code
def hamming_idx(x, y, n):
    return hamming(idx2binary(x, n), idx2binary(y, n))


# Given a graph calculate the graph condensation (all nodes are reduced to strongly
# connected components). Returns the condensation graph, a dictionary mapping
# SCC->[nodes in G], as well as the output of graph_tool's label_components.
# I often use this on the output of graph_sim.prune_stg_edges, or a deterministic stg
def condense(G, directed=True, attractors=True):
    # label_components comes from graph_tool directly
    components = label_components(G, directed=directed, attractors=attractors)
    c_G = gt.Graph()
    c_G.add_vertex(n=len(components[1]))

    vertex_dict = dict()
    for v in c_G.vertices(): vertex_dict[int(v)] = []
    component = components[0]

    for v in G.vertices():
        c = component[v]
        vertex_dict[c].append(v)
        for w in v.out_neighbors():
            cw = component[w]
            if cw == c: continue
            if c_G.edge(c, cw) is None:
                edge = c_G.add_edge(c, cw)
    return c_G, vertex_dict, components


def average_state(idx_list, n):
    av = np.zeros(n)
    for idx in idx_list:
        av = av + np.asarray([float(i) for i in idx2binary(idx, n)]) / (1. * len(idx_list))
    return av


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
            off_leaves, on_leaves = graph_fit.get_leaves_of_regulator(n, i)
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


## Work in progress
def plot_attractors(fname, sep = ','):
    att = pd.read_table(fname, sep=sep, header=0, index_col=0)
    att = att.transpose()
    plt.figure(figsize=(4, 8))
    sns.heatmap(att, cmap='binary', cbar=False, linecolor='w', linewidths=5, square=True,xticklabels = True, yticklabels=True)
    plt.savefig(f"{fname.split('.')[0]}.pdf")

def parent_heatmap(data, regulators_dict, gene):

    regulators = [i for i in regulators_dict[gene]]
    n = len(regulators)

    # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
    heat = np.ones((data.shape[0], 2 ** n))
    for leaf in range(2 ** n):
        binary = idx2binary(leaf, len(regulators))
        binary = [{'0': False, '1': True}[i] for i in binary]
        # binary becomes a list of lists of T and Fs to represent each column
        for i, idx in enumerate(data.index):
            # for each row in data column...
            # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
            df = data.loc[idx]
            val = np.float(data.loc[idx, gene])
            for col, on in enumerate(binary):

                # for each regulator in each column in decision tree...
                regulator = regulators[col]
                # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                # row of data and column of tree with a weight that = probability that that node is on in the data
                # df(regulator) = expression value of regulator in data for that row
                # multiply for each regulator (parent TF) in leaf
                if on:
                    heat[i, leaf] *= np.float(df[regulator])
                else:
                    heat[i, leaf] *= 1 - np.float(df[regulator])

    regulator_order = [i for i in regulators]

    return heat, regulator_order


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def plot_accuracy(data, g, regulators_dict, rules, phenotypes=None, plot_clusters=False, dir_prefix=None,
                  clusters=None, save_plots=None, plot=False, save_df=False, customPalette = sns.color_palette('Set2')):
    try:
        os.mkdir(
            op.join(f'{dir_prefix}', f"{save_plots}"))
    except FileExistsError:
        pass

    h, order = parent_heatmap(data, regulators_dict, g)
    # print("Order",order)
    # print(f"Regulators_dict[{g}]", regulators_dict[g])
    # importance_order = reorder_binary_decision_tree(order, regulators_dict[g])
    rule = rules[g]
    # dot product of weights of test sample and rule will give the predicted value for that sample for that TF
    predicted = (np.dot(h, rule))
    p = pd.DataFrame(predicted, columns=['predicted'], index=data.index)
    p['actual'] = data[g]
    if save_df == True:
        p.to_csv(f'{dir_prefix}/{save_plots}/{g}_validation.csv')
    if plot == True:
        if plot_clusters == True:
            plt.figure()
            predicted = pd.DataFrame(predicted, index=data.index, columns=['predicted'])

            for i in set(clusters['class']):
                clines = data.loc[clusters.loc[clusters['class'] == i].index].index
                sns.scatterplot(x=data.loc[clines][g], y=predicted.loc[clines]['predicted'],
                                label=phenotypes[int(i - 1)])
            plt.xlabel("Actual Normalized Expression")
            plt.ylabel("Predicted Expression from Rule")
            legend_elements = []

            for i, j in enumerate(phenotypes):
                legend_elements.append(Patch(facecolor=customPalette[i], label=j))

            plt.legend(handles=legend_elements, loc='best')
            plt.title(str(g))
            plt.savefig(
                f'{dir_prefix}/{save_plots}/{g}_{save_plots}.pdf')
            plt.close()
        else:
            plt.figure()
            sns.regplot(x=data[g], y=predicted)
            plt.xlabel("Actual Normalized Expression")
            plt.ylabel("Predicted Expression from Rule")
            plt.title(str(g))

            if r2(data[g], predicted) == 0:
                plt.title(str(g))
            else:
                plt.title(str(g) + "\n" + str(round(r2(data[g], predicted), 2)))
            plt.savefig(f'{dir_prefix}/{save_plots}/{g}_{save_plots}.pdf')
            # plt.show()
            plt.close()
    return p

def plot_accuracy_scvelo(data,data_t1, g, regulators_dict, rules, phenotypes=None, plot_clusters=False, dir_prefix=None,
                  clusters=None, save_plots=None, plot=False, save_df=False,customPalette = sns.color_palette('Set2') ):
    try:
        os.mkdir(
            op.join(f'{dir_prefix}', f"{save_plots}"))
    except FileExistsError:
        pass
    try:
        h, order = parent_heatmap(data, regulators_dict, g)
        # print("Order",order)
        # print(f"Regulators_dict[{g}]", regulators_dict[g])
        # importance_order = reorder_binary_decision_tree(order, regulators_dict[g])
        rule = rules[g]
        # dot product of weights of test sample and rule will give the predicted value for that sample for that TF
        predicted = (np.dot(h, rule))
        p = pd.DataFrame(predicted, columns=['predicted'], index=data.index)
        print(len(list(set(p.index).intersection(set(data_t1.index)))))
        p['actual'] = data_t1[g]
        if save_df == True:
            p.to_csv(f'{dir_prefix}/{save_plots}/{g}_validation.csv')
        if plot == True:
            if plot_clusters == True:
                plt.figure()
                predicted = pd.DataFrame(predicted, index=data.index, columns=['predicted'])
                sns.set_palette('Set2')
                for n, c in enumerate(sorted(list(set(clusters['class'])))):

                    clines = data.loc[clusters.loc[clusters['class'] == c].index].index
                    sns.scatterplot(x=data.loc[clines][g], y=predicted.loc[clines]['predicted'],
                                    label=c)
                plt.xlabel("Actual Normalized Expression")
                plt.ylabel("Predicted Expression from Rule")
                legend_elements = []

                for i, j in enumerate(sorted(list(set(clusters['class'])))):
                    legend_elements.append(Patch(facecolor=customPalette[i], label=j))

                plt.legend(handles=legend_elements, loc='best')
                plt.title(str(g))
                plt.savefig(
                    f'{dir_prefix}/{save_plots}/{g}_{save_plots}.pdf')
                plt.close()
            else:
                plt.figure()
                sns.regplot(x=data[g], y=predicted)
                plt.xlabel("Actual Normalized Expression")
                plt.ylabel("Predicted Expression from Rule")
                plt.title(str(g))
                plt.xlim(0,1)
                plt.ylim(0,1)
                if r2(data[g], predicted) == 0:
                    plt.title(str(g))
                else:
                    plt.title(str(g) + "\n" + str(round(r2(data[g], predicted), 2)))
                plt.savefig(f'{dir_prefix}/{save_plots}/{g}_{save_plots}.pdf')
                # plt.show()
                plt.close()
        return p
    except IndexError: print(f"{g} had no parent nodes and cannot be accurately predicted.")

def roc(validation, g, n_thresholds, save_plots, plot=False, save=False, dir_prefix=None):
    tprs = []
    fprs = []
    for i in np.linspace(0, 1, n_thresholds, endpoint=False):
        p, r = calc_roc(validation, i)
        tprs.append(p)
        fprs.append(r)
    # area = auc(fprs, tprs) #### AUC function wasn't working... replace with np.trapz
    area = np.abs(np.trapz(x = fprs, y = tprs))
    if plot == True:
        fig = plt.figure()
        ax = plt.subplot()
        plt.plot(fprs, tprs, '-', marker = 'o')
        plt.title(g + " ROC Curve" + "\n AUC: " + str(round(area,3)))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # ax.plot([0,1], [1,1], ls="--", c=".3")
        # ax.plot([1,1], [0,1], ls="--", c=".2")
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        if save == True:
            plt.savefig(f'{dir_prefix}/{save_plots}/{g}_roc.pdf')
            plt.close()
        else:
            plt.show()
            plt.close()
    return tprs, fprs, area

def calc_roc(validation, threshold):
    # P: True positive over predicted condition positive (of the ones predicted positive, how many are actually
    # positive?)

    # R: True positive over all condition positive (of the actually positive, how many are predicted to be positive?)
    predicted = validation.loc[validation['predicted'] > threshold]
    actual = validation.loc[validation['actual'] > 0.5]
    predicted_neg = validation.loc[validation['predicted'] <= threshold]
    actual_neg = validation.loc[validation['actual'] <= 0.5]
    true_positive = len(set(actual.index).intersection(set(predicted.index)))
    false_positive = len(set(actual_neg.index).intersection(set(predicted.index)))
    true_negative = len(set(actual_neg.index).intersection(set(predicted_neg.index)))
    if len(actual.index.values) == 0 or len(actual_neg.index.values) == 0:
        return -1, -1
    else:
        # print((true_positive+true_negative)/(len(validation)))
        tpr = true_positive / len(actual.index)
        fpr = false_positive / len(actual_neg.index)
        return tpr, fpr

def auc(fpr, tpr): #this function is broken for some reason UGH
    # fpr is x axis, tpr is y axis
    print("Calculating area under discrete ROC curve")
    area = 0
    i_old, j_old = 0, 0
    for c, i in enumerate(fpr):
        j = tpr[c]
        if c == 0:
            i_old = i
            j_old = j
        else:
            area += np.abs(i - i_old) * j_old + .5 * np.abs(i - i_old) * np.abs(j - j_old)
            i_old = i
            j_old = j
    return area

def find_attractors(binarized_data, rules, nodes, regulators_dict, outfile_name, tf_basin, threshold = 0.5, on_nodes = [], off_nodes = []):
    att = dict()
    n = len(nodes)
    for k in binarized_data.keys():
        print(k)
        att[k] = []
        outfile = open(f"{outfile_name}_{k}.txt", 'w+') #if there are no  clusters, comment this out
        outfile.write("start-state,dist-to-start,attractor\n")
        start_states = list(binarized_data[k])
        cnt = 0
        for i in start_states:
            start_states = [i]
            # print(start_states)
            # print("Getting partial STG...")

            # getting entire stg is too costly, so just get stg out to 5 TF neighborhood
            if type(tf_basin) == int:
                if len(on_nodes) == 0 and len(off_nodes) == 0:
                    stg, edge_weights = graph_sim.get_partial_stg(start_states, rules, nodes, regulators_dict, tf_basin)
                else:
                    stg, edge_weights = graph_sim.get_partial_stg(start_states, rules, nodes, regulators_dict, tf_basin,
                                                                  on_nodes=on_nodes, off_nodes=off_nodes)

            elif type(tf_basin) == dict:
                if len(on_nodes) == 0 and len(off_nodes) == 0:
                    stg, edge_weights = graph_sim.get_partial_stg(start_states, rules, nodes, regulators_dict, tf_basin[k])
                else:
                    stg, edge_weights = graph_sim.get_partial_stg(start_states, rules, nodes, regulators_dict, tf_basin[k],
                                                                  on_nodes=on_nodes, off_nodes=off_nodes)
            else: print("tf_basin needs to be an integer or a dictionary of integers for each subtype.")
            # directed stg pruned with threshold .5
            # n = number of nodes that can change (each TF gets chosen with equal probability) EXCEPT nodes that are held ON or OFF (no chance of changing)
            # each edge actually has a probability of being selected * chance of changing
            # print("Pruning STG edges...")
            d_stg = graph_sim.prune_stg_edges(stg, edge_weights, n - len(on_nodes) - len(off_nodes), threshold = threshold)

            # each strongly connected component becomes a single node
            # components[2] tells if its an attractor
            #  components[0] of v tells what components does v belong to
            # print('Condensing STG...')
            c_stg, c_vertex_dict, components = condense(d_stg)

            vidx = stg.vertex_properties['idx']
            # maps graph_tools made up index in partial stg to an index of state that means something to us
            # print("Checking for attractors...")
            for v in stg.vertices():
                # loop through every state in stg and if it's an attractor
                if components[2][components[0][v]]:
                    if v != 0:
                        inspect_state(v, stg,vidx, rules, regulators_dict,nodes,n)
                        outfile.write(f"{start_states[0]},{hamming_idx(vidx[v],start_states[0],n)}, {vidx[v]}\n")
                        print(i, hamming_idx(vidx[v],start_states[0],n), vidx[v])
                        att[k].append(vidx[v])
            cnt += 1
            if cnt%100 == 0: print("...", np.round(cnt/(len(binarized_data[k]))*100, 2), '% done')

        outfile.close()
    for k in att.keys():
        att[k] = list(set(att[k]))
    return att

# look at how likely it is to leave a state
def inspect_state(i, stg, vidx, rules, regulators_dict, nodes, n):
    v = stg.vertex(i)
    for a in zip(nodes, idx2binary(vidx[v],n), graph_sim.get_flip_probs(vidx[v], rules, regulators_dict, nodes)): print(a)
    print(max(graph_sim.get_flip_probs(vidx[v], rules, regulators_dict, nodes)))
    print(sum(graph_sim.get_flip_probs(vidx[v], rules, regulators_dict, nodes)))

def plot_histograms(n_steps_0, n_steps_1, expt_label, bins=20, fname=None, ax=None):
    f, bins = np.histogram(n_steps_0 + n_steps_1, bins=bins)

    frequency_0, steps_0 = np.histogram(n_steps_0, bins=bins)
    density_0 = frequency_0 / (1. * np.sum(frequency_0))
    bin_width_0 = steps_0[1] - steps_0[0]
    gap_0 = bin_width_0 * 0.2

    frequency_1, steps_1 = np.histogram(n_steps_1, bins=bins)
    density_1 = frequency_1 / (1. * np.sum(frequency_1))
    bin_width_1 = steps_1[1] - steps_1[0]
    gap_1 = bin_width_1 * 0.2

    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

    ax.bar(steps_0[:-1] + gap_0 / 4., density_0, width=bin_width_0 - gap_0, color="#4499CC", alpha=0.4,
           label="Control")
    ax.bar(steps_1[:-1] + gap_1 / 4., density_1, width=bin_width_1 - gap_1, color="#CC9944", alpha=0.4,
           label=expt_label)

    ylim = ax.get_ylim()

    avg_0 = np.mean(n_steps_0)
    avg_1 = np.mean(n_steps_1)
    plt.axvline(avg_0,color="#4499CC", linestyle = 'dashed', label = "Control Mean")
    plt.axvline(avg_1,color="#CC9944", linestyle = 'dashed', label = "Perturbation Mean")

    ax.set_ylim(ylim)
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Frequency")
    ax.legend()


    if fname is not None:
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return avg_0, avg_1, ((avg_1 - avg_0) / avg_0)

def random_walks(steady_states, radius, rules, regulators_dict, nodes, dir_prefix, phenotype_name = None, iters=1000, max_steps = 500,
                 perturbations = False, on_nodes = None, off_nodes = None):
    try:
        os.mkdir(op.join(dir_prefix, 'walks'))
    except FileExistsError:
        pass
    try:
        os.mkdir(op.join(dir_prefix, 'perturbations'))
    except FileExistsError:
        pass
    iters = iters
    for start_idx in steady_states: #random walks
        switch_counts_0 = dict()
        for node in nodes: switch_counts_0[node] = 0
        n_steps_to_leave_0 = []
        try:
            os.mkdir(op.join(dir_prefix, 'walks/%d' % start_idx))
        except FileExistsError:
            pass

        outfile = open(op.join(dir_prefix, f"walks/%d/results_radius_{radius}.csv" % start_idx),
                       "w+")
        out_len = open(op.join(dir_prefix, f"walks/%d/len_walks_{radius}.csv" % start_idx), "w+")
        # 1000 iterations; print progress of random walk every 10% of the way
        # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
        # no perturbations
        prog = 0
        for iter_ in range(iters):
            if iter_ / 10 > prog:
                prog = iter_ / 10
                print(prog)
            walk, counts, switches, distances = graph_sim.random_walk_until_leave_basin(start_idx, rules,
                                                                                        regulators_dict, nodes,
                                                                                        radius, max_steps=max_steps)
            n_steps_to_leave_0.append(len(distances))
            for node in switches:
                if node is not None: switch_counts_0[node] += 1
            outfile.write(f"{walk}\n")
            out_len.write(f"{len(walk)}\n")
        outfile.close()
        out_len.close()

        # perturbations
        if perturbations == True:
            if on_nodes == None and off_nodes == None: #run all possible single perturbations
                try:
                    os.mkdir(op.join(dir_prefix, 'perturbations/%d' % start_idx))
                except FileExistsError:
                    pass
                outfile = open(op.join(dir_prefix, f"perturbations/%d/results.csv" % start_idx),
                               "w+")

                for expt_node in nodes:
                    # arrays of # steps when activating or knocking out
                    n_steps_activate = []
                    n_steps_knockout = []
                    prog = 0

                    expt = "%s_activate" % expt_node
                    for iter_ in range(iters):
                        if iter_ / 10 > prog:
                            prog = iter_ / 10
                            print(prog)
                        # to perturb more than one node, add to on_nodes or off_nodes
                        walk_on, counts_on, switches_on, distances_on = graph_sim.random_walk_until_leave_basin(start_idx,
                                                                                                                rules,
                                                                                                                regulators_dict,
                                                                                                                nodes,
                                                                                                                radius,
                                                                                                                max_steps=500,
                                                                                                                on_nodes=[
                                                                                                                    expt_node, ],
                                                                                                                off_nodes=[])

                        n_steps_activate.append(len(distances_on))

                    # mean of non-perturbed vs perturbed: loc_0 and loc_1
                    # histogram plots: inverse gaussian?
                    loc_0, loc_1, stabilized = plot_histograms(n_steps_to_leave_0, n_steps_activate, expt, bins=60,
                                                               fname=op.join(dir_prefix,
                                                                             "perturbations/%d/%s.pdf" % (
                                                                                 start_idx, expt)))
                    #        outfile.write("%s: stablization=%f\n"%(expt, stablized))
                    outfile.write(op.join(dir_prefix, "perturbations/%d,%s,%s,activate,%f\n" % (start_idx, phenotype_name,
                                                                                          expt_node, stabilized)))
                    expt = "%s_knockdown" % expt_node
                    for iter_ in range(iters):
                        walk_off, counts_off, switches_off, distances_off = graph_sim.random_walk_until_leave_basin(
                            start_idx, rules, regulators_dict, nodes, radius,
                            max_steps=500, on_nodes=[], off_nodes=[expt_node, ])

                        n_steps_knockout.append(len(distances_off))

                    loc_0, loc_1, stabilized = plot_histograms(n_steps_to_leave_0, n_steps_knockout, expt, bins=60,
                                                               fname=op.join(dir_prefix,
                                                                             "perturbations/%d/%s.pdf" % (
                                                                                 start_idx,
                                                                                 expt)))
                    outfile.write(
                        op.join(dir_prefix, "perturbations/%d,%s,%s,knockdown,%f\n" % (start_idx, phenotype_name, expt_node,
                                                                                 stabilized)))
                outfile.close()
            else: #run specified perturbation
                try:
                    os.mkdir(op.join(dir_prefix, 'perturbations/%d' % start_idx))
                except FileExistsError:
                    pass
                outfile = open(op.join(dir_prefix, f"perturbations/%d/results_{on_nodes}+_{off_nodes}-.csv" % start_idx),
                               "w+")

                # arrays of # steps when activating or knocking out
                n_steps_activate = []
                n_steps_knockout = []
                prog = 0

                for iter_ in range(iters):
                    if iter_ / 10 > prog:
                        prog = iter_ / 10
                        print(prog)
                    # to perturb more than one node, add to on_nodes or off_nodes
                    walk_on, counts_on, switches_on, distances_on = graph_sim.random_walk_until_leave_basin(
                        start_idx,
                        rules,
                        regulators_dict,
                        nodes,
                        radius,
                        max_steps=500,
                        on_nodes=[
                            on_nodes, ],
                        off_nodes=[
                            off_nodes,
                        ])

                    n_steps_activate.append(len(distances_on))
                expt = f"{on_nodes}+_{off_nodes}-"
                # mean of non-perturbed vs perturbed: loc_0 and loc_1
                # histogram plots: inverse gaussian?
                loc_0, loc_1, stabilized = plot_histograms(n_steps_to_leave_0, n_steps_activate, expt, bins=60,
                                                           fname=op.join(dir_prefix,
                                                                         "perturbations/%d/%s.pdf" % (
                                                                             start_idx, expt)))
                #        outfile.write("%s: stablization=%f\n"%(expt, stablized))
                outfile.write(
                    op.join(dir_prefix, "perturbations/%d,%s,%f\n" % (start_idx, phenotype_name, stabilized)))

                outfile.close()

def split_train_test(data, data_t1, clusters, dir_prefix, fname = None):
    # Split testing and training dataset
    df = list(data.index)

    print("splitting into train and test datasets...")
    kf = ms.StratifiedKFold(n_splits=5)#, random_state=1234)
    train_index, test_index = next(kf.split(df, clusters.loc[df, 'class']))

    T = {'test_cellID': [df[i] for i in test_index], 'test_index': test_index, 'train_index': train_index,
         'train_cellID': [df[i] for i in train_index]}
    with open(f'{dir_prefix}/test_train_indices_{fname}.p', 'wb') as f:
        pickle.dump(T, f)
    test = data.loc[T['test_cellID']]
    data = data.loc[T['train_cellID']]
    test.to_csv(f'{dir_prefix}/test_t0_{fname}.csv')
    data.to_csv(f'{dir_prefix}/train_t0_{fname}.csv')

    if type(data_t1) != type(None):
        test_t1 = data_t1.loc[T['test_cellID']]
        data_t1 = data_t1.loc[T['train_cellID']]
        test_t1.to_csv(f'{dir_prefix}/test_t1_{fname}.csv')
        data_t1.to_csv(f'{dir_prefix}/train_t1_{fname}.csv')
    else:
        test_t1 = None

    clusters_train = clusters.loc[T['train_cellID']]
    clusters_test = clusters.loc[T['test_cellID']]
    clusters_train.to_csv(f'{dir_prefix}/clusters_train_{fname}.csv')
    clusters_test.to_csv(f'{dir_prefix}/clusters_test_{fname}.csv')

    return data, test, data_t1, test_t1, clusters_train, clusters_test
