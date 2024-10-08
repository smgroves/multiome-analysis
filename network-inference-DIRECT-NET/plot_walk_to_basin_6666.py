# Running plot walk to basin interactively for network 6666 to troubleshoot problems with plotting
# %% Imports
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
from sklearn.decomposition import PCA

customPalette = sns.color_palette("tab10")
# %% Load graph and data
# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
print_graph_information = True  # whether to print graph info to {brcd}.txt
plot_network = False
split_train_test = False
write_binarized_data = False
fit_rules = False
run_validation = False
validation_averages = False
find_average_states = False
find_attractors = False
tf_basin = (
    2  # if -1, use average distance between clusters for search basin for attractors.
)
# otherwise use the same size basin for all phenotypes. For single cell data, there may be so many samples that average distance is small.
filter_attractors = False
perturbations = False
stability = False
walk_to_basin = False
plot_walk_to_basin = True
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
notes_for_log = "Walk to attractors for updated DIRECT-NET network with 2020db and indpendent LASSO models, wo sinks"
# notes_for_log = "Perturbations for updated DIRECT-NET network with 2020db and top 8 regulators"
## Set paths
dir_prefix = (
    "/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
)
network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks.csv"
# network_path = "networks/DIRECT-NET_network_2020db_0.1_top8regs_wo_sinks.csv"
data_path = "data/adata_imputed_combined_v3.csv"
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
# brcd = str(random.randint(0,99999))
brcd = str(6666)  # correspond to LASSO network
# brcd = 1112 #corresponds to updated 2020db network 0.1 with manually selected top 8 regulators
print(brcd)
# if rerunning a brcd and data has already been split into training and testing sets, use the below code
# Otherwise, these settings are ignored
data_train_t0_path = f"{brcd}/data_split/train_t0_{fname}.csv"
data_train_t1_path = None  # if no T1, replace with None
data_test_t0_path = f"{brcd}/data_split/test_t0_{fname}.csv"
data_test_t1_path = None  # if no T1, replace with None

# data_train_t0_path = data_path
# data_test_t0_path = data_path

## Set job barcode and random_state
# temp = sys.stdout

job_brcd = str(
    random.randint(0, 99999)
)  # use a job brcd to keep track of multiple jobs for the same brcd
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
if not network_path.endswith(".csv") or not os.path.isfile(dir_prefix + network_path):
    raise Exception("Network path must be a .csv file.  Check file name and location")
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
        data_t0, data_t1, clusters, f"{dir_prefix}/{brcd}/data_split", fname=fname
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
        rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules_scvelo(
            data=data_train_t0,
            data_t1=data_train_t1,
            vertex_dict=vertex_dict,
            plot=False,
            threshold=node_threshold,
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

        if plot_network:
            draw_grn(
                graph,
                vertex_dict,
                rules,
                regulators_dict,
                f"{dir_prefix}/{brcd}/{fname}_network.pdf",
                save_edge_weights=True,
                edge_weights_fname=f"{dir_prefix}/{brcd}/rules/edge_weights.csv",
            )  # , gene2color = gene2color)
    except FileNotFoundError:
        print("Rules file not found. Please set fit_rules to True to generate rules.")
# =================
# %% attractors
attr_color_map = {
    "Arc_1": "red",
    "Arc_2": "purple",
    "Arc_5_Arc_6": "teal",
    "Arc_4": "orange",
    "Arc_5": "blue",
    "Arc_6": "green",
    "Generalist_NE": "darkgrey",
    "Generalist_nonNE": "lightgrey",
}
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"
attractor_dict = bb.utils.get_attractor_dict(ATTRACTOR_DIR, filtered=True)

# "Arc_5",
# "Generalist_NE",
# "Arc_3",
# "Generalist_nonNE",

# %% function


def plot_random_walks(
    walk_path,
    starting_attractors,
    ATTRACTOR_DIR,
    nodes,
    perturb=None,
    num_walks=20,
    binarized_data=None,
    save_as="",
    show_lineplots=True,
    fit_to_data=True,
    plot_vs=False,
    show=False,
    reduction="pca",
    set_colors=None,
):
    """
    Visualization of random walks with and without perturbations

    :param walk_path: file path to the walks folder for plotting (usually long_walks subfolder)
    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :param perturb: name of perturbation to plot (suffix of walk results csv files)
    :param ATTRACTOR_DIR: file path to the attractors folder
    :param nodes: list of nodes in the network
    :param num_walks: number of walks to plot with lines and kde plot
    :param binarized_data: if fit_to_data is True, this is the binarized data as a dictionary
    :param save_as: suffix on plot file name
    :param show_lineplots: if true, plot the lineplots of the walks
    :param fit_to_data: if true, fit the pca to the data instead of only the attractors
    :param plot_vs: if true, plot both the unperturbed and perturbed plot side by side. Note that perturb must be specified.
    :param show: if true, show the plot
    :param reduction: dimensionality reduction method to use. Options are 'pca' and 'umap'
    :param set_colors: dictionary of colors to use for each attractor type (key in attractor_dict); otherwise default seaborn palette is used.
        Can specify single attractor-color mapping to override default palette for that attractor, for example, {'Generalist': 'grey'}
    :return:
    """
    attractor_dict = {}
    attractor_bool_dict = {}
    att_list = []
    attr_filtered = pd.read_csv(
        f"{ATTRACTOR_DIR}/attractors_filtered.txt", sep=",", header=0, index_col=0
    )
    n = len(attr_filtered.columns)

    for i, r in attr_filtered.iterrows():
        attractor_dict[i] = []
        attractor_bool_dict[i] = []
    for i, r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))
        attractor_bool_dict[i].append(
            int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n))
        )
        att_list.append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))

    attr_color_map = bb.utils.make_color_map(
        attractor_dict.keys(), set_colors=set_colors
    )

    if reduction == "pca":
        pca = PCA(n_components=2)
        if fit_to_data:
            binarized_data_df = bb.utils.binarized_data_dict_to_binary_df(
                binarized_data, nodes
            )
            binarized_data_df_new = pca.fit_transform(binarized_data_df)
            att_new = pca.transform(attr_filtered)
        else:
            att_new = pca.fit_transform(attr_filtered)
        comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
        comp = comp.T

        print(
            "Component 1 max and min: ",
            comp[0].idxmax(),
            comp[0].max(),
            comp[0].idxmin(),
            comp[0].min(),
        )
        print(
            "Component 2 max and min: ",
            comp[1].idxmax(),
            comp[1].max(),
            comp[1].idxmin(),
            comp[1].min(),
        )
        print("Explained variance: ", pca.explained_variance_ratio_)
        print("Explained variance sum: ", pca.explained_variance_ratio_.sum())

    elif reduction == "umap":
        umap = UMAP(n_components=2, metric="jaccard")
        if fit_to_data:
            binarized_data_df = bb.utils.binarized_data_dict_to_binary_df(
                binarized_data, nodes
            )
            binarized_data_df_new = umap.fit_transform(binarized_data_df.values)
            att_new = umap.transform(attr_filtered)
        else:
            att_new = umap.fit_transform(attr_filtered)

    data = pd.DataFrame(att_new, columns=["0", "1"])
    data["color"] = [attr_color_map[i] for i in attr_filtered.index]
    # sns.scatterplot(data = data, x = '0',y = '1', hue = 'color')

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
        if plot_vs:
            if perturb is None:
                raise ValueError("If plot_vs is true, perturb must be specified.")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)
            ax1.scatter(
                x=data["0"],
                y=data["1"],
                c=data["color"],
                s=100,
                edgecolors="k",
                zorder=4,
            )
            ax2.scatter(
                x=data["0"],
                y=data["1"],
                c=data["color"],
                s=100,
                edgecolors="k",
                zorder=4,
            )

            # sns.scatterplot(data = data, x = '0',y = '2', hue = 'color')
            # plt.show()

            legend_elements = []

            for i in attr_color_map.keys():
                legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))

            ax1.legend(handles=legend_elements, loc="best")
            ax2.legend(handles=legend_elements, loc="best")

            att2_list = att_list.copy()
            data_walks = pd.DataFrame(columns=["0", "1"])
            try:
                print("Plotting walks without perturbation")
                with open(f"{walk_path}/{start_idx}/results.csv", "r") as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1:
                            pass
                        walk = line.strip()
                        walk = walk.replace("[", "").replace("]", "").split(",")
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == "pca":
                            walk_new = pca.transform(walk_list)
                        elif reduction == "umap":
                            walk_new = umap.transform(walk_list)
                        data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                        data_walks = pd.concat(
                            [data_walks, data_walk], ignore_index=True
                        )
                        data_walk["color"] = [
                            (len(data_walk.index) - i) / len(data_walk.index)
                            for i in data_walk.index
                        ]
                        # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                        #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                        if show_lineplots:
                            sns.lineplot(
                                x=data_walk["0"],
                                y=data_walk["1"],
                                lw=0.3,
                                dashes=True,
                                legend=False,
                                alpha=0.4,
                                zorder=2,
                                color="black",
                                ax=ax1,
                            )
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks:
                            break
                sns.kdeplot(
                    x=data_walks["0"],
                    y=data_walks["1"],
                    shade=True,
                    thresh=0.05,
                    zorder=1,
                    n_levels=20,
                    cbar=True,
                    color=attr_color_map[starting_attractors],
                    ax=ax1,
                )

                # reset data_walks for second half of plot
                data_walks = pd.DataFrame(columns=["0", "1"])

                print("Plotting walks with perturbation")
                with open(
                    f"{walk_path}/{start_idx}/results_{perturb}.csv", "r"
                ) as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1:
                            pass
                        walk = line.strip()
                        walk = walk.replace("[", "").replace("]", "").split(",")
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == "pca":
                            walk_new = pca.transform(walk_list)
                        elif reduction == "umap":
                            walk_new = umap.transform(walk_list)

                        data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                        data_walks = pd.concat(
                            [data_walks, data_walk], ignore_index=True
                        )
                        data_walk["color"] = [
                            (len(data_walk.index) - i) / len(data_walk.index)
                            for i in data_walk.index
                        ]
                        # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                        #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                        if show_lineplots:
                            sns.lineplot(
                                x=data_walk["0"],
                                y=data_walk["1"],
                                lw=0.3,
                                dashes=True,
                                legend=False,
                                alpha=0.4,
                                zorder=2,
                                color="black",
                                ax=ax2,
                            )
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks:
                            break

                sns.kdeplot(
                    x=data_walks["0"],
                    y=data_walks["1"],
                    shade=True,
                    thresh=0.05,
                    zorder=1,
                    n_levels=20,
                    cbar=True,
                    color=attr_color_map[starting_attractors],
                    ax=ax2,
                )

            except:
                continue

            # title for left and right plots
            if perturb.split("_")[1] == "kd":
                perturbation_name = f"{perturb.split('_')[0]} Knockdown"
            elif perturb.split("_")[1] == "act":
                perturbation_name = f"{perturb.split('_')[0]} Activation"

            archetype_name = f"Archetype {starting_attractors.split('_')[1]}"

            plt.suptitle(
                f"{str(num_walks)} Walks from {archetype_name} \n Starting state: {start_idx}  with or without perturbation: {perturbation_name}",
                size=16,
            )
            ax1.set(title="No Perturbation")
            ax2.set(title="With Perturbation")

            # Defining custom 'xlim' and 'ylim' values.
            custom_xlim = (data["0"].min() - 0.3, data["0"].max() + 0.3)
            custom_ylim = (data["1"].min() - 0.3, data["1"].max() + 0.3)

            if reduction == "pca":
                plt.setp(
                    [ax1, ax2],
                    xlim=custom_xlim,
                    ylim=custom_ylim,
                    xlabel="PC 1",
                    ylabel="PC 2",
                )
            elif reduction == "umap":
                plt.setp(
                    [ax1, ax2],
                    xlim=custom_xlim,
                    ylim=custom_ylim,
                    xlabel="UMAP 1",
                    ylabel="UMAP 2",
                )
            # Setting the values for all axes.
            if show:
                plt.show()
            else:
                plt.savefig(
                    f"{walk_path}/{start_idx}/walks_{perturb}_{starting_attractors}{save_as}.png"
                )
                plt.close()

        else:
            for start_idx in attractor_dict[starting_attractors]:
                plt.figure(figsize=(12, 10), dpi=300)
                plt.scatter(
                    x=data["0"],
                    y=data["1"],
                    c=data["color"],
                    s=100,
                    edgecolors="k",
                    zorder=4,
                )
                legend_elements = []
                for i in attr_color_map.keys():
                    legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))
                plt.legend(handles=legend_elements, loc="best")

                att2_list = att_list.copy()
                data_walks = pd.DataFrame(columns=["0", "1"])

                try:
                    if perturb is not None:
                        with open(
                            f"{walk_path}/{start_idx}/results_{perturb}.csv", "r"
                        ) as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1:
                                    pass
                                walk = line.strip()
                                walk = walk.replace("[", "").replace("]", "").split(",")
                                walk_states = [
                                    bb.utils.idx2binary(int(i), n) for i in walk
                                ]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == "pca":
                                    walk_new = pca.transform(walk_list)
                                elif reduction == "umap":
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                                data_walks = pd.concat(
                                    [data_walks, data_walk], ignore_index=True
                                )
                                data_walk["color"] = [
                                    (len(data_walk.index) - i) / len(data_walk.index)
                                    for i in data_walk.index
                                ]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                                if show_lineplots:
                                    sns.lineplot(
                                        x=data_walk["0"],
                                        y=data_walk["1"],
                                        lw=0.3,
                                        dashes=True,
                                        legend=False,
                                        alpha=0.4,
                                        zorder=2,
                                        color="black",
                                    )
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks:
                                    break
                    else:
                        with open(f"{walk_path}/{start_idx}/results.csv", "r") as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1:
                                    pass
                                walk = line.strip()
                                walk = walk.replace("[", "").replace("]", "").split(",")
                                walk_states = [
                                    bb.utils.idx2binary(int(i), n) for i in walk
                                ]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == "pca":
                                    walk_new = pca.transform(walk_list)
                                elif reduction == "umap":
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=["0", "1"])
                                data_walks = pd.concat(
                                    [data_walks, data_walk], ignore_index=True
                                )
                                data_walk["color"] = [
                                    (len(data_walk.index) - i) / len(data_walk.index)
                                    for i in data_walk.index
                                ]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                                if show_lineplots:
                                    sns.lineplot(
                                        x=data_walk["0"],
                                        y=data_walk["1"],
                                        lw=0.3,
                                        dashes=True,
                                        legend=False,
                                        alpha=0.4,
                                        zorder=2,
                                        color="black",
                                    )
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks:
                                    break

                except:
                    continue
                print(data_walks.head())
                sns.kdeplot(
                    x=data_walks["0"],
                    y=data_walks["1"],
                    shade=True,
                    thresh=0.05,
                    zorder=1,
                    n_levels=20,
                    cbar=True,
                    color=attr_color_map[starting_attractors],
                )
                if perturb is not None:
                    plt.title(
                        f"{num_walks} Walks from {starting_attractors} starting state: {start_idx} /n with perturbation: {perturb}"
                    )
                else:
                    plt.title(
                        f"{num_walks} Walks from {starting_attractors} starting state: {start_idx}"
                    )
                plt.xlim(data["0"].min() - 0.3, data["0"].max() + 0.3)
                plt.ylim(data["1"].min() - 0.3, data["1"].max() + 0.3)
                if reduction == "pca":
                    plt.xlabel("PC 1")
                    plt.ylabel("PC 2")
                elif reduction == "umap":
                    plt.xlabel("UMAP 1")
                    plt.ylabel("UMAP 2")
                if show:
                    plt.show()
                else:
                    plt.savefig(
                        f"{walk_path}/{start_idx}/singleplot_walks_{perturb}_{starting_attractors}{save_as}.png"
                    )
                    plt.close()


# %%
walk_path = f"{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks"
starting_attractors = "Generalist_NE"
plot_random_walks(
    walk_path,
    starting_attractors,
    ATTRACTOR_DIR=ATTRACTOR_DIR,
    nodes=nodes,
    perturb="RORA_kd",
    num_walks=5,
    binarized_data=binarized_data_t0,
    save_as="_data-pca",
    show_lineplots=True,
    fit_to_data=True,
    plot_vs=True,
    show=False,
    set_colors={"Generalist_NE": "darkgrey", "Generalist_nonNE": "lightgrey"},
)

# %%
# walk_path = f"{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks"
# starting_attractors = "Arc_5"
# plot_all_random_walks(
#     walk_path,
#     starting_attractors,
#     ATTRACTOR_DIR,
#     perturb="RORA_kd",
#     num_walks=5,
#     binarized_data=binarized_data_t0,
#     nodes=nodes,
#     save_as="",
#     fit_to_data=True,
#     plot_vs=True,
#     show=True,
#     reduction="pca",
#     set_colors={"Generalist_NE": "darkgrey", "Generalist_nonNE": "lightgrey"},
# )

# %%
