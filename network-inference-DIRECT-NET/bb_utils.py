import pandas as pd
import os
import resource
import numpy as np
from graph_tool import all as gt
import booleabayes as bb
from graph_tool import GraphView
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
import scipy.stats as ss
import glob
from sklearn.metrics import *
import sklearn.model_selection as ms
import pickle
from collections import Counter
from sklearn.metrics import pairwise_distances
import matplotlib as mpl
from umap import UMAP
from matplotlib.patches import Patch
from statannot import add_stat_annotation
from matplotlib.patches import Patch
from sklearn import decomposition


# use example:
# make_percentage_popd_df(walk_path=f'{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks',starting_attractors = 'Arc_6', num_walks = 100, radius  = 4, attractor_dict, n = len(nodes), perturbation = 'RORB_kd')
def make_percentage_popd_df(
    walk_path,
    starting_attractors,
    num_walks,
    radius,
    attractor_dict,
    n,
    perturbation=None,
):
    columns = list(attractor_dict.keys())
    columns.append("None")
    columns.append("Start")
    columns.append("Walk")
    popd_df = pd.DataFrame(columns=columns)

    reverse_attr_dist = {}
    for k, v in attractor_dict.items():
        for vx in v:
            reverse_attr_dist[vx] = k

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
        if perturbation is None:
            results_file = f"{walk_path}/{start_idx}/results.csv"
        else:
            results_file = f"{walk_path}/{start_idx}/results_{perturbation}.csv"
        with open(results_file, "r") as file:
            line = file.readline()
            cnt = 1
            while line:
                reach_states_dict = {i: 0 for i in attractor_dict.keys()}
                reach_states_dict["None"] = 0
                reach_states_dict["Start"] = start_idx
                reach_states_dict["Walk"] = cnt

                if cnt == 1:
                    pass
                walk = line.strip()
                walk = walk.replace("[", "").replace("]", "").split(",")
                for step, i in enumerate(walk):
                    min_distance = 100
                    for att in reverse_attr_dist.keys():
                        distance = bb.utils.hamming_idx(int(i), int(att), n)
                        if distance < min_distance:
                            min_distance = distance
                            closest_att = reverse_attr_dist[att]
                    if min_distance <= radius:
                        reach_states_dict[closest_att] += 1
                    else:
                        reach_states_dict["None"] += 1
                popd_df = popd_df.append(reach_states_dict, ignore_index=True)
                cnt += 1
                line = file.readline()
                if cnt == num_walks + 1:
                    break

    if perturbation is None:
        outfile = f"{walk_path}/{starting_attractors}_radius_{radius}_percentages.csv"
    else:
        outfile = f"{walk_path}/{starting_attractors}_radius_{radius}_percentages_{perturbation}.csv"
    popd_df.to_csv(outfile, index=False)
    return popd_df


# use example:
# plot_attractors_reached(walk_path, starting_attractors,perturbations = ['RORB_kd','EGR1_kd','MEIS2_kd'],num_walks = 100,radius= 4, length_walks = 2000)
def plot_attractors_reached(
    walk_path,
    starting_attractors,
    attractor_dict,
    perturbations=[],
    num_walks=100,
    radius=4,
    length_walks=2000,
    show=False,
):
    attractor_groups = list(attractor_dict.keys())
    data_unperturbed = pd.read_csv(
        f"{walk_path}/{starting_attractors}_radius_{radius}_percentages.csv",
        header=0,
        index_col=None,
    )

    # Question 1
    ave_unperturbed = data_unperturbed.groupby("Start").mean().drop(["Walk"], axis=1)
    ave_unperturbed_melt = ave_unperturbed.melt(
        value_vars=attractor_groups.append("None"),
        ignore_index=False,
    )
    ave_unperturbed_melt["perturbation"] = "unperturbed"
    combined = ave_unperturbed_melt.copy()

    data_unperturbed_bool = data_unperturbed.copy()
    end_state_columns = data_unperturbed.drop(["Walk", "Start"], axis=1).columns.values

    for c in end_state_columns:
        data_unperturbed_bool[c] = data_unperturbed_bool[c].apply(
            lambda x: 1 if x > 0 else 0
        )
    ave_unperturbed_bool = (
        data_unperturbed_bool.groupby("Start").mean().drop(["Walk"], axis=1)
    )
    ave_unperturbed_bool_melt = ave_unperturbed_bool.melt(
        value_vars=attractor_groups,
        ignore_index=False,
    )
    ave_unperturbed_bool_melt["perturbation"] = "unperturbed"
    combined_bool = ave_unperturbed_bool_melt.copy()

    for perturbation in perturbations:
        data_perturb = pd.read_csv(
            f"{walk_path}/{starting_attractors}_radius_{radius}_percentages_{perturbation}.csv",
            header=0,
            index_col=None,
        )
        ave_perturb = data_perturb.groupby("Start").mean().drop(["Walk"], axis=1)
        ave_perturb_melt = ave_perturb.melt(
            value_vars=attractor_groups,
            ignore_index=False,
        )
        ave_perturb_melt["perturbation"] = perturbation
        combined = pd.concat([combined, ave_perturb_melt], axis=0)

        data_perturb_bool = data_perturb.copy()
        for c in end_state_columns:
            data_perturb_bool[c] = data_perturb_bool[c].apply(
                lambda x: 1 if x > 0 else 0
            )
        ave_perturb_bool = (
            data_perturb_bool.groupby("Start").mean().drop(["Walk"], axis=1)
        )
        ave_perturb_bool_melt = ave_perturb_bool.melt(
            value_vars=attractor_groups,
            ignore_index=False,
        )
        ave_perturb_bool_melt["perturbation"] = perturbation
        combined_bool = pd.concat([combined_bool, ave_perturb_bool_melt], axis=0)

    combined["value"] = combined["value"].apply(lambda x: x * 100 / length_walks)
    combined_bool["value"] = combined_bool["value"].apply(lambda x: x * 100)

    box_pairs = []
    for perturbation in perturbations:
        box_pairs = box_pairs + [
            ((state, "unperturbed"), (state, perturbation))
            for state in combined["variable"].unique()
        ]
    order = (
        combined.groupby("variable")
        .mean()
        .sort_values("value", ascending=False)
        .index.values
    )
    ax = sns.boxplot(
        data=combined,
        x="variable",
        y="value",
        hue="perturbation",
        linewidth=0.5,
        order=order,
    )
    add_stat_annotation(
        ax,
        data=combined,
        x="variable",
        y="value",
        hue="perturbation",
        box_pairs=box_pairs,
        test="t-test_ind",
        loc="inside",
        verbose=1,
        order=order,
    )
    # plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))

    plt.xticks(rotation=90)
    plt.title(
        f"Average Percentage of Time Spent in Each Attractor Basin \n Starting from {starting_attractors} \n Radius = {radius}, Number of Steps = 2000"
    )
    plt.ylabel("% Time Spent in Each Attractor Basin During Walk")
    plt.xlabel("Attractor Basin")
    plt.tight_layout()
    plt.savefig(
        f"{walk_path}/{starting_attractors}_radius_{radius}_{perturbations}_percentages.pdf"
    )
    plt.close()
    if show:
        plt.show()

    box_pairs = []
    for perturbation in perturbations:
        box_pairs = box_pairs + [
            ((state, "unperturbed"), (state, perturbation))
            for state in combined_bool["variable"].unique()
        ]

    order = (
        combined_bool.groupby("variable")
        .mean()
        .sort_values("value", ascending=False)
        .index.values
    )
    ax = sns.boxplot(
        data=combined_bool,
        x="variable",
        y="value",
        hue="perturbation",
        linewidth=0.5,
        order=order,
    )
    add_stat_annotation(
        ax,
        data=combined_bool,
        x="variable",
        y="value",
        hue="perturbation",
        box_pairs=box_pairs,
        test="t-test_ind",
        loc="inside",
        verbose=1,
        order=order,
    )
    plt.xticks(rotation=90)
    plt.ylabel("% Walks that Reach Attractor Basin")
    plt.xlabel("Attractor Basin")
    plt.title(
        f"Average Percentage of Walks that Reach Each Attractor Basin \n Starting from {starting_attractors} Attractors \n Radius = {radius}, Number of Walks = {num_walks}"
    )
    plt.tight_layout()
    plt.savefig(
        f"{walk_path}/{starting_attractors}_radius_{radius}_{perturbations}_reached.pdf"
    )
    plt.close()
    if show:
        plt.show()


def plot_all_random_walks(
    walk_path,
    starting_attractors,
    ATTRACTOR_DIR,
    nodes,
    perturb=None,
    num_walks=20,
    binarized_data=None,
    save_as="",
    fit_to_data=True,
    plot_vs=False,
    show=False,
    reduction="pca",
    set_colors={"Generalist": "grey"},
):
    """
    Visualization of random walks with and without perturbations

    :param walk_path: file path to the walks folder for plotting (usually long_walks subfolder)
    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :param perturb: name of perturbation to plot (suffix of walk results csv files)
    :param ATTRACTOR_DIR: file path to the attractors folder
    :param num_walks: number of walks to plot with lines and kde plot
    :param binarized_data: if fit_to_data is True, this is the binarized data
    :param save_as: suffix on plot file name
    :param fit_to_data: if true, fit the pca to the data instead of only the attractors
    :param plot_vs: if true, plot both the unperturbed and perturbed plot side by side. Note that perturb must be specified.
    :param show: if true, show the plot
    :param reduction: dimensionality reduction method to use. Options are 'pca' and 'umap'
    :param set_colors: dictionary of colors to use for each attractor type (key in attractor_dict); otherwise default seaborn palette is used.
        Can specify single attractor-color mapping to override default palette for that attractor
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
    attr_color_map = make_color_map(attractor_dict.keys(), set_colors=set_colors)

    if reduction == "pca":
        pca = decomposition.PCA(n_components=2)
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
            binarized_data_df_new = umap.fit_transform(binarized_data_df.values)
            att_new = umap.transform(attr_filtered)
        else:
            att_new = umap.fit_transform(attr_filtered)

    data = pd.DataFrame(att_new, columns=["0", "1"])
    data["color"] = [attr_color_map[i] for i in attr_filtered.index]
    # sns.scatterplot(data = data, x = '0',y = '1', hue = 'color')

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

    legend_elements = []

    for i in attr_color_map.keys():
        legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))

    ax1.legend(handles=legend_elements, loc="best")
    ax2.legend(handles=legend_elements, loc="best")

    att2_list = att_list.copy()
    data_walks = pd.DataFrame(columns=["0", "1"])
    data_walks_perturb = pd.DataFrame(columns=["0", "1"])

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
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
                    data_walks = pd.concat([data_walks, data_walk], ignore_index=True)
                    data_walk["color"] = [
                        (len(data_walk.index) - i) / len(data_walk.index)
                        for i in data_walk.index
                    ]
                    # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                    #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                    cnt += 1
                    line = file.readline()
                    if cnt == num_walks:
                        break
            print("Plotting walks with perturbation")
            with open(f"{walk_path}/{start_idx}/results_{perturb}.csv", "r") as file:
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
                    data_walks_perturb = pd.concat(
                        [data_walks_perturb, data_walk], ignore_index=True
                    )
                    data_walk["color"] = [
                        (len(data_walk.index) - i) / len(data_walk.index)
                        for i in data_walk.index
                    ]
                    cnt += 1
                    line = file.readline()
                    if cnt == num_walks:
                        break

        except:
            continue

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

    sns.kdeplot(
        x=data_walks_perturb["0"],
        y=data_walks_perturb["1"],
        shade=True,
        thresh=0.05,
        zorder=1,
        n_levels=20,
        cbar=True,
        color=attr_color_map[starting_attractors],
        ax=ax2,
    )

    # title for left and right plots
    perturbation_name = ""
    if perturb.split("_")[1] == "kd":
        perturbation_name = f"{perturb.split('_')[0]} Knockdown"
    elif perturb.split("_")[1] == "act":
        perturbation_name = f"{perturb.split('_')[0]} Activation"

    archetype_name = f"Archetype {starting_attractors.split('_')[1]}"

    plt.suptitle(
        f"{str(num_walks)} Walks from {archetype_name} \n  with or without perturbation: {perturbation_name}",
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
        plt.savefig(f"{walk_path}/walks_{perturb}_{starting_attractors}{save_as}.png")
        plt.close()


# NOTE: This function will be integrated into booleabayes version > 0.1.9
def make_jaccard_heatmap(
    fname,
    cmap="viridis",
    set_color=dict(),
    clustered=True,
    figsize=(10, 10),
    save=False,
):
    """
    Function to make heatmap of jaccard distance between attractors from dataframe
    """

    # calculate jaccard distance
    df = pd.read_csv(fname, sep=",", header=0, index_col=0)
    jaccard = 1 - pairwise_distances(df.values, metric="jaccard")
    jaccard_df = pd.DataFrame(jaccard, index=df.index, columns=df.index)
    # plot heatmap
    plt.figure(figsize=figsize)
    lut = dict(
        zip(sorted(df.index.unique()), sns.color_palette("hls", len(df.index.unique())))
    )
    lut.update(set_color)
    row_colors = df.index.map(lut)
    if clustered:
        g = sns.clustermap(
            jaccard_df,
            row_colors=row_colors,
            col_colors=row_colors,
            cmap=cmap,
            yticklabels=False,
            xticklabels=False,
        )
    else:
        g = sns.clustermap(
            jaccard_df,
            row_colors=row_colors,
            col_colors=row_colors,
            cmap=cmap,
            row_cluster=False,
            col_cluster=False,
            yticklabels=False,
            xticklabels=False,
        )
    g.fig.suptitle("Jaccard Similarity Between Attractors")
    plt.subplots_adjust(top=0.95)
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(
        handles,
        lut,
        title="Species",
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )
    if save:
        plt.savefig(f"{fname[0:-4]}_jaccard.pdf", dpi=300)
    else:
        plt.show()


# NOTE: This function will be integrated into booleabayes version > 0.1.9
# function to make a matplotlib color map from a list of strings and the matplotlib default color palette
def make_color_map(attractors, palette="hls", set_colors=None):
    """
    Function to make a matplotlib color map from a list of strings and the matplotlib default color palette
    Parameters
    ----------
    list_of_strings : list
        List of strings
    palette : string
        Matplotlib default color palette
    set_colors : dictionary
        Dictionary of colors for each attractor if user wants to specify particular color
    Returns
    -------
    cmap : dictionary
        Dictionary of colors for each string
    """
    # default matplotlib color palette
    palette = sns.color_palette(palette, n_colors=len(attractors))
    palette = palette.as_hex()
    cmap = {}
    for i, s in enumerate(sorted(attractors)):
        if set_colors is not None:
            if s in set_colors:
                cmap[s] = set_colors[s]
            else:
                cmap[s] = palette[i]
    return cmap


# NOTE: This function will be integrated into booleabayes version > 0.1.9


def binarized_umap_transform(binarized_data):
    """
    Function to perform dimensionality reduction on binarized data using UMAP
    :param binarized_data:
    :return:
    """

    # Recalculate the UMAP
    umap = UMAP(n_components=2, metric="jaccard")
    umap_embedding = umap.fit_transform(binarized_data.values)

    return umap_embedding, umap


# NOTE: This function will be integrated into booleabayes version > 0.1.9


def binarized_data_dict_to_binary_df(binarized_data, nodes):
    """
    Function to convert a dictionary of binarized data to a binary dataframe
    Parameters
    ----------
    binarized_data : dictionary
        Dictionary of binarized data
    nodes : list
        List of nodes in the transcription factor network
    Returns
    -------
    binary_df : dataframe
        Binary dataframe of binarized data
    """
    df = pd.DataFrame(columns=nodes)
    for k in binarized_data.keys():
        att = [bb.utils.idx2binary(x, len(nodes)) for x in binarized_data[k]]
        for a in att:
            att_list = [int(i) for i in a]
            df = df.append(pd.DataFrame(att_list, index=nodes, columns=[k]).T)
    return df


# NOTE: This function will be integrated into booleabayes version > 0.1.9


def _long_random_walk(
    start_state,
    rules,
    regulators_dict,
    nodes,
    max_steps=10000,
    on_nodes=[],
    off_nodes=[],
):
    """
    Function to perform random walks on a BooleaBayes network out to max_steps
    The only difference with this function and bb.utils.random_walks_until_leave_basin() is that this function doesn't
    require a radius parameter, and will just keep walking until max_steps.

    Parameters
    ----------
    start_state : int
        Index of attractor to start walk from
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    max_steps : int
        Max number of steps to take in random walks
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    Returns
    -------
    walk : list
        Path of vertices taken during random walk
    Counter(walk) :
        Histogram of walk
    flipped_nodes : list
        Transcription factors that flipped during walk
    distances : list
        Starting state to next step in walk
    """
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [
        {"0": False, "1": True}[i] for i in bb.utils.idx2binary(start_state, n)
    ]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    next_step = start_bool
    next_idx = bb.utils.state_bool2idx(start_bool)
    distance = 0
    distances = []
    step_i = 0
    while step_i < max_steps:
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes:
                continue
            neighbor_idx, flip = bb.utils.update_node(
                rules, regulators_dict, node, node_i, nodes, node_indices, next_step
            )
            r = r - flip**2 / (1.0 * nu)
            if r <= 0:
                next_step = [
                    {"0": False, "1": True}[i]
                    for i in bb.utils.idx2binary(neighbor_idx, n)
                ]
                next_idx = neighbor_idx
                flipped_nodes.append(node)
                distance = bb.utils.hamming(next_step, start_bool)
                break
        if r > 0:
            flipped_nodes.append(None)
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def random_walk_until_reach_any_basin(
    start_state,
    rules,
    regulators_dict,
    nodes,
    basin_dict,
    radius=2,
    max_steps=10000,
    on_nodes=[],
    off_nodes=[],
):
    """
    Parameters
    ----------
    start_state :
        .
    rules: dictionary
        Dictionary of probabilistic rules for the regulators
    regulators_dict : dictionary
        Dictionary of relevant regulators
    nodes : list
        List of nodes in the transcription factor network
    radius : int
        Radius to stay within during walk
    max_steps : int
        Max number of steps to take in random walks
    on_nodes : list
        Define ON nodes of a perturbation
    off_nodes : list
        Define OFF nodes of a perturbation
    basin_dict: dictionary
        Dictionary of attractors by state index (integer). This may just be the attractor_dict.
    Returns
    -------
    walk : list
        Path of vertices taken during random walk
    Counter(walk) :
    flipped_nodes : list

    distances : list
        All distances to basin
    """
    walk = []
    n = len(nodes)
    node_indices = dict(zip(nodes, range(len(nodes))))
    unperturbed_nodes = [i for i in nodes if not (i in on_nodes + off_nodes)]
    nu = len(unperturbed_nodes)
    flipped_nodes = []

    start_bool = [
        {"0": False, "1": True}[i] for i in bb.utils.idx2binary(start_state, n)
    ]
    for i, node in enumerate(nodes):
        if node in on_nodes:
            start_bool[i] = True
        elif node in off_nodes:
            start_bool[i] = False

    next_step = start_bool
    next_idx = bb.utils.state_bool2idx(start_bool)
    distance = 0
    basin = []
    for cluster in basin_dict.keys():
        for b in basin_dict[cluster]:
            basin.append(b)
    # Random high number to be replaced by actual distances
    min_dist = 200
    for i in basin:
        distance = bb.utils.hamming_idx(start_state, i, len(nodes))
        if distance < min_dist:
            min_dist = distance
    distance = min_dist

    distances = []
    step_i = 0
    while distance >= radius and step_i < max_steps:
        r = np.random.rand()
        for node_i, node in enumerate(nodes):
            if node in on_nodes + off_nodes:
                continue
            neighbor_idx, flip = bb.utils.update_node(
                rules, regulators_dict, node, node_i, nodes, node_indices, next_step
            )
            r = r - flip**2 / (1.0 * nu)
            if r <= 0:
                next_step = [
                    {"0": False, "1": True}[i]
                    for i in bb.utils.idx2binary(neighbor_idx, n)
                ]
                next_idx = neighbor_idx
                flipped_nodes.append(node)

                # If basin is a list, loop through all attractors and find the distance to the closest one
                min_dist = 200  # Random high number to be replaced by actual distances
                for i in basin:
                    distance = bb.utils.hamming_idx(next_idx, i, len(nodes))
                    if distance < min_dist:
                        min_dist = distance
                distance = min_dist

                break
        if r > 0:
            flipped_nodes.append(None)
        distances.append(distance)
        walk.append(next_idx)
        step_i += 1
    return walk, Counter(walk), flipped_nodes, distances


def split_train_test_crossval(
    data, data_t1, clusters, save_dir, fname=None, random_state=1234
):
    """Split a dataset into testing and training dataset

    :param data: Dataset or first timepoint of temporal dataset to be split into training/testing datasets
    :type data: Pandas dataframe
    :param data_t1: Second timepoint of temporal dataset, optional
    :type data_t1: {Pandas dataframe, None}
    :param clusters: Cluster assignments for each sample; see ut.get_clusters() to generate
    :type clusters: Pandas DataFrame
    :param save_dir: File path for saving training and testing sets
    :type save_dir: str
    :param fname: Suffix to add to file names for saving, defaults to None
    :type fname: str, optional
    :return: List of dataframes split into training and testing: `data` (training set, t0), test (testing set, t1), data_t1 (training set, t1), test_t (testing set, t1), clusters_train (cluster IDs of training set), clusters_test (cluster IDs of testing set)
    :rtype: Pandas dataframes
    """
    df = list(data.index)

    # print("Splitting into train and test datasets...")
    kf = ms.StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

    idx = 0
    for train_index, test_index in kf.split(df, clusters.loc[df, "class"]):
        # train_index, test_index = next(kf.split(df, clusters.loc[df, 'class']))

        T = {
            "test_cellID": [df[i] for i in test_index],
            "test_index": test_index,
            "train_index": train_index,
            "train_cellID": [df[i] for i in train_index],
        }

        with open(f"{save_dir}/test_train_indices_{fname}_{idx}.p", "wb") as f:
            pickle.dump(T, f)

        test = data.loc[T["test_cellID"]].copy()
        train = data.loc[T["train_cellID"]].copy()
        test.to_csv(f"{save_dir}/test_t0_{fname}_{idx}.csv")
        train.to_csv(f"{save_dir}/train_t0_{fname}_{idx}.csv")

        clusters_train = clusters.loc[T["train_cellID"]]
        clusters_test = clusters.loc[T["test_cellID"]]
        clusters_train.to_csv(f"{save_dir}/clusters_train_{fname}_{idx}.csv")
        clusters_test.to_csv(f"{save_dir}/clusters_test_{fname}_{idx}.csv")
        idx += 1

    # return data, test, data_t1, test_t1, clusters_train, clusters_test


# function to plot 5 plots in one row using matplotlib
def plot_five_plots(df, title, xlabel, ylabel, save=False, show=False, fname=""):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title)
    for i in range(5):
        axs[i].plot(df.iloc[:, i])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(df.columns[i])
    if save:
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def plot_sklearn_summ_stats(summary_stats, VAL_DIR, fname=""):
    df = pd.melt(summary_stats, id_vars="gene")
    df = df.astype({"variable": "string"})
    q1 = df.groupby(df.variable).quantile(0.25)["value"]
    q3 = df.groupby(df.variable).quantile(0.75)["value"]
    outlier_top_lim = q3 + 1.5 * (q3 - q1)
    outlier_bottom_lim = q1 - 1.5 * (q3 - q1)
    plt.figure()
    sns.boxplot(
        x="variable", y="value", data=pd.melt(summary_stats.drop("gene", axis=1))
    )
    col_dict = {
        i: j
        for i, j in zip(
            summary_stats.drop("gene", axis=1).columns,
            range(len(summary_stats.columns) - 1),
        )
    }
    for row in df.itertuples():
        variable = row.variable
        val = row.value
        if (val > outlier_top_lim[variable]) or (val < outlier_bottom_lim[variable]):
            print(val, row.gene)
            plt.annotate(s=row.gene, xy=(col_dict[variable] + 0.1, val), fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model Metric")
    plt.ylabel("Score")
    plt.title("Metrics for BooleaBayes Rule Fitting across All TFs")
    plt.tight_layout()
    plt.savefig(f"{VAL_DIR}/summary_stats_boxplot{fname}.pdf")


def plot_sklearn_metrics(VAL_DIR, show=False, save=True):
    summary_stats = pd.read_csv(f"{VAL_DIR}/summary_stats.csv", header=0, index_col=0)
    if save:
        try:
            os.mkdir(f"{VAL_DIR}/summary_plots")
        except FileExistsError:
            pass
    metric_dict = {
        "accuracy": {"Best": "1", "Worst": "0"},
        "balanced_accuracy_score": {"Best": "1", "Worst": "0"},
        "f1": {"Best": "1", "Worst": "0"},
        "roc_auc_score": {"Best": "1", "Worst": "0"},
        "precision": {"Best": "1", "Worst": "0"},
        "recall": {"Best": "1", "Worst": "0"},
        "explained_variance": {"Best": "1", "Worst": "0"},
        "max_error": {"Best": "0", "Worst": "High"},
        "r2": {"Best": "1", "Worst": "0"},
        "log-loss": {"Best": "Low", "Worst": "High"},
    }
    for c in sorted(list(set(summary_stats.columns).difference({"gene"}))):
        print(c)
        plt.figure(figsize=(20, 8))
        my_order = summary_stats.sort_values(c)["gene"].values
        sns.barplot(data=summary_stats, x="gene", y=c, order=my_order)
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel(
            f"{c.capitalize()} (Best: {metric_dict[c]['Best']}, Worst: {metric_dict[c]['Worst']})"
        )
        plt.title(c.capitalize())
        if show:
            plt.show()
        if save:
            plt.savefig(f"{VAL_DIR}/summary_plots/{c}.pdf")
            plt.close()


def get_sklearn_metrics(VAL_DIR, plot_cm=True, show=False, save=True, save_stats=True):
    files = glob.glob(f"{VAL_DIR}/accuracy_plots/*.csv")
    summary_stats = pd.DataFrame(
        columns=[
            "gene",
            "accuracy",
            "balanced_accuracy_score",
            "f1",
            "roc_auc_score",
            "precision",
            "recall",
            "explained_variance",
            "max_error",
            "r2",
            "log-loss",
        ]
    )
    for f in files:
        val_df = pd.read_csv(f, header=0, index_col=0)
        val_df["actual_binary"] = [
            {True: 1, False: 0}[x] for x in val_df["actual"] > 0.5
        ]
        val_df["predicted_binary"] = [
            {True: 1, False: 0}[x] for x in val_df["predicted"] > 0.5
        ]
        gene = f.split("/")[-1].split("_")[0]

        # classification stats
        acc = accuracy_score(val_df["actual_binary"], val_df["predicted_binary"])
        bal_acc = balanced_accuracy_score(
            val_df["actual_binary"], val_df["predicted_binary"]
        )

        # roc stats
        f1 = f1_score(val_df["actual_binary"], val_df["predicted_binary"])
        roc_auc = roc_auc_score(
            val_df["actual_binary"], val_df["predicted"]
        )  # use prediction probability instead of binary class
        prec = precision_score(val_df["actual_binary"], val_df["predicted_binary"])
        rec = recall_score(val_df["actual_binary"], val_df["predicted_binary"])

        # regression stats
        expl_var = explained_variance_score(val_df["actual"], val_df["predicted"])
        max_err = max_error(val_df["actual"], val_df["predicted"])
        r2 = r2_score(val_df["actual"], val_df["predicted"])
        ll = log_loss(val_df["actual_binary"], val_df["predicted"])

        summary_stats = summary_stats.append(
            pd.Series(
                [gene, acc, bal_acc, f1, roc_auc, prec, rec, expl_var, max_err, r2, ll],
                index=summary_stats.columns,
            ),
            ignore_index=True,
        )
        if plot_cm:
            plt.figure()
            cm = confusion_matrix(val_df["actual_binary"], val_df["predicted_binary"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(gene)
            if show:
                plt.show()
                plt.close()
            if save:
                plt.savefig(f"{VAL_DIR}/accuracy_plots/{gene}_confusion_matrix.pdf")
                plt.close()

    summary_stats = (
        summary_stats.sort_values("gene").reset_index().drop("index", axis=1)
    )
    if save_stats:
        summary_stats.to_csv(f"{VAL_DIR}/summary_stats.csv")
    return summary_stats


def log_job(
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
    validation,
    validation_averages,
    find_average_states,
    find_attractors,
    tf_basin,
    filter_attractors,
    on_nodes,
    off_nodes,
    perturbations,
    stability,
    time=None,
    linux=False,
    memory=False,
    job_barcode=None,
    notes_for_job="",
):
    print("printing job details to Job_specs.csv")
    T = {}
    if memory:
        if linux:
            T["memory_Mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0
        else:
            T["memory_Mb"] = np.nan
    T["barcode"] = brcd
    T["random_state"] = random_state
    T["dir_prefix"] = dir_prefix
    T["network_path"] = network_path
    T["data_path"] = data_path
    T["data_t1_path"] = data_t1_path
    T["cellID_table"] = cellID_table
    T["node_normalization"] = node_normalization
    T["node_threshold"] = node_threshold
    T["split_train_test"] = split_train_test
    T["write_binarized_data"] = write_binarized_data
    T["fit_rules"] = fit_rules
    T["validation"] = validation
    T["validation_averages"] = validation_averages
    T["find_average_states"] = find_average_states
    T["find_attractors"] = find_attractors
    T["tf_basin"] = (
        tf_basin  # if -1, use average distance between clusters. otherwise use the same size basin for all phenotypes
    )
    T["filter_attractors"] = filter_attractors
    T["on_nodes"] = on_nodes
    T["off_nodes"] = off_nodes
    T["total_time"] = time
    T["job_barcode"] = job_barcode
    T["notes_for_job"] = notes_for_job
    T["perturbations"] = perturbations
    T["stability"] = stability

    T = pd.DataFrame([T])
    if not os.path.isfile(dir_prefix + "Job_specs.csv"):
        T.to_csv(dir_prefix + "Job_specs.csv")
    else:
        with open(dir_prefix + "Job_specs.csv", "a") as f:
            T.to_csv(f, header=False)


def print_graph_info(
    graph,
    vertex_dict,
    nodes,
    fname,
    brcd="",
    dir_prefix="",
    plot=True,
    fillcolor="lightcyan",
    gene2color=None,
    layout=None,
    add_edge_weights=True,
    ew_df=None,
):
    print("==================================")
    print("Graph properties")
    print("==================================")
    print(graph)
    # print("Edge and vertex properties: ", graph.list_properties())
    print("Number of nodes:", len(nodes))
    print("Nodes: ", nodes)
    sources = []
    sinks = []
    for i in range(len(nodes)):
        if graph.vp.source[i] == 1:
            sources.append(graph.vp.name[i])
        if graph.vp.sink[i] == 1:
            sinks.append(graph.vp.name[i])
    print("Sources: ", len(sources), sources)
    print("Sinks: ", len(sinks), sinks)

    # treat network as if it is undirected to ensure largest component includes all nodes and edges
    u = gt.extract_largest_component(graph, directed=False)
    print("Network is a single connected component: ", gt.isomorphism(graph, u))
    if gt.isomorphism(graph, u) == False:
        print("\t Largest component of network: ")
        print("\t", u)
    print("Directed acyclic graph: ", gt.is_DAG(graph))
    print("==================================")

    if plot:
        vertex2gene = graph.vertex_properties["name"]
        vertex_colors = fillcolor
        if gene2color is not None:
            vertex_colors = graph.new_vertex_property("string")
            for gene in gene2color.keys():
                vertex_colors[vertex_dict[gene]] = gene2color[gene]
        edge_weights = graph.new_edge_property("float")
        edge_color = graph.new_edge_property("vector<float>")

        for edge in graph.edges():
            edge_weights[edge] = 0.2
            edge_color[edge] = [0, 0, 0, 1]

        if add_edge_weights:
            min_ew = np.min(ew_df["score"])
            max_ew = np.max(ew_df["score"])

            for edge in graph.edges():
                vs, vt = edge.source(), edge.target()
                source = vertex2gene[vs]
                target = vertex2gene[vt]
                w = float(
                    2
                    * (
                        np.mean(
                            ew_df.loc[
                                (ew_df["source"] == source)
                                & (ew_df["target"] == target)
                            ]["score"].values
                        )
                        - min_ew
                    )
                    / max_ew
                    + 0.2
                )
                if np.isnan(w):
                    edge_weights[edge] = 0.2
                    edge_color[edge] = [0, 0, 0, 0.1]
                else:
                    edge_weights[edge] = w
                    edge_color[edge] = [0, 0, 0, (w - 0.2) / 2]

        graph.edge_properties["edge_weights"] = edge_weights
        graph.edge_properties["edge_color"] = edge_color
        vprops = {
            "text": vertex2gene,
            "shape": "circle",
            "size": 20,
            "pen_width": 1,
            "fill_color": vertex_colors,
        }
        eprops = {"color": edge_color}

        if layout == "circle":
            state = gt.minimize_nested_blockmodel_dl(graph, B_min=10)
            state.draw(
                vprops=vprops,
                output=f"{dir_prefix}/{brcd}/{fname}_simple_circle_network.pdf",
                output_size=(1000, 1000),
            )  # mplfig=ax[0,1])
        else:
            pos = gt.sfdp_layout(graph, mu=1, eweight=edge_weights, max_iter=1000)
            gt.graph_draw(
                graph,
                pos=pos,
                vprops=vprops,
                eprops=eprops,
                edge_pen_width=edge_weights,
                output=f"{dir_prefix}/{brcd}/{fname}_simple_network.pdf",
                output_size=(1000, 1000),
            )


def draw_grn(
    G,
    gene2vertex,
    rules,
    regulators_dict,
    fname,
    gene2group=None,
    gene2color=None,
    type="",
    B_min=5,
    save_edge_weights=True,
    edge_weights_fname="edge_weights.csv",
):
    vertex2gene = G.vertex_properties["name"]

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

    edge_weight_df = pd.DataFrame(
        index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys())
    )
    edge_binary_df = pd.DataFrame(
        index=sorted(regulators_dict.keys()), columns=sorted(regulators_dict.keys())
    )

    edge_markers = G.new_edge_property("string")
    edge_weights = G.new_edge_property("float")
    edge_colors = G.new_edge_property("vector<float>")
    for edge in G.edges():
        edge_colors[edge] = [0.0, 0.0, 0.0, 0.3]
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
            off_leaves, on_leaves = bb.utils.get_leaves_of_regulator(n, i)
            if (
                rule[off_leaves].mean() < rule[on_leaves].mean()
            ):  # The regulator is an activator
                edge_colors[edge] = [0.0, 0.3, 0.0, 0.8]
                edge_binary_df.loc[target, source] = 1
            else:
                edge_markers[edge] = "bar"
                edge_colors[edge] = [0.88, 0.0, 0.0, 0.5]
                edge_binary_df.loc[target, source] = -1

            # note: not sure why I added 0.2 to each edge weight.. skewing act larger and inh smaller?
            edge_weights[edge] = rule[on_leaves].mean() - rule[off_leaves].mean() + 0.2
            edge_weight_df.loc[target, source] = (
                rule[on_leaves].mean() - rule[off_leaves].mean()
            )
    G.edge_properties["edge_weights"] = edge_weights
    if save_edge_weights:
        edge_weight_df.to_csv(edge_weights_fname)
    pos = gt.sfdp_layout(
        G, groups=vertex_group, mu=1, eweight=edge_weights, max_iter=1000
    )
    # pos = gt.arf_layout(G, max_iter=100, dt=1e-4)
    eprops = {
        "color": edge_colors,
        "pen_width": 2,
        "marker_size": 15,
        "end_marker": edge_markers,
    }
    vprops = {
        "text": vertex2gene,
        "shape": "circle",
        "size": 20,
        "pen_width": 1,
        "fill_color": vertex_colors,
    }
    if type == "circle":
        state = gt.minimize_nested_blockmodel_dl(G, B_min=B_min)
        state.draw(vprops=vprops, eprops=eprops)  # mplfig=ax[0,1])
    else:
        gt.graph_draw(
            G,
            pos=pos,
            output=fname,
            vprops=vprops,
            eprops=eprops,
            output_size=(1000, 1000),
        )
    return G, edge_weight_df, edge_binary_df


def plot_stability(
    attractor_dict,
    walks_dir,
    palette=sns.color_palette("tab20"),
    rescaled=True,
    show=False,
    save=True,
    err_style="bars",
):

    df = pd.DataFrame(columns=["cluster", "attr", "radius", "mean", "median", "std"])

    colormap = {i: c for i, c in zip(sorted(attractor_dict.keys()), palette)}
    # folders = glob.glob(f"{walks_dir}/[0-9]*")

    for k in sorted(attractor_dict.keys()):
        print(k)
        for attr in attractor_dict[k]:
            folders = glob.glob(f"{walks_dir}/{attr}/len_walks_[0-9]*")
            for f in folders:
                radius = int(f.split("_")[-1].split(".")[0])
                try:
                    lengths = pd.read_csv(f, header=None, index_col=None)
                except pd.errors.EmptyDataError:
                    continue
                df = df.append(
                    pd.Series(
                        [
                            k,
                            attr,
                            radius,
                            np.mean(lengths[0]),
                            np.median(lengths[0]),
                            np.std(lengths[0]),
                        ],
                        index=["cluster", "attr", "radius", "mean", "median", "std"],
                    ),
                    ignore_index=True,
                )

    ## add walk lengths from random control states to df
    if os.path.exists(f"{walks_dir}/random/"):
        colormap["random"] = "lightgrey"
        random_starts = os.listdir(f"{walks_dir}/random/")
        for state in random_starts:
            folders = glob.glob(f"{walks_dir}/random/{state}/len_walks_[0-9]*")
            for f in folders:
                radius = int(f.split("_")[-1].split(".")[0])
                try:
                    lengths = pd.read_csv(f, header=None, index_col=None)
                except pd.errors.EmptyDataError:
                    continue
                df = df.append(
                    pd.Series(
                        [
                            "random",
                            state,
                            radius,
                            np.mean(lengths[0]),
                            np.median(lengths[0]),
                            np.std(lengths[0]),
                        ],
                        index=["cluster", "attr", "radius", "mean", "median", "std"],
                    ),
                    ignore_index=True,
                )
        if rescaled:
            norm_df = df.copy()[["cluster", "attr", "radius", "mean"]]
            df_agg = df.groupby(["cluster", "radius"]).agg("mean")
            norm = df_agg.xs("random", level="cluster")
            for i, r in norm.iterrows():
                norm_df.loc[norm_df["radius"] == i, "mean"] = (
                    norm_df.loc[norm_df["radius"] == i, "mean"] / r["mean"]
                )
            norm_df = norm_df.sort_values(by="cluster")
            plt.figure()
            sns.lineplot(
                x="radius",
                y="mean",
                err_style=err_style,
                hue="cluster",
                palette=colormap,
                data=norm_df,
                markers=True,
            )
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0,
                title="Attractor Subtypes",
            )

            plt.xticks(list(np.unique(norm_df["radius"])))
            plt.xlabel("Radius of Basin")
            plt.ylabel(
                f"Scaled Mean number of steps to leave basin \n (Fold-change from control mean)"
            )
            plt.title("Scaled Stability of Attractors by Subtype")
            plt.tight_layout()
            if show:
                plt.show()
            if save:
                plt.savefig(f"{walks_dir}/scaled_stability_plot.pdf")
                plt.close()

    df = df.sort_values(by="cluster")
    plt.figure()
    sns.lineplot(
        x="radius",
        y="mean",
        err_style=err_style,
        hue="cluster",
        palette=colormap,
        data=df,
        markers=True,
    )
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        title="Attractor Subtypes",
    )

    plt.xticks(list(np.unique(df["radius"])))
    plt.xlabel("Radius of Basin")
    plt.ylabel("Mean number of steps to leave basin")
    plt.title("Stability of Attractors by Subtype")
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(f"{walks_dir}/stability_plot.pdf")
        plt.close()
    return df
