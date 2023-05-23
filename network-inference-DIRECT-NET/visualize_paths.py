# This code is adapted from Dropbox/Grad School/ Quaranta Lab/SCLC/Network/visualize_paths.py
import booleabayes as bb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os.path as op
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import seaborn as sns
from bb_utils import *

np.random.seed(1)

## Set paths
node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True
fname = "combined"
brcd = str(9999)

data_path = f'data/adata_imputed_combined.csv'
data_train_t0_path = f'{brcd}/data_split/train_t0_{fname}.csv'
data_test_t0_path = f'{brcd}/data_split/test_t0_{fname}.csv'
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

cellID_table = 'data/AA_clusters.csv'
cluster_header_list = ["class"]


# attr_color_map = {"Arc_1": "red", "Arc_2": "purple", "Arc_5_Arc_6": "teal", "Arc_4": "orange", "Arc_5": "blue", "Arc_6": "green",
#                   "Generalist":"grey"}


graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)
v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
n = len(nodes)
n_states = 2 ** n

# =============================================================================
# Load the data and clusters
# =============================================================================
print('Reading in data')

#load the data
data_train_t0 = bb.load.load_data(f'{dir_prefix}/{data_train_t0_path}', nodes, norm=node_normalization, delimiter=',',
                             log1p=False, transpose=True, sample_order=False, fillna=0)
data_test_t0 = bb.load.load_data(f'{dir_prefix}/{data_test_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                  log1p=False, transpose=True, sample_order=False, fillna = 0)
clusters = bb.utils.get_clusters(data_train_t0,data_test=data_test_t0, is_data_split=True,
                                                      cellID_table=f"{dir_prefix}/{cellID_table}",
                                                      cluster_header_list=cluster_header_list)
# =============================================================================
# Read in binarized data
# =============================================================================
print('Binarizing data')
save = False

data_t0 = bb.load.load_data(f'{dir_prefix}/{data_path}', nodes, norm=node_normalization,
                            delimiter=',', log1p=False, transpose=transpose,
                            sample_order=False, fillna=0)
binarized_data_t0 = bb.proc.binarize_data(data_t0, phenotype_labels=clusters, save = save,
                                                save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_t0_{fname}')

binarized_data_train_t0 = bb.proc.binarize_data(data_train_t0, phenotype_labels=clusters, save = save,
                                       save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_train_t0_{fname}')

binarized_data_test = bb.proc.binarize_data(data_test_t0, phenotype_labels=clusters, save = save,
                                            save_dir=f"{dir_prefix}/{brcd}/binarized_data",fname=f'binarized_data_test_t0_{fname}')

binarized_data_df = binarized_data_dict_to_binary_df(binarized_data_t0, nodes)
print(binarized_data_df.head())

# This function will visualize random walks starting from a single attractor (each attractor in starting_attractors, which is a key in attractor_dict) and plot the lineplots of the walks. You can specify how many walks to plot.

# NOTE: This function will be integrated into booleabayes version > 0.1.9
def plot_random_walks(walk_path, starting_attractors, ATTRACTOR_DIR,
                      perturb = None,
                      num_walks = 20,
                      binarized_data_df = None,
                      save_as = "",
                      show_lineplots = True,
                      fit_to_data = True,
                      plot_vs = False,
                      show = False,
                      reduction = 'pca',
                      set_colors={'Generalist': 'grey'}):
    """
    Visualization of random walks with and without perturbations

    :param walk_path: file path to the walks folder for plotting (usually long_walks subfolder)
    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :param perturb: name of perturbation to plot (suffix of walk results csv files)
    :param ATTRACTOR_DIR: file path to the attractors folder
    :param num_walks: number of walks to plot with lines and kde plot
    :param binarized_data_df: if fit_to_data is True, this is the binarized data df
    :param save_as: suffix on plot file name
    :param show_lineplots: if true, plot the lineplots of the walks
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
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    n = len(attr_filtered.columns)

    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []
        attractor_bool_dict[i] = []
    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))
        attractor_bool_dict[i].append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))
        att_list.append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))
    attr_color_map = make_color_map(attractor_dict.keys(), set_colors=set_colors)

    if reduction == 'pca':
        pca = PCA(n_components=2)
        if fit_to_data:
            binarized_data_df_new = pca.fit_transform(binarized_data_df)
            att_new = pca.transform(attr_filtered)
        else:
            att_new = pca.fit_transform(attr_filtered)
        comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
        comp = comp.T

        print("Component 1 max and min: ", comp[0].idxmax(), comp[0].max(), comp[0].idxmin(), comp[0].min())
        print("Component 2 max and min: ", comp[1].idxmax(), comp[1].max(), comp[1].idxmin(), comp[1].min())
        print("Explained variance: ", pca.explained_variance_ratio_)
        print("Explained variance sum: ", pca.explained_variance_ratio_.sum())

    elif reduction == 'umap':
        umap = UMAP(n_components=2, metric='jaccard')
        if fit_to_data:
            binarized_data_df_new = umap.fit_transform(binarized_data_df.values)
            att_new = umap.transform(attr_filtered)
        else:
            att_new = umap.fit_transform(attr_filtered)

    data = pd.DataFrame(att_new, columns=['0', '1'])
    data['color'] = [attr_color_map[i] for i in attr_filtered.index]
    # sns.scatterplot(data = data, x = '0',y = '1', hue = 'color')

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
        if plot_vs:
            if perturb is None:
                raise ValueError("If plot_vs is true, perturb must be specified.")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)
            ax1.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)
            ax2.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)

            # sns.scatterplot(data = data, x = '0',y = '2', hue = 'color')
            # plt.show()

            legend_elements = []

            for i in attr_color_map.keys():
                legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))

            ax1.legend(handles=legend_elements, loc='best')
            ax2.legend(handles=legend_elements, loc='best')

            att2_list = att_list.copy()
            data_walks = pd.DataFrame(columns=['0', '1'])
            try:
                print("Plotting walks without perturbation")
                with open(f"{walk_path}/{start_idx}/results.csv", 'r') as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1: pass
                        walk = line.strip()
                        walk = walk.replace('[', '').replace(']', '').split(',')
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == 'pca':
                            walk_new = pca.transform(walk_list)
                        elif reduction == 'umap':
                            walk_new = umap.transform(walk_list)
                        data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                        data_walks = data_walks.append(data_walk)
                        data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                        # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                        #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                        if show_lineplots:
                            sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=.3, dashes=True, legend=False,
                                     alpha=0.4, zorder=2, color = 'black', ax = ax1)
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks: break
                sns.kdeplot(x=data_walks['0'], y=data_walks['1'], shade=True, thresh=0.05, zorder=1, n_levels=20, cbar=True,
                            color=attr_color_map[starting_attractors], ax = ax1)

                #reset data_walks for second half of plot
                data_walks = pd.DataFrame(columns=['0', '1'])

                print("Plotting walks with perturbation")
                with open(f"{walk_path}/{start_idx}/results_{perturb}.csv", 'r') as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1: pass
                        walk = line.strip()
                        walk = walk.replace('[', '').replace(']', '').split(',')
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == 'pca':
                            walk_new = pca.transform(walk_list)
                        elif reduction == 'umap':
                            walk_new = umap.transform(walk_list)

                        data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                        data_walks = data_walks.append(data_walk)
                        data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                        # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                        #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                        if show_lineplots:
                            sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=.3, dashes=True, legend=False,
                                     alpha=0.4, zorder=2, color = 'black', ax = ax2)
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks: break

                sns.kdeplot(x=data_walks['0'], y=data_walks['1'], shade=True, thresh=0.05, zorder=1, n_levels=20, cbar=True,
                            color=attr_color_map[starting_attractors], ax = ax2)

            except:
                continue

            # title for left and right plots
            if perturb.split("_")[1] == 'kd':
                perturbation_name = f"{perturb.split('_')[0]} Knockdown"
            elif perturb.split("_")[1] == 'act':
                perturbation_name = f"{perturb.split('_')[0]} Activation"

            archetype_name = f"Archetype {starting_attractors.split('_')[1]}"

            plt.suptitle(f'{str(num_walks)} Walks from {archetype_name} \n Starting state: {start_idx}  with or without perturbation: {perturbation_name}',
                         size=16)
            ax1.set(title = "No Perturbation")
            ax2.set(title = "With Perturbation")

            # Defining custom 'xlim' and 'ylim' values.
            custom_xlim = (data['0'].min() - 0.3, data['0'].max() + 0.3)
            custom_ylim = (data['1'].min() - 0.3, data['1'].max() + 0.3)

            if reduction == 'pca':
                plt.setp([ax1, ax2], xlim=custom_xlim, ylim=custom_ylim, xlabel='PC 1', ylabel='PC 2')
            elif reduction == 'umap':
                plt.setp([ax1, ax2], xlim=custom_xlim, ylim=custom_ylim, xlabel='UMAP 1', ylabel='UMAP 2')
            # Setting the values for all axes.
            if show:
                plt.show()
            else:
                plt.savefig(f"{walk_path}/{start_idx}/walks_{perturb}_{starting_attractors}{save_as}.png")

        else:
            for start_idx in attractor_dict[starting_attractors]:
                plt.figure(figsize=(12, 10), dpi=300)
                plt.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)
                legend_elements = []
                for i in attr_color_map.keys():
                    legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))
                plt.legend(handles=legend_elements, loc='best')

                att2_list = att_list.copy()
                data_walks = pd.DataFrame(columns=['0', '1'])

                try:
                    if perturb is not None:
                        with open(f"{walk_path}/{start_idx}/results_{perturb}.csv", 'r') as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1: pass
                                walk = line.strip()
                                walk = walk.replace('[', '').replace(']', '').split(',')
                                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == 'pca':
                                    walk_new = pca.transform(walk_list)
                                elif reduction == 'umap':
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                                if show_lineplots:
                                    sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=.3, dashes=True, legend=False,
                                             alpha=0.4, zorder=2, color = 'black')
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks: break
                    else:
                        with open(f"{walk_path}/{start_idx}/results.csv", 'r') as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1: pass
                                walk = line.strip()
                                walk = walk.replace('[', '').replace(']', '').split(',')
                                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == 'pca':
                                    walk_new = pca.transform(walk_list)
                                elif reduction == 'umap':
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                                if show_lineplots:
                                    sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=.3, dashes=True, legend=False,
                                             alpha=0.4, zorder=2, color = 'black')
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks: break

                except:
                    continue

                sns.kdeplot(x = data_walks['0'], y = data_walks['1'], shade=True, thresh = 0.05,zorder=1, n_levels=20,cbar = True,
                            color = attr_color_map[starting_attractors])
                if perturb is not None:
                    plt.title(f'{num_walks} Walks from {starting_attractors} starting state: {start_idx} /n with perturbation: {perturb}')
                else:
                    plt.title(f'{num_walks} Walks from {starting_attractors} starting state: {start_idx}')
                plt.xlim(data['0'].min() - 0.3, data['0'].max() + 0.3)
                plt.ylim(data['1'].min() - 0.3, data['1'].max() + 0.3)
                if reduction == 'pca':
                    plt.xlabel('PC 1')
                    plt.ylabel('PC 2')
                elif reduction == 'umap':
                    plt.xlabel('UMAP 1')
                    plt.ylabel('UMAP 2')
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{walk_path}/{start_idx}/singleplot_walks_{perturb}_{starting_attractors}{save_as}.png")


# walk_path = f'{dir_prefix}/{brcd}/walks/long_walks/1000_step_walks'
# plot_random_walks(walk_path, starting_attractors = 'Arc_6',
#                   ATTRACTOR_DIR = ATTRACTOR_DIR,
#                   attr_color_map = attr_color_map,
#                       perturb = "RORB_kd",
#                       num_walks = 5,
#                       binarized_data_df = binarized_data_df,
#                       save_as = "_data-pca",
#                       show_lineplots = True,
#                       fit_to_data = True,
#                       plot_vs = True,
#                       show = False)

# walk_path = f'{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks'
# plot_random_walks(walk_path, starting_attractors = 'Arc_6',
#                   ATTRACTOR_DIR = ATTRACTOR_DIR,
#                   attr_color_map = attr_color_map,
#                   perturb = "EGR1_kd",#"RORB_kd",
#                   num_walks = 30,
#                   binarized_data_df = binarized_data_df,
#                   save_as = "_data-pca_nolines",
#                   show_lineplots = False,
#                   fit_to_data = True,
#                   plot_vs = True,
#                   show = False)


# This function will visualize random walks as KDE plots starting from all of the attractors for a given attractor_dict key.
# You can specify how many walks to plot. Lineplot is not an option for these plots because they would generate files that are too large.

def plot_all_random_walks(walk_path, starting_attractors, ATTRACTOR_DIR,
                      perturb = None,
                      num_walks = 20,
                      binarized_data_df = None,
                      save_as = "",
                      fit_to_data = True,
                      plot_vs = False,
                      show = False,
                      reduction = 'pca',
                      set_colors={'Generalist': 'grey'}):
    """
    Visualization of random walks with and without perturbations

    :param walk_path: file path to the walks folder for plotting (usually long_walks subfolder)
    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :param perturb: name of perturbation to plot (suffix of walk results csv files)
    :param ATTRACTOR_DIR: file path to the attractors folder
    :param num_walks: number of walks to plot with lines and kde plot
    :param binarized_data_df: if fit_to_data is True, this is the binarized data df
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
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    n = len(attr_filtered.columns)

    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []
        attractor_bool_dict[i] = []
    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))
        attractor_bool_dict[i].append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))
        att_list.append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))
    attr_color_map = make_color_map(attractor_dict.keys(), set_colors=set_colors)

    if reduction == 'pca':
        pca = PCA(n_components=2)
        if fit_to_data:
            binarized_data_df_new = pca.fit_transform(binarized_data_df)
            att_new = pca.transform(attr_filtered)
        else:
            att_new = pca.fit_transform(attr_filtered)
        comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
        comp = comp.T

        print("Component 1 max and min: ", comp[0].idxmax(), comp[0].max(), comp[0].idxmin(), comp[0].min())
        print("Component 2 max and min: ", comp[1].idxmax(), comp[1].max(), comp[1].idxmin(), comp[1].min())
        print("Explained variance: ", pca.explained_variance_ratio_)
        print("Explained variance sum: ", pca.explained_variance_ratio_.sum())

    elif reduction == 'umap':
        umap = UMAP(n_components=2, metric='jaccard')
        if fit_to_data:
            binarized_data_df_new = umap.fit_transform(binarized_data_df.values)
            att_new = umap.transform(attr_filtered)
        else:
            att_new = umap.fit_transform(attr_filtered)

    data = pd.DataFrame(att_new, columns=['0', '1'])
    data['color'] = [attr_color_map[i] for i in attr_filtered.index]
    # sns.scatterplot(data = data, x = '0',y = '1', hue = 'color')

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
        if plot_vs:
            if perturb is None:
                raise ValueError("If plot_vs is true, perturb must be specified.")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)
            ax1.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)
            ax2.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)

            # sns.scatterplot(data = data, x = '0',y = '2', hue = 'color')
            # plt.show()

            legend_elements = []

            for i in attr_color_map.keys():
                legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))

            ax1.legend(handles=legend_elements, loc='best')
            ax2.legend(handles=legend_elements, loc='best')

            att2_list = att_list.copy()
            data_walks = pd.DataFrame(columns=['0', '1'])
            try:
                print("Plotting walks without perturbation")
                with open(f"{walk_path}/{start_idx}/results.csv", 'r') as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1: pass
                        walk = line.strip()
                        walk = walk.replace('[', '').replace(']', '').split(',')
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == 'pca':
                            walk_new = pca.transform(walk_list)
                        elif reduction == 'umap':
                            walk_new = umap.transform(walk_list)
                        data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                        data_walks = data_walks.append(data_walk)
                        data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                        # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                        #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks: break
                sns.kdeplot(x=data_walks['0'], y=data_walks['1'], shade=True, thresh=0.05, zorder=1, n_levels=20, cbar=True,
                            color=attr_color_map[starting_attractors], ax = ax1)

                #reset data_walks for second half of plot
                data_walks = pd.DataFrame(columns=['0', '1'])

                print("Plotting walks with perturbation")
                with open(f"{walk_path}/{start_idx}/results_{perturb}.csv", 'r') as file:
                    line = file.readline()
                    cnt = 1
                    while line:
                        if cnt == 1: pass
                        walk = line.strip()
                        walk = walk.replace('[', '').replace(']', '').split(',')
                        walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                        walk_list = []
                        for i in walk_states:
                            walk_list.append([int(j) for j in i])
                            att2_list.append([int(j) for j in i])
                        if reduction == 'pca':
                            walk_new = pca.transform(walk_list)
                        elif reduction == 'umap':
                            walk_new = umap.transform(walk_list)

                        data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                        data_walks = data_walks.append(data_walk)
                        data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                        cnt += 1
                        line = file.readline()
                        if cnt == num_walks: break

                sns.kdeplot(x=data_walks['0'], y=data_walks['1'], shade=True, thresh=0.05, zorder=1, n_levels=20, cbar=True,
                            color=attr_color_map[starting_attractors], ax = ax2)

            except:
                continue

            # title for left and right plots
            perturbation_name = ""
            if perturb.split("_")[1] == 'kd':
                perturbation_name = f"{perturb.split('_')[0]} Knockdown"
            elif perturb.split("_")[1] == 'act':
                perturbation_name = f"{perturb.split('_')[0]} Activation"

            archetype_name = f"Archetype {starting_attractors.split('_')[1]}"

            plt.suptitle(f'{str(num_walks)} Walks from {archetype_name} \n Starting state: {start_idx}  with or without perturbation: {perturbation_name}',
                         size=16)
            ax1.set(title = "No Perturbation")
            ax2.set(title = "With Perturbation")

            # Defining custom 'xlim' and 'ylim' values.
            custom_xlim = (data['0'].min() - 0.3, data['0'].max() + 0.3)
            custom_ylim = (data['1'].min() - 0.3, data['1'].max() + 0.3)

            if reduction == 'pca':
                plt.setp([ax1, ax2], xlim=custom_xlim, ylim=custom_ylim, xlabel='PC 1', ylabel='PC 2')
            elif reduction == 'umap':
                plt.setp([ax1, ax2], xlim=custom_xlim, ylim=custom_ylim, xlabel='UMAP 1', ylabel='UMAP 2')
            # Setting the values for all axes.
            if show:
                plt.show()
            else:
                plt.savefig(f"{walk_path}/{start_idx}/walks_{perturb}_{starting_attractors}{save_as}.png")

        else:
            for start_idx in attractor_dict[starting_attractors]:
                plt.figure(figsize=(12, 10), dpi=300)
                plt.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)
                legend_elements = []
                for i in attr_color_map.keys():
                    legend_elements.append(Patch(facecolor=attr_color_map[i], label=i))
                plt.legend(handles=legend_elements, loc='best')

                att2_list = att_list.copy()
                data_walks = pd.DataFrame(columns=['0', '1'])

                try:
                    if perturb is not None:
                        with open(f"{walk_path}/{start_idx}/results_{perturb}.csv", 'r') as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1: pass
                                walk = line.strip()
                                walk = walk.replace('[', '').replace(']', '').split(',')
                                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == 'pca':
                                    walk_new = pca.transform(walk_list)
                                elif reduction == 'umap':
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks: break
                    else:
                        with open(f"{walk_path}/{start_idx}/results.csv", 'r') as file:
                            line = file.readline()
                            cnt = 1
                            while line:
                                if cnt == 1: pass
                                walk = line.strip()
                                walk = walk.replace('[', '').replace(']', '').split(',')
                                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                                walk_list = []
                                for i in walk_states:
                                    walk_list.append([int(j) for j in i])
                                    att2_list.append([int(j) for j in i])
                                if reduction == 'pca':
                                    walk_new = pca.transform(walk_list)
                                elif reduction == 'umap':
                                    walk_new = umap.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                cnt += 1
                                line = file.readline()
                                if cnt == num_walks: break

                except:
                    continue

                sns.kdeplot(x = data_walks['0'], y = data_walks['1'], shade=True, thresh = 0.05,zorder=1, n_levels=20,cbar = True,
                            color = attr_color_map[starting_attractors])
                if perturb is not None:
                    plt.title(f'{num_walks} Walks from {starting_attractors} starting state: {start_idx} /n with perturbation: {perturb}')
                else:
                    plt.title(f'{num_walks} Walks from {starting_attractors} starting state: {start_idx}')
                plt.xlim(data['0'].min() - 0.3, data['0'].max() + 0.3)
                plt.ylim(data['1'].min() - 0.3, data['1'].max() + 0.3)
                if reduction == 'pca':
                    plt.xlabel('PC 1')
                    plt.ylabel('PC 2')
                elif reduction == 'umap':
                    plt.xlabel('UMAP 1')
                    plt.ylabel('UMAP 2')
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{walk_path}/{start_idx}/singleplot_walks_{perturb}_{starting_attractors}{save_as}.png")

