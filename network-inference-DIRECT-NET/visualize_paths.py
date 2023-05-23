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
cellID_table = 'data/AA_clusters.csv'
cluster_header_list = ["class"]

attr_color_map = {"Arc_1": "red", "Arc_2": "purple", "Arc_5_Arc_6": "teal", "Arc_4": "orange", "Arc_5": "blue", "Arc_6": "green",
                  "Generalist":"grey"}


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
# =============================================================================
# Load the attractors
# =============================================================================
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

def plot_random_walks(walk_path, starting_attractors, ATTRACTOR_DIR, attr_color_map,
                      perturb = None,
                      num_walks = 20,
                      binarized_data_df = None,
                      save_as = "",
                      show_lineplots = True,
                      fit_to_data = True,
                      plot_vs = False,
                      show = False):
    """
    Visualization of random walks with and without perturbations

    :param walk_path: file path to the walks folder for plotting (usually long_walks subfolder)
    :param starting_attractors: name of the attractors to start the walk from (key in attractor_dict)
    :param perturb: name of perturbation to plot (suffix of walk results csv files)
    :param ATTRACTOR_DIR: file path to the attractors folder
    :param attr_color_map: dictionary mapping attractor names to colors
    :param num_walks: number of walks to plot with lines and kde plot
    :param binarized_data_df: if fit_to_data is True, this is the binarized data df
    :param save_as: suffix on plot file name
    :param show_lineplots: if true, plot the lineplots of the walks
    :param fit_to_data: if true, fit the pca to the data instead of only the attractors
    :param plot_vs: if true, plot both the unperturbed and perturbed plot side by side. Note that perturb must be specified.
    :param show: if true, show the plot
    :return:
    """
    attractor_dict = {}
    attractor_bool_dict = {}
    att_list = []
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    n = len(attr_filtered.columns)
    print(n)

    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []
        attractor_bool_dict[i] = []
    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))
        attractor_bool_dict[i].append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))
        att_list.append(int(bb.utils.idx2binary(bb.utils.state_bool2idx(list(r)), n)))

    pca = PCA(n_components=2)
    if fit_to_data:
        binarized_data_df_new = pca.fit_transform(binarized_data_df)
        att_new = pca.transform(attr_filtered)
    else:
        att_new = pca.fit_transform(attr_filtered)

    data = pd.DataFrame(att_new, columns=['0', '1'])
    comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
    comp = comp.T

    print("Component 1 max and min: ", comp[0].idxmax(), comp[0].max(), comp[0].idxmin(), comp[0].min())
    print("Component 2 max and min: ", comp[1].idxmax(), comp[1].max(), comp[1].idxmin(), comp[1].min())
    print("Explained variance: ", pca.explained_variance_ratio_)
    print("Explained variance sum: ", pca.explained_variance_ratio_.sum())

    data['color'] = [attr_color_map[i] for i in attr_filtered.index]
    # sns.scatterplot(data = data, x = '0',y = '1', hue = 'color')

    for start_idx in attractor_dict[starting_attractors]:
        print(start_idx)
        if plot_vs:
            if perturb is None:
                raise ValueError("If plot_vs is true, perturb must be specified.")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=600)
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
                        walk_new = pca.transform(walk_list)
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
                        walk_new = pca.transform(walk_list)
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

            # Setting the values for all axes.
            plt.setp([ax1,ax2], xlim=custom_xlim, ylim=custom_ylim)
            if show:
                plt.show()
            else:
                plt.savefig(f"{walk_path}/{start_idx}/walks_{perturb}_{starting_attractors}{save_as}.png")

        else:
            for start_idx in attractor_dict[starting_attractors]:
                plt.figure(figsize=(12, 10), dpi=600)
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
                                walk_new = pca.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
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
                                walk_new = pca.transform(walk_list)
                                data_walk = pd.DataFrame(walk_new, columns=['0', '1'])
                                data_walks = data_walks.append(data_walk)
                                data_walk['color'] = [(len(data_walk.index) - i) / len(data_walk.index) for i in data_walk.index]
                                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
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
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{walk_path}/{start_idx}/singleplot_walks_{perturb}_{starting_attractors}{save_as}.png")

walk_path = f'{dir_prefix}/{brcd}/walks/long_walks/1000_step_walks'
save_as = "_data-pca"
# parameters for plotting walks
starting_attractors = 'Arc_6'
show_lineplots = True

# plot_random_walks(walk_path, starting_attractors, ATTRACTOR_DIR, attr_color_map,
#                       perturb = None,
#                       num_walks = 5,
#                       binarized_data_df = None,
#                       save_as = "",
#                       show_lineplots = True,
#                       fit_to_data = False,
#                       plot_vs = False,
#                       show = True)
#
plot_random_walks(walk_path, starting_attractors, ATTRACTOR_DIR, attr_color_map,
                      perturb = "RORB_kd",
                      num_walks = 20,
                      binarized_data_df = binarized_data_df,
                      save_as = "_data-pca",
                      show_lineplots = True,
                      fit_to_data = True,
                      plot_vs = True,
                      show = False)


def plot_paths(att_list, phenotypes, phenotype_color, radius, start_idx, num_paths = 100, pca_path_reduce = False,
               walk_to_basin = False):
    pca = PCA(n_components=2)
    att_new = pca.fit_transform(att_list)
    data = pd.DataFrame(att_new, columns=['0', '1'])
    comp = pd.DataFrame(pca.components_, index = [0,1], columns = nodes)
    print(comp.T)
    data['color'] = phenotype_color

    plt.figure(figsize=(12, 10), dpi=600)
    plt.scatter(x=data['0'], y=data['1'], c=data['color'], s=100, edgecolors='k', zorder=4)
    legend_elements = []

    for i, j in enumerate(phenotypes):
        if 'null' not in set(phenotype_color):
            if j == 'null': continue
        legend_elements.append(Patch(facecolor=customPalette[i], label=j))

    plt.legend(handles=legend_elements, loc='best')

    start_type = "null"
    if start_idx in NE_attractors:
        start_type = "NE"
    elif start_idx in ML_attractors:
        start_type = "NON-NE"
    elif start_idx in MLH_attractors:
        start_type = "NEv2"
    elif start_idx in NEH_attractors:
        start_type = "NEv1"

    data_walks = pd.DataFrame(columns=['0','1'])
    att2_list = att_list.copy()
    if walk_to_basin == False:
        with open(op.join(dir_prefix,f"Network/walks/walk_to_basin/MYC_network/{start_idx}"
        f"/MYC_results_radius_{radius}.csv"), 'r') as file:

            line = file.readline()
            cnt = 1
            while line:
                if cnt == 1: pass
                walk = line.strip()
                walk = walk.replace('[','').replace(']','').split(',')
                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                walk_list = []
                for i in walk_states:
                    walk_list.append([int(j) for j in i])
                    att2_list.append([int(j) for j in i])
                walk_new = pca.transform(walk_list)
                data_walk = pd.DataFrame(walk_new, columns=['0','1'])
                data_walks = data_walks.append(data_walk)
                data_walk['color'] = [(len(data_walk.index)-i)/len(data_walk.index) for i in data_walk.index]
                plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                            cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                # sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=1, dashes=True, legend=False,
                #              alpha=0.1, zorder=2)
                cnt += 1
                line = file.readline()
                if cnt == num_paths: break
        plt.title( f"PCA of {cnt} Walks from {start_idx} ({start_type})"
                   f"\n Dimensionality Reduction on Attractors")

    else:
        with open(op.join(dir_prefix, f"Network/walks/walk_to_basin/MYC_network/"
        f"{start_idx}/MYC_results_radius_{radius}.csv"),
                  'r') as file:

            line = file.readline()
            cnt = 1
            while line:
                if cnt == 1: pass
                walk = line.strip()
                walk = walk.replace('[','').replace(']','').split(',')
                walk_states = [bb.utils.idx2binary(int(i), n) for i in walk]
                walk_list = []
                for i in walk_states:
                    walk_list.append([int(j) for j in i])
                    att2_list.append([int(j) for j in i])
                walk_new = pca.transform(walk_list)
                data_walk = pd.DataFrame(walk_new, columns=['0','1'])
                data_walks = data_walks.append(data_walk)
                data_walk['color'] = [(len(data_walk.index)-i)/len(data_walk.index) for i in data_walk.index]
                # plt.scatter(x = data_walk['0'], y = data_walk['1'], c = data_walk['color'],
                #             cmap = 'Blues', s = 20, edgecolors='k', zorder = 3)
                sns.lineplot(x=data_walk['0'], y=data_walk['1'], lw=1, dashes=True, legend=False,
                             alpha=0.1, zorder=2)
                cnt += 1
                line = file.readline()
                if cnt == num_paths: break
        if radius == NE_attractors:
            basin_type = "NE"
        elif radius == ML_attractors:
            basin_type = "NON-NE"
        elif radius == MLH_attractors:
            basin_type = "NEv2"
        elif radius == NEH_attractors:
            basin_type = "NEv1"
        plt.title( f"PCA of {cnt} Walks from {start_idx} ({start_type}) to {basin_type}"
                   f"\n Dimensionality Reduction on Attractors")

    sns.kdeplot(data_walks['0'], data_walks['1'], shade=True, shade_lowest=False,zorder=1, n_levels=20,cbar = True)
    plt.show()
    if pca_path_reduce == True:
        plt.figure(figsize=(12, 10), dpi=600)
        att2_new = pca.fit_transform(att2_list)
        data2 = pd.DataFrame(att2_new, columns=['0','1'])
        comp = pd.DataFrame(pca.components_, index=[0, 1], columns=nodes)
        print(comp.T)
        plt.scatter(x = data2.iloc[0:10]['0'], y = data2.iloc[0:10,]['1'], c = data['color'], s = 100, edgecolors='k',
                    zorder = 4)
        legend_elements = []

        for i, j in enumerate(phenotypes):
            if 'null' not in set(phenotype_color):
                if j == 'null': continue
            legend_elements.append(Patch(facecolor=customPalette[i],label=j))

        plt.legend(handles = legend_elements,loc = 'best')
        plt.title(f"PCA of {cnt} Walks from {start_idx} ({start_type}) \n Dimensionality Reduction on All States in Paths")

        sns.kdeplot(data2.iloc[10:]['0'], data2.iloc[10:]['1'], shade=True,
                    shade_lowest=False, zorder=1, n_levels = 20, cbar = True)
        plt.show()


# for radius in [NEH_attractors, ML_attractors,MLH_attractors]:
#     for start_idx in NE_attractors:
#         plot_paths(att_list,phenotypes, phenotype_color, radius, start_idx, walk_to_basin=True)


def check_middle_stop(start_idx, basin, check_stops, radius=2):
    with open(op.join(dir_prefix, f"Network/walks/walk_to_basin/MYC_network/{start_idx}/MYC_results_radius_{basin}.csv"),
              'r') as file:
        line = file.readline()
        cnt = 1
        stopped_NE = None
        stopped_NEH = None
        stopped_MLH = None
        stopped_ML = None
        for k in check_stops:
            if k in NE_attractors:
                stopped_NE = 0
            elif k in ML_attractors:
                stopped_ML = 0
            elif k in MLH_attractors:
                stopped_MLH = 0
            elif k in NEH_attractors:
                stopped_NEH = 0
        while line:
            cnt += 1
            line = file.readline()
            if cnt == 1: pass
            walk = line.strip()
            walk = walk.replace('[', '').replace(']', '').split(',')
            for j in check_stops:
                for i in walk:
                    if i == '': continue
                    dist = bb.utils.hamming_idx(j, int(i), n)
                    if dist < radius:
                        if j in NE_attractors:
                            stopped_NE += 1
                        elif j in ML_attractors:
                            stopped_ML += 1
                        elif j in MLH_attractors:
                            stopped_MLH += 1
                        elif j in NEH_attractors:
                            stopped_NEH += 1
                        break

    return stopped_NE, stopped_NEH, stopped_MLH, stopped_ML

if False:
    with open(op.join(dir_prefix, f"Network/walks/walk_to_basin/stops_MYC.csv"), 'w+') as file:
        file.write("Start, End, Stop, Count\n")
        for start_idx in known_steady_states:
            for basin in basins:
                for stop in basins:
                    phen_basin = None
                    phen_start = None
                    phen_stop =None
                    if start_idx in NE_attractors:
                        phen_start = 'NE'
                    elif start_idx in ML_attractors:
                        phen_start = 'ML'
                    elif start_idx in MLH_attractors:
                        phen_start = 'MLH'
                    elif start_idx in NEH_attractors:
                        phen_start = 'NEH'
                    if basin[0] in NE_attractors:
                        phen_basin = 'NE'
                    elif basin[0] in ML_attractors:
                        phen_basin = 'ML'
                    elif basin[0] in MLH_attractors:
                        phen_basin = 'MLH'
                    elif basin[0] in NEH_attractors:
                        phen_basin = 'NEH'
                    if stop == NE_attractors:
                        phen_stop = 'NE'
                    elif stop == ML_attractors:
                        phen_stop = 'ML'
                    elif stop == MLH_attractors:
                        phen_stop = 'MLH'
                    elif stop == NEH_attractors:
                        phen_stop = 'NEH'
                    if phen_basin == phen_start: continue
                    if phen_start == phen_stop: continue
                    if phen_basin == phen_stop: continue
                    try:
                        NE, NEH, MLH, ML = (check_middle_stop(start_idx,basin, stop, radius=2))
                    except FileNotFoundError: continue
                    if NE != None:
                        print(NE, f" walks from {start_idx} ({phen_start}) to {phen_basin} stop at NE")
                        file.write(f"{phen_start},{phen_basin}, NE,{NE}\n" )
                    if NEH != None:
                        print(NEH, f" walks from {start_idx} ({phen_start}) to {phen_basin} stop at NEH")
                        file.write(f"{phen_start},{phen_basin}, NEH,{NEH}\n" )
                    if MLH != None:
                        print(MLH, f" walks from {start_idx} ({phen_start}) to {phen_basin} stop at MLH")
                        file.write(f"{phen_start},{phen_basin}, MLH,{MLH}\n" )
                    if ML != None:
                        print(ML,f" walks from {start_idx} ({phen_start}) to {phen_basin} stop at ML")
                        file.write(f"{phen_start},{phen_basin}, ML,{ML}\n" )

    file.close()



