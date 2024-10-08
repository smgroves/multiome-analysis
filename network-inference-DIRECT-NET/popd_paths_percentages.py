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
from statannot import add_stat_annotation

np.random.seed(1)

## Set paths
brcd = str(9999)
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)
v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
n = len(nodes)
n_states = 2 ** n

attractor_dict = {}
attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)

for i,r in attr_filtered.iterrows():
    attractor_dict[i] = []
for i,r in attr_filtered.iterrows():
    attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))


## loop through random walks and ask two questions:
# 1. What percentage of the time does the walk spend close to each state?
# 2. Does the walk make it to X state? (Probabilities may not add to 1 here)

def make_percentage_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = None):
    columns = list(attractor_dict.keys())
    columns.append('None')
    columns.append('Start')
    columns.append('Walk')
    popd_df = pd.DataFrame(columns = columns)

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
        with open(results_file, 'r') as file:
            line = file.readline()
            cnt = 1
            while line:
                reach_states_dict = {i: 0 for i in attractor_dict.keys()}
                reach_states_dict['None'] = 0
                reach_states_dict['Start'] = start_idx
                reach_states_dict['Walk'] = cnt

                if cnt == 1: pass
                walk = line.strip()
                walk = walk.replace('[', '').replace(']', '').split(',')
                for step, i in enumerate(walk):
                    min_distance = 100
                    for att in reverse_attr_dist.keys():
                        distance = (bb.utils.hamming_idx(int(i), int(att), n))
                        if distance < min_distance:
                            min_distance = distance
                            closest_att = reverse_attr_dist[att]
                    if min_distance <= radius:
                        reach_states_dict[closest_att] += 1
                    else:
                        reach_states_dict['None'] += 1
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

walk_path = f'{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks'
starting_attractors = 'Arc_6'
num_walks = 100
radius = 4

# make_percentage_popd_df(walk_path=f'{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks',starting_attractors = 'Arc_6', num_walks = 100, radius  = 4, attractor_dict, n = len(nodes), perturbation = 'RORB_kd')
# print("RORB_kd")
# make_percentage_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = 'RORB_kd')
# print("EGR1_kd")
# make_percentage_popd_df(walk_path,starting_attractors, num_walks = 100, radius, attractor_dict, n = len(nodes), perturbation = 'EGR1_kd')
# make_percentage_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = 'TCF7L2_act')
# make_percentage_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = 'MEIS2_kd')

def plot_attractors_reached(walk_path, starting_attractors,perturbations = [],
                            num_walks = 100,radius= 4, length_walks = 2000,
                            show = False):
    data_unperturbed = pd.read_csv(f"{walk_path}/{starting_attractors}_radius_{radius}_percentages.csv", header=0,
                                   index_col=None)

    # Question 1
    ave_unperturbed = data_unperturbed.groupby("Start").mean().drop(["Walk"], axis=1)
    ave_unperturbed_melt = ave_unperturbed.melt(
        value_vars=['Generalist', 'Arc_4', 'Arc_6', 'Arc_1', 'Arc_5', 'Arc_2', 'Arc_5_Arc_6', "None"],
        ignore_index=False)
    ave_unperturbed_melt['perturbation'] = 'unperturbed'
    combined = ave_unperturbed_melt.copy()

    data_unperturbed_bool = data_unperturbed.copy()
    end_state_columns = data_unperturbed.drop(["Walk", 'Start'], axis=1).columns.values

    for c in end_state_columns:
        data_unperturbed_bool[c] = data_unperturbed_bool[c].apply(lambda x: 1 if x > 0 else 0)
    ave_unperturbed_bool = data_unperturbed_bool.groupby("Start").mean().drop(["Walk"], axis=1)
    ave_unperturbed_bool_melt = ave_unperturbed_bool.melt(
        value_vars=['Generalist', 'Arc_4', 'Arc_6', 'Arc_1', 'Arc_5', 'Arc_2', 'Arc_5_Arc_6'], ignore_index=False)
    ave_unperturbed_bool_melt['perturbation'] = 'unperturbed'
    combined_bool = ave_unperturbed_bool_melt.copy()


    for perturbation in perturbations:
        data_perturb = pd.read_csv(f"{walk_path}/{starting_attractors}_radius_{radius}_percentages_{perturbation}.csv", header=0,
                                index_col=None)
        ave_perturb = data_perturb.groupby("Start").mean().drop(["Walk"], axis=1)
        ave_perturb_melt = ave_perturb.melt(
            value_vars=['Generalist', 'Arc_4', 'Arc_6', 'Arc_1', 'Arc_5', 'Arc_2', 'Arc_5_Arc_6', 'None'],
            ignore_index=False)
        ave_perturb_melt['perturbation'] = perturbation
        combined = pd.concat([combined, ave_perturb_melt], axis=0)

        data_perturb_bool = data_perturb.copy()
        for c in end_state_columns:
            data_perturb_bool[c] = data_perturb_bool[c].apply(lambda x: 1 if x > 0 else 0)
        ave_perturb_bool = data_perturb_bool.groupby("Start").mean().drop(["Walk"], axis=1)
        ave_perturb_bool_melt = ave_perturb_bool.melt(
            value_vars=['Generalist', 'Arc_4', 'Arc_6', 'Arc_1', 'Arc_5', 'Arc_2', 'Arc_5_Arc_6'], ignore_index=False)
        ave_perturb_bool_melt['perturbation'] = perturbation
        combined_bool = pd.concat([combined_bool, ave_perturb_bool_melt], axis=0)

    combined['value'] = combined['value'].apply(lambda x: x * 100 / length_walks)
    combined_bool['value'] = combined_bool['value'].apply(lambda x: x * 100)

    box_pairs = []
    for perturbation in perturbations:
        box_pairs = box_pairs + [((state, 'unperturbed'), (state, perturbation)) for state in combined['variable'].unique()]
    order = combined.groupby('variable').mean().sort_values('value', ascending = False).index.values
    ax = sns.boxplot(data=combined, x='variable', y='value', hue='perturbation',linewidth = .5,
                order = order)
    add_stat_annotation(ax, data=combined,x='variable', y='value', hue='perturbation',
                        box_pairs=box_pairs, test='t-test_ind', loc='inside', verbose=1, order = order)
    # plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))

    plt.xticks(rotation=90)
    plt.title(
        f"Average Percentage of Time Spent in Each Attractor Basin \n Starting from {starting_attractors} \n Radius = {radius}, Number of Steps = 2000")
    plt.ylabel("% Time Spent in Each Attractor Basin During Walk")
    plt.xlabel("Attractor Basin")
    plt.tight_layout()
    plt.savefig(f"{walk_path}/{starting_attractors}_radius_{radius}_{perturbations}_percentages.pdf")
    plt.close()
    if show:
        plt.show()

    box_pairs = []
    for perturbation in perturbations:
        box_pairs = box_pairs + [((state, 'unperturbed'), (state, perturbation)) for state in
                                 combined_bool['variable'].unique()]

    order = combined_bool.groupby('variable').mean().sort_values('value', ascending = False).index.values
    ax = sns.boxplot(data=combined_bool, x='variable', y='value', hue='perturbation',linewidth = .5,
                order = order)
    add_stat_annotation(ax, data=combined_bool, x='variable', y='value', hue='perturbation',
                        box_pairs=box_pairs, test='t-test_ind', loc='inside', verbose=1, order=order)
    plt.xticks(rotation=90)
    plt.ylabel("% Walks that Reach Attractor Basin")
    plt.xlabel("Attractor Basin")
    plt.title(
        f"Average Percentage of Walks that Reach Each Attractor Basin \n Starting from {starting_attractors} Attractors \n Radius = {radius}, Number of Walks = {num_walks}")
    plt.tight_layout()
    plt.savefig(f"{walk_path}/{starting_attractors}_radius_{radius}_{perturbations}_reached.pdf")
    plt.close()
    if show:
        plt.show()

plot_attractors_reached(walk_path, starting_attractors,perturbations = ['RORB_kd','EGR1_kd','MEIS2_kd'],num_walks = 100,radius= 4, length_walks = 2000)
# plot_attractors_reached(walk_path, starting_attractors,perturbations = ['TCF7L2_act'],num_walks = 100,radius= 4, length_walks = 2000)