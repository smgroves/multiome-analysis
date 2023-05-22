# Code copied from main_all_data.py. This script will run random walks from attractors to specific basins.

import random
from bb_utils import *


customPalette = sns.color_palette('tab10')
# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
on_nodes = []
off_nodes = []

## Set variables for computation
remove_sinks=False
remove_selfloops=False
remove_sources=False

node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True

fname = "combined"
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'
data_path = f'data/adata_imputed_combined.csv'
t1 = False
data_t1_path = None #if no T1 (i.e. single dataset), replace with None

## Set metadata information
cellID_table = 'data/AA_clusters.csv'
cluster_header_list = ["class"]
brcd = str(9999)
print(brcd)
data_train_t0_path = f'{brcd}/data_split/train_t0_{fname}.csv'
data_test_t0_path = f'{brcd}/data_split/test_t0_{fname}.csv'

job_brcd = str(random.randint(0,99999)) #use a job brcd to keep track of multiple jobs for the same brcd
print(f"Job barcode: {job_brcd}")

random_state = 1234

#########################################

# =============================================================================
# Load the network
# =============================================================================
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)
v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

print("Reading in pre-generated rules...")
rules, regulators_dict = bb.load.load_rules(fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")

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

# =============================================================================
# Calculate likelihood of reaching other attractors
# =============================================================================

# record random walk from one attractor to another for each combination of attractors
# give list of perturbation nodes and repeat walks with perturbed nodes to record #
# that make it from one attractor to another

dir_prefix_walks = op.join(dir_prefix, brcd)

print("Running TF Random Walks...")
ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

attractor_dict = {}
attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
for i,r in attr_filtered.iterrows():
    attractor_dict[i] = []

for i,r in attr_filtered.iterrows():
    attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

print(attractor_dict)

iters = 100
max_steps = 1000
# start_idx = 75109996 #NE

basin_set = set(attractor_dict.keys()).difference({"Arc_6"})
knockdowns = ["RORB",'EGR1']
# for perturb in knockdowns:
#     for start_idx in attractor_dict['Arc_6']:
#         print("Starting state: ",start_idx)
#         for basin_name in basin_set:
#             print("Basin:", basin_name)
#             basin = attractor_dict[basin_name]
#
#             switch_counts_0 = dict()
#             for node in nodes: switch_counts_0[node] = 0
#             n_steps_to_leave_0 = []
#             try:
#                 os.mkdir(f"{dir_prefix}/{brcd}/walks/walk_to_basin/{start_idx}")
#             except FileExistsError: pass
#             outfile = open(f"{dir_prefix}/{brcd}/walks/walk_to_basin/{start_idx}/results_{basin_name}_{perturb}.csv", "w+")
#             out_len = open(f"{dir_prefix}/{brcd}/walks/walk_to_basin/{start_idx}/len_walks_{basin_name}_{perturb}.csv", "w+")
#             outfile_missed = open(f"{dir_prefix}/{brcd}/walks/walk_to_basin/{start_idx}/results_missed_{basin_name}_{perturb}.csv", "w+")
#
#             # 1000 iterations; print progress of random walk every 10% of the way
#             # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
#             # no perturbations
#             for iter_ in range(iters):
#                 if iter_ % 100 == 0:
#                     print(str(iter_/iters*100) + "%")
#                 walk, counts, switches, distances = bb.rw.random_walk_until_reach_basin(start_idx, rules,
#                                                                                             regulators_dict, nodes,
#                                                                                             radius = 2,
#                                                                                             max_steps=max_steps,
#                                                                                             basin = basin,
#                                                                                             off_nodes=[perturb]
#                                                                                             )
#                 n_steps_to_leave_0.append(len(distances))
#                 for node in switches:
#                     if node is not None: switch_counts_0[node] += 1
#                 if len(walk) != max_steps:
#                     outfile.write(f"{walk}\n")
#                     out_len.write(f"{len(walk)}\n")
#                 else:
#                     outfile_missed.write(f"{walk}\n")
#             outfile.close()
#             out_len.close()
#             outfile_missed.close()

# New way to do random walks until reach basin. Instead of looking for a specific basin, keep walking some length of steps
for start_idx in attractor_dict['Arc_6']:
    print("Starting state: ", start_idx)

    switch_counts_0 = dict()
    for node in nodes: switch_counts_0[node] = 0
    n_steps_to_leave_0 = []
    try:
        os.mkdir(f"{dir_prefix}/{brcd}/walks/long_walks/{max_steps}_step_walks/")
    except FileExistsError:
        pass

    try:
        os.mkdir(f"{dir_prefix}/{brcd}/walks/long_walks/{max_steps}_step_walks/{start_idx}")
    except FileExistsError:
        pass



    outfile = open(f"{dir_prefix}/{brcd}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results.csv", "w+")

    # 1000 iterations; print progress of random walk every 10% of the way
    # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
    # no perturbations
    for iter_ in range(iters):
        if iter_ % 10 == 0:
            print(str(iter_ / iters * 100) + "%")
        walk, counts, switches, distances = long_random_walk(start_idx, rules,
                                                             regulators_dict, nodes,
                                                             max_steps=max_steps)
        n_steps_to_leave_0.append(len(distances))
        for node in switches:
            if node is not None: switch_counts_0[node] += 1
        outfile.write(f"{walk}\n")
    outfile.close()

print("Running TF Perturbations...")

for perturb in knockdowns:
    for start_idx in attractor_dict['Arc_6']:
        print("Starting state: ",start_idx)
        print("Perturbation: ",perturb)

        switch_counts_0 = dict()
        for node in nodes: switch_counts_0[node] = 0
        n_steps_to_leave_0 = []

        outfile = open(f"{dir_prefix}/{brcd}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results_{perturb}_kd.csv", "w+")

        # 1000 iterations; print progress of random walk every 10% of the way
        # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
        # no perturbations
        for iter_ in range(iters):
            if iter_ % 10 == 0:
                print(str(iter_/iters*100) + "%")
            walk, counts, switches, distances = long_random_walk(start_idx, rules,
                                                                 regulators_dict, nodes,
                                                                 max_steps=max_steps,
                                                                 off_nodes=[perturb])
            n_steps_to_leave_0.append(len(distances))
            for node in switches:
                if node is not None: switch_counts_0[node] += 1
            outfile.write(f"{walk}\n")
        outfile.close()
