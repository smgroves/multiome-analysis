# Code copied from main_all_data.py. This script will run random walks from attractors to specific basins.

import random
from bb_utils import *
from bb_utils import _long_random_walk
customPalette = sns.color_palette('tab10')
# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'

## Set metadata information
cellID_table = 'data/AA_clusters.csv'
cluster_header_list = ["class"]
brcd = str(9999)
print(brcd)

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

save_dir = f"{dir_prefix}/{brcd}"
# New way to do random walks until reach basin. Instead of looking for a specific basin, keep walking some length of steps
# NOTE: This function will be integrated into booleabayes version > 0.1.9

def long_random_walks(starting_attractors,attractor_dict, rules,regulators_dict, nodes, save_dir,
                      on_nodes = [], off_nodes = [], max_steps = 2000, iters = 100, overwrite_walks = False):
    try:
        os.mkdir(f"{save_dir}/walks/long_walks/")
    except FileExistsError:
        pass
    for s in starting_attractors:
        print(s)
        for start_idx in attractor_dict[s]:
            print("Starting state: ", start_idx)

            switch_counts_0 = dict()
            for node in nodes: switch_counts_0[node] = 0
            n_steps_to_leave_0 = []
            try:
                os.mkdir(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/")
            except FileExistsError:
                pass

            try:
                os.mkdir(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}")
            except FileExistsError:
                if overwrite_walks:
                    pass
                else:
                    continue


            outfile = open(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results.csv", "w+")

            # 1000 iterations; print progress of random walk every 10% of the way
            # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
            # no perturbations
            for iter_ in range(iters):
                if iter_ % 10 == 0:
                    print(str(iter_ / iters * 100) + "%")
                walk, counts, switches, distances = _long_random_walk(start_idx, rules,
                                                                     regulators_dict, nodes,
                                                                     max_steps=max_steps)
                n_steps_to_leave_0.append(len(distances))
                for node in switches:
                    if node is not None: switch_counts_0[node] += 1
                outfile.write(f"{walk}\n")
            outfile.close()

        print("Running TF Perturbations...")

        for perturb in off_nodes:
            for start_idx in attractor_dict[s]:
                print("Starting state: ",start_idx)
                print("Perturbation: ",perturb)

                switch_counts_0 = dict()
                for node in nodes: switch_counts_0[node] = 0
                n_steps_to_leave_0 = []

                outfile = open(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results_{perturb}_kd.csv", "w+")

                # 1000 iterations; print progress of random walk every 10% of the way
                # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
                # no perturbations
                for iter_ in range(iters):
                    if iter_ % 10 == 0:
                        print(str(iter_/iters*100) + "%")
                    walk, counts, switches, distances = _long_random_walk(start_idx, rules,
                                                                         regulators_dict, nodes,
                                                                         max_steps=max_steps,
                                                                         off_nodes=[perturb])
                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None: switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                outfile.close()


        for perturb in on_nodes:
            for start_idx in attractor_dict[s]:
                print("Starting state: ",start_idx)
                print("Perturbation: ",perturb)

                switch_counts_0 = dict()
                for node in nodes: switch_counts_0[node] = 0
                n_steps_to_leave_0 = []

                outfile = open(f"{save_dir}/walks/long_walks/{max_steps}_step_walks/{start_idx}/results_{perturb}_act.csv", "w+")

                # 1000 iterations; print progress of random walk every 10% of the way
                # counts: histogram of walk; switches: count which TFs flipped; distance = starting state to current state; walk until take max steps or leave basin
                # no perturbations
                for iter_ in range(iters):
                    if iter_ % 10 == 0:
                        print(str(iter_/iters*100) + "%")
                    walk, counts, switches, distances = _long_random_walk(start_idx, rules,
                                                                         regulators_dict, nodes,
                                                                         max_steps=max_steps,
                                                                         on_nodes=[perturb])
                    n_steps_to_leave_0.append(len(distances))
                    for node in switches:
                        if node is not None: switch_counts_0[node] += 1
                    outfile.write(f"{walk}\n")
                outfile.close()

knockdowns = ['NFIB','RORB','EGR1']
starting_attractors = ['Arc_5']
long_random_walks(starting_attractors,attractor_dict, rules,regulators_dict, nodes, save_dir,
                      on_nodes = [], off_nodes = knockdowns, max_steps = 2000, iters = 100, overwrite_walks = False)





# basin_set = set(attractor_dict.keys()).difference({"Arc_6"})
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
