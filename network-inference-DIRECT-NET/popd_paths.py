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

print(attractor_dict)

def make_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = None):
    end_states = pd.DataFrame(columns = ['Start','Walk', 'Closest Attractor', 'Distance', 'Step'])

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
                    if min_distance <= radius and closest_att not in [starting_attractors,'Generalist','Arc_5_Arc_6']:
                        print(f"Walk: {cnt}, Closest Attractor: {closest_att}, Distance: {min_distance}, Step: {step}")
                        end_states = end_states.append({'Start': start_idx, "Walk":cnt, 'Closest Attractor': closest_att, 'Distance': min_distance, 'Step': step}, ignore_index=True)
                        break
                if min_distance <= radius and closest_att not in [starting_attractors,'Generalist','Arc_5_Arc_6']:
                    pass
                else:
                    end_states = end_states.append({'Start': start_idx, "Walk": cnt, 'Closest Attractor': "None", 'Distance': 'None', 'Step': 'None'}, ignore_index=True)
                cnt += 1
                line = file.readline()
                if cnt == num_walks + 1:
                    break

    if perturbation is None:
        outfile = f"{walk_path}/{starting_attractors}_radius_{radius}_end_states.csv"
    else:
        outfile = f"{walk_path}/{starting_attractors}_radius_{radius}_end_states_{perturbation}.csv"
    end_states.to_csv(outfile, index=False)

walk_path = f'{dir_prefix}/{brcd}/walks/long_walks/2000_step_walks'
starting_attractors = 'Arc_6'
num_walks = 100
radius = 4

# make_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n)

make_popd_df(walk_path,starting_attractors, num_walks, radius, attractor_dict, n, perturbation = 'RORB_kd')
