import numpy as np
import seaborn as sns
import booleabayes as bb
import os
import os.path as op
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bb_utils import draw_grn

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
brcd = str(1112)
save_dir = f"{dir_prefix}/{brcd}"

edge_weights = pd.read_csv(f"{dir_prefix}/{brcd}/rules/edge_weights.csv", header = 0, index_col=0)
# network_file = f"{dir_prefix}/networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv"
network_file = f"{dir_prefix}/networks/DIRECT-NET_network_2020db_0.1_top8regs_wo_sinks.csv"
nodes = edge_weights.index

def plot_subgraph(keep_nodes, network_file, nodes, edge_weights, keep_parents = True, keep_children = True,
                  save_dir = "", arrows = "straight", show = False, save = True, off_node_arrows_gray = True, weight = 3):

    edge_df = pd.read_csv(network_file, header = None)##network file

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)


    for i,r in edge_df.iterrows():
        G.add_edge(r[0],r[1], weight = edge_weights.loc[r[1],r[0]])

    total_keep = keep_nodes.copy()
    attrs = {}

    if keep_children:
        for n in keep_nodes:
            for successor in G.successors(n):
                total_keep.append(successor)
                attrs[successor] = {"subset":1}
    if keep_parents:
        for n in keep_nodes:
            for predecessor in G.predecessors(n):
                total_keep.append(predecessor)
                attrs[predecessor] = {"subset":3}
    for node in keep_nodes:
        attrs[node] = {"subset":2}

    SG = G.subgraph(nodes = total_keep)
    nx.set_node_attributes(SG, attrs)
    edges = SG.edges()
    weights = [weight*np.abs(SG[u][v]['weight']) for u,v in edges]
    color = []
    for u,v in edges:
        if u in keep_nodes or v in keep_nodes:
            if SG[u][v]['weight'] < 0:
                color.append('red')
            else:
                color.append('green')
        else:
            if off_node_arrows_gray:
                color.append('lightgray')
            else:
                if SG[u][v]['weight'] < 0:
                    color.append('red')
                else:
                    color.append('green')

    print(nx.get_node_attributes(SG, name = 'subset'))
    if arrows == "straight":
        nx.draw_networkx(SG,pos=nx.multipartite_layout(SG,align = 'horizontal'),node_size = 500, font_size = 6,
                     with_labels=True, arrows = True,width = weights, edge_color = color)#,
    elif arrows == "curved":
        nx.draw_networkx(SG,pos=nx.multipartite_layout(SG,align = 'horizontal'),node_size = 500, font_size = 6,
                         with_labels=True, arrows = True,width = weights, edge_color = color,
                        connectionstyle="arc3,rad=0.4")
    else:
        print("arrows must be one of {'curved','straight'}")
    name_plot = ""
    for name in keep_nodes:
        name_plot = name_plot + f"_{name}"
    if show:
        plt.show()
    if save:
        plt.savefig(f"{save_dir}/subnetwork{name_plot}_{arrows}.png",dpi = 300)
        plt.close()

# keep_nodes = ['ZBTB7A']
# ASCL1, RORB, NFIB, EGR1,REST and TCF7L2.

# for g in ['ASCL1', 'RORB', 'NFIB', 'EGR1', 'REST', 'TCF7L2']:
for g in ['TEAD1','RBPJ']:
    plot_subgraph([g], network_file, nodes, edge_weights, keep_parents = True, keep_children = True,
              save_dir = f"{dir_prefix}/{brcd}", arrows = "curved", show = False, save = True, off_node_arrows_gray=True,
              weight = 3)
    plot_subgraph([g], network_file, nodes, edge_weights, keep_parents=True, keep_children=True,
                  save_dir=f"{dir_prefix}/{brcd}", arrows="straight", show=False, save=True, off_node_arrows_gray=True,
                  weight=3)