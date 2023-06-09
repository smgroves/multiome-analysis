import time
import seaborn as sns
import booleabayes as bb
import os
from os.path import exists
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

def save_if_overwrite(network_file, G, attributes = True):
    if exists(network_file):
        if overwrite:
            outfile = open(network_file, "w")
            for edge in G.edges():
                outfile.write("%s,%s" % (edge[0], edge[1]))
                if attributes:
                    for a in list(G[edge[0]][edge[1]].keys()):
                        outfile.write(",%s" % (G[edge[0]][edge[1]][a]))
                outfile.write("\n")
            outfile.close()
        else:
            print("Network file already exists.")
    else:
        outfile = open(network_file, "w")
        for edge in G.edges():
            outfile.write("%s,%s" % (edge[0], edge[1]))
            if attributes:
                for a in list(G[edge[0]][edge[1]].keys()):
                    outfile.write(",%s" % (G[edge[0]][edge[1]][a]))
            outfile.write("\n")
        outfile.close()

def add_connections(G, net_df, thresold_var = None, threshold = 0, parent_node = 'TF motif', child_node = 'Target_gene',
                    add_weight = True, weight = 'motif score', evidence = 'direct-net'):
    for i,r in net_df.iterrows():
        if G.has_edge(r[parent_node], r[child_node]): pass
        else:
            if thresold_var is not None:
                if r[thresold_var] > threshold:
                    if add_weight:
                        G.add_edge(r[parent_node], r[child_node], weight = r[weight], evidence = evidence)
                    else:
                        G.add_edge(r[parent_node], r[child_node],  evidence = evidence)
            else:
                if add_weight:
                    G.add_edge(r[parent_node], r[child_node], weight = r[weight],  evidence = evidence)
                else:
                    G.add_edge(r[parent_node], r[child_node],  evidence = evidence)
    return G


# =============================================================================
# Make network
# =============================================================================
#Instead of using BooleaBayes to make the network with CHIP-seq, we are going to use DIRECT-NET information.
DIRECT_NET_INDIR = "./DIRECT-NET-FILES/"
NETWORK_OUTDIR = './networks/'
overwrite = True

direct_net = pd.read_csv(os.path.join(DIRECT_NET_INDIR,"Direct_net_0.1.csv"), header = 0, index_col = 0)
direct_net['Target_gene'] = [i.upper() for i in direct_net['Target_gene']]

plt.hist(direct_net['motif score'], bins = 30)
plt.show()

threshold = int(input("enter threshold..."))


# "TF motif" column is parent node, "Target gene" is child node
tfs = []
for i,r in direct_net.iterrows():
    tfs.append(r['TF motif'])
    tfs.append(r["Target_gene"])

tfs = list(set(tfs))

G = nx.DiGraph()
for tf in tfs:
    G.add_node(tf)

G = add_connections(G, direct_net, threshold = threshold, thresold_var='motif score')

## Save original DIRECT-NET network
network_file = os.path.join(NETWORK_OUTDIR,f"DIRECT-NET_network_2020db_0.1.csv")
save_if_overwrite(network_file,G)
#
# ## Add RORB edges
# for i,r in direct_net.iterrows():
#     G.add_node('RORB')
#     if r['TF motif'] == 'RORA':
#         if G.has_edge('RORB',r['Target_gene']): pass
#         else:
#             G.add_edge("RORB",r['Target_gene'], weight = r['motif score'], evidence = 'direct-net')
#
# ## Save RORB-augmented network
# network_file = os.path.join(NETWORK_OUTDIR,f"DIRECT-NET_network_threshold_{threshold}_withRORB.csv")
# save_if_overwrite(network_file,G)
#
# print("G in-degree for each node:", G.in_degree())
# print("G out-degree for each node:", G.out_degree())
#
# ## Add information from FigR DORCs
# figr = pd.read_csv(os.path.join(DIRECT_NET_INDIR, "FigR_DORC_TF.csv"), header = 0, index_col=0)
# figr.DORC = [i.upper() for i in figr.DORC]
# figr.Motif = [i.upper() for i in figr.Motif]
#
# G_figr = nx.DiGraph()
# figr_tfs = []
# for i,r in figr.iterrows():
#     figr_tfs.append(r['Motif'])
#     figr_tfs.append(r["DORC"])
# for tf in figr_tfs:
#     G_figr.add_node(tf)
#
# add_connections(G_figr, figr, parent_node='Motif',child_node='DORC', weight='Score', evidence = 'FIGR')
# print("G_figr in-degree for each node:", G_figr.in_degree())
# print("G_figr out-degree for each node:", G_figr.out_degree())
#
# network_file = os.path.join(NETWORK_OUTDIR,"FIGR_network.csv")
# save_if_overwrite(network_file,G_figr)
#
# G_copy = G.copy()
# nodes = G_copy.nodes()
#
# for n in nodes:
#     # If n is a source, look in FIGR for regulators
#     if G.in_degree(n) == 0:
#         print("Source node in G:", n)
#         if n in G_figr.nodes:
#             # add in-edges to n in G_figr
#             G.add_edges_from(G_figr.in_edges(n))
#             print("added", G_figr.in_degree(n), "edges")
#     # If n is a sink, look in FIGR for target genes
#     if G.out_degree(n) == 0:
#         print("Sink node in G:", n)
#         if n in G_figr.nodes:
#             # add in-edges to n in G_figr
#             G.add_edges_from(G_figr.out_edges(n), weight = 100, evidence = 'FIGR')
#             print("added", G_figr.out_degree(n), "edges")
#
# ## Save FIGR-augmented network
# network_file = os.path.join(NETWORK_OUTDIR,f"DIRECT-NET_network_with_FIGR_threshold_{threshold}.csv")
# save_if_overwrite(network_file,G)
#
# for n in G.nodes():
#     if G.in_degree(n) == 0: print(n, " is a source node")
#     if G.out_degree(n) == 0: print(n, " is a sink node")



