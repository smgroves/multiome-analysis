import time
import seaborn as sns
import booleabayes as bb
import os
from os.path import exists
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from bb_utils import *

from booleabayes import enrichr
overwrite = True
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

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks.csv'

net = pd.read_csv(f"{dir_prefix}/{network_path}", header = None)
tfs = set(net[0]).union(set(net[1]))
graph = nx.DiGraph()

for tf in tfs:
    graph.add_node(tf)

for i,r in net.iterrows():
    graph.add_edge(r[0],r[1])
orig_size = len(graph.edges())
print(orig_size)

markers = ['ICAM1','NCAM1',"CD24",'EPCAM','CD44']
for m in markers:
    graph.add_node(m)

for m in markers:
    enrichr.build_tf_network(graph, m, tfs)
    print(len(graph.edges())-orig_size)
    orig_size = len(graph.edges())
    time.sleep(1)
print(graph.nodes())
print(len(graph.edges()))

network_file = f"{dir_prefix}/networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_expanded.csv"
save_if_overwrite(network_file,graph)