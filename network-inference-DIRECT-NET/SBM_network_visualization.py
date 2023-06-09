from bb_utils import *
import booleabayes as bb
import networkx as nx
import leidenalg as la
import kitchen
from networkx.algorithms.community.centrality import girvan_newman
import networkx.algorithms.community as nx_comm

remove_sinks=False
remove_selfloops=False
remove_sources=False
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_threshold_10_2020db.csv'

G = kitchen.net.load_nx_network(f"{dir_prefix}/{network_path}")
kitchen.net.get_graph_info(G)

girvan_newman_communities = list(girvan_newman(G))
modularity_df = pd.DataFrame([[k+1, round(nx_comm.modularity(G, girvan_newman_communities[k]), 6)]
                for k in range(len(girvan_newman_communities))],
                            columns=["k", "modularity"])
modularity_df.plot.bar(x="k", figsize=(10,6), title="Girvan-Newman Community Detection Modularity Scores")