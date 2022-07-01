import enrichr
import networkx as nx
import time


def prune(G_orig, prune_sources = True, prune_sinks = True):
    G = G_orig.copy()
    n = len(G.nodes())
    nold = n + 1

    while (n != nold):
        nold = n
        for tf in list(G.nodes()):
            if prune_sources == True:
                if G.in_degree(tf) == 0:  G.remove_node(tf)
            if prune_sinks == True:
                if G.out_degree(tf) == 0: G.remove_node(tf)
            else:
                if G.in_degree(tf) == 0 and G.out_degree(tf) == 0:G.remove_node(tf)
        n = len(G.nodes())
    return G


def prune_info(G_orig, prune_self_loops=True):
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if 'db' not in edges[target]:
                G.remove_edge(tf, target)
            elif len(edges[target]['db']) < 2:
                G.remove_edge(tf, target)
    return prune(G)


def prune_to_chea(G_orig, prune_self_loops=True):
    G = G_orig.copy()
    for tf in list(G.nodes()):
        edges = G.adj[tf]
        for target in list(edges.keys()):
            if tf == target and prune_self_loops:
                G.remove_edge(tf, target)
                continue
            if 'db' in edges[target]:
                if not True in ['ChEA' in i for i in edges[target]['db']]: G.remove_edge(tf, target)
    #                if len(edges[target]['db']) < 2: G.remove_edge(tf, target)
    return prune(G)
tfs = ['Prdm16','Mecom','Runx1','Ahr','Esr1','Rarb','Npas2','Tcf7l2','Nfia','Sox5','Meis2','Prox1','Ascl1','Tbx15','Rorb','Jund','Fosb','Jun','Fos',
       'Junb','Egr1','Kmt2a','Lmx1b','Nfatc2','Bach1','Hif1a','Hsf2','Six1','Six4','Ets1','Pknox2','Cux2','Tfdp1','Nfib','Pbx1','Tcf12','Zbtb20','Creb1',
       'Bbx','Kdm2a','Lcor','Trps1','Nr3c2','Tcf4','Sox6','Zeb1','Epas1','Smad3','Sp100','Nfix','Bach2','Foxo3','Glis3','Ehf','Nfe2l2','Tcf7l1','Arid5b','Thrb',
       'Cux1','Grhl2','Stat1','Stat2','Nr6a1','Tead1','Nfkb1','Nfyc','Stat4','Pparg']
tfs = [i.upper() for i in tfs]

G = nx.DiGraph()
# prelim_G = nx.DiGraph()
# with open("/Users/sarahmaddox/Dropbox (Vanderbilt)/Quaranta_Lab/SCLC/Network/mothers_network.csv") as infile:
#     for line in infile:
#         line = line.strip().split(',')
#         prelim_G.add_edge(line[0], line[1])

for tf in tfs: G.add_node(tf)

for tf in tfs:
    enrichr.build_tf_network(G, tf, tfs)
    time.sleep(1)

# for edge in prelim_G.edges():
#     if edge[0] in tfs and edge[1] in tfs:
#         G.add_edge(edge[0], edge[1])

outdir = '.'
outfile = open("_0_network.csv", "w")
for edge in G.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()

Gp = prune(G, prune_sinks=False, prune_sources=False)
# Gp.add_edge('NEUROD1',"MYC",db=["ChEA_2013","ChEA_2015"]) #artificially add NEUROD1 --> MYC connection based on Borromeo et al. 2016

outfile = open("_1_network.csv", "w")
for edge in G.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()
Gpp = prune_info(Gp)

# add code to keep INS and GCG even though they don't have out-going edges
Gpp = prune_to_chea(Gp)

outfile = open("_2_network.csv", "w")
for edge in Gpp.edges(): outfile.write("%s,%s\n" % (edge[0], edge[1]))
outfile.close()
