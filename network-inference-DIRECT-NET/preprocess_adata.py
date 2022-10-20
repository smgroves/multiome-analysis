import scvelo as scv
import scanpy as sc
import cellrank as cr
import pandas as pd
import os

DIRECT_NET_INDIR = "./DIRECT-NET-FILES/"

adata = cr.read('../data/M2/adata_04_nodub.h5ad')

direct_net = pd.read_csv(os.path.join(DIRECT_NET_INDIR,"Direct_net.csv"), header = 0, index_col = 0)
direct_net['Target_gene'] = [i.upper() for i in direct_net['Target_gene']]
# "TF motif" column is parent node, "Target gene" is child node
tfs = []
for i,r in direct_net.iterrows():
    tfs.append(r['TF motif'])
    tfs.append(r["Target_gene"])
tfs = list(set(tfs))

figr = pd.read_csv(os.path.join(DIRECT_NET_INDIR, "FigR_DORC_TF.csv"), header = 0, index_col=0)
figr.DORC = [i.upper() for i in figr.DORC]
figr.Motif = [i.upper() for i in figr.Motif]

for i,r in figr.iterrows():
    tfs.append(r['Motif'])
    tfs.append(r["DORC"])

tfs = list(set(tfs))

print(tfs)
print(len(tfs))

print(list(set(tfs).difference(set([i.upper() for i in adata.var_names]))))