import scvelo as scv
import scanpy as sc
import cellrank as cr
import pandas as pd
import os
import numpy as np

DIRECT_NET_INDIR = "./DIRECT-NET-FILES/"

def preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_imputed.csv", extra_genes = None):
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
    if extra_genes is not None:
        for g in extra_genes:
            tfs.append(g)

    overlap = (list(set(tfs).intersection(set([i.upper() for i in adata.var_names]))))
    print(list(set(tfs).difference(set([i.upper() for i in adata.var_names]))))

    adata_net = adata[:,[i.capitalize() for i in overlap]]
    print(adata_net)
    print(adata_net.layers["imputed"][0:10,0:10])
    adata_imputed = pd.DataFrame(adata_net.layers["imputed"], index=adata_net.obs_names, columns=adata_net.var_names)
    adata_imputed.to_csv(f"./data/{outfile_name}")

# adata = cr.read('../data/M2/adata_04_nodub.h5ad')
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_04_nodubs_imputed_M2.csv")

adata = cr.read('../data/combined/adata_02_filtered.h5ad')
preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_imputed_combined.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'])
