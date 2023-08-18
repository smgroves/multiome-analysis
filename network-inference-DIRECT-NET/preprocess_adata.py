import scvelo as scv
import scanpy as sc
import cellrank as cr
import pandas as pd
import os
import numpy as np

DIRECT_NET_INDIR = "./DIRECT-NET-FILES/"


def preprocess_adata(adata, DIRECT_NET_INDIR, Direct_net_file = "Direct_net.csv", outfile_name="adata_imputed.csv", extra_genes=None,
                     imputed_layer='imputed',
                     species='mouse', add_figr = True):
    direct_net = pd.read_csv(os.path.join(DIRECT_NET_INDIR, Direct_net_file), header=0, index_col=0)
    direct_net['Target_gene'] = [i.upper() for i in direct_net['Target_gene']]
    # "TF motif" column is parent node, "Target gene" is child node
    tfs = []
    for i, r in direct_net.iterrows():
        tfs.append(r['TF motif'])
        tfs.append(r["Target_gene"])
    tfs = list(set(tfs))

    if add_figr:
        figr = pd.read_csv(os.path.join(DIRECT_NET_INDIR, "FigR_DORC_TF.csv"), header=0, index_col=0)
        figr.DORC = [i.upper() for i in figr.DORC]
        figr.Motif = [i.upper() for i in figr.Motif]

        for i, r in figr.iterrows():
            tfs.append(r['Motif'])
            tfs.append(r["DORC"])

    tfs = list(set(tfs))

    if extra_genes is not None:
        for g in extra_genes:
            tfs.append(g)

    print(tfs)
    print(len(tfs))

    overlap = (list(set(tfs).intersection(set([i.upper() for i in adata.var_names]))))
    print(list(set(tfs).difference(set([i.upper() for i in adata.var_names]))))

    if species == 'mouse':
        adata_net = adata[:, [i.capitalize() for i in overlap]]
    elif species == 'human':
        adata_net = adata[:, [i.upper() for i in overlap]]
    else:
        print("Species must be mouse or human.")
    print(adata_net)
    print(adata_net.layers[imputed_layer][0:10, 0:10])
    adata_imputed = pd.DataFrame(adata_net.layers[imputed_layer], index=adata_net.obs_names,
                                 columns=adata_net.var_names)
    adata_imputed.to_csv(f"./data/{outfile_name}")


# adata = cr.read('../data/M2/adata_04_nodub.h5ad')
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_04_nodubs_imputed_M2.csv")

# adata = cr.read('../data/combined/adata_02_filtered.h5ad')
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_imputed_combined.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'])
#
# adata = cr.read("../data/external_validation_looms/allografts.h5ad")
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_allografts.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'])


# adata = cr.read("../data/external_validation_looms/5B_allograftdata.h5ad")
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_5B_allografts.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'])

# for allo in ["1L","2L","2LR","3L","5B","TKO-luc","mt2","mt3","mt4","mt4Rf","mt5","mt6"]:
#     adata = cr.read(f"../data/external_validation_looms/{allo}_allograftdata.h5ad")
#     preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = f"adata_{allo}_allografts.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'])

# adata = cr.read(f"../data/external_validation_looms/adata.SCLC.010920.h5ad")
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name= f"adata_human_tumors_MSK.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'],
#                  imputed_layer="imputed_normalized", species = 'human')

# for tumor in ["PleuralEffusion", "RU426", "RU779", "RU1065", "RU1066", "RU1080", "RU1108", "RU1124", "RU1144", "RU1145",
#               "RU1152", "RU1181", "RU1195", "RU1215", "RU1229", "RU1231", "RU1293", "RU1311", "RU1322"]:
#     adata = cr.read(f"../data/external_validation_looms/human_tumors/{tumor}_human_tumor_data.h5ad")
#     preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name=f"human_tumors/adata_{tumor}.csv",
#                      extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1'], imputed_layer="imputed_normalized",
#                      species = 'human')

adata = cr.read('../data/combined/adata_02_filtered.h5ad')
# preprocess_adata(adata, DIRECT_NET_INDIR, outfile_name = "adata_imputed_combined_v2.csv", extra_genes=['CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1','SOX11'])

preprocess_adata(adata, DIRECT_NET_INDIR, Direct_net_file='Direct_net_pval.csv', outfile_name = "adata_imputed_combined_v3.csv",
                 extra_genes=['CD24A','CD44', 'EPCAM', 'ICAM1', 'NCAM1','SOX11', 'HES1', 'NFYC', 'NR6A1', 'RBPJ', 'TFDP1',
                              'ZBTB18'])