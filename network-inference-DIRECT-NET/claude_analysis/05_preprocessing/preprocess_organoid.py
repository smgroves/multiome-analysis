"""Preprocess Organoid_full.h5ad for external validation, matching preprocess_adata.py's
convention for the allograft datasets (same gene panel: DIRECT-NET TF/target genes + FigR
DORC-TF genes + the same fixed extra_genes list used for every other external validation
set in this project): QC filter -> normalize by total UMI count -> log1p -> restrict to
network genes -> MAGIC impute -> add RORA_RORB.

IMPORTANT, caught in review: an earlier version of this script used the `logcounts` layer
assuming it was already log-transformed -- it isn't. Checked directly: `logcounts` is
byte-identical to `X`, and both are raw integer UMI counts (e.g. ASCL1 took only 32
distinct values, all whole numbers 0-31). This produced near-discrete columns that broke
boba-T's [0,1] `norm=0.3` rescaling. This script now runs the real pipeline the allografts'
own preprocessing notebook used (archetype-plasticity-notebooks/external_network_
validation/1-preprocess-tumor-datasets.ipynb): normalize + log1p + MAGIC (same package,
same `solver='approximate'`), not a re-derivation of that notebook's private `mazebox`
QC/filtering step -- percent.mt<20 and nFeature_RNA>200 are used as a standard,
documented substitute (both already look pre-filtered: percent.mt max ~15.9%, min
nFeature_RNA 501, so this is expected to drop few or zero cells).

Note: `condition` in this dataset is shGFP/shRORB1/shRORB2 -- a real RORB-knockdown
organoid experiment, directly relevant as external validation for boba-T's RORA_RORB
perturbation predictions (see 6667/perturbations, walk_to_basin knockdowns).

Run in bobaT_env (has `magic`; celloracle_env has anndata but not `magic`):
    /opt/anaconda3/envs/bobaT_env/bin/python claude_analysis/05_preprocessing/preprocess_organoid.py
"""

import os

import anndata as ad
import magic
import numpy as np
import pandas as pd

BOX = "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/from zenodo"
DIRECT_NET_INDIR = "./DIRECT-NET-FILES/"
OUT_DIR = "data/organoid"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    adata = ad.read_h5ad(os.path.join(BOX, "Organoid_full.h5ad"))
    print(f"Loaded: {adata.shape}")

    keep = (adata.obs["percent.mt"] < 20) & (adata.obs["nFeature_RNA"] > 200)
    print(f"QC filter (percent.mt<20, nFeature_RNA>200): {keep.sum()}/{len(keep)} cells kept")
    adata = adata[keep.values].copy()

    direct_net = pd.read_csv(os.path.join(DIRECT_NET_INDIR, "Direct_net.csv"), header=0, index_col=0)
    direct_net["Target_gene"] = [g.upper() for g in direct_net["Target_gene"]]
    tfs = set(direct_net["TF motif"]).union(direct_net["Target_gene"])

    figr = pd.read_csv(os.path.join(DIRECT_NET_INDIR, "FigR_DORC_TF.csv"), header=0, index_col=0)
    tfs = tfs.union(g.upper() for g in figr["Motif"]).union(g.upper() for g in figr["DORC"])

    extra_genes = ["CD24", "CD44", "EPCAM", "ICAM1", "NCAM1", "HES1", "NFYC", "NR6A1",
                   "RBPJ", "RORA", "RORB", "SOX11", "TFDP1"]
    tfs = {g.upper() for g in tfs} | set(extra_genes)
    print(f"{len(tfs)} candidate genes (network + FigR + extra_genes)")

    all_genes_upper = {g.upper(): g for g in adata.var_names}
    overlap = sorted(set(all_genes_upper) & tfs)
    missing = sorted(tfs - set(all_genes_upper))
    print(f"{len(overlap)} of {len(tfs)} matched in Organoid_full (mouse gene symbols)")
    print("not found:", missing)

    # `logcounts` is actually raw counts (see docstring) -- normalize by total UMI count
    # (genome-wide nCount_RNA, already computed by Seurat) to a common target sum, then
    # log1p, same normalization family as the allografts' log1p_norm step.
    organoid_genes = [all_genes_upper[g] for g in overlap]
    raw = adata[:, organoid_genes].layers["logcounts"]
    raw = raw.toarray() if hasattr(raw, "toarray") else raw
    raw = pd.DataFrame(raw, index=adata.obs_names, columns=[g.upper() for g in organoid_genes])

    size_factor = adata.obs["nCount_RNA"] / adata.obs["nCount_RNA"].median()
    normalized = raw.div(size_factor.values, axis=0)
    log_normalized = np.log1p(normalized)
    print(f"Normalized + log1p. Value range sample (ASCL1): "
          f"{log_normalized['ASCL1'].min():.3f} to {log_normalized['ASCL1'].max():.3f}, "
          f"{log_normalized['ASCL1'].nunique()} distinct values")

    magic_operator = magic.MAGIC(solver="approximate")
    imputed = magic_operator.fit_transform(log_normalized)
    print(f"MAGIC done: {imputed.shape}")

    if "RORA" in imputed.columns and "RORB" in imputed.columns:
        imputed["RORA_RORB"] = imputed[["RORA", "RORB"]].mean(axis=1)

    imputed.index.name = "CellID"
    imputed.to_csv(f"{OUT_DIR}/adata_organoid_v3_RORA_RORB_ave.csv")
    adata.obs.to_csv(f"{OUT_DIR}/organoid_clusters.csv")
    print(f"Wrote {OUT_DIR}/adata_organoid_v3_RORA_RORB_ave.csv: {imputed.shape}")


if __name__ == "__main__":
    main()
