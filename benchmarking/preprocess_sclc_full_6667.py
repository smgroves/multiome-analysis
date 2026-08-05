"""Preprocess the REAL, genome-wide SCLC/AA multiome data behind boba-T run 6667, for a
genuinely independent CellOracle/SCENIC "run from the beginning" (see benchmarking/README.md,
"Sourcing a real SCLC ground truth" -- the earlier from-scratch runs were caught, in review,
to still be restricted to boba-T's 53-gene exports; this uses the actual source data).

Source: Box (`~/Box/multiome_data/Allograft_mnn.rds` per data/converting_seurat_data.R),
mounted at:
    /Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/combined/adata_02_filtered.h5ad
8908 cells x 16719 genes -- the cell count matches 6667's train+test split (7126+1782)
exactly, confirming this is the same cell population, just not gene-restricted. Gene
symbols are MOUSE case convention (allograft = mouse-grown tumor, e.g. "Ascl1" not
"ASCL1") -- upper-cased below to match the human-convention symbols used everywhere else
in this benchmark (same convention already used for the RPR2-mouse ChIP-seq ground truth).

Uses the `raw_counts` layer (real raw UMI counts) -- NOT `imputed`/`imputed_unscaled`
(MAGIC-imputed; CellOracle's own tutorial wants raw or normalized-but-not-imputed data,
since it does its own kNN imputation internally) and NOT `X` (ambiguous/scaled).

Reproduces CellOracle's own documented preprocessing recipe (same one already implemented
for the mouse Tabula Muris benchmark in grn_benchmark/preprocess.py): filter_genes(
min_counts=1) -> normalize_per_cell -> filter_genes_dispersion(cell_ranger, ~3000 HVGs) ->
log1p. HVGs are selected purely by the statistic, with NO manual inclusion of ASCL1 or any
other `6667`-network gene -- whether they survive the cut on their own is itself part of
what the from-scratch comparison is checking.

Run in celloracle_env (needs scanpy):
    /opt/anaconda3/envs/celloracle_env/bin/python preprocess_sclc_full_6667.py
"""

import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

BOX_H5AD = (
    "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/"
    "data_multiome/combined/adata_02_filtered.h5ad"
)
OUT_DIR = "data/sclc_full"
N_TOP_GENES = 3000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[load] reading {BOX_H5AD} (large; first read from Box is slow, this run should be cached)")
    full = ad.read_h5ad(BOX_H5AD)
    print(f"[load] {full.shape[0]} cells x {full.shape[1]} genes")

    adata = ad.AnnData(X=full.layers["raw_counts"].copy(), obs=full.obs.copy())
    adata.var_names = [str(g).upper() for g in full.var_names]  # mouse -> human-convention case
    adata.var_names_make_unique()
    adata.layers["raw_counts"] = adata.X.copy()  # preserved separately; sc.pp below only touches X
    del full

    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata, key_n_counts="n_counts_all")
    filt = sc.pp.filter_genes_dispersion(
        adata.X, flavor="cell_ranger", n_top_genes=min(N_TOP_GENES, adata.n_vars), log=False
    )
    var_genes = adata.var_names[filt.gene_subset].tolist()
    sc.pp.log1p(adata)

    network_genes = "ASCL1,BACH1,BACH2,CREB1,CUX2,EGR1,EHF,EPCAM,ESR1,ETS1,FOS,FOXO3,GRHL2,HES1,HSF2,ICAM1,JUN,JUNB,JUND,KMT2A,LMX1B,MEIS2,NCAM1,NFATC2,NFIA,NFIB,NFIX,NFKB1,NFYC,NR6A1,PBX1,PKNOX2,PROX1,RBPJ,REST,RORA_RORB,RUNX1,SIX1,SIX4,SMAD3,SOX11,SOX9,STAT1,STAT2,TBX15,TCF4,TCF7L1,TCF7L2,TEAD1,TFDP1,THRB,ZBTB20,ZEB1".split(",")
    in_hvg = [g for g in network_genes if g in var_genes]
    not_in_hvg = [g for g in network_genes if g not in var_genes and g in adata.var_names]
    not_present_at_all = [g for g in network_genes if g not in adata.var_names]
    print(f"[hvg] {len(var_genes)} HVGs selected (target {N_TOP_GENES})")
    print(f"[hvg] {len(in_hvg)}/{len(network_genes)} 6667-network genes made the HVG cut: {in_hvg}")
    print(f"[hvg] {len(not_in_hvg)} present in the data but NOT selected as HVG: {not_in_hvg}")
    print(f"[hvg] {len(not_present_at_all)} not present in the data at all (likely RORA_RORB / other renamed nodes): {not_present_at_all}")
    print(f"[hvg] ASCL1 specifically -- in HVG set: {'ASCL1' in var_genes}, present in data: {'ASCL1' in adata.var_names}")

    # log_data: log1p(normalized) over ALL kept genes (genes x cells .mtx, matches
    # grn_benchmark/preprocess.py's convention so existing tooling can read it unmodified).
    X = adata.X.T
    scipy.io.mmwrite(f"{OUT_DIR}/log_data.mtx", sp.csr_matrix(X) if not sp.issparse(X) else X)
    pd.DataFrame({"x": adata.var_names.tolist()}).to_csv(f"{OUT_DIR}/all_genes.csv", index=False)
    pd.DataFrame({"x": var_genes}).to_csv(f"{OUT_DIR}/var_genes.csv", index=False)
    adata.obs.to_csv(f"{OUT_DIR}/meta_data.csv")

    # Raw counts restricted to the HVG set, for methods (CellOracle via
    # import_anndata_as_raw_count, GRNBoost2) that want raw counts rather than log_data.
    hvg_idx = [adata.var_names.get_loc(g) for g in var_genes]
    raw = adata.layers["raw_counts"][:, hvg_idx]
    raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
    raw_hvg = pd.DataFrame(raw, index=adata.obs_names, columns=var_genes)
    raw_hvg.index.name = "CellID"
    raw_hvg.to_csv(f"{OUT_DIR}/raw_counts_hvg.csv")
    print(f"[write] -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
