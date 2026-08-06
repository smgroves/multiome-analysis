"""Fit CellOracle on TKO_final_arc's real expression data using the REAL ATAC-informed base
GRN built by comparison_tko_atac_base_grn.py (Signac peak-to-gene assignment from the
`peaks` ChromatinAssay + gimmemotifs mm10 motif scan -- 115,027 peaks x 1093 TFs x 19,455
target genes), not a generic genome-wide promoter scan. Mirrors
comparison_celloracle_fromscratch_fit_6667.py's "given all the data, own base GRN" pattern.

Inputs (preprocess_tko_rna.py's output, benchmarking/data/tko_full/):
    log_data.mtx    -- log1p-normalized expression, ALL filter_genes-passing genes x cells
    all_genes.csv   -- row order for log_data.mtx
    var_genes.csv   -- the 3000-HVG subset (CellOracle-paper recipe)
    meta_data.csv   -- obs metadata; `identity` column (NE_1..NE_11, Club cells, AT2 cells)
                       used as the cluster_column_name

base GRN gene symbols are Signac/Ensembl mouse case ("Ascl1"); TKO expression genes are
upper-cased ("ASCL1") to match this repo's human-convention gene-naming elsewhere -- upper-case
the base GRN's gene_short_name and TF columns to align before intersecting.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_tko_fit_celloracle_atac.py
"""

from __future__ import annotations

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

DATA_DIR = "data/tko_full"
DN = "../network-inference-DIRECT-NET"
BASE_GRN_PATH = f"{DN}/data/tko_atac/tko_atac_base_grn.parquet"
OUT_DIR = "data/tko_full"


def main():
    all_genes = pd.read_csv(f"{DATA_DIR}/all_genes.csv")["x"].tolist()
    var_genes = pd.read_csv(f"{DATA_DIR}/var_genes.csv")["x"].tolist()
    meta = pd.read_csv(f"{DATA_DIR}/meta_data.csv", index_col=0)

    log_data = scipy.io.mmread(f"{DATA_DIR}/log_data.mtx").tocsr()  # genes x cells
    print(f"[load] log_data {log_data.shape}, {len(all_genes)} genes, {len(var_genes)} HVGs")

    gene_idx = {g: i for i, g in enumerate(all_genes)}
    hvg_rows = [gene_idx[g] for g in var_genes]
    X = log_data[hvg_rows, :].T  # cells x HVGs
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    adata = ad.AnnData(X=X.astype(np.float32), obs=meta, var=pd.DataFrame(index=var_genes))
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.layers["raw_count"] = adata.X.copy()
    print(f"[data] {adata.n_obs} cells x {adata.n_vars} HVGs")

    base = pd.read_parquet(BASE_GRN_PATH)
    id_cols = base[["peak_id", "gene_short_name"]].assign(gene_short_name=lambda d: d["gene_short_name"].str.upper())
    tf_part = base.drop(columns=["peak_id", "gene_short_name"])
    tf_part.columns = tf_part.columns.str.upper()
    tf_part = tf_part.T.groupby(level=0).max().T  # collapse TF columns that collided under upper-casing
    base = pd.concat([id_cols, tf_part], axis=1)

    keep_tfs = sorted(set(tf_part.columns) & set(var_genes))
    base_grn = base[base["gene_short_name"].isin(var_genes)][["peak_id", "gene_short_name"] + keep_tfs]
    n_edges = int((base_grn[keep_tfs].values == 1).sum())
    print(f"[base_grn] restricted to this HVG set: {len(keep_tfs)} candidate TFs, "
          f"{base_grn['gene_short_name'].nunique()} target genes, {n_edges} nonzero peak-TF entries")

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name="identity", embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(20, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix
    coef.to_csv(f"{OUT_DIR}/celloracle_tko_atac_coef_matrix.csv")
    print(f"[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")
    print(f"-> {OUT_DIR}/celloracle_tko_atac_coef_matrix.csv")


if __name__ == "__main__":
    main()
