"""Preprocess TKO_final_arc's RNA data (extracted by extract_tko_atac.R) the same way
preprocess_sclc_full_6667.py did for the "combined" fullscale run: CellOracle-paper HVG
recipe (filter_genes min_counts=1 -> normalize_per_cell -> filter_genes_dispersion
cell_ranger ~3000 -> log1p), no manual gene injection. Mouse gene symbols upper-cased to
match this benchmark's human-convention naming (same as everywhere else in this repo).

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python preprocess_tko_rna.py
"""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

DATA_DIR = "../network-inference-DIRECT-NET/data/tko_atac"
OUT_DIR = "data/tko_full"
N_TOP_GENES = 3000


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = scipy.io.mmread(f"{DATA_DIR}/raw_counts.mtx").T.tocsr().astype(np.float32)  # -> cells x genes
    genes = pd.read_csv(f"{DATA_DIR}/genes.csv")["gene"].tolist()
    cells = pd.read_csv(f"{DATA_DIR}/cells.csv")["cell"].tolist()
    meta = pd.read_csv(f"{DATA_DIR}/tko_clusters.csv", index_col=0)
    print(f"[load] {raw.shape[0]} cells x {raw.shape[1]} genes")

    adata = ad.AnnData(X=raw, obs=meta.reindex(cells))
    adata.var_names = [g.upper() for g in genes]
    adata.var_names_make_unique()
    adata.layers["raw_counts"] = adata.X.copy()

    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata, key_n_counts="n_counts_all")
    filt = sc.pp.filter_genes_dispersion(
        adata.X, flavor="cell_ranger", n_top_genes=min(N_TOP_GENES, adata.n_vars), log=False
    )
    var_genes = adata.var_names[filt.gene_subset].tolist()
    sc.pp.log1p(adata)

    print(f"[hvg] {len(var_genes)} HVGs selected")
    print(f"[hvg] ASCL1 in HVG set: {'ASCL1' in var_genes}")

    X = adata.X.T
    scipy.io.mmwrite(f"{OUT_DIR}/log_data.mtx", sp.csr_matrix(X) if not sp.issparse(X) else X)
    pd.DataFrame({"x": adata.var_names.tolist()}).to_csv(f"{OUT_DIR}/all_genes.csv", index=False)
    pd.DataFrame({"x": var_genes}).to_csv(f"{OUT_DIR}/var_genes.csv", index=False)
    adata.obs.to_csv(f"{OUT_DIR}/meta_data.csv")

    hvg_idx = [adata.var_names.get_loc(g) for g in var_genes]
    raw_hvg = adata.layers["raw_counts"][:, hvg_idx]
    raw_hvg = raw_hvg.toarray() if sp.issparse(raw_hvg) else np.asarray(raw_hvg)
    raw_df = pd.DataFrame(raw_hvg, index=adata.obs_names, columns=var_genes)
    raw_df.index.name = "CellID"
    raw_df.to_csv(f"{OUT_DIR}/raw_counts_hvg.csv")
    print(f"[write] -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
