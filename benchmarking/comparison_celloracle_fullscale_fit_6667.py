"""Comparison 1, the CORRECTED "run from the beginning" step: fit CellOracle on the real,
genome-wide SCLC/AA data behind 6667 (preprocess_sclc_full_6667.py's output -- 2999 HVGs,
same 8908 cells, raw counts, NOT the 53-gene boba-T export the earlier from-scratch attempt
mistakenly used). Uses CellOracle's own raw-count import path (`import_anndata_as_raw_count`)
since we now have real raw UMI counts, and its own human promoter base GRN -- no DIRECT-NET
involvement at any point.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_celloracle_fullscale_fit_6667.py
"""

from __future__ import annotations

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

DATA_DIR = "data/sclc_full"
OUT_PATH = "../network-inference-DIRECT-NET/6667/rules/celloracle_fullscale_coef_matrix.csv"


def main():
    raw = pd.read_csv(f"{DATA_DIR}/raw_counts_hvg.csv", index_col=0)
    meta = pd.read_csv(f"{DATA_DIR}/meta_data.csv", index_col=0)
    meta["class"] = meta["S_0.5threshold"]  # Generalist/Arc_1..6 -- same phenotype convention as 6667's own "class"
    genes = list(raw.columns)
    print(f"[data] {len(genes)} genes (HVGs), {len(raw)} cells")

    base = co.data.load_human_promoter_base_GRN()
    tf_cols = [c for c in base.columns if c not in ("peak_id", "gene_short_name")]
    keep_tfs = [c for c in tf_cols if c in genes]
    base_grn = base[base["gene_short_name"].isin(genes)][["peak_id", "gene_short_name"] + keep_tfs]
    print(f"[base_grn] {len(keep_tfs)} candidate TFs, {base_grn['gene_short_name'].nunique()} target genes matched")

    adata = ad.AnnData(
        X=raw.values.astype(np.float32),
        obs=meta.reindex(raw.index),
        var=pd.DataFrame(index=raw.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)

    oracle = co.Oracle()
    # Real raw counts this time -- CellOracle's actual tutorial-standard entry point (it does
    # its own log-normalization internally), unlike the earlier from-scratch attempt which had
    # to fake this layer because only pre-normalized data was available.
    oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="class",
                                        embedding_name="dummy", transform="log2")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(50, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix
    coef.to_csv(OUT_PATH)
    print(f"[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")

    ascl1_targets = coef.loc["ASCL1"][coef.loc["ASCL1"] != 0] if "ASCL1" in coef.index else None
    if ascl1_targets is not None and len(ascl1_targets):
        print(f"[ASCL1] {len(ascl1_targets)} nonzero outgoing edges; top 10 by |weight|:")
        print(ascl1_targets.abs().sort_values(ascending=False).head(10))
    else:
        print("[ASCL1] zero nonzero outgoing edges")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
