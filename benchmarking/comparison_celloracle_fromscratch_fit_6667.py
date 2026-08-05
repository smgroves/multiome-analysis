"""Comparison 1, "run from the beginning" step: fit CellOracle on 6667's full expression
data using its OWN base GRN (`co.data.load_human_promoter_base_GRN()` -- a genome-wide
promoter motif scan, hg19/gimmemotifs v5, entirely independent of DIRECT-NET), instead of
the DIRECT-NET-restricted base GRN used in comparison3_fit_celloracle_6667.py.

This is the CellOracle counterpart to comparison_scenic_fit_6667.py: both are the "given
all the data, not the DIRECT-NET starter network" versions of comparison 1, run alongside
the already-existing DIRECT-NET-restricted CellOracle/GENIE3 fits for contrast. Uses ALL
of 6667's cells (train+test), since this is structure-only -- no held-out prediction step
here, unlike comparison3_fit_celloracle_6667.py.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_celloracle_fromscratch_fit_6667.py
"""

from __future__ import annotations

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"


def main():
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    clusters_train = pd.read_csv(f"{DN}/6667/data_split/clusters_traincombined.csv", index_col=0)
    clusters_test = pd.read_csv(f"{DN}/6667/data_split/clusters_testcombined.csv", index_col=0)
    full = pd.concat([train, test])  # "given all the data" -- no held-out split needed for structure-only inference
    clusters = pd.concat([clusters_train, clusters_test]).reindex(full.index)
    genes = list(full.columns)
    print(f"[data] {len(genes)} genes, {len(full)} cells (train+test combined)")

    # CellOracle's own base GRN, not DIRECT-NET's -- restricted only to genes present in
    # this expression matrix (both as target and as candidate TF), on both ends.
    base = co.data.load_human_promoter_base_GRN()
    tf_cols = [c for c in base.columns if c not in ("peak_id", "gene_short_name")]
    keep_tfs = [c for c in tf_cols if c in genes]
    base_grn = base[base["gene_short_name"].isin(genes)][["peak_id", "gene_short_name"] + keep_tfs]
    n_candidate_edges = int((base_grn[keep_tfs].values == 1).sum())
    print(f"[base_grn] human promoter base GRN restricted to this gene set: "
          f"{len(keep_tfs)} candidate TFs, {base_grn['gene_short_name'].nunique()} target genes, "
          f"{n_candidate_edges} nonzero peak-TF entries before per-gene aggregation")

    adata = ad.AnnData(
        X=full.values.astype(np.float32),
        obs=clusters,
        var=pd.DataFrame(index=full.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)  # see comparison3_fit_celloracle_6667.py
    adata.layers["raw_count"] = adata.X.copy()  # see comparison3_fit_celloracle_6667.py

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name="class", embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(20, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix
    coef.to_csv(f"{DN}/6667/rules/celloracle_fromscratch_coef_matrix.csv")
    print(f"[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")
    print(f"-> {DN}/6667/rules/celloracle_fromscratch_coef_matrix.csv")


if __name__ == "__main__":
    main()
