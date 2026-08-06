"""Fit CellOracle's ridge regression on Track 1's synthetic BoolODE data, using the SAME
literal true-edge candidate network boba-T got in comparison_hsc_fit_bobat.py -- the
missing half of the boba-T-vs-CellOracle comparison (Track 2's real-data half already
showed boba-T winning decisively; this checks whether that holds on the synthetic,
true-network-handed-directly setup too).

No real ATAC peaks exist for this synthetic system, so base_grn uses one placeholder
"peak" per target gene with a 0/1 column per candidate TF (exactly which edges are in
candidate_network.csv) -- this bypasses motif-scanning entirely, matching how Track 1
handed boba-T the true edges directly rather than making it discover them.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_hsc_fit_celloracle.py
"""
import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

GT_DIR = "data/hsc_ground_truth"
BRCD = "hsc"


def build_base_grn():
    edges = pd.read_csv(f"{GT_DIR}/candidate_network.csv", header=None, names=["source", "target"])
    genes = sorted(set(edges.source) | set(edges.target))
    wide = pd.crosstab(edges["target"], edges["source"]).clip(upper=1)
    for g in genes:
        if g not in wide.columns:
            wide[g] = 0
    wide = wide[genes]
    base_grn = wide.reset_index().rename(columns={"target": "gene_short_name"})
    base_grn.insert(0, "peak_id", base_grn["gene_short_name"] + "_pseudopeak")
    return base_grn, genes


def main():
    base_grn, genes = build_base_grn()
    print(f"[base_grn] {len(base_grn)} pseudo-peaks (1/target gene), {len(genes)} candidate TF columns")

    expr = pd.read_csv(f"{GT_DIR}/expr_bobat.csv", index_col=0)[genes]
    clusters = pd.read_csv(f"{GT_DIR}/clusters_bobat.csv", index_col=0)
    print(f"[data] {expr.shape[0]} cells x {expr.shape[1]} genes (already [0,1] from BoolODE min-max norm)")

    adata = ad.AnnData(
        X=expr.values.astype(np.float32),
        obs=clusters.reindex(expr.index).rename(columns={"class": "class"}),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.layers["raw_count"] = adata.X.copy()  # same CellOracle 0.20.0 gotcha as elsewhere in this project

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name="class", embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(50, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix
    coef.to_csv(f"{GT_DIR}/celloracle_hsc_coef_matrix.csv")
    print(f"\n[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")
    print(coef)


if __name__ == "__main__":
    main()
