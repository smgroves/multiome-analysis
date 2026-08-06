"""Redo CellOracle's R2 evaluation correctly, after finding two real gaps in the earlier
verification: (1) CellOracle's own _getCoefMatrix fits sklearn Ridge with the DEFAULT
fit_intercept=True but only ever saves model.coef_ into coef_matrix -- model.intercept_ is
silently discarded by CellOracle's own code, never persisted anywhere; (2) the ridge is fit
on oracle.adata.layers["imputed_count"] (KNN-smoothed, from knn_imputation()), not the raw
input expression -- the earlier R2 check evaluated against raw expression, a representation
mismatch on top of the missing intercept.

This script refits Ridge itself (same candidate TF sets, same alpha) on TRAIN cells' real
imputed_count (extracted directly from a fitted Oracle object), capturing both coef_ and
intercept_ this time, then evaluates on TEST cells' imputed_count with both applied --
correcting both gaps at once. Run separately for Track 1 (synthetic) and Track 2 (real
Cicero network) via command-line args.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python verify_celloracle_fit.py track1
    /opt/anaconda3/envs/celloracle_env/bin/python verify_celloracle_fit.py track2
"""
import sys

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def build_base_grn_track1():
    edges = pd.read_csv("data/hsc_ground_truth/candidate_network.csv", header=None, names=["source", "target"])
    genes = sorted(set(edges.source) | set(edges.target))
    wide = pd.crosstab(edges["target"], edges["source"]).clip(upper=1)
    for g in genes:
        if g not in wide.columns:
            wide[g] = 0
    wide = wide[genes]
    base_grn = wide.reset_index().rename(columns={"target": "gene_short_name"})
    base_grn.insert(0, "peak_id", base_grn["gene_short_name"] + "_pseudopeak")
    return base_grn, genes


def build_base_grn_track2():
    detail = pd.read_csv("data/hsc_multiome/candidate_network_cicero_real_detail.csv")
    peak_target = detail.groupby("peak")["target"].first()
    panel = ["GATA1", "GATA2", "PU1", "FLI1", "CEBPA", "EKLF", "GFI1", "FOG1", "SCL", "CJUN"]
    wide = pd.crosstab(detail["peak"], detail["source"]).clip(upper=1)
    for tf in panel:
        if tf not in wide.columns:
            wide[tf] = 0
    wide = wide[panel]
    base_grn = wide.reset_index().rename(columns={"peak": "peak_id"})
    base_grn["gene_short_name"] = base_grn["peak_id"].map(peak_target)
    return base_grn[["peak_id", "gene_short_name"] + panel], panel


def run(track):
    if track == "track1":
        base_grn, genes = build_base_grn_track1()
        expr = pd.read_csv("data/hsc_ground_truth/expr_bobat.csv", index_col=0)[genes]
        clusters = pd.read_csv("data/hsc_ground_truth/clusters_bobat.csv", index_col=0)
        test_cells = pd.read_csv("hsc/data_split/test_t0hsc.csv", index_col=0).index
        alpha = 10
    else:
        base_grn, genes = build_base_grn_track2()
        expr = pd.read_csv("data/hsc_multiome/expr_bobat_real.csv", index_col=0)[genes].clip(lower=0)
        clusters = pd.read_csv("data/hsc_multiome/clusters_bobat_real.csv", index_col=0)
        test_cells = pd.read_csv("hsc_multiome_cicero/data_split/test_t0hsc_multiome_cicero.csv", index_col=0).index
        alpha = 10

    common_idx = expr.index.intersection(clusters.index)
    expr = expr.loc[common_idx]
    clusters = clusters.loc[common_idx]

    adata = ad.AnnData(
        X=expr.values.astype(np.float32),
        obs=clusters.rename(columns={clusters.columns[0]: "class"}),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.layers["raw_count"] = adata.X.copy()

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name="class", embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(50, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    # The real, real imputed_count CellOracle's own Ridge is actually fit on -- extracted
    # directly from the fitted Oracle object, not re-derived or approximated.
    imputed = pd.DataFrame(
        oracle.adata.layers["imputed_count"], index=oracle.adata.obs_names, columns=oracle.adata.var_names
    )
    print(f"[{track}] imputed_count: {imputed.shape}")

    test_idx = imputed.index.intersection(test_cells)
    train_idx = imputed.index.difference(test_idx)
    train, test = imputed.loc[train_idx], imputed.loc[test_idx]
    print(f"[{track}] {len(train)} train / {len(test)} test cells (matching boba-T's own split)")

    tf_cols = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    grn_per_target = base_grn.groupby("gene_short_name")[tf_cols].max()

    rows = []
    for gene in genes:
        if gene not in grn_per_target.index:
            continue
        regs = [c for c in tf_cols if grn_per_target.loc[gene, c] == 1 and c != gene]
        if not regs:
            continue
        model = Ridge(alpha=alpha, random_state=123)
        model.fit(train[regs], train[gene])
        pred = model.predict(test[regs])
        r2 = r2_score(test[gene], pred)
        rows.append({"gene": gene, "n_regs": len(regs), "intercept": model.intercept_, "r2_with_intercept": r2})

    res = pd.DataFrame(rows).set_index("gene")
    print(res)
    print(f"\nmean R2 (with intercept, on imputed_count): {res['r2_with_intercept'].mean():.3f}")
    res.to_csv(f"benchmarking_out/comparison_{track}_celloracle_r2_corrected.csv")


if __name__ == "__main__":
    run(sys.argv[1])
