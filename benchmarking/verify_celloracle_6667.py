"""Check whether comparison3_fit_celloracle_6667.py's R2=0.780 is affected by the same
representation mismatch found in the HSC verification (Ridge fit on oracle's internal
imputed_count, but evaluated by applying coef_matrix to raw/non-imputed test values).
Unlike the missing-intercept bug, to01() rescaling does NOT protect against this one --
it only makes the score invariant to a missing additive constant, not to evaluating on a
different data representation than the model was trained on.

Refits Ridge directly on the real imputed_count (extracted from a real fitted Oracle
object) for both train and test, same candidate network, same alpha, same held-out split
already used in the original 6667 comparison -- to see whether the number actually moves.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python verify_celloracle_6667.py
"""
import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
NETWORK_CSV = (
    f"{DN}/networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)


def to01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    clusters_train = pd.read_csv(f"{DN}/6667/data_split/clusters_traincombined.csv", index_col=0)
    clusters_test = pd.read_csv(f"{DN}/6667/data_split/clusters_testcombined.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes, {len(train)} train, {len(test)} test cells")

    edge_list = pd.read_csv(NETWORK_CSV, header=None, names=["source", "target"])
    edge_list = edge_list[edge_list.source.isin(genes) & edge_list.target.isin(genes)]
    base_grn = (
        edge_list.assign(v=1).pivot(index="target", columns="source", values="v")
        .reindex(columns=genes).fillna(0).reset_index()
        .rename(columns={"target": "gene_short_name"})
    )
    base_grn.insert(0, "peak_id", base_grn["gene_short_name"])

    # Build the Oracle on ALL cells (train+test combined) -- imputation is normally a
    # once-over-the-whole-dataset smoothing step, matching how the HSC verification and
    # CellOracle's own standard usage both do it.
    all_expr = pd.concat([train, test])
    all_clusters = pd.concat([clusters_train, clusters_test.rename(columns={clusters_test.columns[0]: clusters_train.columns[0]})])
    all_clusters = all_clusters.reindex(all_expr.index)

    adata = ad.AnnData(
        X=all_expr.values.astype(np.float32),
        obs=all_clusters,
        var=pd.DataFrame(index=all_expr.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.layers["raw_count"] = adata.X.copy()

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name=all_clusters.columns[0], embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(20, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    imputed = pd.DataFrame(oracle.adata.layers["imputed_count"], index=oracle.adata.obs_names, columns=oracle.adata.var_names)
    train_imp = imputed.loc[imputed.index.intersection(train.index)]
    test_imp = imputed.loc[imputed.index.intersection(test.index)]
    print(f"[imputed] {len(train_imp)} train / {len(test_imp)} test cells")
    test_imp.to_csv("benchmarking_out/verify_celloracle_6667_imputed_test.csv")

    tf_cols = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    grn_per_target = base_grn.groupby("gene_short_name")[tf_cols].max()

    rows = []
    for gene in genes:
        if gene not in grn_per_target.index:
            continue
        regs = [c for c in tf_cols if grn_per_target.loc[gene, c] == 1 and c != gene]
        if not regs:
            continue
        model = Ridge(alpha=10, random_state=123)
        model.fit(train_imp[regs], train_imp[gene])
        # (a) corrected: predict on test's OWN imputed_count, with intercept
        pred_correct = model.predict(test_imp[regs])
        r2_correct = r2_score(test_imp[gene], pred_correct)
        # (b) original script's approach: coef-only (no intercept) dot raw test values, then to01 both sides
        pred_original_style = to01((test[regs].to_numpy() * model.coef_).sum(axis=1))
        actual_original_style = to01(test[gene].to_numpy())
        r2_original_style = r2_score(actual_original_style, pred_original_style)
        rows.append({"gene": gene, "n_regs": len(regs), "r2_corrected": r2_correct, "r2_original_style_to01": r2_original_style})

    res = pd.DataFrame(rows).set_index("gene")
    print(res)
    print()
    print(f"mean r2_corrected (real imputed_count, with intercept): {res['r2_corrected'].mean():.3f}")
    print(f"mean r2_original_style_to01 (matches comparison3_fit_celloracle_6667.py's own method): {res['r2_original_style_to01'].mean():.3f}")
    res.to_csv("benchmarking_out/verify_celloracle_6667_representation_check.csv")


if __name__ == "__main__":
    main()
