"""Extends redo_comparison3_6667_fair.py with a 4th method: CellOracle run the way it would
actually be used in the wild -- its own real, dataset-specific ATAC-informed base GRN
(comparison_tko_atac_base_grn.py's TKO_final_arc peaks-derived base GRN), not DIRECT-NET's
228-edge LASSO-restricted candidate network. Same fair-scoring principle as the original
script: every method is fit/evaluated against the identical `imputed_count` target, since
CellOracle's own Ridge is fit on that representation internally and comparing across a
representation mismatch understates/overstates accuracy for reasons unrelated to model
quality (see redo_comparison3_6667_fair.py's docstring for the full story).

The `imputed_count` computation itself (PCA + KNN imputation) does not depend on which base
GRN is later used for the regression step, so it's safe to reuse the *same* train_imp/
test_imp this script builds for CellOracle-DIRECT-NET/GENIE3/boba-T, then additionally fit a
plain sklearn Ridge (alpha=10, with intercept -- same choice as the original fair rescore's
CellOracle refit) using the real-ATAC base GRN's candidate regulators per target instead of
DIRECT-NET's.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python redo_comparison3_6667_fair_with_realatac.py
"""
import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score, roc_auc_score

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
NETWORK_CSV = (
    f"{DN}/networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
REAL_ATAC_BASE_GRN = f"{DN}/data/tko_atac/tko_atac_base_grn.parquet"


def idx2binary(idx, n):
    b = "{0:b}".format(idx)
    return "0" * (n - len(b)) + b


def load_fitted_rules(path):
    out = {}
    with open(path) as f:
        for line in f:
            gene, regs, probs = line.strip().split("|")
            out[gene] = (regs.split(","), [float(p) for p in probs.split(",")])
    return out


def bobat_predict(data, fitted_regulators, rule):
    n = len(fitted_regulators)
    reg_vals = data[fitted_regulators].to_numpy()
    heat = np.ones((reg_vals.shape[0], 2 ** n))
    for leaf in range(2 ** n):
        binary = idx2binary(leaf, n)
        for col, bit in enumerate(binary):
            heat[:, leaf] *= reg_vals[:, col] if bit == "1" else (1 - reg_vals[:, col])
    return heat @ np.array(rule)


def to01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def fit_and_score(candidates_per_target, genes, train_imp, test_imp, model_fn):
    genes_scored, pred, actual = set(), {}, {}
    for gene in genes:
        regs = candidates_per_target.get(gene, [])
        regs = [r for r in regs if r in train_imp.columns and r != gene]
        if not regs:
            continue
        model = model_fn()
        model.fit(train_imp[regs], train_imp[gene])
        pred[gene] = model.predict(test_imp[regs])
        actual[gene] = test_imp[gene].to_numpy()
        genes_scored.add(gene)
    return genes_scored, pred, actual


def main():
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    clusters_train = pd.read_csv(f"{DN}/6667/data_split/clusters_traincombined.csv", index_col=0)
    clusters_test = pd.read_csv(f"{DN}/6667/data_split/clusters_testcombined.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes, {len(train)} train, {len(test)} test cells")

    # DIRECT-NET candidate map (needed to build the Oracle/import_TF_data call -- the base
    # GRN passed here does NOT affect the imputed_count computed below, only the later
    # regression step, so any valid TF_info_matrix works for this part).
    edge_list = pd.read_csv(NETWORK_CSV, header=None, names=["source", "target"])
    edge_list = edge_list[edge_list.source.isin(genes) & edge_list.target.isin(genes)]
    directnet_base_grn = (
        edge_list.assign(v=1).pivot(index="target", columns="source", values="v")
        .reindex(columns=genes).fillna(0).reset_index()
        .rename(columns={"target": "gene_short_name"})
    )
    directnet_base_grn.insert(0, "peak_id", directnet_base_grn["gene_short_name"])
    directnet_tf_cols = [c for c in directnet_base_grn.columns if c not in ("peak_id", "gene_short_name")]
    directnet_per_target = directnet_base_grn.groupby("gene_short_name")[directnet_tf_cols].max()
    directnet_candidates = {
        g: [c for c in directnet_tf_cols if directnet_per_target.loc[g, c] == 1] if g in directnet_per_target.index else []
        for g in genes
    }

    # Real ATAC-informed base GRN, restricted to 6667's 53 genes -- upper-cased to match
    # (Signac/Ensembl mouse case -> this repo's human-convention upper-case symbols).
    real_atac = pd.read_parquet(REAL_ATAC_BASE_GRN)
    id_cols = real_atac[["peak_id", "gene_short_name"]].assign(gene_short_name=lambda d: d["gene_short_name"].str.upper())
    tf_part = real_atac.drop(columns=["peak_id", "gene_short_name"])
    tf_part.columns = tf_part.columns.str.upper()
    tf_part = tf_part.T.groupby(level=0).max().T
    real_atac = pd.concat([id_cols, tf_part], axis=1)
    keep_tfs = sorted(set(tf_part.columns) & set(genes))
    real_atac = real_atac[real_atac["gene_short_name"].isin(genes)][["peak_id", "gene_short_name"] + keep_tfs]
    real_atac_per_target = real_atac.groupby("gene_short_name")[keep_tfs].max()
    real_atac_candidates = {
        g: [c for c in keep_tfs if real_atac_per_target.loc[g, c] == 1] if g in real_atac_per_target.index else []
        for g in genes
    }
    print(f"[real_atac base_grn] {len(keep_tfs)}/{len(genes)} candidate TFs, "
          f"{real_atac['gene_short_name'].nunique()}/{len(genes)} target genes matched")

    # ONE real Oracle, built on train+test combined -- the single source of imputed_count
    # every method (including the new real-ATAC CellOracle variant) is scored against.
    all_expr = pd.concat([train, test])
    all_clusters = pd.concat([
        clusters_train, clusters_test.rename(columns={clusters_test.columns[0]: clusters_train.columns[0]})
    ]).reindex(all_expr.index)

    adata = ad.AnnData(X=all_expr.values.astype(np.float32), obs=all_clusters, var=pd.DataFrame(index=all_expr.columns))
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.layers["raw_count"] = adata.X.copy()

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name=all_clusters.columns[0], embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=directnet_base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(20, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)
    imputed = pd.DataFrame(oracle.adata.layers["imputed_count"], index=oracle.adata.obs_names, columns=oracle.adata.var_names)
    train_imp = imputed.loc[imputed.index.intersection(train.index)]
    test_imp = imputed.loc[imputed.index.intersection(test.index)]
    print(f"[imputed] {len(train_imp)} train / {len(test_imp)} test cells (single shared target for all methods)")

    # boba-T: already-fitted rule, just re-evaluated against the shared target.
    fitted_bobat = load_fitted_rules(f"{DN}/6667/rules/rules_6667.txt")
    bobat_genes, bobat_pred, bobat_actual = set(), {}, {}
    for gene, (regs, rule) in fitted_bobat.items():
        regs = [r for r in regs if r in train_imp.columns]
        if not regs or gene not in test_imp.columns:
            continue
        bobat_pred[gene] = bobat_predict(test_imp, regs, rule)
        bobat_actual[gene] = test_imp[gene].to_numpy()
        bobat_genes.add(gene)

    co_genes, co_pred, co_actual = fit_and_score(
        directnet_candidates, genes, train_imp, test_imp, lambda: Ridge(alpha=10, random_state=123)
    )
    genie3_genes, genie3_pred, genie3_actual = fit_and_score(
        directnet_candidates, genes, train_imp, test_imp,
        lambda: RandomForestRegressor(n_estimators=1000, max_features="sqrt", random_state=0, n_jobs=-1),
    )
    realatac_genes, realatac_pred, realatac_actual = fit_and_score(
        real_atac_candidates, genes, train_imp, test_imp, lambda: Ridge(alpha=10, random_state=123)
    )

    shared = sorted(bobat_genes & co_genes & genie3_genes)
    realatac_covered = sorted(set(shared) & realatac_genes)
    print(f"\n[shared] {len(shared)} genes with surviving candidates in boba-T/CellOracle-DIRECT-NET/GENIE3")
    print(f"[real_atac coverage] {len(realatac_covered)}/{len(shared)} of those genes also have a "
          f"surviving real-ATAC-base candidate regulator")

    rows = []
    methods = [
        ("boba-T", bobat_pred, bobat_actual, shared),
        ("CellOracle (DIRECT-NET base)", co_pred, co_actual, shared),
        ("GENIE3 (DIRECT-NET-restricted)", genie3_pred, genie3_actual, shared),
        ("CellOracle (real ATAC base, not DIRECT-NET-restricted)", realatac_pred, realatac_actual, realatac_covered),
    ]
    for method, pred_d, actual_d, gene_set in methods:
        r2s, aucs, f1s = [], [], []
        for gene in gene_set:
            actual01 = to01(actual_d[gene])
            pred01 = to01(pred_d[gene])
            r2s.append(r2_score(actual_d[gene], pred_d[gene]))
            try:
                aucs.append(roc_auc_score((actual01 > 0.5).astype(int), pred01))
            except ValueError:
                pass
            f1s.append(f1_score((actual01 > 0.5).astype(int), (pred01 > 0.5).astype(int)))
        rows.append({"method": method, "n_genes": len(gene_set), "mean_r2": np.mean(r2s),
                     "mean_auc": np.mean(aucs), "mean_f1": np.mean(f1s)})

    res = pd.DataFrame(rows).set_index("method")
    print(res)
    res.to_csv("benchmarking_out/comparison3_all_methods_vs_bobat_6667_fair_imputed_target_with_realatac.csv")


if __name__ == "__main__":
    main()
