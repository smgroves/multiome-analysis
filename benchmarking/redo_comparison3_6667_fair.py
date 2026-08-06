"""Full redo of comparison 3 (6667, predicted vs. actual expression) against a single,
consistent evaluation target for all three methods -- the fair three-way re-score flagged
as a follow-up after finding that CellOracle's original R2 (0.780) was computed against a
different data representation (raw test values) than its Ridge model was actually trained
on (oracle.adata.layers["imputed_count"], KNN-smoothed).

Rather than picking one method's preferred representation, all three methods (boba-T,
CellOracle, GENIE3) are scored against the SAME real imputed_count test target here --
the methodologically correct notion of "fair": whatever representation is used, it must be
identical across methods, or differences in score partly reflect which target each method
happened to be evaluated against, not real differences in fit quality. GENIE3 is refit
fresh on the matching imputed train data (it was never previously evaluated against
imputed_count at all); boba-T's already-fitted rule and CellOracle's already-fitted
(intercept-corrected) coefficients are just re-evaluated against the same target.

Restricted to the same "42 shared genes" convention the original comparison used (genes
with >=1 surviving candidate regulator in all three methods).

Run in celloracle_env (needs anndata + celloracle for imputation; sklearn for GENIE3):
    /opt/anaconda3/envs/celloracle_env/bin/python redo_comparison3_6667_fair.py
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


def to01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


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
    tf_cols = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    grn_per_target = base_grn.groupby("gene_short_name")[tf_cols].max()

    # ONE real Oracle, built on train+test combined -- the single source of imputed_count
    # every method is scored against below.
    all_expr = pd.concat([train, test])
    all_clusters = pd.concat([
        clusters_train, clusters_test.rename(columns={clusters_test.columns[0]: clusters_train.columns[0]})
    ]).reindex(all_expr.index)

    adata = ad.AnnData(X=all_expr.values.astype(np.float32), obs=all_clusters, var=pd.DataFrame(index=all_expr.columns))
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
    print(f"[imputed] {len(train_imp)} train / {len(test_imp)} test cells (single shared target for all 3 methods)")

    # boba-T: already-fitted rule, just re-evaluated against the shared target.
    fitted_bobat = load_fitted_rules(f"{DN}/6667/rules/rules_6667.txt")
    bobat_genes = set()
    bobat_pred, bobat_actual = {}, {}
    for gene, (regs, rule) in fitted_bobat.items():
        regs = [r for r in regs if r in train_imp.columns]
        if not regs or gene not in test_imp.columns:
            continue
        bobat_pred[gene] = bobat_predict(test_imp, regs, rule)
        bobat_actual[gene] = test_imp[gene].to_numpy()
        bobat_genes.add(gene)

    # CellOracle: refit Ridge myself (intercept included this time), same candidates.
    co_genes = set()
    co_pred, co_actual = {}, {}
    for gene in genes:
        if gene not in grn_per_target.index:
            continue
        regs = [c for c in tf_cols if grn_per_target.loc[gene, c] == 1 and c != gene]
        if not regs:
            continue
        model = Ridge(alpha=10, random_state=123)
        model.fit(train_imp[regs], train_imp[gene])
        co_pred[gene] = model.predict(test_imp[regs])
        co_actual[gene] = test_imp[gene].to_numpy()
        co_genes.add(gene)

    # GENIE3: fit fresh on the SAME imputed train data (never previously evaluated this way).
    genie3_genes = set()
    genie3_pred, genie3_actual = {}, {}
    for gene in genes:
        if gene not in grn_per_target.index:
            continue
        regs = [c for c in tf_cols if grn_per_target.loc[gene, c] == 1 and c != gene]
        if not regs:
            continue
        rf = RandomForestRegressor(n_estimators=1000, max_features="sqrt", random_state=0, n_jobs=-1)
        rf.fit(train_imp[regs], train_imp[gene])
        genie3_pred[gene] = rf.predict(test_imp[regs])
        genie3_actual[gene] = test_imp[gene].to_numpy()
        genie3_genes.add(gene)

    shared = sorted(bobat_genes & co_genes & genie3_genes)
    print(f"\n[shared] {len(shared)} genes with surviving candidates in all 3 methods (vs. 42 originally)")

    rows = []
    for method, pred_d, actual_d in [("boba-T", bobat_pred, bobat_actual), ("CellOracle", co_pred, co_actual), ("GENIE3", genie3_pred, genie3_actual)]:
        r2s, aucs, f1s = [], [], []
        for gene in shared:
            actual01 = to01(actual_d[gene])
            pred01 = to01(pred_d[gene])
            r2s.append(r2_score(actual_d[gene], pred_d[gene]))
            try:
                aucs.append(roc_auc_score((actual01 > 0.5).astype(int), pred01))
            except ValueError:
                pass
            f1s.append(f1_score((actual01 > 0.5).astype(int), (pred01 > 0.5).astype(int)))
        rows.append({"method": method, "n_genes": len(shared), "mean_r2": np.mean(r2s),
                     "mean_auc": np.mean(aucs), "mean_f1": np.mean(f1s)})

    res = pd.DataFrame(rows).set_index("method")
    print(res)
    res.to_csv("benchmarking_out/comparison3_all_methods_vs_bobat_6667_fair_imputed_target.csv")


if __name__ == "__main__":
    main()
