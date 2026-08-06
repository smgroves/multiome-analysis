"""Redo the truth-table/AUC comparison using the SAME correctly-refit Ridge models
(intercept included, fit on real imputed_count) as verify_celloracle_fit.py -- for full
consistency with the corrected R2 numbers. AUROC itself is invariant to adding a constant
(the missing intercept doesn't change *rank order*), but using the wrong data
representation (raw vs. imputed) for the marginalization step could still shift results,
so this reruns it properly rather than assuming AUC was unaffected.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python verify_celloracle_truthtable.py track1
    /opt/anaconda3/envs/celloracle_env/bin/python verify_celloracle_truthtable.py track2
"""
import sys

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from hsc_ground_truth import load_rules, truth_table
from verify_celloracle_fit import build_base_grn_track1, build_base_grn_track2


def run(track):
    if track == "track1":
        base_grn, genes = build_base_grn_track1()
        expr = pd.read_csv("data/hsc_ground_truth/expr_bobat.csv", index_col=0)[genes]
        clusters = pd.read_csv("data/hsc_ground_truth/clusters_bobat.csv", index_col=0)
        norm_data = expr  # already [0,1] via BoolODE min-max; used for true-regulator binning too
        alpha = 10
    else:
        base_grn, genes = build_base_grn_track2()
        expr = pd.read_csv("data/hsc_multiome/expr_bobat_real.csv", index_col=0)[genes].clip(lower=0)
        clusters = pd.read_csv("data/hsc_multiome/clusters_bobat_real.csv", index_col=0)
        norm_data = pd.read_csv("hsc_multiome_cicero/data_split/train_t0hsc_multiome_cicero.csv", index_col=0)
        norm_data = pd.concat([norm_data, pd.read_csv("hsc_multiome_cicero/data_split/test_t0hsc_multiome_cicero.csv", index_col=0)])
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
    imputed = pd.DataFrame(
        oracle.adata.layers["imputed_count"], index=oracle.adata.obs_names, columns=oracle.adata.var_names
    )

    tf_cols = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    grn_per_target = base_grn.groupby("gene_short_name")[tf_cols].max()

    ground_truth_rules = load_rules()
    rows = []
    for gene_lower, (expr_rule, gt_regs_orig) in ground_truth_rules.items():
        gene = gene_lower.upper()
        if gene not in grn_per_target.index or gene not in imputed.columns:
            continue
        gt_regs_upper = [r.upper() for r in gt_regs_orig if r.upper() in norm_data.columns]
        if len(gt_regs_upper) != len(gt_regs_orig):
            continue  # needs EGRNAB, no real data for it (Track 2) -- skip, don't fudge
        regs = [c for c in tf_cols if grn_per_target.loc[gene, c] == 1 and c != gene]
        if not regs:
            continue
        model = Ridge(alpha=alpha, random_state=123)
        model.fit(imputed[regs], imputed[gene])
        predicted_all = model.predict(imputed[regs])  # WITH intercept now, on the real imputed_count

        bin_idx = norm_data.index.intersection(imputed.index)
        n_true = len(gt_regs_upper)
        true_leaf = norm_data.loc[bin_idx, gt_regs_upper].gt(0.5).astype(int).apply(
            lambda row: int("".join(row.astype(str)), 2), axis=1
        )
        predicted = pd.Series(predicted_all, index=imputed.index).loc[bin_idx]

        gt_tt = truth_table(expr_rule, gt_regs_orig)
        gt_labels = gt_tt["output"].to_numpy()

        marginalized = np.full(2 ** n_true, np.nan)
        for leaf in range(2 ** n_true):
            mask = (true_leaf == leaf).to_numpy()
            if mask.sum() > 0:
                marginalized[leaf] = predicted.to_numpy()[mask].mean()
        covered = ~np.isnan(marginalized)
        try:
            auc = roc_auc_score(gt_labels[covered], marginalized[covered]) if len(set(gt_labels[covered])) > 1 else np.nan
        except ValueError:
            auc = np.nan
        rows.append({"gene": gene, "n_true_regulators": n_true, "n_fitted": len(regs),
                     "leaves_covered": int(covered.sum()), "leaves_total": 2 ** n_true, "auc": auc})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print(f"\nmean AUC: {res['auc'].mean():.3f}")
    res.to_csv(f"benchmarking_out/comparison_{track}_celloracle_truth_table_corrected.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1])
