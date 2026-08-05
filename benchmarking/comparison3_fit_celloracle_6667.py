"""Comparison 3 (predicted vs. actual expression, see benchmarking/README.md), fitting step:
fit CellOracle on the exact inputs behind boba-T run 6667 and write CellOracle's own
<gene>_validation.csv files in boba-T's validation-CSV shape, so the two methods' predicted
vs. actual TF expression can be scored identically (see comparison3_score_celloracle_vs_bobat_6667.py).

Uses the SAME train/test split boba-T's rules were fit and validated on
(6667/data_split/{train,test}_t0combined.csv, already in boba-T's post-normalization [0,1]
scale) and the SAME 228-edge DIRECT-NET candidate network as CellOracle's base GRN, so the
comparison isolates rule-fitting quality rather than differing inputs.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison3_fit_celloracle_6667.py
"""

from __future__ import annotations

import os

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
NETWORK_CSV = (
    f"{DN}/networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)


def to01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    # 1. Same train/test split boba-T's rules were fit and validated on.
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    clusters = pd.read_csv(f"{DN}/6667/data_split/clusters_traincombined.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes (boba-T's actual node set), "
          f"{len(train)} train cells, {len(test)} test cells")

    # 2. Same candidate edges boba-T's rules were fit on -> a TF_info_matrix restricted to
    #    exactly this candidate set (every edge in this file has both endpoints in `genes`).
    edge_list = pd.read_csv(NETWORK_CSV, header=None, names=["source", "target"])
    edge_list = edge_list[edge_list.source.isin(genes) & edge_list.target.isin(genes)]
    base_grn = (
        edge_list.assign(v=1)
        .pivot(index="target", columns="source", values="v")
        .reindex(columns=genes)
        .fillna(0)
        .reset_index()
        .rename(columns={"target": "gene_short_name"})
    )
    base_grn.insert(0, "peak_id", base_grn["gene_short_name"])  # no real peaks; CellOracle just needs the column
    print(f"[base_grn] {len(edge_list)} candidate edges over {len(genes)} genes")

    # 3. Same (already boba-T-normalized, [0,1]-scale) expression values -- do NOT re-run
    #    CellOracle's own raw-count normalization, or the two methods are no longer fit on
    #    the same numbers.
    adata = ad.AnnData(
        X=train.values.astype(np.float32),
        obs=clusters.reindex(train.index),
        var=pd.DataFrame(index=train.columns),
    )
    # embedding_name defaults to None in the signature, but import_anndata_as_normalized_count
    # unconditionally reads adata.obsm[embedding_name] -- not actually optional. We don't need
    # a real 2D layout (that's for CellOracle's own plotting), so a placeholder satisfies it.
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    # import_anndata_as_normalized_count's internal QC step (score_cv_vs_mean) reads
    # layers["raw_count"], but the line that would set it from adata.X is commented out in
    # celloracle==0.20.0's own source (oracle_core.py) -- set it ourselves so that QC step
    # (and self.high_var_genes, which simulate_shift checks later) works as intended.
    adata.layers["raw_count"] = adata.X.copy()
    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(
        adata=adata, cluster_column_name="class", embedding_name="dummy"
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(20, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    # GRN_unit="whole" -> one global coef_matrix, matching boba-T's single (non-per-cluster) network.
    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix  # rows = regulators, columns = targets
    coef.to_csv(f"{DN}/6667/rules/celloracle_coef_matrix.csv")
    print(f"[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")

    # 4. Apply the TRAIN-fit coefficients to the held-out TEST cells (never re-fit on test --
    #    this is what makes it comparable to boba-T's held-out validation).
    C = coef.reindex(index=genes, columns=genes).fillna(0)
    X_test = test.reindex(columns=genes)
    pred = X_test.values @ C.values  # held-out predicted expression, cells x genes

    out_dir = f"{DN}/6667/validation/celloracle_validation/accuracy_plots"
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for gi, gene in enumerate(genes):
        if not (C[gene] != 0).any():
            continue  # no surviving regulators for this gene in the shared base GRN -- skip (shared-node rule)
        out = pd.DataFrame(
            {
                "actual": to01(X_test[gene].to_numpy()),  # min-max to [0,1] to match boba-T's 0.5-threshold convention
                "predicted": to01(pred[:, gi]),
            },
            index=test.index,
        )
        out.index.name = "CellID"
        out.to_csv(f"{out_dir}/{gene}_validation.csv")
        n_written += 1
    print(f"[write] {n_written}/{len(genes)} genes had surviving regulators -> {out_dir}")


if __name__ == "__main__":
    main()
