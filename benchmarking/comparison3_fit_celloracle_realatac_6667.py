"""Comparison 3 (predicted vs. actual expression), CellOracle-as-used-in-the-wild variant:
fit CellOracle on the exact same `6667` train/test split and held-out evaluation as
comparison3_fit_celloracle_6667.py, but using the REAL, dataset-specific ATAC-informed base
GRN built by comparison_tko_atac_base_grn.py (Signac peak-to-gene assignment on
TKO_final_arc's own `peaks` ChromatinAssay + gimmemotifs mm10 motif scan) instead of
DIRECT-NET's 228-edge LASSO-restricted candidate network.

This is the fair "how would CellOracle actually be run" comparison: DIRECT-NET's candidate
network is boba-T's own curated starting point, not something CellOracle would ever be
restricted to in practice -- a real CellOracle multiome workflow builds its own base GRN
from the dataset's own ATAC peaks and lets the ridge fit find structure across everything
that base GRN allows, the same way comparison_tko_fit_celloracle_atac.py did for TKO's own
~3,000-HVG panel. This script applies that same real base GRN to `6667`'s own 53-gene
train/test split and scoring convention instead, so the result is directly comparable to
boba-T/CellOracle-DIRECT-NET/GENIE3's numbers in the same table -- TKO_final_arc's scRNA-seq
is the same underlying cells behind `6667` (DIRECT-NET's own ATAC-based candidate-network
construction for `6667` was run on this same data, just not through this project's scripts).

base GRN gene symbols are Signac/Ensembl mouse case ("Ascl1"); `6667`'s genes are
upper-cased ("ASCL1") -- upper-case the base GRN's gene_short_name/TF columns before
restricting to `6667`'s 53-gene set (52/53 targets matched, 47/53 TFs matched; only
RORA_RORB is missing, expected -- it's a combined pseudo-node, not a real gene).

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison3_fit_celloracle_realatac_6667.py
"""

from __future__ import annotations

import os

import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
BASE_GRN_PATH = f"{DN}/data/tko_atac/tko_atac_base_grn.parquet"


def to01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    clusters = pd.read_csv(f"{DN}/6667/data_split/clusters_traincombined.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes (boba-T's actual node set), "
          f"{len(train)} train cells, {len(test)} test cells")

    base = pd.read_parquet(BASE_GRN_PATH)
    id_cols = base[["peak_id", "gene_short_name"]].assign(gene_short_name=lambda d: d["gene_short_name"].str.upper())
    tf_part = base.drop(columns=["peak_id", "gene_short_name"])
    tf_part.columns = tf_part.columns.str.upper()
    tf_part = tf_part.T.groupby(level=0).max().T  # collapse TF columns that collided under upper-casing
    base = pd.concat([id_cols, tf_part], axis=1)

    keep_tfs = sorted(set(tf_part.columns) & set(genes))
    base_grn = base[base["gene_short_name"].isin(genes)][["peak_id", "gene_short_name"] + keep_tfs]
    n_edges = int((base_grn[keep_tfs].values == 1).sum())
    print(f"[base_grn] real ATAC base GRN restricted to 6667's 53-gene set: "
          f"{len(keep_tfs)}/{len(genes)} candidate TFs, {base_grn['gene_short_name'].nunique()}/{len(genes)} "
          f"target genes matched, {n_edges} nonzero peak-TF entries before per-gene aggregation")

    adata = ad.AnnData(
        X=train.values.astype(np.float32),
        obs=clusters.reindex(train.index),
        var=pd.DataFrame(index=train.columns),
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
    coef = oracle.coef_matrix  # rows = regulators, columns = targets
    coef.to_csv(f"{DN}/6667/rules/celloracle_realatac_coef_matrix.csv")
    print(f"[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")

    # Apply the TRAIN-fit coefficients to the held-out TEST cells, same as the DIRECT-NET-
    # restricted version -- never re-fit on test.
    C = coef.reindex(index=genes, columns=genes).fillna(0)
    X_test = test.reindex(columns=genes)
    pred = X_test.values @ C.values

    out_dir = f"{DN}/6667/validation/celloracle_realatac_validation/accuracy_plots"
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for gi, gene in enumerate(genes):
        if not (C[gene] != 0).any():
            continue  # no surviving regulators for this gene -- skip (shared-node rule)
        out = pd.DataFrame(
            {
                "actual": to01(X_test[gene].to_numpy()),
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
