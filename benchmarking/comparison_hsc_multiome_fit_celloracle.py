"""Fit CellOracle's ridge regression on the real, ATAC+motif-derived base GRN (Track 2's
centerpiece deliverable -- "does CellOracle actually look at real ATAC data," built for
the first time in this project by build_hsc_atac_base_grn.py) + the real corrected GSE194122
expression (preprocess_hsc_multiome.py). Same 10-gene HSC panel as boba-T/GENIE3's Track 2
runs, same real cells, so all three methods are directly comparable.

base_grn format matches CellOracle's own convention (one row per peak, gene_short_name =
that peak's assigned target gene, one binary column per candidate TF) -- built from
build_hsc_atac_base_grn.py's per-hit detail table, not from a generic promoter scan.

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_hsc_multiome_fit_celloracle.py
"""
import anndata as ad
import celloracle as co
import numpy as np
import pandas as pd

DATA_DIR = "data/hsc_multiome"
PANEL_GENES = ["GATA1", "GATA2", "PU1", "FLI1", "CEBPA", "EKLF", "GFI1", "FOG1", "SCL", "CJUN"]


def build_base_grn():
    detail = pd.read_csv(f"{DATA_DIR}/candidate_network_atac_real_detail.csv")
    peak_target = detail.groupby("peak")["target"].first()
    wide = pd.crosstab(detail["peak"], detail["source"]).clip(upper=1)
    for tf in PANEL_GENES:
        if tf not in wide.columns:
            wide[tf] = 0
    wide = wide[PANEL_GENES]
    base_grn = wide.reset_index().rename(columns={"peak": "peak_id"})
    base_grn["gene_short_name"] = base_grn["peak_id"].map(peak_target)
    return base_grn[["peak_id", "gene_short_name"] + PANEL_GENES]


def main():
    base_grn = build_base_grn()
    print(f"[base_grn] {len(base_grn)} peaks, {base_grn['gene_short_name'].nunique()} target genes")
    print(base_grn)

    expr = pd.read_csv(f"{DATA_DIR}/expr_bobat_real.csv", index_col=0)
    clusters = pd.read_csv(f"{DATA_DIR}/clusters_bobat_real.csv", index_col=0)
    print(f"[data] {expr.shape[0]} cells x {expr.shape[1]} genes")
    n_neg = (expr < 0).sum().sum()
    print(f"[data] clipping {n_neg} small negative values to 0 (a known property of MAGIC's "
          f"diffusion smoothing near zero, not a real negative expression value) -- "
          f"CellOracle's import_anndata_as_normalized_count requires non-negative input")
    expr = expr.clip(lower=0)

    adata = ad.AnnData(
        X=expr.values.astype(np.float32),
        obs=clusters.reindex(expr.index).rename(columns={"class": "class"}),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obsm["dummy"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    # CellOracle 0.20.0 gotcha (already hit + documented elsewhere in this project):
    # score_cv_vs_mean needs adata.layers["raw_count"], but the line that would set it is
    # commented out in oracle_core.py -- set it ourselves.
    adata.layers["raw_count"] = adata.X.copy()

    oracle = co.Oracle()
    # expr_bobat_real.csv is already MAGIC-imputed log-normalized expression (see
    # preprocess_hsc_multiome.py), not raw counts -- use the normalized-count import path,
    # matching how this project's other already-normalized/imputed inputs are imported
    # into CellOracle (e.g. comparison3_fit_celloracle_6667.py on boba-T's own [0,1] data).
    oracle.import_anndata_as_normalized_count(adata=adata, cluster_column_name="class",
                                                embedding_name="dummy")
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_pca_dims = min(50, adata.shape[0] - 1, adata.shape[1] - 1)
    oracle.knn_imputation(n_pca_dims=n_pca_dims, k=15, balanced=True, n_jobs=4)

    oracle.fit_GRN_for_simulation(GRN_unit="whole", alpha=10)
    coef = oracle.coef_matrix
    coef.to_csv(f"{DATA_DIR}/celloracle_atac_coef_matrix.csv")
    print(f"\n[fit] coef_matrix: {coef.shape}, {int((coef.values != 0).sum())} nonzero entries")
    print(coef)


if __name__ == "__main__":
    main()
