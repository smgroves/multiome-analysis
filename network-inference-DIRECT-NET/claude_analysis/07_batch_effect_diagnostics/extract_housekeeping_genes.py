"""Extract a standard housekeeping-gene panel (batch-invariant by biological expectation)
from the raw full-transcriptome sources of GEMM + organoid + the 5 Ireland et al. 2025
datasets, normalized the same way as the network-gene panel (size-factor by that dataset's
own total-count QC column, then log1p) but WITHOUT MAGIC imputation -- this is a quick
technical check, not a validation input, so per-cell smoothing isn't needed.

Panel: Actb, Gapdh, Tbp, B2m, Ppia, Rplp0, Ywhaz, Sdha, Hprt, Ubc, Polr2a, Tubb5 (12 genes
confirmed present, by direct check, in every one of these 7 datasets' var_names).

If these genes -- which should NOT vary by real biology/cell-state -- show the same kind of
cross-dataset separation the network genes show, that's evidence of a genuine technical
batch effect. If they don't, the network-gene separation is more likely real biology.

Run in bobaT_env (has anndata; no MAGIC needed here):
    /opt/anaconda3/envs/bobaT_env/bin/python claude_analysis/07_batch_effect_diagnostics/extract_housekeeping_genes.py
"""

import os

import anndata as ad
import numpy as np
import pandas as pd

BOX_IRELAND = "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/Ireland_2025_Basal_Cell"
GEMM_PATH = "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/combined/adata_02_filtered.h5ad"
ORGANOID_PATH = "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/from zenodo/Organoid_full.h5ad"
OUT_DIR = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/claude_analysis/07_batch_effect_diagnostics/housekeeping"

HK_GENES = ["Actb", "Gapdh", "Tbp", "B2m", "Ppia", "Rplp0", "Ywhaz", "Sdha", "Hprt", "Ubc", "Polr2a", "Tubb5"]

DATASETS = [
    # name, path, raw_counts_layer, total_count_obs_col, qc_filter (mt_col, mt_thresh, ngene_col, ngene_thresh) or None if already filtered
    dict(name="GEMM_train", path=GEMM_PATH, layer="raw_counts", total_col="total_counts", qc=None),
    dict(name="organoid_combined", path=ORGANOID_PATH, layer="logcounts", total_col="nCount_RNA", qc=("percent.mt", 20, "nFeature_RNA", 200)),
    dict(name="cgrp_k5", path=f"{BOX_IRELAND}/021825_RPM_CGRPvK5_adata3.h5ad", layer="counts", total_col="total_counts", qc=("pct_counts_mito", 20, "n_genes_by_counts", 200)),
    dict(name="organoid_celltag", path=f"{BOX_IRELAND}/030525_RPM_RPMA_WT_Organoids_forCellTagwStates.h5ad", layer="counts", total_col="total_counts", qc=("pct_counts_mito", 20, "n_genes_by_counts", 200)),
    dict(name="tbo_allograft_5khvg", path=f"{BOX_IRELAND}/042725_RPM_TBOAllos_OriginalandAllo3_adata3_5kHVG_subsamplebycluster.h5ad", layer="counts", total_col="total_counts", qc=("pct_counts_mito", 20, "n_genes_by_counts", 200)),
    dict(name="rpr2_allograft", path=f"{BOX_IRELAND}/050925_RPM_TBOAllo_OriginalandAllo3_RPR2_adata2.h5ad", layer="counts", total_col="total_counts", qc=("pct_counts_mito", 20, "n_genes_by_counts", 200)),
    dict(name="celltag_fate_dpt", path=f"{BOX_IRELAND}/050125_RPM_RPMA_TBOAllo_CellTagAnalysis_New_1.2_fate_FAprojection_DPT_final.h5ad", layer="counts", total_col="total_counts", qc=("pct_counts_mito", 20, "n_genes_by_counts", 200)),
]


def extract_one(name, path, layer, total_col, qc):
    print(f"\n=== {name} ===")
    adata = ad.read_h5ad(path)
    print(f"Loaded: {adata.shape}")

    if qc is not None:
        mt_col, mt_thresh, ngene_col, ngene_thresh = qc
        keep = (adata.obs[mt_col] < mt_thresh) & (adata.obs[ngene_col] > ngene_thresh)
        adata = adata[keep.values].copy()
        print(f"QC filter: {keep.sum()}/{len(keep)} cells kept")

    hk_present = [g for g in HK_GENES if g in adata.var_names]
    missing = [g for g in HK_GENES if g not in adata.var_names]
    if missing:
        print(f"WARNING missing housekeeping genes in {name}: {missing}")

    raw = adata[:, hk_present].layers[layer]
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    raw = pd.DataFrame(raw, index=adata.obs_names, columns=hk_present)

    size_factor = adata.obs[total_col] / adata.obs[total_col].median()
    normalized = raw.div(size_factor.values, axis=0)
    log_normalized = np.log1p(normalized)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/hk_{name}.csv"
    log_normalized.index.name = "CellID"
    log_normalized.to_csv(out_path)
    print(f"Wrote {out_path}: {log_normalized.shape}")
    return log_normalized.shape[0]


def main():
    for ds in DATASETS:
        extract_one(ds["name"], ds["path"], ds["layer"], ds["total_col"], ds["qc"])


if __name__ == "__main__":
    main()
