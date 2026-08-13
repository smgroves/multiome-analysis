"""Shared logic for barcode 7779: build an "adaptively rescaled" version of an external
sample's data -- non-flagged genes keep the standard per-sample norm=0.3 quantile-clip;
flagged genes (scale-mismatched vs GEMM, same diagnostic as diagnose_per_gene_scale_
collapse.py) get their RAW values rescored via GEMM's reference GMM's predict_proba
instead (gemm_reference_gmm.pkl, fit_gemm_reference_gmm.py) -- one mechanism that handles
both a shifted-but-still-spread-out distribution (real gradient across GEMM's two
components) and a collapsed one (uniformly close to one component) gracefully.
"""

import pickle

import numpy as np
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
GMM_REF_PATH = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/gemm_reference_gmm.pkl"
MEAN_SHIFT_THRESHOLD = 8.0
STD_RATIO_THRESHOLD = 0.02

_gemm_reference = None
_gemm_mean = None
_gemm_std = None


def _load_gemm_reference():
    global _gemm_reference, _gemm_mean, _gemm_std
    if _gemm_reference is not None:
        return
    with open(GMM_REF_PATH, "rb") as f:
        _gemm_reference = pickle.load(f)
    gemm_full = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm_full.columns = [c.upper() for c in gemm_full.columns]
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_train = gemm_full.loc[gemm_full.index.isin(train_cellids)]
    _gemm_mean = gemm_train.mean()
    _gemm_std = gemm_train.std()


def flag_genes(raw_sample_df, nodes):
    """raw_sample_df: RAW (pre-normalization) values, cells x genes, columns already
    uppercased. Returns the set of genes flagged as scale-mismatched vs GEMM."""
    _load_gemm_reference()
    flagged = set()
    for gene in nodes:
        if gene not in raw_sample_df.columns:
            continue
        d_mean = raw_sample_df[gene].mean()
        d_std = raw_sample_df[gene].std()
        shift = abs((d_mean - _gemm_mean[gene]) / _gemm_std[gene])
        ratio = d_std / _gemm_std[gene]
        if shift > MEAN_SHIFT_THRESHOLD or ratio < STD_RATIO_THRESHOLD:
            flagged.add(gene)
    return flagged


def build_adaptive_data(raw_sample_df, normed_sample_df, nodes, flagged_genes=None):
    """raw_sample_df: RAW values (cells x genes, uppercased columns), same cell index as
    normed_sample_df. normed_sample_df: the standard bb.load.load_data(..., norm=0.3)
    output for the SAME sample. Returns a copy of normed_sample_df with flagged genes'
    columns replaced by GEMM-reference-GMM predict_proba on their raw values."""
    _load_gemm_reference()
    if flagged_genes is None:
        flagged_genes = flag_genes(raw_sample_df, nodes)

    adaptive = normed_sample_df.copy()
    for gene in flagged_genes:
        if gene not in raw_sample_df.columns:
            continue
        ref = _gemm_reference[gene]
        gm = ref["gmm"]
        on_idx = ref["on_idx"]
        d = raw_sample_df.loc[normed_sample_df.index, gene].values.reshape(-1, 1)
        proba_on = gm.predict_proba(d)[:, on_idx]
        adaptive[gene] = proba_on
    return adaptive, flagged_genes
