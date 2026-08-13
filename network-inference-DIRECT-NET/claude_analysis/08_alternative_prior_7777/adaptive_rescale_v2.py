"""Barcode 7779, revised: two EXPLICIT branches for flagged genes, not one blended GMM
mechanism (the unified version failed broadly -- see decisive_test_all_samples.py, net
negative across the 44-sample population, because a GMM fit on GEMM's own narrow range
extrapolates by saturating to 0/1 for genes with EXTREME mean shift, e.g. PBX1, destroying
real relative signal rather than preserving it).

Branch 1 -- mean shift, INTACT variance (std_ratio >= STD_RATIO_THRESHOLD): shift-only
recentering, `raw - sample_mean + gemm_mean` -- preserves the sample's own real spread
exactly, just corrects the absolute location -- then clipped against GEMM's OWN fixed
quantile reference (lq_gemm, uq_gemm from GEMM's raw training data). Using GEMM's reference
quantiles here, not the sample's own, is required -- clipping against the sample's own
quantiles after a location-only shift would just reproduce the standard norm=0.3 output,
silently discarding the correction (same reasoning as why ComBat + per-sample quantiles
canceled out, ireland2025_external_validation/FINDINGS.md sec 4).

Branch 2 -- variance collapse (std_ratio < STD_RATIO_THRESHOLD): REVISED after a real user
catch. The original design used GEMM's reference GMM to make one discrete ON/OFF pole call
per (gene, sample), then shrunk existing values toward that pole (alpha=0.8 pole +
(1-alpha) original). That formula is mathematically guaranteed to force every cell onto the
same side of the 0.5 threshold regardless of the real data (pole=0 -> range [0, 0.2],
pole=1 -> range [0.8, 1.0], for ANY per-cell "original" value in [0,1] -- verified
directly). That's not "correctly detecting a genuinely single-class gene," it's
MANUFACTURING single-class output on every flagged gene unconditionally, exactly the kind
of metric-skewing exclusion-by-construction this whole investigation has been trying to
avoid. If a gene truly is all-ON or all-OFF in a sample, the model predicting that
confidently -- and AUC legitimately coming back NaN, since there's no negative class left
to discriminate against -- is CORRECT behavior to report, not something to route around.

Fixed: variance-collapse genes get the SAME recenter-and-clip-against-GEMM's-window
transform as mean-shift genes, but WITHOUT the z-score/scale step (dividing by a near-zero
sample std is numerically unstable, which is why the branches were split from a single
unified mechanism in the first place -- see decisive_test_all_samples.py's original
one-GMM-for-everything failure). Shift-only recentering lets the real data decide the
outcome instead of forcing one: genuinely collapsed data will honestly land near one pole
(NaN AUC then reported truthfully, not hidden), while any real residual signal survives
rather than being erased. Genes are no longer excluded as scoring targets for this reason
either -- that exclusion was compensating for the forcing bug; individual NaN AUC values
now get excluded from the AUC MEAN only (standard practice), not the gene from analysis.
"""

import pickle

import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
MEAN_SHIFT_THRESHOLD = 8.0
STD_RATIO_THRESHOLD = 0.02

_gemm_mean = None
_gemm_std = None
_gemm_lq = None
_gemm_uq = None
QUANTILE = 0.3


def _load_gemm_reference():
    global _gemm_mean, _gemm_std, _gemm_lq, _gemm_uq
    if _gemm_mean is not None:
        return
    gemm_full = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm_full.columns = [c.upper() for c in gemm_full.columns]
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_train = gemm_full.loc[gemm_full.index.isin(train_cellids)]
    _gemm_mean = gemm_train.mean()
    _gemm_std = gemm_train.std()
    _gemm_lq = gemm_train.quantile(QUANTILE)
    _gemm_uq = gemm_train.quantile(1 - QUANTILE)


def classify_genes(raw_sample_df, nodes):
    """Returns dict {gene: "mean_shift" | "variance_collapse" | None (not flagged)}."""
    _load_gemm_reference()
    classification = {}
    for gene in nodes:
        if gene not in raw_sample_df.columns:
            continue
        d_mean = raw_sample_df[gene].mean()
        d_std = raw_sample_df[gene].std()
        shift = abs((d_mean - _gemm_mean[gene]) / _gemm_std[gene])
        ratio = d_std / _gemm_std[gene]
        if ratio < STD_RATIO_THRESHOLD:
            classification[gene] = "variance_collapse"
        elif shift > MEAN_SHIFT_THRESHOLD:
            classification[gene] = "mean_shift"
        else:
            classification[gene] = None
    return classification


def build_adaptive_data(raw_sample_df, normed_sample_df, nodes, classification=None):
    _load_gemm_reference()
    if classification is None:
        classification = classify_genes(raw_sample_df, nodes)

    adaptive = normed_sample_df.copy()
    flagged_genes = set()
    for gene, kind in classification.items():
        if kind is None or gene not in raw_sample_df.columns:
            continue
        flagged_genes.add(gene)
        raw_vals = raw_sample_df.loc[normed_sample_df.index, gene]

        if kind == "mean_shift":
            adaptive[gene] = rescale_mean_shift_gene(raw_vals, gene).values

        elif kind == "variance_collapse":
            adaptive[gene] = rescale_collapse_gene(raw_vals, gene).values

    return adaptive, flagged_genes


def rescale_collapse_gene(raw_vals, gene):
    """Location-only recentering (no z-score -- dividing by a near-zero sample std is
    numerically unstable) clipped against GEMM's fixed window. Deliberately does NOT force
    a discrete outcome (see module docstring on why the earlier GMM-pole+shrinkage version
    was a real bug, not a feature): if the gene's raw values are genuinely all clustered on
    one side after recentering, every cell will honestly land near the same pole and any
    downstream NaN AUC is a true reflection of that, not an artifact of this transform.
    Any real residual spread in the raw values survives this shift unchanged."""
    _load_gemm_reference()
    recentered = raw_vals - raw_vals.mean() + _gemm_mean[gene]
    lq, uq = _gemm_lq[gene], _gemm_uq[gene]
    return ((recentered - lq) / (uq - lq)).clip(0, 1)


def rescale_mean_shift_gene(raw_vals, gene):
    """Location AND scale correction (upgraded from the original shift-only version, which
    was found to be net-negative overall -- see decisive_test_all_samples_v2.csv's
    classification breakdown: mean_shift-only pairs averaged -0.0085 binary_diff vs.
    variance_collapse's +0.0278, despite both branches targeting genuinely flagged genes.
    A pure location shift doesn't correct for a real difference in the sample's OWN spread
    relative to GEMM's, so it can still miscalibrate after clipping against GEMM's fixed
    window. z-scoring on the sample's own std before re-expressing in GEMM's units fixes
    both location and scale at once.). Shared by 7778 and 7779 -- they differ only in how
    they handle the variance_collapse branch (GMM-pole here in 7779; marginalization in
    7778, see main_validate_7778_adaptive_marginalization.py)."""
    _load_gemm_reference()
    z = (raw_vals - raw_vals.mean()) / raw_vals.std()
    rescaled_raw = z * _gemm_std[gene] + _gemm_mean[gene]
    lq, uq = _gemm_lq[gene], _gemm_uq[gene]
    return ((rescaled_raw - lq) / (uq - lq)).clip(0, 1)
