"""Why does a sample's gene-regulator correlation structure diverge from GEMM's (§4), and
why is that correlated with R2? The sign-flip-rate <-> R2 relationship found in
compare_correlation_shift_across_samples.py is close to tautological on its own -- BoBa-T's
rules ARE that correlation structure, so of course a shift there predicts poor prediction.
It doesn't say WHY the structure shifts for some samples and not others.

Hypothesis: restriction of range. If a sample's cells span less of the underlying
cell-state diversity that GEMM's training tumors span (i.e. its cells are more
homogeneous/less spread out across the archetype/identity axes the network's genes vary
along), within-sample gene-gene correlations will attenuate or flip sign as a statistical
byproduct of reduced variance -- independent of any "real" biological rewiring -- AND
prediction will be poor because the fitted rules were calibrated to distinguish states
across a range of variation this sample barely explores.

Tests this directly: per-sample mean per-gene raw standard deviation (relative to GEMM
training data's own), correlated against mean R2 and against the sign-flip-rate metric.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/diagnose_sample_diversity.py
"""

import glob
import os
import pickle
import re

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"


def gather_sample_paths():
    paths = {}
    for f in glob.glob(f"{DIR_PREFIX}/data/allografts/adata_*_v3_RORA_RORB_ave.csv"):
        name = re.search(r"adata_(.+)_allografts_v3", f).group(1)
        paths[f"allograft_{name}"] = f
    for f in glob.glob(f"{DIR_PREFIX}/data/human_tumor_MSK/adata_*_v3_RORA_RORB_ave.csv"):
        name = re.search(r"adata_(.+)_v3", f).group(1)
        paths[f"human_{name}"] = f
    paths["organoid_shGFP"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB1"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB2"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv"
    paths["organoid_combined"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_v3_RORA_RORB_ave.csv"
    paths["mets_compiled"] = f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv"
    return paths


def gather_r2(name):
    candidates = []
    if name.startswith("allograft_"):
        candidates.append(f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv")
    elif name.startswith("human_"):
        candidates.append(f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv")
    elif name == "organoid_combined":
        candidates.append(f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv")
    else:
        candidates.append(f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv")
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c, index_col=0)["r2"].mean()
    return np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    # NOTE: 6667/data_split/train_t0combined.csv is already node_normalization=0.3-clipped
    # (confirmed directly: exactly 60% of any gene's values are exactly 0 or 1) -- it is
    # NOT raw data despite loading it with norm=None. Use the true raw source file,
    # restricted to the same training cell IDs, for a genuine raw-vs-raw comparison.
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm_raw_full = bb.load.load_data(
        f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    gemm_raw = gemm_raw_full.loc[gemm_raw_full.index.isin(train_cellids)]
    print(f"GEMM raw training data: {len(gemm_raw)}/{len(gemm_raw_full)} cells matched to the train split")
    gemm_mean_std = gemm_raw.std().mean()

    rows = []
    for name, path in gather_sample_paths().items():
        if not os.path.exists(path):
            continue
        d = bb.load.load_data(path, nodes, norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        rows.append({
            "name": name, "mean_gene_std": d.std().mean(), "n_cells": len(d),
            "relative_diversity_vs_gemm": d.std().mean() / gemm_mean_std,
            "mean_r2": gather_r2(name),
        })

    res = pd.DataFrame(rows)
    sign_flip_df = pd.read_csv(f"{OUT_DIR}/all_samples_corr_vs_r2.csv")
    merged = res.merge(sign_flip_df[["name", "strong_sign_flip_rate", "sign_flip_rate"]], on="name")
    merged.to_csv(f"{OUT_DIR}/sample_diversity_vs_signflip_r2.csv", index=False)

    scored = merged.dropna(subset=["mean_r2"])
    print(f"GEMM training data's own mean per-gene raw std (reference): {gemm_mean_std:.4f}")
    print(f"\ncorr(relative_diversity_vs_gemm, mean_r2): {scored['relative_diversity_vs_gemm'].corr(scored['mean_r2']):.3f}")
    print(f"corr(relative_diversity_vs_gemm, strong_sign_flip_rate): {scored['relative_diversity_vs_gemm'].corr(scored['strong_sign_flip_rate']):.3f}")
    print(f"(for reference) corr(strong_sign_flip_rate, mean_r2): {scored['strong_sign_flip_rate'].corr(scored['mean_r2']):.3f}")

    pd.set_option("display.width", 160)
    print("\n=== Bottom 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2").head(8)[["name", "relative_diversity_vs_gemm", "strong_sign_flip_rate", "mean_r2"]].to_string(index=False))
    print("\n=== Top 8 by mean_r2 ===")
    print(scored.sort_values("mean_r2", ascending=False).head(8)[["name", "relative_diversity_vs_gemm", "strong_sign_flip_rate", "mean_r2"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR}/sample_diversity_vs_signflip_r2.csv")


if __name__ == "__main__":
    main()
