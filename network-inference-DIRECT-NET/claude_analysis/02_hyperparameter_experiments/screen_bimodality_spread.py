"""Phase 2, item 1 (BoBa-T_hyperparameters.md sec 5): screen each of the network's genes
for genuine bimodality/spread in their raw (pre-normalization) expression, before trusting
a norm=0.3-fitted rule for that gene. Two diagnostics per gene, on GEMM training data:

1. Hartigan's dip test (via the `diptest` package) on raw (log1p+MAGIC-imputed, but not
   norm=0.3-clipped) expression -- low p-value supports genuine bimodality; a high
   p-value means the fitted "on/off" framing is imposed by node_normalization rather than
   present in the data.
2. Raw spread (IQR, coefficient of variation) vs. post-normalization spread (the fraction
   of cells landing at exact 0 or 1 after norm=0.3) -- a gene with low raw spread but a
   high post-normalization exact-0/1 fraction is exactly the "artificial spread" failure
   mode described in BoBa-T_hyperparameters.md sec 4.

Run in bobaT_env_py3.13 (needs `pip install diptest`):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/02_hyperparameter_experiments/screen_bimodality_spread.py
"""

import os

import diptest
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
DATA_PATH = "data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
OUT_DIR = f"{DIR_PREFIX}/comparisons/phase2_hyperparameters"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    raw = bb.load.load_data(
        f"{DIR_PREFIX}/{DATA_PATH}", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    normed = bb.load.load_data(
        f"{DIR_PREFIX}/{DATA_PATH}", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    rows = []
    for gene in nodes:
        raw_vals = raw[gene].values.astype(float)
        dip, pval = diptest.dipstat(raw_vals), diptest.diptest(raw_vals)[1]
        iqr = np.percentile(raw_vals, 75) - np.percentile(raw_vals, 25)
        cv = raw_vals.std() / raw_vals.mean() if raw_vals.mean() != 0 else np.nan
        frac_extreme_normed = ((normed[gene] == 0) | (normed[gene] == 1)).mean()
        rows.append({
            "gene": gene, "dip_stat": dip, "dip_pval": pval, "raw_iqr": iqr, "raw_cv": cv,
            "raw_mean": raw_vals.mean(), "raw_std": raw_vals.std(),
            "frac_exact_0_or_1_after_norm0.3": frac_extreme_normed,
        })
    df = pd.DataFrame(rows).sort_values("dip_pval", ascending=False)

    # frac_exact_0_or_1_after_norm0.3 is ~2*norm for every gene by construction (the
    # quantile-clip fixes the pinned fraction at the population level, regardless of that
    # gene's actual shape) -- it is NOT a per-gene signal and is kept only for reference.
    # The real per-gene "artificial spread" risk is: not clearly bimodal (dip test can't
    # reject unimodality) AND genuinely low true spread relative to the rest of the
    # network (raw_std in the bottom third across all 53 genes) -- i.e. norm=0.3 is
    # stretching a comparatively small amount of real variation to fill the same [0,1]
    # range as a gene that is both bimodal AND has 10x the raw spread.
    low_spread_cutoff = df["raw_std"].quantile(1 / 3)
    df["low_true_spread"] = df["raw_std"] <= low_spread_cutoff
    df["flagged_artificial_spread"] = (df["dip_pval"] > 0.05) & df["low_true_spread"]

    df.to_csv(f"{OUT_DIR}/bimodality_spread_screen.csv", index=False)

    print(f"{len(df)} genes screened.")
    print(f"Note: frac_exact_0_or_1_after_norm0.3 is ~{2*0.3:.1f} for every gene by construction "
          f"(quantile-clip fixes this at the population level) -- not a per-gene signal on its own.")
    print(f"\n=== Genes flagged for likely artificial spread (dip test p>0.05: not clearly "
          f"bimodal, AND raw_std in the bottom third across the network: genuinely low true spread) ===")
    flagged = df[df["flagged_artificial_spread"]]
    print(flagged[["gene", "dip_pval", "raw_std"]].sort_values("raw_std").to_string(index=False))
    print(f"\n{len(flagged)}/{len(df)} genes flagged.")

    print(f"\n=== Most clearly bimodal genes (lowest dip p-value) ===")
    print(df.sort_values("dip_pval").head(10)[["gene", "dip_pval", "raw_std"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR}/bimodality_spread_screen.csv")


if __name__ == "__main__":
    main()
