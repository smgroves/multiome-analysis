"""Compare 6667 (node_normalization=0.3) vs 6668 (node_normalization=0.4) across all 5
external validation sets, to test whether widening the quantile-clip window (fewer cells
left graded, more pinned to exact 0/1) improves external validation -- most importantly on
organoid_shGFP, the null/control condition this whole test was designed to diagnose. See
/Users/xpz5km/.claude/plans/it-is-looking-like-elegant-patterson.md for the full mechanism
writeup this test is based on.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/01_normalization_sweep/compare_6667_6668_normalization.py
"""

import os

import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
BRCDS = ["6667", "6668"]
VALIDATION_SETS = ["organoid", "organoid_shGFP", "organoid_shRORB1", "organoid_shRORB2", "mets_compiled"]
METRICS = ["r2", "roc_auc_score", "f1", "accuracy"]
OUT_DIR = f"{DIR_PREFIX}/comparisons/6667_vs_6668_norm_sweep"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    long_rows = []
    for brcd in BRCDS:
        for val_set in VALIDATION_SETS:
            path = f"{DIR_PREFIX}/{brcd}/validation/external_validation/{val_set}/summary_stats.csv"
            df = pd.read_csv(path, index_col=0)
            df["brcd"] = brcd
            df["validation_set"] = val_set
            long_rows.append(df)
    long_df = pd.concat(long_rows, ignore_index=True)

    n_groups = long_df.groupby(["brcd", "validation_set"]).ngroups
    assert n_groups == len(BRCDS) * len(VALIDATION_SETS), (
        f"Expected {len(BRCDS) * len(VALIDATION_SETS)} (brcd, validation_set) groups, found {n_groups}"
    )
    long_df.to_csv(f"{OUT_DIR}/long_format_all_genes.csv", index=False)

    aggregate = long_df.groupby(["validation_set", "brcd"])[METRICS].mean().round(4)
    aggregate.to_csv(f"{OUT_DIR}/aggregate_mean_by_set_and_brcd.csv")
    print("=== Mean metrics per (validation_set, brcd) ===")
    print(aggregate)

    # Per-gene diff (6668 - 6667), one column set per validation set
    pivot_frames = []
    for val_set in VALIDATION_SETS:
        sub = long_df[long_df["validation_set"] == val_set]
        wide = sub.pivot(index="gene", columns="brcd", values=METRICS)
        diff = pd.DataFrame(index=wide.index)
        for metric in METRICS:
            diff[f"{metric}_6668_minus_6667"] = wide[(metric, "6668")] - wide[(metric, "6667")]
        diff["validation_set"] = val_set
        pivot_frames.append(diff.reset_index())
    diff_df = pd.concat(pivot_frames, ignore_index=True)
    diff_df.to_csv(f"{OUT_DIR}/per_gene_diff_6668_minus_6667.csv", index=False)

    print("\n=== Key diagnostic: organoid_shGFP (null condition) ===")
    print(aggregate.loc["organoid_shGFP"])
    shgfp_diff = diff_df[diff_df["validation_set"] == "organoid_shGFP"]
    print("\nMean per-gene r2 change (6668 - 6667):", shgfp_diff["r2_6668_minus_6667"].mean().round(4))
    print("Genes that got worse (r2 dropped by >0.05):")
    print(shgfp_diff[shgfp_diff["r2_6668_minus_6667"] < -0.05][["gene", "r2_6668_minus_6667"]])
    print("Genes that got better (r2 improved by >0.05):")
    print(shgfp_diff[shgfp_diff["r2_6668_minus_6667"] > 0.05][["gene", "r2_6668_minus_6667"]])

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
