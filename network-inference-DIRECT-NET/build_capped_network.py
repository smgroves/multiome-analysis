"""Phase 2, item 2 (BoBa-T_hyperparameters.md sec 3): apply the regulators-per-gene cap
sizing heuristic to 6667's 228-edge candidate network. With ~7126 training cells, a gene
with 8 regulators has a 2^8=256-leaf truth table (~28 cells/leaf on average, already
sparse); this caps every gene at CAP=6 regulators (2^6=64 leaves, ~110 cells/leaf on
average), affecting the 10 genes that currently have 7 or 8.

Ranking criterion for which regulators to drop: 6667/rules/strengths.csv, the unsigned
total-relevance strength `get_rules` already computed per edge when it fit rules_6667.txt
(rows=target, columns=regulator -- confirmed directly: row "CUX2" has non-null values in
exactly CUX2's 7 regulator columns) -- keeps each capped gene's CAP highest-fitted-relevance
regulators, drops the rest. This is a pure prefilter on the edge list; no BoBa-T source
change.

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python build_capped_network.py
"""

import os

import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
OUT_NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined_cap6.csv"
)
CAP = 6


def main():
    edges = pd.read_csv(f"{DIR_PREFIX}/{NETWORK_PATH}", header=None, names=["source", "target"])
    strengths = pd.read_csv(f"{DIR_PREFIX}/6667/rules/strengths.csv", index_col=0)

    print(f"Loaded {len(edges)} edges. Regulator counts before capping:")
    counts_before = edges.groupby("target").size().sort_values(ascending=False)
    print(counts_before.head(12))

    keep_rows = []
    dropped = []
    for target, group in edges.groupby("target"):
        regulators = group["source"].tolist()
        if len(regulators) <= CAP:
            keep_rows.extend([(r, target) for r in regulators])
            continue
        if target not in strengths.index:
            print(f"WARNING: {target} not in strengths.csv rows, keeping all {len(regulators)} regulators")
            keep_rows.extend([(r, target) for r in regulators])
            continue
        row = strengths.loc[target]
        ranked = row.loc[[r for r in regulators if r in row.index]].dropna().sort_values(ascending=False)
        keep = list(ranked.index[:CAP])
        drop = [r for r in regulators if r not in keep]
        keep_rows.extend([(r, target) for r in keep])
        for r in drop:
            dropped.append((r, target, row.get(r, float("nan"))))

    capped = pd.DataFrame(keep_rows, columns=["source", "target"])
    counts_after = capped.groupby("target").size().sort_values(ascending=False)

    print(f"\n{len(dropped)} edges dropped (weakest regulator per over-cap gene):")
    dropped_df = pd.DataFrame(dropped, columns=["regulator", "target", "strength"])
    print(dropped_df.to_string(index=False))

    print(f"\nRegulator counts after capping (max should be {CAP}):")
    print(counts_after.head(12))

    capped.to_csv(f"{DIR_PREFIX}/{OUT_NETWORK_PATH}", header=False, index=False)
    print(f"\nWrote {len(capped)} edges to {DIR_PREFIX}/{OUT_NETWORK_PATH}")


if __name__ == "__main__":
    main()
