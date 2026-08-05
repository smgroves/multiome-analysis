"""Fit GENIE3 on the exact inputs behind boba-T run 6667, as a third method alongside
boba-T and CellOracle for comparisons 1 and 3 (see benchmarking/README.md).

GENIE3 (Huynh-Thu et al. 2010) is, per-target-gene, a random-forest regression of that
gene's expression on its candidate regulators' expression; the edge weight TF->target is
that TF's feature importance in the forest. This is a direct sklearn implementation of
that algorithm (not the `arboreto` package) specifically so that, like CellOracle's fit
in comparison3_fit_celloracle_6667.py, each target's candidate regulators can be
restricted to exactly the same 228-edge DIRECT-NET network boba-T and CellOracle were
given -- `arboreto`'s GENIE3 only supports one global candidate-regulator list for every
target, not a per-target restriction, so it can't reproduce that fairness constraint
directly.

Uses the SAME train/test split and candidate network as comparison3_fit_celloracle_6667.py.
Writes:
    network-inference-DIRECT-NET/6667/rules/genie3_importance_matrix.csv  (rows=regulators,
        columns=targets, like celloracle_coef_matrix.csv but unsigned importances)
    network-inference-DIRECT-NET/6667/validation/genie3_validation/accuracy_plots/*.csv

Runs in any env with sklearn/pandas (bobaT_env, celloracle_env, or plain python3 here).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)
    test = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes, {len(train)} train cells, {len(test)} test cells")

    edge_list = pd.read_csv(NETWORK_CSV, header=None, names=["source", "target"])
    edge_list = edge_list[edge_list.source.isin(genes) & edge_list.target.isin(genes)]
    regulators_of = edge_list.groupby("target")["source"].apply(list).to_dict()
    print(f"[base_grn] {len(edge_list)} candidate edges over {len(genes)} genes, "
          f"{len(regulators_of)} targets with >=1 candidate regulator")

    importance = pd.DataFrame(0.0, index=genes, columns=genes)  # rows=regulators, cols=targets
    out_dir = f"{DN}/6667/validation/genie3_validation/accuracy_plots"
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for target, regs in regulators_of.items():
        regs = sorted(set(regs) - {target})  # candidate edges already exclude self-loops, but be safe
        if not regs:
            continue
        X_train, y_train = train[regs].values, train[target].values
        X_test, y_test = test[regs].values, test[target].values

        rf = RandomForestRegressor(n_estimators=1000, max_features="sqrt", random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        importance.loc[regs, target] = rf.feature_importances_

        pred = rf.predict(X_test)
        out = pd.DataFrame(
            {"actual": to01(y_test), "predicted": to01(pred)},  # match boba-T's [0,1]/0.5-threshold convention
            index=test.index,
        )
        out.index.name = "CellID"
        out.to_csv(f"{out_dir}/{target}_validation.csv")
        n_written += 1

    importance.to_csv(f"{DN}/6667/rules/genie3_importance_matrix.csv")
    print(f"[write] {n_written}/{len(genes)} genes had surviving regulators -> {out_dir}")
    print(f"[write] importance matrix -> {DN}/6667/rules/genie3_importance_matrix.csv")


if __name__ == "__main__":
    main()
