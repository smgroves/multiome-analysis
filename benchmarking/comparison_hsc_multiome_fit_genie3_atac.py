"""Fit GENIE3 on the real GSE194122 multiome data (Track 2), same sklearn RandomForest
approach and same ChEA-derived candidate network as boba-T's Track 2 run
(comparison_hsc_multiome_fit_bobat.py) -- reuses boba-T's own train/test split so both
methods are scored on identical held-out cells.

Run in any env with sklearn/pandas (bobaT_env is fine):
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_genie3.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = "data/hsc_multiome"
BRCD = "hsc_multiome"


def to01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    train = pd.read_csv(f"{BRCD}/data_split/train_t0{BRCD}.csv", index_col=0)
    test = pd.read_csv(f"{BRCD}/data_split/test_t0{BRCD}.csv", index_col=0)
    genes = list(train.columns)
    print(f"[data] {len(genes)} genes, {len(train)} train cells, {len(test)} test cells")

    edge_list = pd.read_csv(f"{DATA_DIR}/candidate_network_atac_real.csv", header=None, names=["source", "target"])
    edge_list = edge_list[edge_list.source.isin(genes) & edge_list.target.isin(genes)]
    regulators_of = edge_list.groupby("target")["source"].apply(list).to_dict()
    print(f"[base] {len(edge_list)} candidate edges, {len(regulators_of)} targets with >=1 candidate regulator")

    importance = pd.DataFrame(0.0, index=genes, columns=genes)  # rows=regulators, cols=targets
    out_dir = f"{BRCD}/validation/genie3_atac_validation/accuracy_plots"
    os.makedirs(out_dir, exist_ok=True)

    fitted_regulators = {}
    for target, regs in regulators_of.items():
        regs = sorted(set(regs) - {target})
        if not regs:
            continue
        X_train, y_train = train[regs].values, train[target].values
        X_test, y_test = test[regs].values, test[target].values

        rf = RandomForestRegressor(n_estimators=1000, max_features="sqrt", random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        importance.loc[regs, target] = rf.feature_importances_
        fitted_regulators[target] = regs

        pred = rf.predict(X_test)
        out = pd.DataFrame({"actual": to01(y_test), "predicted": to01(pred)}, index=test.index)
        out.index.name = "CellID"
        out.to_csv(f"{out_dir}/{target}_validation.csv")

    importance.to_csv(f"{BRCD}/rules/genie3_atac_importance_matrix.csv")
    print(f"[write] {len(fitted_regulators)}/{len(genes)} genes fit -> {out_dir}")
    print(f"[write] importance matrix -> {BRCD}/rules/genie3_atac_importance_matrix.csv")
    for gene, regs in fitted_regulators.items():
        print(f"  {gene}: candidate regulators = {regs}")


if __name__ == "__main__":
    main()
