"""Comparison 3 (predicted vs. actual expression), scoring step: run boba-T's own
get_sklearn_metrics on every method's validation CSVs for run 6667, and join them into
one long-format per-gene-per-method comparison table.

Fitting scripts write each method's <gene>_validation.csv (comparison3_fit_celloracle_6667.py
for CellOracle, comparison_genie3_fit_6667.py for GENIE3); boba-T's own 6667 run already
wrote its own. This script only needs boba-T's package, so it runs in bobaT_env (not
celloracle_env):

    /opt/anaconda3/envs/bobaT_env/bin/python comparison3_score_celloracle_vs_bobat_6667.py
"""

import os

import bobaT as bb
import pandas as pd

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
OUT_DIR = f"{REPO}/benchmarking/benchmarking_out"

METRIC_COLS = ["r2", "roc_auc_score", "f1", "accuracy", "balanced_accuracy_score",
               "precision", "recall", "explained_variance", "log-loss"]

# method -> validation dir under 6667/validation/. Add a new method by adding a row here
# once its fitting script has written <gene>_validation.csv under <dir>/accuracy_plots/.
VALIDATION_DIRS = {
    "boba-T": f"{DN}/6667/validation/in_sample_validation",
    "CellOracle": f"{DN}/6667/validation/celloracle_validation",
    "GENIE3": f"{DN}/6667/validation/genie3_validation",
}


def main():
    per_method = {}
    for method, val_dir in VALIDATION_DIRS.items():
        stats = bb.tl.get_sklearn_metrics(val_dir, save=False, plot_cm=False)
        per_method[method] = stats.set_index("gene")[METRIC_COLS]
        print(f"[load] {method}: {len(stats)} genes scored")

    # Shared-node rule: only genes every method could call.
    shared_genes = set.intersection(*(set(df.index) for df in per_method.values()))
    long = pd.concat(
        {m: df.loc[sorted(shared_genes)] for m, df in per_method.items()},
        names=["method", "gene"],
    ).reset_index()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/comparison3_all_methods_vs_bobat_6667.csv"
    long.to_csv(out_path, index=False)

    print(f"\nn shared genes across {len(per_method)} methods: {len(shared_genes)}")
    print(long.groupby("method")[["r2", "roc_auc_score", "f1"]].mean())
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
