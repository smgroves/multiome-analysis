"""Fairness check for verify_celloracle_6667.py's finding: does evaluating against the
real imputed_count (rather than raw test values) inflate R2 for ANY reasonable model
(because smoothing the evaluation target removes real noise), or is it specific to
CellOracle? Computes boba-T's OWN fitted rule's predictions against the identical
imputed_count test cells CellOracle was scored against, using the exact same real
Oracle-computed imputation (not re-derived) for a genuinely fair, single-target comparison.

Run in bobaT_env:
    /opt/anaconda3/envs/bobaT_env/bin/python verify_bobat_vs_imputed_6667.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def idx2binary(idx, n):
    b = "{0:b}".format(idx)
    return "0" * (n - len(b)) + b


def load_fitted_rules(path):
    out = {}
    with open(path) as f:
        for line in f:
            gene, regs, probs = line.strip().split("|")
            out[gene] = (regs.split(","), [float(p) for p in probs.split(",")])
    return out


def bobat_predict(data, fitted_regulators, rule):
    n = len(fitted_regulators)
    reg_vals = data[fitted_regulators].to_numpy()
    n_cells = reg_vals.shape[0]
    heat = np.ones((n_cells, 2 ** n))
    for leaf in range(2 ** n):
        binary = idx2binary(leaf, n)
        for col, bit in enumerate(binary):
            heat[:, leaf] *= reg_vals[:, col] if bit == "1" else (1 - reg_vals[:, col])
    return heat @ np.array(rule)


REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"

test_raw = pd.read_csv(f"{DN}/6667/data_split/test_t0combined.csv", index_col=0)
fitted = load_fitted_rules(f"{DN}/6667/rules/rules_6667.txt")

# real imputed test values, produced by verify_celloracle_6667.py's actual Oracle run
imputed_test = pd.read_csv("benchmarking_out/verify_celloracle_6667_imputed_test.csv", index_col=0) \
    if __import__("os").path.exists("benchmarking_out/verify_celloracle_6667_imputed_test.csv") else None

if imputed_test is None:
    print("Run verify_celloracle_6667.py first with test_imp saved to benchmarking_out/verify_celloracle_6667_imputed_test.csv")
    raise SystemExit(1)

rows = []
for gene, (regs, rule) in fitted.items():
    if gene not in test_raw.columns or gene not in imputed_test.columns:
        continue
    regs = [r for r in regs if r in test_raw.columns]
    if not regs:
        continue
    pred_raw = bobat_predict(test_raw, regs, rule)
    r2_raw = r2_score(test_raw[gene], pred_raw)

    common_idx = imputed_test.index.intersection(test_raw.index)
    pred_imp = bobat_predict(imputed_test.loc[common_idx], regs, rule)
    r2_imp = r2_score(imputed_test.loc[common_idx, gene], pred_imp)

    rows.append({"gene": gene, "n_regs": len(regs), "r2_vs_raw_test": r2_raw, "r2_vs_imputed_test": r2_imp})

res = pd.DataFrame(rows).set_index("gene")
print(res)
print()
print(f"mean R2 vs raw test (boba-T's own original number): {res['r2_vs_raw_test'].mean():.3f}")
print(f"mean R2 vs the SAME imputed test CellOracle was scored against: {res['r2_vs_imputed_test'].mean():.3f}")
