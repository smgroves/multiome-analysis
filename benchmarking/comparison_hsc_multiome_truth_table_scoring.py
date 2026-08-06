"""Truth-table comparison for Track 2 (real data): unlike Tracks 1/3's synthetic data,
there's no literal simulator output to serve as ground truth here. The meaningful analog:
group real cells by their TRUE regulators' (from hsc_ground_truth.py) binarized real
expression state, then check whether each fitted model's prediction -- averaged within
that group -- lands on the correct side of the literal HSC.txt rule's 0/1 label for that
state. AUROC (not raw accuracy) is used since it's scale-invariant: boba-T's prediction is
naturally bounded [0,1], CellOracle's is an unbounded raw linear combination, and AUROC
lets both be compared on the same footing without a method-specific threshold choice.

Compares boba-T+DIRECT-NET vs CellOracle+Cicero (the actual pipelines requested), plus
boba-T+Cicero as a same-network control isolating "network choice" from "fitting method."

Run in bobaT_env:
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_truth_table_scoring.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hsc_ground_truth import load_rules, truth_table

GT_DIR = "."
BOBAT_NORM_PATH = "hsc_multiome_{run}/data_split/train_t0hsc_multiome_{run}.csv"
BOBAT_RULES_PATH = "hsc_multiome_{run}/rules/rules_hsc_multiome_{run}.txt"


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


def score_bobat(run_name):
    data = pd.read_csv(BOBAT_NORM_PATH.format(run=run_name), index_col=0)  # boba-T's own [0,1]-normalized data
    fitted = load_fitted_rules(BOBAT_RULES_PATH.format(run=run_name))
    ground_truth_rules = load_rules()
    rows = []
    for gene_lower, (expr, gt_regs_orig) in ground_truth_rules.items():
        gene = gene_lower.upper()
        gt_regs_upper = [r.upper() for r in gt_regs_orig if r.upper() in data.columns]
        if gene not in fitted or gene not in data.columns or len(gt_regs_upper) == 0:
            continue
        fitted_regs, rule = fitted[gene]
        fitted_regs = [r for r in fitted_regs if r in data.columns]
        if not fitted_regs:
            continue
        predicted = bobat_predict(data, fitted_regs, rule)
        rows.append(_score_one(gene, gt_regs_upper, gt_regs_orig, expr, data, predicted))
    return pd.DataFrame([r for r in rows if r is not None])


def score_celloracle(coef_path, expr_path, run_name):
    coef = pd.read_csv(coef_path, index_col=0)  # rows=source, cols=target, raw scale
    expr = pd.read_csv(expr_path, index_col=0).clip(lower=0)
    norm_data = pd.read_csv(BOBAT_NORM_PATH.format(run=run_name), index_col=0)  # for binning only
    ground_truth_rules = load_rules()
    rows = []
    for gene_lower, (expr_rule, gt_regs_orig) in ground_truth_rules.items():
        gene = gene_lower.upper()
        gt_regs_upper = [r.upper() for r in gt_regs_orig if r.upper() in norm_data.columns]
        if gene not in coef.columns or len(gt_regs_upper) == 0:
            continue
        fitted_regs = list(coef.index[coef[gene] != 0])
        if not fitted_regs:
            continue
        common_idx = norm_data.index.intersection(expr.index)
        predicted = (expr.loc[common_idx, fitted_regs].to_numpy() * coef.loc[fitted_regs, gene].to_numpy()).sum(axis=1)
        predicted = pd.Series(predicted, index=common_idx)
        rows.append(_score_one(gene, gt_regs_upper, gt_regs_orig, expr_rule, norm_data.loc[common_idx], predicted.to_numpy()))
    return pd.DataFrame([r for r in rows if r is not None])


def _score_one(gene, gt_regs_upper, gt_regs_orig, expr_rule, binning_data, predicted):
    # If EGRNAB is among the TRUE regulators, the rule's output genuinely depends on it --
    # dropping it isn't "using fewer regulators," it's evaluating an undefined truth table
    # (eval() would still reference EgrNab and fail, or silently give a wrong marginal).
    # Skip these genes rather than fudge them; Track 2 has no real EGRNAB data throughout.
    if len(gt_regs_upper) != len(gt_regs_orig):
        return None
    n_true = len(gt_regs_upper)
    true_leaf = binning_data[gt_regs_upper].gt(0.5).astype(int).apply(
        lambda row: int("".join(row.astype(str)), 2), axis=1
    ).to_numpy()
    gt_tt = truth_table(expr_rule, gt_regs_orig)
    gt_labels = gt_tt["output"].to_numpy()

    marginalized, leaf_counts = [], []
    for leaf in range(2 ** n_true):
        mask = true_leaf == leaf
        leaf_counts.append(mask.sum())
        marginalized.append(predicted[mask].mean() if mask.sum() > 0 else np.nan)
    marginalized = np.array(marginalized)
    covered = ~np.isnan(marginalized)
    if covered.sum() < 2 or len(set(gt_labels[covered])) < 2:
        return {"gene": gene, "n_true_regulators": n_true, "leaves_covered": int(covered.sum()),
                "leaves_total": 2 ** n_true, "auc": np.nan}
    try:
        auc = roc_auc_score(gt_labels[covered], marginalized[covered])
    except ValueError:
        auc = np.nan
    return {"gene": gene, "n_true_regulators": n_true, "leaves_covered": int(covered.sum()),
            "leaves_total": 2 ** n_true, "auc": auc}


def main():
    print("=== boba-T + DIRECT-NET network ===")
    bobat_dn = score_bobat("directnet")
    print(bobat_dn.to_string(index=False))
    print(f"mean AUC: {bobat_dn['auc'].mean():.3f}\n")

    print("=== boba-T + Cicero network (same-network control) ===")
    bobat_ci = score_bobat("cicero")
    print(bobat_ci.to_string(index=False))
    print(f"mean AUC: {bobat_ci['auc'].mean():.3f}\n")

    print("=== CellOracle + Cicero network ===")
    co_ci = score_celloracle("data/hsc_multiome/celloracle_cicero_coef_matrix.csv",
                              "data/hsc_multiome/expr_bobat_real.csv", "cicero")
    print(co_ci.to_string(index=False))
    print(f"mean AUC: {co_ci['auc'].mean():.3f}\n")

    bobat_dn.to_csv("benchmarking_out/comparison_hsc_multiome_truth_table_bobat_directnet.csv", index=False)
    bobat_ci.to_csv("benchmarking_out/comparison_hsc_multiome_truth_table_bobat_cicero.csv", index=False)
    co_ci.to_csv("benchmarking_out/comparison_hsc_multiome_truth_table_celloracle_cicero.csv", index=False)


if __name__ == "__main__":
    main()
