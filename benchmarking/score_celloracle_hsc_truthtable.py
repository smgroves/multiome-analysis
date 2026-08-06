import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hsc_ground_truth import load_rules, truth_table

coef = pd.read_csv("data/hsc_ground_truth/celloracle_hsc_coef_matrix.csv", index_col=0)
expr = pd.read_csv("data/hsc_ground_truth/expr_bobat.csv", index_col=0)  # all 2000 cells, [0,1]
ground_truth_rules = load_rules()

rows = []
for gene_lower, (expr_rule, gt_regs_orig) in ground_truth_rules.items():
    gene = gene_lower.upper()
    gt_regs_upper = [r.upper() for r in gt_regs_orig]
    if gene not in coef.columns:
        continue
    fitted_regs = list(coef.index[coef[gene] != 0])
    if not fitted_regs:
        continue
    predicted = (expr[fitted_regs].to_numpy() * coef.loc[fitted_regs, gene].to_numpy()).sum(axis=1)

    n_true = len(gt_regs_upper)
    true_leaf = expr[gt_regs_upper].gt(0.5).astype(int).apply(
        lambda row: int("".join(row.astype(str)), 2), axis=1
    ).to_numpy()
    gt_tt = truth_table(expr_rule, gt_regs_orig)
    gt_labels = gt_tt["output"].to_numpy()

    marginalized = np.full(2 ** n_true, np.nan)
    for leaf in range(2 ** n_true):
        mask = true_leaf == leaf
        if mask.sum() > 0:
            marginalized[leaf] = predicted[mask].mean()
    covered = ~np.isnan(marginalized)
    try:
        auc = roc_auc_score(gt_labels[covered], marginalized[covered]) if len(set(gt_labels[covered])) > 1 else np.nan
    except ValueError:
        auc = np.nan
    rows.append({"gene": gene, "n_true_regulators": n_true, "n_fitted": len(fitted_regs),
                 "leaves_covered": int(covered.sum()), "leaves_total": 2 ** n_true, "auc": auc})

res = pd.DataFrame(rows)
res.to_csv("benchmarking_out/comparison_hsc_celloracle_truth_table.csv", index=False)
print(res.to_string(index=False))
print(f"\nmean AUC: {res['auc'].mean():.3f}")
