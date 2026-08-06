"""Score Track 3's ChEA-fitted rule against the literal ground-truth truth table, via
empirical marginalization -- unlike Track 1's comparison_hsc_truth_table_scoring.py, Track
3's fitted regulator set generally differs in size/identity from the true regulator set
(see benchmarking/README.md's 3-way coverage table), so the fitted rule's 2^m-leaf
probability array isn't directly indexable by the true 2^n-leaf truth table.

Marginalization approach: for each gene, evaluate the FITTED rule on every one of the
2000 simulated cells (using bobaT's own parent_heatmap-style fuzzy weighting over the
FITTED regulators' continuous values -- reimplemented here directly, no bobaT_env needed),
then group cells by their TRUE regulators' binarized state (same MSB-first convention as
hsc_ground_truth.py's itertools.product) and average the fitted rule's per-cell prediction
within each true leaf. This is the real empirical distribution of "what does the fitted
rule predict, given the true regulators' state" -- marginalizing over any extra fitted
regulators using their actual joint distribution in the data, and simply not conditioning
on any true regulator the fit never had access to.

Run in bobaT_env (needs numpy/pandas only, but keep consistent with the rest of this repo):
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_chea_truth_table_scoring.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hsc_ground_truth import load_rules, truth_table

GT_DIR = "data/hsc_ground_truth"
RULES_PATH = "hsc_chea/rules/rules_hsc_chea.txt"
OUT_PATH = "benchmarking_out/comparison_hsc_chea_truth_table.csv"


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


def predict_all_cells(data, fitted_regulators, rule):
    """bobaT's parent_heatmap + dot(heat, rule), reimplemented directly on the full dataset."""
    n = len(fitted_regulators)
    reg_vals = data[fitted_regulators].to_numpy()  # cells x n, continuous [0,1]
    n_cells = reg_vals.shape[0]
    heat = np.ones((n_cells, 2 ** n))
    for leaf in range(2 ** n):
        binary = idx2binary(leaf, n)
        for col, bit in enumerate(binary):
            if bit == "1":
                heat[:, leaf] *= reg_vals[:, col]
            else:
                heat[:, leaf] *= 1 - reg_vals[:, col]
    return heat @ np.array(rule)  # per-cell predicted probability


def main():
    data = pd.read_csv(f"{GT_DIR}/expr_bobat.csv", index_col=0)  # all 2000 cells, [0,1]
    ground_truth_rules = load_rules()
    fitted = load_fitted_rules(RULES_PATH)

    rows = []
    for gene_lower, (expr, gt_regulators_orig) in ground_truth_rules.items():
        gene = gene_lower.upper()
        if gene not in fitted:
            continue
        fitted_regs, rule = fitted[gene]
        predicted = predict_all_cells(data, fitted_regs, rule)  # per-cell, using FITTED regs

        gt_regs_upper = [r.upper() for r in gt_regulators_orig]
        n_true = len(gt_regs_upper)
        true_leaf = data[gt_regs_upper].gt(0.5).astype(int).apply(
            lambda row: int("".join(row.astype(str)), 2), axis=1
        )
        gt_tt = truth_table(expr, gt_regulators_orig)  # same MSB-first order as true_leaf above
        gt_labels = gt_tt["output"].to_numpy()

        marginalized = np.full(2 ** n_true, np.nan)
        leaf_counts = np.zeros(2 ** n_true, dtype=int)
        for leaf in range(2 ** n_true):
            mask = (true_leaf == leaf).to_numpy()
            leaf_counts[leaf] = mask.sum()
            if mask.sum() > 0:
                marginalized[leaf] = predicted[mask].mean()

        covered = ~np.isnan(marginalized)
        pred_binary = (marginalized[covered] >= 0.5).astype(int)
        gt_covered = gt_labels[covered]
        accuracy = (pred_binary == gt_covered).mean() if covered.sum() > 0 else float("nan")
        try:
            auc = roc_auc_score(gt_covered, marginalized[covered]) if len(set(gt_covered)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        overlap_frac = len(set(fitted_regs) & set(gt_regs_upper)) / n_true
        rows.append({
            "gene": gene,
            "n_true_regulators": n_true,
            "n_fitted_regulators": len(fitted_regs),
            "regulator_overlap_frac": round(overlap_frac, 3),
            "leaves_covered": int(covered.sum()),
            "leaves_total": 2 ** n_true,
            "min_leaf_n": int(leaf_counts[covered].min()) if covered.sum() > 0 else 0,
            "truth_table_accuracy": accuracy,
            "truth_table_auc": auc,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)
    print(result.to_string(index=False))
    valid = result.dropna(subset=["truth_table_accuracy"])
    print(f"\nmean truth-table accuracy (genes with >=1 covered leaf): {valid['truth_table_accuracy'].mean():.3f}")
    print(f"mean truth-table AUC: {valid['truth_table_auc'].mean():.3f}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
