"""Score boba-T's FITTED RULE against the HSC model's actual ground-truth Boolean rule --
not "is there an edge" (comparison 1) but "is the combinatorial LOGIC right." This is the
comparison that isolates boba-T's stated advantage: a method that only outputs one scalar
edge weight per (regulator, target) pair has no object comparable to a truth table at all;
boba-T's fitted rule literally is one, so it can be compared truth-table-to-truth-table.

Bit-order convention verified directly from bobaT's own source (not guessed), since a
truth table is meaningless if the two sides don't agree on which bit is which regulator:
    bobaT/tl.py:parent_heatmap -- for regulators_dict[gene] = [r0, r1, ..., r(n-1)],
    leaf index i's binary representation ut.idx2binary(i, n) (MSB-first, bobaT/utils.py)
    has its j-th character = state of regulators[j]. So r0 is the MSB. This script rebuilds
    the ground-truth truth table using each gene's *actual fitted* regulator order (from
    rules_hsc.txt), not hsc_ground_truth.py's alphabetical order, specifically so both
    sides use an identical bit convention -- comparing two truth tables built with
    different regulator orderings would silently misscore every gene with >1 regulator.

Run in any env with pandas/numpy/sklearn (bobaT_env is fine, already has them):
    /opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_truth_table_scoring.py
"""

import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score

from hsc_ground_truth import load_rules, truth_table

RULES_PATH = "hsc/rules/rules_hsc.txt"
OUT_PATH = "benchmarking_out/comparison_hsc_truth_table_bobat.csv"


def load_bobat_rules(path: str) -> dict[str, tuple[list[str], list[float]]]:
    """Parse rules_hsc.txt: `gene|reg1,reg2,...|p0,p1,...,p_(2^n-1)` per line."""
    out = {}
    with open(path) as f:
        for line in f:
            gene, regs, probs = line.strip().split("|")
            out[gene] = (regs.split(","), [float(p) for p in probs.split(",")])
    return out


def main():
    ground_truth_rules = load_rules()  # {gene: (expr, regulators)}, original HSC.txt case
    bobat_rules = load_bobat_rules(RULES_PATH)  # {GENE: (regulators_upper, probs)}

    rows = []
    for gene_lower, (expr, gt_regulators) in ground_truth_rules.items():
        gene = gene_lower.upper()
        if gene not in bobat_rules:
            continue
        fitted_regs_upper, probs = bobat_rules[gene]

        # Rebuild the ground-truth truth table using boba-T's OWN fitted regulator order,
        # so both sides share one bit convention. Requires the same regulator SET (already
        # verified to match exactly for all 11 HSC genes) -- map upper-case fitted names
        # back to the original-case tokens `expr` uses for eval().
        case_map = {r.upper(): r for r in gt_regulators}
        if set(fitted_regs_upper) != set(case_map):
            rows.append({"gene": gene, "note": "regulator set mismatch, not scored"})
            continue
        ordered_original_case = [case_map[r] for r in fitted_regs_upper]
        gt_tt = truth_table(expr, ordered_original_case)  # index order matches idx2binary's MSB-first convention
        gt_labels = gt_tt["output"].to_numpy()

        n = len(fitted_regs_upper)
        assert len(probs) == 2 ** n == len(gt_labels)

        pred_binary = [1 if p >= 0.5 else 0 for p in probs]
        accuracy = sum(int(p == g) for p, g in zip(pred_binary, gt_labels)) / len(gt_labels)
        try:
            auc = roc_auc_score(gt_labels, probs) if len(set(gt_labels)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        rows.append({
            "gene": gene,
            "n_regulators": n,
            "regulators": ",".join(fitted_regs_upper),
            "truth_table_accuracy": accuracy,
            "truth_table_auc": auc,
            "ground_truth_on_fraction": gt_labels.mean(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)
    print(result.to_string(index=False))
    print(f"\nmean truth-table accuracy: {result['truth_table_accuracy'].mean():.3f}")
    print(f"mean truth-table AUC: {result['truth_table_auc'].mean():.3f}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
