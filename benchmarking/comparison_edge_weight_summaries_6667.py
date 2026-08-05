"""Alternative summaries of boba-T's fitted Boolean rules as edge weights.

boba-T's `get_rules` (bobaT/tl.py:244) fits, per target gene with n regulators, a full
truth table `rule` of length 2**n: P(target ON | this combination of regulator states),
one entry per leaf of the binary decision tree. `signed_strengths.csv` / `strengths.csv`
(what every other script in this benchmark treats as "the edge weight") collapse that
whole truth table down to ONE number per regulator by summing the regulator's ON-vs-OFF
effect across every possible combination of the *other* regulators
(bobaT/tl.py:52 detect_irrelevant_regulator: `tot_dif`/`signed_tot_dif`).

That summary treats every combination of co-regulators as equally likely, and averages
away any context-dependence -- exactly the two things flagged as suspect:
  - an edge active only on some leaves (context-/canalizing-dependent regulation) gets
    diluted by summing/averaging over many leaves where it may not matter;
  - an edge whose effect flips sign across contexts can partially cancel in a sum,
    understating a regulator that is genuinely influential but not in one fixed direction.

This script builds several alternative per-(regulator,target) summaries from the SAME
already-fitted truth tables in `rules_6667.txt` (no need to refit boba-T) and scores each
against the three real ChIP-seq ground truths already used in comparison 1, via
comparison 2's ranking metrics (AUROC/EPR) -- comparison 1's binary edge-existence metrics
(Jaccard/precision/recall/F1) can't distinguish these, since which regulators survive at
all is already decided at fit time (by `max_dif < threshold` pruning); only which
regulators rank as *strongest* changes, which is exactly what AUROC/EPR is sensitive to.

Summaries computed, per regulator r of target t (contexts = every combination of t's
*other* regulators; `rule[on]`/`rule[off]` = the truth-table entries with r's bit flipped,
context held fixed):
    sum_abs      boba-T's current strengths.csv:        sum_context |rule[on]-rule[off]|
    sum_signed   boba-T's current signed_strengths.csv:  sum_context  rule[on]-rule[off]
    mean_abs     sum_abs / n_contexts                    -- comparable across targets with
                 different regulator counts, unlike the raw sum (n_contexts = 2**(n-1)
                 differs per target, so sum_abs alone systematically favors regulators of
                 heavily-co-regulated targets).
    mean_signed  sum_signed / n_contexts
    max_abs      max_context |rule[on]-rule[off]|        -- boba-T computes this internally
                 (`max_dif`) to decide pruning, then DISCARDS it; never written to any
                 output file. Captures a regulator whose effect is concentrated in one or
                 few contexts (canalizing / conditional regulation) even if its average
                 effect is small.
    max_signed   the SIGNED difference at the context of maximum |diff| -- can disagree in
                 sign with sum_signed/mean_signed for a genuinely context-dependent
                 regulator (activates in some contexts, represses in others); sum_signed
                 would average that toward zero, max_signed preserves the sign where the
                 effect is strongest.
    dataw_abs    context-weighted mean of |rule[on]-rule[off]|, weighted by how often that
                 combination of the *other* regulators actually occurs in the real training
                 data (recomputes boba-T's own per-cell leaf-membership weighting, `heat`
                 in get_rules, which is computed and then discarded) -- contexts that never
                 really occur in the data (e.g. mutually exclusive TF programs) stop
                 contributing to the summary, contexts that dominate the real cell
                 population dominate the summary too.
    dataw_signed same weighting applied to the signed difference.

Run in any env with pandas/numpy (no boba-T/graph_tool import needed -- this only reads
rules_6667.txt as a plain text file and reimplements boba-T's tiny index-arithmetic
helpers directly, see idx2binary/get_leaves_of_regulator below, copied from
bobaT/utils.py to avoid a graph_tool dependency for this analysis).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from grn_benchmark.edges import matrix_to_edges
from grn_benchmark.loaders import load_sclc_chipseq_gt
from grn_benchmark.metrics import run_beeline_comparison, structure_metrics

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
GT_DIR = "data/sclc_chipseq_gt"
OUT_DIR = "benchmarking_out"

GROUND_TRUTHS = {
    "borromeo2016_ascl1_human": f"{GT_DIR}/borromeo2016_ascl1_human_chip.csv",
    "borromeo2016_ascl1_mouse": f"{GT_DIR}/borromeo2016_ascl1_mouse_chip.csv",
    "pozo2021_ascl1_direct": f"{GT_DIR}/pozo2021_ascl1_direct.csv",
}


# --- tiny index-arithmetic helpers, copied from bobaT/utils.py (idx2binary,
# get_leaves_of_regulator) so this script doesn't need to import boba-T / graph_tool. ---

def idx2binary(idx: int, n: int) -> str:
    binary = "{0:b}".format(idx)
    return "0" * (n - len(binary)) + binary


def get_leaves_of_regulator(n_leaves: int, index: int):
    step_size = int(round(n_leaves / (2 ** (index + 1))))
    num_steps = int(round(n_leaves / step_size / 2))
    off_leaves, on_leaves = [], []
    base = 0
    for _ in range(num_steps):
        off_leaves.extend(range(base, base + step_size))
        base += step_size
        on_leaves.extend(range(base, base + step_size))
        base += step_size
    return off_leaves, on_leaves


def load_rules(fname: str):
    rules, regulators_dict = {}, {}
    with open(fname) as f:
        for line in f:
            gene, regs, rule = line.strip().split("|")
            regulators_dict[gene] = regs.split(",")
            rules[gene] = np.asarray([float(v) for v in rule.split(",")])
    return rules, regulators_dict


def leaf_weights_from_data(regulator_values: np.ndarray) -> np.ndarray:
    """Empirical P(this exact combination of regulator states) over the training data,
    recomputing boba-T's own `heat` (get_rules, bobaT/tl.py:293) from data alone -- it
    doesn't depend on the target gene's fitted rule, only on regulator_values
    (cells x n_regulators, soft-binarized in [0,1]).
    """
    n_cells, n = regulator_values.shape
    n_leaves = 2 ** n
    weights = np.zeros(n_leaves)
    for leaf in range(n_leaves):
        binary = idx2binary(leaf, n)
        per_cell = np.ones(n_cells)
        for col, bit in enumerate(binary):
            p_on = regulator_values[:, col]
            per_cell *= p_on if bit == "1" else (1 - p_on)
        weights[leaf] = per_cell.mean()
    return weights  # sums to ~1 across leaves


def edge_summaries_for_gene(gene: str, regs: list, rule: np.ndarray, data: pd.DataFrame) -> dict:
    """Returns {regulator: {summary_name: value}} for one target gene's fitted rule."""
    n = len(regs)
    leaf_w = None
    if all(r in data.columns for r in regs):
        leaf_w = leaf_weights_from_data(data[regs].values)

    out = {}
    for k, reg in enumerate(regs):
        off_leaves, on_leaves = get_leaves_of_regulator(2 ** n, k)
        diffs = rule[on_leaves] - rule[off_leaves]  # signed, one per context
        abs_diffs = np.abs(diffs)

        n_contexts = len(diffs)
        i_max = int(np.argmax(abs_diffs))

        summaries = {
            "sum_abs": abs_diffs.sum(),
            "sum_signed": diffs.sum(),
            "mean_abs": abs_diffs.mean(),
            "mean_signed": diffs.mean(),
            "max_abs": abs_diffs[i_max],
            "max_signed": diffs[i_max],
        }
        if leaf_w is not None:
            off_w = leaf_w[off_leaves]
            on_w = leaf_w[on_leaves]
            context_w = off_w + on_w  # P(the other n-1 regulators = this combination)
            total_w = context_w.sum()
            if total_w > 0:
                summaries["dataw_abs"] = float(np.sum(context_w * abs_diffs) / total_w)
                summaries["dataw_signed"] = float(np.sum(context_w * diffs) / total_w)
            else:
                summaries["dataw_abs"] = summaries["dataw_signed"] = 0.0
        out[reg] = summaries
    return out


def build_all_matrices(rules: dict, regulators_dict: dict, data: pd.DataFrame, nodes: list):
    """One target(rows) x regulator(cols) matrix per summary method."""
    summary_names = ["sum_abs", "sum_signed", "mean_abs", "mean_signed",
                      "max_abs", "max_signed", "dataw_abs", "dataw_signed"]
    matrices = {name: pd.DataFrame(0.0, index=nodes, columns=nodes) for name in summary_names}

    for gene, regs in regulators_dict.items():
        if regs == [gene]:
            continue  # boba-T's placeholder for "no real regulators found" -- not a fitted edge
        per_reg = edge_summaries_for_gene(gene, regs, rules[gene], data)
        for reg, summaries in per_reg.items():
            for name, val in summaries.items():
                matrices[name].loc[gene, reg] = val
    return matrices


def local_auroc(mat: pd.DataFrame, gt: pd.DataFrame, source: str = "ASCL1") -> dict:
    """AUROC/EPR restricted to candidates = source -> every OTHER gene actually in boba-T's
    53-gene network (52 candidates), not source -> every gene in the ground truth's much
    larger universe (hundreds to thousands, nearly all of which boba-T never scores at all).

    Why this is needed: comparison 2's usual `run_beeline_comparison` scores the full
    ground-truth gene universe as the candidate set. With an ASCL1-only ground truth and a
    53-gene network, only ~13 of those candidates ever get a nonzero score from boba-T at
    all -- everything else is tied at exactly 0. roc_auc_score is then almost entirely
    determined by "does any nonzero score beat the sea of zero-scored negatives", which is
    true for every summary method here regardless of how those 13 edges are RELATIVELY
    ranked against each other -- so the different summaries came out numerically identical
    to 6 decimal places when this script's first pass used run_beeline_comparison directly.
    Restricting the candidate universe to boba-T's own 52 possible ASCL1-sourced edges makes
    the metric actually sensitive to how the 13 nonzero-scored edges are ordered relative to
    each other and to the ~39 candidates boba-T scored as exactly 0.
    """
    targets = [g for g in mat.index if g != source]
    scores = mat.loc[targets, source].abs().values if source in mat.columns else np.zeros(len(targets))
    gt_targets = set(gt.loc[gt.source == source, "target"])
    y_true = np.array([1 if t in gt_targets else 0 for t in targets])
    n_gold = int(y_true.sum())
    if n_gold == 0 or n_gold == len(y_true):
        return {"local_auroc": np.nan, "local_epr": np.nan, "local_n_gold": n_gold}
    auroc = roc_auc_score(y_true, scores)
    k = n_gold
    top_k = np.argsort(-scores)[:k]
    epr = y_true[top_k].mean() / (n_gold / len(y_true))
    return {"local_auroc": auroc, "local_epr": epr, "local_n_gold": n_gold}


def main():
    rules, regulators_dict = load_rules(f"{DN}/6667/rules/rules_6667.txt")
    nodes = list(regulators_dict.keys())
    train = pd.read_csv(f"{DN}/6667/data_split/train_t0combined.csv", index_col=0)

    matrices = build_all_matrices(rules, regulators_dict, train, nodes)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for summary_name, mat in matrices.items():
        # matrix_to_edges expects rows=targets, columns=regulators -> regulators_on_columns=True
        edges = matrix_to_edges(mat, regulators_on_columns=True)
        mat.to_csv(f"{DN}/6667/rules/edge_summary_{summary_name}.csv")

        for gt_name, gt_path in GROUND_TRUTHS.items():
            gt = load_sclc_chipseq_gt(gt_path)
            nodes_gt = set(gt.source) | set(gt.target)
            beeline = run_beeline_comparison({summary_name: edges}, gt, nodes_gt).iloc[0]
            struct = structure_metrics(edges, gt)
            local = local_auroc(mat, gt)
            rows.append({
                "summary": summary_name, "ground_truth": gt_name,
                "auroc": beeline["auroc"], "epr": beeline["epr"],
                "n_gold": beeline["n_gold"], "n_candidates": beeline["n_candidates"],
                "sign_concordance": struct["sign_concordance"],
                **local,
            })

    results = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/comparison_edge_weight_summaries_6667.csv"
    results.to_csv(out_path, index=False)

    print("global AUROC (ground-truth-gene-universe candidates -- see local_auroc()'s"
          " docstring for why this is ~uninformative here, identical across summaries):")
    print(results.pivot(index="summary", columns="ground_truth", values="auroc").round(4))

    print("\nlocal AUROC (candidates restricted to boba-T's own 52 possible ASCL1-> edges"
          " -- THIS is the one that actually differentiates the summary methods):")
    print(results.pivot(index="summary", columns="ground_truth", values="local_auroc").round(4))

    print("\nlocal EPR:")
    print(results.pivot(index="summary", columns="ground_truth", values="local_epr").round(3))

    print("\nsign_concordance (only meaningful vs. pozo2021_ascl1_direct, the signed GT):")
    print(results.pivot(index="summary", columns="ground_truth", values="sign_concordance").round(3))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
