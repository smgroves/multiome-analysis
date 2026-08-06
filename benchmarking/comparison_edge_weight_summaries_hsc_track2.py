"""Edge-weight summarization comparison (see comparison_edge_weight_summaries_6667.py for
the method and full rationale), rerun on HSC Track 2 -- boba-T fit on REAL GSE194122 human
bone-marrow multiome data, against four independently-built real candidate networks (ChEA,
ATAC+motif, DIRECT-NET, Cicero) -- instead of the 6667 SCLC network.

Why Track 2 is a better test bed than 6667's ChIP-seq ground truths: those only ever gave
4-25 shared nodes to score against (ASCL1-only, single source TF) -- too little N to
resolve differences between summary methods except by luck. Track 2 has a LITERAL,
unambiguous Boolean ground truth (Krumsiek et al. 2011's HSC model, see
benchmarking/hsc_ground_truth.py) for every one of the ~9-10 fitted genes, each with its
OWN independently-known true regulator set, scored across four independently-built real
candidate networks at once -- more genes, more networks, and (new here) enough information
to test the user's original motivating concern directly: does a regulator's TRUE role
being conditional/combinatorial (an AND/OR/NOT gate with its co-regulators) actually predict
which summary method recovers it better, not just whether summary method matters at all.

Two analyses:
    1. Structure recovery (like 6667): pooled, per-network AUROC of "is this candidate
       regulator a TRUE regulator of this gene", scored with each of the 8 summaries as
       the ranking score -- across all 4 real Track 2 networks.
    2. NEW -- canalization check: for every TRUE regulator of every HSC gene, compute its
       ground-truth "context-dependence" directly from the literal truth table (how much
       its effect varies depending on the OTHER true regulators' states -- exactly the
       quantity a sum/mean can wash out and a max/data-weighted summary is designed to
       preserve). Split true regulators into "conditional" (truly canalized/context-
       dependent in the real model) vs "unconditional", and check whether max/dataw-style
       summaries recover the conditional ones better than sum/mean-style summaries do,
       specifically among the harder, genuinely combinatorial cases -- the concrete,
       ground-truth-backed version of the concern that motivated this whole comparison.

Run in any env with pandas/numpy/sklearn (no boba-T/graph_tool import needed, same as
comparison_edge_weight_summaries_6667.py, whose helpers this script reuses directly).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from comparison_edge_weight_summaries_6667 import (
    edge_summaries_for_gene,
    load_rules,
)
from hsc_ground_truth import load_rules as load_ground_truth_rules, truth_table

OUT_DIR = "benchmarking_out"

# run suffix -> (rules file, training-data file, human label)
NETWORKS = {
    "chea": ("hsc_multiome/rules/rules_hsc_multiome.txt",
             "hsc_multiome/data_split/train_t0hsc_multiome.csv", "ChEA"),
    "atac": ("hsc_multiome_atac/rules/rules_hsc_multiome_atac.txt",
             "hsc_multiome_atac/data_split/train_t0hsc_multiome_atac.csv", "real ATAC+motif"),
    "directnet": ("hsc_multiome_directnet/rules/rules_hsc_multiome_directnet.txt",
                  "hsc_multiome_directnet/data_split/train_t0hsc_multiome_directnet.csv", "DIRECT-NET"),
    "cicero": ("hsc_multiome_cicero/rules/rules_hsc_multiome_cicero.txt",
               "hsc_multiome_cicero/data_split/train_t0hsc_multiome_cicero.csv", "Cicero"),
}

SUMMARY_NAMES = ["sum_abs", "sum_signed", "mean_abs", "mean_signed",
                 "max_abs", "max_signed", "dataw_abs", "dataw_signed"]


def build_summaries_for_network(rules_path: str, data_path: str) -> pd.DataFrame:
    """Long-format (target, regulator, summary_name, value) for every fitted edge."""
    rules, regulators_dict = load_rules(rules_path)
    data = pd.read_csv(data_path, index_col=0)

    rows = []
    for gene, regs in regulators_dict.items():
        if regs == [gene]:
            continue  # boba-T's "no real regulators survived" fallback -- not a fitted edge
        per_reg = edge_summaries_for_gene(gene, regs, rules[gene], data)
        for reg, summaries in per_reg.items():
            for name, val in summaries.items():
                rows.append({"target": gene, "regulator": reg, "summary": name, "value": val})
    return pd.DataFrame(rows)


def ground_truth_regulators() -> dict[str, set[str]]:
    gt_rules = load_ground_truth_rules()
    return {gene.upper(): {r.upper() for r in regs} for gene, (_, regs) in gt_rules.items()}


def analysis_1_structure_recovery(all_summaries: dict[str, pd.DataFrame], gt_regs: dict) -> pd.DataFrame:
    """Pooled AUROC per (network, summary): does the summary rank true regulators above
    false-candidate ones, across every fitted gene in that network at once."""
    rows = []
    for network, df in all_summaries.items():
        for summary in SUMMARY_NAMES:
            sub = df[df["summary"] == summary].copy()
            sub["y_true"] = [
                1 if reg in gt_regs.get(target, set()) else 0
                for reg, target in zip(sub["regulator"], sub["target"])
            ]
            sub["score"] = sub["value"].abs()
            n_pos, n_neg = sub["y_true"].sum(), (1 - sub["y_true"]).sum()
            auroc = roc_auc_score(sub["y_true"], sub["score"]) if 0 < n_pos < len(sub) else np.nan
            rows.append({"network": network, "summary": summary, "auroc": auroc,
                          "n_true_edges": int(n_pos), "n_candidate_edges": len(sub)})
    return pd.DataFrame(rows)


def true_context_dependence() -> pd.DataFrame:
    """For every TRUE regulator of every HSC gene: how context-dependent is its effect in
    the LITERAL ground-truth rule? Same per-context-diff machinery as the fitted-rule
    summaries, applied to the ground-truth truth table instead -- so "conditional" here
    means "verified against the actual Boolean formula," not inferred/guessed.
    """
    gt_rules = load_ground_truth_rules()
    rows = []
    for gene_lower, (expr, regs) in gt_rules.items():
        gene = gene_lower.upper()
        regs_upper = [r.upper() for r in regs]
        if len(regs_upper) < 2:
            continue  # context-dependence is undefined with a single regulator (0 "other" regulators)
        tt = truth_table(expr, regs)
        n = len(regs_upper)
        rule = tt["output"].to_numpy(dtype=float)
        for k, reg in enumerate(regs_upper):
            # same off/on leaf pairing convention as edge_summaries_for_gene / bobaT/tl.py
            mask_on = (np.arange(2 ** n) >> (n - 1 - k)) & 1 == 1
            on_vals, off_vals = rule[mask_on], rule[~mask_on]
            diffs = on_vals - off_vals  # aligned since itertools.product iterates in the same MSB-first order
            abs_diffs = np.abs(diffs)
            # In a LITERAL 0/1 Boolean truth table, every true regulator's max_abs is exactly
            # 1 and min_abs is exactly 0 -- confirmed empirically (see README/chat write-up):
            # any regulator that appears in a genuine AND/OR/NOT rule has at least one context
            # where flipping it flips the output (max=1) and at least one where it doesn't
            # matter at all (min=0), so `max - min` is uniformly 1 and carries no information.
            # canalization_degree = true_max_abs - true_mean_abs (= 1 - mean_abs here) is the
            # graded quantity instead: how CONCENTRATED the regulator's influence is across
            # contexts. Near 0 = matters in almost every context (closer to unconditional in
            # effect, even though the rule is still technically nonlinear); near 1 = matters in
            # only a handful of the 2**(n-1) contexts (highly canalized/conditional).
            canalization_degree = float(abs_diffs.max() - abs_diffs.mean())
            rows.append({"gene": gene, "regulator": reg, "canalization_degree": canalization_degree,
                         "true_max_abs": abs_diffs.max(), "true_mean_abs": abs_diffs.mean()})
    return pd.DataFrame(rows)


def analysis_2_canalization_check(all_summaries: dict[str, pd.DataFrame], context_dep: pd.DataFrame) -> pd.DataFrame:
    """Among TRUE regulators only (so this is about ranking WITHIN the true set, not
    true-vs-false), does a regulator's rank under sum/mean-style summaries drop relative to
    its rank under max/dataw-style summaries as its TRUE canalization_degree increases? A
    regulator whose sum_abs rank is much lower than its max_abs rank (rank_drop > 0, since
    rank 1 = highest/best) is exactly the "washed out by averaging over irrelevant contexts"
    failure mode this whole comparison was about -- correlating rank_drop against the
    continuous, ground-truth-verified canalization_degree (not a binary split, which turned out
    not to separate anything -- see true_context_dependence()'s docstring) is the direct
    test of whether that failure mode is real and predictable from the true logic.
    """
    rows = []
    for network, df in all_summaries.items():
        pivot = df.pivot_table(index=["target", "regulator"], columns="summary", values="value")
        for target in pivot.index.get_level_values("target").unique():
            grp = pivot.loc[target]
            if len(grp) < 2:
                continue  # rank is meaningless with <2 candidates
            ranks = {s: grp[s].abs().rank(ascending=False) for s in SUMMARY_NAMES}
            for reg in grp.index:
                match = context_dep[(context_dep.gene == target) & (context_dep.regulator == reg)]
                if match.empty:
                    continue  # not a TRUE regulator of this gene (or true set has <2 regulators)
                rows.append({
                    "network": network, "gene": target, "regulator": reg,
                    "canalization_degree": match["canalization_degree"].iloc[0],
                    "rank_drop_sum_vs_max": ranks["max_abs"].loc[reg] - ranks["sum_abs"].loc[reg],
                    "rank_drop_mean_vs_dataw": ranks["dataw_abs"].loc[reg] - ranks["mean_abs"].loc[reg],
                })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gt_regs = ground_truth_regulators()

    all_summaries = {}
    for network, (rules_path, data_path, label) in NETWORKS.items():
        print(f"[fit] {label} ({network})")
        all_summaries[network] = build_summaries_for_network(rules_path, data_path)
        all_summaries[network].to_csv(f"{OUT_DIR}/edge_summaries_hsc_{network}.csv", index=False)

    print("\n=== Analysis 1: structure recovery (pooled AUROC, true vs. false candidate regulators) ===")
    structure = analysis_1_structure_recovery(all_summaries, gt_regs)
    structure.to_csv(f"{OUT_DIR}/comparison_edge_weight_summaries_hsc_structure.csv", index=False)
    print(structure.pivot(index="summary", columns="network", values="auroc").round(3))
    print("\nn_true_edges / n_candidate_edges per network (same across summaries):")
    print(structure.drop_duplicates("network").set_index("network")[["n_true_edges", "n_candidate_edges"]])

    print("\n=== Analysis 2: does true context-dependence predict which summary wins? ===")
    context_dep = true_context_dependence()
    context_dep.to_csv(f"{OUT_DIR}/hsc_true_context_dependence.csv", index=False)
    print(f"\nTrue regulators' ground-truth canalization_degree (0=unconditional, "
          f"1=fully canalized): range {context_dep['canalization_degree'].min():.3f}-"
          f"{context_dep['canalization_degree'].max():.3f}, "
          f"{(context_dep['canalization_degree'] > 0).sum()}/{len(context_dep)} > 0")
    print(context_dep.sort_values("canalization_degree").to_string(index=False))

    canal = analysis_2_canalization_check(all_summaries, context_dep)
    canal.to_csv(f"{OUT_DIR}/comparison_edge_weight_summaries_hsc_canalization.csv", index=False)
    print("\nSpearman correlation of canalization_degree vs. rank_drop (positive = the more "
          "context-dependent a true regulator's real role is, the more max_abs/dataw_abs "
          "ranks it higher than sum_abs/mean_abs does -- i.e. averaging is hiding it "
          "specifically for genuinely canalized regulators):")
    if canal["canalization_degree"].nunique() > 1:
        print(canal[["canalization_degree", "rank_drop_sum_vs_max", "rank_drop_mean_vs_dataw"]]
              .corr(method="spearman").loc[["canalization_degree"], :])
    else:
        print("canalization_degree has no variation among this network's TRUE regulators (n="
              f"{len(canal)}) -- correlation undefined.")
    print(f"\n(n={len(canal)} true-regulator observations across all 4 networks)")

    print(f"\n-> {OUT_DIR}/comparison_edge_weight_summaries_hsc_{{structure,canalization}}.csv, "
          f"hsc_true_context_dependence.csv")


if __name__ == "__main__":
    main()
